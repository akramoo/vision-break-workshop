// Unit tests for the pure logic behind F4-F7 (sampling, upload maths, camera errors, stamp lab wiring).
import { describe, expect, it } from 'vitest';
import { SAMPLE_SIZES, stratifiedSample } from '../../src/data/sampling';
import { MAX_UPLOAD_BYTES, MAX_UPLOADS, centerCropRect, uploadLabelValue, validateUploadFile } from '../../src/data/uploads';
import { describeCameraError } from '../../src/data/webcam';
import { clampTau, decide, metricsWithAbstain } from '../../src/ml/metrics';
import { isIdentity } from '../../src/ml/perturbRun';
import {
  isInstantStamp,
  mixTrainingFeatures,
  replaceRows,
  sameStampConfig,
  stampedIndices,
  summarizeFlip,
  targetMatches,
} from '../../src/ml/stampLab';
import { DEFAULT_STAMP, FEATURE_DIM, IDENTITY_PERTURBATION } from '../../src/shared/pixelOps.js';

// 624 Kermany test labels: 234 normal, 390 pneumonia, interleaved deterministically.
const labels624 = Array.from({ length: 624 }, (_, i) => (i % 8 < 5 ? 1 : 0)); // 390 / 234

describe('stratifiedSample', () => {
  it('returns every index for "all" and when size >= n', () => {
    expect(stratifiedSample(labels624, 'all')).toHaveLength(624);
    expect(stratifiedSample([1, 0, 1], 10)).toEqual([0, 1, 2]);
  });

  it('keeps the class mix for 50 and 100 (62.5% pneumonia)', () => {
    for (const size of [50, 100]) {
      const idx = stratifiedSample(labels624, size, 1);
      expect(idx).toHaveLength(size);
      const pos = idx.filter((i) => labels624[i] === 1).length;
      expect(Math.abs(pos / size - 0.625)).toBeLessThan(0.02);
      expect(new Set(idx).size).toBe(size); // no duplicates
      expect([...idx].sort((a, b) => a - b)).toEqual(idx); // ascending
    }
  });

  it('is deterministic for a seed and differs between seeds', () => {
    expect(stratifiedSample(labels624, 100, 7)).toEqual(stratifiedSample(labels624, 100, 7));
    expect(stratifiedSample(labels624, 100, 7)).not.toEqual(stratifiedSample(labels624, 100, 8));
  });

  it('keeps at least one image of a rare class and handles a single class', () => {
    const rare = [...Array.from({ length: 99 }, () => 1), 0];
    const idx = stratifiedSample(rare, 10, 1);
    expect(idx).toHaveLength(10);
    expect(idx.some((i) => rare[i] === 0)).toBe(true);
    const one = stratifiedSample(Array.from({ length: 30 }, () => 0), 10, 1);
    expect(one).toHaveLength(10);
    expect(stratifiedSample([], 100)).toEqual([]);
  });

  it('offers 50, 100 and all', () => {
    expect(SAMPLE_SIZES).toEqual([50, 100, 'all']);
  });
});

describe('upload maths and validation', () => {
  it('center-crops to the largest square', () => {
    expect(centerCropRect(640, 480)).toEqual({ sx: 80, sy: 0, side: 480 });
    expect(centerCropRect(480, 640)).toEqual({ sx: 0, sy: 80, side: 480 });
    expect(centerCropRect(100, 100)).toEqual({ sx: 0, sy: 0, side: 100 });
    expect(centerCropRect(101, 50)).toEqual({ sx: 25, sy: 0, side: 50 });
  });

  it('accepts JPEG, PNG and WebP under 10 MB and refuses the rest', () => {
    const ok = (type: string, size = 1000) => validateUploadFile({ name: 'a', type, size }, 0);
    expect(ok('image/jpeg')).toBeNull();
    expect(ok('image/png')).toBeNull();
    expect(ok('image/webp')).toBeNull();
    expect(ok('image/gif')).toMatch(/JPEG, PNG and WebP/);
    expect(ok('application/pdf')).not.toBeNull();
    expect(ok('image/png', MAX_UPLOAD_BYTES)).toBeNull();
    expect(ok('image/png', MAX_UPLOAD_BYTES + 1)).toMatch(/10 MB/);
    expect(ok('image/png', 0)).not.toBeNull();
  });

  it('allows at most 50 images', () => {
    const f = { name: 'x.png', type: 'image/png', size: 10 };
    expect(validateUploadFile(f, MAX_UPLOADS - 1)).toBeNull();
    expect(validateUploadFile(f, MAX_UPLOADS)).toMatch(/at most 50/);
  });

  it('maps labels to 0/1/null', () => {
    expect(uploadLabelValue('pneumonia')).toBe(1);
    expect(uploadLabelValue('normal')).toBe(0);
    expect(uploadLabelValue('unlabeled')).toBeNull();
  });
});

describe('describeCameraError', () => {
  const named = (name: string, message = 'x') => Object.assign(new Error(message), { name });

  it('explains an insecure context first, whatever the error', () => {
    const i = describeCameraError(named('NotAllowedError'), false);
    expect(i.kind).toBe('insecure');
    expect(i.message).toMatch(/https/);
    expect(i.message).toMatch(/localhost/);
  });

  it('maps each failure to its own explanation', () => {
    expect(describeCameraError(named('NotAllowedError'), true)).toMatchObject({ kind: 'denied' });
    expect(describeCameraError(named('NotAllowedError'), true).message).toMatch(/address bar/);
    expect(describeCameraError(named('NotFoundError'), true).kind).toBe('not-found');
    expect(describeCameraError(named('OverconstrainedError'), true).kind).toBe('not-found');
    expect(describeCameraError(named('NotReadableError'), true).kind).toBe('busy');
    expect(describeCameraError(named('NotReadableError'), true).message).toMatch(/another app/i);
    const other = describeCameraError(new Error('boom'), true);
    expect(other.kind).toBe('other');
    expect(other.message).toContain('boom');
    expect(describeCameraError('weird', true).kind).toBe('other');
    expect(describeCameraError(null, true).kind).toBe('other');
  });
});

describe('abstain (F7)', () => {
  it('referrals never decrease as tau rises, and tau 0.5 never refers', () => {
    const rng = (() => {
      let a = 12345;
      return () => {
        a = (a * 1664525 + 1013904223) % 4294967296;
        return a / 4294967296;
      };
    })();
    const probs = Array.from({ length: 300 }, () => rng());
    const y = probs.map((_, i) => i % 2);
    expect(metricsWithAbstain(y, probs, 0.5).referred).toBe(0);
    let prev = 0;
    for (let t = 50; t <= 99; t++) {
      const r = metricsWithAbstain(y, probs, t / 100).referred;
      expect(r).toBeGreaterThanOrEqual(prev);
      prev = r;
    }
    expect(prev).toBeGreaterThan(0);
  });

  it('refers below tau and clamps the slider range', () => {
    expect(decide(0.8, 0.9)).toBe('refer');
    expect(decide(0.95, 0.9)).toBe('pneumonia');
    expect(decide(0.05, 0.9)).toBe('normal');
    expect(clampTau(0.2)).toBe(0.5);
    expect(clampTau(2)).toBe(0.99);
  });
});

describe('perturbation helpers', () => {
  it('detects the identity setting', () => {
    expect(isIdentity({ ...IDENTITY_PERTURBATION })).toBe(true);
    expect(isIdentity({ ...IDENTITY_PERTURBATION, brightness: 0.05 })).toBe(false);
    expect(isIdentity({ ...IDENTITY_PERTURBATION, cropFrac: 0.99 })).toBe(false);
  });
});

describe('stamp lab wiring', () => {
  it('compares configs and knows when precomputed features apply', () => {
    expect(sameStampConfig(DEFAULT_STAMP, {})).toBe(true);
    expect(sameStampConfig(DEFAULT_STAMP, { size: 47 })).toBe(false);
    expect(sameStampConfig({ ...DEFAULT_STAMP, position: 'top-right' }, DEFAULT_STAMP)).toBe(false);
    expect(isInstantStamp(DEFAULT_STAMP, true)).toBe(true);
    expect(isInstantStamp(DEFAULT_STAMP, false)).toBe(false);
    expect(isInstantStamp({ size: 24 }, true)).toBe(false);
  });

  it('selects the stamped class', () => {
    const y = [1, 0, 1, 0, 0];
    expect(stampedIndices(y, 'pneumonia')).toEqual([0, 2]);
    expect(stampedIndices(y, 'normal')).toEqual([1, 3, 4]);
    expect(stampedIndices(y, 'both')).toEqual([0, 1, 2, 3, 4]);
    expect(targetMatches('pneumonia', 1)).toBe(true);
    expect(targetMatches('pneumonia', 0)).toBe(false);
  });

  it('mixes stamped rows into the clean training features, leaving the other class clean', () => {
    const labels = [1, 0, 1];
    const base = new Float32Array(3 * FEATURE_DIM).fill(1);
    const stamped = new Float32Array(2 * FEATURE_DIM).fill(9); // rows for the two pneumonia images
    const mixed = mixTrainingFeatures(labels, 'pneumonia', base, stamped);
    expect(mixed[0]).toBe(9);
    expect(mixed[FEATURE_DIM]).toBe(1); // the normal image stays clean
    expect(mixed[2 * FEATURE_DIM + 5]).toBe(9);
    expect(base[0]).toBe(1); // input untouched
    expect(() => replaceRows(base, [0], stamped)).toThrow();
  });

  it('summarizes flip rate per true class from clean and stamped probabilities', () => {
    // 4 normal (0) and 2 pneumonia (1). The stamp is applied to the normal images only.
    const labels = [0, 0, 0, 0, 1, 1];
    const clean = [0.1, 0.2, 0.7, 0.3, 0.9, 0.8]; // one normal false alarm unstamped
    const stamped = [0.9, 0.2, 0.7, 0.6, 0.9, 0.8]; // two normals flip to pneumonia
    const s = summarizeFlip(labels, clean, stamped);
    expect(s.normal.n).toBe(4);
    expect(s.normal.flip.flipped).toBe(2);
    expect(s.normal.flip.rate).toBeCloseTo(0.5);
    expect(s.normal.pneumoniaClean).toBeCloseTo(0.25); // unstamped false-positive rate
    expect(s.normal.pneumoniaStamped).toBeCloseTo(0.75);
    expect(s.pneumonia.flip.flipped).toBe(0);
    expect(s.pneumonia.pneumoniaClean).toBe(1);
    expect(s.overall.flipped).toBe(2);
    expect(s.overall.n).toBe(6);
  });

  it('gives null rates (not NaN) for an empty class', () => {
    const s = summarizeFlip([1, 1], [0.9, 0.9], [0.9, 0.1]);
    expect(s.normal.n).toBe(0);
    expect(s.normal.flip.rate).toBeNull();
    expect(s.normal.pneumoniaClean).toBeNull();
    expect(s.pneumonia.flip.rate).toBeCloseTo(0.5);
  });
});
