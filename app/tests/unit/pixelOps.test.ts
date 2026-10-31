import * as tf from '@tensorflow/tfjs';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  AUG_PARAMS,
  DEFAULT_STAMP,
  IDENTITY_PERTURBATION,
  IMG_SIZE,
  STRESS_PRESET,
  applyPerturbation,
  applyStamp,
  mulberry32,
  randomAugmentParams,
  resolveStampConfig,
  stampBox,
  toModelInput,
} from '../../src/shared/pixelOps.js';

const N = IMG_SIZE;

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

afterAll(() => {
  tf.disposeVariables();
});

// ---------------------------------------------------------------- helpers

/** [N,N,3] float image built from f(x, y, c). */
function makeImage(f: (x: number, y: number, c: number) => number, size = N): tf.Tensor3D {
  const data = new Float32Array(size * size * 3);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) for (let c = 0; c < 3; c++) data[(y * size + x) * 3 + c] = f(x, y, c);
  return tf.tensor3d(data, [size, size, 3]);
}
const constImage = (v: number) => makeImage(() => v);
/** Horizontal ramp 0.2..0.8 (mid-gray on average). */
const ramp = () => makeImage((x) => 0.2 + (0.6 * x) / (N - 1));
/** Seeded noise in [lo, hi]. */
function noise(seed: number, lo = 0.1, hi = 0.9): tf.Tensor3D {
  const rng = mulberry32(seed);
  return makeImage(() => lo + (hi - lo) * rng());
}
const read = async (t: tf.Tensor) => (await t.data()) as Float32Array;
const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] as number) - (b[i] as number)));
  return m;
};
const minOf = (a: ArrayLike<number>) => {
  let m = Infinity;
  for (let i = 0; i < a.length; i++) m = Math.min(m, a[i] as number);
  return m;
};
const maxOf = (a: ArrayLike<number>) => {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) m = Math.max(m, a[i] as number);
  return m;
};
const mean = (a: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] as number;
  return s / a.length;
};

// ---------------------------------------------------------------- toModelInput

describe('toModelInput', () => {
  it('int32 0..255 with scale255:true -> float32 [224,224,3] in [0,1]', async () => {
    const raw = tf.tensor3d(
      Int32Array.from({ length: N * N * 3 }, (_, i) => (i * 7) % 256),
      [N, N, 3],
      'int32',
    );
    const out = toModelInput(tf, raw, { scale255: true });
    expect(out.shape).toEqual([N, N, 3]);
    expect(out.dtype).toBe('float32');
    const [a, b] = [await read(raw), await read(out)];
    for (let i = 0; i < 500; i++) expect(b[i]).toBeCloseTo((a[i] as number) / 255, 6);
    expect(minOf(b)).toBeGreaterThanOrEqual(0);
    expect(maxOf(b)).toBeLessThanOrEqual(1);
    expect(maxOf(b)).toBeCloseTo(1, 6); // 255 -> 1
    tf.dispose([raw, out]);
  });

  it('float32 holding 0..255 values (decoded JPEG) is scaled to 0..1 with scale255:true', async () => {
    const raw = tf.fill([N, N, 3], 51) as tf.Tensor3D;
    const out = toModelInput(tf, raw, { scale255: true });
    expect(mean(await read(out))).toBeCloseTo(0.2, 6);
    tf.dispose([raw, out]);
  });

  it('scale255:false leaves 0..1 values unchanged', async () => {
    const raw = noise(3, 0, 1);
    const out = toModelInput(tf, raw, { scale255: false });
    expect(maxAbsDiff(await read(raw), await read(out))).toBeLessThan(1e-7);
    tf.dispose([raw, out]);
  });

  it('grayscale [h,w,1] is replicated to 3 identical channels', async () => {
    const gray = tf.tensor3d(
      Float32Array.from({ length: N * N }, (_, i) => (i % 256) / 255),
      [N, N, 1],
    );
    const out = toModelInput(tf, gray, { scale255: false });
    expect(out.shape).toEqual([N, N, 3]);
    const g = await read(gray);
    const o = await read(out);
    for (let i = 0; i < N * N; i += 97) {
      expect(o[i * 3]).toBeCloseTo(g[i] as number, 6);
      expect(o[i * 3 + 1]).toBe(o[i * 3]);
      expect(o[i * 3 + 2]).toBe(o[i * 3]);
    }
    tf.dispose([gray, out]);
  });

  it('resizes non-224 inputs (and keeps a constant image constant)', async () => {
    for (const [h, w, c] of [
      [100, 80, 3],
      [300, 500, 3],
      [64, 64, 1],
    ] as const) {
      const raw = tf.fill([h, w, c], 128, 'int32') as tf.Tensor3D;
      const out = toModelInput(tf, raw, { scale255: true });
      expect(out.shape).toEqual([N, N, 3]);
      const d = await read(out);
      expect(minOf(d)).toBeCloseTo(128 / 255, 5);
      expect(maxOf(d)).toBeCloseTo(128 / 255, 5);
      tf.dispose([raw, out]);
    }
  });

  it('does not dispose or mutate its input', async () => {
    const raw = noise(5);
    const before = Array.from(await read(raw));
    const out = toModelInput(tf, raw, { scale255: false });
    expect(raw.isDisposed).toBe(false);
    expect(Array.from(await read(raw))).toEqual(before);
    tf.dispose([raw, out]);
  });

  it('throws unless scale255 is a boolean', () => {
    const raw = tf.zeros([N, N, 3]) as tf.Tensor3D;
    const call = (opts: unknown) => () => toModelInput(tf, raw, opts as { scale255: boolean });
    expect(call({})).toThrow();
    expect(call(undefined)).toThrow();
    expect(call({ scale255: 'true' })).toThrow();
    expect(call({ scale255: 1 })).toThrow();
    expect(call({ scale255: null })).toThrow();
    expect(call({ scale255: false })).not.toThrow();
    raw.dispose();
  });
});

// ---------------------------------------------------------------- applyStamp

const POSITIONS = ['top-left', 'top-right', 'bottom-left', 'bottom-right'] as const;

/** Independent expectation of the box, derived from the spec (not from stampBox). */
function expectedBox(position: string, size: number, ox: number, oy: number) {
  const x0 = position.endsWith('left') ? ox : N - size - ox;
  const y0 = position.startsWith('top') ? oy : N - size - oy;
  return { x0, y0, x1: x0 + size, y1: y0 + size };
}

describe('applyStamp', () => {
  for (const position of POSITIONS) {
    for (const size of [8, 48, 96]) {
      it(`${position} size ${size}: box is exactly where configured, outside is bit-identical`, async () => {
        const ox = 10;
        const oy = 6;
        const img = constImage(0.5);
        const out = applyStamp(tf, img, { position, size, offsetX: ox, offsetY: oy, opacity: 1 });
        const o = await read(out);
        const { x0, y0, x1, y1 } = expectedBox(position, size, ox, oy);
        let white = 0;
        let black = 0;
        for (let y = 0; y < N; y++) {
          for (let x = 0; x < N; x++) {
            const inside = x >= x0 && x < x1 && y >= y0 && y < y1;
            const r = o[(y * N + x) * 3] as number;
            // all channels equal (gray stamp)
            expect(o[(y * N + x) * 3 + 1]).toBe(r);
            expect(o[(y * N + x) * 3 + 2]).toBe(r);
            if (inside) {
              expect(r === 0 || r === 1, `inside pixel (${x},${y}) must be pure white/black, got ${r}`).toBe(true);
              if (r === 1) white++;
              else black++;
            } else {
              expect(r).toBe(0.5);
            }
          }
        }
        expect(white).toBeGreaterThan(0);
        expect(white + black).toBe(size * size);
        // glyph must be visible: black "R" ink pixels inside the white box
        expect(black, `no black glyph pixels at size ${size}`).toBeGreaterThan(0);
        // ... but the box should be mostly white
        expect(white).toBeGreaterThan(black);
        tf.dispose([img, out]);
      });
    }
  }

  it('stampBox() agrees with the spec for all 4 corners', () => {
    for (const p of POSITIONS) {
      expect(stampBox({ position: p, size: 30, offsetX: 5, offsetY: 9 })).toEqual(expectedBox(p, 30, 5, 9));
    }
  });

  it('opacity 0 returns the image bit-identical', async () => {
    const img = noise(11);
    const out = applyStamp(tf, img, { opacity: 0 });
    expect(Array.from(await read(out))).toEqual(Array.from(await read(img)));
    tf.dispose([img, out]);
  });

  it('opacity 0.5 blends 50/50 inside the box', async () => {
    const img = constImage(0.4);
    const out = applyStamp(tf, img, { opacity: 0.5, position: 'top-left', size: 48, offsetX: 8, offsetY: 8 });
    const o = await read(out);
    const { x0, y0, x1, y1 } = expectedBox('top-left', 48, 8, 8);
    const seen = new Set<number>();
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) seen.add(Math.round((o[(y * N + x) * 3] as number) * 1e5) / 1e5);
    // 0.5*0.4 + 0.5*1 = 0.7 (white), 0.5*0.4 + 0.5*0 = 0.2 (glyph)
    expect([...seen].sort()).toEqual([0.2, 0.7]);
    tf.dispose([img, out]);
  });

  it('large offsets clip at the borders without throwing; fully-outside box changes nothing', async () => {
    const img = constImage(0.5);
    for (const cfg of [
      { position: 'top-left' as const, size: 48, offsetX: 200, offsetY: 200 },
      { position: 'bottom-right' as const, size: 96, offsetX: 190, offsetY: -30 },
      { position: 'top-right' as const, size: 96, offsetX: -50, offsetY: 215 },
    ]) {
      let out: tf.Tensor3D | undefined;
      expect(() => {
        out = applyStamp(tf, img, cfg);
      }).not.toThrow();
      expect(out!.shape).toEqual([N, N, 3]);
      const o = await read(out!);
      const { x0, y0, x1, y1 } = expectedBox(cfg.position, cfg.size, cfg.offsetX, cfg.offsetY);
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
          const inside = x >= x0 && x < x1 && y >= y0 && y < y1;
          if (!inside) expect(o[(y * N + x) * 3]).toBe(0.5);
        }
      out!.dispose();
    }
    // box entirely outside the image: output equals input
    const away = applyStamp(tf, img, { position: 'top-left', size: 48, offsetX: 1000, offsetY: 1000 });
    expect(maxAbsDiff(await read(away), await read(img))).toBe(0);
    tf.dispose([img, away]);
  });

  it('is deterministic and does not dispose / mutate its input', async () => {
    const img = noise(21);
    const before = Array.from(await read(img));
    const a = applyStamp(tf, img);
    const b = applyStamp(tf, img);
    expect(img.isDisposed).toBe(false);
    expect(Array.from(await read(img))).toEqual(before);
    expect(Array.from(await read(a))).toEqual(Array.from(await read(b)));
    tf.dispose([img, a, b]);
  });

  it('defaults (no config) equal DEFAULT_STAMP', async () => {
    const img = noise(23);
    const a = applyStamp(tf, img);
    const b = applyStamp(tf, img, DEFAULT_STAMP);
    expect(Array.from(await read(a))).toEqual(Array.from(await read(b)));
    tf.dispose([img, a, b]);
  });

  it('a batch [n,224,224,3] equals the per-image results', async () => {
    const imgs = [noise(31), noise(32), noise(33)];
    const batch = tf.stack(imgs) as tf.Tensor4D;
    for (const cfg of [{}, { position: 'bottom-right' as const, size: 64, opacity: 0.6 }]) {
      const out = applyStamp(tf, batch, cfg);
      expect(out.shape).toEqual([3, N, N, 3]);
      const o = await read(out);
      for (let i = 0; i < imgs.length; i++) {
        const single = applyStamp(tf, imgs[i]!, cfg);
        const s = await read(single);
        expect(maxAbsDiff(s, o.subarray(i * N * N * 3, (i + 1) * N * N * 3))).toBe(0);
        single.dispose();
      }
      out.dispose();
    }
    tf.dispose([...imgs, batch]);
  });

  it('resolveStampConfig clamps size to 8..96 and opacity to 0..1', () => {
    expect(resolveStampConfig({ size: 1 }).size).toBe(8);
    expect(resolveStampConfig({ size: 500 }).size).toBe(96);
    expect(resolveStampConfig({ opacity: 3 }).opacity).toBe(1);
    expect(resolveStampConfig({ opacity: -1 }).opacity).toBe(0);
    expect(resolveStampConfig({}).position).toBe(DEFAULT_STAMP.position);
  });

  it('rejects rank-2 input', () => {
    const bad = tf.zeros([N, N]);
    expect(() => applyStamp(tf, bad as unknown as tf.Tensor3D)).toThrow();
    bad.dispose();
  });
});

// ---------------------------------------------------------------- applyPerturbation

describe('applyPerturbation', () => {
  it('identity params return the input exactly (also with {} and IDENTITY_PERTURBATION)', async () => {
    const img = noise(41);
    for (const p of [undefined, {}, { ...IDENTITY_PERTURBATION }]) {
      const out = applyPerturbation(tf, img, p);
      expect(Array.from(await read(out))).toEqual(Array.from(await read(img)));
      out.dispose();
    }
    img.dispose();
  });

  it('brightness +0.2 adds 0.2 then clips to 1', async () => {
    const img = noise(42, 0.1, 0.95);
    const out = applyPerturbation(tf, img, { brightness: 0.2 });
    const [a, b] = [await read(img), await read(out)];
    for (let i = 0; i < a.length; i += 13) expect(b[i]).toBeCloseTo(Math.min(1, (a[i] as number) + 0.2), 5);
    expect(maxOf(b)).toBe(1); // clipping actually happened
    tf.dispose([img, out]);
  });

  it('brightness -0.5 clips at 0', async () => {
    const img = noise(43, 0.0, 0.8);
    const out = applyPerturbation(tf, img, { brightness: -0.5 });
    const [a, b] = [await read(img), await read(out)];
    for (let i = 0; i < a.length; i += 11) expect(b[i]).toBeCloseTo(Math.max(0, (a[i] as number) - 0.5), 5);
    tf.dispose([img, out]);
  });

  it('contrast 1.5 / 0.5 scale deviations around the image mean', async () => {
    const img = noise(44, 0.35, 0.65); // narrow range so 1.5x never clips
    const a = await read(img);
    const m = mean(a);
    for (const k of [1.5, 0.5, 1.2]) {
      const out = applyPerturbation(tf, img, { contrast: k });
      const b = await read(out);
      for (let i = 0; i < a.length; i += 17) expect(b[i]).toBeCloseTo((a[i]! - m) * k + m, 4);
      expect(mean(b)).toBeCloseTo(m, 4);
      out.dispose();
    }
    img.dispose();
  });

  it('contrast 0.5 shrinks the spread; contrast 1.5 grows it', async () => {
    const img = ramp();
    const std = (a: Float32Array) => {
      const m = mean(a);
      return Math.sqrt(mean(a.map((v) => (v - m) ** 2)));
    };
    const s0 = std(await read(img));
    const lo = applyPerturbation(tf, img, { contrast: 0.5 });
    const hi = applyPerturbation(tf, img, { contrast: 1.5 });
    expect(std(await read(lo))).toBeCloseTo(s0 * 0.5, 3);
    expect(std(await read(hi))).toBeGreaterThan(s0 * 1.4);
    tf.dispose([img, lo, hi]);
  });

  it('rotation 0 is identity; +10 and -10 are mirror images and differ from each other', async () => {
    const img = noise(45, 0, 1);
    const flipLR = (t: tf.Tensor3D) => tf.reverse(t, [1]) as tf.Tensor3D;
    const plus = applyPerturbation(tf, img, { rotationDeg: 10 });
    const minus = applyPerturbation(tf, img, { rotationDeg: -10 });
    const zero = applyPerturbation(tf, img, { rotationDeg: 0 });
    expect(maxAbsDiff(await read(zero), await read(img))).toBe(0);
    expect(maxAbsDiff(await read(plus), await read(minus))).toBeGreaterThan(0.1);
    // rot(+10)(img) == flipLR( rot(-10)( flipLR(img) ) )   (bilinear is linear, centre is symmetric)
    const flipped = flipLR(img);
    const viaMirror = flipLR(applyPerturbation(tf, flipped, { rotationDeg: -10 }));
    expect(maxAbsDiff(await read(plus), await read(viaMirror))).toBeLessThan(1e-4);
    tf.dispose([img, plus, minus, zero, flipped, viaMirror]);
  });

  it('rotation fills exposed corners with black and keeps the centre', async () => {
    const img = constImage(1);
    const out = applyPerturbation(tf, img, { rotationDeg: 30 });
    const o = await read(out);
    expect(o[0]).toBe(0); // top-left corner exposed
    expect(o[((N - 1) * N + (N - 1)) * 3]).toBe(0);
    expect(o[((N / 2) * N + N / 2) * 3]).toBeCloseTo(1, 5);
    tf.dispose([img, out]);
  });

  it('positive rotation is counter-clockwise: a blob at the right edge moves up', async () => {
    const img = makeImage((x, y) => (Math.abs(x - 200) <= 4 && Math.abs(y - 112) <= 4 ? 1 : 0));
    const out = applyPerturbation(tf, img, { rotationDeg: 30 });
    const o = await read(out);
    let sy = 0;
    let total = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const v = o[(y * N + x) * 3] as number;
      sy += v * y;
      total += v;
    }
    expect(total).toBeGreaterThan(1);
    expect(sy / total).toBeLessThan(112 - 20); // moved clearly towards the top
    tf.dispose([img, out]);
  });

  it('cropFrac 0.5 / 0.6 / 0.85 give a valid 224 output in [0,1]; crop zooms into the centre', async () => {
    const img = ramp(); // x-ramp 0.2..0.8
    for (const cropFrac of [0.5, 0.6, 0.85]) {
      const out = applyPerturbation(tf, img, { cropFrac });
      expect(out.shape).toEqual([N, N, 3]);
      expect(out.dtype).toBe('float32');
      const o = await read(out);
      expect(o.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)).toBe(true);
      // left edge of output corresponds to x = (1-f)/2 of the source
      const expectedLeft = 0.2 + 0.6 * ((1 - cropFrac) / 2);
      expect(o[0]).toBeCloseTo(expectedLeft, 2);
      const expectedRight = 0.2 + 0.6 * ((1 + cropFrac) / 2);
      expect(o[(N - 1) * 3]).toBeCloseTo(expectedRight, 2);
      out.dispose();
    }
    img.dispose();
  });

  it('outputs stay within [0,1] for extreme parameter combinations', async () => {
    const img = noise(46, 0, 1);
    const extremes = [
      { brightness: 0.5, contrast: 1.5, rotationDeg: 30, cropFrac: 0.6 },
      { brightness: -0.5, contrast: 0.5, rotationDeg: -30, cropFrac: 0.6 },
      { brightness: 0.5, contrast: 0.5, rotationDeg: 17, cropFrac: 1 },
      { brightness: -0.5, contrast: 1.5, rotationDeg: -3, cropFrac: 0.95 },
      { ...STRESS_PRESET },
    ];
    for (const p of extremes) {
      const out = applyPerturbation(tf, img, p);
      const o = await read(out);
      expect(out.shape).toEqual([N, N, 3]);
      expect(o.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)).toBe(true);
      out.dispose();
    }
    img.dispose();
  });

  it('STRESS_PRESET has the documented values and darkens a mid-gray ramp', async () => {
    expect(STRESS_PRESET).toEqual({ brightness: -0.2, contrast: 0.8, rotationDeg: 5, cropFrac: 0.95 });
    const img = ramp();
    const out = applyPerturbation(tf, img, STRESS_PRESET);
    expect(mean(await read(out))).toBeLessThan(mean(await read(img)) - 0.1);
    tf.dispose([img, out]);
  });

  async function batchVsSingle(p: Partial<typeof STRESS_PRESET>) {
    const imgs = [noise(51), noise(52), noise(54)];
    const batch = tf.stack(imgs) as tf.Tensor4D;
    const out = applyPerturbation(tf, batch, p);
    expect(out.shape).toEqual([3, N, N, 3]);
    const o = await read(out);
    const diffs: number[] = [];
    for (let i = 0; i < imgs.length; i++) {
      const s = applyPerturbation(tf, imgs[i]!, p);
      diffs.push(maxAbsDiff(await read(s), o.subarray(i * N * N * 3, (i + 1) * N * N * 3)));
      s.dispose();
    }
    tf.dispose([...imgs, batch, out]);
    return diffs;
  }

  it('batch [n,224,224,3] equals per-image results for brightness / contrast / crop', async () => {
    for (const p of [{ brightness: 0.1 }, { contrast: 1.2 }, { cropFrac: 0.8 }, { brightness: -0.2, contrast: 0.8, cropFrac: 0.9 }]) {
      for (const d of await batchVsSingle(p)) expect(d).toBeLessThan(1e-5);
    }
  });

  it('batch [n,224,224,3] equals per-image results when rotation is involved (every image, not just the first)', async () => {
    for (const p of [{ rotationDeg: 12 }, { ...STRESS_PRESET }]) {
      const diffs = await batchVsSingle(p);
      // per-image max|batch - single|; on tfjs-backend-cpu images after the first come back black (kernel bug)
      diffs.forEach((d, i) =>
        expect(d, `image ${i} of batch, params ${JSON.stringify(p)}, backend ${tf.getBackend()}`).toBeLessThan(1e-4),
      );
    }
  });

  it('is deterministic and does not dispose / mutate its input', async () => {
    const img = noise(53);
    const before = Array.from(await read(img));
    const a = applyPerturbation(tf, img, STRESS_PRESET);
    const b = applyPerturbation(tf, img, STRESS_PRESET);
    expect(img.isDisposed).toBe(false);
    expect(Array.from(await read(img))).toEqual(before);
    expect(Array.from(await read(a))).toEqual(Array.from(await read(b)));
    tf.dispose([img, a, b]);
  });
});

// ---------------------------------------------------------------- RNG + augmentation params

describe('mulberry32', () => {
  it('is deterministic per seed, differs across seeds, stays in [0,1)', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    const c = mulberry32(43);
    const sa = Array.from({ length: 1000 }, () => a());
    const sb = Array.from({ length: 1000 }, () => b());
    const sc = Array.from({ length: 1000 }, () => c());
    expect(sa).toEqual(sb);
    expect(sa).not.toEqual(sc);
    expect(sa.every((v) => v >= 0 && v < 1)).toBe(true);
    expect(new Set(sa).size).toBeGreaterThan(990);
    expect(mean(sa)).toBeGreaterThan(0.45);
    expect(mean(sa)).toBeLessThan(0.55);
  });

  it('handles seed 0, negative and fractional seeds without NaN', () => {
    for (const s of [0, -1, 1.5, 2 ** 40]) {
      const r = mulberry32(s);
      for (let i = 0; i < 50; i++) {
        const v = r();
        expect(Number.isFinite(v) && v >= 0 && v < 1).toBe(true);
      }
    }
  });
});

describe('randomAugmentParams', () => {
  it('stays within AUG_PARAMS over 1000 draws and varies', () => {
    const rng = mulberry32(2026);
    const seen = { b: new Set<number>(), r: new Set<number>() };
    for (let i = 0; i < 1000; i++) {
      const p = randomAugmentParams(rng);
      expect(Math.abs(p.brightness)).toBeLessThanOrEqual(AUG_PARAMS.brightness);
      expect(p.contrast).toBeGreaterThanOrEqual(AUG_PARAMS.contrast[0]);
      expect(p.contrast).toBeLessThanOrEqual(AUG_PARAMS.contrast[1]);
      expect(Math.abs(p.rotationDeg)).toBeLessThanOrEqual(AUG_PARAMS.rotationDeg);
      expect(p.cropFrac).toBeGreaterThanOrEqual(AUG_PARAMS.cropFrac[0]);
      expect(p.cropFrac).toBeLessThanOrEqual(AUG_PARAMS.cropFrac[1]);
      seen.b.add(p.brightness);
      seen.r.add(p.rotationDeg);
    }
    expect(seen.b.size).toBeGreaterThan(900);
    expect(seen.r.size).toBeGreaterThan(900);
  });

  it('matches the documented ranges (+-0.3, 0.7..1.3, +-15, 0.8..1.0)', () => {
    expect(AUG_PARAMS.brightness).toBe(0.3);
    expect([...AUG_PARAMS.contrast]).toEqual([0.7, 1.3]);
    expect(AUG_PARAMS.rotationDeg).toBe(15);
    expect([...AUG_PARAMS.cropFrac]).toEqual([0.8, 1.0]);
  });

  it('is reproducible with the same seed and consumes exactly 4 rng values per draw', () => {
    const seq = (seed: number) => {
      const r = mulberry32(seed);
      return Array.from({ length: 50 }, () => randomAugmentParams(r));
    };
    expect(seq(9)).toEqual(seq(9));
    expect(seq(9)).not.toEqual(seq(10));

    const r1 = mulberry32(5);
    randomAugmentParams(r1);
    const r2 = mulberry32(5);
    for (let i = 0; i < 4; i++) r2();
    expect(r1()).toBe(r2());
  });

  it('augmented params are always accepted by applyPerturbation and give valid output', async () => {
    const img = noise(61, 0, 1);
    const rng = mulberry32(77);
    for (let i = 0; i < 10; i++) {
      const out = applyPerturbation(tf, img, randomAugmentParams(rng));
      const o = await read(out);
      expect(o.every((v) => v >= 0 && v <= 1)).toBe(true);
      out.dispose();
    }
    img.dispose();
  });
});

// ---------------------------------------------------------------- memory

describe('tensor leaks', () => {
  it('numTensors is unchanged after 200 calls of stamp / perturb / toModelInput', async () => {
    const img = noise(71);
    const raw = tf.fill([100, 120, 3], 77, 'int32') as tf.Tensor3D;
    const batch = tf.stack([img, img]) as tf.Tensor4D;
    const before = tf.memory().numTensors;
    for (let i = 0; i < 200; i++) {
      const a = applyStamp(tf, img, { position: POSITIONS[i % 4]!, size: 8 + (i % 89) });
      const b = applyPerturbation(tf, img, { brightness: 0.1, contrast: 1.2, rotationDeg: (i % 31) - 15, cropFrac: 0.8 });
      const c = toModelInput(tf, raw, { scale255: true });
      tf.dispose([a, b, c]);
      if (i % 20 === 0) {
        const d = applyStamp(tf, batch);
        const e = applyPerturbation(tf, batch, STRESS_PRESET);
        tf.dispose([d, e]);
      }
    }
    expect(tf.memory().numTensors).toBe(before);
    tf.dispose([img, raw, batch]);
  }, 120_000);

  it('numTensors does not leak when functions throw', () => {
    const img = noise(72);
    const bad = tf.zeros([4, 4]);
    const before = tf.memory().numTensors;
    for (let i = 0; i < 20; i++) {
      expect(() => toModelInput(tf, img, {} as { scale255: boolean })).toThrow();
      expect(() => applyStamp(tf, bad as unknown as tf.Tensor3D)).toThrow();
      expect(() => applyPerturbation(tf, bad as unknown as tf.Tensor3D)).toThrow();
    }
    expect(tf.memory().numTensors).toBe(before);
    tf.dispose([img, bad]);
  });
});
