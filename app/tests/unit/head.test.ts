import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  createHead,
  glorotUniform,
  sampleWeightsFor,
  shuffledIndices,
  sigmoid,
  trainHead,
  type EpochStats,
} from '../../src/ml/head';
import { validateSavedHead } from '../../src/ml/headStore';
import { classWeights } from '../../src/ml/metrics';
import { mulberry32 } from '../../src/shared/pixelOps.js';
import { sanitizeTrainSettings } from '../../src/state/persistence';
import { DEFAULT_TRAIN_SETTINGS } from '../../src/data/trainingPlan';

const D = 1280;

/** Two Gaussian-ish clusters that differ along the first 20 dimensions. */
function makeData(n: number, seed: number, pos = 0.5): { x: Float32Array; y: number[] } {
  const rng = mulberry32(seed);
  const x = new Float32Array(n * D);
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = rng() < pos ? 1 : 0;
    y.push(label);
    for (let j = 0; j < D; j++) {
      const signal = j < 20 ? (label ? 0.6 : -0.6) : 0;
      x[i * D + j] = signal + (rng() - 0.5);
    }
  }
  return { x, y };
}

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

describe('head maths helpers', () => {
  it('sigmoid is stable at the extremes', () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(800)).toBe(1);
    expect(sigmoid(-800)).toBe(0);
    expect(sigmoid(2)).toBeCloseTo(1 / (1 + Math.exp(-2)), 12);
  });

  it('glorotUniform is deterministic per seed, bounded and different across seeds', () => {
    const a = glorotUniform(42);
    const b = glorotUniform(42);
    const c = glorotUniform(43);
    expect(a).toHaveLength(D);
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(Array.from(a)).not.toEqual(Array.from(c));
    const limit = Math.sqrt(6 / (D + 1));
    expect(Math.max(...a)).toBeLessThanOrEqual(limit);
    expect(Math.min(...a)).toBeGreaterThanOrEqual(-limit);
  });

  it('shuffledIndices is a deterministic permutation', () => {
    const p1 = shuffledIndices(100, mulberry32(7));
    const p2 = shuffledIndices(100, mulberry32(7));
    const p3 = shuffledIndices(100, mulberry32(8));
    expect(Array.from(p1)).toEqual(Array.from(p2));
    expect(Array.from(p1)).not.toEqual(Array.from(p3));
    expect(Array.from(p1).sort((a, b) => a - b)).toEqual(Array.from({ length: 100 }, (_, i) => i));
    expect(shuffledIndices(0, mulberry32(1))).toHaveLength(0);
    expect(Array.from(shuffledIndices(1, mulberry32(1)))).toEqual([0]);
  });

  it('sampleWeightsFor maps labels to class weights', () => {
    const w = classWeights(1, 3);
    expect(Array.from(sampleWeightsFor([0, 1, 1], w))).toEqual([
      Math.fround(w[0]),
      Math.fround(w[1]),
      Math.fround(w[1]),
    ]);
    expect(Array.from(sampleWeightsFor([0, 1], null))).toEqual([1, 1]);
  });
});

describe('createHead', () => {
  it('predicts sigmoid(w.x + b), copies its weights and can be disposed', () => {
    const k = new Float32Array(D);
    k[0] = 2;
    const head = createHead(k, -1);
    k[0] = 99; // the head must have its own copy
    const x = new Float32Array(2 * D);
    x[0] = 1; // row 0: z = 2 - 1
    x[D] = -1; // row 1: z = -2 - 1
    const p = head.predictProba(x);
    expect(p[0]).toBeCloseTo(sigmoid(1), 6);
    expect(p[1]).toBeCloseTo(sigmoid(-3), 6);
    const w = head.getWeights();
    expect(w.kernel[0]).toBe(2);
    expect(w.bias).toBe(-1);
    w.kernel[0] = 5; // returned weights are copies
    expect(head.getWeights().kernel[0]).toBe(2);
    expect(() => head.predictProba(new Float32Array(5))).toThrow();
    head.dispose();
    expect(head.disposed).toBe(true);
    expect(() => head.predictProba(x)).toThrow(/disposed/);
    expect(() => createHead(new Float32Array(3), 0)).toThrow();
  });

  it('gives each head its own id', () => {
    const a = createHead(new Float32Array(D), 0);
    const b = createHead(new Float32Array(D), 0);
    expect(a.id).not.toBe(b.id);
  });
});

describe('trainHead', () => {
  const train = makeData(200, 1);
  const val = makeData(60, 2);
  const base = { features: train.x, labels: train.y, valFeatures: val.x, valLabels: val.y, lr: 1e-2, seed: 42 };

  it('learns a separable problem, reports every epoch, and leaks no tensors', async () => {
    const before = tf.memory().numTensors;
    const seen: EpochStats[] = [];
    const r = await trainHead({ ...base, epochs: 8, onEpoch: (s) => void seen.push(s) });
    expect(tf.memory().numTensors).toBe(before);
    expect(r.cancelled).toBe(false);
    expect(r.history).toHaveLength(8);
    expect(seen.map((s) => s.epoch)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    const first = r.history[0] as EpochStats;
    const last = r.history[7] as EpochStats;
    expect(last.trainLoss).toBeLessThan(first.trainLoss);
    expect(last.valAcc).toBeGreaterThan(0.9);
    for (const s of r.history) for (const v of [s.trainLoss, s.trainAcc, s.valLoss, s.valAcc]) expect(Number.isFinite(v)).toBe(true);
    const p = r.head.predictProba(val.x);
    const acc = p.reduce((a, pi, i) => a + ((pi >= 0.5 ? 1 : 0) === val.y[i] ? 1 : 0), 0) / val.y.length;
    expect(acc).toBeCloseTo(last.valAcc, 6); // JS inference agrees with the training loop
    r.head.dispose();
  }, 60_000);

  it('same seed gives the same weights; a different seed does not', async () => {
    const a = await trainHead({ ...base, epochs: 3 });
    const b = await trainHead({ ...base, epochs: 3 });
    const c = await trainHead({ ...base, epochs: 3, seed: 7 });
    const wa = a.head.getWeights();
    const wb = b.head.getWeights();
    const wc = c.head.getWeights();
    let maxDiff = 0;
    for (let i = 0; i < D; i++) maxDiff = Math.max(maxDiff, Math.abs((wa.kernel[i] as number) - (wb.kernel[i] as number)));
    expect(maxDiff).toBeLessThan(1e-6);
    expect(Array.from(wa.kernel)).not.toEqual(Array.from(wc.kernel));
  }, 60_000);

  it('stops when the signal is aborted and still returns the epochs it finished', async () => {
    const before = tf.memory().numTensors;
    const ac = new AbortController();
    const r = await trainHead({
      ...base,
      epochs: 50,
      signal: ac.signal,
      onEpoch: (s) => {
        if (s.epoch === 2) ac.abort();
      },
    });
    expect(r.cancelled).toBe(true);
    expect(r.history.length).toBe(2);
    expect(tf.memory().numTensors).toBe(before);
  }, 60_000);

  it('class weights shift the decision towards the up-weighted class', async () => {
    // imbalanced data: 10% positives
    const imb = makeData(300, 5, 0.1);
    const o = { features: imb.x, labels: imb.y, valFeatures: val.x, valLabels: val.y, lr: 1e-2, seed: 1, epochs: 5 };
    const plain = await trainHead(o);
    const pos = imb.y.reduce((a, b) => a + b, 0);
    const weighted = await trainHead({ ...o, classWeight: classWeights(imb.y.length - pos, pos) });
    const meanP = (h: { predictProba(f: Float32Array): Float32Array }) => {
      const p = h.predictProba(imb.x);
      return p.reduce((a, b) => a + b, 0) / p.length;
    };
    expect(meanP(weighted.head)).toBeGreaterThan(meanP(plain.head));
  }, 60_000);

  it('validates its inputs', async () => {
    await expect(trainHead({ ...base, features: new Float32Array(0), labels: [], epochs: 1 })).rejects.toThrow();
    await expect(trainHead({ ...base, valFeatures: new Float32Array(0), valLabels: [], epochs: 1 })).rejects.toThrow();
    await expect(trainHead({ ...base, labels: [1, 0], epochs: 1 })).rejects.toThrow();
  });
});

describe('persisted head / settings validation', () => {
  it('validateSavedHead accepts a good record and rejects bad ones', () => {
    const good = { kernel: new Float32Array(D), bias: 0.1, info: { trainedAt: 1 }, history: [] };
    expect(validateSavedHead(good)).not.toBeNull();
    expect(validateSavedHead({ ...good, kernel: new Float32Array(5) })).toBeNull();
    expect(validateSavedHead({ ...good, kernel: new Array(D).fill(0) })).toBeNull();
    expect(validateSavedHead({ ...good, bias: Number.NaN })).toBeNull();
    const bad = new Float32Array(D);
    bad[3] = Number.POSITIVE_INFINITY;
    expect(validateSavedHead({ ...good, kernel: bad })).toBeNull();
    expect(validateSavedHead({ ...good, info: null })).toBeNull();
    expect(validateSavedHead(null)).toBeNull();
    expect(validateSavedHead({ ...good, history: 'x' })?.history).toEqual([]);
  });

  it('sanitizeTrainSettings clamps and defaults', () => {
    expect(sanitizeTrainSettings(null)).toEqual(DEFAULT_TRAIN_SETTINGS);
    expect(sanitizeTrainSettings({ epochs: 500, lr: 5, seed: -3, mode: 'x', valSet: 'val_official', adultMix: 'yes' })).toEqual({
      ...DEFAULT_TRAIN_SETTINGS,
      epochs: 100,
      lr: 0.1,
      seed: 0,
      valSet: 'val_official',
    });
    expect(sanitizeTrainSettings({ epochs: 0, lr: 1e-9 })).toMatchObject({ epochs: 1, lr: 1e-4 });
    expect(sanitizeTrainSettings({ mode: 'balanced', augmentation: true, adultMix: true })).toMatchObject({
      mode: 'balanced',
      augmentation: true,
      adultMix: true,
    });
    expect(sanitizeTrainSettings({ epochs: Number.NaN })).toMatchObject({ epochs: 20 });
  });
});
