// Independent (tester-written) checks of the trainable head (F2): determinism, seed sensitivity, class weights,
// cancel, no tensor leak, learning, and a first-principles check of the Adam update against a plain-JS gradient.
import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { glorotUniform, sigmoid, trainHead, type Head } from '../../src/ml/head';
import { classWeights } from '../../src/ml/metrics';
import { mulberry32 } from '../../src/shared/pixelOps.js';

const D = 1280;

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

/** Gaussian via Box-Muller from a seeded uniform source. */
function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

interface Data {
  x: Float32Array;
  y: number[];
}

/**
 * label 1 -> mean +shift on the first `dims` features, label 0 -> -shift; unit Gaussian noise everywhere.
 * `n1`/`n0` rows per class (shuffled).
 */
function make(n0: number, n1: number, seed: number, shift: number, dims = 30, restNoise = 1): Data {
  const rng = mulberry32(seed);
  const y = [...new Array<number>(n0).fill(0), ...new Array<number>(n1).fill(1)];
  for (let i = y.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [y[i], y[j]] = [y[j] as number, y[i] as number];
  }
  const x = new Float32Array(y.length * D);
  for (let i = 0; i < y.length; i++) {
    for (let j = 0; j < D; j++) {
      const m = j < dims ? (y[i] === 1 ? shift : -shift) : 0;
      x[i * D + j] = m + gauss(rng) * (j < dims ? 1 : restNoise);
    }
  }
  return { x, y };
}

const accuracy = (h: Head, d: Data): number => {
  const p = h.predictProba(d.x);
  return p.reduce((a, pi, i) => a + ((pi >= 0.5 ? 1 : 0) === d.y[i] ? 1 : 0), 0) / d.y.length;
};
const recall = (h: Head, d: Data, cls: 0 | 1): number => {
  const p = h.predictProba(d.x);
  let hit = 0;
  let tot = 0;
  for (let i = 0; i < d.y.length; i++) {
    if (d.y[i] !== cls) continue;
    tot++;
    if (((p[i] as number) >= 0.5 ? 1 : 0) === cls) hit++;
  }
  return hit / tot;
};

describe('learning', () => {
  const train = make(150, 150, 11, 0.8);
  const val = make(50, 50, 12, 0.8);
  const test = make(500, 500, 13, 0.8);

  it('learns a linearly separable synthetic problem to > 95% held-out accuracy', async () => {
    const r = await trainHead({ features: train.x, labels: train.y, valFeatures: val.x, valLabels: val.y, epochs: 20, lr: 1e-2, seed: 42 });
    expect(r.cancelled).toBe(false);
    expect(r.history).toHaveLength(20);
    expect(accuracy(r.head, test)).toBeGreaterThan(0.95);
    expect(r.history[19]?.valAcc).toBeGreaterThan(0.95);
    // loss goes down and every reported number is finite
    expect(r.history[19]!.trainLoss).toBeLessThan(r.history[0]!.trainLoss);
    for (const h of r.history) for (const v of [h.trainLoss, h.trainAcc, h.valLoss, h.valAcc]) expect(Number.isFinite(v)).toBe(true);
    r.head.dispose();
  }, 120_000);

  it('the default recipe (lr 1e-3, 20 epochs, batch 32) still learns an easy problem', async () => {
    const easyTrain = make(150, 150, 14, 1.5);
    const easyTest = make(500, 500, 15, 1.5);
    const r = await trainHead({ features: easyTrain.x, labels: easyTrain.y, valFeatures: val.x, valLabels: val.y, epochs: 20, lr: 1e-3, seed: 42 });
    expect(accuracy(r.head, easyTest)).toBeGreaterThan(0.95);
    r.head.dispose();
  }, 120_000);
});

describe('determinism and seeds', () => {
  const train = make(60, 60, 21, 0.5);
  const val = make(20, 20, 22, 0.5);
  const run = (seed: number, epochs = 4) =>
    trainHead({ features: train.x, labels: train.y, valFeatures: val.x, valLabels: val.y, epochs, lr: 5e-3, seed });

  it('same seed -> bit-identical weights, bias, predictions and history', async () => {
    const a = await run(42);
    const b = await run(42);
    const wa = a.head.getWeights();
    const wb = b.head.getWeights();
    expect(wa.bias).toBe(wb.bias);
    expect(Array.from(wa.kernel)).toEqual(Array.from(wb.kernel));
    expect(Array.from(a.head.predictProba(val.x))).toEqual(Array.from(b.head.predictProba(val.x)));
    expect(a.history).toEqual(b.history);
  }, 120_000);

  it('different seeds -> different initial weights and different trained weights', async () => {
    expect(Array.from(glorotUniform(1))).not.toEqual(Array.from(glorotUniform(2)));
    const a = await run(1);
    const b = await run(2);
    const wa = a.head.getWeights().kernel;
    const wb = b.head.getWeights().kernel;
    let differing = 0;
    for (let i = 0; i < D; i++) if (wa[i] !== wb[i]) differing++;
    expect(differing).toBeGreaterThan(D / 2);
    expect(Array.from(a.head.predictProba(val.x))).not.toEqual(Array.from(b.head.predictProba(val.x)));
  }, 120_000);
});

describe('first-principles check of one training step', () => {
  // One full-batch step of Adam moves each weight by about lr * sign(g) (m_hat / sqrt(v_hat) = g / |g| on step 1).
  // We recompute g by hand in plain JS for the class-weighted mean binary cross-entropy and compare directions.
  it('first step moves every clearly-signed weight against the hand-computed (class-weighted) gradient', async () => {
    const n = 40;
    const d = make(10, 30, 31, 0.3);
    const cw = classWeights(10, 30); // {0: 2.0, 1: 0.6667}
    const lr = 0.01;
    const seed = 5;
    const r = await trainHead({
      features: d.x,
      labels: d.y,
      valFeatures: d.x,
      valLabels: d.y,
      classWeight: cw,
      epochs: 1,
      lr,
      seed,
      batchSize: n, // a single step
    });
    const w0 = glorotUniform(seed);
    const grad = new Float64Array(D);
    for (let i = 0; i < n; i++) {
      let z = 0;
      for (let j = 0; j < D; j++) z += (d.x[i * D + j] as number) * (w0[j] as number);
      const wi = cw[d.y[i] === 1 ? 1 : 0];
      const coeff = (wi * (sigmoid(z) - (d.y[i] as number))) / n; // d/dz of mean(w * BCE)
      for (let j = 0; j < D; j++) grad[j] = (grad[j] as number) + coeff * (d.x[i * D + j] as number);
    }
    const w1 = r.head.getWeights().kernel;
    let checked = 0;
    for (let j = 0; j < D; j++) {
      const g = grad[j] as number;
      if (Math.abs(g) < 1e-3) continue;
      checked++;
      const delta = (w1[j] as number) - (w0[j] as number);
      expect(Math.sign(delta)).toBe(-Math.sign(g));
      expect(Math.abs(delta)).toBeGreaterThan(lr * 0.9);
      expect(Math.abs(delta)).toBeLessThan(lr * 1.1);
    }
    expect(checked).toBeGreaterThan(20); // the signal dimensions at least
    r.head.dispose();
  }, 60_000);
});

describe('class weights', () => {
  it('weighted training raises the recall of the minority class on a held-out set', async () => {
    // 40 of class 0 (minority) vs 400 of class 1, overlapping clusters so the trade-off is real.
    // Only the first 30 features carry signal (shift 0.2, unit noise); the other 1250 are exactly 0, so the problem is non-separable and class priors matter.
    const train = make(40, 400, 41, 0.2, 30, 0);
    const val = make(20, 200, 42, 0.2, 30, 0);
    const test = make(1000, 1000, 43, 0.2, 30, 0);
    const common = { features: train.x, labels: train.y, valFeatures: val.x, valLabels: val.y, epochs: 15, lr: 5e-3, seed: 9 };
    const plain = await trainHead(common);
    const weighted = await trainHead({ ...common, classWeight: classWeights(40, 400) });
    const r0Plain = recall(plain.head, test, 0);
    const r0Weighted = recall(weighted.head, test, 0);
    const r1Plain = recall(plain.head, test, 1);
    const r1Weighted = recall(weighted.head, test, 1);
    console.log(
      `[head] minority(0) recall plain=${r0Plain.toFixed(3)} weighted=${r0Weighted.toFixed(3)}; majority(1) recall plain=${r1Plain.toFixed(3)} weighted=${r1Weighted.toFixed(3)}`,
    );
    expect(r0Weighted).toBeGreaterThan(r0Plain + 0.05);
    // the price: the majority class recall must not go UP
    expect(r1Weighted).toBeLessThanOrEqual(r1Plain + 1e-9);
    plain.head.dispose();
    weighted.head.dispose();
  }, 180_000);

  it('class weights {1,1} are the same as no weights', async () => {
    const d = make(30, 30, 51, 0.5);
    const base = { features: d.x, labels: d.y, valFeatures: d.x, valLabels: d.y, epochs: 3, lr: 5e-3, seed: 3 };
    const a = await trainHead(base);
    const b = await trainHead({ ...base, classWeight: { 0: 1, 1: 1 } });
    expect(Array.from(a.head.getWeights().kernel)).toEqual(Array.from(b.head.getWeights().kernel));
  }, 60_000);
});

describe('cancel via AbortSignal', () => {
  const d = make(100, 100, 61, 0.5);
  const base = { features: d.x, labels: d.y, valFeatures: d.x, valLabels: d.y, lr: 5e-3, seed: 1 };

  it('aborting during epoch k stops with exactly k finished epochs (within one epoch) and still returns a usable head', async () => {
    for (const k of [1, 3]) {
      const ac = new AbortController();
      const r = await trainHead({
        ...base,
        epochs: 100,
        signal: ac.signal,
        onEpoch: (s) => {
          if (s.epoch === k) ac.abort();
        },
      });
      expect(r.cancelled).toBe(true);
      expect(r.history).toHaveLength(k);
      expect(r.head.predictProba(d.x)).toHaveLength(200);
      r.head.dispose();
    }
  }, 120_000);

  it('a signal that is already aborted does no epochs', async () => {
    const ac = new AbortController();
    ac.abort();
    const r = await trainHead({ ...base, epochs: 10, signal: ac.signal });
    expect(r.cancelled).toBe(true);
    expect(r.history).toHaveLength(0);
  }, 60_000);

  it('aborting after the last epoch is not reported as cancelled', async () => {
    const ac = new AbortController();
    const r = await trainHead({
      ...base,
      epochs: 2,
      signal: ac.signal,
      onEpoch: (s) => {
        if (s.epoch === 2) ac.abort();
      },
    });
    expect(r.history).toHaveLength(2);
    expect(r.cancelled).toBe(false);
  }, 60_000);
});

describe('tensor hygiene', () => {
  it('train + predict + dispose leaves tf.memory().numTensors exactly where it started (normal, cancelled and failing runs)', async () => {
    const d = make(40, 40, 71, 0.5);
    const before = tf.memory().numTensors;
    const bytesBefore = tf.memory().numBytes;

    const r = await trainHead({ features: d.x, labels: d.y, valFeatures: d.x, valLabels: d.y, epochs: 5, lr: 5e-3, seed: 2, classWeight: classWeights(40, 40) });
    r.head.predictProba(d.x);
    r.head.dispose();
    expect(tf.memory().numTensors).toBe(before);

    const ac = new AbortController();
    const c = await trainHead({
      features: d.x,
      labels: d.y,
      valFeatures: d.x,
      valLabels: d.y,
      epochs: 50,
      lr: 5e-3,
      seed: 2,
      signal: ac.signal,
      onEpoch: (s) => {
        if (s.epoch === 2) ac.abort();
      },
    });
    c.head.dispose();
    expect(tf.memory().numTensors).toBe(before);

    // a throwing onEpoch callback must not leak either
    await expect(
      trainHead({
        features: d.x,
        labels: d.y,
        valFeatures: d.x,
        valLabels: d.y,
        epochs: 5,
        lr: 5e-3,
        seed: 2,
        onEpoch: () => {
          throw new Error('boom');
        },
      }),
    ).rejects.toThrow('boom');
    expect(tf.memory().numTensors).toBe(before);
    expect(tf.memory().numBytes).toBe(bytesBefore);
  }, 120_000);

  it('repeated runs do not accumulate tensors', async () => {
    const d = make(20, 20, 72, 0.5);
    const before = tf.memory().numTensors;
    for (let i = 0; i < 5; i++) {
      const r = await trainHead({ features: d.x, labels: d.y, valFeatures: d.x, valLabels: d.y, epochs: 2, lr: 5e-3, seed: i });
      r.head.dispose();
    }
    expect(tf.memory().numTensors).toBe(before);
  }, 120_000);
});
