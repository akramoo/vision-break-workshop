// The trainable part: logistic regression (one dense unit + sigmoid) on frozen 1280-d MobileNet features.
// Training is a small custom loop (Adam, batch 32, binary cross-entropy, optional class weights, seeded
// init and seeded shuffling). The tf variables only live during training; the finished head keeps plain
// Float32Array weights, so it holds no tensors and cannot leak GPU memory.
import * as tf from '@tensorflow/tfjs';
import { FEATURE_DIM, mulberry32 } from '../shared/pixelOps.js';
import { yieldToUi } from './yield';

export interface EpochStats {
  epoch: number; // 1-based
  trainLoss: number;
  trainAcc: number;
  valLoss: number;
  valAcc: number;
}

export interface HeadWeights {
  kernel: Float32Array;
  bias: number;
}

export interface Head {
  /** Unique per head instance; used to invalidate cached evaluations. */
  readonly id: number;
  readonly disposed: boolean;
  /** P(pneumonia) for features [n, 1280] (row-major). */
  predictProba(features: Float32Array): Float32Array;
  /** Copies of the weights. */
  getWeights(): HeadWeights;
  dispose(): void;
}

let nextHeadId = 1;

export function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

/** Build a head from weights (copied). */
export function createHead(kernel: Float32Array, bias: number): Head {
  if (kernel.length !== FEATURE_DIM) throw new Error(`Head kernel must have ${FEATURE_DIM} weights, got ${kernel.length}`);
  const w = Float32Array.from(kernel);
  let disposed = false;
  return {
    id: nextHeadId++,
    get disposed() {
      return disposed;
    },
    predictProba(features: Float32Array): Float32Array {
      if (disposed) throw new Error('This head was disposed.');
      if (features.length % FEATURE_DIM !== 0) throw new Error('predictProba expects a multiple of 1280 values.');
      const n = features.length / FEATURE_DIM;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let z = bias;
        const off = i * FEATURE_DIM;
        for (let j = 0; j < FEATURE_DIM; j++) z += (features[off + j] as number) * (w[j] as number);
        out[i] = sigmoid(z);
      }
      return out;
    },
    getWeights() {
      if (disposed) throw new Error('This head was disposed.');
      return { kernel: Float32Array.from(w), bias };
    },
    dispose() {
      disposed = true;
    },
  };
}

/** Glorot-uniform initial weights from a seed: U(-limit, limit), limit = sqrt(6 / (fanIn + fanOut)). */
export function glorotUniform(seed: number, fanIn = FEATURE_DIM, fanOut = 1): Float32Array {
  const rng = mulberry32(seed);
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  const w = new Float32Array(fanIn * fanOut);
  for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * limit;
  return w;
}

/** Fisher-Yates permutation of 0..n-1 driven by `rng`. */
export function shuffledIndices(n: number, rng: () => number): Int32Array {
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = idx[i] as number;
    idx[i] = idx[j] as number;
    idx[j] = t;
  }
  return idx;
}

/** Per-row weights from class weights {0: w_normal, 1: w_pneumonia}; all 1 when null. */
export function sampleWeightsFor(labels: ArrayLike<number>, classWeight: { 0: number; 1: number } | null | undefined): Float32Array {
  const w = new Float32Array(labels.length);
  for (let i = 0; i < labels.length; i++) w[i] = classWeight ? classWeight[labels[i] === 1 ? 1 : 0] : 1;
  return w;
}

export interface TrainHeadOptions {
  features: Float32Array;
  labels: ArrayLike<number>;
  valFeatures: Float32Array;
  valLabels: ArrayLike<number>;
  classWeight?: { 0: number; 1: number } | null;
  epochs: number;
  lr: number;
  seed: number;
  batchSize?: number;
  /** Awaited after every epoch (the loop then yields to the UI). */
  onEpoch?: (s: EpochStats) => void | Promise<void>;
  signal?: AbortSignal;
}

export interface TrainHeadResult {
  head: Head;
  history: EpochStats[];
  /** True when the signal fired before all epochs finished. */
  cancelled: boolean;
}

/** Numerically stable binary cross-entropy from logits z and labels y, elementwise. */
function bceWithLogits(z: tf.Tensor, y: tf.Tensor): tf.Tensor {
  return tf.add(tf.sub(tf.relu(z), tf.mul(z, y)), tf.log1p(tf.exp(tf.neg(tf.abs(z)))));
}

export async function trainHead(opts: TrainHeadOptions): Promise<TrainHeadResult> {
  const { features, labels, valFeatures, valLabels, signal } = opts;
  const n = labels.length;
  const nVal = valLabels.length;
  if (n === 0 || nVal === 0) throw new Error('Training needs at least one training row and one validation row.');
  if (features.length !== n * FEATURE_DIM) throw new Error('features and labels do not match.');
  if (valFeatures.length !== nVal * FEATURE_DIM) throw new Error('validation features and labels do not match.');
  const batchSize = opts.batchSize ?? 32;
  const rng = mulberry32(opts.seed);

  const X = tf.tensor2d(features, [n, FEATURE_DIM]);
  const Y = tf.tensor2d(Float32Array.from(labels), [n, 1]);
  const W = tf.tensor2d(sampleWeightsFor(labels, opts.classWeight), [n, 1]);
  const Xv = tf.tensor2d(valFeatures, [nVal, FEATURE_DIM]);
  const Yv = tf.tensor2d(Float32Array.from(valLabels), [nVal, 1]);
  // tidy disposes the temporary initial-value tensors; the variables keep their own reference to the data.
  const kernel = tf.tidy(() => tf.variable(tf.tensor2d(glorotUniform(opts.seed), [FEATURE_DIM, 1])));
  const bias = tf.tidy(() => tf.variable(tf.scalar(0)));
  const optimizer = tf.train.adam(opts.lr);
  const history: EpochStats[] = [];
  let cancelled = false;

  const logits = (x: tf.Tensor): tf.Tensor => tf.add(tf.matMul(x, kernel), bias);
  // loss and accuracy of a whole set, as [loss, acc]
  const evalSet = (x: tf.Tensor, y: tf.Tensor): tf.Tensor =>
    tf.tidy(() => {
      const z = logits(x);
      const loss = tf.mean(bceWithLogits(z, y));
      const acc = tf.mean(tf.cast(tf.equal(tf.cast(tf.greaterEqual(z, 0), 'float32'), y), 'float32'));
      return tf.stack([loss, acc]);
    });

  try {
    for (let epoch = 1; epoch <= opts.epochs; epoch++) {
      const order = shuffledIndices(n, rng);
      for (let start = 0; start < n; start += batchSize) {
        if (signal?.aborted) {
          cancelled = true;
          break;
        }
        const batch = order.subarray(start, Math.min(n, start + batchSize));
        tf.tidy(() => {
          const idx = tf.tensor1d(batch, 'int32');
          const xb = tf.gather(X, idx);
          const yb = tf.gather(Y, idx);
          const wb = tf.gather(W, idx);
          optimizer.minimize(
            () => tf.mean(tf.mul(bceWithLogits(logits(xb), yb), wb)) as tf.Scalar,
            false,
            [kernel, bias],
          );
        });
      }
      if (cancelled) break;
      const stats = tf.tidy(() => tf.concat([evalSet(X, Y), evalSet(Xv, Yv)]));
      const [trainLoss, trainAcc, valLoss, valAcc] = Array.from(await stats.data());
      stats.dispose();
      const entry: EpochStats = {
        epoch,
        trainLoss: trainLoss as number,
        trainAcc: trainAcc as number,
        valLoss: valLoss as number,
        valAcc: valAcc as number,
      };
      history.push(entry);
      await opts.onEpoch?.(entry);
      await yieldToUi();
      if (signal?.aborted && epoch < opts.epochs) {
        cancelled = true;
        break;
      }
    }
    const w = Float32Array.from(await kernel.data());
    const b = (await bias.data())[0] as number;
    return { head: createHead(w, b), history, cancelled };
  } finally {
    optimizer.dispose();
    kernel.dispose();
    bias.dispose();
    X.dispose();
    Y.dispose();
    W.dispose();
    Xv.dispose();
    Yv.dispose();
  }
}
