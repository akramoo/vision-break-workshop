// Independent math audit of the app's calculations (metrics, loss/gradient/Adam, real-data pipeline).
//
// Run:   cd app && npx vitest run tests/unit/mathAudit.test.ts
// Then:  scripts/.venv/Scripts/python scripts/tests/audit_math.py
//
// The real-data part trains the app's own head (src/ml/head.ts) on the shipped float16 embeddings with the
// default recipe and writes probabilities + the app's metrics to $MATH_AUDIT_OUT (default: <tmp>/vision-math-audit)
// so the Python script can recompute every number independently and require exact agreement.
import * as tf from '@tensorflow/tfjs';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildLoadedVariant, getAugRows, getRowsByIds, parseIndex, type LoadedVariant, type Variant } from '../../src/data/embeddings';
import { EVAL_SETS, evalSetIds } from '../../src/data/evalSets';
import { idsBySplit, labelValue, parseManifest, type ManifestEntry } from '../../src/data/manifest';
import { stratifiedSample } from '../../src/data/sampling';
import { buildTrainingPlan, DEFAULT_TRAIN_SETTINGS, trainingRowLabels } from '../../src/data/trainingPlan';
import { glorotUniform, shuffledIndices, sigmoid, trainHead, type Head } from '../../src/ml/head';
import {
  accuracyOfProbs,
  alwaysPneumoniaBaseline,
  classWeights,
  decide,
  flipRate,
  formatPercent,
  metricsWithAbstain,
  rangeOf,
  worstCase,
} from '../../src/ml/metrics';
import { summarizeResults } from '../../src/ml/results';
import { replaceRows, stampedIndices, summarizeFlip } from '../../src/ml/stampLab';
import { formatDuration } from '../../src/ui/format';
import { FEATURE_DIM, mulberry32 } from '../../src/shared/pixelOps.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.resolve(here, '../../public/data');
const OUT = process.env['MATH_AUDIT_OUT'] ?? path.join(os.tmpdir(), 'vision-math-audit');
const TAUS = [0.5, 0.7, 0.8, 0.9];

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  mkdirSync(OUT, { recursive: true });
});

// ---------------------------------------------------------------------------------------------
// 1. metrics.ts closed-form checks
// ---------------------------------------------------------------------------------------------
describe('metrics closed form', () => {
  it('always-pneumonia baseline: Kermany 62.5/100/0, adult 50/100/0, bal 50', () => {
    const k = alwaysPneumoniaBaseline([...Array(234).fill(0), ...Array(390).fill(1)]);
    expect([k.accuracy, k.sensitivity, k.specificity, k.balancedAccuracy]).toEqual([0.625, 1, 0, 0.5]);
    expect(k.confusion).toEqual({ tp: 390, fn: 0, fp: 234, tn: 0 });
    const a = alwaysPneumoniaBaseline([...Array(100).fill(0), ...Array(100).fill(1)]);
    expect([a.accuracy, a.sensitivity, a.specificity]).toEqual([0.5, 1, 0]);
    expect(formatPercent(k.accuracy)).toBe('62.5%');
  });
  it('class weights N/(2Nc)', () => {
    const w = classWeights(255, 765);
    expect(w[0]).toBeCloseTo(2, 12);
    expect(w[1]).toBeCloseTo(2 / 3, 12);
    const m = classWeights(355, 865);
    expect(m[0]).toBeCloseTo(1220 / 710, 12);
    expect(m[1]).toBeCloseTo(1220 / 1730, 12);
    // mean sample weight is exactly 1 for these weights (so batch-size normalisation keeps the loss scale)
    expect((255 * w[0] + 765 * w[1]) / 1020).toBeCloseTo(1, 12);
  });
  it('decide / confidence boundaries', () => {
    expect(decide(0.5, 0.5)).toBe('pneumonia'); // p = 0.5 is pneumonia
    expect(decide(0.4999, 0.5)).toBe('normal');
    expect(decide(0.7, 0.7)).toBe('pneumonia'); // confidence == tau -> answered
    expect(1 - 0.3).toBe(0.7);
    expect(decide(0.3, 0.7)).toBe('normal'); // confidence exactly 0.7 -> answered
    expect(decide(Math.fround(0.7), 0.7)).toBe('refer'); // float32 0.7 < 0.7 (double)
  });
  it('abstain: refer-all = 0.5, tau .5 = balanced accuracy, empty class -> null', () => {
    const y = [0, 0, 0, 1, 1, 1, 1];
    const p = [0.55, 0.45, 0.4, 0.6, 0.52, 0.49, 0.9];
    const all = metricsWithAbstain(y, p, 0.99);
    expect(all.halfCreditScore).toBe(0.5);
    expect(all.accuracy).toBeNull();
    const half = metricsWithAbstain(y, p, 0.5);
    expect(half.halfCreditScore).toBeCloseTo(half.balancedAccuracy as number, 15);
    expect(metricsWithAbstain([1, 1], [0.9, 0.2], 0.5).halfCreditScore).toBeNull();
  });
  it('worstCase / flipRate / rangeOf / formatPercent', () => {
    expect(worstCase({ a: 0.6, b: null, c: 0.55 })).toEqual({ score: 0.55, set: 'c' });
    const f = flipRate([0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 1]);
    expect(f.normal).toEqual({ n: 2, flipped: 1, rate: 0.5 });
    expect(f.pneumonia).toEqual({ n: 2, flipped: 1, rate: 0.5 });
    expect(rangeOf([0.8125, null, 0.875])).toEqual({ min: 0.8125, max: 0.875, spread: 0.0625 });
    expect(formatPercent(null)).toBe('n/a');
    expect(formatPercent(-0)).toBe('0.0%');
    expect(formatPercent(13 / 16)).toBe('81.3%');
    // 0.9995 * 100 = 99.95000000000002 -> rounds up; ordinary half-up rounding
    expect(formatPercent(0.9995)).toBe('100.0%');
    expect(formatPercent(1 / 16)).toBe('6.3%');
  });
  it('formatDuration rounds seconds before splitting minutes', () => {
    expect(formatDuration(119_600)).toBe('2 min 0 s');
  });
});

// ---------------------------------------------------------------------------------------------
// 2. loss / gradient / Adam / minibatch: float64 re-implementation vs trainHead (float32 tf)
// ---------------------------------------------------------------------------------------------
const D = FEATURE_DIM;
function bce(z: number, y: number): number {
  return Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
}
/** Weighted batch loss exactly as head.ts defines it: mean over the batch of w_i * bce_i. */
function batchLoss(x: Float64Array[], y: number[], sw: number[], w: Float64Array, b: number): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    let z = b;
    for (let j = 0; j < D; j++) z += (x[i] as Float64Array)[j]! * w[j]!;
    s += sw[i]! * bce(z, y[i]!);
  }
  return s / x.length;
}
function batchGrad(x: Float64Array[], y: number[], sw: number[], w: Float64Array, b: number) {
  const gw = new Float64Array(D);
  let gb = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i] as Float64Array;
    let z = b;
    for (let j = 0; j < D; j++) z += xi[j]! * w[j]!;
    const r = (sw[i]! * (sigmoid(z) - y[i]!)) / x.length;
    for (let j = 0; j < D; j++) gw[j]! += r * xi[j]!;
    gb += r;
  }
  return { gw, gb };
}

describe('loss, gradient and Adam', () => {
  const n = 10;
  const rng = mulberry32(7);
  const xs: Float64Array[] = [];
  const ys: number[] = [];
  const flat = new Float32Array(n * D);
  for (let i = 0; i < n; i++) {
    const r = new Float64Array(D);
    ys.push(i % 3 === 0 ? 0 : 1);
    for (let j = 0; j < D; j++) {
      const v = Math.fround((rng() - 0.5) * 0.2 + (j < 20 ? (ys[i] === 1 ? 0.1 : -0.1) : 0));
      r[j] = v;
      flat[i * D + j] = v;
    }
    xs.push(r);
  }
  const cw = classWeights(4, 6);

  it('analytic gradient matches central finite differences', () => {
    const w = Float64Array.from(glorotUniform(3));
    const sw = ys.map((l) => cw[l as 0 | 1]);
    const { gw, gb } = batchGrad(xs, ys, sw, w, 0.1);
    const h = 1e-6;
    for (const j of [0, 5, 19, 700, 1279]) {
      const wp = Float64Array.from(w);
      const wm = Float64Array.from(w);
      wp[j]! += h;
      wm[j]! -= h;
      const fd = (batchLoss(xs, ys, sw, wp, 0.1) - batchLoss(xs, ys, sw, wm, 0.1)) / (2 * h);
      expect(Math.abs(fd - gw[j]!)).toBeLessThan(1e-8);
    }
    const fdb = (batchLoss(xs, ys, sw, w, 0.1 + h) - batchLoss(xs, ys, sw, w, 0.1 - h)) / (2 * h);
    expect(Math.abs(fdb - gb)).toBeLessThan(1e-8);
  });

  it('trainHead == float64 Adam(b1 .9, b2 .999, eps 1e-7) with seeded shuffle, partial last batch, class weights', async () => {
    const lr = 0.01;
    const epochs = 3;
    const batchSize = 4; // 10 rows -> batches 4, 4, 2 (partial)
    const seed = 11;
    const res = await trainHead({
      features: flat,
      labels: ys,
      valFeatures: flat,
      valLabels: ys,
      classWeight: cw,
      epochs,
      lr,
      seed,
      batchSize,
    });
    // reference
    const w = Float64Array.from(glorotUniform(seed));
    let b = 0;
    const mW = new Float64Array(D);
    const vW = new Float64Array(D);
    let mB = 0;
    let vB = 0;
    let t = 0;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-7;
    const srng = mulberry32(seed);
    for (let e = 0; e < epochs; e++) {
      const order = shuffledIndices(n, srng);
      for (let s = 0; s < n; s += batchSize) {
        const idx = Array.from(order.subarray(s, Math.min(n, s + batchSize)));
        const g = batchGrad(
          idx.map((i) => xs[i] as Float64Array),
          idx.map((i) => ys[i] as number),
          idx.map((i) => cw[ys[i] as 0 | 1]),
          w,
          b,
        );
        t++;
        const c1 = 1 - b1 ** t;
        const c2 = 1 - b2 ** t;
        for (let j = 0; j < D; j++) {
          mW[j] = b1 * mW[j]! + (1 - b1) * g.gw[j]!;
          vW[j] = b2 * vW[j]! + (1 - b2) * g.gw[j]! ** 2;
          w[j]! -= (lr * (mW[j]! / c1)) / (Math.sqrt(vW[j]! / c2) + eps);
        }
        mB = b1 * mB + (1 - b1) * g.gb;
        vB = b2 * vB + (1 - b2) * g.gb ** 2;
        b -= (lr * (mB / c1)) / (Math.sqrt(vB / c2) + eps);
      }
    }
    const got = res.head.getWeights();
    let maxDiff = 0;
    for (let j = 0; j < D; j++) maxDiff = Math.max(maxDiff, Math.abs(got.kernel[j]! - w[j]!));
    console.log(`adam: max abs diff tf vs float64 reference = ${maxDiff.toExponential(2)} (bias ${Math.abs(got.bias - b).toExponential(2)})`);
    expect(maxDiff).toBeLessThan(1e-5); // float32 vs float64 over 9 Adam steps of size ~1e-2 (measured 7.8e-7)
    expect(Math.abs(got.bias - b)).toBeLessThan(1e-5);
    // per-epoch stats use UNWEIGHTED mean BCE and accuracy at z >= 0 on the full sets
    const last = res.history[epochs - 1]!;
    let l = 0;
    let acc = 0;
    const pr = res.head.predictProba(flat);
    for (let i = 0; i < n; i++) {
      const pi = pr[i]!;
      l += -(ys[i]! * Math.log(pi) + (1 - ys[i]!) * Math.log(1 - pi));
      acc += (pi >= 0.5 ? 1 : 0) === ys[i] ? 1 : 0;
    }
    expect(last.trainLoss).toBeCloseTo(l / n, 4);
    expect(last.trainAcc).toBeCloseTo(acc / n, 6);
    res.head.dispose();
  });

  it('sigmoid is stable at extreme logits', () => {
    for (const z of [-1000, -50, 0, 50, 1000]) expect(Number.isFinite(sigmoid(z))).toBe(true);
    expect(sigmoid(0)).toBe(0.5);
  });
});

// ---------------------------------------------------------------------------------------------
// 3. real data pipeline
// ---------------------------------------------------------------------------------------------
const manifest: ManifestEntry[] = parseManifest(JSON.parse(readFileSync(path.join(dataDir, 'manifest.json'), 'utf8')));
const modelSha: string = JSON.parse(readFileSync(path.resolve(here, '../../public/models/mobilenet_v2/model.sha256.json'), 'utf8')).combined;
const loaded = {} as Record<Variant, LoadedVariant>;
function load(v: Variant): LoadedVariant {
  if (!loaded[v]) {
    const idx = parseIndex(JSON.parse(readFileSync(path.join(dataDir, 'embeddings', `${v}.index.json`), 'utf8')), v, modelSha);
    loaded[v] = buildLoadedVariant(idx, new Uint8Array(readFileSync(path.join(dataDir, 'embeddings', `${v}.f16.bin`))));
  }
  return loaded[v];
}
const byId = new Map(manifest.map((e) => [e.id, e]));
const labelsOf = (ids: string[]) => ids.map((id) => labelValue(byId.get(id) as ManifestEntry));

function metricsJson(labels: number[], probs: ArrayLike<number>) {
  const out: Record<string, unknown> = {};
  for (const tau of TAUS) out[String(tau)] = metricsWithAbstain(labels, probs, tau);
  return out;
}

describe('real data pipeline (default recipe, seed 42)', () => {
  it(
    'trains, evaluates and writes app outputs for the Python audit',
    async () => {
      const out: Record<string, unknown> = { taus: TAUS };
      // plans and weights
      const plans: Record<string, unknown> = {};
      for (const [name, patch] of [
        ['realistic', {}],
        ['balanced', { mode: 'balanced' }],
        ['balanced_adult', { mode: 'balanced', adultMix: true }],
        ['balanced_adult_aug', { mode: 'balanced', adultMix: true, augmentation: true }],
        ['realistic_aug', { augmentation: true }],
      ] as const) {
        const p = buildTrainingPlan(manifest, { ...DEFAULT_TRAIN_SETTINGS, ...patch });
        plans[name] = { nRows: p.nRows, nNormal: p.nNormal, nPneumonia: p.nPneumonia, weights: p.weights, balancedWeights: p.balancedWeights, nBase: p.baseIds.length, nVal: p.valIds.length, rowLabels: trainingRowLabels(p).length };
      }
      out['plans'] = plans;

      const plan = buildTrainingPlan(manifest, DEFAULT_TRAIN_SETTINGS);
      const base = load('base');
      const X = getRowsByIds(base, plan.baseIds);
      const Xv = getRowsByIds(base, plan.valIds);
      // aug rows are id-major and line up with trainingRowLabels
      const augPlan = buildTrainingPlan(manifest, { ...DEFAULT_TRAIN_SETTINGS, augmentation: true });
      const augRows = getAugRows(load('aug'), augPlan.baseIds);
      expect(augRows.length / D).toBe(2 * augPlan.baseIds.length);
      expect(trainingRowLabels(augPlan).slice(augPlan.baseIds.length, augPlan.baseIds.length + 4)).toEqual([
        augPlan.baseLabels[0], augPlan.baseLabels[0], augPlan.baseLabels[1], augPlan.baseLabels[1],
      ]);

      const t0 = performance.now();
      const res = await trainHead({ features: X, labels: plan.baseLabels, valFeatures: Xv, valLabels: plan.valLabels, classWeight: plan.weights, epochs: 20, lr: 1e-3, seed: 42 });
      out['trainMs'] = performance.now() - t0;
      out['history'] = res.history;
      const head = res.head;
      const w = head.getWeights();
      out['weights'] = { kernel: Array.from(w.kernel), bias: w.bias };
      out['plan'] = { baseIds: plan.baseIds, valIds: plan.valIds };

      // predictProba (JS) vs tf sigmoid(xW + b)
      const tfP = tf.tidy(() => tf.sigmoid(tf.add(tf.matMul(tf.tensor2d(Xv, [plan.valIds.length, D]), tf.tensor2d(w.kernel, [D, 1])), w.bias)).dataSync());
      const jsP = head.predictProba(Xv);
      let md = 0;
      for (let i = 0; i < jsP.length; i++) md = Math.max(md, Math.abs(jsP[i]! - tfP[i]!));
      out['predictProbaVsTfMaxAbs'] = md;
      expect(md).toBeLessThan(1e-5);
      out['valAccFromProbs'] = accuracyOfProbs(plan.valLabels, jsP);

      const sets: Record<string, unknown> = {};
      const evals: Record<string, { labels: number[]; probs: Float32Array; headId: number; usedPrecomputed: boolean; tau: number; timestamp: number }> = {};
      for (const def of EVAL_SETS) {
        const ids = evalSetIds(manifest, def);
        const labels = labelsOf(ids);
        const probs = head.predictProba(getRowsByIds(load(def.variant), ids));
        sets[def.id] = { ids, labels, probs: Array.from(probs), metrics: metricsJson(labels, probs), baseline: alwaysPneumoniaBaseline(labels) };
        evals[def.id] = { labels, probs, headId: head.id, usedPrecomputed: true, tau: 0.5, timestamp: 0 };
      }
      out['sets'] = sets;
      const summaries: Record<string, unknown> = {};
      for (const tau of TAUS) summaries[String(tau)] = summarizeResults(evals, head.id, tau);
      out['summaries'] = summaries;

      // perturb sample (Break it): stratified, seed 1
      const perturb: Record<string, unknown> = {};
      for (const setId of ['kermany_test', 'adult_test'] as const) {
        const s = sets[setId] as { labels: number[]; probs: number[] };
        for (const size of [50, 100] as const) {
          const idx = stratifiedSample(s.labels, size, 1);
          const lab = idx.map((i) => s.labels[i]!);
          perturb[`${setId}_${size}`] = {
            idx,
            nPos: lab.reduce((a, b) => a + b, 0),
            baseline: alwaysPneumoniaBaseline(lab),
            cleanMetrics: metricsWithAbstain(lab, idx.map((i) => s.probs[i]!), 0.5),
          };
        }
      }
      out['perturb'] = perturb;

      // stamp lab: before (this head) and after (head trained with the stamp on pneumonia training images)
      const testIds = idsBySplit(manifest, 'kermany', 'test');
      const testLabels = labelsOf(testIds);
      const clean = getRowsByIds(base, testIds);
      const nIdx = stampedIndices(testLabels, 'normal');
      const stampedTest = replaceRows(clean, nIdx, getRowsByIds(load('stamped'), nIdx.map((i) => testIds[i]!)));
      const pIdx = stampedIndices(plan.baseLabels, 'pneumonia');
      const stampedTrain = replaceRows(X, pIdx, getRowsByIds(load('stamped'), pIdx.map((i) => plan.baseIds[i]!)));
      const stampRes = await trainHead({ features: stampedTrain, labels: plan.baseLabels, valFeatures: Xv, valLabels: plan.valLabels, classWeight: plan.weights, epochs: 20, lr: 1e-3, seed: 42 });
      const flipOf = (h: Head) => {
        const c = h.predictProba(clean);
        const s = h.predictProba(stampedTest);
        return { cleanProbs: Array.from(c), stampedProbs: Array.from(s), summary: summarizeFlip(testLabels, c, s), cleanMetrics: metricsWithAbstain(testLabels, c, 0.5) };
      };
      out['stamp'] = { labels: testLabels, before: flipOf(head), after: flipOf(stampRes.head) };
      stampRes.head.dispose();

      // balanced mode (weighted) for the weighted-vs-unweighted comparison
      const pb = buildTrainingPlan(manifest, { ...DEFAULT_TRAIN_SETTINGS, mode: 'balanced' });
      const resB = await trainHead({ features: X, labels: pb.baseLabels, valFeatures: Xv, valLabels: pb.valLabels, classWeight: pb.weights, epochs: 20, lr: 1e-3, seed: 42 });
      const balSets: Record<string, unknown> = {};
      for (const def of EVAL_SETS) {
        const ids = evalSetIds(manifest, def);
        balSets[def.id] = { probs: Array.from(resB.head.predictProba(getRowsByIds(load(def.variant), ids))) };
      }
      out['balanced'] = { sets: balSets, history: resB.history };
      resB.head.dispose();

      // Repeat x5 (seeds 1..5): accuracy on val180, val16, test (as trainRunner.runRepeat)
      const ids16 = idsBySplit(manifest, 'kermany', 'val_official');
      const f16 = getRowsByIds(base, ids16);
      const fTest = clean;
      const repeat = [];
      for (let s = 1; s <= 5; s++) {
        const r = await trainHead({ features: X, labels: plan.baseLabels, valFeatures: Xv, valLabels: plan.valLabels, classWeight: plan.weights, epochs: 20, lr: 1e-3, seed: s });
        repeat.push({
          seed: s,
          val180: accuracyOfProbs(plan.valLabels, r.head.predictProba(Xv)),
          val16: accuracyOfProbs(labelsOf(ids16), r.head.predictProba(f16)),
          test: accuracyOfProbs(testLabels, r.head.predictProba(fTest)),
          historyValAcc: r.history[19]!.valAcc,
        });
        r.head.dispose();
      }
      out['repeat'] = repeat;
      out['repeatRange'] = {
        val16: rangeOf(repeat.map((r) => r.val16)),
        val180: rangeOf(repeat.map((r) => r.val180)),
        test: rangeOf(repeat.map((r) => r.test)),
      };

      writeFileSync(path.join(OUT, 'app_outputs.json'), JSON.stringify(out));
      // decoded base embeddings (float32) for the numpy float16 cross-check
      writeFileSync(path.join(OUT, 'base_decoded.f32'), Buffer.from(base.data.buffer, base.data.byteOffset, base.data.byteLength));
      head.dispose();

      // the numbers quoted in the docs
      const k = (sets['kermany_test'] as { metrics: Record<string, { accuracy: number; sensitivity: number; specificity: number }> }).metrics['0.5']!;
      console.log('valAcc', res.history[19]!.valAcc, 'kermany', k.accuracy, k.sensitivity, k.specificity);
    },
    600_000,
  );
});
