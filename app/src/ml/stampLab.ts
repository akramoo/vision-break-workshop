// Stamp lab (F6): train a head on stamped images of one class and measure how often a stamp flips predictions.
// Pure helpers (no tf) are unit-tested; the feature plumbing reuses featureStore for the default stamp
// (precomputed `stamped` variant) and extracts live for custom stamp settings.
import * as tf from '@tensorflow/tfjs';
import { evalSetIds, evalSetDef } from '../data/evalSets';
import { LIVE_BATCH, getFeatures, mergeSource, type FeatureProgress, type FeatureResult, type FeatureSource } from '../data/featureStore';
import { loadImageTensor } from '../data/images';
import { labelValue, type ManifestEntry } from '../data/manifest';
import { buildTrainingPlan, type TrainSettings } from '../data/trainingPlan';
import { DEFAULT_STAMP, FEATURE_DIM, applyStamp, resolveStampConfig, type StampConfig } from '../shared/pixelOps.js';
import type { Extractor } from './extractor';
import { trainHead, type EpochStats, type Head, type TrainHeadResult } from './head';
import { flipRate, hardPredictions, ratio, type FlipRate } from './metrics';
import { throwIfAborted, yieldToUi } from './yield';

export type StampTarget = 'pneumonia' | 'normal' | 'both';
export type StampClass = StampTarget;

export const STAMP_TARGETS: readonly { id: StampTarget; label: string }[] = [
  { id: 'pneumonia', label: 'Pneumonia images' },
  { id: 'normal', label: 'Normal images' },
  { id: 'both', label: 'Both classes' },
];

/** The stamp-trained model ("after"). The normal Train result stays in session.head ("before"). */
export interface StampedHeadRecord {
  head: Head;
  config: StampConfig;
  target: StampTarget;
  trainedAt: number;
  /** True only if every feature file really came from the precomputed files (not the switch alone). */
  usedPrecomputed: boolean;
  /** The training recipe that was used (the same as the normal model's, when there is one). */
  recipe: Pick<TrainSettings, 'mode' | 'adultMix' | 'epochs' | 'lr' | 'seed'>;
  completedEpochs: number;
  history: EpochStats[];
  durationMs: number;
}

export function sameStampConfig(a: Partial<StampConfig>, b: Partial<StampConfig>): boolean {
  const x = resolveStampConfig(a);
  const y = resolveStampConfig(b);
  return (
    x.size === y.size &&
    x.position === y.position &&
    x.offsetX === y.offsetX &&
    x.offsetY === y.offsetY &&
    Math.abs(x.opacity - y.opacity) < 1e-9
  );
}

/** True when the precomputed `stamped` embeddings can be used (default stamp + precomputed switch on). */
export function isInstantStamp(cfg: Partial<StampConfig>, usePrecomputed: boolean): boolean {
  return usePrecomputed && sameStampConfig(cfg, DEFAULT_STAMP);
}

/** Does the stamp go on an image with this label? */
export function targetMatches(target: StampTarget, label: number): boolean {
  return target === 'both' || (target === 'pneumonia' ? label === 1 : label !== 1);
}

/** Indices of the rows whose label is in the stamped class. */
export function stampedIndices(labels: ArrayLike<number>, target: StampTarget): number[] {
  const out: number[] = [];
  for (let i = 0; i < labels.length; i++) if (targetMatches(target, labels[i] as number)) out.push(i);
  return out;
}

/** Copy of `clean` [n, 1280] where the rows listed in `indices` are replaced by `rows` [indices.length, 1280]. */
export function replaceRows(clean: Float32Array, indices: readonly number[], rows: Float32Array): Float32Array {
  if (rows.length !== indices.length * FEATURE_DIM) throw new Error('replaceRows: rows do not match indices');
  const out = Float32Array.from(clean);
  indices.forEach((row, k) => out.set(rows.subarray(k * FEATURE_DIM, (k + 1) * FEATURE_DIM), row * FEATURE_DIM));
  return out;
}

export interface ClassCalls {
  n: number;
  /** Share predicted pneumonia, images unstamped. */
  pneumoniaClean: number | null;
  /** Share predicted pneumonia when the stamp is applied (only to the stamped class). */
  pneumoniaStamped: number | null;
  /** Share whose prediction changed. */
  flip: FlipRate;
}

export interface FlipSummary {
  normal: ClassCalls;
  pneumonia: ClassCalls;
  overall: FlipRate;
}

/** Per-true-class flip rate and pneumonia-call rates from probabilities on clean and stamped copies of the same images. */
export function summarizeFlip(labels: ArrayLike<number>, cleanProbs: ArrayLike<number>, stampedProbs: ArrayLike<number>): FlipSummary {
  const clean = hardPredictions(cleanProbs);
  const stamped = hardPredictions(stampedProbs);
  const f = flipRate(labels, clean, stamped);
  const calls = (cls: 0 | 1, flip: FlipRate): ClassCalls => {
    let n = 0;
    let c = 0;
    let s = 0;
    for (let i = 0; i < labels.length; i++) {
      if ((labels[i] === 1 ? 1 : 0) !== cls) continue;
      n++;
      c += clean[i] as number;
      s += stamped[i] as number;
    }
    return { n, pneumoniaClean: ratio(c, n), pneumoniaStamped: ratio(s, n), flip };
  };
  return { normal: calls(0, f.normal), pneumonia: calls(1, f.pneumonia), overall: f.overall };
}

// ---- features of stamped images ----
const customCache = new Map<string, Float32Array>();

export function clearStampCache(): void {
  customCache.clear();
}

function cfgKey(cfg: StampConfig): string {
  return `${cfg.size}|${cfg.position}|${cfg.offsetX}|${cfg.offsetY}|${cfg.opacity}`;
}

export interface StampedFeatureOptions {
  ids: readonly string[];
  cfg: Partial<StampConfig>;
  usePrecomputed: boolean;
  entries: readonly ManifestEntry[];
  extractor: Extractor | null;
  signal?: AbortSignal;
  onProgress?: (p: FeatureProgress) => void;
  onFallback?: (reason: string) => void;
  onPrecomputedOk?: () => void;
}

/**
 * Features [ids.length, 1280] of the images with the stamp applied. The default stamp uses the precomputed `stamped`
 * variant (instant) or the feature store's live path; any other stamp is extracted live here (cached per config).
 */
export async function getStampedFeatures(o: StampedFeatureOptions): Promise<FeatureResult> {
  const cfg = resolveStampConfig(o.cfg);
  if (sameStampConfig(cfg, DEFAULT_STAMP)) {
    return getFeatures({
      ids: o.ids,
      variant: 'stamped',
      usePrecomputed: o.usePrecomputed,
      entries: o.entries,
      extractor: o.extractor,
      ...(o.signal ? { signal: o.signal } : {}),
      ...(o.onProgress ? { onProgress: o.onProgress } : {}),
      ...(o.onFallback ? { onFallback: o.onFallback } : {}),
      ...(o.onPrecomputedOk ? { onPrecomputedOk: o.onPrecomputedOk } : {}),
    });
  }
  const key = cfgKey(cfg);
  const byId = new Map(o.entries.map((e) => [e.id, e]));
  const missing = o.ids.filter((id) => !customCache.has(`${key}:${id}`));
  if (missing.length > 0) {
    if (!o.extractor) throw new Error('The MobileNet model is not ready yet, so features cannot be extracted.');
    const t0 = performance.now();
    let done = 0;
    for (let start = 0; start < missing.length; start += LIVE_BATCH) {
      throwIfAborted(o.signal);
      const batch = missing.slice(start, start + LIVE_BATCH);
      // allSettled: if one image fails, the ones that did load are disposed instead of leaking.
      const settled = await Promise.allSettled(
        batch.map((id) => {
          const e = byId.get(id);
          if (!e) return Promise.reject(new Error(`Unknown image id ${id}`));
          return loadImageTensor(e, o.signal);
        }),
      );
      const imgs = settled.flatMap((s) => (s.status === 'fulfilled' ? [s.value] : []));
      const failed = settled.find((s): s is PromiseRejectedResult => s.status === 'rejected');
      if (failed) {
        imgs.forEach((t) => t.dispose());
        throw failed.reason instanceof Error ? failed.reason : new Error(String(failed.reason));
      }
      let input: tf.Tensor4D;
      try {
        input = tf.tidy(() => tf.stack(imgs.map((t) => applyStamp(tf, t, cfg))) as tf.Tensor4D);
      } finally {
        imgs.forEach((t) => t.dispose());
      }
      let feats: Float32Array;
      try {
        feats = await (o.extractor as Extractor).extractBatch(input);
      } finally {
        input.dispose();
      }
      batch.forEach((id, i) => customCache.set(`${key}:${id}`, feats.slice(i * FEATURE_DIM, (i + 1) * FEATURE_DIM)));
      done += batch.length;
      const elapsed = performance.now() - t0;
      o.onProgress?.({
        phase: 'extract',
        done,
        total: missing.length,
        fraction: done / missing.length,
        etaMs: (elapsed / done) * (missing.length - done),
      });
      await yieldToUi();
    }
  }
  const out = new Float32Array(o.ids.length * FEATURE_DIM);
  o.ids.forEach((id, i) => {
    const row = customCache.get(`${key}:${id}`);
    if (!row) throw new Error(`Missing stamped features for ${id}`);
    out.set(row, i * FEATURE_DIM);
  });
  return { features: out, source: 'live' };
}

// ---- training ----
export interface StampTrainArgs {
  entries: readonly ManifestEntry[];
  extractor: Extractor | null;
  usePrecomputed: boolean;
  settings: TrainSettings;
  cfg: StampConfig;
  target: StampTarget;
  signal: AbortSignal;
  onProgress?: (p: FeatureProgress) => void;
  onEpoch?: (s: EpochStats) => void;
  onFallback?: (reason: string) => void;
  onPrecomputedOk?: () => void;
}

/** Features of the training rows: clean everywhere, except the stamped class which uses stamped features. */
export function mixTrainingFeatures(labels: ArrayLike<number>, target: StampTarget, base: Float32Array, stamped: Float32Array): Float32Array {
  return replaceRows(base, stampedIndices(labels, target), stamped);
}

/** Train a NEW head on Kermany train (+ adult train if the mix is on) with the stamp on the chosen class. */
export async function trainStampedHead(a: StampTrainArgs): Promise<TrainHeadResult & { source: FeatureSource }> {
  const plan = buildTrainingPlan(a.entries, { ...a.settings, augmentation: false });
  const common = {
    entries: a.entries,
    extractor: a.extractor,
    usePrecomputed: a.usePrecomputed,
    signal: a.signal,
    ...(a.onProgress ? { onProgress: a.onProgress } : {}),
    ...(a.onFallback ? { onFallback: a.onFallback } : {}),
    ...(a.onPrecomputedOk ? { onPrecomputedOk: a.onPrecomputedOk } : {}),
  };
  const base = await getFeatures({ ids: plan.baseIds, variant: 'base', ...common });
  const idx = stampedIndices(plan.baseLabels, a.target);
  const stamped = await getStampedFeatures({ ids: idx.map((i) => plan.baseIds[i] as string), cfg: a.cfg, ...common });
  const features = replaceRows(base.features, idx, stamped.features);
  const val = await getFeatures({ ids: plan.valIds, variant: 'base', ...common });
  const source = mergeSource(mergeSource(base.source, stamped.source), val.source);
  const result = await trainHead({
    features,
    labels: plan.baseLabels,
    valFeatures: val.features,
    valLabels: plan.valLabels,
    classWeight: plan.weights,
    epochs: a.settings.epochs,
    lr: a.settings.lr,
    seed: a.settings.seed,
    signal: a.signal,
    ...(a.onEpoch ? { onEpoch: a.onEpoch } : {}),
  });
  return { ...result, source };
}

// ---- flip rate ----
export type FlipSetId = 'kermany_test' | 'adult_test';

export interface HeadFlip {
  cleanProbs: Float32Array;
  stampedProbs: Float32Array;
  summary: FlipSummary;
}

export interface FlipResult {
  setId: FlipSetId;
  stampClass: StampClass;
  config: StampConfig;
  ids: string[];
  labels: number[];
  before: HeadFlip;
  after: HeadFlip | null;
  beforeHeadId: number;
  afterHeadId: number | null;
  /** Where the features really came from. */
  source: FeatureSource;
  durationMs: number;
}

export interface FlipArgs {
  setId: FlipSetId;
  stampClass: StampClass;
  cfg: StampConfig;
  before: Head;
  after: Head | null;
  entries: readonly ManifestEntry[];
  extractor: Extractor | null;
  usePrecomputed: boolean;
  signal: AbortSignal;
  onProgress?: (p: FeatureProgress) => void;
  onFallback?: (reason: string) => void;
  onPrecomputedOk?: () => void;
}

export async function computeFlip(a: FlipArgs): Promise<FlipResult> {
  const t0 = performance.now();
  const def = evalSetDef(a.setId);
  const ids = evalSetIds(a.entries, def);
  const byId = new Map(a.entries.map((e) => [e.id, e]));
  const labels = ids.map((id) => labelValue(byId.get(id) as ManifestEntry));
  const common = {
    entries: a.entries,
    extractor: a.extractor,
    usePrecomputed: a.usePrecomputed,
    signal: a.signal,
    ...(a.onProgress ? { onProgress: a.onProgress } : {}),
    ...(a.onFallback ? { onFallback: a.onFallback } : {}),
    ...(a.onPrecomputedOk ? { onPrecomputedOk: a.onPrecomputedOk } : {}),
  };
  const cleanRes = await getFeatures({ ids, variant: 'base', ...common });
  const clean = cleanRes.features;
  const idx = stampedIndices(labels, a.stampClass);
  const stampedRes = await getStampedFeatures({ ids: idx.map((i) => ids[i] as string), cfg: a.cfg, ...common });
  const stamped = replaceRows(clean, idx, stampedRes.features);
  const run = (h: Head): HeadFlip => {
    const cleanProbs = h.predictProba(clean);
    const stampedProbs = h.predictProba(stamped);
    return { cleanProbs, stampedProbs, summary: summarizeFlip(labels, cleanProbs, stampedProbs) };
  };
  return {
    setId: a.setId,
    stampClass: a.stampClass,
    config: a.cfg,
    ids,
    labels,
    before: run(a.before),
    after: a.after ? run(a.after) : null,
    beforeHeadId: a.before.id,
    afterHeadId: a.after ? a.after.id : null,
    source: mergeSource(cleanRes.source, stampedRes.source),
    durationMs: performance.now() - t0,
  };
}
