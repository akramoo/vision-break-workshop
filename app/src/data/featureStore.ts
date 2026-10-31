// One entry point for "give me the 1280-number features of these images".
// Precomputed (shipped float16 files) when allowed and healthy, otherwise live MobileNet extraction.
// If the precomputed files fail in any way, we silently fall back to live and tell the caller via onFallback.
import * as tf from '@tensorflow/tfjs';
import { loadModelHash, type Extractor } from '../ml/extractor';
import { throwIfAborted, yieldToUi } from '../ml/yield';
import { augSeed } from '../shared/augSeed.js';
import {
  DEFAULT_STAMP,
  FEATURE_DIM,
  STRESS_PRESET,
  applyPerturbation,
  applyStamp,
  mulberry32,
  randomAugmentParams,
} from '../shared/pixelOps.js';
import { clearEmbeddingCache, getAugRows, getRowsByIds, loadVariant, rowKey, type Variant } from './embeddings';
import { loadImageTensor } from './images';
import type { ManifestEntry } from './manifest';

export interface FeatureProgress {
  phase: 'download' | 'extract';
  done: number;
  total: number;
  /** 0..1 */
  fraction: number;
  /** Estimated milliseconds left; null until a rate is known. */
  etaMs: number | null;
}

export interface GetFeaturesOptions {
  /** Manifest ids, in the order the rows are wanted. */
  ids: readonly string[];
  variant: Variant;
  usePrecomputed: boolean;
  /** Needed for the live path (image paths) and as a fallback. */
  entries: readonly ManifestEntry[];
  /** Needed for the live path. */
  extractor: Extractor | null;
  /** aug only: which views to return per id (default [0, 1]). Rows are id-major: id0v0, id0v1, id1v0, ... */
  views?: readonly number[];
  onProgress?: (p: FeatureProgress) => void;
  signal?: AbortSignal;
  /** Called once with a readable reason when precomputed features could not be used. */
  onFallback?: (reason: string) => void;
  /** Called when the precomputed files were loaded and used successfully. */
  onPrecomputedOk?: () => void;
}

export const LIVE_BATCH = 16;

/** Where the numbers actually came from (which can differ from the switch when precomputed files fail). */
export type FeatureSource = 'precomputed' | 'live';

export interface FeatureResult {
  features: Float32Array;
  source: FeatureSource;
}

/** 'live' if either part was extracted live. */
export function mergeSource(a: FeatureSource, b: FeatureSource): FeatureSource {
  return a === 'live' || b === 'live' ? 'live' : 'precomputed';
}

/** Resolve/reject like `p`, but reject with an AbortError as soon as `signal` fires (p itself keeps running for other waiters). */
export function raceAbort<T>(p: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return p;
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(new DOMException('Aborted', 'AbortError'));
    if (signal.aborted) {
      onAbort();
      return;
    }
    signal.addEventListener('abort', onAbort, { once: true });
    p.then(
      (v) => {
        signal.removeEventListener('abort', onAbort);
        resolve(v);
      },
      (e: unknown) => {
        signal.removeEventListener('abort', onAbort);
        reject(e);
      },
    );
  });
}

// ---- model hash (needed to validate precomputed features) ----
let hashPromise: Promise<string | null> | null = null;
export function getModelSha256(): Promise<string | null> {
  if (!hashPromise) {
    const p = loadModelHash().then((h) => h?.combined ?? null);
    hashPromise = p;
    void p.then((v) => {
      if (v === null && hashPromise === p) hashPromise = null;
    });
  }
  return hashPromise;
}

// ---- live feature cache (the four named variants only) ----
const liveCache = new Map<string, Float32Array>();
let msPerImage: number | null = null;

export function clearFeatureCache(): void {
  liveCache.clear();
  clearEmbeddingCache();
}

/** Measured live extraction speed (ms per image) of the last live run, or null. */
export function measuredMsPerImage(): number | null {
  return msPerImage;
}

/** How many of these items still need live extraction (not in the live cache). */
export function countUncachedLive(variant: Variant, ids: readonly string[], views: readonly number[] = [0, 1]): number {
  let n = 0;
  for (const id of ids) {
    if (variant === 'aug') {
      for (const v of views) if (!liveCache.has(`${variant}:${rowKey(variant, id, v)}`)) n++;
    } else if (!liveCache.has(`${variant}:${id}`)) n++;
  }
  return n;
}

function throttled(fn: ((p: FeatureProgress) => void) | undefined, ms = 100): ((p: FeatureProgress) => void) | undefined {
  if (!fn) return undefined;
  let last = 0;
  return (p) => {
    const now = performance.now();
    if (p.fraction >= 1 || now - last >= ms) {
      last = now;
      fn(p);
    }
  };
}

interface Item {
  id: string;
  view: number;
  key: string;
}

function itemsFor(variant: Variant, ids: readonly string[], views: readonly number[]): Item[] {
  if (variant === 'aug') {
    return ids.flatMap((id) => views.map((view) => ({ id, view, key: rowKey('aug', id, view) })));
  }
  return ids.map((id) => ({ id, view: 0, key: id }));
}

/** The pixel transform of a variant. Same maths and seeds as scripts/precompute_embeddings.mjs. */
function transform(variant: Variant, item: Item, img: tf.Tensor3D): tf.Tensor3D {
  switch (variant) {
    case 'base':
      return tf.clone(img);
    case 'stamped':
      return applyStamp(tf, img, DEFAULT_STAMP);
    case 'stress':
      return applyPerturbation(tf, img, STRESS_PRESET);
    case 'aug':
      return applyPerturbation(tf, img, randomAugmentParams(mulberry32(augSeed(item.id, item.view))));
  }
}

async function liveFeatures(opts: GetFeaturesOptions, items: Item[]): Promise<Float32Array> {
  const { variant, extractor, entries, signal } = opts;
  const byId = new Map(entries.map((e) => [e.id, e]));
  const missing = items.filter((it) => !liveCache.has(`${variant}:${it.key}`));
  if (missing.length > 0) {
    if (!extractor) throw new Error('The MobileNet model is not ready yet, so features cannot be extracted.');
    const onProgress = throttled(opts.onProgress);
    const t0 = performance.now();
    let done = 0;
    for (let start = 0; start < missing.length; start += LIVE_BATCH) {
      throwIfAborted(signal);
      const batch = missing.slice(start, start + LIVE_BATCH);
      const settled = await Promise.allSettled(
        batch.map((it) => {
          const entry = byId.get(it.id);
          if (!entry) return Promise.reject(new Error(`Unknown image id ${it.id}`));
          return loadImageTensor(entry, signal);
        }),
      );
      const loaded = settled.flatMap((s) => (s.status === 'fulfilled' ? [s.value] : []));
      const failed = settled.find((s): s is PromiseRejectedResult => s.status === 'rejected');
      if (failed) {
        loaded.forEach((t) => t.dispose());
        throw failed.reason instanceof Error ? failed.reason : new Error(String(failed.reason));
      }
      let feats: Float32Array;
      let input: tf.Tensor4D;
      try {
        input = tf.tidy(() => {
          const transformed = loaded.map((img, i) => transform(variant, batch[i] as Item, img));
          return tf.stack(transformed) as tf.Tensor4D;
        });
      } finally {
        loaded.forEach((t) => t.dispose());
      }
      try {
        feats = await extractor.extractBatch(input);
      } finally {
        input.dispose();
      }
      batch.forEach((it, i) => {
        liveCache.set(`${variant}:${it.key}`, feats.slice(i * FEATURE_DIM, (i + 1) * FEATURE_DIM));
      });
      done += batch.length;
      const elapsed = performance.now() - t0;
      msPerImage = elapsed / done;
      onProgress?.({
        phase: 'extract',
        done,
        total: missing.length,
        fraction: done / missing.length,
        etaMs: (elapsed / done) * (missing.length - done),
      });
      await yieldToUi();
    }
  }
  const out = new Float32Array(items.length * FEATURE_DIM);
  items.forEach((it, i) => {
    const row = liveCache.get(`${variant}:${it.key}`);
    if (!row) throw new Error(`Missing live features for ${it.key}`);
    out.set(row, i * FEATURE_DIM);
  });
  return out;
}

/**
 * Features [n, 1280] (row-major Float32Array) for `ids`. For variant 'aug' there are
 * ids.length * views.length rows, id-major. Throws an AbortError when `signal` fires.
 */
export async function getFeatures(opts: GetFeaturesOptions): Promise<FeatureResult> {
  const { variant, usePrecomputed, signal } = opts;
  const views = opts.views ?? [0, 1];
  throwIfAborted(signal);
  if (usePrecomputed) {
    try {
      const sha = await getModelSha256();
      const download = throttled(opts.onProgress);
      // The shared download never sees this caller's signal; this caller just stops waiting when it aborts.
      const loaded = await raceAbort(
        loadVariant(variant, {
          modelSha256: sha,
          onProgress: (fraction) =>
            download?.({ phase: 'download', done: Math.round(fraction * 100), total: 100, fraction, etaMs: null }),
        }),
        signal,
      );
      const features = variant === 'aug' ? getAugRows(loaded, opts.ids, views) : getRowsByIds(loaded, opts.ids);
      opts.onPrecomputedOk?.();
      return { features, source: 'precomputed' };
    } catch (e) {
      if (signal?.aborted) throw e;
      opts.onFallback?.(e instanceof Error ? e.message : String(e));
    }
  }
  return { features: await liveFeatures(opts, itemsFor(variant, opts.ids, views)), source: 'live' };
}

/** Download + validate one precomputed variant ahead of time (Data tab). Throws on any problem. */
export async function preloadVariant(variant: Variant, onProgress?: (fraction: number) => void): Promise<void> {
  const sha = await getModelSha256();
  await loadVariant(variant, { modelSha256: sha, ...(onProgress ? { onProgress } : {}) });
}
