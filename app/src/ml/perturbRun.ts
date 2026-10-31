// Live perturbation experiment (F5): perturb a sample of images, run MobileNet + the head, abortable between batches.
// The decoded 224x224 pixels of the sample are cached as Float32Array (at most CACHE_LIMIT images, ~600 KB each).
import * as tf from '@tensorflow/tfjs';
import { loadImageTensor } from '../data/images';
import { getUploadPixels } from '../data/uploads';
import { IDENTITY_PERTURBATION, applyPerturbation, type PerturbationParams } from '../shared/pixelOps.js';
import type { Extractor } from './extractor';
import type { Head } from './head';
import { pixelsToBatch } from './predictImage';
import { throwIfAborted, yieldToUi } from './yield';

export const PERTURB_BATCH = 16;
export const CACHE_LIMIT = 100;

export interface PerturbItem {
  /** Manifest id or upload id. */
  id: string;
  /** 1 = pneumonia, 0 = normal, null = unlabeled (uploads only). */
  label: 0 | 1 | null;
  /** Path under public/data/ for shipped images; absent for uploads. */
  path?: string;
}

export interface PerturbRunStats {
  started: number;
  completed: number;
  aborted: number;
}

/** Counters for tests / the debug panel: how many runs started, finished and were cancelled by a newer slider move. */
export const perturbStats: PerturbRunStats = { started: 0, completed: 0, aborted: 0 };

const pixelCache = new Map<string, Float32Array>();
const baselineCache = new Map<string, Float32Array>();

export function clearPerturbCache(): void {
  pixelCache.clear();
  baselineCache.clear();
}

export function isIdentity(p: PerturbationParams): boolean {
  const i = IDENTITY_PERTURBATION;
  return p.brightness === i.brightness && p.contrast === i.contrast && p.rotationDeg === i.rotationDeg && p.cropFrac === i.cropFrac;
}

async function pixelsOf(item: PerturbItem, cacheable: boolean, signal: AbortSignal): Promise<Float32Array> {
  if (!item.path) {
    const px = getUploadPixels(item.id);
    if (!px) throw new Error(`The uploaded image ${item.id} is no longer available.`);
    return px;
  }
  const hit = pixelCache.get(item.id);
  if (hit) return hit;
  const t = await loadImageTensor({ path: item.path }, signal);
  try {
    const data = (await t.data()) as Float32Array;
    if (cacheable && pixelCache.size < CACHE_LIMIT) pixelCache.set(item.id, data);
    return data;
  } finally {
    t.dispose();
  }
}

export interface PerturbRunArgs {
  items: readonly PerturbItem[];
  params: PerturbationParams;
  extractor: Extractor;
  head: Head;
  /** Identifies (set, sample) so the unperturbed predictions are reused between slider moves. */
  cacheKey: string;
  signal: AbortSignal;
  onProgress?: (done: number, total: number) => void;
}

export interface PerturbRunResult {
  /** P(pneumonia) per item, unperturbed, from the same pipeline. */
  baseProbs: Float32Array;
  /** P(pneumonia) per item with `params` applied. */
  probs: Float32Array;
  durationMs: number;
}

/**
 * One pass over the sample in batches of 16: load pixels, (unperturbed predictions if not cached), perturbed predictions.
 * Throws an AbortError as soon as `signal` fires (checked before and after every batch). Identity parameters reuse
 * the unperturbed predictions, so they reproduce them exactly.
 */
export async function runPerturbation(a: PerturbRunArgs): Promise<PerturbRunResult> {
  const { items, params, extractor, head, signal } = a;
  perturbStats.started++;
  const t0 = performance.now();
  const key = `${a.cacheKey}|head${head.id}`;
  const cachedBase = baselineCache.get(key);
  const needBase = !cachedBase;
  const identity = isIdentity(params);
  const cacheable = items.length <= CACHE_LIMIT;
  const baseProbs = cachedBase ?? new Float32Array(items.length);
  const probs = new Float32Array(items.length);
  try {
    for (let start = 0; start < items.length; start += PERTURB_BATCH) {
      throwIfAborted(signal);
      const chunk = items.slice(start, start + PERTURB_BATCH);
      const px = await Promise.all(chunk.map((it) => pixelsOf(it, cacheable, signal)));
      throwIfAborted(signal);
      const x = pixelsToBatch(px);
      let xp: tf.Tensor4D | null = null;
      try {
        if (needBase) baseProbs.set(head.predictProba(await extractor.extractBatch(x)), start);
        if (identity) {
          probs.set(baseProbs.subarray(start, start + chunk.length), start);
        } else {
          throwIfAborted(signal);
          xp = applyPerturbation(tf, x, params);
          probs.set(head.predictProba(await extractor.extractBatch(xp)), start);
        }
      } finally {
        x.dispose();
        xp?.dispose();
      }
      throwIfAborted(signal);
      a.onProgress?.(Math.min(items.length, start + chunk.length), items.length);
      await yieldToUi();
    }
    throwIfAborted(signal);
    if (needBase) baselineCache.set(key, baseProbs);
    perturbStats.completed++;
    return { baseProbs, probs, durationMs: performance.now() - t0 };
  } catch (e) {
    if (signal.aborted) perturbStats.aborted++;
    throw e;
  }
}

/** Pixels of one item with `params` applied, as flat float32 [224*224*3] (for the before/after preview). */
export async function perturbedPixels(item: PerturbItem, params: PerturbationParams, signal: AbortSignal): Promise<{ before: Float32Array; after: Float32Array }> {
  const before = await pixelsOf(item, true, signal);
  const out = tf.tidy(() => applyPerturbation(tf, tf.tensor3d(before, [224, 224, 3]), params));
  try {
    return { before, after: (await out.data()) as Float32Array };
  } finally {
    out.dispose();
  }
}
