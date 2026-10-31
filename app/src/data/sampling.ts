// Deterministic, class-stratified sample selection (pure, no tf, no DOM).
import { mulberry32 } from '../shared/pixelOps.js';

export type SampleSize = 50 | 100 | 'all';
export const SAMPLE_SIZES: readonly SampleSize[] = [50, 100, 'all'];

function shuffled(items: readonly number[], rng: () => number): number[] {
  const a = items.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = a[i] as number;
    a[i] = a[j] as number;
    a[j] = t;
  }
  return a;
}

/**
 * Indices (ascending) of a sample of `size` items that keeps the class mix of `labels` (1 = pneumonia, else 0).
 * Same inputs and seed always give the same sample. 'all' or size >= n returns every index.
 * Each class present keeps at least one item when size >= 2.
 */
export function stratifiedSample(labels: ArrayLike<number>, size: SampleSize | number, seed = 1): number[] {
  const n = labels.length;
  const all = Array.from({ length: n }, (_, i) => i);
  if (size === 'all' || size >= n) return all;
  if (size <= 0) return [];
  const pos = all.filter((i) => labels[i] === 1);
  const neg = all.filter((i) => labels[i] !== 1);
  let takePos = Math.round((size * pos.length) / n);
  if (pos.length > 0 && neg.length > 0 && size >= 2) takePos = Math.min(Math.max(takePos, 1), size - 1);
  takePos = Math.min(takePos, pos.length);
  let takeNeg = Math.min(size - takePos, neg.length);
  takePos = Math.min(pos.length, size - takeNeg); // give unused slots back to the other class
  const rng = mulberry32(seed);
  const picked = [...shuffled(pos, rng).slice(0, takePos), ...shuffled(neg, rng).slice(0, takeNeg)];
  return picked.sort((a, b) => a - b);
}
