// Small helpers shared by precompute_embeddings.mjs and tests/verify_embeddings.mjs.
// The augmentation seed scheme now lives in app/src/shared/augSeed.js (re-exported here).
export { AUG_SEED_SCHEME, augSeed, fnv1a32 } from '../app/src/shared/augSeed.js';

/** Cosine similarity of two equal-length float arrays (accumulated in float64). */
export function cosine(a, b, aOff = 0, bOff = 0, n = a.length) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[aOff + i];
    const y = b[bOff + i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
