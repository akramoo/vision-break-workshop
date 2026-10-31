// Deterministic per-(image, view) seeds for the precomputed `aug` embeddings.
// Pure JS, no Node APIs: used by scripts/precompute_embeddings.mjs (and its verifier) and,
// later, by the app to re-derive the exact augmentation parameters of a precomputed row:
//   randomAugmentParams(mulberry32(augSeed(id, view)))   (see pixelOps.js)

/** Name of the seed scheme recorded in aug.index.json. */
export const AUG_SEED_SCHEME = 'fnv1a32(`${id}|${view}`) -> mulberry32(seed) -> randomAugmentParams(rng)';

/**
 * 32-bit FNV-1a hash of a string (hashed over its UTF-8 bytes).
 * @param {string} str
 * @returns {number} uint32
 */
export function fnv1a32(str) {
  const bytes = new TextEncoder().encode(str);
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/**
 * Seed of the augmented view `view` (0 or 1) of the manifest entry `id`.
 * @param {string} id
 * @param {number} view
 * @returns {number} uint32
 */
export function augSeed(id, view) {
  return fnv1a32(`${id}|${view}`);
}
