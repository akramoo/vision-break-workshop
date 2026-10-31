/** Name of the seed scheme recorded in aug.index.json. */
export const AUG_SEED_SCHEME: string;
/** 32-bit FNV-1a hash of a string (over its UTF-8 bytes). */
export function fnv1a32(str: string): number;
/** Seed of the augmented view `view` (0 or 1) of the manifest entry `id`. */
export function augSeed(id: string, view: number): number;
