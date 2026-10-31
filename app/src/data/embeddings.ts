// Loader for the shipped MobileNet features (public/data/embeddings/<variant>.index.json + .f16.bin).
// Rows are ALWAYS looked up by id, never by position. Any problem (missing file, wrong size,
// other model, different stamp / stress / augmentation settings) throws EmbeddingsError so the
// caller can fall back to live extraction. No crypto.subtle: it is unavailable on insecure origins.
import { AUG_SEED_SCHEME } from '../shared/augSeed.js';
import { decodeFloat16 } from '../shared/float16.js';
import { AUG_PARAMS, DEFAULT_STAMP, FEATURE_DIM, STRESS_PRESET } from '../shared/pixelOps.js';

export type Variant = 'base' | 'stamped' | 'aug' | 'stress';
export const VARIANTS: readonly Variant[] = ['base', 'stamped', 'aug', 'stress'];

export type EmbeddingsErrorCode = 'fetch' | 'format' | 'size' | 'model-hash' | 'settings-mismatch' | 'missing-id';

export class EmbeddingsError extends Error {
  readonly code: EmbeddingsErrorCode;
  constructor(code: EmbeddingsErrorCode, message: string) {
    super(message);
    this.name = 'EmbeddingsError';
    this.code = code;
  }
}

export interface EmbeddingIndex {
  variant: Variant;
  dim: number;
  dtype: string;
  rows: number;
  bytes: number;
  modelSha256: string;
  ids: string[];
  views?: number[];
  stamp?: unknown;
  stressPreset?: unknown;
  augParams?: unknown;
  seedScheme?: unknown;
}

export interface LoadedVariant {
  variant: Variant;
  rows: number;
  /** Decoded rows, row-major [rows, 1280]. */
  data: Float32Array;
  /** Row key -> row number. The key is `id`, or `${id}|${view}` for the aug variant. */
  keyToRow: Map<string, number>;
}

/** Key of a row: the id, or `${id}|${view}` for the aug variant. */
export function rowKey(variant: Variant, id: string, view = 0): string {
  return variant === 'aug' ? `${id}|${view}` : id;
}

/** Deep equality for plain JSON values (key order does not matter). */
export function sameJson(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((v, i) => sameJson(v, b[i]));
  }
  const ka = Object.keys(a as object);
  const kb = Object.keys(b as object);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => k in (b as object) && sameJson((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/**
 * Validate a parsed index.json. `modelSha256` is the hash of the model that is actually loaded;
 * a different value means the features came from another model and must not be used.
 */
export function parseIndex(raw: unknown, variant: Variant, modelSha256: string | null): EmbeddingIndex {
  const bad = (code: EmbeddingsErrorCode, msg: string) => new EmbeddingsError(code, `${variant}: ${msg}`);
  if (typeof raw !== 'object' || raw === null) throw bad('format', 'the index file is not an object.');
  const x = raw as Record<string, unknown>;
  if (x['variant'] !== variant) throw bad('format', `the index describes variant "${String(x['variant'])}".`);
  if (x['dim'] !== FEATURE_DIM) throw bad('format', `dim is ${String(x['dim'])}, expected ${FEATURE_DIM}.`);
  if (x['dtype'] !== 'float16') throw bad('format', `dtype is ${String(x['dtype'])}, expected float16.`);
  const rows = x['rows'];
  const bytes = x['bytes'];
  const ids = x['ids'];
  if (typeof rows !== 'number' || !Number.isInteger(rows) || rows < 0) throw bad('format', 'rows is invalid.');
  if (!Array.isArray(ids) || ids.length !== rows || !ids.every((i) => typeof i === 'string')) {
    throw bad('format', 'ids does not match rows.');
  }
  if (bytes !== rows * FEATURE_DIM * 2) throw bad('size', `bytes (${String(bytes)}) is not rows * ${FEATURE_DIM} * 2.`);
  const sha = x['modelSha256'];
  if (typeof sha !== 'string') throw bad('model-hash', 'the index has no model hash.');
  if (modelSha256 === null) throw bad('model-hash', 'the loaded model has no hash to compare with.');
  if (sha !== modelSha256) throw bad('model-hash', 'these features were computed with a different model.');

  let views: number[] | undefined;
  if (variant === 'aug') {
    const v = x['views'];
    if (!Array.isArray(v) || v.length !== rows || !v.every((n) => typeof n === 'number')) {
      throw bad('format', 'the aug index has no valid views list.');
    }
    views = v as number[];
    if (!sameJson(x['augParams'], AUG_PARAMS)) throw bad('settings-mismatch', 'augmentation settings differ from this app.');
    if (x['seedScheme'] !== AUG_SEED_SCHEME) throw bad('settings-mismatch', 'augmentation seed scheme differs from this app.');
  } else if (variant === 'stamped') {
    if (!sameJson(x['stamp'], DEFAULT_STAMP)) throw bad('settings-mismatch', 'the stamp settings differ from this app.');
  } else if (variant === 'stress') {
    if (!sameJson(x['stressPreset'], STRESS_PRESET)) throw bad('settings-mismatch', 'the stress preset differs from this app.');
  }

  return {
    variant,
    dim: FEATURE_DIM,
    dtype: 'float16',
    rows,
    bytes,
    modelSha256: sha,
    ids: ids as string[],
    ...(views ? { views } : {}),
  };
}

/** Decode the binary payload and build the key -> row map. */
export function buildLoadedVariant(index: EmbeddingIndex, bin: Uint8Array): LoadedVariant {
  if (bin.byteLength !== index.rows * FEATURE_DIM * 2) {
    throw new EmbeddingsError(
      'size',
      `${index.variant}: the data file has ${bin.byteLength} bytes, expected ${index.rows * FEATURE_DIM * 2}.`,
    );
  }
  const keyToRow = new Map<string, number>();
  for (let i = 0; i < index.rows; i++) {
    const id = index.ids[i] as string;
    const key = rowKey(index.variant, id, index.views?.[i] ?? 0);
    if (keyToRow.has(key)) throw new EmbeddingsError('format', `${index.variant}: duplicate row key ${key}.`);
    keyToRow.set(key, i);
  }
  return { variant: index.variant, rows: index.rows, data: decodeFloat16(bin), keyToRow };
}

/**
 * Copy the rows for `keys` (see rowKey) into a new Float32Array [keys.length, 1280], in the
 * order requested. Throws EmbeddingsError('missing-id') if any key is absent.
 */
export function getRowsByKeys(loaded: LoadedVariant, keys: readonly string[]): Float32Array {
  const out = new Float32Array(keys.length * FEATURE_DIM);
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i] as string;
    const row = loaded.keyToRow.get(key);
    if (row === undefined) throw new EmbeddingsError('missing-id', `${loaded.variant}: no precomputed row for ${key}.`);
    out.set(loaded.data.subarray(row * FEATURE_DIM, (row + 1) * FEATURE_DIM), i * FEATURE_DIM);
  }
  return out;
}

/** Rows for manifest ids (non-aug variants). */
export function getRowsByIds(loaded: LoadedVariant, ids: readonly string[]): Float32Array {
  return getRowsByKeys(loaded, ids);
}

/** Rows for the aug variant: for each id, one row per view, id-major (id0v0, id0v1, id1v0, ...). */
export function getAugRows(loaded: LoadedVariant, ids: readonly string[], views: readonly number[] = [0, 1]): Float32Array {
  const keys: string[] = [];
  for (const id of ids) for (const v of views) keys.push(rowKey('aug', id, v));
  return getRowsByKeys(loaded, keys);
}

export interface LoadOptions {
  /** Hash of the loaded model; null makes the load fail with 'model-hash'. */
  modelSha256: string | null;
  /** Called with the fraction (0..1) of the data file downloaded (only for the caller that started the download). */
  onProgress?: (fraction: number) => void;
  /** Defaults to `${import.meta.env.BASE_URL}data/embeddings/`. */
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

function defaultBaseUrl(): string {
  return `${import.meta.env.BASE_URL}data/embeddings/`;
}

async function fetchChecked(fetchImpl: typeof fetch, url: string, signal?: AbortSignal): Promise<Response> {
  let res: Response;
  try {
    res = await fetchImpl(url, signal ? { signal } : undefined);
  } catch (e) {
    if (signal?.aborted) throw e;
    throw new EmbeddingsError('fetch', `Could not download ${url} (${e instanceof Error ? e.message : String(e)}).`);
  }
  if (!res.ok) throw new EmbeddingsError('fetch', `${url} returned HTTP ${res.status}.`);
  return res;
}

async function fetchBinary(
  fetchImpl: typeof fetch,
  url: string,
  expected: number,
  onProgress: ((f: number) => void) | undefined,
  signal: AbortSignal | undefined,
): Promise<Uint8Array> {
  const res = await fetchChecked(fetchImpl, url, signal);
  const reader = res.body?.getReader();
  if (!reader) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(1);
    return buf;
  }
  const out = new Uint8Array(expected);
  let got = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.byteLength > expected) {
        void reader.cancel();
        throw new EmbeddingsError('size', `${url} is larger than the ${expected} bytes its index promises.`);
      }
      out.set(value, got);
      got += value.byteLength;
      onProgress?.(expected > 0 ? got / expected : 1);
    }
  } catch (e) {
    if (e instanceof EmbeddingsError || signal?.aborted) throw e;
    throw new EmbeddingsError('fetch', `Download of ${url} failed (${e instanceof Error ? e.message : String(e)}).`);
  }
  if (got !== expected) throw new EmbeddingsError('size', `${url} has ${got} bytes, expected ${expected}.`);
  return out;
}

const cache = new Map<Variant, Promise<LoadedVariant>>();

/**
 * Load (once) and cache one variant. A failed load is not cached, so a later call retries.
 * Concurrent callers share the same download. The download is deliberately NOT tied to any caller's AbortSignal
 * (the files are at most ~6 MB): if the first caller gives up, the others still get the data. Callers that want
 * to stop waiting race the returned promise against their own signal.
 */
export function loadVariant(variant: Variant, opts: LoadOptions): Promise<LoadedVariant> {
  const hit = cache.get(variant);
  if (hit) return hit;
  const base = opts.baseUrl ?? defaultBaseUrl();
  const fetchImpl = opts.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  const p = (async () => {
    const res = await fetchChecked(fetchImpl, `${base}${variant}.index.json`);
    let json: unknown;
    try {
      json = JSON.parse(await res.text());
    } catch {
      throw new EmbeddingsError('format', `${variant}: the index file is not valid JSON.`);
    }
    const index = parseIndex(json, variant, opts.modelSha256);
    const bin = await fetchBinary(fetchImpl, `${base}${variant}.f16.bin`, index.bytes, opts.onProgress, undefined);
    return buildLoadedVariant(index, bin);
  })();
  cache.set(variant, p);
  p.catch(() => {
    if (cache.get(variant) === p) cache.delete(variant);
  });
  return p;
}

export function isVariantLoaded(variant: Variant): boolean {
  return cache.has(variant);
}

export function clearEmbeddingCache(): void {
  cache.clear();
}
