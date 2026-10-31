import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  EmbeddingsError,
  buildLoadedVariant,
  clearEmbeddingCache,
  getAugRows,
  getRowsByIds,
  loadVariant,
  parseIndex,
  rowKey,
  sameJson,
  type EmbeddingIndex,
} from '../../src/data/embeddings';
import { AUG_SEED_SCHEME } from '../../src/shared/augSeed.js';
import { encodeFloat16ToBytes } from '../../src/shared/float16.js';
import { AUG_PARAMS, DEFAULT_STAMP, STRESS_PRESET } from '../../src/shared/pixelOps.js';

const DIM = 1280;
const SHA = 'a'.repeat(64);
const here = path.dirname(fileURLToPath(import.meta.url));
const embDir = path.resolve(here, '../../public/data/embeddings');
const modelSha: string = JSON.parse(
  readFileSync(path.resolve(here, '../../public/models/mobilenet_v2/model.sha256.json'), 'utf8'),
).combined;

function baseIndex(extra: Record<string, unknown> = {}, ids = ['a', 'b', 'c']): Record<string, unknown> {
  return {
    variant: 'base',
    dim: DIM,
    dtype: 'float16',
    rows: ids.length,
    bytes: ids.length * DIM * 2,
    modelSha256: SHA,
    ids,
    ...extra,
  };
}

/** Row i is filled with the value i + 1 (exactly representable in float16). */
function binFor(rows: number): Uint8Array {
  const f = new Float32Array(rows * DIM);
  for (let r = 0; r < rows; r++) f.fill(r + 1, r * DIM, (r + 1) * DIM);
  return encodeFloat16ToBytes(f);
}

describe('parseIndex', () => {
  it('accepts a good base index', () => {
    const idx = parseIndex(baseIndex(), 'base', SHA);
    expect(idx.rows).toBe(3);
    expect(idx.ids).toEqual(['a', 'b', 'c']);
  });

  it('fails with model-hash on a different model, or when our model has no hash', () => {
    expect(() => parseIndex(baseIndex(), 'base', 'b'.repeat(64))).toThrow(EmbeddingsError);
    try {
      parseIndex(baseIndex(), 'base', 'b'.repeat(64));
    } catch (e) {
      expect((e as EmbeddingsError).code).toBe('model-hash');
    }
    expect(() => parseIndex(baseIndex(), 'base', null)).toThrow(/hash/);
    expect(() => parseIndex(baseIndex({ modelSha256: undefined }), 'base', SHA)).toThrow(/hash/);
  });

  it('checks dtype, dim, rows, bytes and the variant name', () => {
    expect(() => parseIndex(baseIndex({ dtype: 'float32' }), 'base', SHA)).toThrow(/dtype/);
    expect(() => parseIndex(baseIndex({ dim: 512 }), 'base', SHA)).toThrow(/dim/);
    expect(() => parseIndex(baseIndex({ rows: 4 }), 'base', SHA)).toThrow(/ids/);
    expect(() => parseIndex(baseIndex({ bytes: 7 }), 'base', SHA)).toThrow(/bytes/);
    expect(() => parseIndex(baseIndex({ variant: 'stamped' }), 'base', SHA)).toThrow(/variant/);
    expect(() => parseIndex(null, 'base', SHA)).toThrow(EmbeddingsError);
    expect(() => parseIndex('x', 'base', SHA)).toThrow(EmbeddingsError);
  });

  it('treats a stamp / stress / augmentation mismatch like a hash mismatch', () => {
    const stamped = (stamp: unknown) => baseIndex({ variant: 'stamped', stamp });
    expect(() => parseIndex(stamped({ ...DEFAULT_STAMP }), 'stamped', SHA)).not.toThrow();
    expect(() => parseIndex(stamped({ ...DEFAULT_STAMP, size: 32 }), 'stamped', SHA)).toThrow(/stamp/);
    expect(() => parseIndex(stamped(undefined), 'stamped', SHA)).toThrow(/stamp/);

    const stress = (stressPreset: unknown) => baseIndex({ variant: 'stress', stressPreset });
    expect(() => parseIndex(stress({ ...STRESS_PRESET }), 'stress', SHA)).not.toThrow();
    expect(() => parseIndex(stress({ ...STRESS_PRESET, rotationDeg: 9 }), 'stress', SHA)).toThrow(/stress/);

    const aug = (extra: Record<string, unknown>) =>
      baseIndex({ variant: 'aug', views: [0, 1, 0], augParams: AUG_PARAMS, seedScheme: AUG_SEED_SCHEME, ...extra }, ['a', 'a', 'b']);
    expect(() => parseIndex(aug({}), 'aug', SHA)).not.toThrow();
    expect(() => parseIndex(aug({ augParams: { ...AUG_PARAMS, brightness: 0.9 } }), 'aug', SHA)).toThrow(/augmentation/);
    expect(() => parseIndex(aug({ seedScheme: 'other' }), 'aug', SHA)).toThrow(/seed/);
    expect(() => parseIndex(aug({ views: undefined }), 'aug', SHA)).toThrow(/views/);
  });

  it('accepts the shipped index files for the real model (settings and hash agree with this app)', () => {
    for (const v of ['base', 'stamped', 'aug', 'stress'] as const) {
      const raw: unknown = JSON.parse(readFileSync(path.join(embDir, `${v}.index.json`), 'utf8'));
      const idx = parseIndex(raw, v, modelSha);
      expect(idx.rows).toBe(idx.ids.length);
    }
  });
});

describe('sameJson', () => {
  it('compares plain JSON ignoring key order', () => {
    expect(sameJson({ a: 1, b: [1, 2] }, { b: [1, 2], a: 1 })).toBe(true);
    expect(sameJson({ a: 1 }, { a: 2 })).toBe(false);
    expect(sameJson({ a: 1 }, { a: 1, b: 1 })).toBe(false);
    expect(sameJson([1, 2], [1, 2, 3])).toBe(false);
    expect(sameJson(null, {})).toBe(false);
  });
});

describe('rows are mapped by id, never by position', () => {
  const idx = parseIndex(baseIndex({}, ['c', 'a', 'b']), 'base', SHA);
  const loaded = buildLoadedVariant(idx, binFor(3));

  it('returns the rows in the order requested', () => {
    const rows = getRowsByIds(loaded, ['a', 'c', 'b', 'a']);
    expect(rows).toHaveLength(4 * DIM);
    // file order is c, a, b -> fill values 1, 2, 3
    expect(rows[0]).toBe(2); // a
    expect(rows[DIM]).toBe(1); // c
    expect(rows[2 * DIM]).toBe(3); // b
    expect(rows[3 * DIM]).toBe(2); // a again
    expect(rows[DIM - 1]).toBe(2);
  });

  it('throws missing-id for an unknown id', () => {
    expect(() => getRowsByIds(loaded, ['a', 'zzz'])).toThrow(/zzz/);
    try {
      getRowsByIds(loaded, ['zzz']);
    } catch (e) {
      expect((e as EmbeddingsError).code).toBe('missing-id');
    }
  });

  it('checks the byte length (rows * 1280 * 2)', () => {
    expect(() => buildLoadedVariant(idx, binFor(2))).toThrow(/bytes/);
    expect(() => buildLoadedVariant(idx, new Uint8Array(3 * DIM * 2 + 2))).toThrow(EmbeddingsError);
  });

  it('rejects duplicate ids in a non-aug variant', () => {
    const dup = parseIndex(baseIndex({}, ['a', 'a', 'b']), 'base', SHA);
    expect(() => buildLoadedVariant(dup, binFor(3))).toThrow(/duplicate/);
  });

  it('keys aug rows by id|view', () => {
    const augIdx: EmbeddingIndex = parseIndex(
      baseIndex({ variant: 'aug', views: [0, 1, 0, 1], augParams: AUG_PARAMS, seedScheme: AUG_SEED_SCHEME }, ['x', 'x', 'y', 'y']),
      'aug',
      SHA,
    );
    const aug = buildLoadedVariant(augIdx, binFor(4));
    expect(rowKey('aug', 'x', 1)).toBe('x|1');
    const rows = getAugRows(aug, ['y', 'x']);
    expect(rows).toHaveLength(4 * DIM);
    expect([rows[0], rows[DIM], rows[2 * DIM], rows[3 * DIM]]).toEqual([3, 4, 1, 2]); // y0 y1 x0 x1
    expect(getAugRows(aug, ['x'], [1])[0]).toBe(2);
    expect(() => getAugRows(aug, ['x'], [2])).toThrow(/x\|2/);
  });
});

describe('loadVariant (injected fetch)', () => {
  beforeEach(() => clearEmbeddingCache());

  function fakeFetch(files: Record<string, string | Uint8Array | number>): typeof fetch {
    return (async (url: string | URL | Request) => {
      const u = String(url);
      const name = u.slice(u.lastIndexOf('/') + 1);
      const f = files[name];
      if (f === undefined || typeof f === 'number') return new Response('nope', { status: typeof f === 'number' ? f : 404 });
      return new Response(f as BodyInit, { status: 200 });
    }) as typeof fetch;
  }

  const good = (): Record<string, string | Uint8Array> => ({
    'base.index.json': JSON.stringify(baseIndex()),
    'base.f16.bin': binFor(3),
  });

  it('loads, validates, caches and reports progress', async () => {
    let calls = 0;
    const inner = fakeFetch(good());
    const counting = ((...a: Parameters<typeof fetch>) => {
      calls++;
      return inner(...a);
    }) as typeof fetch;
    const seen: number[] = [];
    const v = await loadVariant('base', { modelSha256: SHA, baseUrl: '/x/', fetchImpl: counting, onProgress: (f) => seen.push(f) });
    expect(v.rows).toBe(3);
    expect(seen[seen.length - 1]).toBe(1);
    expect(getRowsByIds(v, ['b'])[0]).toBe(2);
    const before = calls;
    await loadVariant('base', { modelSha256: SHA, baseUrl: '/x/', fetchImpl: counting });
    expect(calls).toBe(before); // served from the cache
  });

  it('fails (and does not cache the failure) on HTTP errors, truncated data and a model mismatch', async () => {
    await expect(loadVariant('base', { modelSha256: SHA, baseUrl: '/x/', fetchImpl: fakeFetch({}) })).rejects.toMatchObject({
      code: 'fetch',
    });
    await expect(
      loadVariant('base', {
        modelSha256: SHA,
        baseUrl: '/x/',
        fetchImpl: fakeFetch({ 'base.index.json': JSON.stringify(baseIndex()), 'base.f16.bin': binFor(2) }),
      }),
    ).rejects.toMatchObject({ code: 'size' });
    await expect(loadVariant('base', { modelSha256: 'f'.repeat(64), baseUrl: '/x/', fetchImpl: fakeFetch(good()) })).rejects.toMatchObject({
      code: 'model-hash',
    });
    await expect(
      loadVariant('base', { modelSha256: SHA, baseUrl: '/x/', fetchImpl: fakeFetch({ 'base.index.json': '<html>', 'base.f16.bin': 404 }) }),
    ).rejects.toMatchObject({ code: 'format' });
    // a later good attempt works: failures were not cached
    const ok = await loadVariant('base', { modelSha256: SHA, baseUrl: '/x/', fetchImpl: fakeFetch(good()) });
    expect(ok.rows).toBe(3);
  });
});
