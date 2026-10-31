// Independent (tester-written) error-path and lookup checks for the embeddings loader (F1: "a load failure shows a
// clear banner and the app still works in live mode"). Drives the public loadVariant() with a fake fetch.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  EmbeddingsError,
  clearEmbeddingCache,
  getAugRows,
  getRowsByIds,
  loadVariant,
  rowKey,
  type Variant,
} from '../../src/data/embeddings';
import { getFeatures } from '../../src/data/featureStore';
import { AUG_SEED_SCHEME } from '../../src/shared/augSeed.js';
import { encodeFloat16ToBytes } from '../../src/shared/float16.js';
import { AUG_PARAMS, DEFAULT_STAMP, STRESS_PRESET } from '../../src/shared/pixelOps.js';

const DIM = 1280;
const SHA = 'c'.repeat(64);
const here = path.dirname(fileURLToPath(import.meta.url));
const pub = path.resolve(here, '../../public');

/** Row i holds the constant (i+1) in every column: exactly representable in float16, so row identity is obvious. */
function bin(rows: number): Uint8Array {
  const f = new Float32Array(rows * DIM);
  for (let r = 0; r < rows; r++) f.fill(r + 1, r * DIM, (r + 1) * DIM);
  return encodeFloat16ToBytes(f);
}

function index(variant: Variant, ids: string[], extra: Record<string, unknown> = {}): Record<string, unknown> {
  const base: Record<string, unknown> = {
    variant,
    dim: DIM,
    dtype: 'float16',
    rows: ids.length,
    bytes: ids.length * DIM * 2,
    modelSha256: SHA,
    ids,
  };
  if (variant === 'aug') {
    base['views'] = ids.map((_, i) => i % 2);
    base['augParams'] = AUG_PARAMS;
    base['seedScheme'] = AUG_SEED_SCHEME;
  }
  if (variant === 'stamped') base['stamp'] = DEFAULT_STAMP;
  if (variant === 'stress') base['stressPreset'] = STRESS_PRESET;
  return { ...base, ...extra };
}

type Files = Record<string, string | Uint8Array | number | 'throw'>;
function fakeFetch(files: Files): typeof fetch {
  return (async (url: string | URL | Request) => {
    const name = String(url).split('/').pop() as string;
    const f = files[name];
    if (f === 'throw') throw new TypeError('Failed to fetch');
    if (f === undefined) return new Response('not found', { status: 404 });
    if (typeof f === 'number') return new Response('err', { status: f });
    return new Response(f as BodyInit, { status: 200 });
  }) as typeof fetch;
}
const load = (v: Variant, files: Files, sha: string | null = SHA) =>
  loadVariant(v, { modelSha256: sha, baseUrl: '/e/', fetchImpl: fakeFetch(files) });

beforeEach(() => clearEmbeddingCache());

describe('every failure is a typed EmbeddingsError (so the UI can fall back to live)', () => {
  const ids = ['a', 'b', 'c'];
  const goodIdx = JSON.stringify(index('base', ids));

  it('missing index file or missing data file -> code "fetch"', async () => {
    await expect(load('base', {})).rejects.toBeInstanceOf(EmbeddingsError);
    await expect(load('base', {})).rejects.toMatchObject({ code: 'fetch' });
    await expect(load('base', { 'base.index.json': goodIdx })).rejects.toMatchObject({ code: 'fetch' });
    await expect(load('base', { 'base.index.json': goodIdx, 'base.f16.bin': 500 })).rejects.toMatchObject({ code: 'fetch' });
    await expect(load('base', { 'base.index.json': 'throw' })).rejects.toMatchObject({ code: 'fetch' });
  });

  it('data file too short or too long -> "size"', async () => {
    await expect(load('base', { 'base.index.json': goodIdx, 'base.f16.bin': bin(2) })).rejects.toMatchObject({ code: 'size' });
    await expect(load('base', { 'base.index.json': goodIdx, 'base.f16.bin': bin(4) })).rejects.toMatchObject({ code: 'size' });
    // one byte short
    const b = bin(3);
    await expect(load('base', { 'base.index.json': goodIdx, 'base.f16.bin': b.slice(0, b.length - 1) })).rejects.toMatchObject({
      code: 'size',
    });
  });

  it('index says wrong byte count -> "size"', async () => {
    const idx = JSON.stringify(index('base', ids, { bytes: 3 * DIM * 2 + 2 }));
    await expect(load('base', { 'base.index.json': idx, 'base.f16.bin': bin(3) })).rejects.toMatchObject({ code: 'size' });
  });

  it('wrong dim / wrong dtype / wrong variant name / rows != ids / bad JSON -> "format"', async () => {
    for (const bad of [
      index('base', ids, { dim: 1024 }),
      index('base', ids, { dtype: 'float32' }),
      index('base', ids, { variant: 'stress' }),
      index('base', ids, { rows: 2, bytes: 2 * DIM * 2 }),
      index('base', ids, { ids: ['a', 'b', 7] }),
    ]) {
      await expect(load('base', { 'base.index.json': JSON.stringify(bad), 'base.f16.bin': bin(3) })).rejects.toMatchObject({
        code: 'format',
      });
    }
    await expect(load('base', { 'base.index.json': '<!doctype html>', 'base.f16.bin': bin(3) })).rejects.toMatchObject({ code: 'format' });
    // SPA-style fallback: server answers 200 with HTML for a missing file
    await expect(load('base', { 'base.index.json': '<html><body>404</body></html>', 'base.f16.bin': bin(3) })).rejects.toMatchObject({
      code: 'format',
    });
  });

  it('duplicate ids in a non-aug variant -> "format"', async () => {
    const idx = JSON.stringify(index('base', ['a', 'a', 'b']));
    await expect(load('base', { 'base.index.json': idx, 'base.f16.bin': bin(3) })).rejects.toMatchObject({ code: 'format' });
  });

  it('model hash mismatch (or no hash for the loaded model) -> "model-hash"', async () => {
    const files = { 'base.index.json': goodIdx, 'base.f16.bin': bin(3) };
    await expect(load('base', files, 'd'.repeat(64))).rejects.toMatchObject({ code: 'model-hash' });
    await expect(load('base', files, null)).rejects.toMatchObject({ code: 'model-hash' });
    const noHash = index('base', ids);
    delete noHash['modelSha256'];
    await expect(load('base', { ...files, 'base.index.json': JSON.stringify(noHash) })).rejects.toMatchObject({ code: 'model-hash' });
  });

  it('stamp / stress / augmentation settings that differ from this app -> "settings-mismatch"', async () => {
    const otherStamp = { ...DEFAULT_STAMP, size: DEFAULT_STAMP.size + 8 };
    await expect(
      load('stamped', { 'stamped.index.json': JSON.stringify(index('stamped', ids, { stamp: otherStamp })), 'stamped.f16.bin': bin(3) }),
    ).rejects.toMatchObject({ code: 'settings-mismatch' });
    const otherStress = { ...STRESS_PRESET, brightness: -0.4 };
    await expect(
      load('stress', { 'stress.index.json': JSON.stringify(index('stress', ids, { stressPreset: otherStress })), 'stress.f16.bin': bin(3) }),
    ).rejects.toMatchObject({ code: 'settings-mismatch' });
    await expect(
      load('aug', { 'aug.index.json': JSON.stringify(index('aug', ids, { seedScheme: 'other' })), 'aug.f16.bin': bin(3) }),
    ).rejects.toMatchObject({ code: 'settings-mismatch' });
    await expect(
      load('aug', { 'aug.index.json': JSON.stringify(index('aug', ids, { augParams: { ...AUG_PARAMS, rotate: 99 } })), 'aug.f16.bin': bin(3) }),
    ).rejects.toMatchObject({ code: 'settings-mismatch' });
    // and the matching settings load fine
    const okStamp = await load('stamped', { 'stamped.index.json': JSON.stringify(index('stamped', ids)), 'stamped.f16.bin': bin(3) });
    expect(okStamp.rows).toBe(3);
  });

  it('a failure is not cached: fixing the file makes the next load succeed', async () => {
    await expect(load('base', {})).rejects.toBeInstanceOf(EmbeddingsError);
    const ok = await load('base', { 'base.index.json': goodIdx, 'base.f16.bin': bin(3) });
    expect(ok.rows).toBe(3);
  });
});

describe('rows are looked up by id, never by file position', () => {
  const ids = ['r0', 'r1', 'r2', 'r3', 'r4'];
  const files = { 'base.index.json': JSON.stringify(index('base', ids)), 'base.f16.bin': bin(5) };

  it('any permutation of requested ids returns the matching rows (row i holds value i+1)', async () => {
    const v = await load('base', files);
    const want = ['r3', 'r0', 'r4', 'r1'];
    const out = getRowsByIds(v, want);
    expect(out.length).toBe(4 * DIM);
    expect(Array.from({ length: 4 }, (_, i) => out[i * DIM])).toEqual([4, 1, 5, 2]);
    // every column of a row is the same value (no row bleed / off-by-one stride)
    for (let i = 0; i < 4; i++) {
      expect(out[i * DIM]).toBe(out[i * DIM + DIM - 1]);
    }
    // a subset in reverse
    expect(Array.from(getRowsByIds(v, ['r4', 'r3']).filter((_, i) => i % DIM === 0))).toEqual([5, 4]);
  });

  it('shuffling the ids in the FILE (index permuted together with the data) still returns the right rows', async () => {
    // The data file holds rows [row for 'z', row for 'y', row for 'x'] = values 1, 2, 3
    const shuffled = ['z', 'y', 'x'];
    const v = await load('base', { 'base.index.json': JSON.stringify(index('base', shuffled)), 'base.f16.bin': bin(3) });
    expect(getRowsByIds(v, ['x'])[0]).toBe(3);
    expect(getRowsByIds(v, ['y'])[0]).toBe(2);
    expect(getRowsByIds(v, ['z'])[0]).toBe(1);
    expect(getRowsByIds(v, ['x', 'y', 'z'])[0]).toBe(3);
  });

  it('an unknown id throws "missing-id" (not a silent zero row)', async () => {
    const v = await load('base', files);
    expect(() => getRowsByIds(v, ['r1', 'nope'])).toThrow(EmbeddingsError);
    try {
      getRowsByIds(v, ['nope']);
    } catch (e) {
      expect((e as EmbeddingsError).code).toBe('missing-id');
    }
  });

  it('aug rows are keyed id|view and returned id-major', async () => {
    // file order: (p,0)=1 (p,1)=2 (q,0)=3 (q,1)=4 -- views list alternates 0,1 but ids differ per pair
    const augIdx = index('aug', ['p', 'p', 'q', 'q'], { views: [0, 1, 0, 1] });
    const v = await load('aug', { 'aug.index.json': JSON.stringify(augIdx), 'aug.f16.bin': bin(4) });
    expect(rowKey('aug', 'p', 1)).toBe('p|1');
    expect(v.keyToRow.get('p|0')).toBe(0);
    expect(v.keyToRow.get('q|1')).toBe(3);
    const out = getAugRows(v, ['q', 'p']);
    expect(Array.from({ length: 4 }, (_, i) => out[i * DIM])).toEqual([3, 4, 1, 2]); // q0 q1 p0 p1
    expect(getAugRows(v, ['p'], [1])[0]).toBe(2); // only view 1 of p
    expect(() => getAugRows(v, ['p'], [2])).toThrow(/p\|2/);
    // a base-style lookup by bare id must NOT work on the aug variant
    expect(() => getRowsByIds(v, ['p'])).toThrow(EmbeddingsError);
  });

  it('duplicate (id, view) pairs in an aug index are rejected', async () => {
    const augIdx = index('aug', ['p', 'p'], { views: [0, 0] });
    await expect(load('aug', { 'aug.index.json': JSON.stringify(augIdx), 'aug.f16.bin': bin(2) })).rejects.toMatchObject({ code: 'format' });
  });
});

describe('the shipped embedding files are complete for the shipped manifest', () => {
  const manifest = JSON.parse(readFileSync(path.join(pub, 'data/manifest.json'), 'utf8')) as {
    id: string;
    split: string;
    source: string;
  }[];
  const idx = (v: string) => JSON.parse(readFileSync(path.join(pub, `data/embeddings/${v}.index.json`), 'utf8')) as {
    ids: string[];
    views?: number[];
    rows: number;
  };

  it('base covers every manifest id exactly once', () => {
    const b = idx('base');
    expect(new Set(b.ids).size).toBe(b.ids.length);
    expect(new Set(b.ids)).toEqual(new Set(manifest.map((m) => m.id)));
  });

  it('stress covers exactly the 624 Kermany test ids', () => {
    const s = idx('stress');
    const test = manifest.filter((m) => m.source === 'kermany' && m.split === 'test').map((m) => m.id);
    expect(test).toHaveLength(624);
    expect(new Set(s.ids)).toEqual(new Set(test));
  });

  it('aug has both views for every training-pool image (Kermany train + adult pool)', () => {
    const a = idx('aug');
    const keys = new Set(a.ids.map((id, i) => `${id}|${a.views?.[i]}`));
    expect(keys.size).toBe(a.rows);
    const pool = manifest.filter((m) => m.split === 'train' || m.split === 'train_adult');
    expect(pool).toHaveLength(1220);
    for (const m of pool) for (const v of [0, 1]) expect(keys.has(`${m.id}|${v}`)).toBe(true);
  });

  it('the shipped base file really loads from disk and its rows are finite with the right length', async () => {
    const b = idx('base');
    const sha = JSON.parse(readFileSync(path.join(pub, 'data/embeddings/base.index.json'), 'utf8')).modelSha256 as string;
    const bytes = new Uint8Array(readFileSync(path.join(pub, 'data/embeddings/base.f16.bin')));
    const v = await loadVariant('base', {
      modelSha256: sha,
      baseUrl: '/real/',
      fetchImpl: (async (u: string | URL | Request) => {
        const name = String(u).split('/').pop() as string;
        return name.endsWith('.json')
          ? new Response(readFileSync(path.join(pub, 'data/embeddings', name), 'utf8'))
          : new Response(bytes);
      }) as typeof fetch,
    });
    expect(v.rows).toBe(b.rows);
    const row = getRowsByIds(v, [b.ids[1234] as string]);
    expect(row.length).toBe(DIM);
    expect(Array.from(row).every(Number.isFinite)).toBe(true);
    expect(Array.from(row).some((x) => x !== 0)).toBe(true); // MobileNet features after ReLU6 average pool: non-zero
    // features are post-ReLU6 average pool, so they live in [0, 6]
    expect(Math.min(...row)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...row)).toBeLessThanOrEqual(6.01);
  });
});

describe('getFeatures falls back (and says so) when precomputed features are unusable', () => {
  it('invokes onFallback exactly once with a readable reason, then goes to the live path', async () => {
    let reasons: string[] = [];
    // vitest has no relative-URL fetch; the model-hash lookup and the embedding fetch both fail -> typed error -> fallback.
    const p = getFeatures({
      ids: ['x'],
      variant: 'base',
      usePrecomputed: true,
      entries: [],
      extractor: null,
      onFallback: (r) => reasons.push(r),
    });
    // live path with no extractor and an unknown id: must reject with a clear message, not hang or crash silently
    await expect(p).rejects.toThrow(/MobileNet model is not ready/);
    expect(reasons).toHaveLength(1);
    expect(reasons[0]?.length).toBeGreaterThan(10);
    reasons = [];
    // precomputed switched OFF: no fallback notice, straight to live
    await expect(
      getFeatures({ ids: ['x'], variant: 'base', usePrecomputed: false, entries: [], extractor: null, onFallback: (r) => reasons.push(r) }),
    ).rejects.toThrow(/MobileNet model is not ready/);
    expect(reasons).toEqual([]);
  });
});
