// Independent data-integrity tests for the precomputed embedding files
// (app/public/data/embeddings/{base,stamped,aug,stress}.{f16.bin,index.json}).
// Does not import anything from scripts/ and recomputes the aug seeds with its own FNV-1a.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { decodeFloat16 } from '../../src/shared/float16.js';
import { AUG_PARAMS, DEFAULT_STAMP, STRESS_PRESET, mulberry32, randomAugmentParams } from '../../src/shared/pixelOps.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.resolve(here, '../../public/data');
const embDir = path.join(dataDir, 'embeddings');
const modelHashFile = path.resolve(here, '../../public/models/mobilenet_v2/model.sha256.json');
const DIM = 1280;

interface ManifestEntry {
  id: string;
  path: string;
  label: string;
  split: string;
  source: string;
}
interface Index {
  variant: string;
  dim: number;
  dtype: string;
  littleEndian: boolean;
  rows: number;
  bytes: number;
  binSha256: string;
  modelSha256: string;
  ids: string[];
  views?: number[];
  seeds?: number[];
  [k: string]: unknown;
}

const manifest: ManifestEntry[] = JSON.parse(readFileSync(path.join(dataDir, 'manifest.json'), 'utf8'));
const manifestIds = new Set(manifest.map((e) => e.id));
const byId = new Map(manifest.map((e) => [e.id, e]));
const modelSha: string = JSON.parse(readFileSync(modelHashFile, 'utf8')).combined;

const VARIANTS = ['base', 'stamped', 'aug', 'stress'] as const;
type Variant = (typeof VARIANTS)[number];

const jpgCache = new Map<string, string>();
function jpgHash(id: string): string {
  let h = jpgCache.get(id);
  if (!h) {
    h = createHash('sha256').update(readFileSync(path.join(dataDir, (byId.get(id) as ManifestEntry).path))).digest('hex');
    jpgCache.set(id, h);
  }
  return h;
}

const sha256 = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');

const idx = {} as Record<Variant, Index>;
const bin = {} as Record<Variant, Buffer>;
const feats = {} as Record<Variant, Float32Array>;
for (const v of VARIANTS) {
  idx[v] = JSON.parse(readFileSync(path.join(embDir, `${v}.index.json`), 'utf8'));
  bin[v] = readFileSync(path.join(embDir, `${v}.f16.bin`));
  feats[v] = decodeFloat16(bin[v]);
}

function row(v: Variant, i: number): Float32Array {
  return feats[v].subarray(i * DIM, (i + 1) * DIM);
}
function cos(a: Float32Array, b: Float32Array): number {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return d / Math.sqrt(na * nb);
}
function stats(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  const mean = xs.reduce((p, c) => p + c, 0) / xs.length;
  return { min: s[0], p05: s[Math.floor(s.length * 0.05)], median: s[Math.floor(s.length / 2)], mean, max: s[s.length - 1] };
}

/** Independent FNV-1a 32 over the UTF-8 bytes of the string. */
function fnv1a(str: string): number {
  let h = 2166136261;
  for (const byte of Buffer.from(str, 'utf8')) {
    h = (h ^ byte) >>> 0;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

// expected id lists (from the manifest, in manifest order)
const expected: Record<Variant, string[]> = {
  base: manifest.map((e) => e.id),
  stamped: manifest.map((e) => e.id),
  aug: manifest
    .filter((e) => e.split === 'train' || e.split === 'train_adult')
    .flatMap((e) => [e.id, e.id]),
  stress: manifest.filter((e) => e.source === 'kermany' && e.split === 'test').map((e) => e.id),
};
const expectedRows: Record<Variant, number> = { base: 2240, stamped: 2240, aug: 2440, stress: 624 };

describe('manifest sanity (the premise of the tests below)', () => {
  it('has 2240 unique ids', () => {
    expect(manifest.length).toBe(2240);
    expect(manifestIds.size).toBe(2240);
  });
});

describe.each(VARIANTS)('%s: files and index', (v) => {
  const i = () => idx[v];

  it('bin file size == rows * 1280 * 2', () => {
    const size = statSync(path.join(embDir, `${v}.f16.bin`)).size;
    expect(size).toBe(expectedRows[v] * DIM * 2);
    expect(size).toBe(i().rows * DIM * 2);
    expect(size).toBe(i().bytes);
  });

  it('index has the required fields with the required values', () => {
    expect(i().variant).toBe(v);
    expect(i().dim).toBe(1280);
    expect(i().dtype).toBe('float16');
    expect(i().littleEndian).toBe(true);
    expect(i().rows).toBe(expectedRows[v]);
    expect(i().modelSha256).toBe(modelSha);
    expect(typeof i().binSha256).toBe('string');
    expect(Array.isArray(i().ids)).toBe(true);
    expect(i().ids.length).toBe(i().rows);
  });

  it('binSha256 matches the real sha256 of the .bin', () => {
    expect(sha256(bin[v])).toBe(i().binSha256);
  });

  it('every id exists in manifest.json, and the id list equals the expected set in the expected order', () => {
    for (const id of i().ids) expect(manifestIds.has(id)).toBe(true);
    expect(i().ids).toEqual(expected[v]);
  });

  it('no NaN / Infinity, all values in [0, 6] (ReLU6 after average pool)', () => {
    const f = feats[v];
    expect(f.length).toBe(expectedRows[v] * DIM);
    let bad = 0;
    let mn = Infinity;
    let mx = -Infinity;
    for (let k = 0; k < f.length; k++) {
      const x = f[k];
      if (!Number.isFinite(x)) bad++;
      if (x! < mn) mn = x as number;
      if (x! > mx) mx = x as number;
    }
    expect(bad).toBe(0);
    expect(mn).toBeGreaterThanOrEqual(0);
    expect(mx).toBeLessThanOrEqual(6);
    expect(mx).toBeGreaterThan(0.5); // not a degenerate near-zero file
  });

  it('no row is all zeros, all rows have non-trivial variance, rows are only duplicated where the source JPEGs are byte-identical', () => {
    const zeroRows: number[] = [];
    const lowVar: number[] = [];
    const seen = new Map<string, number>();
    const badDups: string[] = [];
    for (let r = 0; r < expectedRows[v]; r++) {
      const x = row(v, r);
      let sum = 0;
      let nz = 0;
      for (let k = 0; k < DIM; k++) {
        sum += x[k]!;
        if (x[k] !== 0) nz++;
      }
      if (nz === 0) zeroRows.push(r);
      const mean = sum / DIM;
      let vr = 0;
      for (let k = 0; k < DIM; k++) vr += (x[k]! - mean) ** 2;
      if (vr / DIM < 1e-4) lowVar.push(r);
      const h = sha256(Buffer.from(x.buffer, x.byteOffset, x.byteLength));
      if (seen.has(h)) {
        // identical feature rows are only acceptable when the underlying images are the same file
        const a = idx[v].ids[seen.get(h) as number];
        const b = idx[v].ids[r];
        const same = a === b || jpgHash(a as string) === jpgHash(b as string);
        if (!same) badDups.push(a + ' ~ ' + b);
      } else seen.set(h, r);
    }
    expect(zeroRows).toEqual([]);
    expect(lowVar).toEqual([]);
    expect(badDups).toEqual([]);
  });
});

describe('stamped vs base', () => {
  it('differs for every row (cosine < 1) but stays similar (cosine > 0.5)', () => {
    expect(idx.stamped.ids).toEqual(idx.base.ids);
    const cs: number[] = [];
    const notLess: number[] = [];
    const low: number[] = [];
    for (let r = 0; r < 2240; r++) {
      const c = cos(row('base', r), row('stamped', r));
      cs.push(c);
      if (!(c < 1)) notLess.push(r);
      if (!(c > 0.5)) low.push(r);
    }
    // eslint-disable-next-line no-console
    console.log('stamped-vs-base cosine', JSON.stringify(stats(cs)));
    expect(notLess).toEqual([]);
    expect(low).toEqual([]);
  });

  it('stamp effect is not byte-identical anywhere', () => {
    let identical = 0;
    for (let r = 0; r < 2240; r++) {
      const a = row('base', r);
      const b = row('stamped', r);
      let same = true;
      for (let k = 0; k < DIM; k++) if (a[k] !== b[k]) { same = false; break; }
      if (same) identical++;
    }
    expect(identical).toBe(0);
  });
});

describe('stress vs base', () => {
  it('stress rows differ from the base row of the same id (cosine < 1) yet remain related (cosine > 0.3)', () => {
    const baseIndex = new Map(idx.base.ids.map((id, i) => [id, i]));
    const cs: number[] = [];
    for (let r = 0; r < 624; r++) {
      const bi = baseIndex.get(idx.stress.ids[r] as string);
      expect(bi).toBeDefined();
      cs.push(cos(row('stress', r), row('base', bi as number)));
    }
    // eslint-disable-next-line no-console
    console.log('stress-vs-base cosine', JSON.stringify(stats(cs)));
    expect(cs.filter((c) => !(c < 1)).length).toBe(0);
    expect(cs.filter((c) => !(c > 0.3)).length).toBe(0);
  });
});

describe('aug', () => {
  const a = idx.aug;

  it('views array is 0,1,0,1,...; consecutive pairs share an id; 2 views per train/train_adult image; nothing else', () => {
    expect(a.views).toBeDefined();
    const views = a.views as number[];
    expect(views.length).toBe(2440);
    for (let r = 0; r < 2440; r++) expect(views[r]).toBe(r % 2);
    for (let r = 0; r < 2440; r += 2) expect(a.ids[r]).toBe(a.ids[r + 1]);
    const counts = new Map<string, number>();
    for (const id of a.ids) counts.set(id, (counts.get(id) ?? 0) + 1);
    expect([...counts.values()].every((c) => c === 2)).toBe(true);
    expect(counts.size).toBe(1220);
    const splitCount: Record<string, number> = {};
    for (const id of counts.keys()) {
      const e = byId.get(id) as ManifestEntry;
      splitCount[`${e.source}/${e.split}`] = (splitCount[`${e.source}/${e.split}`] ?? 0) + 1;
    }
    expect(splitCount).toEqual({ 'kermany/train': 1020, 'nih/train_adult': 200 });
    // no val / test ids leaked into the training augmentation pool
    for (const id of counts.keys()) {
      const sp = (byId.get(id) as ManifestEntry).split;
      expect(['train', 'train_adult']).toContain(sp);
    }
  });

  it('view 0 != view 1 for (almost) every row, and both differ from the base row of the same id', () => {
    const baseIndex = new Map(idx.base.ids.map((id, i) => [id, i]));
    let differ = 0;
    const csViews: number[] = [];
    const csBase: number[] = [];
    for (let r = 0; r < 2440; r += 2) {
      const c = cos(row('aug', r), row('aug', r + 1));
      csViews.push(c);
      if (c < 1) differ++;
      const b = row('base', baseIndex.get(a.ids[r] as string) as number);
      csBase.push(cos(row('aug', r), b), cos(row('aug', r + 1), b));
    }
    // eslint-disable-next-line no-console
    console.log('aug view0-vs-view1 cosine', JSON.stringify(stats(csViews)), 'aug-vs-base', JSON.stringify(stats(csBase)));
    expect(differ).toBeGreaterThanOrEqual(1220 * 0.99);
    expect(csBase.filter((c) => !(c < 1)).length).toBe(0);
    expect(csBase.filter((c) => !(c > 0.2)).length).toBe(0);
  });

  it('recorded seeds equal fnv1a32("id|view") recomputed independently (all 2440 rows)', () => {
    const seeds = a.seeds as number[];
    expect(seeds.length).toBe(2440);
    const bad: string[] = [];
    for (let r = 0; r < 2440; r++) {
      const want = fnv1a(`${a.ids[r]}|${(a.views as number[])[r]}`);
      if (seeds[r] !== want) bad.push(`${r}: ${seeds[r]} != ${want}`);
    }
    expect(bad.slice(0, 5)).toEqual([]);
    expect(bad.length).toBe(0);
  });

  it('seed -> mulberry32 -> randomAugmentParams is deterministic, stays inside the documented ranges, and views of one id get different params (20 random rows)', () => {
    const seeds = a.seeds as number[];
    const pick = (() => {
      let s = 2024;
      return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
      };
    })();
    for (let k = 0; k < 20; k++) {
      const r = Math.floor(pick() * 1220) * 2;
      const p0 = randomAugmentParams(mulberry32(seeds[r] as number));
      const p0b = randomAugmentParams(mulberry32(seeds[r] as number));
      const p1 = randomAugmentParams(mulberry32(seeds[r + 1] as number));
      expect(p0).toEqual(p0b);
      expect(p0).not.toEqual(p1);
      expect(Math.abs(p0.brightness)).toBeLessThanOrEqual(AUG_PARAMS.brightness);
      expect(p0.contrast).toBeGreaterThanOrEqual(AUG_PARAMS.contrast[0]);
      expect(p0.contrast).toBeLessThanOrEqual(AUG_PARAMS.contrast[1]);
      expect(Math.abs(p0.rotationDeg)).toBeLessThanOrEqual(AUG_PARAMS.rotationDeg);
      expect(p0.cropFrac).toBeGreaterThanOrEqual(AUG_PARAMS.cropFrac[0]);
      expect(p0.cropFrac).toBeLessThanOrEqual(AUG_PARAMS.cropFrac[1]);
    }
    // index-recorded aug parameters equal the pixelOps constants
    const rec = a.augParams as { brightness: number; contrast: number[]; rotationDeg: number; cropFrac: number[] };
    expect(rec.brightness).toBe(AUG_PARAMS.brightness);
    expect([...rec.contrast]).toEqual([...AUG_PARAMS.contrast]);
    expect(rec.rotationDeg).toBe(AUG_PARAMS.rotationDeg);
    expect([...rec.cropFrac]).toEqual([...AUG_PARAMS.cropFrac]);
  });

  it('seeds are unique', () => {
    expect(new Set(a.seeds as number[]).size).toBe(2440);
  });
});

describe('files present / no stray temp files', () => {
  it('has exactly the 8 expected files in the embeddings folder', () => {
    for (const v of VARIANTS) {
      expect(existsSync(path.join(embDir, `${v}.f16.bin`))).toBe(true);
      expect(existsSync(path.join(embDir, `${v}.index.json`))).toBe(true);
      expect(existsSync(path.join(embDir, `${v}.f16.bin.tmp`))).toBe(false);
      expect(existsSync(path.join(embDir, `${v}.index.json.tmp`))).toBe(false);
    }
  });
});

describe('index files record the pixelOps constants they were built with', () => {
  // Guards against a stale stress/stamped file after STRESS_PRESET or DEFAULT_STAMP is retuned.
  it('stress.index.stressPreset equals STRESS_PRESET', () => {
    expect((idx.stress as unknown as { stressPreset: unknown }).stressPreset).toEqual(STRESS_PRESET);
  });
  it('stamped.index.stamp equals DEFAULT_STAMP', () => {
    expect((idx.stamped as unknown as { stamp: unknown }).stamp).toEqual(DEFAULT_STAMP);
  });
});
