import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { LOG_KEY } from '../../src/state/persistence';
import {
  LOG_TAGS,
  MAX_HYPOTHESIS_CHARS,
  MAX_LOG_ENTRIES,
  MAX_THUMB_CHARS,
  TAG_INFO,
  buildLogConfig,
  buildLogExport,
  estimateLogSize,
  labelName,
  loadLog,
  logExportFileName,
  makeLogEntry,
  saveLog,
  tagCounts,
  validateEntries,
  validateEntry,
  type LogDraft,
  type LogEntry,
} from '../../src/state/failureLog';
import { MemStorage, installStorage } from './storageStub';

/** A realistic 96 px q0.7 JPEG data URL is about 4-6 KB of base64; use a 6 KB one. */
const THUMB = `data:image/jpeg;base64,${'A'.repeat(6000)}`;

const MODEL = { mode: 'realistic', adultMix: false, augmentation: false, epochs: 20, lr: 0.001, seed: 42 } as const;

function draft(over: Partial<LogDraft> = {}): LogDraft {
  return {
    thumbnail: THUMB,
    setName: 'kermany_test',
    imageId: 'kermany-test-normal-0001',
    trueLabel: 'normal',
    predictedLabel: 'pneumonia',
    confidence: 0.93,
    probPneumonia: 0.93,
    referred: false,
    config: buildLogConfig(MODEL, 0.5, true, { stamped: true, stamp: { size: 24, position: 'top-left' } }),
    tag: 'imbalance',
    hypothesis: 'Three times more pneumonia in training.',
    ...over,
  };
}

let store: MemStorage;
let original: Storage | undefined;
beforeEach(() => {
  store = new MemStorage();
  original = installStorage(store);
});
afterEach(() => {
  installStorage(original);
});

describe('tags', () => {
  it('has exactly the seven tags, in order', () => {
    expect([...LOG_TAGS]).toEqual(['imbalance', 'shortcut', 'shift', 'quality', 'closed-world', 'evaluation', 'other']);
  });
  it('explains every tag in one line', () => {
    for (const t of LOG_TAGS) {
      expect(TAG_INFO[t].length).toBeGreaterThan(20);
      expect(TAG_INFO[t]).not.toContain('\n');
    }
    expect(Object.keys(TAG_INFO).sort()).toEqual([...LOG_TAGS].sort());
  });
  it('counts entries by tag, zero for unused tags', () => {
    const entries = [draft({ tag: 'shift' }), draft({ tag: 'shift' }), draft({ tag: 'other' })].map((d) => makeLogEntry(d));
    expect(tagCounts(entries)).toEqual({ imbalance: 0, shortcut: 0, shift: 2, quality: 0, 'closed-world': 0, evaluation: 0, other: 1 });
  });
});

describe('schema validator', () => {
  const good = (): LogEntry => makeLogEntry(draft(), 1_700_000_000_000);

  it('accepts a well-formed entry and returns an equal clean copy', () => {
    const e = good();
    expect(validateEntry(JSON.parse(JSON.stringify(e)))).toEqual(e);
  });

  it('makeLogEntry assigns a unique id and the timestamp, and trims the hypothesis', () => {
    const a = makeLogEntry(draft({ hypothesis: '  because  ' }), 5);
    const b = makeLogEntry(draft(), 5);
    expect(a.createdAt).toBe(5);
    expect(a.hypothesis).toBe('because');
    expect(a.id).not.toBe(b.id);
  });

  it('accepts a missing trueLabel as null and keeps the "no finding" name', () => {
    const e = JSON.parse(JSON.stringify(good()));
    delete e.trueLabel;
    expect(validateEntry(e)?.trueLabel).toBeNull();
    e.trueLabel = 'no finding';
    e.predictedLabel = 'no finding';
    expect(validateEntry(e)?.trueLabel).toBe('no finding');
  });

  it('rejects malformed entries', () => {
    const base = JSON.parse(JSON.stringify(good()));
    const bad: Record<string, unknown>[] = [
      { ...base, id: '' },
      { ...base, id: 5 },
      { ...base, createdAt: 'yesterday' },
      { ...base, createdAt: Number.NaN },
      { ...base, thumbnail: 'http://evil/x.jpg' },
      { ...base, thumbnail: `data:image/jpeg;base64,${'A'.repeat(MAX_THUMB_CHARS)}` },
      { ...base, setName: '' },
      { ...base, trueLabel: 'maybe' },
      { ...base, predictedLabel: 'refer' },
      { ...base, probPneumonia: 1.2 },
      { ...base, probPneumonia: '0.5' },
      { ...base, confidence: -0.1 },
      { ...base, referred: 'no' },
      { ...base, tag: 'whatever' },
      { ...base, tag: undefined },
      { ...base, hypothesis: 42 },
      { ...base, config: null },
      { ...base, config: { ...base.config, mode: 'fancy' } },
      { ...base, config: { ...base.config, tau: 'high' } },
      { ...base, config: { ...base.config, usePrecomputed: 1 } },
    ];
    for (const b of bad) expect(validateEntry(b), JSON.stringify(b).slice(0, 80)).toBeNull();
    for (const junk of [null, undefined, 3, 'x', [], true]) expect(validateEntry(junk)).toBeNull();
  });

  it('drops unknown top-level fields and truncates an over-long hypothesis', () => {
    const e = { ...JSON.parse(JSON.stringify(good())), extra: 'x', hypothesis: 'h'.repeat(MAX_HYPOTHESIS_CHARS + 50) };
    const v = validateEntry(e);
    expect(v).not.toBeNull();
    expect(v).not.toHaveProperty('extra');
    expect(v?.hypothesis).toHaveLength(MAX_HYPOTHESIS_CHARS);
  });

  it('validateEntries keeps the good ones, drops malformed and duplicate ids, and caps at 100', () => {
    const a = good();
    const many = Array.from({ length: 130 }, (_, i) => makeLogEntry(draft(), i));
    expect(validateEntries([a, { nope: 1 }, a, 'x'])).toEqual([a]);
    expect(validateEntries(many)).toHaveLength(MAX_LOG_ENTRIES);
    expect(validateEntries('not an array')).toEqual([]);
  });

  it('labelName maps to the three label names', () => {
    expect(labelName(true)).toBe('pneumonia');
    expect(labelName(false)).toBe('normal');
    expect(labelName(false, 'no finding')).toBe('no finding');
    expect(labelName(true, 'no finding')).toBe('pneumonia');
  });

  it('buildLogConfig carries the model settings, tau, feature mode and the extras', () => {
    const c = buildLogConfig(MODEL, 0.8, false, { perturb: { brightness: -0.2 } });
    expect(c).toMatchObject({ ...MODEL, tau: 0.8, usePrecomputed: false, perturb: { brightness: -0.2 } });
  });
});

describe('export', () => {
  it('has the documented shape and the disclaimer', () => {
    const entries = [makeLogEntry(draft(), 10), makeLogEntry(draft({ tag: 'shortcut' }), 11)];
    const out = buildLogExport(entries, new Date('2026-11-01T09:05:00Z'), '9.9.9');
    expect(Object.keys(out).sort()).toEqual(['appVersion', 'disclaimer', 'entries', 'exportedAt']);
    expect(out.disclaimer).toBe('Educational only. Not for clinical use.');
    expect(out.exportedAt).toBe('2026-11-01T09:05:00.000Z');
    expect(out.appVersion).toBe('9.9.9');
    expect(out.entries).toEqual(entries);
    // survives JSON and re-validation
    expect(validateEntries(JSON.parse(JSON.stringify(out)).entries)).toEqual(entries);
  });

  it('uses the real app version by default', () => {
    expect(buildLogExport([]).appVersion).toMatch(/^\d+\.\d+\.\d+/);
  });

  it('names the file vision-failures-YYYYMMDD-HHMM.json (local time)', () => {
    expect(logExportFileName(new Date(2026, 10, 2, 9, 5))).toBe('vision-failures-20261102-0905.json');
    expect(logExportFileName(new Date(2026, 0, 31, 23, 59))).toBe('vision-failures-20260131-2359.json');
    expect(logExportFileName()).toMatch(/^vision-failures-\d{8}-\d{4}\.json$/);
  });
});

describe('persistence', () => {
  it('round-trips under the versioned key', () => {
    const entries = [makeLogEntry(draft(), 1), makeLogEntry(draft({ tag: 'quality', trueLabel: null }), 2)];
    expect(LOG_KEY).toBe('vbw:v1:log');
    expect(saveLog(entries)).toBe(true);
    expect(store.getItem('vbw:v1:log')).not.toBeNull();
    expect(loadLog()).toEqual(entries);
  });

  it('an empty log removes the key', () => {
    saveLog([makeLogEntry(draft())]);
    saveLog([]);
    expect(store.getItem(LOG_KEY)).toBeNull();
    expect(loadLog()).toEqual([]);
  });

  it('drops malformed entries on load and survives garbage', () => {
    const ok = makeLogEntry(draft(), 3);
    store.setItem(LOG_KEY, JSON.stringify([ok, { id: 'x' }, 7]));
    expect(loadLog()).toEqual([ok]);
    for (const g of ['{oops', 'null', '{}', '"str"', '']) {
      store.setItem(LOG_KEY, g);
      expect(loadLog()).toEqual([]);
    }
  });

  it('reports a quota error instead of throwing', () => {
    store.quota = 1000;
    expect(saveLog([makeLogEntry(draft())])).toBe(false);
    expect(store.getItem(LOG_KEY)).toBeNull();
  });

  it('reports failure when storage is blocked', () => {
    installStorage(undefined);
    expect(saveLog([makeLogEntry(draft())])).toBe(false);
    expect(loadLog()).toEqual([]);
  });
});

describe('size', () => {
  it('100 entries with realistic thumbnails and long hypotheses stay under 1 MB', () => {
    const entries = Array.from({ length: MAX_LOG_ENTRIES }, (_, i) =>
      makeLogEntry(draft({ hypothesis: 'x'.repeat(MAX_HYPOTHESIS_CHARS), imageId: `kermany-test-pneumonia-${String(i).padStart(4, '0')}` }), i),
    );
    const size = estimateLogSize(entries);
    expect(size).toBeLessThan(1_000_000);
    // and it really fits in a quota of 1 MB of characters
    store.quota = 1_000_000;
    expect(saveLog(entries)).toBe(true);
    expect(loadLog()).toHaveLength(MAX_LOG_ENTRIES);
  });

  it('even the largest allowed thumbnail keeps 100 entries under 1 MB', () => {
    const max = `data:image/jpeg;base64,${'A'.repeat(MAX_THUMB_CHARS - 23)}`;
    const entries = Array.from({ length: MAX_LOG_ENTRIES }, (_, i) =>
      makeLogEntry(draft({ thumbnail: max, hypothesis: 'x'.repeat(MAX_HYPOTHESIS_CHARS) }), i),
    );
    expect(max.length).toBe(MAX_THUMB_CHARS);
    expect(estimateLogSize(entries)).toBeLessThan(1_000_000);
  });
});
