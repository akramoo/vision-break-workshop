import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_SETTINGS,
  KEY_PREFIX,
  clearPersisted,
  loadSettings,
  loadTau,
  loadTrainSettings,
  safeParse,
  saveSettings,
  saveTau,
  saveTrainSettings,
} from '../../src/state/persistence';
import { DEFAULT_TRAIN_SETTINGS } from '../../src/data/trainingPlan';

/** Minimal in-memory Storage. */
class MemStorage implements Storage {
  private m = new Map<string, string>();
  get length() {
    return this.m.size;
  }
  clear() {
    this.m.clear();
  }
  getItem(k: string) {
    return this.m.has(k) ? (this.m.get(k) as string) : null;
  }
  key(i: number) {
    return [...this.m.keys()][i] ?? null;
  }
  removeItem(k: string) {
    this.m.delete(k);
  }
  setItem(k: string, v: string) {
    this.m.set(k, String(v));
  }
}

const g = globalThis as unknown as { localStorage?: Storage };
let original: Storage | undefined;
let store: MemStorage;
const settingsKey = `${KEY_PREFIX}settings`;

beforeEach(() => {
  original = g.localStorage;
  store = new MemStorage();
  Object.defineProperty(globalThis, 'localStorage', { value: store, configurable: true, writable: true });
});
afterEach(() => {
  Object.defineProperty(globalThis, 'localStorage', { value: original, configurable: true, writable: true });
});

describe('persistence', () => {
  it('uses a versioned key prefix', () => {
    expect(KEY_PREFIX).toMatch(/^vbw:v\d+:$/);
  });

  it('defaults: usePrecomputed is ON when nothing is stored', () => {
    expect(DEFAULT_SETTINGS.usePrecomputed).toBe(true);
    expect(loadSettings()).toEqual({ usePrecomputed: true });
  });

  it('round-trips settings, writing under the versioned key', () => {
    saveSettings({ usePrecomputed: false });
    expect(store.getItem(settingsKey)).not.toBeNull();
    expect(loadSettings()).toEqual({ usePrecomputed: false });
    saveSettings({ usePrecomputed: true });
    expect(loadSettings()).toEqual({ usePrecomputed: true });
  });

  it('survives garbage / wrong-shaped stored values by returning defaults', () => {
    for (const garbage of ['{not json', '', 'null', '[]', '42', '"str"', '{"usePrecomputed":"no"}', '{"usePrecomputed":0}', '{}', '\u0000￿']) {
      store.setItem(settingsKey, garbage);
      expect(loadSettings(), `stored: ${JSON.stringify(garbage)}`).toEqual({ usePrecomputed: true });
    }
  });

  it('ignores data stored under an older/other version key', () => {
    store.setItem('vbw:v0:settings', JSON.stringify({ usePrecomputed: false }));
    store.setItem('settings', JSON.stringify({ usePrecomputed: false }));
    expect(loadSettings().usePrecomputed).toBe(true);
  });

  it('does not throw when storage is unavailable or throws', () => {
    const throwing = {
      getItem() {
        throw new Error('denied');
      },
      setItem() {
        throw new Error('quota');
      },
      removeItem() {
        throw new Error('denied');
      },
      key() {
        throw new Error('denied');
      },
      length: 3,
    } as unknown as Storage;
    Object.defineProperty(globalThis, 'localStorage', { value: throwing, configurable: true, writable: true });
    expect(loadSettings()).toEqual({ usePrecomputed: true });
    expect(() => saveSettings({ usePrecomputed: false })).not.toThrow();
    expect(() => clearPersisted()).not.toThrow();

    Object.defineProperty(globalThis, 'localStorage', { value: undefined, configurable: true, writable: true });
    expect(loadSettings()).toEqual({ usePrecomputed: true });
    expect(() => saveSettings({ usePrecomputed: false })).not.toThrow();
    expect(() => clearPersisted()).not.toThrow();
  });

  it('clearPersisted removes only vbw: keys', () => {
    saveSettings({ usePrecomputed: false });
    store.setItem('vbw:v1:log', '[]');
    store.setItem('other:key', 'keep');
    clearPersisted();
    expect(store.getItem(settingsKey)).toBeNull();
    expect(store.getItem('vbw:v1:log')).toBeNull();
    expect(store.getItem('other:key')).toBe('keep');
    expect(loadSettings().usePrecomputed).toBe(true);
  });

  it('does not store default values, so a fresh or reset session leaves no vbw: keys', () => {
    saveSettings({ ...DEFAULT_SETTINGS });
    saveTrainSettings({ ...DEFAULT_TRAIN_SETTINGS });
    saveTau(0.5);
    expect(store.length).toBe(0);
    saveSettings({ usePrecomputed: false });
    saveTrainSettings({ ...DEFAULT_TRAIN_SETTINGS, epochs: 5 });
    saveTau(0.9);
    expect(store.length).toBe(3);
    expect(loadTau()).toBe(0.9);
    expect(loadTrainSettings().epochs).toBe(5);
    // going back to the defaults removes the keys again
    saveSettings({ ...DEFAULT_SETTINGS });
    saveTrainSettings({ ...DEFAULT_TRAIN_SETTINGS });
    saveTau(0.5);
    expect(store.length).toBe(0);
  });

  it('safeParse returns the fallback for null and invalid JSON, parsed value otherwise', () => {
    expect(safeParse(null, 'fb')).toBe('fb');
    expect(safeParse('{oops', { a: 1 })).toEqual({ a: 1 });
    expect(safeParse('{"a":2}', { a: 1 })).toEqual({ a: 2 });
    expect(safeParse('false', true)).toBe(false);
  });
});
