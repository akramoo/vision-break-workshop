// A minimal in-memory Storage for tests that exercise localStorage-backed persistence.
export class MemStorage implements Storage {
  private m = new Map<string, string>();
  /** Max total characters of values; setItem throws a QuotaExceededError-like error beyond it. */
  quota = Infinity;
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
    const used = [...this.m.entries()].reduce((n, [key, val]) => (key === k ? n : n + val.length), 0);
    if (used + String(v).length > this.quota) {
      const e = new Error('The quota has been exceeded.');
      e.name = 'QuotaExceededError';
      throw e;
    }
    this.m.set(k, String(v));
  }
  keys(): string[] {
    return [...this.m.keys()];
  }
}

/** Replace globalThis.localStorage; returns the previous value so the test can restore it. */
export function installStorage(store: Storage | undefined): Storage | undefined {
  const g = globalThis as unknown as { localStorage?: Storage };
  const original = g.localStorage;
  Object.defineProperty(globalThis, 'localStorage', { value: store, configurable: true, writable: true });
  return original;
}
