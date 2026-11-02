// Persist the trained head (1281 numbers) in IndexedDB so a page refresh does not lose it.
// Every function swallows errors: persistence is a convenience and must never break the app.
import { FEATURE_DIM } from '../shared/pixelOps.js';
import { currentGeneration } from './generation';
import type { EpochStats } from './head';

const DB_NAME = 'vbw';
const STORE = 'head';
const KEY = 'current';

/** JSON-safe description of how a head was trained. */
export interface HeadInfo {
  trainedAt: number;
  seed: number;
  epochs: number;
  completedEpochs: number;
  lr: number;
  mode: 'realistic' | 'balanced';
  adultMix: boolean;
  augmentation: boolean;
  valSet: 'val' | 'val_official';
  nRows: number;
  nNormal: number;
  nPneumonia: number;
  classWeights: { normal: number; pneumonia: number } | null;
  usedPrecomputed: boolean;
  durationMs: number;
  /** Validation accuracy after the last epoch (optional: heads saved by older versions do not have it). */
  finalValAcc?: number;
  restored: boolean;
}

export interface SavedHead {
  kernel: Float32Array;
  bias: number;
  info: HeadInfo;
  history: EpochStats[];
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB is not available'));
      return;
    }
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB error'));
    req.onblocked = () => reject(new Error('IndexedDB is blocked'));
  });
}

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const req = run(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(req.result);
      tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
      tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
    });
  } finally {
    db.close();
  }
}

/**
 * Store the head. `generation` is the session generation the caller started in: if Reset session has run since,
 * nothing is written (checked at the moment the transaction is created, so it also orders correctly against
 * the reset's own delete).
 */
export async function saveHead(rec: SavedHead, generation: number = currentGeneration()): Promise<boolean> {
  try {
    let skipped = false;
    await withStore('readwrite', (s) => {
      if (currentGeneration() !== generation) {
        skipped = true;
        return s.count() as unknown as IDBRequest<IDBValidKey>; // a harmless read keeps the transaction valid
      }
      return s.put({ ...rec, kernel: Float32Array.from(rec.kernel) }, KEY);
    });
    return !skipped;
  } catch {
    return false;
  }
}

/** Check a stored record's shape; null if it is not a valid head. */
export function validateSavedHead(raw: unknown): SavedHead | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const r = raw as Partial<SavedHead>;
  if (!(r.kernel instanceof Float32Array) || r.kernel.length !== FEATURE_DIM) return null;
  if (typeof r.bias !== 'number' || !Number.isFinite(r.bias)) return null;
  if (typeof r.info !== 'object' || r.info === null) return null;
  if (!r.kernel.every((v) => Number.isFinite(v))) return null;
  return { kernel: r.kernel, bias: r.bias, info: r.info, history: Array.isArray(r.history) ? r.history : [] };
}

export async function loadSavedHead(): Promise<SavedHead | null> {
  try {
    const raw = await withStore<unknown>('readonly', (s) => s.get(KEY));
    return validateSavedHead(raw);
  } catch {
    return null;
  }
}

export async function clearSavedHead(): Promise<void> {
  try {
    await withStore('readwrite', (s) => s.delete(KEY));
  } catch {
    // ignore
  }
}
