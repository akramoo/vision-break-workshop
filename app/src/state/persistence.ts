// Versioned, defensive localStorage helpers. Bump the version in KEY_PREFIX when
// a stored shape changes incompatibly; old keys are simply ignored.
import { DEFAULT_TRAIN_SETTINGS, type TrainSettings } from '../data/trainingPlan';
import { clampTau, MIN_TAU } from '../ml/metrics';

export const KEY_PREFIX = 'vbw:v1:';
const SETTINGS_KEY = `${KEY_PREFIX}settings`;
const TRAIN_KEY = `${KEY_PREFIX}train`;
const TAU_KEY = `${KEY_PREFIX}tau`;
export const LOG_KEY = `${KEY_PREFIX}log`;
export const SNAPSHOTS_KEY = `${KEY_PREFIX}snapshots`;
export const STEPS_KEY = `${KEY_PREFIX}steps`;

export interface Settings {
  /** Use shipped MobileNet features instead of extracting them live. */
  usePrecomputed: boolean;
}

export const DEFAULT_SETTINGS: Settings = { usePrecomputed: true };

function storage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null; // blocked by privacy settings
  }
}

/** Raw stored text for a key, or null (missing, or storage blocked). */
export function readStored(key: string): string | null {
  try {
    return storage()?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

/**
 * Write (or, with null, remove) a key. Returns false when the browser refused the write (quota exceeded or
 * storage blocked), so callers can warn and keep the data in memory.
 */
export function writeStored(key: string, text: string | null): boolean {
  try {
    const s = storage();
    if (!s) return false;
    if (text === null) s.removeItem(key);
    else s.setItem(key, text);
    return true;
  } catch {
    return false;
  }
}

/** JSON.parse that returns `fallback` instead of throwing. */
export function safeParse<T>(text: string | null, fallback: T): T {
  if (text === null) return fallback;
  try {
    return JSON.parse(text) as T;
  } catch {
    return fallback;
  }
}

export function loadSettings(): Settings {
  let text: string | null = null;
  try {
    text = storage()?.getItem(SETTINGS_KEY) ?? null;
  } catch {
    text = null;
  }
  const raw = safeParse<unknown>(text, null);
  if (typeof raw === 'object' && raw !== null) {
    const { usePrecomputed } = raw as { usePrecomputed?: unknown };
    return {
      usePrecomputed: typeof usePrecomputed === 'boolean' ? usePrecomputed : DEFAULT_SETTINGS.usePrecomputed,
    };
  }
  return { ...DEFAULT_SETTINGS };
}

/** Defaults are not stored (so a fresh or just-reset session leaves no keys behind). */
export function saveSettings(settings: Settings): void {
  // quota exceeded or storage disabled: settings just won't persist
  writeStored(SETTINGS_KEY, settings.usePrecomputed === DEFAULT_SETTINGS.usePrecomputed ? null : JSON.stringify(settings));
}

function clampInt(v: unknown, lo: number, hi: number, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : fallback;
}

/** Turn any stored value into valid training settings (unknown fields dropped, bad values defaulted/clamped). */
export function sanitizeTrainSettings(raw: unknown): TrainSettings {
  const d = DEFAULT_TRAIN_SETTINGS;
  const o = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>;
  const lr = typeof o['lr'] === 'number' && Number.isFinite(o['lr']) ? Math.min(0.1, Math.max(1e-4, o['lr'])) : d.lr;
  return {
    mode: o['mode'] === 'balanced' || o['mode'] === 'realistic' ? o['mode'] : d.mode,
    adultMix: typeof o['adultMix'] === 'boolean' ? o['adultMix'] : d.adultMix,
    augmentation: typeof o['augmentation'] === 'boolean' ? o['augmentation'] : d.augmentation,
    epochs: clampInt(o['epochs'], 1, 100, d.epochs),
    lr,
    seed: clampInt(o['seed'], 0, 2_147_483_647, d.seed),
    valSet: o['valSet'] === 'val' || o['valSet'] === 'val_official' ? o['valSet'] : d.valSet,
  };
}

export function loadTrainSettings(): TrainSettings {
  let text: string | null = null;
  try {
    text = storage()?.getItem(TRAIN_KEY) ?? null;
  } catch {
    text = null;
  }
  return sanitizeTrainSettings(safeParse<unknown>(text, null));
}

export function saveTrainSettings(settings: TrainSettings): void {
  const d = DEFAULT_TRAIN_SETTINGS;
  const isDefault = (Object.keys(d) as (keyof TrainSettings)[]).every((k) => settings[k] === d[k]);
  writeStored(TRAIN_KEY, isDefault ? null : JSON.stringify(settings));
}

/** Saved abstain threshold (0.50..0.99); 0.50 = never refer. */
export function loadTau(): number {
  let text: string | null = null;
  try {
    text = storage()?.getItem(TAU_KEY) ?? null;
  } catch {
    text = null;
  }
  const v = safeParse<unknown>(text, null);
  return clampTau(typeof v === 'number' ? v : MIN_TAU);
}

export function saveTau(tau: number): void {
  writeStored(TAU_KEY, tau === MIN_TAU ? null : JSON.stringify(tau));
}

/** Remove every key this app wrote (used by "Reset session"). */
export function clearPersisted(): void {
  try {
    const s = storage();
    if (!s) return;
    const keys: string[] = [];
    for (let i = 0; i < s.length; i++) {
      const k = s.key(i);
      if (k !== null && k.startsWith('vbw:')) keys.push(k);
    }
    keys.forEach((k) => s.removeItem(k));
  } catch {
    // ignore
  }
}
