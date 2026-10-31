// Versioned, defensive localStorage helpers. Bump the version in KEY_PREFIX when
// a stored shape changes incompatibly; old keys are simply ignored.
export const KEY_PREFIX = 'vbw:v1:';
const SETTINGS_KEY = `${KEY_PREFIX}settings`;

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

export function saveSettings(settings: Settings): void {
  try {
    storage()?.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // quota exceeded or storage disabled: settings just won't persist
  }
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
