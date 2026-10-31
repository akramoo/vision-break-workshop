// Typed access to public/data/manifest.json (the list of every shipped X-ray). Framework-free.

export type Label = 'normal' | 'pneumonia';
export type Source = 'kermany' | 'nih';
export type Split = 'train' | 'val' | 'val_official' | 'test' | 'train_adult' | 'test_adult';

export interface ManifestEntry {
  id: string;
  /** Path relative to public/data/. */
  path: string;
  label: Label;
  split: Split;
  source: Source;
}

export class ManifestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ManifestError';
  }
}

const LABELS: readonly string[] = ['normal', 'pneumonia'];
const SOURCES: readonly string[] = ['kermany', 'nih'];
const SPLITS: readonly string[] = ['train', 'val', 'val_official', 'test', 'train_adult', 'test_adult'];

/** Validate the parsed JSON; throws ManifestError with a readable message. */
export function parseManifest(raw: unknown): ManifestEntry[] {
  if (!Array.isArray(raw) || raw.length === 0) throw new ManifestError('manifest.json is empty or not a list.');
  const seen = new Set<string>();
  const out: ManifestEntry[] = [];
  raw.forEach((e: unknown, i) => {
    const o = (typeof e === 'object' && e !== null ? e : {}) as Record<string, unknown>;
    const { id, path, label, split, source } = o;
    if (
      typeof id !== 'string' ||
      typeof path !== 'string' ||
      typeof label !== 'string' ||
      !LABELS.includes(label) ||
      typeof split !== 'string' ||
      !SPLITS.includes(split) ||
      typeof source !== 'string' ||
      !SOURCES.includes(source)
    ) {
      throw new ManifestError(`manifest.json entry ${i} is malformed.`);
    }
    if (seen.has(id)) throw new ManifestError(`manifest.json has a duplicate id: ${id}.`);
    seen.add(id);
    out.push({ id, path, label: label as Label, split: split as Split, source: source as Source });
  });
  return out;
}

/** Base URL for everything under public/data/ (relative, so it works with `base: './'`). */
export function dataBaseUrl(base: string = import.meta.env.BASE_URL): string {
  return `${base}data/`;
}

let cached: Promise<ManifestEntry[]> | null = null;

/** Fetch and validate the manifest once; a failed load is not cached. */
export function loadManifest(): Promise<ManifestEntry[]> {
  if (cached) return cached;
  const p = (async () => {
    let res: Response;
    try {
      res = await fetch(`${dataBaseUrl()}manifest.json`);
    } catch (e) {
      throw new ManifestError(`Could not download manifest.json (${e instanceof Error ? e.message : String(e)}).`);
    }
    if (!res.ok) throw new ManifestError(`manifest.json returned HTTP ${res.status}.`);
    let json: unknown;
    try {
      json = JSON.parse(await res.text());
    } catch {
      throw new ManifestError('manifest.json is not valid JSON.');
    }
    return parseManifest(json);
  })();
  cached = p;
  p.catch(() => {
    if (cached === p) cached = null;
  });
  return p;
}

export interface EntryFilter {
  source?: Source;
  split?: Split;
  label?: Label;
}

export function selectEntries(entries: readonly ManifestEntry[], f: EntryFilter): ManifestEntry[] {
  return entries.filter(
    (e) =>
      (f.source === undefined || e.source === f.source) &&
      (f.split === undefined || e.split === f.split) &&
      (f.label === undefined || e.label === f.label),
  );
}

/** Ids (manifest order) of one source/split, optionally one label. */
export function idsBySplit(entries: readonly ManifestEntry[], source: Source, split: Split, label?: Label): string[] {
  return selectEntries(entries, label === undefined ? { source, split } : { source, split, label }).map((e) => e.id);
}

/** 1 = pneumonia, 0 = normal / no finding. */
export function labelValue(e: Pick<ManifestEntry, 'label'>): 0 | 1 {
  return e.label === 'pneumonia' ? 1 : 0;
}

/** Words shown to students: adult negatives (NIH "normal") are "no finding". */
export function displayLabel(e: Pick<ManifestEntry, 'label' | 'source'>): 'pneumonia' | 'normal' | 'no finding' {
  if (e.label === 'pneumonia') return 'pneumonia';
  return e.source === 'nih' ? 'no finding' : 'normal';
}

export interface Count {
  normal: number;
  pneumonia: number;
  total: number;
}

export function countOf(entries: readonly ManifestEntry[]): Count {
  let normal = 0;
  let pneumonia = 0;
  for (const e of entries) {
    if (e.label === 'pneumonia') pneumonia++;
    else normal++;
  }
  return { normal, pneumonia, total: normal + pneumonia };
}

/** Counts keyed by "source/split". */
export function countsBySplit(entries: readonly ManifestEntry[]): Record<string, Count> {
  const groups = new Map<string, ManifestEntry[]>();
  for (const e of entries) {
    const k = `${e.source}/${e.split}`;
    const g = groups.get(k);
    if (g) g.push(e);
    else groups.set(k, [e]);
  }
  const out: Record<string, Count> = {};
  for (const [k, g] of groups) out[k] = countOf(g);
  return out;
}

/** Look up helper: id -> entry. */
export function indexById(entries: readonly ManifestEntry[]): Map<string, ManifestEntry> {
  return new Map(entries.map((e) => [e.id, e]));
}
