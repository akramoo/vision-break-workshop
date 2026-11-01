import { decide, type Decision } from '../ml/metrics';

/** One evaluated X-ray, with the decision the model makes at the current abstain threshold. */
export interface PredRow {
  id: string;
  /** P(pneumonia). */
  p: number;
  d: Decision;
  /** True label: 1 = pneumonia, 0 = normal / no finding. */
  truth: number;
  /** Answered (not referred) and answered wrongly. */
  wrong: boolean;
  /** Position in the evaluated set (dataset order). */
  i: number;
}

export type ViewMode = 'mistakes-first' | 'only-mistakes' | 'only-referred' | 'only-correct' | 'dataset';
export type ClassFilter = 'all' | 'pneumonia' | 'negative';

export const PAGE_SIZES = [20, 40, 100] as const;
export const DEFAULT_PAGE_SIZE = 40;

export const VIEW_MODES: readonly ViewMode[] = ['mistakes-first', 'only-mistakes', 'only-referred', 'only-correct', 'dataset'];

/** All rows of an evaluated set in dataset order. */
export function buildRows(ids: readonly string[], probs: ArrayLike<number>, labels: ArrayLike<number>, tau: number): PredRow[] {
  return ids.map((id, i) => {
    const p = probs[i] as number;
    const d = decide(p, tau);
    const truth = labels[i] as number;
    const wrong = d !== 'refer' && (d === 'pneumonia') !== (truth === 1);
    return { id, p, d, truth, wrong, i };
  });
}

export interface Categories {
  total: number;
  mistakes: number;
  referred: number;
  correct: number;
}

/** Mistakes + referred + correct always add up to the total. */
export function categoryCounts(rows: readonly PredRow[]): Categories {
  let mistakes = 0;
  let referred = 0;
  for (const r of rows) {
    if (r.wrong) mistakes++;
    else if (r.d === 'refer') referred++;
  }
  return { total: rows.length, mistakes, referred, correct: rows.length - mistakes - referred };
}

export interface ClassCounts {
  total: number;
  pneumonia: number;
  negative: number;
}

export function classCounts(rows: readonly PredRow[]): ClassCounts {
  let pneumonia = 0;
  for (const r of rows) if (r.truth === 1) pneumonia++;
  return { total: rows.length, pneumonia, negative: rows.length - pneumonia };
}

function matchesClass(r: PredRow, cls: ClassFilter): boolean {
  return cls === 'all' || (cls === 'pneumonia') === (r.truth === 1);
}

/**
 * Filter by true class, then filter/order by view mode. Deterministic and stable: rows that tie keep their dataset order,
 * and the input array is never modified.
 */
export function applyView(rows: readonly PredRow[], mode: ViewMode, cls: ClassFilter): PredRow[] {
  const byClass = cls === 'all' ? rows : rows.filter((r) => matchesClass(r, cls));
  switch (mode) {
    case 'only-mistakes':
      return byClass.filter((r) => r.wrong);
    case 'only-referred':
      return byClass.filter((r) => r.d === 'refer');
    case 'only-correct':
      return byClass.filter((r) => !r.wrong && r.d !== 'refer');
    case 'dataset':
      return byClass.slice();
    case 'mistakes-first':
      return [...byClass.filter((r) => r.wrong), ...byClass.filter((r) => !r.wrong)];
  }
}

/** 'Only referred' makes no sense without abstention (tau = 0.5 refers nothing); fall back to the default order. */
export function effectiveMode(mode: ViewMode, tau: number): ViewMode {
  return mode === 'only-referred' && !(tau > 0.5) ? 'mistakes-first' : mode;
}

export interface PageInfo {
  /** 1-based, clamped into 1..pages. */
  page: number;
  pages: number;
  /** 0-based slice bounds [start, end). */
  start: number;
  end: number;
  total: number;
  size: number;
}

export function pageInfo(total: number, page: number, size: number): PageInfo {
  const s = Math.max(1, Math.floor(size));
  const t = Math.max(0, Math.floor(total));
  const pages = Math.max(1, Math.ceil(t / s));
  const cur = Number.isFinite(page) ? Math.min(Math.max(1, Math.floor(page)), pages) : 1;
  const start = Math.min((cur - 1) * s, t);
  return { page: cur, pages, start, end: Math.min(start + s, t), total: t, size: s };
}

export function pageLabel(info: PageInfo): string {
  if (info.total === 0) return 'No rows match this filter';
  return `Page ${info.page} of ${info.pages} (rows ${info.start + 1}-${info.end} of ${info.total})`;
}

export function countsLabel(c: Categories): string {
  return `Mistakes ${c.mistakes} | Referred ${c.referred} | Correct ${c.correct}`;
}
