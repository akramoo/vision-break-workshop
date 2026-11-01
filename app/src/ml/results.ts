// F9: the results card maths (pure; no React, no tf). Everything here is recomputed from the stored
// probabilities and labels of each evaluation, so changing tau never shows stale numbers.
import { EVAL_SETS, evalSetDef, type EvalSetId } from '../data/evalSets';
import { DISCLAIMER } from '../ui/disclaimer';
import { APP_VERSION } from '../version';
import type { ModelConfig } from './modelConfig';
import {
  alwaysPneumoniaBaseline,
  metricsWithAbstain,
  worstCase,
  type AbstainMetrics,
  type Scores,
} from './metrics';

/** The three sets of the worst-case score, in display order. */
export const RESULT_SET_IDS: readonly EvalSetId[] = EVAL_SETS.map((s) => s.id);

/** Numbers of one set, JSON-safe (null = not defined, shown as "n/a"). */
export interface SetScores {
  n: number;
  accuracy: number | null;
  sensitivity: number | null;
  specificity: number | null;
  balancedAccuracy: number | null;
  referred: number;
  referredFraction: number | null;
  coverage: number | null;
  /** Mean over the two classes of (correct + 0.5 * referred) / n_class. */
  halfCreditScore: number | null;
}

export function setScoresOf(m: AbstainMetrics): SetScores {
  return {
    n: m.n,
    accuracy: m.accuracy,
    sensitivity: m.sensitivity,
    specificity: m.specificity,
    balancedAccuracy: m.balancedAccuracy,
    referred: m.referred,
    referredFraction: m.referredFraction,
    coverage: m.coverage,
    halfCreditScore: m.halfCreditScore,
  };
}

/** The minimum of the three half-credit scores and the set that set it. Needs all three sets. */
export interface Leaderboard {
  score: number;
  set: EvalSetId;
}

/** What the card needs from one evaluation. */
export interface EvalInput {
  labels: ArrayLike<number>;
  probs: ArrayLike<number>;
  headId: number;
  usedPrecomputed: boolean;
  tau: number;
  timestamp: number;
}

export interface SetSummary {
  setId: EvalSetId;
  label: string;
  scores: SetScores;
  /** "Always say pneumonia" on the same labels. At tau it never refers, so half credit = balanced accuracy. */
  baseline: Pick<Scores, 'accuracy' | 'sensitivity' | 'specificity' | 'balancedAccuracy'> & { halfCreditScore: number | null };
  usedPrecomputed: boolean;
  /** tau the evaluation was run at (the numbers above use the CURRENT tau). */
  evaluatedAtTau: number;
  timestamp: number;
}

export interface ResultsSummary {
  tau: number;
  perSet: Partial<Record<EvalSetId, SetSummary>>;
  evaluatedSets: EvalSetId[];
  missingSets: EvalSetId[];
  complete: boolean;
  /** Worst case over the three sets; null until all three are evaluated. */
  leaderboard: Leaderboard | null;
}

/** Leaderboard value from per-set half-credit scores: null unless all three sets have a score. */
export function leaderboardOf(scores: Partial<Record<EvalSetId, number | null | undefined>>): Leaderboard | null {
  const all = RESULT_SET_IDS.every((id) => typeof scores[id] === 'number' && !Number.isNaN(scores[id]));
  if (!all) return null;
  const w = worstCase(scores as Record<string, number | null | undefined>);
  return w ? { score: w.score, set: w.set as EvalSetId } : null;
}

/** Recompute every set at `tau` from stored probabilities. Evaluations of another head are ignored. */
export function summarizeResults(
  evals: Partial<Record<EvalSetId, EvalInput | undefined>>,
  headId: number | null,
  tau: number,
): ResultsSummary {
  const perSet: Partial<Record<EvalSetId, SetSummary>> = {};
  for (const id of RESULT_SET_IDS) {
    const r = evals[id];
    if (!r || headId === null || r.headId !== headId) continue;
    const m = metricsWithAbstain(r.labels, r.probs, tau);
    const b = alwaysPneumoniaBaseline(r.labels);
    perSet[id] = {
      setId: id,
      label: evalSetDef(id).label,
      scores: setScoresOf(m),
      baseline: {
        accuracy: b.accuracy,
        sensitivity: b.sensitivity,
        specificity: b.specificity,
        balancedAccuracy: b.balancedAccuracy,
        halfCreditScore: b.balancedAccuracy,
      },
      usedPrecomputed: r.usedPrecomputed,
      evaluatedAtTau: r.tau,
      timestamp: r.timestamp,
    };
  }
  const evaluatedSets = RESULT_SET_IDS.filter((id) => perSet[id] !== undefined);
  const missingSets = RESULT_SET_IDS.filter((id) => perSet[id] === undefined);
  const leaderboard = leaderboardOf({
    kermany_test: perSet.kermany_test?.scores.halfCreditScore,
    adult_test: perSet.adult_test?.scores.halfCreditScore,
    stress: perSet.stress?.scores.halfCreditScore,
  });
  return { tau, perSet, evaluatedSets, missingSets, complete: missingSets.length === 0, leaderboard };
}

// ---- deltas between snapshots ----

/** A fraction as the percentage the screen shows it (one decimal): 0.8125 -> 81.3, same rounding as formatPercent. */
export function shownPercent(fraction: number): number {
  return Number((fraction * 100).toFixed(1));
}

/**
 * Difference in percentage points (after - before), computed from the values as they are displayed (rounded to 0.1 points),
 * so "81.3% to 87.5%" shows a change of 6.2 points, not 6.3. Null if either is missing.
 */
export function deltaPoints(before: number | null | undefined, after: number | null | undefined): number | null {
  if (typeof before !== 'number' || typeof after !== 'number') return null;
  if (Number.isNaN(before) || Number.isNaN(after)) return null;
  return Number((shownPercent(after) - shownPercent(before)).toFixed(1));
}

/** "+12.3 pts", "-4.0 pts", "0.0 pts", "n/a". Tiny differences round to 0.0 without a sign. */
export function formatDelta(points: number | null, digits = 1): string {
  if (points === null || Number.isNaN(points)) return 'n/a';
  const rounded = Number(points.toFixed(digits));
  if (rounded === 0) return `${(0).toFixed(digits)} pts`;
  return `${rounded > 0 ? '+' : '-'}${Math.abs(rounded).toFixed(digits)} pts`;
}

// ---- results JSON ----

export interface ResultsExportSnapshot {
  id: string;
  name: string;
  createdAt: number;
  config: unknown;
  perSet: unknown;
  leaderboard: Leaderboard | null;
}

export interface ResultsExport {
  disclaimer: string;
  exportedAt: string;
  appVersion: string;
  /** The model that produced these numbers. */
  config: ModelConfig & { tau: number; stampTrained: boolean };
  tau: number;
  leaderboard: (Leaderboard & { setLabel: string; coverage: number | null; tau: number }) | null;
  explanation: string;
  perSet: Partial<Record<EvalSetId, SetSummary>>;
  snapshots: ResultsExportSnapshot[];
  failureCounts: Record<string, number>;
}

export const HALF_CREDIT_EXPLANATION =
  'Half credit: each referred case counts as half right. Referring everything scores exactly 50, the same as always saying pneumonia (50 on every set), so abstaining cannot game the leaderboard.';

export function buildResultsExport(args: {
  summary: ResultsSummary;
  config: ModelConfig;
  snapshots: readonly ResultsExportSnapshot[];
  failureCounts: Record<string, number>;
  now?: Date;
  version?: string;
}): ResultsExport {
  const { summary } = args;
  const lb = summary.leaderboard;
  return {
    disclaimer: DISCLAIMER,
    exportedAt: (args.now ?? new Date()).toISOString(),
    appVersion: args.version ?? APP_VERSION,
    config: { ...args.config, tau: summary.tau, stampTrained: false },
    tau: summary.tau,
    leaderboard: lb
      ? {
          ...lb,
          setLabel: evalSetDef(lb.set).label,
          coverage: summary.perSet[lb.set]?.scores.coverage ?? null,
          tau: summary.tau,
        }
      : null,
    explanation: HALF_CREDIT_EXPLANATION,
    perSet: summary.perSet,
    snapshots: [...args.snapshots],
    failureCounts: args.failureCounts,
  };
}

/** "vision-results-20261031-1542.json" (local time). */
export function resultsExportFileName(now: Date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `vision-results-${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}-${p(now.getHours())}${p(now.getMinutes())}.json`;
}
