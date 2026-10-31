// Pure evaluation maths (no tf, no DOM). Labels and predictions are numbers:
// 1 = pneumonia (the "positive" class), 0 = normal / no finding.
// Every ratio with an empty denominator is null (the UI shows "n/a"), never NaN.

export type Decision = 'pneumonia' | 'normal' | 'refer';

export interface Confusion {
  /** Sick, called pneumonia (caught). */
  tp: number;
  /** Sick, called normal (missed). */
  fn: number;
  /** Healthy, called pneumonia (false alarm). */
  fp: number;
  /** Healthy, called normal (cleared). */
  tn: number;
}

export interface Scores {
  accuracy: number | null;
  /** Share of sick children caught: tp / (tp + fn). */
  sensitivity: number | null;
  /** Share of healthy children correctly cleared: tn / (tn + fp). */
  specificity: number | null;
  /** Mean of sensitivity and specificity. */
  balancedAccuracy: number | null;
}

export const MIN_TAU = 0.5;
export const MAX_TAU = 0.99;

export function ratio(num: number, den: number): number | null {
  return den > 0 ? num / den : null;
}

export function confusionTotal(c: Confusion): number {
  return c.tp + c.fn + c.fp + c.tn;
}

/** Count outcomes. labels/preds are 0/1 arrays of equal length. */
export function confusionMatrix(labels: ArrayLike<number>, preds: ArrayLike<number>): Confusion {
  if (labels.length !== preds.length) {
    throw new Error(`confusionMatrix: ${labels.length} labels but ${preds.length} predictions`);
  }
  const c: Confusion = { tp: 0, fn: 0, fp: 0, tn: 0 };
  for (let i = 0; i < labels.length; i++) {
    const y = labels[i] === 1;
    const p = preds[i] === 1;
    if (y && p) c.tp++;
    else if (y) c.fn++;
    else if (p) c.fp++;
    else c.tn++;
  }
  return c;
}

export function accuracy(c: Confusion): number | null {
  return ratio(c.tp + c.tn, confusionTotal(c));
}
export function sensitivity(c: Confusion): number | null {
  return ratio(c.tp, c.tp + c.fn);
}
export function specificity(c: Confusion): number | null {
  return ratio(c.tn, c.tn + c.fp);
}
export function balancedAccuracy(c: Confusion): number | null {
  const s = sensitivity(c);
  const p = specificity(c);
  return s === null || p === null ? null : (s + p) / 2;
}

export function scoresOf(c: Confusion): Scores {
  return {
    accuracy: accuracy(c),
    sensitivity: sensitivity(c),
    specificity: specificity(c),
    balancedAccuracy: balancedAccuracy(c),
  };
}

/** The "always say pneumonia" baseline for a set of true labels. */
export function alwaysPneumoniaBaseline(labels: ArrayLike<number>): Scores & { confusion: Confusion } {
  const preds = new Array<number>(labels.length).fill(1);
  const confusion = confusionMatrix(labels, preds);
  return { confusion, ...scoresOf(confusion) };
}

/** Class weights w_c = N / (2 * N_c). Both 1 when a class is empty (nothing to balance). */
export function classWeights(nNormal: number, nPneumonia: number): { 0: number; 1: number } {
  if (nNormal <= 0 || nPneumonia <= 0) return { 0: 1, 1: 1 };
  const n = nNormal + nPneumonia;
  return { 0: n / (2 * nNormal), 1: n / (2 * nPneumonia) };
}

export function clampTau(tau: number): number {
  if (!Number.isFinite(tau)) return MIN_TAU;
  return Math.min(MAX_TAU, Math.max(MIN_TAU, tau));
}

/** Confidence of a probability: max(p, 1 - p). */
export function confidenceOf(p: number): number {
  return Math.max(p, 1 - p);
}

/**
 * Decision rule: pneumonia if p >= 0.5, else normal; "refer" when the confidence
 * max(p, 1 - p) is below tau. tau = 0.5 never refers. A NaN probability is referred.
 */
export function decide(p: number, tau: number): Decision {
  if (Number.isNaN(p)) return 'refer';
  if (confidenceOf(p) < tau) return 'refer';
  return p >= 0.5 ? 'pneumonia' : 'normal';
}

export interface AbstainMetrics extends Scores {
  n: number;
  referred: number;
  /** referred / n, null when n = 0. */
  referredFraction: number | null;
  /** 1 - referredFraction, null when n = 0. */
  coverage: number | null;
  /** Confusion over the cases that were NOT referred. */
  confusion: Confusion;
  /** Per-class half-credit: (correct + 0.5 * referred) / n_class. */
  classCredit: { normal: number | null; pneumonia: number | null };
  referredByClass: { normal: number; pneumonia: number };
  /** Mean of the two class credits; equals balanced accuracy at tau = 0.5. Null if a class is empty. */
  halfCreditScore: number | null;
}

/** Metrics when low-confidence cases are referred to a doctor. Scores use only non-referred cases. */
export function metricsWithAbstain(labels: ArrayLike<number>, probs: ArrayLike<number>, tau: number): AbstainMetrics {
  if (labels.length !== probs.length) {
    throw new Error(`metricsWithAbstain: ${labels.length} labels but ${probs.length} probabilities`);
  }
  const confusion: Confusion = { tp: 0, fn: 0, fp: 0, tn: 0 };
  const nClass = { normal: 0, pneumonia: 0 };
  const correct = { normal: 0, pneumonia: 0 };
  const referredByClass = { normal: 0, pneumonia: 0 };
  let referred = 0;
  for (let i = 0; i < labels.length; i++) {
    const sick = labels[i] === 1;
    const cls = sick ? 'pneumonia' : 'normal';
    nClass[cls]++;
    const d = decide(probs[i] as number, tau);
    if (d === 'refer') {
      referred++;
      referredByClass[cls]++;
      continue;
    }
    const callSick = d === 'pneumonia';
    if (sick && callSick) confusion.tp++;
    else if (sick) confusion.fn++;
    else if (callSick) confusion.fp++;
    else confusion.tn++;
    if (sick === callSick) correct[cls]++;
  }
  const n = labels.length;
  const credit = (cls: 'normal' | 'pneumonia') => ratio(correct[cls] + 0.5 * referredByClass[cls], nClass[cls]);
  const classCredit = { normal: credit('normal'), pneumonia: credit('pneumonia') };
  const halfCreditScore =
    classCredit.normal === null || classCredit.pneumonia === null
      ? null
      : (classCredit.normal + classCredit.pneumonia) / 2;
  return {
    n,
    referred,
    referredFraction: ratio(referred, n),
    coverage: n > 0 ? (n - referred) / n : null,
    confusion,
    ...scoresOf(confusion),
    classCredit,
    referredByClass,
    halfCreditScore,
  };
}

/** The lowest score over all sets (ignoring sets without a score), and which set it came from. */
export function worstCase(scoresBySet: Record<string, number | null | undefined>): { score: number; set: string } | null {
  let worst: { score: number; set: string } | null = null;
  for (const [set, score] of Object.entries(scoresBySet)) {
    if (typeof score !== 'number' || Number.isNaN(score)) continue;
    if (worst === null || score < worst.score) worst = { score, set };
  }
  return worst;
}

export interface FlipRate {
  n: number;
  flipped: number;
  rate: number | null;
}

/** How many predictions changed between two runs over the same cases, per TRUE class. */
export function flipRate(
  labels: ArrayLike<number>,
  before: ArrayLike<number>,
  after: ArrayLike<number>,
): { normal: FlipRate; pneumonia: FlipRate; overall: FlipRate } {
  if (labels.length !== before.length || labels.length !== after.length) {
    throw new Error('flipRate: labels, before and after must have the same length');
  }
  const acc = { normal: { n: 0, flipped: 0 }, pneumonia: { n: 0, flipped: 0 } };
  for (let i = 0; i < labels.length; i++) {
    const a = acc[labels[i] === 1 ? 'pneumonia' : 'normal'];
    a.n++;
    if (before[i] !== after[i]) a.flipped++;
  }
  const fin = (x: { n: number; flipped: number }): FlipRate => ({ ...x, rate: ratio(x.flipped, x.n) });
  const total = { n: acc.normal.n + acc.pneumonia.n, flipped: acc.normal.flipped + acc.pneumonia.flipped };
  return { normal: fin(acc.normal), pneumonia: fin(acc.pneumonia), overall: fin(total) };
}

/** "62.5%", or "n/a" for null / NaN. */
export function formatPercent(v: number | null | undefined, digits = 1): string {
  if (typeof v !== 'number' || Number.isNaN(v)) return 'n/a';
  return `${(v * 100).toFixed(digits)}%`;
}

/** Hard 0/1 predictions at the p >= 0.5 decision rule (no abstention). */
export function hardPredictions(probs: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(probs.length);
  for (let i = 0; i < probs.length; i++) out[i] = (probs[i] as number) >= 0.5 ? 1 : 0;
  return out;
}

/** Accuracy of probabilities against 0/1 labels at the 0.5 rule; null for an empty set. */
export function accuracyOfProbs(labels: ArrayLike<number>, probs: ArrayLike<number>): number | null {
  return accuracy(confusionMatrix(labels, hardPredictions(probs)));
}

/** Smallest / largest of the numeric values and their difference; null if there are none. */
export function rangeOf(values: readonly (number | null | undefined)[]): { min: number; max: number; spread: number } | null {
  const v = values.filter((x): x is number => typeof x === 'number' && !Number.isNaN(x));
  if (v.length === 0) return null;
  const min = Math.min(...v);
  const max = Math.max(...v);
  return { min, max, spread: max - min };
}
