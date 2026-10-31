import { describe, expect, it } from 'vitest';
import {
  accuracy,
  accuracyOfProbs,
  alwaysPneumoniaBaseline,
  balancedAccuracy,
  classWeights,
  clampTau,
  confusionMatrix,
  decide,
  flipRate,
  formatPercent,
  hardPredictions,
  metricsWithAbstain,
  rangeOf,
  scoresOf,
  sensitivity,
  specificity,
  worstCase,
} from '../../src/ml/metrics';

describe('confusion matrix and scores', () => {
  it('counts the four boxes', () => {
    const labels = [1, 1, 1, 0, 0, 0, 0];
    const preds = [1, 1, 0, 0, 0, 1, 0];
    const c = confusionMatrix(labels, preds);
    expect(c).toEqual({ tp: 2, fn: 1, fp: 1, tn: 3 });
    expect(c.tp + c.fn + c.fp + c.tn).toBe(labels.length);
    expect(accuracy(c)).toBeCloseTo(5 / 7, 10);
    expect(sensitivity(c)).toBeCloseTo(2 / 3, 10);
    expect(specificity(c)).toBeCloseTo(3 / 4, 10);
    expect(balancedAccuracy(c)).toBeCloseTo((2 / 3 + 3 / 4) / 2, 10);
  });

  it('rejects mismatched lengths', () => {
    expect(() => confusionMatrix([1, 0], [1])).toThrow();
  });

  it('returns null (never NaN) for empty denominators', () => {
    const empty = confusionMatrix([], []);
    expect(scoresOf(empty)).toEqual({ accuracy: null, sensitivity: null, specificity: null, balancedAccuracy: null });
    // only healthy cases: sensitivity and balanced accuracy are undefined
    const onlyHealthy = confusionMatrix([0, 0, 0], [0, 0, 1]);
    expect(sensitivity(onlyHealthy)).toBeNull();
    expect(specificity(onlyHealthy)).toBeCloseTo(2 / 3, 10);
    expect(balancedAccuracy(onlyHealthy)).toBeNull();
    // only sick cases
    const onlySick = confusionMatrix([1, 1], [1, 0]);
    expect(specificity(onlySick)).toBeNull();
    expect(sensitivity(onlySick)).toBe(0.5);
  });

  it('formats null / NaN as n/a', () => {
    expect(formatPercent(null)).toBe('n/a');
    expect(formatPercent(undefined)).toBe('n/a');
    expect(formatPercent(Number.NaN)).toBe('n/a');
    expect(formatPercent(0.625)).toBe('62.5%');
    expect(formatPercent(1)).toBe('100.0%');
    expect(formatPercent(0.12345, 2)).toBe('12.35%');
  });
});

describe('always-pneumonia baseline', () => {
  it('is exactly 62.5% / 100% / 0% on the Kermany test counts (390 pneumonia, 234 normal)', () => {
    const labels = [...new Array<number>(390).fill(1), ...new Array<number>(234).fill(0)];
    const b = alwaysPneumoniaBaseline(labels);
    expect(b.accuracy).toBe(0.625);
    expect(b.sensitivity).toBe(1);
    expect(b.specificity).toBe(0);
    expect(b.balancedAccuracy).toBe(0.5);
    expect(b.confusion).toEqual({ tp: 390, fn: 0, fp: 234, tn: 0 });
  });

  it('is 50% on the balanced adult set and n/a-safe on an empty set', () => {
    const adult = [...new Array<number>(100).fill(1), ...new Array<number>(100).fill(0)];
    expect(alwaysPneumoniaBaseline(adult).accuracy).toBe(0.5);
    expect(alwaysPneumoniaBaseline([]).accuracy).toBeNull();
  });
});

describe('classWeights', () => {
  it('gives normal 2.00 / pneumonia 0.67 for the default 255 / 765', () => {
    const w = classWeights(255, 765);
    expect(w[0]).toBeCloseTo(2, 10);
    expect(w[1]).toBeCloseTo(2 / 3, 10);
    expect(w[0].toFixed(2)).toBe('2.00');
    expect(w[1].toFixed(2)).toBe('0.67');
  });

  it('recomputes with the adult mix (355 normal, 865 pneumonia)', () => {
    const w = classWeights(255 + 100, 765 + 100);
    expect(w[0]).toBeCloseTo((355 + 865) / (2 * 355), 10);
    expect(w[1]).toBeCloseTo((355 + 865) / (2 * 865), 10);
  });

  it('weights sum back to N: sum_c N_c * w_c = N', () => {
    const w = classWeights(100, 300);
    expect(100 * w[0] + 300 * w[1]).toBeCloseTo(400, 10);
  });

  it('is neutral when a class is empty', () => {
    expect(classWeights(0, 10)).toEqual({ 0: 1, 1: 1 });
    expect(classWeights(10, 0)).toEqual({ 0: 1, 1: 1 });
    expect(classWeights(0, 0)).toEqual({ 0: 1, 1: 1 });
  });
});

describe('decide', () => {
  it('uses p >= 0.5 for pneumonia and never refers at tau 0.5', () => {
    expect(decide(0.5, 0.5)).toBe('pneumonia');
    expect(decide(0.4999, 0.5)).toBe('normal');
    for (const p of [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]) expect(decide(p, 0.5)).not.toBe('refer');
  });

  it('refers when the confidence max(p, 1-p) is below tau', () => {
    expect(decide(0.7, 0.8)).toBe('refer');
    expect(decide(0.3, 0.8)).toBe('refer');
    expect(decide(0.9, 0.8)).toBe('pneumonia');
    expect(decide(0.1, 0.8)).toBe('normal');
    expect(decide(0.8, 0.8)).toBe('pneumonia'); // exactly tau is not below tau
  });

  it('refers a NaN probability', () => {
    expect(decide(Number.NaN, 0.5)).toBe('refer');
  });

  it('clampTau keeps tau in [0.5, 0.99]', () => {
    expect(clampTau(0.2)).toBe(0.5);
    expect(clampTau(1.5)).toBe(0.99);
    expect(clampTau(0.8)).toBe(0.8);
    expect(clampTau(Number.NaN)).toBe(0.5);
  });
});

describe('metricsWithAbstain', () => {
  const labels = [1, 1, 1, 1, 0, 0, 0, 0];
  const probs = [0.95, 0.6, 0.3, 0.9, 0.05, 0.4, 0.7, 0.2];

  it('at tau 0.5 refers nothing and half-credit equals balanced accuracy', () => {
    const m = metricsWithAbstain(labels, probs, 0.5);
    expect(m.referred).toBe(0);
    expect(m.coverage).toBe(1);
    expect(m.referredFraction).toBe(0);
    expect(m.n).toBe(8);
    expect(m.confusion).toEqual({ tp: 3, fn: 1, fp: 1, tn: 3 });
    expect(m.halfCreditScore).toBeCloseTo(m.balancedAccuracy as number, 12);
    expect(m.halfCreditScore).toBeCloseTo((3 / 4 + 3 / 4) / 2, 12);
  });

  it('refers the unsure cases and gives them half credit', () => {
    // tau 0.8: confidence = max(p, 1-p) -> 0.95, 0.6, 0.7, 0.9 | 0.95, 0.6, 0.7, 0.8
    // refer: p=0.6 (sick), p=0.3 (sick), p=0.4 (healthy), p=0.7 (healthy)
    const m = metricsWithAbstain(labels, probs, 0.8);
    expect(m.referred).toBe(4);
    expect(m.referredByClass).toEqual({ normal: 2, pneumonia: 2 });
    expect(m.coverage).toBe(0.5);
    expect(m.confusion).toEqual({ tp: 2, fn: 0, fp: 0, tn: 2 });
    expect(m.accuracy).toBe(1);
    // class credit = (correct + 0.5 * referred) / n_class = (2 + 1) / 4 for both classes
    expect(m.classCredit.pneumonia).toBeCloseTo(0.75, 12);
    expect(m.classCredit.normal).toBeCloseTo(0.75, 12);
    expect(m.halfCreditScore).toBeCloseTo(0.75, 12);
  });

  it('referrals never decrease as tau rises', () => {
    let last = -1;
    for (let tau = 0.5; tau <= 0.99; tau += 0.01) {
      const r = metricsWithAbstain(labels, probs, tau).referred;
      expect(r).toBeGreaterThanOrEqual(last);
      last = r;
    }
  });

  it('refers everything at tau 0.99 when nothing is that confident; half credit is then exactly 0.5', () => {
    const m = metricsWithAbstain(labels, [0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4], 0.99);
    expect(m.referred).toBe(8);
    expect(m.coverage).toBe(0);
    expect(m.accuracy).toBeNull();
    expect(m.sensitivity).toBeNull();
    expect(m.specificity).toBeNull();
    expect(m.halfCreditScore).toBe(0.5);
  });

  it('is null-safe for empty and single-class sets', () => {
    const empty = metricsWithAbstain([], [], 0.7);
    expect(empty.n).toBe(0);
    expect(empty.coverage).toBeNull();
    expect(empty.referredFraction).toBeNull();
    expect(empty.halfCreditScore).toBeNull();
    const oneClass = metricsWithAbstain([1, 1], [0.9, 0.2], 0.5);
    expect(oneClass.halfCreditScore).toBeNull();
    expect(oneClass.classCredit.normal).toBeNull();
    expect(oneClass.sensitivity).toBe(0.5);
    for (const v of Object.values(empty)) {
      if (typeof v === 'number') expect(Number.isNaN(v)).toBe(false);
    }
  });

  it('rejects mismatched lengths', () => {
    expect(() => metricsWithAbstain([1], [0.5, 0.5], 0.5)).toThrow();
  });

  it('hardPredictions / accuracyOfProbs follow the 0.5 rule', () => {
    expect(Array.from(hardPredictions([0.2, 0.5, 0.9]))).toEqual([0, 1, 1]);
    expect(accuracyOfProbs([0, 1, 1], [0.2, 0.5, 0.1])).toBeCloseTo(2 / 3, 12);
    expect(accuracyOfProbs([], [])).toBeNull();
  });
});

describe('worstCase', () => {
  it('returns the minimum score and its set', () => {
    expect(worstCase({ kermany_test: 0.9, adult_test: 0.61, stress: 0.7 })).toEqual({ score: 0.61, set: 'adult_test' });
  });
  it('ignores sets without a score, and is null when none has one', () => {
    expect(worstCase({ a: null, b: 0.8, c: undefined })).toEqual({ score: 0.8, set: 'b' });
    expect(worstCase({ a: null })).toBeNull();
    expect(worstCase({})).toBeNull();
    expect(worstCase({ a: Number.NaN, b: 0.5 })).toEqual({ score: 0.5, set: 'b' });
  });
});

describe('flipRate', () => {
  it('counts changed predictions per TRUE class', () => {
    const labels = [0, 0, 0, 0, 1, 1];
    const before = [0, 0, 0, 1, 1, 1];
    const after = [1, 1, 0, 1, 1, 0];
    const f = flipRate(labels, before, after);
    expect(f.normal).toEqual({ n: 4, flipped: 2, rate: 0.5 });
    expect(f.pneumonia).toEqual({ n: 2, flipped: 1, rate: 0.5 });
    expect(f.overall.flipped).toBe(3);
    expect(f.overall.rate).toBeCloseTo(0.5, 12);
  });
  it('has null rate for an absent class and rejects bad lengths', () => {
    const f = flipRate([1, 1], [1, 0], [0, 0]);
    expect(f.normal.rate).toBeNull();
    expect(f.pneumonia.rate).toBe(0.5);
    expect(() => flipRate([1], [1, 0], [0, 0])).toThrow();
  });
});

describe('rangeOf', () => {
  it('gives min, max and spread, skipping nulls', () => {
    expect(rangeOf([0.5, null, 0.75, 0.625])).toEqual({ min: 0.5, max: 0.75, spread: 0.25 });
    expect(rangeOf([])).toBeNull();
    expect(rangeOf([null])).toBeNull();
  });
});
