// Independent (tester-written) checks of src/ml/metrics.ts. Every expected value below was worked out by hand
// from the acceptance criteria (F3, F7, "Decision rule", "Worst-case score"), not copied from the code.
import { describe, expect, it } from 'vitest';
import {
  alwaysPneumoniaBaseline,
  classWeights,
  confidenceOf,
  confusionMatrix,
  decide,
  flipRate,
  formatPercent,
  hardPredictions,
  metricsWithAbstain,
  scoresOf,
  worstCase,
} from '../../src/ml/metrics';
import { mulberry32 } from '../../src/shared/pixelOps.js';

/** n copies of a value. */
const rep = (v: number, n: number): number[] => new Array<number>(n).fill(v);

describe('confusion + scores by hand', () => {
  // 10 sick (label 1), 10 healthy (label 0). Model catches 7 sick, clears 6 healthy.
  const labels = [...rep(1, 10), ...rep(0, 10)];
  const preds = [...rep(1, 7), ...rep(0, 3), ...rep(0, 6), ...rep(1, 4)];

  it('tp=7 fn=3 tn=6 fp=4', () => {
    expect(confusionMatrix(labels, preds)).toEqual({ tp: 7, fn: 3, fp: 4, tn: 6 });
  });

  it('accuracy 13/20, sensitivity 7/10, specificity 6/10, balanced 0.65', () => {
    const s = scoresOf(confusionMatrix(labels, preds));
    expect(s.accuracy).toBeCloseTo(0.65, 12);
    expect(s.sensitivity).toBeCloseTo(0.7, 12);
    expect(s.specificity).toBeCloseTo(0.6, 12);
    expect(s.balancedAccuracy).toBeCloseTo(0.65, 12);
  });

  it('balanced accuracy differs from accuracy on an imbalanced set (3 sick, 9 healthy)', () => {
    // catches 3/3 sick, clears 3/9 healthy: acc = 6/12 = 0.5, balanced = (1 + 1/3)/2
    const l = [...rep(1, 3), ...rep(0, 9)];
    const p = [...rep(1, 3), ...rep(0, 3), ...rep(1, 6)];
    const s = scoresOf(confusionMatrix(l, p));
    expect(s.accuracy).toBeCloseTo(0.5, 12);
    expect(s.balancedAccuracy).toBeCloseTo((1 + 1 / 3) / 2, 12);
  });

  it('an empty class gives null (never NaN) for every score that needs it', () => {
    // no healthy cases at all
    const s = scoresOf(confusionMatrix([1, 1], [1, 0]));
    expect(s.accuracy).toBeCloseTo(0.5, 12);
    expect(s.sensitivity).toBeCloseTo(0.5, 12);
    expect(s.specificity).toBeNull();
    expect(s.balancedAccuracy).toBeNull();
    // no sick cases at all
    const t = scoresOf(confusionMatrix([0, 0, 0], [0, 0, 1]));
    expect(t.sensitivity).toBeNull();
    expect(t.balancedAccuracy).toBeNull();
    expect(t.specificity).toBeCloseTo(2 / 3, 12);
    // totally empty set
    const e = scoresOf(confusionMatrix([], []));
    for (const v of Object.values(e)) expect(v).toBeNull();
    for (const v of [...Object.values(s), ...Object.values(t), ...Object.values(e)]) expect(Number.isNaN(v as number)).toBe(false);
    expect(formatPercent(e.accuracy)).toBe('n/a');
    expect(formatPercent(Number.NaN)).toBe('n/a');
    expect(formatPercent(0.625)).toBe('62.5%');
    expect(formatPercent(1)).toBe('100.0%');
    expect(formatPercent(0)).toBe('0.0%');
  });

  it('property: sensitivity + miss rate = 1 and specificity + false-alarm rate = 1, and the four boxes sum to n', () => {
    const rng = mulberry32(2024);
    for (let trial = 0; trial < 200; trial++) {
      const n = 2 + Math.floor(rng() * 80);
      const y = Array.from({ length: n }, () => (rng() < 0.4 ? 1 : 0));
      const p = Array.from({ length: n }, () => (rng() < 0.5 ? 1 : 0));
      const c = confusionMatrix(y, p);
      expect(c.tp + c.fn + c.fp + c.tn).toBe(n);
      const sick = y.filter((v) => v === 1).length;
      const healthy = n - sick;
      expect(c.tp + c.fn).toBe(sick);
      expect(c.tn + c.fp).toBe(healthy);
      const s = scoresOf(c);
      if (sick > 0) {
        expect((s.sensitivity as number) + c.fn / sick).toBeCloseTo(1, 12);
      } else {
        expect(s.sensitivity).toBeNull();
      }
      if (healthy > 0) {
        expect((s.specificity as number) + c.fp / healthy).toBeCloseTo(1, 12);
      } else {
        expect(s.specificity).toBeNull();
      }
      if (sick > 0 && healthy > 0) {
        expect(s.balancedAccuracy).toBeCloseTo(((s.sensitivity as number) + (s.specificity as number)) / 2, 12);
      }
    }
  });
});

describe('always-pneumonia baseline', () => {
  it('234 normal + 390 pneumonia -> 62.5 / 100 / 0 (and balanced exactly 50)', () => {
    const labels = [...rep(0, 234), ...rep(1, 390)];
    const b = alwaysPneumoniaBaseline(labels);
    expect(b.confusion).toEqual({ tp: 390, fn: 0, fp: 234, tn: 0 });
    expect(b.accuracy).toBe(0.625);
    expect(b.sensitivity).toBe(1);
    expect(b.specificity).toBe(0);
    expect(b.balancedAccuracy).toBe(0.5);
    expect(formatPercent(b.accuracy)).toBe('62.5%');
    expect(formatPercent(b.sensitivity)).toBe('100.0%');
    expect(formatPercent(b.specificity)).toBe('0.0%');
  });
  it('100 + 100 adult -> 50 / 100 / 0; all-healthy set has null sensitivity', () => {
    const b = alwaysPneumoniaBaseline([...rep(0, 100), ...rep(1, 100)]);
    expect([b.accuracy, b.sensitivity, b.specificity]).toEqual([0.5, 1, 0]);
    const h = alwaysPneumoniaBaseline(rep(0, 5));
    expect(h.accuracy).toBe(0);
    expect(h.sensitivity).toBeNull();
  });
});

describe('classWeights N / (2 N_c)', () => {
  it('255 / 765 -> normal 2.0, pneumonia 2/3', () => {
    const w = classWeights(255, 765);
    expect(w[0]).toBe(1020 / 510); // exactly 2
    expect(w[0]).toBe(2);
    expect(w[1]).toBeCloseTo(2 / 3, 12);
    expect(w[0].toFixed(2)).toBe('2.00');
    expect(w[1].toFixed(2)).toBe('0.67');
  });
  it('adult mix 355 / 865 -> 1220/710 and 1220/1730', () => {
    const w = classWeights(355, 865);
    expect(w[0]).toBeCloseTo(1220 / 710, 12);
    expect(w[1]).toBeCloseTo(1220 / 1730, 12);
    expect(w[0].toFixed(2)).toBe('1.72');
    expect(w[1].toFixed(2)).toBe('0.71');
  });
  it('a balanced set gives 1 / 1, and total weight is N for any split', () => {
    expect(classWeights(50, 50)).toEqual({ 0: 1, 1: 1 });
    const rng = mulberry32(5);
    for (let i = 0; i < 100; i++) {
      const a = 1 + Math.floor(rng() * 500);
      const b = 1 + Math.floor(rng() * 500);
      const w = classWeights(a, b);
      expect(a * w[0] + b * w[1]).toBeCloseTo(a + b, 9);
      expect(a * w[0]).toBeCloseTo(b * w[1], 9); // each class contributes N/2
    }
  });
});

describe('decide() boundaries', () => {
  it('tau = 0.5 never refers, for any probability including exactly 0.5, 0 and 1', () => {
    for (const p of [0, 1e-9, 0.25, 0.4999999, 0.5, 0.5000001, 0.75, 1 - 1e-9, 1]) {
      expect(decide(p, 0.5)).not.toBe('refer');
    }
    const rng = mulberry32(3);
    for (let i = 0; i < 1000; i++) expect(decide(rng(), 0.5)).not.toBe('refer');
  });
  it('p exactly 0.5 is pneumonia (>=), just below is normal', () => {
    expect(decide(0.5, 0.5)).toBe('pneumonia');
    expect(decide(0.4999999, 0.5)).toBe('normal');
  });
  it('p exactly at tau (confidence == tau) is answered, not referred', () => {
    expect(decide(0.75, 0.75)).toBe('pneumonia'); // confidence 0.75, not < 0.75
    expect(decide(0.25, 0.75)).toBe('normal'); // confidence 1 - 0.25 = 0.75
    expect(decide(0.7499, 0.75)).toBe('refer');
    expect(decide(0.2501, 0.75)).toBe('refer');
  });
  it('confidence is symmetric around 0.5: decide(p) refers iff decide(1 - p) refers', () => {
    for (let k = 0; k <= 64; k++) {
      const p = k / 64; // dyadic, so 1 - p is exact
      expect(confidenceOf(p)).toBe(confidenceOf(1 - p));
      for (const tau of [0.5, 0.6, 0.75, 0.9, 0.99]) {
        expect(decide(p, tau) === 'refer').toBe(decide(1 - p, tau) === 'refer');
      }
    }
  });
  it('the call (when not referred) is independent of tau', () => {
    for (const p of [0.01, 0.2, 0.49, 0.5, 0.51, 0.8, 0.99]) {
      const d = decide(p, 0.5);
      for (const tau of [0.5, 0.6, 0.9]) {
        const dt = decide(p, tau);
        if (dt !== 'refer') expect(dt).toBe(d);
      }
    }
  });
});

describe('metricsWithAbstain: hand-computed half credit', () => {
  // truth:   1     1     1     1     0     0     0     0
  // p:     0.95  0.55  0.25  0.52  0.10  0.45  0.80  0.05
  // tau = 0.7 (confidence needed >= 0.7):
  //   0.95 -> conf .95 -> pneumonia, sick   -> TP
  //   0.55 -> conf .55 -> REFER (sick)
  //   0.25 -> conf .75 -> normal,    sick   -> FN
  //   0.52 -> conf .52 -> REFER (sick)
  //   0.10 -> conf .90 -> normal,  healthy  -> TN
  //   0.45 -> conf .55 -> REFER (healthy)
  //   0.80 -> conf .80 -> pneumonia, healthy-> FP
  //   0.05 -> conf .95 -> normal,  healthy  -> TN
  const labels = [1, 1, 1, 1, 0, 0, 0, 0];
  const probs = [0.95, 0.55, 0.25, 0.52, 0.1, 0.45, 0.8, 0.05];

  it('tau 0.7: referred 3, coverage 5/8, confusion over answered cases only', () => {
    const m = metricsWithAbstain(labels, probs, 0.7);
    expect(m.n).toBe(8);
    expect(m.referred).toBe(3);
    expect(m.referredFraction).toBeCloseTo(3 / 8, 12);
    expect(m.coverage).toBeCloseTo(5 / 8, 12);
    expect(m.referredByClass).toEqual({ normal: 1, pneumonia: 2 });
    expect(m.confusion).toEqual({ tp: 1, fn: 1, fp: 1, tn: 2 });
    expect(m.accuracy).toBeCloseTo(3 / 5, 12);
    expect(m.sensitivity).toBeCloseTo(1 / 2, 12);
    expect(m.specificity).toBeCloseTo(2 / 3, 12);
  });

  it('tau 0.7: class credit = (correct + 0.5 * referred) / n_class; score is their mean', () => {
    const m = metricsWithAbstain(labels, probs, 0.7);
    // pneumonia: 1 correct + 0.5 * 2 referred = 2 -> 2/4 = 0.5
    expect(m.classCredit.pneumonia).toBeCloseTo(0.5, 12);
    // normal: 2 correct + 0.5 * 1 referred = 2.5 -> 2.5/4 = 0.625
    expect(m.classCredit.normal).toBeCloseTo(0.625, 12);
    expect(m.halfCreditScore).toBeCloseTo(0.5625, 12);
  });

  it('tau 0.5 refers nobody and the half-credit score equals balanced accuracy of the 0.5 rule', () => {
    const m = metricsWithAbstain(labels, probs, 0.5);
    expect(m.referred).toBe(0);
    expect(m.coverage).toBe(1);
    // 0.5 rule: 0.95 TP, 0.55 TP, 0.25 FN, 0.52 TP, 0.10 TN, 0.45 TN, 0.80 FP, 0.05 TN
    expect(m.confusion).toEqual({ tp: 3, fn: 1, fp: 1, tn: 3 });
    expect(m.halfCreditScore).toBeCloseTo((3 / 4 + 3 / 4) / 2, 12);
    expect(m.halfCreditScore).toBeCloseTo(m.balancedAccuracy as number, 12);
  });

  it('property: at tau 0.5 the score equals balanced accuracy for random sets with both classes', () => {
    const rng = mulberry32(77);
    for (let t = 0; t < 100; t++) {
      const n = 10 + Math.floor(rng() * 90);
      const y = Array.from({ length: n }, (_, i) => (i < 2 ? i : rng() < 0.5 ? 1 : 0)); // guarantees both classes
      const p = Array.from({ length: n }, () => rng());
      const m = metricsWithAbstain(y, p, 0.5);
      const ref = scoresOf(confusionMatrix(y, hardPredictions(p)));
      expect(m.referred).toBe(0);
      expect(m.halfCreditScore as number).toBeCloseTo(ref.balancedAccuracy as number, 12);
    }
  });

  it('property: the referral count is monotone non-decreasing in tau, and coverage non-increasing', () => {
    const rng = mulberry32(99);
    for (let t = 0; t < 60; t++) {
      const n = 5 + Math.floor(rng() * 200);
      const y = Array.from({ length: n }, () => (rng() < 0.5 ? 1 : 0));
      const p = Array.from({ length: n }, () => rng());
      let prevRef = -1;
      let prevCov = 2;
      for (let tau = 0.5; tau <= 0.99 + 1e-9; tau += 0.01) {
        const m = metricsWithAbstain(y, p, tau);
        expect(m.referred).toBeGreaterThanOrEqual(prevRef);
        expect(m.coverage as number).toBeLessThanOrEqual(prevCov + 1e-12);
        prevRef = m.referred;
        prevCov = m.coverage as number;
      }
    }
  });

  it('refer-everything gives exactly half credit; empty set gives nulls', () => {
    const m = metricsWithAbstain([1, 0, 1, 0], [0.6, 0.6, 0.4, 0.4], 0.99);
    expect(m.referred).toBe(4);
    expect(m.halfCreditScore).toBe(0.5);
    expect(m.accuracy).toBeNull();
    expect(m.sensitivity).toBeNull();
    const e = metricsWithAbstain([], [], 0.8);
    expect(e.coverage).toBeNull();
    expect(e.referredFraction).toBeNull();
    expect(e.halfCreditScore).toBeNull();
  });

  it('a missing class makes the half-credit score null rather than NaN', () => {
    const m = metricsWithAbstain([1, 1], [0.9, 0.6], 0.7);
    expect(m.classCredit.normal).toBeNull();
    expect(m.halfCreditScore).toBeNull();
    expect(Number.isNaN(m.classCredit.pneumonia as number)).toBe(false);
  });
});

describe('worstCase', () => {
  it('picks the minimum and reports which set', () => {
    expect(worstCase({ kermany_test: 0.91, adult_test: 0.62, stress: 0.75 })).toEqual({ score: 0.62, set: 'adult_test' });
  });
  it('ignores null / undefined / NaN entries; all-missing -> null', () => {
    expect(worstCase({ kermany_test: 0.9, adult_test: null, stress: undefined })).toEqual({ score: 0.9, set: 'kermany_test' });
    expect(worstCase({ a: Number.NaN, b: 0.4 })).toEqual({ score: 0.4, set: 'b' });
    expect(worstCase({ a: null, b: undefined })).toBeNull();
    expect(worstCase({})).toBeNull();
  });
  it('a score of exactly 0 is a real score, not "missing"', () => {
    expect(worstCase({ a: 0.5, b: 0 })).toEqual({ score: 0, set: 'b' });
  });
});

describe('flipRate per true class', () => {
  it('counts flips within each TRUE class and overall', () => {
    const y = [0, 0, 0, 1, 1];
    const before = [0, 0, 1, 1, 1];
    const after = [1, 0, 1, 0, 1];
    const f = flipRate(y, before, after);
    expect(f.normal).toMatchObject({ n: 3, flipped: 1 });
    expect(f.normal.rate).toBeCloseTo(1 / 3, 12);
    expect(f.pneumonia).toMatchObject({ n: 2, flipped: 1 });
    expect(f.pneumonia.rate).toBeCloseTo(0.5, 12);
    expect(f.overall).toMatchObject({ n: 5, flipped: 2 });
    expect(f.overall.rate).toBeCloseTo(0.4, 12);
  });
  it('identical runs flip nothing; the stamp story: all normals flip to pneumonia = 100%', () => {
    const y = [0, 0, 1];
    expect(flipRate(y, [0, 0, 1], [0, 0, 1]).overall.rate).toBe(0);
    const f = flipRate(y, [0, 0, 1], [1, 1, 1]);
    expect(f.normal.rate).toBe(1);
    expect(f.pneumonia.rate).toBe(0);
  });
  it('absent class -> null rate, not NaN', () => {
    const f = flipRate([0, 0], [0, 0], [1, 0]);
    expect(f.pneumonia.rate).toBeNull();
    expect(f.pneumonia.n).toBe(0);
  });
});
