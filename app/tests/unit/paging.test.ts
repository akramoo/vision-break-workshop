import { describe, expect, it } from 'vitest';
import {
  DEFAULT_PAGE_SIZE,
  PAGE_SIZES,
  applyView,
  buildRows,
  categoryCounts,
  classCounts,
  countsLabel,
  effectiveMode,
  pageInfo,
  pageLabel,
  type PredRow,
} from '../../src/ui/paging';

// 10 rows: truth = [1,1,1,1,0,0,0,0,1,0]; probabilities chosen so the answers at tau 0.5 are known.
const ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
const labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0];
//                 ok   FN   ok   ok   ok   FP   ok   ok   FN   ok
const probs = [0.9, 0.2, 0.95, 0.7, 0.1, 0.8, 0.3, 0.05, 0.4, 0.45];

describe('buildRows', () => {
  it('keeps dataset order and marks wrong answers', () => {
    const rows = buildRows(ids, probs, labels, 0.5);
    expect(rows.map((r) => r.id)).toEqual(ids);
    expect(rows.map((r) => r.i)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(rows.filter((r) => r.wrong).map((r) => r.id)).toEqual(['b', 'f', 'i']);
    expect(rows.every((r) => r.d !== 'refer')).toBe(true);
  });

  it('refers uncertain rows and never counts them as wrong', () => {
    const rows = buildRows(ids, probs, labels, 0.8);
    const referred = rows.filter((r) => r.d === 'refer').map((r) => r.id);
    expect(referred).toEqual(['d', 'g', 'i', 'j']);
    expect(rows.find((r) => r.id === 'i')?.wrong).toBe(false); // 0.4 -> confidence 0.6 < 0.8
    expect(rows.find((r) => r.id === 'b')?.wrong).toBe(true); // 0.2 -> confidence 0.8, answered normal, truth pneumonia
  });
});

describe('counts', () => {
  it('mistakes + referred + correct = total at every tau', () => {
    for (const tau of [0.5, 0.6, 0.8, 0.99]) {
      const c = categoryCounts(buildRows(ids, probs, labels, tau));
      expect(c.mistakes + c.referred + c.correct).toBe(10);
      expect(c.total).toBe(10);
    }
  });

  it('matches the hand-counted values at tau 0.5', () => {
    const c = categoryCounts(buildRows(ids, probs, labels, 0.5));
    expect(c).toEqual({ total: 10, mistakes: 3, referred: 0, correct: 7 });
    expect(countsLabel(c)).toBe('Mistakes 3 | Referred 0 | Correct 7');
  });

  it('counts true classes', () => {
    expect(classCounts(buildRows(ids, probs, labels, 0.5))).toEqual({ total: 10, pneumonia: 5, negative: 5 });
  });

  it('handles an empty set', () => {
    expect(categoryCounts([])).toEqual({ total: 0, mistakes: 0, referred: 0, correct: 0 });
    expect(classCounts([])).toEqual({ total: 0, pneumonia: 0, negative: 0 });
  });
});

describe('applyView', () => {
  const rows = buildRows(ids, probs, labels, 0.8);
  const idsOf = (r: readonly PredRow[]) => r.map((x) => x.id);

  it('mistakes first: wrong ones first (most confident first), then the rest in dataset order', () => {
    const out = applyView(rows, 'mistakes-first', 'all');
    expect(out).toHaveLength(rows.length);
    const wrong = rows.filter((r) => r.wrong);
    // at tau 0.8 the wrong answers are b (p 0.2, confidence 0.8) and f (p 0.8, confidence 0.8): a tie, so dataset order
    expect(idsOf(wrong)).toEqual(['b', 'f']);
    expect(idsOf(out.slice(0, wrong.length))).toEqual(['b', 'f']);
    const rest = idsOf(out.slice(wrong.length));
    expect(rest).toEqual(idsOf(rows.filter((r) => !r.wrong)));
  });

  it('orders mistakes by descending confidence max(p, 1 - p); ties keep dataset order', () => {
    // all wrong at tau 0.5: truth is the opposite of the answer
    const ids2 = ['a', 'b', 'c', 'd', 'e', 'f'];
    const probs2 = [0.6, 0.02, 0.9, 0.3, 0.98, 0.1]; // confidences .6 .98 .9 .7 .98 .9
    const labels2 = [0, 1, 0, 1, 0, 1];
    const r2 = buildRows(ids2, probs2, labels2, 0.5);
    expect(r2.every((r) => r.wrong)).toBe(true);
    const expected = ['b', 'e', 'c', 'f', 'd', 'a']; // .98 .98 .9 .9 .7 .6 with ties b<e and c<f in dataset order
    expect(idsOf(applyView(r2, 'only-mistakes', 'all'))).toEqual(expected);
    expect(idsOf(applyView(r2, 'mistakes-first', 'all'))).toEqual(expected);
    // the class filter keeps the same relative order
    expect(idsOf(applyView(r2, 'only-mistakes', 'pneumonia'))).toEqual(['b', 'f', 'd']);
    expect(idsOf(applyView(r2, 'only-mistakes', 'negative'))).toEqual(['e', 'c', 'a']);
  });

  it('a confident miss on a late page of dataset order comes first in mistakes first', () => {
    const n = 100;
    const idsN = Array.from({ length: n }, (_, i) => `x${i}`);
    // 98 false alarms with confidence 0.6, then 2 confident misses at the very end
    const probsN = idsN.map((_, i) => (i < 98 ? 0.6 : 0.01));
    const labelsN = idsN.map((_, i) => (i < 98 ? 0 : 1));
    const out = applyView(buildRows(idsN, probsN, labelsN, 0.5), 'mistakes-first', 'all');
    expect(idsOf(out.slice(0, 3))).toEqual(['x98', 'x99', 'x0']);
    expect(idsOf(out.slice(0, 40)).slice(0, 2)).toEqual(['x98', 'x99']);
  });

  it('only referred: most confident first, ties in dataset order; only correct stays in dataset order', () => {
    const ids3 = ['a', 'b', 'c', 'd', 'e'];
    const probs3 = [0.55, 0.7, 0.45, 0.3, 0.95]; // tau 0.8: a, b, c referred (conf .55 .7 .55), d referred? conf .7 -> yes, e answered
    const labels3 = [1, 0, 0, 1, 1];
    const r3 = buildRows(ids3, probs3, labels3, 0.8);
    expect(idsOf(r3.filter((r) => r.d === 'refer'))).toEqual(['a', 'b', 'c', 'd']);
    expect(idsOf(applyView(r3, 'only-referred', 'all'))).toEqual(['b', 'd', 'a', 'c']);
    expect(idsOf(applyView(r3, 'only-correct', 'all'))).toEqual(['e']);
  });

  it('ordering is stable across repeated calls and does not modify the input', () => {
    const r2 = buildRows(['a', 'b', 'c'], [0.1, 0.9, 0.1], [1, 0, 1], 0.5); // three wrong answers, equal confidence
    const copy = r2.map((r) => ({ ...r }));
    for (const mode of ['mistakes-first', 'only-mistakes'] as const) {
      expect(idsOf(applyView(r2, mode, 'all'))).toEqual(['a', 'b', 'c']);
      expect(applyView(r2, mode, 'all')).toEqual(applyView(r2, mode, 'all'));
    }
    expect(r2).toEqual(copy);
  });

  it('dataset order returns every row in place', () => {
    expect(idsOf(applyView(rows, 'dataset', 'all'))).toEqual(ids);
  });

  it('the three filters partition the set', () => {
    const m = applyView(rows, 'only-mistakes', 'all');
    const r = applyView(rows, 'only-referred', 'all');
    const c = applyView(rows, 'only-correct', 'all');
    expect(m.length + r.length + c.length).toBe(rows.length);
    expect(new Set([...idsOf(m), ...idsOf(r), ...idsOf(c)]).size).toBe(rows.length);
    expect(m.every((x) => x.wrong)).toBe(true);
    expect(r.every((x) => x.d === 'refer')).toBe(true);
    expect(c.every((x) => !x.wrong && x.d !== 'refer')).toBe(true);
  });

  it('filters by TRUE class, also when combined with a mode', () => {
    const p = applyView(rows, 'dataset', 'pneumonia');
    const n = applyView(rows, 'dataset', 'negative');
    expect(p.every((x) => x.truth === 1)).toBe(true);
    expect(n.every((x) => x.truth === 0)).toBe(true);
    expect(p.length + n.length).toBe(rows.length);
    const mp = applyView(rows, 'only-mistakes', 'pneumonia');
    expect(mp.every((x) => x.wrong && x.truth === 1)).toBe(true);
  });

  it('is deterministic and does not modify its input', () => {
    const copy = rows.map((r) => ({ ...r }));
    const a = applyView(rows, 'mistakes-first', 'negative');
    const b = applyView(rows, 'mistakes-first', 'negative');
    expect(a).toEqual(b);
    expect(rows).toEqual(copy);
    expect(applyView(rows, 'dataset', 'all')).not.toBe(rows);
  });

  it('returns an empty list when nothing matches', () => {
    const none = buildRows(ids, probs, labels, 0.5);
    expect(applyView(none, 'only-referred', 'all')).toEqual([]);
  });
});

describe('effectiveMode', () => {
  it('drops "only referred" when nothing can be referred', () => {
    expect(effectiveMode('only-referred', 0.5)).toBe('mistakes-first');
    expect(effectiveMode('only-referred', 0.7)).toBe('only-referred');
    expect(effectiveMode('only-mistakes', 0.5)).toBe('only-mistakes');
  });
});

describe('pageInfo / pageLabel', () => {
  it('the Kermany test set at 40 per page', () => {
    expect(PAGE_SIZES).toEqual([20, 40, 100]);
    expect(DEFAULT_PAGE_SIZE).toBe(40);
    const first = pageInfo(624, 1, 40);
    expect(first).toMatchObject({ page: 1, pages: 16, start: 0, end: 40 });
    expect(pageLabel(first)).toBe('Page 1 of 16 (rows 1-40 of 624)');
    expect(pageLabel(pageInfo(624, 3, 40))).toBe('Page 3 of 16 (rows 81-120 of 624)');
    expect(pageLabel(pageInfo(624, 16, 40))).toBe('Page 16 of 16 (rows 601-624 of 624)');
  });

  it('other page sizes', () => {
    expect(pageInfo(624, 1, 20).pages).toBe(32);
    expect(pageInfo(624, 1, 100).pages).toBe(7);
    expect(pageLabel(pageInfo(624, 7, 100))).toBe('Page 7 of 7 (rows 601-624 of 624)');
    expect(pageInfo(200, 5, 40)).toMatchObject({ pages: 5, start: 160, end: 200 });
    expect(pageInfo(40, 1, 40).pages).toBe(1);
    expect(pageInfo(41, 2, 40)).toMatchObject({ start: 40, end: 41, pages: 2 });
  });

  it('clamps out-of-range pages', () => {
    expect(pageInfo(624, 0, 40).page).toBe(1);
    expect(pageInfo(624, -5, 40).page).toBe(1);
    expect(pageInfo(624, 99, 40).page).toBe(16);
    expect(pageInfo(624, Number.NaN, 40).page).toBe(1);
  });

  it('an empty list has one empty page', () => {
    const e = pageInfo(0, 3, 40);
    expect(e).toMatchObject({ page: 1, pages: 1, start: 0, end: 0 });
    expect(pageLabel(e)).toBe('No rows match this filter');
  });

  it('slicing every page reproduces the list exactly once', () => {
    const list = Array.from({ length: 624 }, (_, i) => i);
    for (const size of PAGE_SIZES) {
      const seen: number[] = [];
      const { pages } = pageInfo(list.length, 1, size);
      for (let p = 1; p <= pages; p++) {
        const info = pageInfo(list.length, p, size);
        expect(info.end - info.start).toBeLessThanOrEqual(size);
        seen.push(...list.slice(info.start, info.end));
      }
      expect(seen).toEqual(list);
    }
  });
});
