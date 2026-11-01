import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { EvalSetId } from '../../src/data/evalSets';
import { formatPercent, metricsWithAbstain } from '../../src/ml/metrics';
import { describeModelConfig, modelConfigOf } from '../../src/ml/modelConfig';
import {
  RESULT_SET_IDS,
  buildResultsExport,
  deltaPoints,
  formatDelta,
  leaderboardOf,
  resultsExportFileName,
  shownPercent,
  summarizeResults,
  type EvalInput,
  type SetScores,
} from '../../src/ml/results';
import { SNAPSHOTS_KEY, STEPS_KEY } from '../../src/state/persistence';
import {
  MAX_SNAPSHOTS,
  defaultSnapshotName,
  loadSnapshots,
  makeSnapshot,
  saveSnapshots,
  validateSnapshot,
  validateSnapshots,
  type Snapshot,
  type SnapshotConfig,
} from '../../src/state/snapshots';
import { ALL_STEP_IDS, STEP_GROUPS, loadSteps, sanitizeSteps, saveSteps, toggleStepId } from '../../src/state/steps';
import { MemStorage, installStorage } from './storageStub';

function ev(labels: number[], probs: number[], over: Partial<EvalInput> = {}): EvalInput {
  return { labels, probs, headId: 7, usedPrecomputed: true, tau: 0.5, timestamp: 1, ...over };
}

// 4 pneumonia + 4 normal. Model: 3/4 pneumonia right, 2/4 normal right.
const L = [1, 1, 1, 1, 0, 0, 0, 0];
const P = [0.9, 0.9, 0.9, 0.2, 0.1, 0.1, 0.6, 0.7];

describe('summarizeResults / worst case', () => {
  it('the leaderboard value is the minimum of the three half-credit scores and names that set', () => {
    // kermany: 100% / 100%  -> 1.0 ; adult: 3/4 and 2/4 -> 0.625 ; stress: 2/4 and 3/4 -> 0.625 (tie goes to the first set)
    const kermany = ev(L, [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]);
    const adult = ev(L, P);
    const stress = ev(L, [0.9, 0.9, 0.2, 0.2, 0.1, 0.1, 0.1, 0.8]);
    const s = summarizeResults({ kermany_test: kermany, adult_test: adult, stress }, 7, 0.5);
    expect(s.complete).toBe(true);
    expect(s.perSet.kermany_test?.scores.halfCreditScore).toBe(1);
    expect(s.perSet.adult_test?.scores.halfCreditScore).toBeCloseTo(0.625, 10);
    expect(s.perSet.stress?.scores.halfCreditScore).toBeCloseTo(0.625, 10);
    expect(s.leaderboard?.score).toBeCloseTo(0.625, 10);
    expect(s.leaderboard?.set).toBe('adult_test');
  });

  it('equals the min of metricsWithAbstain(...).halfCreditScore computed by hand', () => {
    const sets: Record<EvalSetId, EvalInput> = {
      kermany_test: ev(L, [0.95, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.55]),
      adult_test: ev(L, [0.6, 0.55, 0.4, 0.3, 0.45, 0.7, 0.8, 0.2]),
      stress: ev(L, [0.9, 0.2, 0.2, 0.2, 0.1, 0.9, 0.9, 0.3]),
    };
    const tau = 0.7;
    const s = summarizeResults(sets, 7, tau);
    const hand = RESULT_SET_IDS.map((id) => metricsWithAbstain(sets[id].labels, sets[id].probs, tau).halfCreditScore as number);
    expect(s.leaderboard?.score).toBe(Math.min(...hand));
    expect(RESULT_SET_IDS[hand.indexOf(Math.min(...hand))]).toBe(s.leaderboard?.set);
  });

  it('refer everything scores exactly 50 on every set, equal to the baseline', () => {
    const sets = { kermany_test: ev(L, P), adult_test: ev(L, P), stress: ev(L, P) };
    const s = summarizeResults(sets, 7, 0.99);
    // P has no value with confidence >= 0.99, so every case is referred
    for (const id of RESULT_SET_IDS) {
      expect(s.perSet[id]?.scores.referred).toBe(8);
      expect(s.perSet[id]?.scores.halfCreditScore).toBe(0.5);
      expect(s.perSet[id]?.baseline.halfCreditScore).toBe(0.5);
    }
    expect(s.leaderboard?.score).toBe(0.5);
  });

  it('is recomputed when tau changes (no stale numbers) and reports the tau it was evaluated at', () => {
    const e = ev(L, P, { tau: 0.5 });
    const at50 = summarizeResults({ kermany_test: e }, 7, 0.5);
    const at90 = summarizeResults({ kermany_test: e }, 7, 0.9);
    expect(at50.perSet.kermany_test?.scores.referred).toBe(0);
    expect(at90.perSet.kermany_test?.scores.referred).toBeGreaterThan(0);
    expect(at90.perSet.kermany_test?.scores.halfCreditScore).not.toBe(at50.perSet.kermany_test?.scores.halfCreditScore);
    expect(at90.tau).toBe(0.9);
    expect(at90.perSet.kermany_test?.evaluatedAtTau).toBe(0.5);
  });

  it('has no leaderboard until all three sets are evaluated, and ignores evaluations of another head', () => {
    const part = summarizeResults({ kermany_test: ev(L, P), adult_test: ev(L, P) }, 7, 0.5);
    expect(part.complete).toBe(false);
    expect(part.missingSets).toEqual(['stress']);
    expect(part.leaderboard).toBeNull();
    const other = summarizeResults({ kermany_test: ev(L, P, { headId: 3 }), adult_test: ev(L, P), stress: ev(L, P) }, 7, 0.5);
    expect(other.evaluatedSets).toEqual(['adult_test', 'stress']);
    expect(other.leaderboard).toBeNull();
    expect(summarizeResults({ kermany_test: ev(L, P) }, null, 0.5).evaluatedSets).toEqual([]);
  });

  it('the always-pneumonia baseline scores 50% half credit on every set, accuracy = share of pneumonia', () => {
    const kermanyLabels = [...Array<number>(390).fill(1), ...Array<number>(234).fill(0)];
    const adultLabels = [...Array<number>(100).fill(1), ...Array<number>(100).fill(0)];
    const s = summarizeResults(
      {
        kermany_test: ev(kermanyLabels, kermanyLabels.map(() => 0.9)),
        adult_test: ev(adultLabels, adultLabels.map(() => 0.9)),
        stress: ev(kermanyLabels, kermanyLabels.map(() => 0.9)),
      },
      7,
      0.5,
    );
    expect(s.perSet.kermany_test?.baseline).toMatchObject({ accuracy: 0.625, sensitivity: 1, specificity: 0, halfCreditScore: 0.5 });
    expect(s.perSet.adult_test?.baseline).toMatchObject({ accuracy: 0.5, halfCreditScore: 0.5 });
    expect(s.perSet.stress?.baseline.halfCreditScore).toBe(0.5);
  });

  it('leaderboardOf needs all three numeric scores', () => {
    expect(leaderboardOf({ kermany_test: 0.9, adult_test: 0.6, stress: 0.7 })).toEqual({ score: 0.6, set: 'adult_test' });
    expect(leaderboardOf({ kermany_test: 0.9, adult_test: null, stress: 0.7 })).toBeNull();
    expect(leaderboardOf({ kermany_test: 0.9, adult_test: 0.6 })).toBeNull();
    expect(leaderboardOf({ kermany_test: 0.9, adult_test: Number.NaN, stress: 0.7 })).toBeNull();
  });
});

describe('deltas', () => {
  it('computes percentage points and formats them with a sign', () => {
    expect(deltaPoints(0.5, 0.62)).toBeCloseTo(12, 10);
    expect(deltaPoints(0.8, 0.7)).toBeCloseTo(-10, 10);
    // the change is computed from the displayed values: 81.3% -> 87.5% is 6.2 points (exact difference 6.25)
    expect(formatPercent(0.8125)).toBe('81.3%');
    expect(formatPercent(0.875)).toBe('87.5%');
    expect(deltaPoints(0.8125, 0.875)).toBe(6.2);
    expect(shownPercent(0.8125)).toBe(81.3);
    expect(deltaPoints(null, 0.7)).toBeNull();
    expect(deltaPoints(0.7, undefined)).toBeNull();
    expect(formatDelta(12.34)).toBe('+12.3 pts');
    expect(formatDelta(-4)).toBe('-4.0 pts');
    expect(formatDelta(0)).toBe('0.0 pts');
    expect(formatDelta(0.04)).toBe('0.0 pts');
    expect(formatDelta(-0.04)).toBe('0.0 pts');
    expect(formatDelta(null)).toBe('n/a');
  });
});

describe('model config', () => {
  const settings = { mode: 'balanced', adultMix: true, augmentation: false, epochs: 30, lr: 0.01, seed: 1 } as const;
  it('prefers the trained head record over the current Train settings', () => {
    const info = { mode: 'realistic', adultMix: false, augmentation: true, completedEpochs: 12, lr: 0.001, seed: 42 } as const;
    expect(modelConfigOf(info, settings)).toEqual({ mode: 'realistic', adultMix: false, augmentation: true, epochs: 12, lr: 0.001, seed: 42 });
    expect(modelConfigOf(null, settings)).toEqual({ mode: 'balanced', adultMix: true, augmentation: false, epochs: 30, lr: 0.01, seed: 1 });
  });
  it('describes the config on one line', () => {
    expect(describeModelConfig(modelConfigOf(null, settings))).toBe('balanced, adult mix on, augmentation off, 30 epochs, lr 1.0e-2, seed 1');
  });
});

describe('results JSON', () => {
  it('has the disclaimer, config, per-set metrics, snapshots, leaderboard and failure counts', () => {
    const sets = { kermany_test: ev(L, P), adult_test: ev(L, P), stress: ev(L, P) };
    const summary = summarizeResults(sets, 7, 0.5);
    const config = modelConfigOf(null, { mode: 'realistic', adultMix: false, augmentation: false, epochs: 20, lr: 0.001, seed: 42 });
    const out = buildResultsExport({ summary, config, snapshots: [], failureCounts: { shift: 2 }, now: new Date('2026-11-02T10:00:00Z'), version: '1.2.3' });
    expect(Object.keys(out).sort()).toEqual(
      ['appVersion', 'config', 'disclaimer', 'exportedAt', 'explanation', 'failureCounts', 'leaderboard', 'perSet', 'snapshots', 'tau'].sort(),
    );
    expect(out.disclaimer).toBe('Educational only. Not for clinical use.');
    expect(out.config).toMatchObject({ mode: 'realistic', epochs: 20, tau: 0.5, stampTrained: false });
    expect(out.leaderboard).toMatchObject({ set: summary.leaderboard?.set, score: summary.leaderboard?.score, tau: 0.5 });
    expect(Object.keys(out.perSet)).toEqual(['kermany_test', 'adult_test', 'stress']);
    expect(out.failureCounts).toEqual({ shift: 2 });
    expect(JSON.parse(JSON.stringify(out)).perSet.adult_test.scores.halfCreditScore).toBeCloseTo(0.625, 10);
    expect(resultsExportFileName(new Date(2026, 10, 2, 9, 5))).toBe('vision-results-20261102-0905.json');
  });

  it('has a null leaderboard when a set is missing', () => {
    const summary = summarizeResults({ kermany_test: ev(L, P) }, 7, 0.5);
    const out = buildResultsExport({ summary, config: modelConfigOf(null, { mode: 'realistic', adultMix: false, augmentation: false, epochs: 20, lr: 0.001, seed: 42 }), snapshots: [], failureCounts: {} });
    expect(out.leaderboard).toBeNull();
  });
});

// ---- snapshots ----

const CONFIG: SnapshotConfig = { mode: 'realistic', adultMix: false, augmentation: false, epochs: 20, lr: 0.001, seed: 42, tau: 0.5, stampTrained: false, usePrecomputed: true };

function scores(half: number): SetScores {
  return { n: 100, accuracy: half, sensitivity: half, specificity: half, balancedAccuracy: half, referred: 0, referredFraction: 0, coverage: 1, halfCreditScore: half };
}
function snap(name: string, k: number, a: number, s: number): Snapshot {
  return makeSnapshot({ name, existing: [], config: CONFIG, perSet: { kermany_test: scores(k), adult_test: scores(a), stress: scores(s) }, now: 1000 });
}

describe('snapshots', () => {
  let store: MemStorage;
  let original: Storage | undefined;
  beforeEach(() => {
    store = new MemStorage();
    original = installStorage(store);
  });
  afterEach(() => installStorage(original));

  it('names them Before fix, After fix 1, After fix 2, ...', () => {
    expect(defaultSnapshotName([])).toBe('Before fix');
    expect(defaultSnapshotName([{ name: 'Before fix' }])).toBe('After fix 1');
    expect(defaultSnapshotName([{ name: 'Before fix' }, { name: 'After fix 1' }])).toBe('After fix 2');
    expect(defaultSnapshotName([{ name: 'a' }, { name: 'After fix 1' }, { name: 'After fix 3' }])).toBe('After fix 2');
  });

  it('makeSnapshot computes the worst case and falls back to the default name for a blank one', () => {
    const s = makeSnapshot({ name: '   ', existing: [{ name: 'Before fix' }], config: CONFIG, perSet: { kermany_test: scores(0.9), adult_test: scores(0.6), stress: scores(0.7) } });
    expect(s.name).toBe('After fix 1');
    expect(s.leaderboard).toEqual({ score: 0.6, set: 'adult_test' });
    expect(snap('x'.repeat(100), 1, 1, 1).name).toHaveLength(40);
  });

  it('round-trips through localStorage under vbw:v1:snapshots', () => {
    const list = [snap('Before fix', 0.9, 0.55, 0.6), snap('After fix 1', 0.88, 0.7, 0.8)];
    expect(SNAPSHOTS_KEY).toBe('vbw:v1:snapshots');
    expect(saveSnapshots(list)).toBe(true);
    expect(loadSnapshots()).toEqual(list);
    saveSnapshots([]);
    expect(store.getItem(SNAPSHOTS_KEY)).toBeNull();
  });

  it('keeps at most 8, drops malformed ones and recomputes the leaderboard instead of trusting it', () => {
    const many = Array.from({ length: 12 }, (_, i) => ({ ...snap(`s${i}`, 0.9, 0.7, 0.8), id: `id${i}` }));
    expect(validateSnapshots(many)).toHaveLength(MAX_SNAPSHOTS);
    expect(MAX_SNAPSHOTS).toBe(8);
    const good = snap('ok', 0.9, 0.7, 0.8);
    const tampered = { ...good, leaderboard: { score: 0.99, set: 'stress' } };
    expect(validateSnapshot(tampered)?.leaderboard).toEqual({ score: 0.7, set: 'adult_test' });
    expect(validateSnapshots([good, { ...good, id: 'b', perSet: { kermany_test: scores(1) } }, 'junk', null])).toEqual([good]);
    expect(validateSnapshot({ ...good, config: { ...good.config, mode: 'x' } })).toBeNull();
    expect(validateSnapshot({ ...good, name: '' })).toBeNull();
  });
});

// ---- steps ----

describe('steps checklist', () => {
  let store: MemStorage;
  let original: Storage | undefined;
  beforeEach(() => {
    store = new MemStorage();
    original = installStorage(store);
  });
  afterEach(() => installStorage(original));

  it('has the Day 1 and Day 2 steps from the plan', () => {
    expect(STEP_GROUPS.map((g) => [g.title, g.steps.length])).toEqual([['Day 1', 6], ['Day 2', 5]]);
    expect(STEP_GROUPS[0]?.steps.map((s) => s.text)).toEqual([
      'Load data and read the class counts',
      'Train your first model',
      "Compare with 'always say pneumonia'",
      'Evaluate on the adult X-rays',
      'Break it three ways (sliders, upload, webcam on a cup)',
      'Log each failure with a tag',
    ]);
    expect(STEP_GROUPS[1]?.steps.map((s) => s.text)).toEqual([
      'Run the stamp experiment and record the flip rate',
      'Test on the adult set and note the drop',
      'Apply two fixes (balance, adult mix, augmentation, refer threshold)',
      'Re-measure all three sets and save a snapshot',
      'Post your worst-case score',
    ]);
    expect(new Set(ALL_STEP_IDS).size).toBe(ALL_STEP_IDS.length);
  });

  it('toggles, persists under vbw:v1:steps and reloads only known ids', () => {
    let done: string[] = [];
    done = toggleStepId(done, 'd1-train');
    done = toggleStepId(done, 'd2-post');
    expect(done).toEqual(['d1-train', 'd2-post']);
    expect(toggleStepId(done, 'd1-train')).toEqual(['d2-post']);
    expect(toggleStepId(done, 'bogus')).toEqual(done);
    expect(STEPS_KEY).toBe('vbw:v1:steps');
    saveSteps(done);
    expect(loadSteps()).toEqual(done);
    store.setItem(STEPS_KEY, JSON.stringify(['d1-data', 'nope', 'd1-data', 5]));
    expect(loadSteps()).toEqual(['d1-data']);
    saveSteps([]);
    expect(store.getItem(STEPS_KEY)).toBeNull();
    expect(sanitizeSteps('x')).toEqual([]);
  });
});
