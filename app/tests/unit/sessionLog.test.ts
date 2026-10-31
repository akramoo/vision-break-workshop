import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MAX_LOG_ENTRIES, buildLogConfig, makeLogEntry, type LogDraft } from '../../src/state/failureLog';
import { initialState, reducer } from '../../src/state/session';
import { makeSnapshot, MAX_SNAPSHOTS } from '../../src/state/snapshots';
import { MemStorage, installStorage } from './storageStub';

const draft = (over: Partial<LogDraft> = {}): LogDraft => ({
  thumbnail: `data:image/jpeg;base64,${'A'.repeat(100)}`,
  setName: 'webcam',
  trueLabel: null,
  predictedLabel: 'pneumonia',
  confidence: 0.8,
  probPneumonia: 0.8,
  referred: false,
  config: buildLogConfig({ mode: 'realistic', adultMix: false, augmentation: false, epochs: 20, lr: 0.001, seed: 42 }, 0.5, true),
  tag: 'closed-world',
  hypothesis: '',
  ...over,
});

const perf = { n: 10, accuracy: 0.8, sensitivity: 0.8, specificity: 0.8, balancedAccuracy: 0.8, referred: 0, referredFraction: 0, coverage: 1, halfCreditScore: 0.8 };
const snapshot = (name: string) =>
  makeSnapshot({
    name,
    existing: [],
    config: { mode: 'realistic', adultMix: false, augmentation: false, epochs: 20, lr: 0.001, seed: 42, tau: 0.5, stampTrained: false, usePrecomputed: true },
    perSet: { kermany_test: perf, adult_test: perf, stress: perf },
  });

let original: Storage | undefined;
beforeEach(() => {
  original = installStorage(new MemStorage());
});
afterEach(() => installStorage(original));

describe('session reducer: failure log, snapshots, steps, reset', () => {
  it('adds, edits and deletes log entries', () => {
    let s = initialState();
    const e = makeLogEntry(draft());
    s = reducer(s, { type: 'logAdd', entry: e });
    expect(s.failureLog).toEqual([e]);
    s = reducer(s, { type: 'logAdd', entry: e }); // same id twice is ignored
    expect(s.failureLog).toHaveLength(1);
    s = reducer(s, { type: 'logUpdate', id: e.id, patch: { tag: 'shift', hypothesis: '  adults look different ' } });
    expect(s.failureLog[0]).toMatchObject({ tag: 'shift', hypothesis: 'adults look different' });
    s = reducer(s, { type: 'logUpdate', id: e.id, patch: { tag: 'quality' } });
    expect(s.failureLog[0]).toMatchObject({ tag: 'quality', hypothesis: 'adults look different' });
    s = reducer(s, { type: 'logDelete', id: e.id });
    expect(s.failureLog).toEqual([]);
  });

  it('refuses entries beyond the cap', () => {
    let s = initialState();
    for (let i = 0; i < MAX_LOG_ENTRIES + 5; i++) s = reducer(s, { type: 'logAdd', entry: makeLogEntry(draft(), i) });
    expect(s.failureLog).toHaveLength(MAX_LOG_ENTRIES);
  });

  it('keeps at most 8 snapshots and deletes by id', () => {
    let s = initialState();
    for (let i = 0; i < MAX_SNAPSHOTS + 3; i++) s = reducer(s, { type: 'snapshotAdd', snapshot: snapshot(`s${i}`) });
    expect(s.snapshots).toHaveLength(8);
    const first = s.snapshots[0];
    s = reducer(s, { type: 'snapshotDelete', id: first?.id ?? '' });
    expect(s.snapshots).toHaveLength(7);
    expect(s.snapshots.some((x) => x.id === first?.id)).toBe(false);
  });

  it('toggles checklist steps and opens/closes the panel', () => {
    let s = initialState();
    s = reducer(s, { type: 'toggleStep', id: 'd1-train' });
    expect(s.stepsDone).toEqual(['d1-train']);
    s = reducer(s, { type: 'toggleStep', id: 'd1-train' });
    expect(s.stepsDone).toEqual([]);
    s = reducer(s, { type: 'setStepsOpen', open: true });
    expect(s.stepsOpen).toBe(true);
  });

  it('resetSession empties log, snapshots, steps, uploads, head and goes back to tau 0.5', () => {
    let s = initialState();
    s = reducer(s, { type: 'logAdd', entry: makeLogEntry(draft()) });
    s = reducer(s, { type: 'snapshotAdd', snapshot: snapshot('Before fix') });
    s = reducer(s, { type: 'toggleStep', id: 'd2-post' });
    s = reducer(s, { type: 'setTau', tau: 0.9 });
    s = reducer(s, { type: 'resetSession' });
    expect(s.failureLog).toEqual([]);
    expect(s.snapshots).toEqual([]);
    expect(s.stepsDone).toEqual([]);
    expect(s.abstainTau).toBe(0.5);
    expect(s.head).toBeNull();
    expect(s.evalResults).toEqual({});
    expect(s.settings.usePrecomputed).toBe(true);
  });

  it('initialState loads the persisted log, snapshots and steps', () => {
    let s = initialState();
    s = reducer(s, { type: 'logAdd', entry: makeLogEntry(draft()) });
    s = reducer(s, { type: 'snapshotAdd', snapshot: snapshot('Before fix') });
    s = reducer(s, { type: 'toggleStep', id: 'd1-data' });
    // the provider's effects persist these; do the same by hand
    const g = globalThis as unknown as { localStorage: Storage };
    g.localStorage.setItem('vbw:v1:log', JSON.stringify(s.failureLog));
    g.localStorage.setItem('vbw:v1:snapshots', JSON.stringify(s.snapshots));
    g.localStorage.setItem('vbw:v1:steps', JSON.stringify(s.stepsDone));
    const again = initialState();
    expect(again.failureLog).toEqual(s.failureLog);
    expect(again.snapshots).toEqual(s.snapshots);
    expect(again.stepsDone).toEqual(['d1-data']);
  });
});
