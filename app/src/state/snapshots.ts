// F9: named snapshots of the results card ("Before fix", "After fix 1", ...). Pure + localStorage.
import type { EvalSetId } from '../data/evalSets';
import type { ModelConfig } from '../ml/modelConfig';
import { RESULT_SET_IDS, leaderboardOf, type Leaderboard, type SetScores } from '../ml/results';
import { SNAPSHOTS_KEY, readStored, safeParse, writeStored } from './persistence';

export const MAX_SNAPSHOTS = 8;
export const MAX_SNAPSHOT_NAME = 40;

export interface SnapshotConfig extends ModelConfig {
  tau: number;
  /** Was the model trained on stamped images? The main model never is. */
  stampTrained: boolean;
  /** Feature mode that was switched on when the snapshot was taken. */
  usePrecomputed: boolean;
}

export interface Snapshot {
  id: string;
  name: string;
  createdAt: number;
  config: SnapshotConfig;
  perSet: Record<EvalSetId, SetScores>;
  leaderboard: Leaderboard | null;
}

/** "Before fix", then "After fix 1", "After fix 2", ... (first number not already used). */
export function defaultSnapshotName(existing: readonly { name: string }[]): string {
  if (existing.length === 0) return 'Before fix';
  const used = new Set(existing.map((s) => s.name));
  for (let n = 1; ; n++) {
    const name = `After fix ${n}`;
    if (!used.has(name)) return name;
  }
}

let counter = 0;

export function makeSnapshot(args: {
  name: string;
  existing: readonly { name: string }[];
  config: SnapshotConfig;
  perSet: Record<EvalSetId, SetScores>;
  now?: number;
}): Snapshot {
  const now = args.now ?? Date.now();
  counter += 1;
  const name = args.name.trim().slice(0, MAX_SNAPSHOT_NAME) || defaultSnapshotName(args.existing);
  return {
    id: `snap-${now.toString(36)}-${counter.toString(36)}`,
    name,
    createdAt: now,
    config: args.config,
    perSet: args.perSet,
    leaderboard: leaderboardOf({
      kermany_test: args.perSet.kermany_test.halfCreditScore,
      adult_test: args.perSet.adult_test.halfCreditScore,
      stress: args.perSet.stress.halfCreditScore,
    }),
  };
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}
const numOrNull = (v: unknown): number | null | undefined =>
  v === null ? null : typeof v === 'number' && Number.isFinite(v) ? v : undefined;

function validSetScores(raw: unknown): SetScores | null {
  if (!isObject(raw)) return null;
  const n = raw['n'];
  const referred = raw['referred'];
  if (typeof n !== 'number' || !Number.isFinite(n) || typeof referred !== 'number' || !Number.isFinite(referred)) return null;
  const f = {
    accuracy: numOrNull(raw['accuracy']),
    sensitivity: numOrNull(raw['sensitivity']),
    specificity: numOrNull(raw['specificity']),
    balancedAccuracy: numOrNull(raw['balancedAccuracy']),
    referredFraction: numOrNull(raw['referredFraction']),
    coverage: numOrNull(raw['coverage']),
    halfCreditScore: numOrNull(raw['halfCreditScore']),
  };
  if (Object.values(f).some((v) => v === undefined)) return null;
  return { n, referred, ...(f as Record<keyof typeof f, number | null>) };
}

function validConfig(raw: unknown): SnapshotConfig | null {
  if (!isObject(raw)) return null;
  const { mode, adultMix, augmentation, epochs, lr, seed, tau, stampTrained, usePrecomputed } = raw;
  const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
  if (mode !== 'realistic' && mode !== 'balanced') return null;
  if (typeof adultMix !== 'boolean' || typeof augmentation !== 'boolean') return null;
  if (typeof stampTrained !== 'boolean' || typeof usePrecomputed !== 'boolean') return null;
  if (!num(epochs) || !num(lr) || !num(seed) || !num(tau)) return null;
  return { mode, adultMix, augmentation, epochs, lr, seed, tau, stampTrained, usePrecomputed };
}

/** One stored snapshot -> a clean copy, or null if malformed. The leaderboard is recomputed, never trusted. */
export function validateSnapshot(raw: unknown): Snapshot | null {
  if (!isObject(raw)) return null;
  const { id, name, createdAt } = raw;
  if (typeof id !== 'string' || id.length === 0 || id.length > 80) return null;
  if (typeof name !== 'string' || name.length === 0 || name.length > MAX_SNAPSHOT_NAME) return null;
  if (typeof createdAt !== 'number' || !Number.isFinite(createdAt)) return null;
  const config = validConfig(raw['config']);
  if (!config) return null;
  const ps = raw['perSet'];
  if (!isObject(ps)) return null;
  const perSet = {} as Record<EvalSetId, SetScores>;
  for (const setId of RESULT_SET_IDS) {
    const s = validSetScores(ps[setId]);
    if (!s) return null;
    perSet[setId] = s;
  }
  const leaderboard = leaderboardOf({
    kermany_test: perSet.kermany_test.halfCreditScore,
    adult_test: perSet.adult_test.halfCreditScore,
    stress: perSet.stress.halfCreditScore,
  });
  return { id, name, createdAt, config, perSet, leaderboard };
}

export function validateSnapshots(raw: unknown): Snapshot[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: Snapshot[] = [];
  for (const item of raw) {
    const s = validateSnapshot(item);
    if (!s || seen.has(s.id)) continue;
    seen.add(s.id);
    out.push(s);
    if (out.length >= MAX_SNAPSHOTS) break;
  }
  return out;
}

export function loadSnapshots(): Snapshot[] {
  return validateSnapshots(safeParse<unknown>(readStored(SNAPSHOTS_KEY), null));
}

/** Store the snapshots; none removes the key. False if the browser refused the write. */
export function saveSnapshots(snaps: readonly Snapshot[]): boolean {
  return writeStored(SNAPSHOTS_KEY, snaps.length === 0 ? null : JSON.stringify(snaps));
}
