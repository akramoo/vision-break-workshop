// Pure description of WHICH images a training run uses, derived from the manifest and the training settings.
import { classWeights } from '../ml/metrics';
import { idsBySplit, labelValue, selectEntries, type ManifestEntry } from './manifest';

export type TrainMode = 'realistic' | 'balanced';
export type ValSet = 'val' | 'val_official';

export interface TrainSettings {
  /** realistic: train on the natural 1:3 mix. balanced: weight classes by N / (2 * N_c). */
  mode: TrainMode;
  /** Day-2 fix: add the adult training pool (+200 images). */
  adultMix: boolean;
  /** Day-2 fix: add 2 augmented views of every training image. */
  augmentation: boolean;
  epochs: number;
  lr: number;
  seed: number;
  valSet: ValSet;
}

export const DEFAULT_TRAIN_SETTINGS: Readonly<TrainSettings> = {
  mode: 'realistic',
  adultMix: false,
  augmentation: false,
  epochs: 20,
  lr: 1e-3,
  seed: 42,
  valSet: 'val',
};

export const AUG_VIEWS: readonly number[] = [0, 1];

export interface TrainingPlan {
  /** Training images, manifest order (Kermany train, then adult pool when mixed in). */
  baseIds: string[];
  baseLabels: number[];
  valIds: string[];
  valLabels: number[];
  augmentation: boolean;
  /** Totals over the ACTUAL training rows (base + augmented views). */
  nRows: number;
  nNormal: number;
  nPneumonia: number;
  /** Class weights that balanced mode would use for these rows. */
  balancedWeights: { 0: number; 1: number };
  /** Class weights actually used: null in realistic mode. */
  weights: { 0: number; 1: number } | null;
}

export function buildTrainingPlan(entries: readonly ManifestEntry[], s: TrainSettings): TrainingPlan {
  const base = [
    ...selectEntries(entries, { source: 'kermany', split: 'train' }),
    ...(s.adultMix ? selectEntries(entries, { source: 'nih', split: 'train_adult' }) : []),
  ];
  const baseIds = base.map((e) => e.id);
  const baseLabels: number[] = base.map(labelValue);
  const valIds = idsBySplit(entries, 'kermany', s.valSet);
  const valById = new Map(entries.map((e) => [e.id, e]));
  const valLabels = valIds.map((id) => labelValue(valById.get(id) as ManifestEntry));
  const mult = s.augmentation ? 1 + AUG_VIEWS.length : 1;
  const basePneumonia = baseLabels.reduce((a, b) => a + b, 0);
  const nPneumonia = basePneumonia * mult;
  const nNormal = (baseLabels.length - basePneumonia) * mult;
  const balancedWeights = classWeights(nNormal, nPneumonia);
  return {
    baseIds,
    baseLabels,
    valIds,
    valLabels,
    augmentation: s.augmentation,
    nRows: nNormal + nPneumonia,
    nNormal,
    nPneumonia,
    balancedWeights,
    weights: s.mode === 'balanced' ? balancedWeights : null,
  };
}

/** Labels of all training rows: the base rows, then (if augmenting) each base label once per view, id-major. */
export function trainingRowLabels(plan: TrainingPlan): number[] {
  const out = [...plan.baseLabels];
  if (plan.augmentation) for (const l of plan.baseLabels) for (let v = 0; v < AUG_VIEWS.length; v++) out.push(l);
  return out;
}
