// What the current model was trained with. Used by the failure log and the results card.
import type { TrainMode, TrainSettings } from '../data/trainingPlan';
import type { HeadInfo } from './headStore';

export interface ModelConfig {
  mode: TrainMode;
  adultMix: boolean;
  augmentation: boolean;
  /** Epochs actually completed when a head exists, else the epochs setting. */
  epochs: number;
  lr: number;
  seed: number;
}

/** The trained head's own record wins (it is what was really used); otherwise the Train settings. */
export function modelConfigOf(
  info: Pick<HeadInfo, 'mode' | 'adultMix' | 'augmentation' | 'completedEpochs' | 'lr' | 'seed'> | null,
  settings: Pick<TrainSettings, 'mode' | 'adultMix' | 'augmentation' | 'epochs' | 'lr' | 'seed'>,
): ModelConfig {
  if (info) {
    return {
      mode: info.mode,
      adultMix: info.adultMix,
      augmentation: info.augmentation,
      epochs: info.completedEpochs,
      lr: info.lr,
      seed: info.seed,
    };
  }
  return {
    mode: settings.mode,
    adultMix: settings.adultMix,
    augmentation: settings.augmentation,
    epochs: settings.epochs,
    lr: settings.lr,
    seed: settings.seed,
  };
}

/** One line for humans: "realistic, no adult mix, no augmentation, 20 epochs, lr 1.0e-3, seed 42". */
export function describeModelConfig(c: ModelConfig): string {
  return [
    c.mode,
    c.adultMix ? 'adult mix on' : 'adult mix off',
    c.augmentation ? 'augmentation on' : 'augmentation off',
    `${c.epochs} epochs`,
    `lr ${c.lr.toExponential(1)}`,
    `seed ${c.seed}`,
  ].join(', ');
}
