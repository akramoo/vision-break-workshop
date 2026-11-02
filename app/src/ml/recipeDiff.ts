// Is the recipe on the Data tab still the one the trained model was made with?
import type { TrainSettings } from '../data/trainingPlan';
import type { HeadInfo } from './headStore';

export type RecipeField = 'mode' | 'adultMix' | 'augmentation' | 'epochs' | 'lr' | 'seed' | 'valSet';

/** Plain names for the fields, in the order they are listed to the student. */
export const RECIPE_FIELD_LABELS: Readonly<Record<RecipeField, string>> = {
  mode: 'class balance',
  adultMix: 'adult X-rays',
  augmentation: 'augmentation',
  epochs: 'epochs',
  lr: 'learning rate',
  seed: 'seed',
  valSet: 'validation set',
};

const FIELDS = Object.keys(RECIPE_FIELD_LABELS) as RecipeField[];

/** What a trained head remembers about its recipe. Old or hand-made records may lack fields; those are skipped. */
export type RecipeRecord = Partial<Pick<HeadInfo, RecipeField>>;

function sameLr(a: number, b: number): boolean {
  // Learning rates come from a log slider and a JSON round trip: compare with a relative tolerance, never with ===.
  return Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(a), Math.abs(b));
}

function differs(field: RecipeField, trained: unknown, current: unknown): boolean {
  if (trained === undefined || trained === null) return false; // not recorded, so nothing to compare
  if (field === 'lr') {
    return typeof trained === 'number' && typeof current === 'number' && Number.isFinite(trained) && !sameLr(trained, current);
  }
  return trained !== current;
}

/**
 * The recipe fields where the settings now on the Data/Train tabs differ from what the model was trained with.
 * Empty when there is no model, or when everything that was recorded still matches.
 * The epochs compared are the ones that were asked for (a run stopped early still counts as trained with that setting).
 */
export function recipeDiffers(info: RecipeRecord | null, settings: Readonly<TrainSettings>): RecipeField[] {
  if (!info) return [];
  return FIELDS.filter((f) => differs(f, info[f], settings[f]));
}
