import { describe, expect, it } from 'vitest';
import { DEFAULT_TRAIN_SETTINGS, type TrainSettings } from '../../src/data/trainingPlan';
import type { HeadInfo } from '../../src/ml/headStore';
import { recipeDiffers, type RecipeRecord } from '../../src/ml/recipeDiff';
import { lrFromSlider } from '../../src/ui/format';
import { modelStripText, NO_MODEL_TEXT, RECIPE_CHANGED_TEXT } from '../../src/ui/modelStripText';
import { evalSetNote } from '../../src/ui/evalNote';

const settings = (over: Partial<TrainSettings> = {}): TrainSettings => ({ ...DEFAULT_TRAIN_SETTINGS, ...over });

const info = (over: Partial<HeadInfo> = {}): HeadInfo => ({
  trainedAt: Date.UTC(2026, 10, 2, 14, 32),
  seed: 42,
  epochs: 20,
  completedEpochs: 20,
  lr: 1e-3,
  mode: 'realistic',
  adultMix: false,
  augmentation: false,
  valSet: 'val',
  nRows: 1020,
  nNormal: 255,
  nPneumonia: 765,
  classWeights: null,
  usedPrecomputed: true,
  durationMs: 3400,
  restored: false,
  ...over,
});

describe('recipeDiffers', () => {
  it('is empty without a model or when nothing changed', () => {
    expect(recipeDiffers(null, settings())).toEqual([]);
    expect(recipeDiffers(info(), settings())).toEqual([]);
  });

  it('lists every field that changed, in a fixed order', () => {
    const s = settings({ mode: 'balanced', adultMix: true, augmentation: true, epochs: 5, lr: 0.01, seed: 7, valSet: 'val_official' });
    expect(recipeDiffers(info(), s)).toEqual(['mode', 'adultMix', 'augmentation', 'epochs', 'lr', 'seed', 'valSet']);
    expect(recipeDiffers(info(), settings({ mode: 'balanced' }))).toEqual(['mode']);
    expect(recipeDiffers(info(), settings({ seed: 43, epochs: 21 }))).toEqual(['epochs', 'seed']);
  });

  it('compares the epochs that were asked for, not the ones completed', () => {
    expect(recipeDiffers(info({ completedEpochs: 6 }), settings())).toEqual([]);
    expect(recipeDiffers(info({ epochs: 6, completedEpochs: 6 }), settings())).toEqual(['epochs']);
  });

  it('treats equal learning rates as equal through float noise, a JSON round trip and the slider', () => {
    expect(recipeDiffers(info({ lr: 0.1 + 0.2 - 0.2 }), settings({ lr: 0.1 }))).toEqual([]);
    const stored = JSON.parse(JSON.stringify(0.00031)) as number;
    expect(recipeDiffers(info({ lr: stored }), settings({ lr: lrFromSlider(Math.log10(0.00031)) }))).toEqual([]);
    expect(recipeDiffers(info({ lr: 1e-3 }), settings({ lr: 1.1e-3 }))).toEqual(['lr']);
    expect(recipeDiffers(info({ lr: 1e-3 }), settings({ lr: 1.0000001e-3 }))).toEqual(['lr']);
  });

  it('works for restored heads and skips fields a stored record does not have', () => {
    expect(recipeDiffers(info({ restored: true }), settings())).toEqual([]);
    expect(recipeDiffers(info({ restored: true }), settings({ augmentation: true }))).toEqual(['augmentation']);
    const old: RecipeRecord = { seed: 42, mode: 'realistic' };
    expect(recipeDiffers(old, settings())).toEqual([]);
    expect(recipeDiffers(old, settings({ seed: 1, adultMix: true }))).toEqual(['seed']);
    expect(recipeDiffers({}, settings({ mode: 'balanced' }))).toEqual([]);
    expect(recipeDiffers({ lr: Number.NaN }, settings())).toEqual([]);
  });
});

describe('modelStripText', () => {
  const fixedTime = () => '14:32';

  it('describes the default model in one plain line', () => {
    const t = modelStripText(info(), 0.989, fixedTime);
    expect(t).toBe(
      'Model in use: Realistic classes, adult X-rays off, augmentation off, seed 42, 20 epochs, learning rate 1.0e-3. ' +
        'Trained 14:32 (took 3.4 s). Validation accuracy 98.9% on our 180-image validation set.',
    );
  });

  it('reflects the recipe, the official validation set and a restored head', () => {
    const t = modelStripText(
      info({ mode: 'balanced', adultMix: true, augmentation: true, seed: 7, completedEpochs: 8, lr: 0.01, valSet: 'val_official', restored: true, durationMs: 12_400 }),
      0.9375,
      fixedTime,
    );
    expect(t).toContain('Balanced classes, adult X-rays on, augmentation on, seed 7, 8 epochs, learning rate 1.0e-2.');
    expect(t).toContain('(took 12 s)');
    expect(t).toContain('Validation accuracy 93.8% on the official 16-image set.');
    expect(t.endsWith('Restored from an earlier session.')).toBe(true);
  });

  it('leaves the accuracy out when it is unknown, and does not claim a restore otherwise', () => {
    const t = modelStripText(info(), null, fixedTime);
    expect(t).not.toContain('Validation accuracy');
    expect(t).not.toContain('Restored');
  });

  it('keeps the fixed sentences the e2e test and the guide rely on', () => {
    expect(NO_MODEL_TEXT).toBe('No model yet: train one on the Train tab.');
    expect(RECIPE_CHANGED_TEXT).toBe('The recipe on the Data tab has changed since this model was trained. Press Train again to apply it.');
  });
});

describe('evalSetNote', () => {
  it('names the set and its size and explains the difference to the Train tab', () => {
    const k = evalSetNote('kermany_test', 624);
    expect(k).toContain('Kermany test set (624 images the model never trained on)');
    expect(k).toContain('validation accuracy on the Train tab');
    expect(evalSetNote('adult_test', 200)).toContain('Adult test set (200 images');
    expect(evalSetNote('stress', 624)).toContain('degraded copies of the Kermany test images (624 images');
  });
});
