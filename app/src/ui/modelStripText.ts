// The one-line description of the model on screen, as plain text (kept pure so it can be unit tested).
import type { HeadInfo } from '../ml/headStore';
import { formatPercent } from '../ml/metrics';
import { formatDuration, formatLr } from './format';

export const NO_MODEL_TEXT = 'No model yet: train one on the Train tab.';
export const RECIPE_CHANGED_TEXT = 'The recipe on the Data tab has changed since this model was trained. Press Train again to apply it.';

export function clockTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** "Model in use: Realistic classes, adult X-rays off, augmentation off, seed 42, 20 epochs, learning rate 1.0e-3. ..." */
export function modelStripText(
  info: HeadInfo,
  valAcc: number | null,
  formatTime: (ms: number) => string = clockTime,
): string {
  const balance = info.mode === 'balanced' ? 'Balanced' : 'Realistic';
  const recipe = [
    `${balance} classes`,
    `adult X-rays ${info.adultMix ? 'on' : 'off'}`,
    `augmentation ${info.augmentation ? 'on' : 'off'}`,
    `seed ${info.seed}`,
    `${info.completedEpochs} epochs`,
    `learning rate ${formatLr(info.lr)}`,
  ].join(', ');
  const parts = [`Model in use: ${recipe}.`];
  parts.push(`Trained ${formatTime(info.trainedAt)} (took ${formatDuration(info.durationMs)}).`);
  if (valAcc !== null && Number.isFinite(valAcc)) {
    const set = info.valSet === 'val_official' ? 'the official 16-image set' : 'our 180-image validation set';
    parts.push(`Validation accuracy ${formatPercent(valAcc)} on ${set}.`);
  }
  if (info.restored) parts.push('Restored from an earlier session.');
  return parts.join(' ');
}
