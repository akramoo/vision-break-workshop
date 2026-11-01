export { formatPercent } from '../ml/metrics';

/**
 * A confidence as a whole percent, rounded DOWN: 0.7995 shows "79%", never "80%", so a prediction that was referred at an
 * 80% threshold can not be shown as "80% sure". The tiny epsilon keeps 0.57 (= 56.999...) at 57.
 */
export function floorPercent(fraction: number): string {
  if (!Number.isFinite(fraction)) return 'n/a';
  return `${Math.floor(fraction * 100 + 1e-9)}%`;
}

/** Learning rate for display: 0.001 -> "1.0e-3". */
export function formatLr(lr: number): string {
  const [m, e] = lr.toExponential(1).split('e');
  return `${m}e${e}`;
}

/** Slider position (log10) -> learning rate with 2 significant digits. */
export function lrFromSlider(pos: number): number {
  return Number((10 ** pos).toPrecision(2));
}

export function sliderFromLr(lr: number): number {
  return Math.log10(lr);
}

/** 3400 -> "3.4 s" (one decimal below 10 s); 59400 -> "59 s"; 125000 -> "2 min 5 s"; 119600 -> "2 min 0 s". */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '-';
  const s = ms / 1000;
  if (s < 9.95) return `${s.toFixed(1)} s`;
  // Round the total first, then split, so 119.6 s reads "2 min 0 s" and never "1 min 60 s".
  const total = Math.round(s);
  if (total < 60) return `${total} s`;
  const m = Math.floor(total / 60);
  return `${m} min ${total - m * 60} s`;
}
