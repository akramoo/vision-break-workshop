export { formatPercent } from '../ml/metrics';

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

/** 3400 -> "3.4 s"; 125000 -> "2 min 5 s". */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '-';
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
}
