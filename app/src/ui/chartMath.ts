// Small pure helpers for the hand-rolled SVG charts.

/** "Nice" tick values covering [min, max]; the first/last tick bound the data. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  let lo = min;
  let hi = max;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  if (hi <= lo) hi = lo + 1;
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const start = Math.floor(lo / step + 1e-9) * step;
  const end = Math.ceil(hi / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let v = start, i = 0; v <= end + step * 1e-6 && i < 100; v = start + ++i * step) {
    ticks.push(Number(v.toPrecision(12)));
  }
  return ticks;
}

/** Linear map of v from [d0, d1] to [r0, r1]; returns r0 when the domain is empty. */
export function scaleLinear(v: number, d0: number, d1: number, r0: number, r1: number): number {
  return d1 === d0 ? r0 : r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

/** SVG path "M x y L x y ..." for a series; points are (index + 1, value) skipping non-finite values. */
export function linePath(values: readonly number[], x: (i: number) => number, y: (v: number) => number): string {
  let d = '';
  values.forEach((v, i) => {
    if (!Number.isFinite(v)) return;
    d += `${d === '' ? 'M' : 'L'}${x(i + 1).toFixed(1)} ${y(v).toFixed(1)} `;
  });
  return d.trim();
}
