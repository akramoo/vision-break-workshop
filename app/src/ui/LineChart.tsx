import { useId } from 'react';
import { linePath, niceTicks, scaleLinear } from './chartMath';

export interface ChartSeries {
  name: string;
  values: readonly number[];
  color: string;
  dashed?: boolean;
}

interface LineChartProps {
  title: string;
  series: readonly ChartSeries[];
  /** Number of epochs on the x axis (the axis is fixed so the line grows into the chart). */
  xMax: number;
  yMin?: number | 'auto';
  yMax?: number | 'auto';
  yLabel: string;
  xLabel?: string;
  formatY?: (v: number) => string;
  testId?: string;
}

const W = 560;
const H = 300;
const M = { left: 64, right: 16, top: 16, bottom: 48 };

/** Hand-rolled SVG line chart: axes, ticks, legend, accessible title and summary. */
export function LineChart({
  title,
  series,
  xMax,
  yMin = 'auto',
  yMax = 'auto',
  yLabel,
  xLabel = 'Epoch',
  formatY,
  testId,
}: LineChartProps) {
  const uid = useId();
  const all = series.flatMap((s) => s.values.filter(Number.isFinite));
  const dataMin = all.length ? Math.min(...all) : 0;
  const dataMax = all.length ? Math.max(...all) : 1;
  const lo = yMin === 'auto' ? dataMin : yMin;
  const hi = yMax === 'auto' ? dataMax : yMax;
  const yTicks = niceTicks(lo, hi, 5);
  const y0 = yTicks[0] as number;
  const y1 = yTicks[yTicks.length - 1] as number;
  const xEnd = Math.max(1, xMax);
  const xTicks = niceTicks(0, xEnd, 5).filter((t) => t >= 0 && t <= xEnd && Number.isInteger(t));
  const px = (e: number) => scaleLinear(e, 0, xEnd, M.left, W - M.right);
  const py = (v: number) => scaleLinear(v, y0, y1, H - M.bottom, M.top);
  const fmt = formatY ?? ((v: number) => String(Number(v.toPrecision(3))));

  const summary = series
    .map((s) => {
      const last = s.values[s.values.length - 1];
      return last === undefined ? `${s.name}: no data yet` : `${s.name}: ${fmt(last)} after epoch ${s.values.length}`;
    })
    .join('; ');

  return (
    <figure className="chart" data-testid={testId}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-labelledby={`${uid}-t ${uid}-d`}
        className="chart__svg"
        preserveAspectRatio="xMidYMid meet"
      >
        <title id={`${uid}-t`}>{title}</title>
        <desc id={`${uid}-d`}>{summary}</desc>
        <rect x="0" y="0" width={W} height={H} fill="#ffffff" />
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={M.left} x2={W - M.right} y1={py(t)} y2={py(t)} stroke="#d1d5db" strokeWidth="1" />
            <text x={M.left - 8} y={py(t) + 5} textAnchor="end" className="chart__tick">
              {fmt(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={px(t)} x2={px(t)} y1={H - M.bottom} y2={H - M.bottom + 5} stroke="#111827" strokeWidth="1.5" />
            <text x={px(t)} y={H - M.bottom + 22} textAnchor="middle" className="chart__tick">
              {t}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={M.left} y1={M.top} y2={H - M.bottom} stroke="#111827" strokeWidth="2" />
        <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke="#111827" strokeWidth="2" />
        <text x={(M.left + W - M.right) / 2} y={H - 8} textAnchor="middle" className="chart__label">
          {xLabel}
        </text>
        <text
          transform={`translate(16 ${(M.top + H - M.bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
          className="chart__label"
        >
          {yLabel}
        </text>
        {series.map((s) => (
          <g key={s.name}>
            <path
              d={linePath(s.values, px, py)}
              fill="none"
              stroke={s.color}
              strokeWidth="3"
              strokeLinejoin="round"
              {...(s.dashed ? { strokeDasharray: '8 5' } : {})}
            />
            {s.values.length > 0 && s.values.length < 3
              ? s.values.map((v, i) => <circle key={i} cx={px(i + 1)} cy={py(v)} r="4" fill={s.color} />)
              : null}
          </g>
        ))}
      </svg>
      <figcaption className="chart__legend">
        <strong className="chart__title">{title}</strong>
        {series.map((s) => (
          <span key={s.name} className="chart__key">
            <svg width="34" height="10" aria-hidden="true">
              <line
                x1="0"
                x2="34"
                y1="5"
                y2="5"
                stroke={s.color}
                strokeWidth="3"
                {...(s.dashed ? { strokeDasharray: '8 5' } : {})}
              />
            </svg>
            {s.name}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
