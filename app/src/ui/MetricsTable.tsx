import { formatPercent, type Scores } from '../ml/metrics';

export interface MetricRow {
  key: string;
  name: string;
  scores: Scores;
  /** Referral figures; only shown when the table has showAbstain. */
  referred?: { n: number; fraction: number | null; coverage: number | null };
  emphasis?: boolean;
}

interface Props {
  rows: readonly MetricRow[];
  showAbstain: boolean;
  testId?: string;
  /** Who the X-rays belong to: the Kermany sets are children, the adult set (and uploads) are just "patients". */
  who?: 'children' | 'patients';
}

/** Scores side by side. Percentages are "n/a" when the denominator is empty. */
export function MetricsTable({ rows, showAbstain, testId = 'metrics-table', who = 'children' }: Props) {
  return (
    <table className="metrics" data-testid={testId}>
      <thead>
        <tr>
          <th scope="col">Predictor</th>
          <th scope="col">Accuracy</th>
          <th scope="col">
            Sensitivity <small>sick {who} caught</small>
          </th>
          <th scope="col">
            Specificity <small>{who === 'children' ? 'healthy children' : 'patients with no finding'} cleared</small>
          </th>
          <th scope="col">Balanced accuracy</th>
          {showAbstain ? (
            <>
              <th scope="col">Referred</th>
              <th scope="col">Coverage</th>
            </>
          ) : null}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr
            key={r.key}
            className={r.emphasis ? 'metrics__row--model' : 'metrics__row--baseline'}
            data-testid={`metrics-row-${r.key}`}
          >
            <th scope="row">{r.name}</th>
            <td data-testid={`${r.key}-accuracy`}>{formatPercent(r.scores.accuracy)}</td>
            <td data-testid={`${r.key}-sensitivity`}>{formatPercent(r.scores.sensitivity)}</td>
            <td data-testid={`${r.key}-specificity`}>{formatPercent(r.scores.specificity)}</td>
            <td data-testid={`${r.key}-balanced`}>{formatPercent(r.scores.balancedAccuracy)}</td>
            {showAbstain ? (
              <>
                <td data-testid={`${r.key}-referred`}>
                  {r.referred ? `${r.referred.n} (${formatPercent(r.referred.fraction)})` : '-'}
                </td>
                <td data-testid={`${r.key}-coverage`}>{r.referred ? formatPercent(r.referred.coverage) : '-'}</td>
              </>
            ) : null}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
