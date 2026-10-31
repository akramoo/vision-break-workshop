import { confusionTotal, formatPercent, ratio, type Confusion } from '../ml/metrics';

interface Props {
  confusion: Confusion;
  /** What the healthy class is called: "normal" (children) or "no finding" (adults). */
  negativeName: 'normal' | 'no finding';
  referred?: number;
  testId?: string;
}

/** 2x2 confusion matrix. Rows are the TRUE class, columns the PREDICTED class. */
export function ConfusionMatrix({ confusion: c, negativeName, referred = 0, testId = 'confusion-matrix' }: Props) {
  const rowNeg = c.tn + c.fp;
  const rowPos = c.fn + c.tp;
  const cell = (count: number, rowTotal: number, good: boolean, tag: string, id: string) => (
    <td className={`cm__cell ${good ? 'cm__cell--good' : 'cm__cell--bad'}`} data-testid={`cm-${id}`}>
      <span className="cm__count">{count}</span>
      <span className="cm__pct">{formatPercent(ratio(count, rowTotal))} of row</span>
      <span className="cm__tag">{tag}</span>
    </td>
  );
  return (
    <div className="cm" data-testid={testId}>
      <table className="cm__table">
        <caption className="cm__caption">
          Confusion matrix: {confusionTotal(c)} X-rays{referred > 0 ? ` (${referred} referred, not counted)` : ''}
        </caption>
        <thead>
          <tr>
            <td className="cm__corner" />
            <th scope="col" colSpan={2} className="cm__colgroup">
              Model said
            </th>
          </tr>
          <tr>
            <td className="cm__corner">True class</td>
            <th scope="col">{negativeName}</th>
            <th scope="col">pneumonia</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">
              {negativeName} <small>(n = {rowNeg})</small>
            </th>
            {cell(c.tn, rowNeg, true, 'correctly cleared', 'tn')}
            {cell(c.fp, rowNeg, false, 'false alarm', 'fp')}
          </tr>
          <tr>
            <th scope="row">
              pneumonia <small>(n = {rowPos})</small>
            </th>
            {cell(c.fn, rowPos, false, 'missed', 'fn')}
            {cell(c.tp, rowPos, true, 'caught', 'tp')}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
