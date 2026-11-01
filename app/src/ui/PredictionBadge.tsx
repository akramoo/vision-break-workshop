import { confidenceOf, decide } from '../ml/metrics';
import { floorPercent } from './format';

interface Props {
  /** P(pneumonia). */
  p: number;
  tau: number;
  /** Name of the negative class ("normal" or "no finding"). */
  negativeName?: string;
  testId?: string;
}

/** The model's answer for one image: class + confidence bar, or "Refer to a doctor" below the threshold. */
export function PredictionBadge({ p, tau, negativeName = 'normal', testId }: Props) {
  const d = decide(p, tau);
  const conf = confidenceOf(p);
  const refer = d === 'refer';
  const name = refer ? 'Refer to a doctor' : d === 'pneumonia' ? 'pneumonia' : negativeName;
  return (
    <span className={`pred-badge pred-badge--${refer ? 'refer' : d === 'pneumonia' ? 'pneumonia' : 'normal'}`} {...(testId ? { 'data-testid': testId } : {})} data-decision={d}>
      <strong>{name}</strong>
      <span className="pred-badge__conf">
        {floorPercent(conf)} sure
        <span className="confbar" aria-hidden="true">
          <span className="confbar__fill" style={{ width: `${Math.floor(conf * 100 + 1e-9)}%` }} />
        </span>
      </span>
    </span>
  );
}
