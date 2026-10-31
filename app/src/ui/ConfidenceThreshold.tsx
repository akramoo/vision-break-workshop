import { MAX_TAU, MIN_TAU } from '../ml/metrics';
import { useSession } from '../state/session';

/** The "Refer to a doctor" threshold. One shared value (state.abstainTau), shown on Evaluate and in Break it. */
export function ConfidenceThreshold() {
  const { state, dispatch } = useSession();
  const tau = state.abstainTau;
  const pct = Math.round(tau * 100);
  return (
    <div className="card tau" data-testid="tau-control">
      <label htmlFor="tau-slider" className="tau__label">
        Refer to a doctor when the model is less than <strong data-testid="tau-value">{pct}%</strong> sure
      </label>
      <div className="field__row">
        <input
          id="tau-slider"
          data-testid="tau-slider"
          type="range"
          min={MIN_TAU}
          max={MAX_TAU}
          step={0.01}
          value={tau}
          onChange={(e) => dispatch({ type: 'setTau', tau: Math.round(Number(e.target.value) * 100) / 100 })}
        />
        <span className="muted">{tau <= MIN_TAU ? '50% = the model always answers (never refers)' : `refers when confidence < ${pct}%`}</span>
      </div>
      <p className="muted tau__explain" data-testid="tau-explain">
        Metrics are computed on cases the model answered; referred cases get half credit in the worst-case score.
      </p>
    </div>
  );
}
