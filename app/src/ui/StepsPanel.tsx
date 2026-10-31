import { useSession } from '../state/session';
import { STEP_GROUPS } from '../state/steps';

/** Collapsible "Workshop steps" checklist (Day 1 / Day 2). Ticking is manual; the ticks survive a reload. */
export function StepsPanel() {
  const { state, dispatch } = useSession();
  const done = new Set(state.stepsDone);
  const total = STEP_GROUPS.reduce((n, g) => n + g.steps.length, 0);
  return (
    <details
      className="steps-panel no-print"
      open={state.stepsOpen}
      onToggle={(e) => {
        const open = (e.currentTarget as HTMLDetailsElement).open;
        if (open !== state.stepsOpen) dispatch({ type: 'setStepsOpen', open });
      }}
      data-testid="steps-panel"
    >
      <summary data-testid="steps-summary">
        Workshop steps{' '}
        <span className="muted" data-testid="steps-progress">
          ({done.size} of {total} done)
        </span>
      </summary>
      <div className="steps-panel__body">
        {STEP_GROUPS.map((g) => (
          <fieldset key={g.day} className="steps-group" data-testid={`steps-day-${g.day}`}>
            <legend>{g.title}</legend>
            <ul>
              {g.steps.map((s) => (
                <li key={s.id}>
                  <label>
                    <input
                      type="checkbox"
                      checked={done.has(s.id)}
                      onChange={() => dispatch({ type: 'toggleStep', id: s.id })}
                      data-testid={`step-${s.id}`}
                    />{' '}
                    <span>{s.text}</span>
                  </label>
                </li>
              ))}
            </ul>
          </fieldset>
        ))}
      </div>
    </details>
  );
}
