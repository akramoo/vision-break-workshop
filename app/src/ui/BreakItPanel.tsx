import { useSession, type BreakTab } from '../state/session';
import { ConfidenceThreshold } from './ConfidenceThreshold';
import { ModelStrip } from './ModelStrip';
import { PerturbPanel } from './PerturbPanel';
import { StampPanel } from './StampPanel';
import { UploadResults, UploadsManager } from './UploadsPanel';
import { WebcamPanel } from './WebcamPanel';

const SUBTABS: { id: BreakTab; label: string }[] = [
  { id: 'perturb', label: 'Perturb' },
  { id: 'stamp', label: 'Stamp' },
  { id: 'images', label: 'Your images & webcam' },
];

/** "Break it": three experiments that expose what the model really learned, plus the shared Refer-to-a-doctor threshold. */
export function BreakItPanel() {
  const { state, dispatch } = useSession();
  const active = state.breakTab;

  return (
    <>
      <h2>Break it</h2>
      <ModelStrip />
      <p className="lead">
        A model that scores well on the exam can still be fragile or wrong for the wrong reasons. Try to break yours, and log what you find.
      </p>
      <div role="group" aria-label="Break it experiments" className="subtabs">
        {SUBTABS.map((t) => (
          <button
            key={t.id}
            type="button"
            id={`subtab-${t.id}`}
            aria-pressed={active === t.id}
            aria-controls="break-subpanel"
            className={`subtab${active === t.id ? ' subtab--active' : ''}`}
            data-testid={`break-subtab-${t.id}`}
            onClick={() => dispatch({ type: 'setBreakTab', tab: t.id })}
          >
            {t.label}
          </button>
        ))}
      </div>
      <ConfidenceThreshold />
      <div id="break-subpanel" data-testid={`break-panel-${active}`}>
        {active === 'perturb' ? (
          <PerturbPanel />
        ) : active === 'stamp' ? (
          <StampPanel />
        ) : (
          <>
            <UploadsManager />
            <UploadResults />
            <WebcamPanel />
          </>
        )}
      </div>
    </>
  );
}
