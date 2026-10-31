import { useSession } from '../state/session';
import { DISCLAIMER } from './disclaimer';

const BACKEND_LABEL = { webgl: 'WebGL', wasm: 'WASM', cpu: 'CPU', none: 'No backend' } as const;

export function Header() {
  const { state, dispatch } = useSession();
  const { backend, model } = state;

  const backendLabel = backend ? BACKEND_LABEL[backend.backend] : 'Detecting…';
  const baseTip = backend
    ? (backend.reason ?? `Running on ${backendLabel}.`)
    : 'Detecting the fastest available compute backend.';
  const backendTip = backend?.softwareRenderer
    ? `${baseTip} Software rendering: WebGL is emulated on the CPU${backend.renderer ? ` (${backend.renderer})` : ''}, so this is slower than a real GPU.`
    : baseTip;

  const modelLabel =
    model.state === 'ready'
      ? 'Ready'
      : model.state === 'error'
        ? 'Model unavailable'
        : model.progress > 0 && model.progress < 1
          ? `Loading model… ${Math.round(model.progress * 100)}%`
          : 'Loading model…';

  return (
    <header className="app-header">
      <div className="app-header__top">
        <h1 className="app-title">Build a Vision Model, Then Break It</h1>
        <div className="app-header__status">
          <span
            className={`badge badge--backend${backend?.fellBack ? ' badge--warn' : ''}`}
            title={backendTip}
            data-testid="backend-badge"
          >
            {backendLabel}
          </span>
          <span className={`badge badge--model badge--${model.state}`} data-testid="model-status" role="status">
            {modelLabel}
          </span>
          <button
            type="button"
            className="btn btn--ghost"
            aria-haspopup="dialog"
            onClick={() => dispatch({ type: 'setSettingsOpen', open: true })}
          >
            Settings
          </button>
        </div>
      </div>
      <p className="disclaimer" data-testid="disclaimer-header">
        {DISCLAIMER}
      </p>
    </header>
  );
}
