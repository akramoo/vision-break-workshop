import { useEffect, useRef } from 'react';
import { resetSession, useSession } from '../state/session';

export function SettingsDrawer() {
  const { state, dispatch } = useSession();
  const closeRef = useRef<HTMLButtonElement>(null);
  const open = state.settingsOpen;

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dispatch({ type: 'setSettingsOpen', open: false });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, dispatch]);

  if (!open) return null;
  const close = () => dispatch({ type: 'setSettingsOpen', open: false });
  const backend = state.backend;

  return (
    <div className="drawer-backdrop" onClick={close}>
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        data-testid="settings-drawer"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="drawer__head">
          <h2 id="settings-title">Settings</h2>
          <button ref={closeRef} type="button" className="btn btn--ghost" onClick={close}>
            Close
          </button>
        </div>

        <label className="switch">
          <input
            type="checkbox"
            role="switch"
            checked={state.settings.usePrecomputed}
            onChange={(e) => dispatch({ type: 'setUsePrecomputed', value: e.target.checked })}
          />
          <span>
            <strong>Use precomputed features</strong>
            <small>
              On: skip running MobileNet on every image (fast). Off: extract live on this laptop (slower, same
              result). You can switch at any time: the next training or evaluation uses the new mode, and every
              result says which mode produced it.
            </small>
          </span>
        </label>

        <dl className="drawer__info">
          <dt>Backend</dt>
          <dd>{backend ? `${backend.backend}${backend.reason ? ` (${backend.reason})` : ''}` : 'detecting…'}</dd>
          <dt>Model SHA-256</dt>
          <dd className="mono">{state.modelHash?.combined ?? 'not available'}</dd>
        </dl>

        <p className="muted">
          Reset clears the trained models, uploads, camera, failure log, snapshots, checklist and saved settings. The MobileNet model stays
          loaded.
        </p>
        <button
          type="button"
          className="btn btn--danger"
          data-testid="reset-session"
          onClick={() => {
            if (
              window.confirm(
                'Reset the session? The trained models, failure log, snapshots, checklist and saved settings on this computer will be cleared.',
              )
            ) {
              void resetSession(dispatch);
            }
          }}
        >
          Reset session
        </button>
      </aside>
    </div>
  );
}
