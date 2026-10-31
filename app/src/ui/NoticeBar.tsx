import { useSession } from '../state/session';

/** Non-fatal messages (for example "precomputed features unavailable, using live extraction"). */
export function NoticeBar() {
  const { state, dispatch } = useSession();
  const n = state.notice;
  if (!n) return null;
  return (
    <div className={`notice notice--${n.level}`} role="status" data-testid="notice">
      <span>{n.message}</span>
      <button type="button" className="btn btn--small" onClick={() => dispatch({ type: 'dismissNotice' })}>
        Dismiss
      </button>
    </div>
  );
}
