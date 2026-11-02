import { recipeDiffers, RECIPE_FIELD_LABELS } from '../ml/recipeDiff';
import { useSession } from '../state/session';
import { modelStripText, NO_MODEL_TEXT, RECIPE_CHANGED_TEXT } from './modelStripText';

/**
 * "Which model produced the numbers on this screen?" One compact strip shared by Train, Evaluate, Break it and Results.
 * With a model it describes the recipe it was trained with, and warns when the Data tab's recipe has moved on since.
 */
/** `emptyText={false}` hides the "No model yet" line when there is no head (the Train tab does not want it). */
export function ModelStrip({ emptyText = true }: { emptyText?: boolean }) {
  const { state, dispatch } = useSession();
  const { head, headInfo, train, activeTab } = state;

  if (!head || !headInfo) {
    if (!emptyText) return null;
    return (
      <div className="model-strip model-strip--empty" data-testid="model-strip">
        <p data-testid="model-strip-text">{NO_MODEL_TEXT}</p>
      </div>
    );
  }

  // The head's own record of its last epoch; heads saved before that field existed fall back to the stored curve.
  const last = train.history[train.history.length - 1];
  const valAcc = headInfo.finalValAcc ?? last?.valAcc ?? null;
  const changed = recipeDiffers(headInfo, state.trainSettings);

  return (
    <div className="model-strip" data-testid="model-strip">
      <p data-testid="model-strip-text">{modelStripText(headInfo, valAcc)}</p>
      {changed.length > 0 ? (
        <div className="model-strip__warn" role="status" data-testid="recipe-warning">
          <span>
            {RECIPE_CHANGED_TEXT}{' '}
            <span className="muted" data-testid="recipe-warning-fields">
              (Changed: {changed.map((f) => RECIPE_FIELD_LABELS[f]).join(', ')}.)
            </span>
          </span>
          {activeTab !== 'train' ? (
            <button
              type="button"
              className="btn btn--small"
              data-testid="recipe-warning-go"
              onClick={() => dispatch({ type: 'setTab', tab: 'train' })}
            >
              Go to Train
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
