import { useEffect, useMemo, useState } from 'react';
import { buildTrainingPlan, type ValSet } from '../data/trainingPlan';
import { formatPercent, rangeOf } from '../ml/metrics';
import { shownPercent } from '../ml/results';
import { cancelRun, REPEAT_RUNS, runRepeat, runTraining, type RunEnv } from '../ml/trainRunner';
import { useSession, type RepeatRun, type SessionState } from '../state/session';
import { formatDuration, formatLr, lrFromSlider, sliderFromLr } from './format';
import { LineChart } from './LineChart';
import { ModelStrip } from './ModelStrip';
import { ProgressBar } from './ProgressBar';

const TRAIN_COLOR = '#1d4ed8';
const VAL_COLOR = '#b45309';

function useElapsed(startedAt: number | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (startedAt === null) return;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(id);
  }, [startedAt]);
  return startedAt === null ? 0 : Math.max(0, now - startedAt);
}

function featureLine(p: { phase: string; done: number; total: number; fraction: number; etaMs: number | null } | null): string {
  if (!p) return 'Preparing the features...';
  if (p.phase === 'download') return `Loading precomputed features... ${Math.round(p.fraction * 100)}%`;
  const eta = p.etaMs === null ? '' : `, about ${formatDuration(p.etaMs)} left`;
  return `Running MobileNet on image ${p.done} of ${p.total}${eta}`;
}

export function TrainPanel() {
  const { state, dispatch } = useSession();
  const { trainSettings: ts, train, repeat, head, headInfo, manifest } = state;
  const status = train.status;
  const running = status.state === 'running';
  const repeating = repeat.state === 'running';
  const busy = running || repeating;
  const modelReady = state.model.state === 'ready';
  const manifestReady = manifest.state === 'ready';
  const canStart = modelReady && manifestReady && !busy;
  const elapsed = useElapsed(running ? status.startedAt : null);

  const env = (): RunEnv | null =>
    manifest.state === 'ready'
      ? {
          dispatch,
          entries: manifest.entries,
          extractor: state.extractor,
          usePrecomputed: state.settings.usePrecomputed,
          hasPreviousHead: head !== null,
        }
      : null;

  const plan = useMemo(
    () => buildTrainingPlan(manifest.state === 'ready' ? manifest.entries : [], ts),
    [manifest, ts],
  );

  let disabledReason = '';
  if (!modelReady) disabledReason = 'Waiting for the MobileNet model to load...';
  else if (!manifestReady) disabledReason = 'Waiting for the X-ray list to load...';

  const hist = train.history;
  const last = hist[hist.length - 1];
  const doneStatus = status.state === 'done' || status.state === 'cancelled' ? status : null;

  let statusText: string;
  if (status.state === 'running') {
    statusText =
      status.phase === 'features'
        ? featureLine(status.featureProgress)
        : `Training: epoch ${hist.length} of ${status.epochs}, ${formatDuration(elapsed)} elapsed`;
  } else if (doneStatus) {
    statusText =
      doneStatus.state === 'done'
        ? `Finished ${doneStatus.completedEpochs} epochs in ${formatDuration(doneStatus.durationMs)}.`
        : doneStatus.completedEpochs > 0
          ? `Stopped after epoch ${doneStatus.completedEpochs} of ${doneStatus.epochs}. ${
              doneStatus.keptPrevious
                ? 'Training cancelled; your previous model is kept.'
                : 'There was no earlier model, so the one trained so far was kept.'
            }`
          : 'Training was cancelled before the first epoch finished.';
    if (last) statusText += ` Validation accuracy ${formatPercent(last.valAcc)}.`;
  } else if (status.state === 'error') {
    statusText = `Training failed: ${status.message}`;
  } else if (head && headInfo?.restored) {
    statusText = 'A trained model from your last visit was restored. Train again to replace it.';
  } else {
    statusText = 'Not trained yet. Click Train.';
  }

  return (
    <>
      <h2>Train</h2>
      <ModelStrip emptyText={false} />
      <p className="lead">
        MobileNet turns every X-ray into 1,280 numbers and stays frozen. We train only a tiny final layer (logistic regression) that
        turns those numbers into <strong>P(pneumonia)</strong>. Watch the error (loss) fall and the accuracy rise.
      </p>

      <div className="card" data-testid="train-recipe">
        <p className="recipe">
          Recipe (change it in the Data tab): <strong>{ts.mode}</strong> class balance
          {ts.adultMix ? ', adult X-rays mixed in' : ''}
          {ts.augmentation ? ', augmentation on' : ''}. <span className="muted">{plan.nRows} training rows.</span>
        </p>
        <fieldset className="choice" disabled={busy}>
          <legend>Check progress on</legend>
          {(
            [
              ['val', 'Our validation set (180 images)'],
              ['val_official', 'Official validation set (16 images)'],
            ] as [ValSet, string][]
          ).map(([v, text]) => (
            <label key={v} className="choice__option">
              <input
                type="radio"
                name="val-set"
                value={v}
                checked={ts.valSet === v}
                onChange={() => dispatch({ type: 'setTrainSettings', patch: { valSet: v } })}
              />
              <span>{text}</span>
            </label>
          ))}
        </fieldset>

        <div className="actions">
          <button
            type="button"
            className="btn btn--primary"
            disabled={!canStart}
            data-testid="train-button"
            onClick={() => {
              const e = env();
              if (e) void runTraining(e, ts);
            }}
          >
            {head ? 'Train again' : 'Train'}
          </button>
          <button type="button" className="btn" disabled={!busy} data-testid="cancel-button" onClick={() => cancelRun()}>
            Cancel
          </button>
          {disabledReason ? <span className="muted">{disabledReason}</span> : null}
        </div>

        <p className="train-status" role="status" aria-live="polite" data-testid="train-status">
          {statusText}
        </p>
        {running && status.phase === 'features' ? (
          <ProgressBar value={status.featureProgress?.fraction ?? 0} label="Preparing features" />
        ) : null}
        {running && status.phase === 'training' ? <ProgressBar value={hist.length / status.epochs} label="Training progress" /> : null}
      </div>

      <div className="charts">
        <LineChart
          title="Loss (lower is better)"
          testId="chart-loss"
          xMax={running ? status.epochs : Math.max(ts.epochs, hist.length)}
          yMin={0}
          yLabel="Loss"
          series={[
            { name: 'Training', values: hist.map((h) => h.trainLoss), color: TRAIN_COLOR },
            { name: 'Validation', values: hist.map((h) => h.valLoss), color: VAL_COLOR, dashed: true },
          ]}
        />
        <LineChart
          title="Accuracy (higher is better)"
          testId="chart-accuracy"
          xMax={running ? status.epochs : Math.max(ts.epochs, hist.length)}
          yMax={1}
          yLabel="Accuracy"
          formatY={(v) => `${Math.round(v * 100)}%`}
          series={[
            { name: 'Training', values: hist.map((h) => h.trainAcc), color: TRAIN_COLOR },
            { name: 'Validation', values: hist.map((h) => h.valAcc), color: VAL_COLOR, dashed: true },
          ]}
        />
      </div>

      <details className="advanced" data-testid="advanced">
        <summary>Advanced settings</summary>
        <div className="advanced__body">
          <label className="field">
            <span>Epochs (passes over the training data): {ts.epochs}</span>
            <input
              type="number"
              min={1}
              max={100}
              step={1}
              value={ts.epochs}
              disabled={busy}
              data-testid="epochs-input"
              onChange={(e) => dispatch({ type: 'setTrainSettings', patch: { epochs: Number(e.target.value) || 1 } })}
            />
          </label>
          <label className="field">
            <span>
              Learning rate (step size): <strong data-testid="lr-value">{formatLr(ts.lr)}</strong>
            </span>
            <input
              type="range"
              min={-4}
              max={-1}
              step={0.05}
              value={sliderFromLr(ts.lr)}
              disabled={busy}
              data-testid="lr-slider"
              onChange={(e) => dispatch({ type: 'setTrainSettings', patch: { lr: lrFromSlider(Number(e.target.value)) } })}
            />
          </label>
          <div className="field">
            <label htmlFor="seed-input">Seed (same seed, same result)</label>
            <div className="field__row">
              <input
                id="seed-input"
                type="number"
                min={0}
                step={1}
                value={ts.seed}
                disabled={busy}
                data-testid="seed-input"
                onChange={(e) => dispatch({ type: 'setTrainSettings', patch: { seed: Number(e.target.value) || 0 } })}
              />
              <button
                type="button"
                className="btn"
                disabled={busy}
                data-testid="new-seed"
                onClick={() => dispatch({ type: 'setTrainSettings', patch: { seed: 1 + Math.floor(Math.random() * 9999) } })}
              >
                New seed
              </button>
            </div>
          </div>
        </div>
      </details>

      <RepeatPanel
        state={state}
        canStart={canStart}
        onStart={() => {
          const e = env();
          if (e) void runRepeat(e, ts);
        }}
      />
    </>
  );
}

function pctRange(values: (number | null)[]): string {
  const r = rangeOf(values);
  if (!r) return 'n/a';
  return `${formatPercent(r.min)} to ${formatPercent(r.max)}`;
}

function spreadPts(values: (number | null)[]): string {
  const r = rangeOf(values);
  // From the displayed (0.1-rounded) endpoints, so "81.3% to 87.5%" reads 6.2 points and not 6.3.
  return r ? `${(shownPercent(r.max) - shownPercent(r.min)).toFixed(1)} points` : 'n/a';
}

function RepeatPanel({ state, canStart, onStart }: { state: SessionState; canStart: boolean; onStart: () => void }) {
  const { repeat } = state;
  const running = repeat.state === 'running';
  const runs: RepeatRun[] = repeat.runs;
  const col = (pick: (r: RepeatRun) => number | null) => runs.map(pick);

  return (
    <section className="card" aria-labelledby="repeat-title" data-testid="repeat-panel">
      <h3 id="repeat-title">Repeat x5 with new seeds</h3>
      <p>
        Train the same recipe five times with seeds 1 to 5. Compare how much the score moves on the tiny 16-image validation set,
        on our 180-image validation set and on the big test set.
      </p>
      <div className="actions">
        <button type="button" className="btn" disabled={!canStart} onClick={onStart} data-testid="repeat-button">
          Repeat x5 with new seeds
        </button>
        <button type="button" className="btn" disabled={!running} onClick={() => cancelRun()} data-testid="repeat-cancel">
          Cancel
        </button>
      </div>
      <p role="status" aria-live="polite" data-testid="repeat-status">
        {repeat.state === 'running'
          ? repeat.featureProgress
            ? featureLine(repeat.featureProgress)
            : `Run ${repeat.current} of ${repeat.total}${repeat.epoch > 0 ? `, epoch ${repeat.epoch} of ${state.trainSettings.epochs}` : ''}...`
          : repeat.state === 'done'
            ? `Finished ${REPEAT_RUNS} runs.`
            : repeat.state === 'cancelled'
              ? `Cancelled after ${runs.length} finished run(s).`
              : repeat.state === 'error'
                ? `Failed: ${repeat.message ?? 'unknown error'}`
                : ''}
      </p>
      {runs.length > 0 ? (
        <table className="metrics" data-testid="repeat-table">
          <thead>
            <tr>
              <th scope="col">Run (seed)</th>
              <th scope="col">Validation accuracy, 16 images</th>
              <th scope="col">Validation accuracy, 180 images</th>
              <th scope="col">Kermany test accuracy</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.seed} data-testid={`repeat-row-${r.seed}`}>
                <th scope="row">Seed {r.seed}</th>
                <td>{formatPercent(r.val16)}</td>
                <td>{formatPercent(r.val180)}</td>
                <td>{formatPercent(r.test)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr data-testid="repeat-range">
              <th scope="row">Lowest to highest</th>
              <td>{pctRange(col((r) => r.val16))}</td>
              <td>{pctRange(col((r) => r.val180))}</td>
              <td>{pctRange(col((r) => r.test))}</td>
            </tr>
            <tr data-testid="repeat-spread">
              <th scope="row">Spread</th>
              <td data-testid="spread-val16">{spreadPts(col((r) => r.val16))}</td>
              <td data-testid="spread-val180">{spreadPts(col((r) => r.val180))}</td>
              <td data-testid="spread-test">{spreadPts(col((r) => r.test))}</td>
            </tr>
          </tfoot>
        </table>
      ) : null}
      {runs.length > 1 ? (
        <p className="muted">
          One image on a 16-image set is worth 6.25 points, so its score jumps around from run to run. A score you cannot repeat is
          not evidence.
        </p>
      ) : null}
    </section>
  );
}
