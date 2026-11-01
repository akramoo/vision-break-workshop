import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { EVAL_SETS, evalSetDef, type EvalSetId } from '../data/evalSets';
import { evaluateSet } from '../ml/evaluate';
import { formatPercent } from '../ml/metrics';
import { describeModelConfig, modelConfigOf } from '../ml/modelConfig';
import {
  HALF_CREDIT_EXPLANATION,
  RESULT_SET_IDS,
  buildResultsExport,
  deltaPoints,
  formatDelta,
  resultsExportFileName,
  summarizeResults,
  type SetScores,
} from '../ml/results';
import { FALLBACK_NOTICE } from '../ml/trainRunner';
import { isAbortError } from '../ml/yield';
import { LOG_TAGS, tagCounts } from '../state/failureLog';
import { useSession } from '../state/session';
import { MAX_SNAPSHOTS, defaultSnapshotName, makeSnapshot, type Snapshot } from '../state/snapshots';
import { DISCLAIMER } from './disclaimer';
import { downloadJson } from './download';
import { ProgressBar } from './ProgressBar';

interface RunState {
  running: boolean;
  index: number;
  total: number;
  label: string;
  fraction: number;
  error: string | null;
  cancelled: boolean;
}
const IDLE_RUN: RunState = { running: false, index: 0, total: 0, label: '', fraction: 0, error: null, cancelled: false };

const featureMode = (precomputed: boolean) => (precomputed ? 'precomputed' : 'live extraction');
const tauPct = (tau: number) => `${Math.round(tau * 100)}%`;

export function ResultsPanel() {
  const { state, dispatch } = useSession();
  const { head, headInfo, abstainTau: tau, evalResults, manifest, snapshots, failureLog } = state;
  const usePrecomputed = state.settings.usePrecomputed;
  const [run, setRun] = useState<RunState>(IDLE_RUN);
  const [snapName, setSnapName] = useState('');
  const acRef = useRef<AbortController | null>(null);

  // Always recomputed from the stored probabilities at the CURRENT tau: nothing here can be stale.
  const summary = useMemo(() => summarizeResults(evalResults, head?.id ?? null, tau), [evalResults, head, tau]);
  const config = modelConfigOf(headInfo, state.trainSettings);
  const counts = useMemo(() => tagCounts(failureLog), [failureLog]);
  const lb = summary.leaderboard;
  const lbSet = lb ? summary.perSet[lb.set] : undefined;
  const entries = manifest.state === 'ready' ? manifest.entries : null;
  const canRun = head !== null && entries !== null && state.model.state === 'ready' && !run.running;

  // A different head (or leaving the tab) cancels a run in flight.
  const headId = head?.id ?? 0;
  useEffect(
    () => () => {
      acRef.current?.abort();
    },
    [headId],
  );

  const evaluatedAtOtherTau = summary.evaluatedSets.some((id) => Math.abs((summary.perSet[id]?.evaluatedAtTau ?? tau) - tau) > 1e-9);
  const otherModeSets = summary.evaluatedSets.filter((id) => summary.perSet[id]?.usedPrecomputed !== usePrecomputed);

  async function evaluateAll(force: boolean) {
    if (!head || !entries || run.running) return;
    const todo = EVAL_SETS.filter((d) => force || summary.perSet[d.id] === undefined);
    if (todo.length === 0) return;
    const ac = new AbortController();
    acRef.current = ac;
    setRun({ ...IDLE_RUN, running: true, total: todo.length, label: todo[0]?.label ?? '' });
    let warned = false;
    try {
      for (let i = 0; i < todo.length; i++) {
        const def = todo[i];
        if (!def) continue;
        setRun((r) => ({ ...r, index: i, label: def.label, fraction: 0 }));
        const result = await evaluateSet({
          def,
          entries,
          head,
          extractor: state.extractor,
          usePrecomputed,
          tau,
          signal: ac.signal,
          onProgress: (p) => {
            if (!ac.signal.aborted) setRun((r) => ({ ...r, fraction: p.fraction }));
          },
          onPrecomputedOk: () => dispatch({ type: 'featuresStatus', status: { state: 'ready' } }),
          onFallback: (reason) => {
            dispatch({ type: 'featuresStatus', status: { state: 'failed', reason } });
            if (!warned) {
              warned = true;
              dispatch({ type: 'notice', level: 'warn', message: `${FALLBACK_NOTICE} (${reason})` });
            }
          },
        });
        if (ac.signal.aborted) break;
        dispatch({ type: 'evalDone', result });
      }
      setRun({ ...IDLE_RUN, cancelled: ac.signal.aborted });
    } catch (e) {
      if (isAbortError(e) || ac.signal.aborted) setRun({ ...IDLE_RUN, cancelled: true });
      else setRun({ ...IDLE_RUN, error: e instanceof Error ? e.message : String(e) });
    } finally {
      if (acRef.current === ac) acRef.current = null;
    }
  }

  const defaultName = defaultSnapshotName(snapshots);
  function saveSnapshot() {
    if (!summary.complete || snapshots.length >= MAX_SNAPSHOTS) return;
    const perSet = {} as Record<EvalSetId, SetScores>;
    for (const id of RESULT_SET_IDS) {
      const s = summary.perSet[id];
      if (!s) return;
      perSet[id] = s.scores;
    }
    dispatch({
      type: 'snapshotAdd',
      snapshot: makeSnapshot({
        name: snapName,
        existing: snapshots,
        config: { ...config, tau, stampTrained: false, usePrecomputed },
        perSet,
      }),
    });
    setSnapName('');
  }

  function downloadResults() {
    downloadJson(
      resultsExportFileName(),
      buildResultsExport({ summary, config, snapshots, failureCounts: counts }),
    );
  }

  return (
    <>
      <h2>Results</h2>
      <p className="lead no-print">
        Your results card, like the printed handout. The number to post on the leaderboard is the <strong>worst-case score</strong>: the weakest
        of the three test sets. A model is only as good as its worst set.
      </p>

      <div className="card no-print results-controls" data-testid="results-controls">
        <div className="actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => void evaluateAll(false)}
            disabled={!canRun || summary.missingSets.length === 0}
            data-testid="results-eval-all"
          >
            Evaluate all three sets now
          </button>
          <button type="button" className="btn" onClick={() => void evaluateAll(true)} disabled={!canRun || !head} data-testid="results-eval-rerun">
            Re-run all three with the current settings
          </button>
          {run.running ? (
            <button type="button" className="btn" onClick={() => acRef.current?.abort()} data-testid="results-eval-cancel">
              Cancel
            </button>
          ) : null}
          <span className="muted" data-testid="results-eval-status">
            {summary.evaluatedSets.length} of {RESULT_SET_IDS.length} sets evaluated. Features: {featureMode(usePrecomputed)}.
          </span>
        </div>
        {run.running ? (
          <div role="status" aria-live="polite" data-testid="results-running">
            <p>
              Scoring {run.label} (set {run.index + 1} of {run.total})...
            </p>
            <ProgressBar value={(run.index + run.fraction) / Math.max(1, run.total)} label="Evaluating the three test sets" />
          </div>
        ) : null}
        {run.cancelled ? (
          <p className="muted" data-testid="results-cancelled">
            Cancelled. Sets that finished are kept.
          </p>
        ) : null}
        {run.error ? (
          <p className="inline-warn inline-warn--strong" role="alert" data-testid="results-error">
            Evaluation failed: {run.error}
          </p>
        ) : null}
        <div className="actions">
          <button type="button" className="btn" onClick={downloadResults} disabled={!head} data-testid="results-download">
            Download results JSON
          </button>
          <button type="button" className="btn" onClick={() => window.print()} data-testid="results-print">
            Print
          </button>
        </div>
      </div>

      <article
        className="results-card"
        id="results-card"
        aria-label="Results card"
        data-testid="results-card"
        data-tau={tau}
        data-complete={summary.complete}
        data-recomputed={evaluatedAtOtherTau}
      >
        <h3 className="results-card__title">Results card</h3>
        <p className="disclaimer" data-testid="results-disclaimer">
          {DISCLAIMER}
        </p>

        {!head ? (
          <div className="empty" data-testid="results-empty">
            <p>
              <strong>No trained model yet.</strong> Train one (Train tab), then evaluate it on the three test sets to get your leaderboard value.
            </p>
            <button type="button" className="btn btn--primary no-print" onClick={() => dispatch({ type: 'setTab', tab: 'train' })}>
              Go to Train
            </button>
          </div>
        ) : (
          <>
            <section className="leaderboard" aria-labelledby="lb-title" data-testid="leaderboard">
              <h4 id="lb-title">Leaderboard value (worst-case score)</h4>
              {lb && lbSet ? (
                <div className="leaderboard__row">
                  <p className="leaderboard__value" data-testid="leaderboard-value" data-score={lb.score}>
                    {formatPercent(lb.score)}
                  </p>
                  <ul className="leaderboard__meta">
                    <li>
                      Set that set the minimum: <strong data-testid="leaderboard-set">{evalSetDef(lb.set).label}</strong>
                    </li>
                    <li>
                      Refer threshold (tau): <strong data-testid="leaderboard-tau">{tauPct(tau)}</strong>
                    </li>
                    <li>
                      Coverage on that set: <strong data-testid="leaderboard-coverage">{formatPercent(lbSet.scores.coverage)}</strong>
                    </li>
                  </ul>
                </div>
              ) : (
                <p className="leaderboard__empty" data-testid="leaderboard-incomplete">
                  Not complete yet. {summary.evaluatedSets.length} of {RESULT_SET_IDS.length} test sets are evaluated; press "Evaluate all three sets now" to get
                  the leaderboard value.
                </p>
              )}
              <p className="note" data-testid="leaderboard-explain">
                {HALF_CREDIT_EXPLANATION} The score is the lowest of the three sets' half-credit scores; at tau = 50% (never refer) it is balanced accuracy.
              </p>
            </section>

            {evaluatedAtOtherTau ? (
              <p className="inline-warn" data-testid="results-recomputed">
                Recomputed for the current threshold (tau = {tauPct(tau)}). The sets were evaluated at a different tau; the stored predictions are re-scored,
                so these numbers are current. Snapshots keep the tau they were saved with.
              </p>
            ) : null}
            {otherModeSets.length > 0 ? (
              <p className="inline-warn" data-testid="results-mode-note">
                {otherModeSets.map((id) => evalSetDef(id).label).join(', ')} {otherModeSets.length === 1 ? 'was' : 'were'} evaluated with{' '}
                {featureMode(!usePrecomputed)}, but the switch is now {featureMode(usePrecomputed)}. Re-run all three to use the current mode.
              </p>
            ) : null}

            <section aria-labelledby="persets-title">
              <h4 id="persets-title">Scores per test set</h4>
              <table className="metrics results-table" data-testid="results-table">
                <thead>
                  <tr>
                    <th scope="col">Test set</th>
                    <th scope="col">Predictor</th>
                    <th scope="col">{tau > 0.5 ? 'Accuracy (answered cases)' : 'Accuracy'}</th>
                    <th scope="col">{tau > 0.5 ? 'Sensitivity (answered)' : 'Sensitivity'}</th>
                    <th scope="col">{tau > 0.5 ? 'Specificity (answered)' : 'Specificity'}</th>
                    <th scope="col">Balanced accuracy</th>
                    <th scope="col">Referred n (%)</th>
                    <th scope="col">Coverage</th>
                    <th scope="col">Half-credit score</th>
                    <th scope="col">Features</th>
                  </tr>
                </thead>
                {RESULT_SET_IDS.map((id) => {
                  const s = summary.perSet[id];
                  const worst = lb?.set === id;
                  if (!s) {
                    return (
                      <tbody key={id} data-testid={`res-${id}`}>
                        <tr>
                          <th scope="row">{evalSetDef(id).label}</th>
                          <td colSpan={9} className="muted" data-testid={`res-${id}-missing`}>
                            not evaluated yet
                          </td>
                        </tr>
                      </tbody>
                    );
                  }
                  const m = s.scores;
                  return (
                    <tbody key={id} data-testid={`res-${id}`} data-worst={worst}>
                      <tr className="metrics__row--model">
                        <th scope="row" rowSpan={2}>
                          {s.label}
                          {worst ? <small className="worst-tag"> (worst case)</small> : null}
                          <small className="muted"> n = {m.n}</small>
                        </th>
                        <td>Our model</td>
                        <td data-testid={`res-${id}-model-accuracy`}>{formatPercent(m.accuracy)}</td>
                        <td data-testid={`res-${id}-model-sensitivity`}>{formatPercent(m.sensitivity)}</td>
                        <td data-testid={`res-${id}-model-specificity`}>{formatPercent(m.specificity)}</td>
                        <td data-testid={`res-${id}-model-balanced`}>{formatPercent(m.balancedAccuracy)}</td>
                        <td data-testid={`res-${id}-model-referred`}>
                          {m.referred} ({formatPercent(m.referredFraction)})
                        </td>
                        <td data-testid={`res-${id}-model-coverage`}>{formatPercent(m.coverage)}</td>
                        <td data-testid={`res-${id}-model-score`}>
                          <strong>{formatPercent(m.halfCreditScore)}</strong>
                        </td>
                        <td data-testid={`res-${id}-model-mode`}>{featureMode(s.usedPrecomputed)}</td>
                      </tr>
                      <tr className="metrics__row--baseline">
                        <td>Always say "pneumonia"</td>
                        <td data-testid={`res-${id}-baseline-accuracy`}>{formatPercent(s.baseline.accuracy)}</td>
                        <td data-testid={`res-${id}-baseline-sensitivity`}>{formatPercent(s.baseline.sensitivity)}</td>
                        <td data-testid={`res-${id}-baseline-specificity`}>{formatPercent(s.baseline.specificity)}</td>
                        <td data-testid={`res-${id}-baseline-balanced`}>{formatPercent(s.baseline.balancedAccuracy)}</td>
                        <td>0 (0.0%)</td>
                        <td>100.0%</td>
                        <td data-testid={`res-${id}-baseline-score`}>{formatPercent(s.baseline.halfCreditScore)}</td>
                        <td>-</td>
                      </tr>
                    </tbody>
                  );
                })}
              </table>
              <p className="muted results-footnote">
                Accuracy, sensitivity and specificity use the cases the model answered. The half-credit score counts every referred case as half right.
              </p>
            </section>

            <section aria-labelledby="config-title">
              <h4 id="config-title">The model behind these numbers</h4>
              <dl className="config-grid" data-testid="results-config">
                <dt>Balance mode</dt>
                <dd data-testid="cfg-mode">{config.mode}</dd>
                <dt>Adult X-rays mixed in</dt>
                <dd data-testid="cfg-adultMix">{config.adultMix ? 'yes' : 'no'}</dd>
                <dt>Augmentation</dt>
                <dd data-testid="cfg-augmentation">{config.augmentation ? 'yes' : 'no'}</dd>
                <dt>Epochs</dt>
                <dd data-testid="cfg-epochs">{config.epochs}</dd>
                <dt>Learning rate</dt>
                <dd data-testid="cfg-lr">{config.lr.toExponential(1)}</dd>
                <dt>Seed</dt>
                <dd data-testid="cfg-seed">{config.seed}</dd>
                <dt>Refer threshold (tau)</dt>
                <dd data-testid="cfg-tau">{tauPct(tau)}</dd>
                <dt>Trained on stamped images</dt>
                <dd data-testid="cfg-stamp">no</dd>
              </dl>
            </section>

            <section aria-labelledby="tags-title">
              <h4 id="tags-title">Failures you logged</h4>
              <ul className="tag-counts" data-testid="results-tag-counts">
                {LOG_TAGS.map((t) => (
                  <li key={t} className="tag-chip tag-chip--static" data-testid={`res-tag-${t}`}>
                    {t} <strong>{counts[t]}</strong>
                  </li>
                ))}
              </ul>
              <p className="muted">{failureLog.length} {failureLog.length === 1 ? 'entry' : 'entries'} in total.</p>
            </section>
          </>
        )}

        <section aria-labelledby="snap-title" data-testid="snapshots-section">
          <h4 id="snap-title">Snapshots: before and after your fixes</h4>
          <div className="snap-save no-print">
            <label htmlFor="snapshot-name">Snapshot name</label>
            <input
              id="snapshot-name"
              type="text"
              data-testid="snapshot-name"
              value={snapName}
              maxLength={40}
              placeholder={defaultName}
              onChange={(e) => setSnapName(e.target.value)}
            />
            <button
              type="button"
              className="btn btn--primary"
              onClick={saveSnapshot}
              disabled={!summary.complete || snapshots.length >= MAX_SNAPSHOTS}
              data-testid="snapshot-save"
            >
              Save snapshot
            </button>
            <span className="muted" data-testid="snapshot-count">
              {snapshots.length} of {MAX_SNAPSHOTS} saved
              {!summary.complete ? '. Evaluate all three sets first.' : snapshots.length >= MAX_SNAPSHOTS ? '. Delete one to save another.' : '.'}
            </span>
          </div>
          {snapshots.length === 0 ? (
            <p className="muted" data-testid="snapshots-empty">
              No snapshots yet. Save one now ("Before fix"), apply a fix, retrain, re-evaluate all three sets and save "After fix 1". The columns appear
              side by side with the change in points.
            </p>
          ) : (
            <SnapshotTable snapshots={snapshots} onDelete={(id) => dispatch({ type: 'snapshotDelete', id })} />
          )}
        </section>

        <p className="disclaimer results-card__foot">{DISCLAIMER}</p>
      </article>
    </>
  );
}

const SNAP_METRICS: readonly { key: 'accuracy' | 'sensitivity' | 'specificity' | 'halfCreditScore'; label: string }[] = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'sensitivity', label: 'Sensitivity' },
  { key: 'specificity', label: 'Specificity' },
  { key: 'halfCreditScore', label: 'Half-credit score' },
];

function SnapshotTable({ snapshots, onDelete }: { snapshots: readonly Snapshot[]; onDelete: (id: string) => void }) {
  const cell = (testId: string, value: number | null, prev: number | null | undefined, bold = false) => {
    const d = prev === undefined ? null : deltaPoints(prev, value);
    return (
      <td data-testid={testId}>
        <span className={bold ? 'snap-value snap-value--bold' : 'snap-value'}>{formatPercent(value)}</span>
        {prev !== undefined ? (
          <span className={`delta${d === null ? '' : d > 0.05 ? ' delta--up' : d < -0.05 ? ' delta--down' : ''}`} data-testid={`${testId}-delta`}>
            {' '}
            {formatDelta(d)}
          </span>
        ) : null}
      </td>
    );
  };
  return (
    <div className="snap-wrap">
      <table className="metrics snap-table" data-testid="snapshot-table">
        <thead>
          <tr>
            <th scope="col">Measure</th>
            {snapshots.map((s, i) => (
              <th scope="col" key={s.id} data-testid={`snap-col-${i}`}>
                <span data-testid={`snap-name-${i}`}>{s.name}</span>
                <small className="muted"> {new Date(s.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</small>
                <button
                  type="button"
                  className="btn btn--small no-print"
                  aria-label={`Delete snapshot ${s.name}`}
                  onClick={() => onDelete(s.id)}
                  data-testid={`snap-delete-${i}`}
                >
                  Delete
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr className="snap-lb">
            <th scope="row">Leaderboard value (worst case)</th>
            {snapshots.map((s, i) => {
              const prev = i > 0 ? (snapshots[i - 1]?.leaderboard?.score ?? null) : undefined;
              const d = prev === undefined ? null : deltaPoints(prev, s.leaderboard?.score ?? null);
              return (
                <td key={s.id} data-testid={`snap-${i}-leaderboard`}>
                  <strong className="snap-value">{formatPercent(s.leaderboard?.score ?? null)}</strong>
                  {s.leaderboard ? <small className="muted"> {evalSetDef(s.leaderboard.set).label}</small> : null}
                  {prev !== undefined ? (
                    <span className={`delta${d === null ? '' : d > 0.05 ? ' delta--up' : d < -0.05 ? ' delta--down' : ''}`} data-testid={`snap-${i}-leaderboard-delta`}>
                      {' '}
                      {formatDelta(d)}
                    </span>
                  ) : null}
                </td>
              );
            })}
          </tr>
          <tr>
            <th scope="row">Model</th>
            {snapshots.map((s, i) => (
              <td key={s.id} className="snap-config" data-testid={`snap-${i}-config`}>
                {describeModelConfig(s.config)}, tau {tauPct(s.config.tau)}, {featureMode(s.config.usePrecomputed)}
                {s.config.stampTrained ? ', stamp-trained' : ''}
              </td>
            ))}
          </tr>
        </tbody>
        {RESULT_SET_IDS.map((id) => (
          <tbody key={id}>
            <tr className="snap-set">
              <th scope="rowgroup" colSpan={snapshots.length + 1}>
                {evalSetDef(id).label}
              </th>
            </tr>
            {SNAP_METRICS.map((m) => (
              <tr key={m.key}>
                <th scope="row">{m.label}</th>
                {snapshots.map((s, i) => (
                  <Fragment key={s.id}>
                    {cell(`snap-${i}-${id}-${m.key}`, s.perSet[id][m.key], i > 0 ? (snapshots[i - 1]?.perSet[id][m.key] ?? null) : undefined, m.key === 'halfCreditScore')}
                  </Fragment>
                ))}
              </tr>
            ))}
          </tbody>
        ))}
      </table>
      <p className="muted results-footnote">Each column after the first shows the change in percentage points compared with the column before it.</p>
    </div>
  );
}
