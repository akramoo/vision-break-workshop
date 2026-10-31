import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { EVAL_SETS, UPLOADS_LABEL, evalSetDef, type EvalSetDef, type SetChoice } from '../data/evalSets';
import type { FeatureProgress } from '../data/featureStore';
import { imageUrl } from '../data/images';
import { displayLabel, indexById, type ManifestEntry } from '../data/manifest';
import { evaluateSet } from '../ml/evaluate';
import { entryPixels } from '../ml/logPixels';
import {
  alwaysPneumoniaBaseline,
  confidenceOf,
  decide,
  formatPercent,
  metricsWithAbstain,
} from '../ml/metrics';
import { FALLBACK_NOTICE } from '../ml/trainRunner';
import { isAbortError } from '../ml/yield';
import { labelName, type LogDraftBase } from '../state/failureLog';
import { useSession, type EvalResult } from '../state/session';
import { ConfidenceThreshold } from './ConfidenceThreshold';
import { ConfusionMatrix } from './ConfusionMatrix';
import { formatDuration } from './format';
import { LogForm, useLogConfig } from './LogThis';
import { MetricsTable, type MetricRow } from './MetricsTable';
import { pixelsToDataUrl } from './PixelCanvas';
import { ProgressBar } from './ProgressBar';
import { UploadResults, UploadsManager } from './UploadsPanel';

const LIST_LIMIT = 40;

export function EvaluatePanel() {
  const { state, dispatch } = useSession();
  const { head, manifest } = state;
  const choice = state.evalSelection.set;
  const isUploads = choice === 'uploads';
  const def = evalSetDef(choice === 'uploads' ? 'kermany_test' : choice);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<FeatureProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);

  const modelReady = state.model.state === 'ready';
  const entries: readonly ManifestEntry[] = manifest.state === 'ready' ? manifest.entries : [];
  const result = head ? state.evalResults[def.id] : undefined;
  const fresh = head !== null && result !== undefined && result.headId === head.id;

  // Refs so the effect below only re-runs when the head / set / precomputed switch change.
  const extractorRef = useRef(state.extractor);
  extractorRef.current = state.extractor;
  const tauRef = useRef(state.abstainTau);
  tauRef.current = state.abstainTau;
  const freshRef = useRef(fresh);
  freshRef.current = fresh;

  const manifestReady = manifest.state === 'ready';
  const usePrecomputed = state.settings.usePrecomputed;
  useEffect(() => {
    if (isUploads || !head || !manifestReady || !modelReady || freshRef.current) return;
    const ac = new AbortController();
    setRunning(true);
    setError(null);
    setProgress(null);
    let warned = false;
    void evaluateSet({
      def,
      entries,
      head,
      extractor: extractorRef.current,
      usePrecomputed,
      tau: tauRef.current,
      signal: ac.signal,
      onProgress: (p) => {
        if (!ac.signal.aborted) setProgress(p);
      },
      onPrecomputedOk: () => dispatch({ type: 'featuresStatus', status: { state: 'ready' } }),
      onFallback: (reason) => {
        dispatch({ type: 'featuresStatus', status: { state: 'failed', reason } });
        if (!warned) {
          warned = true;
          dispatch({ type: 'notice', level: 'warn', message: `${FALLBACK_NOTICE} (${reason})` });
        }
      },
    }).then(
      (r) => {
        if (ac.signal.aborted) return;
        dispatch({ type: 'evalDone', result: r });
        setRunning(false);
      },
      (e: unknown) => {
        if (isAbortError(e) || ac.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
        setRunning(false);
      },
    );
    return () => {
      ac.abort();
      setRunning(false);
    };
    // `entries` is stable once the manifest is ready.
  }, [head, def.id, isUploads, manifestReady, modelReady, usePrecomputed, retry, dispatch]);

  return (
    <>
      <h2>Evaluate</h2>
      <p className="lead">
        Now the exam. We score the trained model on X-rays it has never seen. The key question is not "how high is the number?" but{' '}
        <strong>"is it better than a model that never looks at the picture?"</strong>
      </p>

      {!head ? (
        <div className="empty" data-testid="evaluate-empty">
          <p>
            <strong>No trained model yet.</strong> Go to the Train tab and click Train; it only takes a few seconds.
          </p>
          <button type="button" className="btn btn--primary" onClick={() => dispatch({ type: 'setTab', tab: 'train' })}>
            Go to Train
          </button>
        </div>
      ) : (
        <>
          <SetPicker selected={choice} onSelect={(id) => dispatch({ type: 'setEvalSet', set: id })} />
          <p className="muted" data-testid="eval-set-description">
            {isUploads
              ? 'Pictures you add yourself. Only the ones you label count towards the scores; unlabeled ones just get a prediction.'
              : def.description}
          </p>
          <ConfidenceThreshold />

          {!modelReady ? <p role="status">Waiting for the MobileNet model to finish loading...</p> : null}
          {isUploads ? (
            <div data-testid="eval-uploads">
              <UploadsManager />
              <UploadResults />
            </div>
          ) : null}
          {!isUploads && running ? (
            <div role="status" aria-live="polite" data-testid="eval-running">
              <p>
                {progress && progress.phase === 'extract'
                  ? `Running MobileNet on image ${progress.done} of ${progress.total}${
                      progress.etaMs === null ? '' : `, about ${formatDuration(progress.etaMs)} left`
                    }`
                  : `Scoring ${def.label}...`}
              </p>
              <ProgressBar value={progress?.fraction ?? 0} label="Evaluating" />
            </div>
          ) : null}
          {!isUploads && error ? (
            <div className="error-banner" role="alert" data-testid="eval-error">
              <strong className="error-banner__title">Evaluation failed</strong>
              <p>{error}</p>
              <button type="button" className="btn" onClick={() => setRetry((n) => n + 1)}>
                Try again
              </button>
            </div>
          ) : null}

          {!isUploads && fresh && result ? <Results def={def} result={result} tau={state.abstainTau} entries={entries} /> : null}
        </>
      )}
    </>
  );
}

function SetPicker({ selected, onSelect }: { selected: SetChoice; onSelect: (id: SetChoice) => void }) {
  return (
    <fieldset className="segmented" data-testid="eval-set-picker">
      <legend>Test on</legend>
      {EVAL_SETS.map((s) => (
        <label key={s.id} className={`segmented__option${selected === s.id ? ' segmented__option--on' : ''}`}>
          <input
            type="radio"
            name="eval-set"
            value={s.id}
            checked={selected === s.id}
            onChange={() => onSelect(s.id)}
            data-testid={`eval-set-${s.id}`}
          />
          <span>{s.label}</span>
        </label>
      ))}
      <label className={`segmented__option${selected === 'uploads' ? ' segmented__option--on' : ''}`}>
        <input
          type="radio"
          name="eval-set"
          value="uploads"
          checked={selected === 'uploads'}
          onChange={() => onSelect('uploads')}
          data-testid="eval-set-uploads"
        />
        <span>{UPLOADS_LABEL}</span>
      </label>
    </fieldset>
  );
}

function Results({ def, result, tau, entries }: { def: EvalSetDef; result: EvalResult; tau: number; entries: readonly ManifestEntry[] }) {
  const m = useMemo(() => metricsWithAbstain(result.labels, result.probs, tau), [result, tau]);
  const baseline = useMemo(() => alwaysPneumoniaBaseline(result.labels), [result]);
  const showAbstain = tau > 0.5;

  const rows: MetricRow[] = [
    {
      key: 'model',
      name: 'Our model',
      scores: m,
      emphasis: true,
      referred: { n: m.referred, fraction: m.referredFraction, coverage: m.coverage },
    },
    { key: 'baseline', name: 'Always say "pneumonia"', scores: baseline },
  ];

  const sick = m.confusion.tp + m.confusion.fn;
  const adult = def.negativeName === 'no finding';
  return (
    <div data-testid="eval-results" data-set={def.id}>
      <h3>
        {def.label}: {result.ids.length} X-rays
      </h3>
      <MetricsTable rows={rows} showAbstain={showAbstain} who={adult ? 'patients' : 'children'} />
      <p className="muted" data-testid="eval-mode">
        Features for this evaluation:{' '}
        <strong>{result.usedPrecomputed ? 'precomputed (shipped with the app)' : 'extracted live on this laptop'}</strong>.
      </p>
      {showAbstain ? (
        <p className="muted" data-testid="abstain-note">
          Referred cases ("Refer to a doctor") are left out of the scores and counted in Referred. Coverage is the share of X-rays
          the model dared to answer.
        </p>
      ) : null}
      <ul className="explain">
        <li>
          <strong>Sensitivity</strong> = sick {adult ? 'patients' : 'children'} caught: of the {sick} {adult ? 'patients' : 'children'} with
          pneumonia that the model answered about, how many it flagged.
        </li>
        <li>
          <strong>Specificity</strong> = {adult ? 'patients with no finding' : 'healthy children'} correctly cleared: of the{' '}
          {adult ? 'no-finding patients' : 'healthy ones'}, how many it let go home.
        </li>
        <li>
          The "always say pneumonia" row never looks at the picture. Its accuracy is just the share of pneumonia in this set, and
          our model has to beat that to be worth anything.
        </li>
      </ul>

      <h3>Where the mistakes are</h3>
      <ConfusionMatrix confusion={m.confusion} negativeName={def.negativeName} referred={m.referred} />
      <p className="note" data-testid="cost-note">
        A missed sick child (a "missed" box) usually costs more than a false alarm: a false alarm means an extra check, a missed
        pneumonia can mean a child goes home untreated.
      </p>

      <PredictionList def={def} result={result} tau={tau} entries={entries} />
    </div>
  );
}

function PredictionList({ def, result, tau, entries }: { def: EvalSetDef; result: EvalResult; tau: number; entries: readonly ManifestEntry[] }) {
  const [mistakesFirst, setMistakesFirst] = useState(true);
  const byId = useMemo(() => indexById(entries), [entries]);
  const logConfig = useLogConfig();
  const [logOpen, setLogOpen] = useState<string | null>(null);
  const [logged, setLogged] = useState<ReadonlySet<string>>(new Set());

  const buildDraft = (r: { id: string; p: number; truth: number; d: ReturnType<typeof decide> }) => async (): Promise<LogDraftBase> => {
    const entry = byId.get(r.id);
    if (!entry) throw new Error('This X-ray is not in the list of shipped images.');
    const px = await entryPixels(entry, def.id === 'stress');
    return {
      thumbnail: pixelsToDataUrl(px),
      setName: def.id,
      imageId: r.id,
      trueLabel: labelName(r.truth === 1, def.negativeName),
      predictedLabel: labelName(r.p >= 0.5, def.negativeName),
      confidence: confidenceOf(r.p),
      probPneumonia: r.p,
      referred: r.d === 'refer',
      config: logConfig(),
    };
  };

  const rows = useMemo(() => {
    const all = result.ids.map((id, i) => {
      const p = result.probs[i] as number;
      const d = decide(p, tau);
      const truth = result.labels[i] as number;
      const wrong = d !== 'refer' && (d === 'pneumonia') !== (truth === 1);
      return { id, p, d, truth, wrong, i };
    });
    const ordered = mistakesFirst ? [...all.filter((r) => r.wrong), ...all.filter((r) => !r.wrong)] : all;
    return ordered.slice(0, LIST_LIMIT);
  }, [result, tau, mistakesFirst]);

  return (
    <section aria-labelledby="pred-title">
      <h3 id="pred-title">Individual predictions</h3>
      <label className="toggle">
        <input type="checkbox" checked={mistakesFirst} onChange={(e) => setMistakesFirst(e.target.checked)} /> Show mistakes first
      </label>
      <table className="metrics preds" data-testid="prediction-list">
        <thead>
          <tr>
            <th scope="col">X-ray</th>
            <th scope="col">True label</th>
            <th scope="col">Model says</th>
            <th scope="col">Confidence</th>
            <th scope="col">Failure log</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const entry = byId.get(r.id);
            const truthName = entry ? displayLabel(entry) : r.truth === 1 ? 'pneumonia' : def.negativeName;
            const says = r.d === 'refer' ? 'Refer to a doctor' : r.d === 'pneumonia' ? 'pneumonia' : def.negativeName;
            return (
              <Fragment key={r.id}>
              <tr className={r.wrong ? 'pred--wrong' : r.d === 'refer' ? 'pred--refer' : ''} data-wrong={r.wrong} data-testid="pred-row">
                <td>
                  {entry ? <img src={imageUrl(entry.path)} alt="" width={48} height={48} loading="lazy" /> : null}
                  <span className="mono">{r.id}</span>
                </td>
                <td>{truthName}</td>
                <td>
                  {says}
                  {r.wrong ? <strong> (wrong)</strong> : null}
                </td>
                <td>{formatPercent(confidenceOf(r.p))}</td>
                <td>
                  <button
                    type="button"
                    className="btn btn--small"
                    aria-expanded={logOpen === r.id}
                    data-testid="pred-log"
                    onClick={() => setLogOpen(logOpen === r.id ? null : r.id)}
                  >
                    Log this
                  </button>
                  {logged.has(r.id) && logOpen !== r.id ? <span className="log-this__saved" role="status"> Saved</span> : null}
                </td>
              </tr>
              {logOpen === r.id ? (
                <tr className="log-row">
                  <td colSpan={5}>
                    <LogForm
                      build={buildDraft(r)}
                      onCancel={() => setLogOpen(null)}
                      onSaved={() => {
                        setLogged((s) => new Set(s).add(r.id));
                        setLogOpen(null);
                      }}
                    />
                  </td>
                </tr>
              ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
      <p className="muted">Showing the first {LIST_LIMIT} rows of {result.ids.length}.</p>
    </section>
  );
}
