import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { EVAL_SETS, UPLOADS_LABEL, evalSetDef, type EvalSetDef, type SetChoice } from '../data/evalSets';
import type { FeatureProgress } from '../data/featureStore';
import { imageUrl } from '../data/images';
import { displayLabel, indexById, type ManifestEntry } from '../data/manifest';
import { evaluateSet } from '../ml/evaluate';
import { predictedCamClass } from '../ml/gradcam';
import type { Head } from '../ml/head';
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
import { HeatmapOverlay } from './HeatmapOverlay';
import { LogForm, useLogConfig } from './LogThis';
import { MetricsTable, type MetricRow } from './MetricsTable';
import { Pager } from './Pager';
import {
  DEFAULT_PAGE_SIZE,
  applyView,
  buildRows,
  categoryCounts,
  classCounts,
  countsLabel,
  effectiveMode,
  pageInfo,
  type ClassFilter,
  type ViewMode,
} from './paging';
import { pixelsToDataUrl } from './PixelCanvas';
import { ProgressBar } from './ProgressBar';
import { UploadResults, UploadsManager } from './UploadsPanel';

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

          {!isUploads && fresh && result ? <Results def={def} result={result} tau={state.abstainTau} entries={entries} head={head} /> : null}
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

function Results({ def, result, tau, entries, head }: { def: EvalSetDef; result: EvalResult; tau: number; entries: readonly ManifestEntry[]; head: Head }) {
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

      <PredictionList def={def} result={result} tau={tau} entries={entries} head={head} />
    </div>
  );
}

function PredictionList({ def, result, tau, entries, head }: { def: EvalSetDef; result: EvalResult; tau: number; entries: readonly ManifestEntry[]; head: Head }) {
  const [mode, setMode] = useState<ViewMode>('mistakes-first');
  const [cls, setCls] = useState<ClassFilter>('all');
  const [pageSize, setPageSize] = useState<number>(DEFAULT_PAGE_SIZE);
  const [page, setPage] = useState(1);
  const byId = useMemo(() => indexById(entries), [entries]);
  const logConfig = useLogConfig();
  const [logOpen, setLogOpen] = useState<string | null>(null);
  const [logged, setLogged] = useState<ReadonlySet<string>>(new Set());
  const [heatOpen, setHeatOpen] = useState<string | null>(null);

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

  const allRows = useMemo(() => buildRows(result.ids, result.probs, result.labels, tau), [result, tau]);
  const totals = useMemo(() => categoryCounts(allRows), [allRows]);
  const classes = useMemo(() => classCounts(allRows), [allRows]);
  const showReferred = tau > 0.5;
  const eMode = effectiveMode(mode, tau);
  // Counts on the chips follow the true-class filter, so each chip says how many rows it would show.
  const inClass = useMemo(() => applyView(allRows, 'dataset', cls), [allRows, cls]);
  const chipCounts = useMemo(() => categoryCounts(inClass), [inClass]);
  const view = useMemo(() => applyView(allRows, eMode, cls), [allRows, eMode, cls]);
  const info = pageInfo(view.length, page, pageSize);
  const rows = view.slice(info.start, info.end);

  // Back to page 1 (and close open forms) when the set, the threshold or the head changes.
  const [seen, setSeen] = useState({ result, tau, headId: head.id });
  if (seen.result !== result || seen.tau !== tau || seen.headId !== head.id) {
    setSeen({ result, tau, headId: head.id });
    setPage(1);
    setLogOpen(null);
    setHeatOpen(null);
  }

  const changeView = (next: { mode?: ViewMode; cls?: ClassFilter; size?: number }) => {
    if (next.mode !== undefined) setMode(next.mode);
    if (next.cls !== undefined) setCls(next.cls);
    if (next.size !== undefined) setPageSize(next.size);
    setPage(1);
    setLogOpen(null);
    setHeatOpen(null);
  };
  const goToPage = (p: number) => {
    setPage(p);
    setLogOpen(null);
    setHeatOpen(null);
  };

  const modeChips: { id: ViewMode; text: string; count: number | null }[] = [
    { id: 'mistakes-first', text: 'Mistakes first (most confident first)', count: null },
    { id: 'only-mistakes', text: 'Only mistakes', count: chipCounts.mistakes },
    ...(showReferred ? [{ id: 'only-referred' as const, text: 'Only referred', count: chipCounts.referred }] : []),
    { id: 'only-correct', text: 'Only correct', count: chipCounts.correct },
    { id: 'dataset', text: 'Dataset order', count: null },
  ];
  const classChips: { id: ClassFilter; text: string; count: number }[] = [
    { id: 'all', text: 'All', count: classes.total },
    { id: 'pneumonia', text: 'pneumonia', count: classes.pneumonia },
    { id: 'negative', text: def.negativeName, count: classes.negative },
  ];

  return (
    <section aria-labelledby="pred-title">
      <h3 id="pred-title">Individual predictions</h3>
      <p data-testid="pred-counts">
        <strong>{countsLabel(totals)}</strong>{' '}
        <span className="muted">of {totals.total} X-rays</span>
      </p>
      <fieldset className="segmented chips" data-testid="pred-mode">
        <legend>Show</legend>
        {modeChips.map((c) => (
          <label key={c.id} className={`segmented__option${eMode === c.id ? ' segmented__option--on' : ''}`}>
            <input type="radio" name="pred-mode" value={c.id} checked={eMode === c.id} onChange={() => changeView({ mode: c.id })} data-testid={`pred-mode-${c.id}`} />
            <span>{c.count === null ? c.text : `${c.text} (${c.count})`}</span>
          </label>
        ))}
      </fieldset>
      {eMode === 'dataset' || eMode === 'only-correct' ? null : (
        <p className="muted" data-testid="pred-order-note">
          {eMode === 'only-referred' ? 'Referred cases' : 'Mistakes'} are listed with the most confident answer first.
        </p>
      )}
      <fieldset className="segmented chips" data-testid="pred-class">
        <legend>True label</legend>
        {classChips.map((c) => (
          <label key={c.id} className={`segmented__option${cls === c.id ? ' segmented__option--on' : ''}`}>
            <input type="radio" name="pred-class" value={c.id} checked={cls === c.id} onChange={() => changeView({ cls: c.id })} data-testid={`pred-class-${c.id}`} />
            <span>{`${c.text} (${c.count})`}</span>
          </label>
        ))}
      </fieldset>
      <Pager info={info} onPage={goToPage} onPageSize={(size) => changeView({ size })} id="pred" />
      <table className="metrics preds" data-testid="prediction-list">
        <thead>
          <tr>
            <th scope="col">X-ray</th>
            <th scope="col">True label</th>
            <th scope="col">Model says</th>
            <th scope="col">Confidence</th>
            <th scope="col">Failure log</th>
            <th scope="col">Heatmap</th>
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
                <td>
                  <button
                    type="button"
                    className="btn btn--small"
                    aria-expanded={heatOpen === r.id}
                    data-testid="pred-heatmap-btn"
                    onClick={() => setHeatOpen(heatOpen === r.id ? null : r.id)}
                  >
                    Heatmap
                  </button>
                </td>
              </tr>
              {heatOpen === r.id ? (
                <tr className="heatmap-row" data-testid="pred-heatmap-row">
                  <td colSpan={6}>
                    {entry ? (
                      <RowHeatmap entry={entry} stress={def.id === 'stress'} head={head} testId="pred-heatmap" p={r.p} />
                    ) : (
                      <p className="muted">This X-ray is not in the list of shipped images.</p>
                    )}
                  </td>
                </tr>
              ) : null}
              {logOpen === r.id ? (
                <tr className="log-row">
                  <td colSpan={6}>
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
      {view.length === 0 ? (
        <p className="muted" data-testid="pred-empty">
          No X-ray matches this filter.
        </p>
      ) : (
        <Pager info={info} onPage={goToPage} onPageSize={(size) => changeView({ size })} id="pred-bottom" live={false} />
      )}
    </section>
  );
}

/** Loads the pixels the model saw for one shipped X-ray (with the stress transform for the stress set) and shows its heatmap. */
function RowHeatmap({ entry, stress, head, testId, p }: { entry: ManifestEntry; stress: boolean; head: Head; testId: string; p: number }) {
  const [pixels, setPixels] = useState<Float32Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setPixels(null);
    setError(null);
    entryPixels(entry, stress).then(
      (p) => {
        if (!cancelled) setPixels(p);
      },
      (e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [entry, stress]);
  if (error) {
    return (
      <p className="inline-warn" role="alert">
        Could not load this X-ray: {error}
      </p>
    );
  }
  if (!pixels) return <p className="muted" role="status">Loading the X-ray...</p>;
  return <HeatmapOverlay pixels={pixels} heads={[{ id: 'main', label: 'Your model', head }]} size={224} autoShow testId={testId} initialClass={predictedCamClass(p)} />;
}
