import { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import type { FeatureProgress } from '../data/featureStore';
import { evalSetDef, evalSetIds } from '../data/evalSets';
import { imageUrl, loadImageTensor } from '../data/images';
import { indexById, labelValue, selectEntries, type ManifestEntry } from '../data/manifest';
import { alwaysPneumoniaBaseline, formatPercent, metricsWithAbstain } from '../ml/metrics';
import { predictPixelArrays } from '../ml/predictImage';
import {
  STAMP_TARGETS,
  computeFlip,
  isInstantStamp,
  sameStampConfig,
  trainStampedHead,
  type ClassCalls,
  type FlipResult,
  type FlipSetId,
  type StampClass,
  type StampTarget,
} from '../ml/stampLab';
import { currentGeneration } from '../ml/generation';
import { FALLBACK_NOTICE, isRunActive } from '../ml/trainRunner';
import type { Head } from '../ml/head';
import { isAbortError } from '../ml/yield';
import { DEFAULT_STAMP, STAMP_POSITIONS, applyStamp, resolveStampConfig, type StampConfig, type StampPosition } from '../shared/pixelOps.js';
import { labelName } from '../state/failureLog';
import { useSession } from '../state/session';
import { formatDuration } from './format';
import { HeatmapOverlay, type HeatmapHead } from './HeatmapOverlay';
import { EntryEditor, useAddLog, useLogConfig } from './LogThis';
import { MetricsTable, type MetricRow } from './MetricsTable';
import { PixelCanvas, pixelsToDataUrl } from './PixelCanvas';
import { PredictionBadge } from './PredictionBadge';
import { ProgressBar } from './ProgressBar';

const PAGE_SIZE = 12;
const BROWSE_COUNT = 24;

function stampPixels(pixels: Float32Array, cfg: StampConfig): Promise<Float32Array> {
  const out = tf.tidy(() => applyStamp(tf, tf.tensor3d(pixels, [224, 224, 3]), cfg));
  return out.data().then((d) => {
    out.dispose();
    return d as Float32Array;
  });
}

function progressLine(p: FeatureProgress | null): string {
  if (!p) return 'Preparing the features...';
  if (p.phase === 'download') return `Loading precomputed features... ${Math.round(p.fraction * 100)}%`;
  return `Running MobileNet on image ${p.done} of ${p.total}${p.etaMs === null ? '' : `, about ${formatDuration(p.etaMs)} left`}`;
}

export function StampPanel() {
  const { state, dispatch } = useSession();
  const { head, stampedHead, manifest, abstainTau: tau } = state;
  const logConfig = useLogConfig();
  const addLog = useAddLog();
  const [lastLoggedId, setLastLoggedId] = useState<string | null>(null);
  const entries: readonly ManifestEntry[] = manifest.state === 'ready' ? manifest.entries : [];
  const usePrecomputed = state.settings.usePrecomputed;
  const modelReady = state.model.state === 'ready';
  const ready = modelReady && manifest.state === 'ready';

  const [cfg, setCfg] = useState<StampConfig>(() => resolveStampConfig(stampedHead?.config ?? DEFAULT_STAMP));
  const [target, setTarget] = useState<StampTarget>(stampedHead?.target ?? 'pneumonia');
  const [flipSet, setFlipSet] = useState<FlipSetId>('kermany_test');
  const [stampClass, setStampClass] = useState<StampClass>('normal');

  const [training, setTraining] = useState<{ running: boolean; phase: 'features' | 'training'; epoch: number; progress: FeatureProgress | null }>({
    running: false,
    phase: 'features',
    epoch: 0,
    progress: null,
  });
  const [trainMsg, setTrainMsg] = useState<string | null>(null);
  const [flip, setFlip] = useState<FlipResult | null>(null);
  const [flipRun, setFlipRun] = useState<{ running: boolean; progress: FeatureProgress | null; error: string | null }>({
    running: false,
    progress: null,
    error: null,
  });
  const trainAc = useRef<AbortController | null>(null);
  const flipAc = useRef<AbortController | null>(null);

  const cfgKey = `${cfg.size}|${cfg.position}|${cfg.offsetX}|${cfg.offsetY}|${cfg.opacity}`;
  const instant = isInstantStamp(cfg, usePrecomputed);
  const headId = head?.id ?? 0;
  const stampedHeadId = stampedHead?.head.id ?? 0;

  // Abort whatever is running when the panel goes away.
  useEffect(
    () => () => {
      trainAc.current?.abort();
      flipAc.current?.abort();
    },
    [],
  );

  // The stamp model is trained with the SAME recipe as the normal ("before") model when there is one, so the only
  // difference between the two is the stamp. (Augmentation is not used here: there are no stamped augmented features.)
  const recipe = useMemo(
    () =>
      state.headInfo
        ? {
            ...state.trainSettings,
            mode: state.headInfo.mode,
            adultMix: state.headInfo.adultMix,
            epochs: state.headInfo.epochs,
            lr: state.headInfo.lr,
            seed: state.headInfo.seed,
            valSet: state.headInfo.valSet,
            augmentation: false,
          }
        : { ...state.trainSettings, augmentation: false },
    [state.headInfo, state.trainSettings],
  );

  function noticeFallback() {
    let warned = false;
    return (reason: string) => {
      dispatch({ type: 'featuresStatus', status: { state: 'failed', reason } });
      if (!warned) {
        warned = true;
        dispatch({ type: 'notice', level: 'warn', message: `${FALLBACK_NOTICE} (${reason})` });
      }
    };
  }

  async function runFlip(before: Head, after: Head | null, flipCfg: StampConfig) {
    flipAc.current?.abort();
    const ac = new AbortController();
    flipAc.current = ac;
    setFlipRun({ running: true, progress: null, error: null });
    try {
      const r = await computeFlip({
        setId: flipSet,
        stampClass,
        cfg: flipCfg,
        before,
        after,
        entries,
        extractor: state.extractor,
        usePrecomputed,
        signal: ac.signal,
        onProgress: (p) => {
          if (!ac.signal.aborted) setFlipRun((f) => ({ ...f, progress: p }));
        },
        onPrecomputedOk: () => dispatch({ type: 'featuresStatus', status: { state: 'ready' } }),
        onFallback: noticeFallback(),
      });
      if (ac.signal.aborted) return;
      setFlip(r);
      setFlipRun({ running: false, progress: null, error: null });
    } catch (e) {
      if (flipAc.current !== ac) return;
      setFlipRun({ running: false, progress: null, error: isAbortError(e) || ac.signal.aborted ? null : e instanceof Error ? e.message : String(e) });
    }
  }

  // With the default stamp and precomputed features the flip rate is instant, so keep it up to date automatically.
  useEffect(() => {
    if (!ready || !head || !instant) return;
    void runFlip(head, stampedHead?.head ?? null, cfg);
    return () => flipAc.current?.abort();
    // runFlip reads the other inputs through its closure; these are the triggers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, headId, stampedHeadId, flipSet, stampClass, instant, cfgKey]);

  async function trainStamped() {
    if (!head || !ready || trainAc.current) return;
    const ac = new AbortController();
    trainAc.current = ac;
    const trainCfg = cfg;
    const gen = currentGeneration();
    const t0 = performance.now();
    setTrainMsg(null);
    setTraining({ running: true, phase: 'features', epoch: 0, progress: null });
    try {
      const res = await trainStampedHead({
        entries,
        extractor: state.extractor,
        usePrecomputed,
        settings: recipe,
        cfg: trainCfg,
        target,
        signal: ac.signal,
        onProgress: (p) => {
          if (!ac.signal.aborted) setTraining((t) => ({ ...t, progress: p }));
        },
        onEpoch: (s) => setTraining((t) => ({ ...t, phase: 'training', epoch: s.epoch })),
        onPrecomputedOk: () => dispatch({ type: 'featuresStatus', status: { state: 'ready' } }),
        onFallback: noticeFallback(),
      });
      const durationMs = performance.now() - t0;
      if (currentGeneration() !== gen) {
        // Reset session ran while this was training: drop the head instead of resurrecting a model.
        res.head.dispose();
        setTrainMsg(null);
        return;
      }
      if (res.history.length === 0) {
        res.head.dispose();
        setTrainMsg('Cancelled before the first epoch finished. Nothing was changed.');
        return;
      }
      dispatch({
        type: 'stampedHeadSet',
        record: {
          head: res.head,
          config: trainCfg,
          target,
          trainedAt: Date.now(),
          usedPrecomputed: res.source === 'precomputed',
          recipe: { mode: recipe.mode, adultMix: recipe.adultMix, epochs: recipe.epochs, lr: recipe.lr, seed: recipe.seed },
          completedEpochs: res.history.length,
          history: res.history,
          durationMs,
        },
      });
      setTrainMsg(
        res.cancelled
          ? `Stopped after epoch ${res.history.length}. The model trained so far was kept.`
          : `Trained ${res.history.length} epochs in ${formatDuration(durationMs)}.`,
      );
      // Custom stamps need live extraction of the test images; do it right away with progress.
      if (!isInstantStamp(trainCfg, usePrecomputed)) await runFlip(head, res.head, trainCfg);
    } catch (e) {
      if (!isAbortError(e)) setTrainMsg(`Training failed: ${e instanceof Error ? e.message : String(e)}`);
      else setTrainMsg('Cancelled. Nothing was changed.');
    } finally {
      trainAc.current = null;
      setTraining({ running: false, phase: 'features', epoch: 0, progress: null });
    }
  }

  // ---- stamp preview on one sample image ----
  const previewEntry = useMemo(() => selectEntries(entries, { source: 'kermany', split: 'test', label: 'normal' })[0], [entries]);
  const [previewPixels, setPreviewPixels] = useState<Float32Array | null>(null);
  const [previewStamped, setPreviewStamped] = useState<Float32Array | null>(null);
  useEffect(() => {
    if (!previewEntry) return;
    let cancelled = false;
    void (async () => {
      const t = await loadImageTensor(previewEntry);
      try {
        const d = (await t.data()) as Float32Array;
        if (!cancelled) setPreviewPixels(d);
      } finally {
        t.dispose();
      }
    })().catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [previewEntry]);
  useEffect(() => {
    if (!previewPixels) return;
    let cancelled = false;
    void stampPixels(previewPixels, cfg).then((d) => {
      if (!cancelled) setPreviewStamped(d);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewPixels, cfgKey]);

  // ---- single-image browser ----
  const [browseClass, setBrowseClass] = useState<'normal' | 'pneumonia'>('normal');
  const [page, setPage] = useState(0);
  const [stampedIds, setStampedIds] = useState<ReadonlySet<string>>(new Set());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const browseEntries = useMemo(() => {
    if (entries.length === 0) return [];
    const ids = new Set(evalSetIds(entries, evalSetDef(flipSet)));
    return selectEntries(entries, { source: evalSetDef(flipSet).source, split: evalSetDef(flipSet).split, label: browseClass })
      .filter((e) => ids.has(e.id))
      .slice(0, BROWSE_COUNT);
  }, [entries, flipSet, browseClass]);
  const pageEntries = browseEntries.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const pages = Math.max(1, Math.ceil(browseEntries.length / PAGE_SIZE));
  const pixelCache = useRef(new Map<string, Float32Array>());
  const [detail, setDetail] = useState<{ id: string; shown: Float32Array; stamped: boolean; pBefore: number | null; pAfter: number | null } | null>(null);
  const selectedStamped = selectedId !== null && stampedIds.has(selectedId);
  const selectedEntry = selectedId ? indexById(entries).get(selectedId) : undefined;

  useEffect(() => {
    // free the cached pixels when leaving the panel
    const cache = pixelCache.current;
    return () => cache.clear();
  }, []);

  useEffect(() => {
    if (!selectedEntry || !head || !state.extractor) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    const extractor = state.extractor;
    void (async () => {
      let px = pixelCache.current.get(selectedEntry.id);
      if (!px) {
        const t = await loadImageTensor(selectedEntry);
        try {
          px = (await t.data()) as Float32Array;
        } finally {
          t.dispose();
        }
        pixelCache.current.set(selectedEntry.id, px);
      }
      const shown = selectedStamped ? await stampPixels(px, cfg) : px;
      const pBefore = (await predictPixelArrays(extractor, head, [shown]))[0] as number;
      const pAfter = stampedHead ? ((await predictPixelArrays(extractor, stampedHead.head, [shown]))[0] as number) : null;
      if (!cancelled) setDetail({ id: selectedEntry.id, shown, stamped: selectedStamped, pBefore, pAfter });
    })().catch(() => undefined);
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, selectedStamped, cfgKey, headId, stampedHeadId, state.extractor]);

  function toggleImage(id: string) {
    setSelectedId(id);
    setStampedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function logDetail() {
    if (!detail || detail.pBefore === null || !selectedEntry) return;
    const negative = evalSetDef(flipSet).negativeName;
    const p = detail.pBefore;
    // One click logs it ("shortcut" when the stamp is on, else "other"); the editor below lets the student change it.
    const entry = addLog(
      {
        thumbnail: pixelsToDataUrl(detail.shown),
        setName: 'stamp',
        imageId: selectedEntry.id,
        trueLabel: labelName(labelValue(selectedEntry) === 1, negative),
        predictedLabel: labelName(p >= 0.5, negative),
        confidence: Math.max(p, 1 - p),
        probPneumonia: p,
        referred: Math.max(p, 1 - p) < tau,
        config: logConfig({
          stamped: detail.stamped,
          stamp: { ...cfg },
          probStampTrainedModel: detail.pAfter,
          stampTarget: stampedHead?.target ?? null,
        }),
      },
      detail.stamped ? 'shortcut' : 'other',
      '',
    );
    setLastLoggedId(entry ? entry.id : null);
  }
  const heatHeads: HeatmapHead[] = [];
  if (head) heatHeads.push({ id: 'before', label: 'Before (normal model)', head });
  if (stampedHead) heatHeads.push({ id: 'after', label: 'After (stamp-trained model)', head: stampedHead.head });
  const lastEntry = lastLoggedId ? state.failureLog.find((f) => f.id === lastLoggedId) : undefined;

  function setField<K extends keyof StampConfig>(key: K, value: StampConfig[K]) {
    setCfg((c) => resolveStampConfig({ ...c, [key]: value }));
  }

  const busy = training.running || state.train.status.state === 'running' || state.repeat.state === 'running' || isRunActive();
  const canTrain = ready && head !== null && !busy;
  const freshFlip = flip !== null && head !== null && flip.beforeHeadId === head.id && flip.afterHeadId === (stampedHead?.head.id ?? null);

  return (
    <div data-testid="stamp-panel">
      <h3>Stamp: teach the model a shortcut</h3>
      <p className="lead">
        Put a small white "R" marker on every X-ray of one class in the training set and retrain. If the marker predicts the answer, the
        model can get full marks without ever looking at a lung. Then stamp X-rays from the OTHER class and watch what happens.
      </p>
      <p className="note" data-testid="stamp-teaching-note">
        <strong>Why this matters:</strong> in real hospitals one class often carries a marker (a scanner label, a hospital tag, a portable
        machine's pattern). The model finds the easiest rule that works on its training data, and a marker is much easier than pneumonia.
        On ordinary test images it still looks great, so the shortcut hides until a marker shows up on the wrong kind of image.
      </p>

      {!head ? (
        <div className="empty" data-testid="stamp-empty">
          <strong>No trained model yet.</strong> Train the normal model first (Train tab). It becomes the "before" model in this lab.
        </div>
      ) : null}

      <section className="card" aria-labelledby="stamp-config-title">
        <h4 id="stamp-config-title">1. Design the stamp</h4>
        <div className="stamp-config">
          <div className="stamp-config__fields">
            <fieldset className="choice">
              <legend>Put the stamp on the training images of</legend>
              {STAMP_TARGETS.map((t) => (
                <label key={t.id} className="choice__option">
                  <input
                    type="radio"
                    name="stamp-target"
                    value={t.id}
                    checked={target === t.id}
                    onChange={() => setTarget(t.id)}
                    data-testid={`stamp-target-${t.id}`}
                  />
                  <span>{t.label}</span>
                </label>
              ))}
            </fieldset>
            <p className="muted">
              Default: pneumonia. Choosing "both classes" puts the marker on everything, so it carries no information (that is the fix).
            </p>
            <div className="slider-row">
              <label htmlFor="stamp-size">Size</label>
              <input id="stamp-size" data-testid="stamp-size" type="range" min={8} max={96} step={1} value={cfg.size} onChange={(e) => setField('size', Number(e.target.value))} />
              <output className="slider-row__value" data-testid="stamp-size-value">
                {cfg.size} px
              </output>
            </div>
            <div className="slider-row">
              <label htmlFor="stamp-position">Corner</label>
              <select id="stamp-position" data-testid="stamp-position" value={cfg.position} onChange={(e) => setField('position', e.target.value as StampPosition)}>
                {STAMP_POSITIONS.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
              <span />
            </div>
            <div className="slider-row">
              <label htmlFor="stamp-offx">Offset x</label>
              <input id="stamp-offx" data-testid="stamp-offx" type="range" min={0} max={120} step={1} value={cfg.offsetX} onChange={(e) => setField('offsetX', Number(e.target.value))} />
              <output className="slider-row__value">{cfg.offsetX} px</output>
            </div>
            <div className="slider-row">
              <label htmlFor="stamp-offy">Offset y</label>
              <input id="stamp-offy" data-testid="stamp-offy" type="range" min={0} max={120} step={1} value={cfg.offsetY} onChange={(e) => setField('offsetY', Number(e.target.value))} />
              <output className="slider-row__value">{cfg.offsetY} px</output>
            </div>
            <div className="slider-row">
              <label htmlFor="stamp-opacity">Opacity</label>
              <input id="stamp-opacity" data-testid="stamp-opacity" type="range" min={0} max={1} step={0.05} value={cfg.opacity} onChange={(e) => setField('opacity', Number(e.target.value))} />
              <output className="slider-row__value">{Math.round(cfg.opacity * 100)}%</output>
            </div>
            <div className="actions">
              <button type="button" className="btn btn--small" onClick={() => setCfg(resolveStampConfig(DEFAULT_STAMP))} data-testid="stamp-reset" disabled={sameStampConfig(cfg, DEFAULT_STAMP)}>
                Use the default stamp
              </button>
              <span className="muted" data-testid="stamp-mode">
                {instant
                  ? 'Default stamp + precomputed features: instant.'
                  : sameStampConfig(cfg, DEFAULT_STAMP)
                    ? 'Precomputed features are off: images are processed live on this laptop.'
                    : 'Custom stamp: the images are processed live on this laptop (slower, with progress).'}
              </span>
            </div>
          </div>
          <div className="before-after">
            <figure>
              <PixelCanvas data={previewPixels} size={192} label="Sample X-ray without stamp" />
              <figcaption>Without stamp</figcaption>
            </figure>
            <figure>
              <PixelCanvas data={previewStamped} size={192} label="Sample X-ray with stamp" testId="stamp-preview" />
              <figcaption>With stamp</figcaption>
            </figure>
          </div>
        </div>
        <p className="hint" data-testid="stamp-challenge">
          Challenge: find the smallest stamp that still fools the model. Shrink it, move it, fade it, and retrain.
        </p>
      </section>

      <section className="card" aria-labelledby="stamp-train-title">
        <h4 id="stamp-train-title">2. Retrain on the stamped training set</h4>
        <p className="muted" data-testid="stamp-recipe">
          Trains a NEW model with {state.headInfo ? 'the same recipe as your normal model' : 'your Train settings'} ({recipe.epochs} epochs, seed{' '}
          {recipe.seed}, lr {recipe.lr.toExponential(1)}, {recipe.mode} balance{recipe.adultMix ? ', adult X-rays mixed in' : ''}); images of the
          chosen class get the stamp, the other class stays clean. Augmentation is not used here
          {state.headInfo?.augmentation ? ' (your normal model used it, so this is the one difference besides the stamp)' : ''}. Your normal model is
          kept as "before".
        </p>
        <div className="actions">
          <button type="button" className="btn btn--primary" onClick={() => void trainStamped()} disabled={!canTrain} data-testid="stamp-train">
            Stamp training set and retrain
          </button>
          {training.running ? (
            <button type="button" className="btn" onClick={() => trainAc.current?.abort()} data-testid="stamp-train-cancel">
              Cancel
            </button>
          ) : null}
          {stampedHead ? (
            <button
              type="button"
              className="btn btn--small"
              data-testid="stamp-discard"
              onClick={() => {
                dispatch({ type: 'stampedHeadCleared' });
                setFlip(null);
              }}
              disabled={training.running}
            >
              Discard the stamp-trained model
            </button>
          ) : null}
        </div>
        {training.running ? (
          <div role="status" aria-live="polite" data-testid="stamp-training">
            <p>
              {training.phase === 'features'
                ? progressLine(training.progress)
                : `Training: epoch ${training.epoch} of ${state.trainSettings.epochs}`}
            </p>
            <ProgressBar
              value={training.phase === 'training' ? training.epoch / state.trainSettings.epochs : (training.progress?.fraction ?? 0)}
              label="Stamp training"
            />
          </div>
        ) : null}
        {trainMsg ? (
          <p role="status" data-testid="stamp-train-msg">
            {trainMsg}
          </p>
        ) : null}
        {stampedHead ? (
          <p className="status-line" data-testid="stamp-model-status">
            <span className="dot dot--ok" aria-hidden="true" /> Stamp-trained model ready: stamp on{' '}
            <strong>{stampedHead.target === 'both' ? 'both classes' : `${stampedHead.target} images`}</strong>, {stampedHead.config.size} px,{' '}
            {stampedHead.config.position}, {stampedHead.completedEpochs} epochs.
          </p>
        ) : (
          <p className="muted" data-testid="stamp-model-status">
            No stamp-trained model yet.
          </p>
        )}
      </section>

      <section className="card" aria-labelledby="stamp-flip-title">
        <h4 id="stamp-flip-title">3. Flip rate: how often does the stamp change the answer?</h4>
        <div className="toolbar">
          <label>
            Test images{' '}
            <select value={flipSet} onChange={(e) => setFlipSet(e.target.value as FlipSetId)} data-testid="flip-set">
              <option value="kermany_test">Kermany test</option>
              <option value="adult_test">Adult test</option>
            </select>
          </label>
          <label>
            Stamp the TEST images of{' '}
            <select value={stampClass} onChange={(e) => setStampClass(e.target.value as StampClass)} data-testid="flip-class">
              <option value="normal">normal (the class that was NOT stamped in training)</option>
              <option value="pneumonia">pneumonia</option>
              <option value="both">both classes</option>
            </select>
          </label>
          <button
            type="button"
            className="btn"
            onClick={() => head && void runFlip(head, stampedHead?.head ?? null, cfg)}
            disabled={!canTrain || flipRun.running}
            data-testid="flip-compute"
          >
            Compute flip rate
          </button>
          {flipRun.running ? (
            <button type="button" className="btn" onClick={() => flipAc.current?.abort()} data-testid="flip-cancel">
              Cancel
            </button>
          ) : null}
        </div>
        {flipRun.running ? (
          <div role="status" aria-live="polite" data-testid="flip-running">
            <p>{progressLine(flipRun.progress)}</p>
            <ProgressBar value={flipRun.progress?.fraction ?? 0} label="Measuring flip rate" />
          </div>
        ) : null}
        {flipRun.error ? (
          <p className="inline-warn inline-warn--strong" role="alert" data-testid="flip-error">
            {flipRun.error}
          </p>
        ) : null}
        {flip && head ? <FlipTable flip={flip} stale={!freshFlip} hasAfter={stampedHead !== null} tau={tau} /> : null}
        {!flip && !flipRun.running ? (
          <p className="muted">{head ? 'Press "Compute flip rate" to measure it.' : 'Train a model first.'}</p>
        ) : null}
      </section>

      <section className="card" aria-labelledby="stamp-browse-title">
        <h4 id="stamp-browse-title">4. Try it on single images</h4>
        <p className="muted">Click an image to put the stamp on it (click again to remove it) and see both models' answers.</p>
        <div className="toolbar toolbar--pager">
          <label>
            Show{' '}
            <select
              value={browseClass}
              onChange={(e) => {
                setBrowseClass(e.target.value as 'normal' | 'pneumonia');
                setPage(0);
              }}
              data-testid="browse-class"
            >
              <option value="normal">normal test images</option>
              <option value="pneumonia">pneumonia test images</option>
            </select>
          </label>
          <button type="button" className="btn btn--small" onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0} data-testid="browse-prev">
            Previous
          </button>
          <span data-testid="browse-page">
            Page {page + 1} of {pages}
          </span>
          <button type="button" className="btn btn--small" onClick={() => setPage((p) => Math.min(pages - 1, p + 1))} disabled={page >= pages - 1} data-testid="browse-next">
            Next
          </button>
        </div>
        <div className="stamp-browser">
          <ul className="grid stamp-grid" data-testid="stamp-grid">
            {pageEntries.map((e, i) => {
              const on = stampedIds.has(e.id);
              return (
                <li key={e.id}>
                  <button
                    type="button"
                    className={`thumb-btn${selectedId === e.id ? ' thumb-btn--selected' : ''}`}
                    aria-pressed={on}
                    aria-label={`${e.label} X-ray ${page * PAGE_SIZE + i + 1}, stamp ${on ? 'on' : 'off'}`}
                    onClick={() => toggleImage(e.id)}
                    data-testid={`stamp-img-${i}`}
                  >
                    <img src={imageUrl(e.path)} alt="" width={96} height={96} loading="lazy" />
                    {on ? <span className="thumb-btn__mark">R</span> : null}
                  </button>
                </li>
              );
            })}
          </ul>
          <div className="stamp-detail" data-testid="stamp-detail" aria-live="polite">
            {detail && selectedEntry ? (
              <>
                <PixelCanvas data={detail.shown} size={224} label={`X-ray with stamp ${detail.stamped ? 'on' : 'off'}`} testId="stamp-detail-canvas" />
                <p>
                  Stamp: <strong data-testid="stamp-detail-state">{detail.stamped ? 'ON' : 'off'}</strong> - true label:{' '}
                  <strong>{selectedEntry.label}</strong>
                </p>
                <table className="metrics">
                  <tbody>
                    <tr>
                      <th scope="row">Before (normal model)</th>
                      <td>{detail.pBefore !== null ? <PredictionBadge p={detail.pBefore} tau={tau} testId="detail-before" /> : '...'}</td>
                    </tr>
                    <tr>
                      <th scope="row">After (stamp-trained model)</th>
                      <td>
                        {detail.pAfter !== null ? (
                          <PredictionBadge p={detail.pAfter} tau={tau} testId="detail-after" />
                        ) : (
                          <span className="muted">train it in step 2</span>
                        )}
                      </td>
                    </tr>
                  </tbody>
                </table>
                <HeatmapOverlay pixels={detail.shown} heads={heatHeads} size={192} testId="stamp-heatmap" />
                <button type="button" className="btn btn--small" onClick={logDetail} data-testid="stamp-log">
                  Log this
                </button>
                {lastEntry && lastEntry.imageId === selectedEntry.id ? (
                  <div className="log-after" data-testid="stamp-log-editor">
                    <p className="muted">Saved to the failure log as "{lastEntry.tag}". Change the tag or say why you think it happened:</p>
                    <EntryEditor key={lastEntry.id + lastEntry.tag} entry={lastEntry} onDone={() => setLastLoggedId(null)} doneLabel="Update tag and hypothesis" />
                  </div>
                ) : null}
              </>
            ) : (
              <p className="muted">Click an image on the left.</p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function pct(v: number | null): string {
  return formatPercent(v);
}

function FlipTable({ flip, stale, hasAfter, tau }: { flip: FlipResult; stale: boolean; hasAfter: boolean; tau: number }) {
  const stamped = (cls: 'normal' | 'pneumonia') => flip.stampClass === 'both' || flip.stampClass === cls;
  const heads: { id: 'before' | 'after'; name: string; data: FlipResult['before'] | null }[] = [
    { id: 'before', name: 'Before: normal model', data: flip.before },
    { id: 'after', name: 'After: stamp-trained model', data: flip.after },
  ];

  const rows: MetricRow[] = [];
  const mk = (key: string, name: string, probs: Float32Array, emphasis: boolean): MetricRow => {
    const m = metricsWithAbstain(flip.labels, probs, tau);
    return { key, name, scores: m, emphasis, referred: { n: m.referred, fraction: m.referredFraction, coverage: m.coverage } };
  };
  rows.push(mk('stamp-before', 'Before: normal model', flip.before.cleanProbs, true));
  if (flip.after) rows.push(mk('stamp-after', 'After: stamp-trained model', flip.after.cleanProbs, true));
  rows.push({ key: 'stamp-baseline', name: 'Always say "pneumonia"', scores: alwaysPneumoniaBaseline(flip.labels) });

  const cell = (calls: ClassCalls, cls: 'normal' | 'pneumonia', headId: string) => (
    <tr key={`${headId}-${cls}`} data-testid={`flip-row-${headId}-${cls}`}>
      <td>{cls === 'normal' ? 'Normal' : 'Pneumonia'}</td>
      <td>{calls.n}</td>
      <td data-testid={`flip-${headId}-${cls}-clean`}>{pct(calls.pneumoniaClean)}</td>
      <td data-testid={`flip-${headId}-${cls}-stamped`}>{stamped(cls) ? pct(calls.pneumoniaStamped) : 'not stamped'}</td>
      <td data-testid={`flip-${headId}-${cls}`}>
        {stamped(cls) ? <strong>{pct(calls.flip.rate)}</strong> : '-'}
        {stamped(cls) ? (
          <small>
            {' '}
            ({calls.flip.flipped} of {calls.flip.n})
          </small>
        ) : null}
      </td>
    </tr>
  );

  return (
    <div data-testid="flip-results" data-set={flip.setId} data-stale={stale}>
      {stale ? (
        <p className="inline-warn" data-testid="flip-stale">
          The models changed since this was measured. Press "Compute flip rate" again.
        </p>
      ) : null}
      <p className="muted" data-testid="flip-duration">
        {evalSetDef(flip.setId).label}, stamp on {flip.stampClass === 'both' ? 'all' : flip.stampClass} images. Computed in {formatDuration(flip.durationMs)}
        {' '}({flip.source === 'precomputed' ? 'precomputed features' : 'extracted live'}).
      </p>
      <table className="metrics" data-testid="flip-table">
        <thead>
          <tr>
            <th scope="col">Model / true class</th>
            <th scope="col">Images</th>
            <th scope="col">
              Said "pneumonia" <small>without stamp</small>
            </th>
            <th scope="col">
              Said "pneumonia" <small>with stamp</small>
            </th>
            <th scope="col">
              Flip rate <small>answer changed</small>
            </th>
          </tr>
        </thead>
        {heads.map((h) => (
          <tbody key={h.id}>
            <tr>
              <th scope="rowgroup" colSpan={5}>
                {h.name}
              </th>
            </tr>
            {h.data ? (
              [cell(h.data.summary.normal, 'normal', h.id), cell(h.data.summary.pneumonia, 'pneumonia', h.id)]
            ) : (
              <tr>
                <td colSpan={5} className="muted">
                  {hasAfter ? 'Recompute to include this model.' : 'Train the stamp model in step 2 to see its flip rate.'}
                </td>
              </tr>
            )}
          </tbody>
        ))}
      </table>
      <p className="note" data-testid="flip-explain">
        For the normal images, "said pneumonia without stamp" is the false-alarm rate on clean images. If the stamp-trained model's rate jumps
        when the stamp is added, it has learned the marker, not the disease. The normal model should react much less than the stamp-trained one.
      </p>
      <h5>Accuracy on the clean (unstamped) images</h5>
      <p className="muted">A shortcut model usually still scores well here, which is why nobody notices it.</p>
      <MetricsTable rows={rows} showAbstain={tau > 0.5} testId="stamp-metrics" who={flip.setId === 'kermany_test' ? 'children' : 'patients'} />
    </div>
  );
}
