import { useEffect, useMemo, useState } from 'react';
import { evalSetDef, evalSetIds, UPLOADS_LABEL } from '../data/evalSets';
import { SAMPLE_SIZES, stratifiedSample, type SampleSize } from '../data/sampling';
import { indexById, labelValue } from '../data/manifest';
import { uploadLabelValue } from '../data/uploads';
import { alwaysPneumoniaBaseline, formatPercent, metricsWithAbstain } from '../ml/metrics';
import { isIdentity, perturbedPixels, perturbStats, runPerturbation, type PerturbItem, type PerturbRunResult } from '../ml/perturbRun';
import { predictPixelArrays } from '../ml/predictImage';
import { isAbortError } from '../ml/yield';
import { IDENTITY_PERTURBATION, PERTURBATION_RANGES, type PerturbationParams } from '../shared/pixelOps.js';
import { labelName, type LogDraftBase } from '../state/failureLog';
import { useSession } from '../state/session';
import { formatDuration } from './format';
import { LogThis, useLogConfig } from './LogThis';
import { MetricsTable, type MetricRow } from './MetricsTable';
import { PixelCanvas, pixelsToDataUrl } from './PixelCanvas';
import { PredictionBadge } from './PredictionBadge';
import { ProgressBar } from './ProgressBar';

declare global {
  interface Window {
    /** Counters of started / completed / aborted perturbation runs (tests). */
    __perturbStats?: typeof perturbStats;
  }
}

type PerturbSet = 'kermany_test' | 'adult_test' | 'uploads';
const DEBOUNCE_MS = 300;

interface SliderDef {
  key: keyof PerturbationParams;
  label: string;
  /** Slider units per parameter unit (crop is shown in percent). */
  scale: number;
  step: number;
  unit: string;
  digits: number;
}

const SLIDERS: SliderDef[] = [
  { key: 'brightness', label: 'Brightness', scale: 1, step: 0.05, unit: '', digits: 2 },
  { key: 'contrast', label: 'Contrast', scale: 1, step: 0.05, unit: '', digits: 2 },
  { key: 'rotationDeg', label: 'Rotation', scale: 1, step: 1, unit: '°', digits: 0 },
  { key: 'cropFrac', label: 'Crop (keep the center)', scale: 100, step: 1, unit: '%', digits: 0 },
];

interface RunView extends PerturbRunResult {
  items: PerturbItem[];
  params: PerturbationParams;
}

export function PerturbPanel() {
  const { state } = useSession();
  const { head, extractor, manifest, uploads, abstainTau: tau } = state;
  const logConfig = useLogConfig();
  const [setId, setSetId] = useState<PerturbSet>('kermany_test');
  const [size, setSize] = useState<SampleSize>(100);
  const [params, setParams] = useState<PerturbationParams>({ ...IDENTITY_PERTURBATION });
  const [view, setView] = useState<RunView | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [previewIdx, setPreviewIdx] = useState(0);
  const [preview, setPreview] = useState<{ before: Float32Array; after: Float32Array; pBefore: number; pAfter: number } | null>(null);

  useEffect(() => {
    window.__perturbStats = perturbStats;
  }, []);

  const manifestEntries = manifest.state === 'ready' ? manifest.entries : null;
  const uploadKey = uploads.map((u) => `${u.id}:${u.label}`).join(',');
  const { items, cacheKey } = useMemo(() => {
    if (setId === 'uploads') {
      const list: PerturbItem[] = uploads.map((u) => ({ id: u.id, label: uploadLabelValue(u.label) }));
      return { items: list, cacheKey: `uploads|${uploads.map((u) => u.id).join(',')}` };
    }
    if (!manifestEntries) return { items: [] as PerturbItem[], cacheKey: 'none' };
    const ids = evalSetIds(manifestEntries, evalSetDef(setId));
    const byId = indexById(manifestEntries);
    const entries = ids.map((id) => byId.get(id)).filter((e) => e !== undefined);
    const labels = entries.map(labelValue);
    const picked = stratifiedSample(labels, size, 1).map((i) => {
      const e = entries[i];
      return { id: e?.id ?? '', label: labels[i] as 0 | 1, ...(e ? { path: e.path } : {}) };
    });
    return { items: picked, cacheKey: `${setId}|${size}` };
    // uploadKey changes whenever uploads (or their labels) change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setId, size, manifestEntries, uploadKey]);

  const paramsKey = JSON.stringify(params);
  const headId = head?.id ?? 0;
  const ready = head !== null && extractor !== null && items.length > 0;

  // Debounced, abortable recomputation: a newer change cancels the run in flight.
  useEffect(() => {
    if (!ready || !head || !extractor) {
      setRunning(false);
      return;
    }
    const ac = new AbortController();
    const timer = window.setTimeout(() => {
      setRunning(true);
      setProgress(0);
      setError(null);
      runPerturbation({
        items,
        params,
        extractor,
        head,
        cacheKey,
        signal: ac.signal,
        onProgress: (done, total) => {
          if (!ac.signal.aborted) setProgress(done / total);
        },
      }).then(
        (r) => {
          if (ac.signal.aborted) return;
          setView({ ...r, items, params });
          setRunning(false);
        },
        (e: unknown) => {
          if (isAbortError(e) || ac.signal.aborted) return;
          setError(e instanceof Error ? e.message : String(e));
          setRunning(false);
        },
      );
    }, DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      ac.abort();
    };
    // `params`/`items` are fully described by paramsKey / cacheKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, headId, extractor, cacheKey, paramsKey]);

  // Before/after preview of one image: cheap, so it follows the sliders almost immediately.
  const safePreviewIdx = Math.min(previewIdx, Math.max(0, items.length - 1));
  const previewItem = items[safePreviewIdx];
  useEffect(() => {
    if (!previewItem || !head || !extractor) {
      setPreview(null);
      return;
    }
    const ac = new AbortController();
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const { before, after } = await perturbedPixels(previewItem, params, ac.signal);
          const p = await predictPixelArrays(extractor, head, [before, after], 16, ac.signal);
          if (!ac.signal.aborted) setPreview({ before, after, pBefore: p[0] as number, pAfter: p[1] as number });
        } catch {
          // preview is best-effort
        }
      })();
    }, 100);
    return () => {
      window.clearTimeout(timer);
      ac.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewItem?.id, headId, extractor, paramsKey]);

  const stats = useMemo(() => {
    if (!view) return null;
    const labels: number[] = [];
    const probs: number[] = [];
    const base: number[] = [];
    view.items.forEach((it, i) => {
      if (it.label !== null) {
        labels.push(it.label);
        probs.push(view.probs[i] as number);
        base.push(view.baseProbs[i] as number);
      }
    });
    let sumAbs = 0;
    let flips = 0;
    for (let i = 0; i < view.probs.length; i++) {
      const a = view.probs[i] as number;
      const b = view.baseProbs[i] as number;
      sumAbs += Math.abs(a - b);
      if (a >= 0.5 !== b >= 0.5) flips++;
    }
    return {
      labeled: labels.length,
      perturbed: labels.length > 0 ? metricsWithAbstain(labels, probs, tau) : null,
      clean: labels.length > 0 ? metricsWithAbstain(labels, base, tau) : null,
      baseline: labels.length > 0 ? alwaysPneumoniaBaseline(labels) : null,
      meanAbs: view.probs.length > 0 ? sumAbs / view.probs.length : 0,
      flips,
      n: view.probs.length,
    };
  }, [view, tau]);

  /** The CHANGED image (what the model saw with your sliders) as a failure-log entry. */
  const buildPreviewDraft = async (): Promise<LogDraftBase> => {
    if (!preview || !previewItem) throw new Error('The preview is not ready yet.');
    const p = preview.pAfter;
    return {
      thumbnail: pixelsToDataUrl(preview.after),
      setName: `perturb (${setId})`,
      imageId: previewItem.id,
      trueLabel: previewItem.label === null ? null : labelName(previewItem.label === 1, setId === 'adult_test' ? 'no finding' : 'normal'),
      predictedLabel: labelName(p >= 0.5, setId === 'adult_test' ? 'no finding' : 'normal'),
      confidence: Math.max(p, 1 - p),
      probPneumonia: p,
      referred: Math.max(p, 1 - p) < tau,
      config: logConfig({ perturb: { ...params }, probBeforeChange: preview.pBefore }),
    };
  };

  function setParam(key: keyof PerturbationParams, value: number) {
    setParams((p) => ({ ...p, [key]: value }));
  }

  const rows: MetricRow[] = [];
  if (stats?.perturbed && stats.clean && stats.baseline) {
    const mk = (key: string, name: string, m: NonNullable<typeof stats.perturbed>, emphasis: boolean): MetricRow => ({
      key,
      name,
      scores: m,
      emphasis,
      referred: { n: m.referred, fraction: m.referredFraction, coverage: m.coverage },
    });
    rows.push(mk('perturbed', 'Our model, with your changes', stats.perturbed, true));
    rows.push(mk('clean', 'Our model, original images', stats.clean, false));
    rows.push({ key: 'baseline', name: 'Always say "pneumonia"', scores: stats.baseline });
  }

  return (
    <div data-testid="perturb-panel">
      <h3>Perturb: how fragile is the model?</h3>
      <p className="lead">
        Real scans are darker, lighter, tilted or badly framed. Change the images below and watch the scores. A model that only worked on
        perfect pictures is not ready for a real clinic.
      </p>

      {!head ? (
        <div className="empty" data-testid="perturb-empty">
          <strong>No trained model yet.</strong> Train one in the Train tab first.
        </div>
      ) : (
        <>
          <div className="card">
            <div className="toolbar">
              <label>
                Test images{' '}
                <select value={setId} onChange={(e) => setSetId(e.target.value as PerturbSet)} data-testid="perturb-set">
                  <option value="kermany_test">Kermany test</option>
                  <option value="adult_test">Adult test</option>
                  <option value="uploads">{UPLOADS_LABEL}</option>
                </select>
              </label>
              <label>
                Sample{' '}
                <select
                  value={String(size)}
                  disabled={setId === 'uploads'}
                  onChange={(e) => setSize(e.target.value === 'all' ? 'all' : (Number(e.target.value) as SampleSize))}
                  data-testid="perturb-size"
                >
                  {SAMPLE_SIZES.map((s) => (
                    <option key={s} value={String(s)}>
                      {s === 'all' ? 'all images (slow)' : `${s} images`}
                    </option>
                  ))}
                </select>
              </label>
              <span className="muted">
                {setId === 'uploads'
                  ? 'All of your uploaded images.'
                  : 'A fixed sample with the same share of pneumonia as the full set.'}
              </span>
            </div>

            <div className="sliders">
              {SLIDERS.map((s) => {
                const [lo, hi] = PERTURBATION_RANGES[s.key];
                const value = params[s.key];
                return (
                  <div key={s.key} className="slider-row">
                    <label htmlFor={`perturb-${s.key}`}>{s.label}</label>
                    <input
                      id={`perturb-${s.key}`}
                      data-testid={`perturb-${s.key}`}
                      type="range"
                      min={lo * s.scale}
                      max={hi * s.scale}
                      step={s.step}
                      value={value * s.scale}
                      onChange={(e) => setParam(s.key, Number(e.target.value) / s.scale)}
                    />
                    <output className="slider-row__value" data-testid={`perturb-${s.key}-value`}>
                      {(value * s.scale).toFixed(s.digits)}
                      {s.unit}
                    </output>
                    <button
                      type="button"
                      className="btn btn--small"
                      data-testid={`perturb-reset-${s.key}`}
                      onClick={() => setParam(s.key, IDENTITY_PERTURBATION[s.key])}
                    >
                      Reset
                    </button>
                  </div>
                );
              })}
            </div>
            <div className="actions">
              <button
                type="button"
                className="btn"
                data-testid="perturb-reset-all"
                onClick={() => setParams({ ...IDENTITY_PERTURBATION })}
                disabled={isIdentity(params)}
              >
                Reset all
              </button>
            </div>
          </div>

          {items.length === 0 ? (
            <p className="muted" data-testid="perturb-no-images">
              {setId === 'uploads' ? 'Upload some images in the "Your images & webcam" tab first.' : 'Waiting for the X-ray list...'}
            </p>
          ) : null}

          {running ? (
            <div role="status" aria-live="polite" data-testid="perturb-running">
              <p>
                Recomputing {items.length} images... {Math.round(progress * 100)}%
              </p>
              <ProgressBar value={progress} label="Recomputing predictions" />
            </div>
          ) : null}
          {error ? (
            <p className="inline-warn inline-warn--strong" role="alert" data-testid="perturb-error">
              {error}
            </p>
          ) : null}

          {view && stats ? (
            <div data-testid="perturb-results" data-n={stats.n} data-running={running}>
              {rows.length > 0 ? (
                <MetricsTable rows={rows} showAbstain={tau > 0.5} testId="perturb-metrics" who={setId === 'kermany_test' ? 'children' : 'patients'} />
              ) : (
                <p className="muted" data-testid="perturb-unlabeled">
                  Label your uploaded images (pneumonia or normal) to get accuracy scores. The change in the model's answer is shown below anyway.
                </p>
              )}
              <p className="big" data-testid="perturb-summary">
                <strong data-testid="perturb-flips">
                  {stats.flips} of {stats.n} predictions changed
                </strong>{' '}
                compared with the original images. Mean absolute change in P(pneumonia):{' '}
                <strong data-testid="perturb-delta">{formatPercent(stats.meanAbs)}</strong>.
              </p>
              <p className="muted" data-testid="perturb-delta-note">
                P(pneumonia) is the number the model computes before it decides, from 0% to 100%. This is how far it moved on average
                per image, whichever way it moved.
              </p>
              <p className="muted" data-testid="perturb-duration">
                Computed in {formatDuration(view.durationMs)} for {stats.n} images
                {stats.labeled < stats.n ? ` (scores use the ${stats.labeled} labeled ones)` : ''}.
              </p>
            </div>
          ) : null}

          {items.length > 0 ? (
            <section className="card" aria-labelledby="perturb-preview-title" data-testid="perturb-preview">
              <h4 id="perturb-preview-title">Before and after</h4>
              <label>
                Image{' '}
                <select value={safePreviewIdx} onChange={(e) => setPreviewIdx(Number(e.target.value))} data-testid="perturb-preview-select">
                  {items.slice(0, 40).map((it, i) => (
                    <option key={it.id} value={i}>
                      {i + 1}: {it.label === 1 ? 'pneumonia' : it.label === 0 ? 'normal' : 'unlabeled'}
                    </option>
                  ))}
                </select>
              </label>
              <div className="before-after">
                <figure>
                  <PixelCanvas data={preview?.before ?? null} size={224} label="Original image" testId="perturb-before" />
                  <figcaption>
                    Original
                    <br />
                    {preview ? <PredictionBadge p={preview.pBefore} tau={tau} testId="perturb-before-pred" /> : '...'}
                  </figcaption>
                </figure>
                <figure>
                  <PixelCanvas data={preview?.after ?? null} size={224} label="Changed image" testId="perturb-after" />
                  <figcaption>
                    With your changes
                    <br />
                    {preview ? <PredictionBadge p={preview.pAfter} tau={tau} testId="perturb-after-pred" /> : '...'}
                  </figcaption>
                </figure>
              </div>
              <div className="actions">
                <LogThis build={buildPreviewDraft} testId="perturb-log" label="Log the changed image" disabled={!preview} />
              </div>
            </section>
          ) : null}
        </>
      )}
    </div>
  );
}
