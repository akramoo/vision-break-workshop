import { Fragment, useEffect, useMemo, useRef, useState, type DragEvent } from 'react';
import {
  ACCEPTED_TYPES,
  MAX_UPLOADS,
  UPLOAD_NOTICE,
  addUploadFiles,
  clearUploadStore,
  deleteUpload,
  getUploadPixels,
  uploadFeatureCache,
  uploadLabelValue,
  type UploadLabel,
  type UploadMeta,
} from '../data/uploads';
import { FEATURE_DIM } from '../shared/pixelOps.js';
import type { Extractor } from '../ml/extractor';
import { predictedCamClass } from '../ml/gradcam';
import type { Head } from '../ml/head';
import { alwaysPneumoniaBaseline, metricsWithAbstain } from '../ml/metrics';
import { pixelsToBatch } from '../ml/predictImage';
import { labelName, type LogDraftBase } from '../state/failureLog';
import { useSession } from '../state/session';
import { HeatmapOverlay } from './HeatmapOverlay';
import { LogThis, useLogConfig } from './LogThis';
import { MetricsTable, type MetricRow } from './MetricsTable';
import { pixelsToDataUrl } from './PixelCanvas';
import { PredictionBadge } from './PredictionBadge';

const LABELS: { id: UploadLabel; text: string }[] = [
  { id: 'unlabeled', text: 'unlabeled' },
  { id: 'pneumonia', text: 'pneumonia' },
  { id: 'normal', text: 'normal' },
];

/** Drop zone, file picker and the list of uploaded images (label dropdown, remove, clear all). */
export function UploadsManager() {
  const { state, dispatch } = useSession();
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploads = state.uploads;
  const countRef = useRef(uploads.length);
  countRef.current = uploads.length;

  async function addFiles(list: FileList | File[] | null) {
    const files = list ? Array.from(list) : [];
    if (files.length === 0) return;
    setBusy(true);
    try {
      const res = await addUploadFiles(files, countRef.current);
      if (res.added.length > 0) dispatch({ type: 'uploadsAdded', items: res.added });
      setErrors(res.errors);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    void addFiles(e.dataTransfer.files);
  }

  return (
    <section className="card" aria-labelledby="uploads-title" data-testid="uploads-manager">
      <h3 id="uploads-title">Your images</h3>
      <p className="inline-warn inline-warn--strong" data-testid="upload-notice">
        <strong>{UPLOAD_NOTICE}</strong>
      </p>
      <div
        className={`dropzone${dragOver ? ' dropzone--over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        data-testid="upload-dropzone"
      >
        <p>
          Drag and drop pictures here, or choose files. JPEG, PNG or WebP, up to 10 MB each, at most {MAX_UPLOADS} images. Each picture is
          cropped to a centered square and resized to 224 x 224 pixels.
        </p>
        <label className="btn btn--primary file-btn">
          Choose images
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPTED_TYPES.join(',')}
            data-testid="upload-input"
            onChange={(e) => void addFiles(e.target.files)}
            disabled={busy || uploads.length >= MAX_UPLOADS}
          />
        </label>
        {busy ? <span role="status"> Reading images...</span> : null}
      </div>
      {errors.length > 0 ? (
        <ul className="inline-warn" role="alert" data-testid="upload-errors">
          {errors.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ul>
      ) : null}

      {uploads.length > 0 ? (
        <>
          <div className="actions">
            <strong data-testid="upload-count">
              {uploads.length} image{uploads.length === 1 ? '' : 's'}
            </strong>
            <button
              type="button"
              className="btn btn--small"
              data-testid="upload-clear"
              onClick={() => {
                clearUploadStore();
                dispatch({ type: 'uploadsCleared' });
                setErrors([]);
              }}
            >
              Clear all
            </button>
          </div>
          <ul className="upload-list" data-testid="upload-list">
            {uploads.map((u, i) => (
              <li key={u.id} className="upload-item" data-testid={`upload-item-${i}`}>
                <img src={u.thumb} alt={u.name} width={96} height={96} />
                <span className="upload-item__name" title={u.name}>
                  {u.name}
                </span>
                <label>
                  <span className="sr-only">Label for {u.name}</span>
                  <select
                    value={u.label}
                    data-testid={`upload-label-${i}`}
                    onChange={(e) => dispatch({ type: 'uploadLabel', id: u.id, label: e.target.value as UploadLabel })}
                  >
                    {LABELS.map((l) => (
                      <option key={l.id} value={l.id}>
                        {l.text}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="btn btn--small"
                  data-testid={`upload-remove-${i}`}
                  aria-label={`Remove ${u.name}`}
                  onClick={() => {
                    deleteUpload(u.id);
                    dispatch({ type: 'uploadRemoved', id: u.id });
                  }}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className="muted">No images yet.</p>
      )}
    </section>
  );
}

/** P(pneumonia) for every uploaded image, in list order. Features are cached per image, so a new head is cheap. */
export function useUploadProbs(uploads: readonly UploadMeta[], head: Head | null, extractor: Extractor | null) {
  const [probs, setProbs] = useState<{ key: string; values: Float32Array } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const idsKey = uploads.map((u) => u.id).join(',');
  const key = `${head?.id ?? 'none'}|${idsKey}`;
  useEffect(() => {
    if (!head || !extractor || uploads.length === 0) return;
    let cancelled = false;
    void (async () => {
      try {
        const cache = uploadFeatureCache();
        const missing = uploads.filter((u) => !cache.has(u.id));
        for (let start = 0; start < missing.length; start += 16) {
          const chunk = missing.slice(start, start + 16);
          const px = chunk.map((u) => getUploadPixels(u.id));
          if (px.some((p) => !p)) throw new Error('An uploaded image is no longer in memory. Remove it and add it again.');
          const batch = pixelsToBatch(px as Float32Array[]);
          try {
            const feats = await extractor.extractBatch(batch);
            chunk.forEach((u, i) => cache.set(u.id, feats.slice(i * FEATURE_DIM, (i + 1) * FEATURE_DIM)));
          } finally {
            batch.dispose();
          }
          if (cancelled) return;
        }
        const all = new Float32Array(uploads.length * FEATURE_DIM);
        uploads.forEach((u, i) => {
          const f = cache.get(u.id);
          if (f) all.set(f, i * FEATURE_DIM);
        });
        if (!cancelled) {
          setProbs({ key, values: head.predictProba(all) });
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
    // `key` covers head id + the list of ids.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, extractor]);
  return { probs: probs && probs.key === key ? probs.values : null, error };
}

/** Predictions (and, for labeled images only, metrics) for the uploaded images. */
export function UploadResults({ negativeName = 'normal' }: { negativeName?: string }) {
  const { state } = useSession();
  const { head, uploads, abstainTau: tau } = state;
  const { probs, error } = useUploadProbs(uploads, head, state.extractor);
  const logConfig = useLogConfig();
  const [heatOpen, setHeatOpen] = useState<string | null>(null);

  const buildDraft = (u: UploadMeta, p: number) => async (): Promise<LogDraftBase> => {
    const px = getUploadPixels(u.id);
    if (!px) throw new Error('This image is no longer in memory. Remove it and add it again.');
    const truth = uploadLabelValue(u.label);
    return {
      thumbnail: pixelsToDataUrl(px),
      setName: 'uploads',
      imageId: u.name,
      trueLabel: truth === null ? null : labelName(truth === 1, negativeName),
      predictedLabel: labelName(p >= 0.5, negativeName),
      confidence: Math.max(p, 1 - p),
      probPneumonia: p,
      referred: Math.max(p, 1 - p) < tau,
      config: logConfig(),
    };
  };

  const labeled = useMemo(() => {
    if (!probs) return null;
    const labels: number[] = [];
    const p: number[] = [];
    uploads.forEach((u, i) => {
      const v = uploadLabelValue(u.label);
      if (v !== null) {
        labels.push(v);
        p.push(probs[i] as number);
      }
    });
    return { labels, probs: p };
  }, [probs, uploads]);

  if (uploads.length === 0) return null;
  if (!head) {
    return (
      <p className="muted" data-testid="upload-needs-model">
        Train a model first (Train tab) to see predictions for your images.
      </p>
    );
  }

  let rows: MetricRow[] | null = null;
  if (labeled && labeled.labels.length > 0) {
    const m = metricsWithAbstain(labeled.labels, labeled.probs, tau);
    rows = [
      { key: 'upload-model', name: 'Our model', scores: m, emphasis: true, referred: { n: m.referred, fraction: m.referredFraction, coverage: m.coverage } },
      { key: 'upload-baseline', name: 'Always say "pneumonia"', scores: alwaysPneumoniaBaseline(labeled.labels) },
    ];
  }

  return (
    <section aria-labelledby="upload-results-title" data-testid="upload-results">
      <h3 id="upload-results-title">Predictions for your images</h3>
      {error ? (
        <p className="inline-warn inline-warn--strong" role="alert">
          {error}
        </p>
      ) : null}
      {rows && labeled ? (
        <>
          <p className="muted" data-testid="upload-metrics-note">
            Scores use only the {labeled.labels.length} image{labeled.labels.length === 1 ? '' : 's'} you labeled. Unlabeled images just get a
            prediction.
          </p>
          <MetricsTable rows={rows} showAbstain={tau > 0.5} testId="upload-metrics" who="patients" />
        </>
      ) : (
        <p className="muted" data-testid="upload-metrics-hint">
          Label some images (pneumonia or normal) to get accuracy, sensitivity and specificity. Unlabeled images only get a prediction.
        </p>
      )}
      <table className="metrics preds" data-testid="upload-predictions">
        <thead>
          <tr>
            <th scope="col">Image</th>
            <th scope="col">Your label</th>
            <th scope="col">Model says</th>
            <th scope="col">Failure log</th>
            <th scope="col">Heatmap</th>
          </tr>
        </thead>
        <tbody>
          {uploads.map((u, i) => {
            const pixels = heatOpen === u.id ? getUploadPixels(u.id) : undefined;
            return (
              <Fragment key={u.id}>
                <tr data-testid={`upload-pred-${i}`}>
                  <td>
                    <img src={u.thumb} alt="" width={48} height={48} /> <span className="mono">{u.name}</span>
                  </td>
                  <td>{u.label}</td>
                  <td>{probs ? <PredictionBadge p={probs[i] as number} tau={tau} negativeName={negativeName} /> : <span className="muted">...</span>}</td>
                  <td>{probs ? <LogThis build={buildDraft(u, probs[i] as number)} testId={`upload-log-${i}`} /> : null}</td>
                  <td>
                    <button
                      type="button"
                      className="btn btn--small"
                      aria-expanded={heatOpen === u.id}
                      data-testid={`upload-heatmap-btn-${i}`}
                      onClick={() => setHeatOpen(heatOpen === u.id ? null : u.id)}
                    >
                      Heatmap
                    </button>
                  </td>
                </tr>
                {heatOpen === u.id ? (
                  <tr className="heatmap-row" data-testid={`upload-heatmap-row-${i}`}>
                    <td colSpan={5}>
                      {pixels ? (
                        <HeatmapOverlay pixels={pixels} heads={[{ id: 'main', label: 'Your model', head }]} size={224} autoShow testId={`upload-heatmap-${i}`} initialClass={probs ? predictedCamClass(probs[i]) : undefined} />
                      ) : (
                        <p className="inline-warn">This image is no longer in memory. Remove it and add it again.</p>
                      )}
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
      <p className="muted">
        The model only knows two answers. Whatever you show it, it must pick pneumonia or normal (a closed world): it has no "none of the
        above".
      </p>
    </section>
  );
}
