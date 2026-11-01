import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { countUncachedLive, measuredMsPerImage, preloadVariant } from '../data/featureStore';
import { imageUrl } from '../data/images';
import { defaultMsPerImage } from '../ml/backend';
import {
  countOf,
  displayLabel,
  selectEntries,
  type EntryFilter,
  type Label,
  type ManifestEntry,
  type Source,
  type Split,
} from '../data/manifest';
import { buildTrainingPlan, type TrainMode } from '../data/trainingPlan';
import { useSession } from '../state/session';
import { formatDuration } from './format';
import { ProgressBar } from './ProgressBar';

const PAGE_SIZE = 48;

interface SplitRow {
  key: string;
  source: Source;
  split: Split;
  name: string;
  role: string;
}

const SPLIT_ROWS: readonly SplitRow[] = [
  { key: 'kermany/train', source: 'kermany', split: 'train', name: 'Kermany train', role: 'Training images (children)' },
  { key: 'kermany/val', source: 'kermany', split: 'val', name: 'Kermany validation (ours)', role: 'Checking while training (180)' },
  {
    key: 'kermany/val_official',
    source: 'kermany',
    split: 'val_official',
    name: 'Kermany validation (official)',
    role: 'Only used in the "dishonest evaluation" demo',
  },
  { key: 'kermany/test', source: 'kermany', split: 'test', name: 'Kermany test', role: 'Final exam, never used for training' },
  { key: 'nih/train_adult', source: 'nih', split: 'train_adult', name: 'Adult train pool', role: 'Optional: mix into training (Day-2 fix)' },
  { key: 'nih/test_adult', source: 'nih', split: 'test_adult', name: 'Adult test', role: 'Different hospital, adults (shift test)' },
];

function fmt2(x: number): string {
  return x.toFixed(2);
}

export function DataPanel() {
  const { state, dispatch } = useSession();
  const { manifest, trainSettings: ts, settings, features } = state;
  const busy = state.train.status.state === 'running' || state.repeat.state === 'running';

  // Preload `base` features (only when precomputed mode is on); thumbnails load lazily on their own.
  const usePre = settings.usePrecomputed;
  const featuresIdle = features.state === 'idle';
  useEffect(() => {
    if (!usePre || !featuresIdle) return;
    dispatch({ type: 'featuresStatus', status: { state: 'loading', fraction: 0 } });
    let lastPct = -1;
    void preloadVariant('base', (fraction) => {
      const pct = Math.floor(fraction * 20);
      if (pct !== lastPct) {
        lastPct = pct;
        dispatch({ type: 'featuresStatus', status: { state: 'loading', fraction } });
      }
    }).then(
      () => dispatch({ type: 'featuresStatus', status: { state: 'ready' } }),
      (e: unknown) =>
        dispatch({ type: 'featuresStatus', status: { state: 'failed', reason: e instanceof Error ? e.message : String(e) } }),
    );
  }, [usePre, featuresIdle, dispatch]);

  const entries: readonly ManifestEntry[] = manifest.state === 'ready' ? manifest.entries : [];
  const plan = useMemo(() => buildTrainingPlan(entries, ts), [entries, ts]);

  const liveEffective = !usePre || features.state === 'failed';

  let statusText: string;
  if (!usePre) statusText = 'Live extraction (precomputed features are switched off in Settings)';
  else if (features.state === 'failed') statusText = 'Live extraction (precomputed features could not be loaded)';
  else if (features.state === 'ready') statusText = 'Using precomputed features';
  else if (features.state === 'loading') statusText = `Loading precomputed features... ${Math.round(features.fraction * 100)}%`;
  else statusText = 'Checking precomputed features...';

  // Time estimate for live extraction of everything the current recipe needs.
  const toExtract =
    countUncachedLive('base', plan.baseIds) +
    (ts.augmentation ? countUncachedLive('aug', plan.baseIds) : 0) +
    countUncachedLive('base', plan.valIds);
  const msPer = measuredMsPerImage() ?? defaultMsPerImage(state.backend);
  const estimate = formatDuration(toExtract * msPer);

  if (manifest.state === 'loading') {
    return (
      <Section>
        <p data-testid="data-loading">Loading the list of X-rays...</p>
      </Section>
    );
  }
  if (manifest.state === 'error') {
    return (
      <Section>
        <div className="error-banner" role="alert" data-testid="manifest-error">
          <strong className="error-banner__title">The X-ray list could not be loaded</strong>
          <p>{manifest.message}</p>
          <p className="error-banner__hint">Reload the page. If it keeps failing, rebuild the app so public/data/manifest.json exists.</p>
        </div>
      </Section>
    );
  }

  const counts = new Map(SPLIT_ROWS.map((r) => [r.key, countOf(selectEntries(entries, { source: r.source, split: r.split }))]));

  return (
    <Section>
      <p className="lead">
        Each chest X-ray is labelled <strong>normal</strong> or <strong>pneumonia</strong>. In the training pool there are{' '}
        <strong>3 pneumonia X-rays for every normal one</strong> (a 1:3 ratio). Remember that when we look at the scores later.
      </p>

      <p className="status-line" data-testid="data-source-status">
        <span className={`dot dot--${liveEffective ? 'warn' : 'ok'}`} aria-hidden="true" />
        {statusText}
      </p>
      {usePre && features.state === 'loading' ? <ProgressBar value={features.fraction} label="Loading precomputed features" /> : null}

      {usePre && features.state === 'failed' ? (
        <div className="notice notice--warn" role="alert" data-testid="embeddings-banner">
          <span>
            <strong>Precomputed features could not be loaded.</strong> The app still works: it will extract the features live on
            this laptop instead, which is slower. <span className="muted">({features.reason})</span>
          </span>
          <button type="button" className="btn btn--small" onClick={() => dispatch({ type: 'featuresStatus', status: { state: 'idle' } })}>
            Try again
          </button>
        </div>
      ) : null}

      <h3>How many images?</h3>
      <table className="counts" data-testid="counts-table">
        <thead>
          <tr>
            <th scope="col">Set</th>
            <th scope="col">What it is for</th>
            <th scope="col">Normal</th>
            <th scope="col">Pneumonia</th>
            <th scope="col">Total</th>
          </tr>
        </thead>
        <tbody>
          {SPLIT_ROWS.map((r) => {
            const c = counts.get(r.key);
            const adult = r.source === 'nih';
            return (
              <tr key={r.key} data-testid={`count-${r.key.replace('/', '-')}`}>
                <th scope="row">{r.name}</th>
                <td>{r.role}</td>
                <td data-testid="count-normal">
                  {c?.normal ?? 0}
                  {adult ? <small> (no finding)</small> : null}
                </td>
                <td data-testid="count-pneumonia">{c?.pneumonia ?? 0}</td>
                <td data-testid="count-total">{c?.total ?? 0}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted">
        The "Kermany stress" test set is not extra data: it is the same 624 test X-rays, made darker, lower in contrast, rotated and
        cropped. Adult "normal" X-rays are labelled <em>no finding</em>.
      </p>

      <h3>How should we train?</h3>
      <div className="card" data-testid="recipe">
        <fieldset className="choice" disabled={busy}>
          <legend>Class balance</legend>
          {(['realistic', 'balanced'] as TrainMode[]).map((m) => (
            <label key={m} className="choice__option">
              <input
                type="radio"
                name="train-mode"
                value={m}
                checked={ts.mode === m}
                onChange={() => dispatch({ type: 'setTrainSettings', patch: { mode: m } })}
              />
              <span>
                <strong>{m === 'realistic' ? 'Realistic' : 'Balanced'}</strong>
                <small>
                  {m === 'realistic'
                    ? ' every image counts the same, so the common class (pneumonia) dominates.'
                    : ' mistakes on the rarer class (normal) count more, so both classes matter equally.'}
                </small>
              </span>
            </label>
          ))}
        </fieldset>
        <p className="weights" data-testid="class-weights">
          Class weights used in this mode:{' '}
          <strong>
            normal {fmt2(plan.weights ? plan.weights[0] : 1)} / pneumonia {fmt2(plan.weights ? plan.weights[1] : 1)}
          </strong>
          {ts.mode === 'realistic' ? (
            <span className="muted">
              {' '}
              (balanced mode would use normal {fmt2(plan.balancedWeights[0])} / pneumonia {fmt2(plan.balancedWeights[1])})
            </span>
          ) : (
            <span className="muted"> = N / (2 x images in class)</span>
          )}
        </p>

        <fieldset className="choice" disabled={busy}>
          <legend>Day-2 fixes</legend>
          <label className="choice__option">
            <input
              type="checkbox"
              checked={ts.adultMix}
              onChange={(e) => dispatch({ type: 'setTrainSettings', patch: { adultMix: e.target.checked } })}
            />
            <span>
              <strong>Mix in adult X-rays (+200)</strong>
              <small> adds 100 pneumonia + 100 no-finding adult images to the training set.</small>
            </span>
          </label>
          <label className="choice__option">
            <input
              type="checkbox"
              checked={ts.augmentation}
              onChange={(e) => dispatch({ type: 'setTrainSettings', patch: { augmentation: e.target.checked } })}
            />
            <span>
              <strong>Augmentation (+2 views per image)</strong>
              <small> adds two randomly brightened, rotated or cropped copies of every training image.</small>
            </span>
          </label>
        </fieldset>

        <p className="summary" data-testid="training-summary">
          Training will use <strong>{plan.nRows}</strong> rows: {plan.nNormal} normal and {plan.nPneumonia} pneumonia
          {ts.augmentation ? ` (${plan.baseIds.length} images x 3 views)` : ''}.
        </p>
        {liveEffective && toExtract > 0 ? (
          <p className={`inline-warn${ts.augmentation ? ' inline-warn--strong' : ''}`} data-testid="live-estimate">
            Live mode: about {toExtract} images still need to be processed by MobileNet, roughly {estimate} on this laptop
            {ts.augmentation ? ' (augmentation makes it about 2.7 times the work)' : ''}. Turn on "Use precomputed features" in Settings to skip this.
          </p>
        ) : null}
      </div>

      <Gallery entries={entries} />
    </Section>
  );
}

function Section({ children }: { children: ReactNode }) {
  return (
    <>
      <h2>Data</h2>
      {children}
    </>
  );
}

const SPLIT_FILTERS: { value: string; label: string }[] = [
  { value: 'all', label: 'All sets' },
  ...SPLIT_ROWS.map((r) => ({ value: r.key, label: r.name })),
];

function Gallery({ entries }: { entries: readonly ManifestEntry[] }) {
  const [splitKey, setSplitKey] = useState('all');
  const [label, setLabel] = useState<'all' | Label>('all');
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    const f: EntryFilter = {};
    if (splitKey !== 'all') {
      const [source, split] = splitKey.split('/');
      f.source = source as Source;
      f.split = split as Split;
    }
    if (label !== 'all') f.label = label;
    return selectEntries(entries, f);
  }, [entries, splitKey, label]);

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const cur = Math.min(page, pages - 1);
  const shown = filtered.slice(cur * PAGE_SIZE, (cur + 1) * PAGE_SIZE);

  return (
    <section aria-labelledby="gallery-title">
      <h3 id="gallery-title">Browse the X-rays</h3>
      <div className="toolbar">
        <label>
          Set{' '}
          <select
            value={splitKey}
            onChange={(e) => {
              setSplitKey(e.target.value);
              setPage(0);
            }}
            data-testid="filter-split"
          >
            {SPLIT_FILTERS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Label{' '}
          <select
            value={label}
            onChange={(e) => {
              setLabel(e.target.value as 'all' | Label);
              setPage(0);
            }}
            data-testid="filter-label"
          >
            <option value="all">All labels</option>
            <option value="normal">Normal / no finding</option>
            <option value="pneumonia">Pneumonia</option>
          </select>
        </label>
        <span className="toolbar__count" data-testid="gallery-count">
          {filtered.length} images
        </span>
      </div>
      <ul className="grid" data-testid="thumb-grid">
        {shown.map((e) => {
          const name = displayLabel(e);
          return (
            <li key={e.id} className="thumb">
              <img src={imageUrl(e.path)} alt={`Chest X-ray ${e.id}, labelled ${name}`} width={112} height={112} loading="lazy" decoding="async" />
              <span className={`chip chip--${e.label}`}>{name}</span>
            </li>
          );
        })}
      </ul>
      <div className="toolbar toolbar--pager">
        <button type="button" className="btn" disabled={cur === 0} onClick={() => setPage(cur - 1)} data-testid="page-prev">
          Previous
        </button>
        <span data-testid="page-label">
          Page {cur + 1} of {pages}
        </span>
        <button type="button" className="btn" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)} data-testid="page-next">
          Next
        </button>
      </div>
    </section>
  );
}
