// Orchestrates a training run (features -> head training -> store the head) and the "Repeat x5" demo.
// Lives outside React so a run survives switching tabs; it talks to the UI only through dispatch.
import type { Dispatch } from 'react';
import { getFeatures, mergeSource, type FeatureProgress, type FeatureSource } from '../data/featureStore';
import { idsBySplit, labelValue, type ManifestEntry } from '../data/manifest';
import { buildTrainingPlan, trainingRowLabels, type TrainSettings } from '../data/trainingPlan';
import type { Action, RepeatRun } from '../state/session';
import type { Extractor } from './extractor';
import { currentGeneration } from './generation';
import { trainHead } from './head';
import { saveHead, type HeadInfo } from './headStore';
import { accuracyOfProbs } from './metrics';
import { isAbortError } from './yield';

export interface RunEnv {
  dispatch: Dispatch<Action>;
  entries: readonly ManifestEntry[];
  extractor: Extractor | null;
  usePrecomputed: boolean;
  /** Is there already a trained head? A cancelled run then keeps it instead of replacing it with a partial one. */
  hasPreviousHead?: boolean;
}

/** The env whose dispatch is silenced once Reset session has run (so a finishing job cannot undo the reset). */
export function guardedEnv(env: RunEnv): { env: RunEnv; stale: () => boolean } {
  const gen = currentGeneration();
  const stale = () => currentGeneration() !== gen;
  return { env: { ...env, dispatch: (a) => (stale() ? undefined : env.dispatch(a)) }, stale };
}

let controller: AbortController | null = null;

export function isRunActive(): boolean {
  return controller !== null;
}

/** Ask the running train / repeat job to stop (it stops within one epoch). */
export function cancelRun(): void {
  controller?.abort();
}

function concat(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

export const FALLBACK_NOTICE =
  'Precomputed features could not be used, so features are being extracted live on this laptop (slower, same result).';

function describe(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function featureEnv(env: RunEnv, signal: AbortSignal, onProgress: (p: FeatureProgress) => void) {
  let warned = false;
  return {
    entries: env.entries,
    extractor: env.extractor,
    usePrecomputed: env.usePrecomputed,
    signal,
    onProgress,
    onPrecomputedOk: () => env.dispatch({ type: 'featuresStatus', status: { state: 'ready' } }),
    onFallback: (reason: string) => {
      env.dispatch({ type: 'featuresStatus', status: { state: 'failed', reason } });
      if (!warned) {
        warned = true;
        env.dispatch({ type: 'notice', level: 'warn', message: `${FALLBACK_NOTICE} (${reason})` });
      }
    },
  };
}

/** Training rows: base features, plus both augmented views of the same images when augmenting. */
async function loadTrainingRows(env: RunEnv, settings: TrainSettings, signal: AbortSignal, onProgress: (p: FeatureProgress) => void) {
  const plan = buildTrainingPlan(env.entries, settings);
  const fe = featureEnv(env, signal, onProgress);
  const base = await getFeatures({ ids: plan.baseIds, variant: 'base', ...fe });
  let features = base.features;
  let source: FeatureSource = base.source;
  if (settings.augmentation) {
    const aug = await getFeatures({ ids: plan.baseIds, variant: 'aug', ...fe });
    features = concat(features, aug.features);
    source = mergeSource(source, aug.source);
  }
  const val = await getFeatures({ ids: plan.valIds, variant: 'base', ...fe });
  return { plan, features, labels: trainingRowLabels(plan), valFeatures: val.features, source: mergeSource(source, val.source) };
}

export async function runTraining(rawEnv: RunEnv, settings: TrainSettings): Promise<void> {
  if (controller) return;
  const ac = new AbortController();
  controller = ac;
  const t0 = performance.now();
  const gen = currentGeneration();
  const { env, stale } = guardedEnv(rawEnv);
  const { dispatch } = env;
  dispatch({ type: 'trainStart', epochs: settings.epochs, startedAt: Date.now() });
  try {
    const { plan, features, labels, valFeatures, source } = await loadTrainingRows(env, settings, ac.signal, (progress) =>
      dispatch({ type: 'trainFeatureProgress', progress }),
    );
    dispatch({ type: 'trainPhase', phase: 'training' });
    const result = await trainHead({
      features,
      labels,
      valFeatures,
      valLabels: plan.valLabels,
      classWeight: plan.weights,
      epochs: settings.epochs,
      lr: settings.lr,
      seed: settings.seed,
      signal: ac.signal,
      onEpoch: (stats) => dispatch({ type: 'trainEpoch', stats }),
    });
    const durationMs = performance.now() - t0;
    if (stale() || result.history.length === 0) {
      // Reset session ran while we were training, or nothing was learned: throw the head away.
      result.head.dispose();
      dispatch({ type: 'trainFinished', state: 'cancelled', completedEpochs: 0, durationMs });
      return;
    }
    if (result.cancelled && env.hasPreviousHead) {
      // Keep the earlier, fully trained model; the half-trained one is discarded (state and IndexedDB stay as they were).
      result.head.dispose();
      dispatch({ type: 'trainFinished', state: 'cancelled', completedEpochs: result.history.length, durationMs, keptPrevious: true });
      return;
    }
    const info: HeadInfo = {
      trainedAt: Date.now(),
      seed: settings.seed,
      epochs: settings.epochs,
      completedEpochs: result.history.length,
      lr: settings.lr,
      mode: settings.mode,
      adultMix: settings.adultMix,
      augmentation: settings.augmentation,
      valSet: settings.valSet,
      nRows: plan.nRows,
      nNormal: plan.nNormal,
      nPneumonia: plan.nPneumonia,
      classWeights: plan.weights ? { normal: plan.weights[0], pneumonia: plan.weights[1] } : null,
      usedPrecomputed: source === 'precomputed',
      durationMs,
      restored: false,
    };
    dispatch({ type: 'headSet', head: result.head, info, history: result.history });
    const w = result.head.getWeights();
    void saveHead({ kernel: w.kernel, bias: w.bias, info, history: result.history }, gen);
    dispatch({
      type: 'trainFinished',
      state: result.cancelled ? 'cancelled' : 'done',
      completedEpochs: result.history.length,
      durationMs,
    });
  } catch (e) {
    if (isAbortError(e)) {
      dispatch({ type: 'trainFinished', state: 'cancelled', completedEpochs: 0, durationMs: performance.now() - t0 });
    } else {
      dispatch({ type: 'trainFailed', message: describe(e) });
    }
  } finally {
    controller = null;
  }
}

export const REPEAT_RUNS = 5;

/**
 * Train REPEAT_RUNS times with seeds 1..5 and record accuracy on both validation sets and the Kermany test set.
 * The user's current head is left alone; every run's head is disposed.
 */
export async function runRepeat(rawEnv: RunEnv, settings: TrainSettings): Promise<void> {
  if (controller) return;
  const ac = new AbortController();
  controller = ac;
  const { env } = guardedEnv(rawEnv);
  const { dispatch } = env;
  dispatch({ type: 'repeatStart', total: REPEAT_RUNS });
  try {
    const onProgress = (progress: FeatureProgress) => dispatch({ type: 'repeatFeatureProgress', progress });
    const { plan, features, labels, valFeatures } = await loadTrainingRows(env, settings, ac.signal, onProgress);
    const fe = featureEnv(env, ac.signal, onProgress);
    const byId = new Map(env.entries.map((e) => [e.id, e]));
    const labelsOf = (ids: string[]) => ids.map((id) => labelValue(byId.get(id) as ManifestEntry));
    const ids180 = idsBySplit(env.entries, 'kermany', 'val');
    const ids16 = idsBySplit(env.entries, 'kermany', 'val_official');
    const idsTest = idsBySplit(env.entries, 'kermany', 'test');
    const f180 = settings.valSet === 'val' ? valFeatures : (await getFeatures({ ids: ids180, variant: 'base', ...fe })).features;
    const f16 = (await getFeatures({ ids: ids16, variant: 'base', ...fe })).features;
    const fTest = (await getFeatures({ ids: idsTest, variant: 'base', ...fe })).features;
    const l180 = labelsOf(ids180);
    const l16 = labelsOf(ids16);
    const lTest = labelsOf(idsTest);

    for (let i = 1; i <= REPEAT_RUNS; i++) {
      dispatch({ type: 'repeatProgress', current: i, epoch: 0 });
      const result = await trainHead({
        features,
        labels,
        valFeatures,
        valLabels: plan.valLabels,
        classWeight: plan.weights,
        epochs: settings.epochs,
        lr: settings.lr,
        seed: i,
        signal: ac.signal,
        onEpoch: (s) => dispatch({ type: 'repeatProgress', current: i, epoch: s.epoch }),
      });
      try {
        if (result.cancelled) {
          dispatch({ type: 'repeatEnd', state: 'cancelled' });
          return;
        }
        const run: RepeatRun = {
          seed: i,
          val180: accuracyOfProbs(l180, result.head.predictProba(f180)),
          val16: accuracyOfProbs(l16, result.head.predictProba(f16)),
          test: accuracyOfProbs(lTest, result.head.predictProba(fTest)),
        };
        dispatch({ type: 'repeatRun', run });
      } finally {
        result.head.dispose();
      }
    }
    dispatch({ type: 'repeatEnd', state: 'done' });
  } catch (e) {
    if (isAbortError(e)) dispatch({ type: 'repeatEnd', state: 'cancelled' });
    else dispatch({ type: 'repeatFailed', message: describe(e) });
  } finally {
    controller = null;
  }
}
