// Score a trained head on one of the test sets.
import { evalSetIds, type EvalSetDef } from '../data/evalSets';
import { getFeatures, type FeatureProgress } from '../data/featureStore';
import { labelValue, type ManifestEntry } from '../data/manifest';
import type { EvalResult } from '../state/session';
import type { Extractor } from './extractor';
import type { Head } from './head';
import { metricsWithAbstain } from './metrics';

export interface EvaluateArgs {
  def: EvalSetDef;
  entries: readonly ManifestEntry[];
  head: Head;
  extractor: Extractor | null;
  usePrecomputed: boolean;
  tau: number;
  signal?: AbortSignal;
  onProgress?: (p: FeatureProgress) => void;
  onFallback?: (reason: string) => void;
  onPrecomputedOk?: () => void;
}

export async function evaluateSet(a: EvaluateArgs): Promise<EvalResult> {
  const ids = evalSetIds(a.entries, a.def);
  const byId = new Map(a.entries.map((e) => [e.id, e]));
  const labels = ids.map((id) => labelValue(byId.get(id) as ManifestEntry));
  const features = await getFeatures({
    ids,
    variant: a.def.variant,
    usePrecomputed: a.usePrecomputed,
    entries: a.entries,
    extractor: a.extractor,
    ...(a.signal ? { signal: a.signal } : {}),
    ...(a.onProgress ? { onProgress: a.onProgress } : {}),
    ...(a.onFallback ? { onFallback: a.onFallback } : {}),
    ...(a.onPrecomputedOk ? { onPrecomputedOk: a.onPrecomputedOk } : {}),
  });
  const probs = a.head.predictProba(features.features);
  return {
    setId: a.def.id,
    headId: a.head.id,
    ids,
    labels,
    probs,
    metrics: metricsWithAbstain(labels, probs, a.tau),
    tau: a.tau,
    timestamp: Date.now(),
    // what really happened: precomputed files can fail and fall back to live extraction even when the switch is on
    usedPrecomputed: features.source === 'precomputed',
  };
}
