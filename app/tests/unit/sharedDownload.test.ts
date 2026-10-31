import { beforeEach, describe, expect, it, vi } from 'vitest';
import { clearEmbeddingCache, getRowsByIds, loadVariant } from '../../src/data/embeddings';
import { raceAbort } from '../../src/data/featureStore';
import { bumpGeneration, currentGeneration } from '../../src/ml/generation';
import { guardedEnv, type RunEnv } from '../../src/ml/trainRunner';
import { encodeFloat16ToBytes } from '../../src/shared/float16.js';
import { isAbortError } from '../../src/ml/yield';

const DIM = 1280;
const SHA = 'a'.repeat(64);

function files(): Record<string, string | Uint8Array> {
  const f = new Float32Array(3 * DIM);
  for (let r = 0; r < 3; r++) f.fill(r + 1, r * DIM, (r + 1) * DIM);
  return {
    'base.index.json': JSON.stringify({ variant: 'base', dim: DIM, dtype: 'float16', rows: 3, bytes: 3 * DIM * 2, modelSha256: SHA, ids: ['a', 'b', 'c'] }),
    'base.f16.bin': encodeFloat16ToBytes(f),
  };
}

describe('the shared precomputed download is not tied to the first caller', () => {
  beforeEach(() => clearEmbeddingCache());

  it('first caller aborts, second caller still gets the data from the same single download', async () => {
    let gate!: () => void;
    const opened = new Promise<void>((r) => (gate = r));
    const served = files();
    const calls: string[] = [];
    const slowFetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const u = String(url);
      calls.push(u);
      expect(init?.signal, 'no caller signal may reach the shared fetch').toBeUndefined();
      await opened; // hold every response until both callers are waiting
      const body = served[u.slice(u.lastIndexOf('/') + 1)] as BodyInit;
      return new Response(body, { status: 200 });
    }) as typeof fetch;
    const load = () => loadVariant('base', { modelSha256: SHA, baseUrl: '/x/', fetchImpl: slowFetch });

    const acA = new AbortController();
    const acB = new AbortController();
    const a = raceAbort(load(), acA.signal);
    const b = raceAbort(load(), acB.signal);
    acA.abort(); // e.g. the Stamp tab's flip-rate run is cancelled
    await expect(a).rejects.toSatisfy(isAbortError);
    gate();
    const loaded = await b;
    expect(getRowsByIds(loaded, ['c'])[0]).toBe(3);
    expect(calls.filter((c) => c.endsWith('.f16.bin'))).toHaveLength(1);
    expect(calls.filter((c) => c.endsWith('.index.json'))).toHaveLength(1);
  });

  it('a caller whose signal is already aborted never waits, and a later caller is served from the cache', async () => {
    const fetchImpl = vi.fn(async (url: string | URL | Request) => {
      const u = String(url);
      return new Response(files()[u.slice(u.lastIndexOf('/') + 1)] as BodyInit, { status: 200 });
    }) as unknown as typeof fetch;
    const ac = new AbortController();
    ac.abort();
    await expect(raceAbort(loadVariant('base', { modelSha256: SHA, baseUrl: '/x/', fetchImpl }), ac.signal)).rejects.toSatisfy(isAbortError);
    const again = await loadVariant('base', { modelSha256: SHA, baseUrl: '/x/', fetchImpl });
    expect(again.rows).toBe(3);
    expect(fetchImpl).toHaveBeenCalledTimes(2); // index + bin, once
  });

  it('raceAbort passes results and errors through when the signal never fires', async () => {
    const ac = new AbortController();
    await expect(raceAbort(Promise.resolve(5), ac.signal)).resolves.toBe(5);
    await expect(raceAbort(Promise.reject(new Error('boom')), ac.signal)).rejects.toThrow('boom');
    await expect(raceAbort(Promise.resolve(6))).resolves.toBe(6);
  });
});

describe('Reset session generation guard', () => {
  it('silences dispatches of a job that started before the reset', () => {
    const seen: string[] = [];
    const env: RunEnv = { dispatch: (a) => void seen.push(a.type), entries: [], extractor: null, usePrecomputed: true };
    const { env: guarded, stale } = guardedEnv(env);
    guarded.dispatch({ type: 'dismissNotice' });
    expect(stale()).toBe(false);
    expect(seen).toEqual(['dismissNotice']);
    const g = currentGeneration();
    bumpGeneration();
    expect(currentGeneration()).toBe(g + 1);
    expect(stale()).toBe(true);
    guarded.dispatch({ type: 'dismissNotice' });
    guarded.dispatch({ type: 'headCleared' });
    expect(seen).toEqual(['dismissNotice']); // nothing after the reset
    // a job started after the reset is live again
    const fresh = guardedEnv(env);
    fresh.env.dispatch({ type: 'dismissNotice' });
    expect(seen).toEqual(['dismissNotice', 'dismissNotice']);
  });
});
