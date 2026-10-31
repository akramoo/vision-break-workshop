import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
// Vendored WASM binaries: Vite emits them into dist/assets and rewrites the
// URLs relative to the bundle, so no CDN is needed and `base: './'` works.
import wasmUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import wasmSimdUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import wasmThreadedSimdUrl from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';

export type BackendName = 'webgl' | 'wasm' | 'cpu';

export interface BackendInfo {
  /** Backend actually in use, or 'none' if even CPU failed (practically never). */
  backend: BackendName | 'none';
  /** Human-readable explanation, set when we fell back or the user forced a backend. */
  reason?: string;
  /** True when the first-choice backend was not the one we ended up with. */
  fellBack: boolean;
}

const ORDER: BackendName[] = ['webgl', 'wasm', 'cpu'];

let wasmConfigured = false;
function configureWasm(): void {
  if (wasmConfigured) return;
  setWasmPaths({
    'tfjs-backend-wasm.wasm': wasmUrl,
    'tfjs-backend-wasm-simd.wasm': wasmSimdUrl,
    'tfjs-backend-wasm-threaded-simd.wasm': wasmThreadedSimdUrl,
  });
  wasmConfigured = true;
}

/** Read `?backend=webgl|wasm|cpu` from the URL; anything else is ignored. */
export function backendFromQuery(search: string = globalThis.location?.search ?? ''): BackendName | null {
  const value = new URLSearchParams(search).get('backend');
  return value === 'webgl' || value === 'wasm' || value === 'cpu' ? value : null;
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Switch to a backend and prove it computes. Returns an error message or null on success. */
async function tryBackend(name: BackendName): Promise<string | null> {
  try {
    if (name === 'wasm') configureWasm();
    const ok = await tf.setBackend(name);
    if (!ok) return `${name} backend could not be initialised`;
    await tf.ready();
    if (tf.getBackend() !== name) return `${name} backend is not active`;
    const out = tf.tidy(() => tf.add(tf.tensor1d([1, 2]), tf.tensor1d([3, 4])));
    const values = await out.data();
    out.dispose();
    if (values[0] !== 4 || values[1] !== 6) return `${name} backend returned wrong results`;
    return null;
  } catch (e) {
    return errMessage(e);
  }
}

/**
 * Pick the best available backend: WebGL, then WASM, then CPU. `?backend=` puts
 * the requested backend first (the others remain as fallbacks). Never throws.
 */
export async function initBackend(): Promise<BackendInfo> {
  const forced = backendFromQuery();
  const order = forced ? [forced, ...ORDER.filter((b) => b !== forced)] : ORDER;
  const failures: string[] = [];

  for (const name of order) {
    const error = await tryBackend(name);
    if (error === null) {
      const fellBack = name !== order[0];
      let reason: string | undefined;
      if (fellBack) {
        reason = `${failures.join('; ')}. Using ${name.toUpperCase()} instead.`;
      } else if (forced) {
        reason = `Forced by ?backend=${forced}.`;
      }
      return reason !== undefined ? { backend: name, reason, fellBack } : { backend: name, fellBack };
    }
    failures.push(`${name.toUpperCase()} unavailable (${error})`);
  }
  return {
    backend: 'none',
    reason: `No TensorFlow.js backend could start: ${failures.join('; ')}. Try a different browser.`,
    fellBack: true,
  };
}

/**
 * Call `onLost` if the WebGL context is lost (GPU reset, too many tabs, driver
 * issue). Returns an unsubscribe function; a no-op when WebGL is not active.
 */
export function onWebglContextLost(onLost: () => void): () => void {
  if (tf.getBackend() !== 'webgl') return () => undefined;
  const backend = tf.backend() as unknown as {
    gpgpu?: { gl?: { canvas?: EventTarget } };
  };
  const canvas = backend.gpgpu?.gl?.canvas;
  if (!canvas) return () => undefined;
  const handler = (event: Event) => {
    event.preventDefault();
    onLost();
  };
  canvas.addEventListener('webglcontextlost', handler);
  return () => canvas.removeEventListener('webglcontextlost', handler);
}
