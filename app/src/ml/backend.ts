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
  /** True when WebGL runs on a software renderer (SwiftShader, llvmpipe, ...): "WebGL" but no real GPU. */
  softwareRenderer: boolean;
  /** Unmasked WebGL renderer string, when the browser exposes it (it may be hidden for privacy). */
  renderer?: string;
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

const SOFTWARE_RENDERER = /swiftshader|llvmpipe|software|basic render/i;

/** True when an (unmasked) WebGL renderer string names a CPU-based rasteriser rather than a GPU. */
export function isSoftwareRenderer(renderer: string | undefined | null): boolean {
  return renderer ? SOFTWARE_RENDERER.test(renderer) : false;
}

type GlLike = {
  getExtension(name: string): unknown;
  getParameter(pname: number): unknown;
};

function rendererOf(gl: GlLike | null | undefined): string | undefined {
  try {
    const ext = gl?.getExtension('WEBGL_debug_renderer_info') as { UNMASKED_RENDERER_WEBGL: number } | null | undefined;
    if (!gl || !ext) return undefined;
    const value = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL);
    return typeof value === 'string' && value ? value : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Unmasked WebGL renderer name, read once at backend init. Prefers the context TensorFlow.js already
 * created (no second context); falls back to a throwaway canvas. Undefined when it cannot be read.
 */
function detectRenderer(): string | undefined {
  try {
    if (tf.getBackend() === 'webgl') {
      const backend = tf.backend() as unknown as { gpgpu?: { gl?: GlLike } };
      const fromTf = rendererOf(backend.gpgpu?.gl);
      if (fromTf) return fromTf;
    }
    if (typeof document === 'undefined') return undefined;
    const canvas = document.createElement('canvas');
    return rendererOf((canvas.getContext('webgl2') ?? canvas.getContext('webgl')) as GlLike | null);
  } catch {
    return undefined;
  }
}

/**
 * Typical live-extraction speed in ms per image before anything has been measured: about 25 on a real GPU,
 * 120 on software WebGL (measured ~19 and ~116), and rough figures for WASM and plain CPU.
 */
export function defaultMsPerImage(info: Pick<BackendInfo, 'backend' | 'softwareRenderer'> | null | undefined): number {
  switch (info?.backend) {
    case 'webgl':
      return info.softwareRenderer ? 120 : 25;
    case 'wasm':
      return 350;
    case 'cpu':
      return 900;
    default:
      return 900;
  }
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
      const renderer = name === 'webgl' ? detectRenderer() : undefined;
      const softwareRenderer = name === 'webgl' && isSoftwareRenderer(renderer);
      let reason: string | undefined;
      if (fellBack) {
        reason = `${failures.join('; ')}. Using ${name.toUpperCase()} instead.`;
      } else if (forced) {
        reason = `Forced by ?backend=${forced}.`;
      }
      return {
        backend: name,
        fellBack,
        softwareRenderer,
        ...(reason !== undefined ? { reason } : {}),
        ...(renderer !== undefined ? { renderer } : {}),
      };
    }
    failures.push(`${name.toUpperCase()} unavailable (${error})`);
  }
  return {
    backend: 'none',
    reason: `No TensorFlow.js backend could start: ${failures.join('; ')}. Try a different browser.`,
    fellBack: true,
    softwareRenderer: false,
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
