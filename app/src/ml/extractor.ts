import * as tf from '@tensorflow/tfjs';
import { FEATURE_DIM, FEATURE_OUTPUT_NODE, IMG_SIZE } from '../shared/pixelOps.js';

const MODEL_DIR = 'models/mobilenet_v2';

/** Relative URL of a file in the vendored model folder (works with `base: './'`). */
function modelUrl(file: string): string {
  return `${import.meta.env.BASE_URL}${MODEL_DIR}/${file}`;
}

export class ModelLoadError extends Error {
  readonly hint: string;
  constructor(message: string, hint: string) {
    super(message);
    this.name = 'ModelLoadError';
    this.hint = hint;
  }
}

export interface Extractor {
  readonly model: tf.GraphModel;
  /** Features for a batch [n, 224, 224, 3] in [0, 1]. Returns n * 1280 floats, row-major. */
  extractBatch(imgs01: tf.Tensor4D): Promise<Float32Array>;
  /** Features for one image [224, 224, 3] in [0, 1]. Returns 1280 floats. */
  extractFromTensor3D(img01: tf.Tensor3D): Promise<Float32Array>;
  dispose(): void;
}

const REINSTALL_HINT = 'Run "node scripts/fetch_model.mjs" to download it into app/public/models/mobilenet_v2/, then rebuild.';

interface ModelJson {
  modelTopology: unknown;
  format?: string;
  generatedBy?: string;
  convertedBy?: string | null;
  weightsManifest: { paths: string[]; weights: tf.io.WeightsManifestEntry[] }[];
}

function isModelJson(x: unknown): x is ModelJson {
  if (typeof x !== 'object' || x === null) return false;
  const m = x as Partial<ModelJson>;
  return (
    m.modelTopology !== undefined &&
    Array.isArray(m.weightsManifest) &&
    m.weightsManifest.every((g) => Array.isArray(g.paths) && Array.isArray(g.weights))
  );
}

async function fetchChecked(url: string, what: string): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (e) {
    throw new ModelLoadError(
      `Could not download the ${what} (${e instanceof Error ? e.message : String(e)}).`,
      'Check your connection to the server, then reload the page.',
    );
  }
  if (!res.ok) {
    throw new ModelLoadError(`The MobileNet model is incomplete: the ${what} returned HTTP ${res.status} (${url}).`, REINSTALL_HINT);
  }
  return res;
}

/**
 * Download model.json and every weight shard ourselves, then hand tf.io.fromMemory the bytes.
 * tf.loadGraphModel's own HTTP loader leaves one unhandled promise rejection per failed
 * shard; fetching here keeps every failure inside one awaited chain and gives clear messages.
 */
async function fetchModelArtifacts(onProgress?: (fraction: number) => void): Promise<tf.io.ModelArtifacts> {
  const url = modelUrl('model.json');
  const res = await fetchChecked(url, 'model file');
  let json: unknown;
  try {
    json = JSON.parse(await res.text());
  } catch {
    json = null; // dev servers answer unknown paths with index.html (HTTP 200)
  }
  if (!isModelJson(json)) {
    throw new ModelLoadError(
      `The MobileNet model is missing or damaged (${url} is not a valid model description).`,
      REINSTALL_HINT,
    );
  }

  const paths = json.weightsManifest.flatMap((g) => g.paths);
  let done = 0;
  const shards = await Promise.all(
    paths.map(async (path) => {
      const r = await fetchChecked(modelUrl(path), `weight file ${path}`);
      const buf = await r.arrayBuffer();
      done += 1;
      onProgress?.(done / paths.length);
      return buf;
    }),
  );
  return {
    modelTopology: json.modelTopology as tf.io.ModelArtifacts['modelTopology'],
    weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
    weightData: tf.io.concatenateArrayBuffers(shards),
    ...(json.format !== undefined ? { format: json.format } : {}),
    ...(json.generatedBy !== undefined ? { generatedBy: json.generatedBy } : {}),
  };
}

/**
 * Load MobileNet v2 (feature vector) and warm it up on zeros so the first real
 * call is fast. Throws ModelLoadError with a hint when the files are missing.
 */
export async function loadExtractor(onProgress?: (fraction: number) => void): Promise<Extractor> {
  const artifacts = await fetchModelArtifacts(onProgress);

  let model: tf.GraphModel;
  try {
    model = await tf.loadGraphModel(tf.io.fromMemory(artifacts));
  } catch (e) {
    throw new ModelLoadError(
      `The MobileNet model failed to load (${e instanceof Error ? e.message : String(e)}).`,
      'The model files may be damaged. Re-run "node scripts/fetch_model.mjs" and rebuild.',
    );
  }

  const run = (imgs: tf.Tensor4D): tf.Tensor2D =>
    tf.tidy(() => {
      const out = model.execute({ images: imgs }, FEATURE_OUTPUT_NODE) as tf.Tensor;
      return tf.reshape(out, [imgs.shape[0], FEATURE_DIM]);
    });

  async function extractBatch(imgs01: tf.Tensor4D): Promise<Float32Array> {
    const [, h, w, c] = imgs01.shape;
    if (h !== IMG_SIZE || w !== IMG_SIZE || c !== 3) {
      throw new Error(`extractBatch expects [n, ${IMG_SIZE}, ${IMG_SIZE}, 3], got [${imgs01.shape.join(', ')}]`);
    }
    const feats = run(imgs01);
    try {
      return (await feats.data()) as Float32Array;
    } finally {
      feats.dispose();
    }
  }

  async function extractFromTensor3D(img01: tf.Tensor3D): Promise<Float32Array> {
    const batch = tf.expandDims(img01, 0) as tf.Tensor4D;
    try {
      return await extractBatch(batch);
    } finally {
      batch.dispose();
    }
  }

  // Warm-up: compiles shaders / wasm kernels. The result is read back so the
  // work really happens before we report "Ready".
  const warm = tf.tidy(() => run(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]) as tf.Tensor4D));
  await warm.data();
  warm.dispose();

  return { model, extractBatch, extractFromTensor3D, dispose: () => model.dispose() };
}

export interface ModelHash {
  /** Combined SHA-256 of model.json + weight shards, written by scripts/fetch_model.mjs. */
  combined: string;
}

/** Read models/mobilenet_v2/model.sha256.json; null if absent or malformed. */
export async function loadModelHash(): Promise<ModelHash | null> {
  try {
    const res = await fetch(modelUrl('model.sha256.json'));
    if (!res.ok) return null;
    const data: unknown = JSON.parse(await res.text());
    if (typeof data === 'object' && data !== null && 'combined' in data) {
      const { combined } = data as { combined: unknown };
      if (typeof combined === 'string') return { combined };
    }
    return null;
  } catch {
    return null;
  }
}
