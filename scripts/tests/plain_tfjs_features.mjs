// Helper for verify_embeddings.mjs: recompute features with plain @tensorflow/tfjs (pure-JS CPU
// backend, from app/node_modules), i.e. the same code path the browser app uses, from raw RGB
// pixels. Runs in its own process because it must not share a tfjs instance with tfjs-node.
//
//   node plain_tfjs_features.mjs <job.json> <out.f32.bin>
//
// job.json: { pixels: "<raw uint8 file, n * 224*224*3>", rows: [{variant, params?}] }
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import * as px from '../../app/src/shared/pixelOps.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const require = createRequire(path.join(root, 'app', 'package.json'));
const tf = require('@tensorflow/tfjs');
await tf.setBackend('cpu');
await tf.ready();

const [jobPath, outPath] = process.argv.slice(2);
const job = JSON.parse(readFileSync(jobPath, 'utf8'));
const pixels = readFileSync(job.pixels);
const imgBytes = px.IMG_SIZE * px.IMG_SIZE * 3;

// Load exactly like app/src/ml/extractor.ts (fromMemory).
const dir = path.join(root, 'app', 'public', 'models', 'mobilenet_v2');
const modelJson = JSON.parse(readFileSync(path.join(dir, 'model.json'), 'utf8'));
const shards = modelJson.weightsManifest.flatMap((g) => g.paths).map((p) => readFileSync(path.join(dir, p)));
const model = await tf.loadGraphModel(
  tf.io.fromMemory({
    modelTopology: modelJson.modelTopology,
    weightSpecs: modelJson.weightsManifest.flatMap((g) => g.weights),
    weightData: tf.io.concatenateArrayBuffers(shards.map((b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))),
    format: modelJson.format,
    generatedBy: modelJson.generatedBy,
  }),
);

const out = new Float32Array(job.rows.length * px.FEATURE_DIM);
for (const [i, row] of job.rows.entries()) {
  const feats = tf.tidy(() => {
    const raw = tf.tensor3d(new Uint8Array(pixels.buffer, pixels.byteOffset + i * imgBytes, imgBytes), [px.IMG_SIZE, px.IMG_SIZE, 3], 'int32');
    let x = px.toModelInput(tf, raw, { scale255: true });
    if (row.variant === 'stamped') x = px.applyStamp(tf, x, px.DEFAULT_STAMP);
    else if (row.variant === 'stress') x = px.applyPerturbation(tf, x, px.STRESS_PRESET);
    else if (row.variant === 'aug') x = px.applyPerturbation(tf, x, row.params);
    const y = model.execute({ images: tf.expandDims(x, 0) }, px.FEATURE_OUTPUT_NODE);
    return tf.reshape(y, [px.FEATURE_DIM]);
  });
  out.set(feats.dataSync(), i * px.FEATURE_DIM);
  feats.dispose();
}
writeFileSync(outPath, Buffer.from(out.buffer));
console.log(`plain tfjs ${tf.version.tfjs} (${tf.getBackend()}): ${job.rows.length} rows`);
