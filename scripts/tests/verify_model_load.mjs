// D0 acceptance: "Node loads the model and the output shape is [1,1280]".
// Uses @tensorflow/tfjs from app/node_modules (read-only; no installs). CPU backend.
// Run: node scripts/tests/verify_model_load.mjs
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const dir = path.join(root, 'app', 'public', 'models', 'mobilenet_v2');
const require = createRequire(path.join(root, 'app', 'package.json'));
const tf = require('@tensorflow/tfjs');
await tf.setBackend('cpu');
await tf.ready();

const modelJson = JSON.parse(readFileSync(path.join(dir, 'model.json'), 'utf8'));
const weightSpecs = modelJson.weightsManifest.flatMap((g) => g.weights);
const bufs = modelJson.weightsManifest.flatMap((g) => g.paths).map((p) => readFileSync(path.join(dir, p)));
const all = Buffer.concat(bufs);
const weightData = all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength);

const handler = {
  load: async () => ({
    modelTopology: modelJson.modelTopology,
    weightSpecs,
    weightData,
    format: modelJson.format,
    generatedBy: modelJson.generatedBy,
    convertedBy: modelJson.convertedBy,
    signature: modelJson.signature,
    userDefinedMetadata: modelJson.userDefinedMetadata,
  }),
};

let failed = 0;
const check = (name, ok, detail) => {
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  [${detail}]`);
};

const model = await tf.loadGraphModel(handler);
check('loadGraphModel succeeds', true, `inputs=${model.inputs.map((i) => i.name + JSON.stringify(i.shape))}`);

const x = tf.fill([1, 224, 224, 3], 0.5);
const y = model.predict(x);
const out = Array.isArray(y) ? y[0] : y;
check('output shape is [1,1280]', JSON.stringify(out.shape) === '[1,1280]', JSON.stringify(out.shape));
const v = out.dataSync();
check('output finite and non-trivial', v.every(Number.isFinite) && v.some((a) => a !== 0), `min=${Math.min(...v).toFixed(3)} max=${Math.max(...v).toFixed(3)}`);

const x2 = tf.stack([tf.fill([224, 224, 3], 0.2), tf.fill([224, 224, 3], 0.8)]);
const y2 = model.predict(x2);
check('batch of 2 -> [2,1280]', JSON.stringify(y2.shape) === '[2,1280]', JSON.stringify(y2.shape));

console.log(failed ? `\n${failed} check(s) FAILED` : '\nall checks passed');
process.exit(failed ? 1 : 0);
