// Independent verification of D0. Run: node scripts/tests/verify_model.mjs  (zero deps)
import { createHash } from 'node:crypto';
import { readFileSync, existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.resolve(here, '..', '..', 'app', 'public', 'models', 'mobilenet_v2');
const sha = (b) => createHash('sha256').update(b).digest('hex');
const DT = { float32: 4, int32: 4, uint8: 1, int8: 1, uint16: 2, float16: 2 };

let failed = 0;
let total = 0;
function check(name, fn) {
  total++;
  try {
    const d = fn();
    console.log(`PASS  ${name}${d ? `  [${d}]` : ''}`);
  } catch (e) {
    failed++;
    console.log(`FAIL  ${name}: ${e.message}`);
  }
}
function assert(c, msg) {
  if (!c) throw new Error(msg);
}

let modelBuf, model, shardPaths, shardBufs, source, combinedJson;

check('model.json parses as graph-model', () => {
  modelBuf = readFileSync(path.join(dir, 'model.json'));
  model = JSON.parse(modelBuf.toString('utf8'));
  assert(model.format === 'graph-model', `format=${model.format}`);
  assert(model.modelTopology && Array.isArray(model.modelTopology.node), 'no modelTopology.node[]');
  assert(Array.isArray(model.weightsManifest) && model.weightsManifest.length > 0, 'no weightsManifest');
  return `format=${model.format}, generatedBy=${model.generatedBy}, ${model.modelTopology.node.length} nodes`;
});

check('weightsManifest paths exist (3 shards, in order)', () => {
  shardPaths = model.weightsManifest.flatMap((g) => g.paths);
  assert(JSON.stringify(shardPaths) === JSON.stringify([1, 2, 3].map((i) => `group1-shard${i}of3.bin`)), `paths=${shardPaths}`);
  for (const p of shardPaths) assert(existsSync(path.join(dir, p)), `missing ${p}`);
  shardBufs = shardPaths.map((p) => readFileSync(path.join(dir, p)));
  return shardPaths.join(', ');
});

check('shard bytes sum == bytes implied by weights manifest (shapes*dtype)', () => {
  let implied = 0;
  for (const g of model.weightsManifest)
    for (const w of g.weights) {
      const dt = w.quantization ? w.quantization.dtype : w.dtype;
      assert(DT[dt], `unknown dtype ${dt} for ${w.name}`);
      implied += w.shape.reduce((a, b) => a * b, 1) * DT[dt];
    }
  const sum = shardBufs.reduce((n, b) => n + b.length, 0);
  assert(sum === implied, `shards total ${sum} != implied ${implied}`);
  for (const [i, p] of shardPaths.entries()) assert(statSync(path.join(dir, p)).size === shardBufs[i].length, `size mismatch ${p}`);
  return `${sum} bytes (${shardBufs.map((b) => b.length).join('+')})`;
});

check('SOURCE.md per-file SHA-256 and byte counts match files', () => {
  source = readFileSync(path.join(dir, 'SOURCE.md'), 'utf8');
  const files = [['model.json', modelBuf], ...shardPaths.map((p, i) => [p, shardBufs[i]])];
  for (const [name, buf] of files) {
    const row = source.split('\n').find((l) => l.includes(`\`${name}\``) && l.startsWith('|'));
    assert(row, `no SOURCE.md table row for ${name}`);
    const m = row.match(/\|\s*`[^`]+`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]+)`\s*\|/);
    assert(m, `unparseable row for ${name}: ${row}`);
    assert(m[2].length === 64, `${name}: hash length ${m[2].length}`);
    assert(Number(m[1]) === buf.length, `${name}: SOURCE.md bytes ${m[1]} != ${buf.length}`);
    assert(m[2] === sha(buf), `${name}: SOURCE.md sha ${m[2]} != actual ${sha(buf)}`);
  }
  return '4/4 files';
});

check('combined hash (model.json + shards 1..3) matches model.sha256.json and SOURCE.md', () => {
  combinedJson = JSON.parse(readFileSync(path.join(dir, 'model.sha256.json'), 'utf8'));
  const actual = sha(Buffer.concat([modelBuf, ...shardBufs]));
  assert(combinedJson.combined === actual, `model.sha256.json ${combinedJson.combined} != actual ${actual}`);
  assert(source.includes(actual), 'combined hash not in SOURCE.md');
  return actual;
});

check('input placeholder shape [-1,224,224,3]', () => {
  const ph = model.modelTopology.node.filter((n) => n.op === 'Placeholder');
  assert(ph.length === 1, `expected 1 Placeholder, found ${ph.length}`);
  const dims = ph[0].attr.shape.shape.dim.map((d) => Number(d.size));
  assert(JSON.stringify(dims) === '[-1,224,224,3]', `shape=${JSON.stringify(dims)}`);
  return `${ph[0].name} ${JSON.stringify(dims)}`;
});

check('output node module_apply_default/hub_output/feature_vector/SpatialSqueeze exists', () => {
  const name = 'module_apply_default/hub_output/feature_vector/SpatialSqueeze';
  const n = model.modelTopology.node.find((x) => x.name === name);
  assert(n, 'node not found');
  if (model.signature?.outputs) {
    const outs = JSON.stringify(model.signature.outputs);
    assert(outs.includes('SpatialSqueeze'), `signature outputs do not reference it: ${outs}`);
  }
  return `op=${n.op}`;
});

check('SOURCE.md mentions Apache-2.0 and the source URL', () => {
  assert(/Apache-2\.0/.test(source), 'no Apache-2.0');
  assert(source.includes('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1/model.json'), 'URL missing');
  assert(source.includes('[-1, 224, 224, 3]') && source.includes('1280'), 'input/output docs missing');
});

console.log(`\n${total - failed}/${total} checks passed`);
process.exit(failed ? 1 : 0);
