// D3 acceptance checks for the precomputed embeddings (needs `cd scripts && npm install`).
//
//   node scripts/tests/verify_embeddings.mjs [--samples 10] [--seed 1] [--min-cos 0.999]
//
// 1. structure: row counts, file sizes, ids vs manifest, model hash, dtype/dim
// 2. numbers: no NaN/Inf, mean/std
// 3. sanity: cosine(base, stamped), cosine(base, stress), cosine(base, aug views)
// 4. parity: for N random rows per variant, recompute the features from the JPEG in a child
//    process with plain @tensorflow/tfjs (pure-JS CPU = browser code path) and compare with
//    the stored float16 rows (min cosine must be >= --min-cos).
// Exit code 1 if any hard check fails.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { decodeFloat16 } from '../../app/src/shared/float16.js';
import { IMG_SIZE, mulberry32, randomAugmentParams } from '../../app/src/shared/pixelOps.js';
import { augSeed } from '../../app/src/shared/augSeed.js';
import { cosine } from '../embedding_common.mjs';
import { ensureTfjsNode } from '../ensure_tfjs_node.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const dataDir = path.join(root, 'app', 'public', 'data');
const embDir = path.join(dataDir, 'embeddings');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, all) => (v.startsWith('--') ? [...a, [v.slice(2), all[i + 1]]] : a), []));
const SAMPLES = Number(args.samples ?? 10);
const MIN_COS = Number(args['min-cos'] ?? 0.999);
const rng = mulberry32(Number(args.seed ?? 1));

process.env.TF_CPP_MIN_LOG_LEVEL ??= '2';
ensureTfjsNode();
const require = createRequire(path.join(root, 'scripts', 'package.json'));
const tf = require('@tensorflow/tfjs-node');

let failures = 0;
const check = (ok, msg) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`);
  if (!ok) failures++;
};
const stats = (xs) => {
  let min = Infinity, max = -Infinity, sum = 0;
  for (const x of xs) { min = Math.min(min, x); max = Math.max(max, x); sum += x; }
  return { min, mean: sum / xs.length, max };
};
const f = (x, d = 5) => x.toFixed(d);

const manifest = JSON.parse(readFileSync(path.join(dataDir, 'manifest.json'), 'utf8'));
const byId = new Map(manifest.map((e) => [e.id, e]));
const expectedIds = {
  base: manifest.map((e) => e.id),
  stamped: manifest.map((e) => e.id),
  stress: manifest.filter((e) => e.source === 'kermany' && e.split === 'test').map((e) => e.id),
  aug: manifest.filter((e) => e.split === 'train' || e.split === 'train_adult').flatMap((e) => [e.id, e.id]),
};
const modelSha = JSON.parse(readFileSync(path.join(root, 'app', 'public', 'models', 'mobilenet_v2', 'model.sha256.json'), 'utf8')).combined;

const V = {};
for (const name of ['base', 'stamped', 'aug', 'stress']) {
  console.log(`\n== ${name}`);
  const idx = JSON.parse(readFileSync(path.join(embDir, `${name}.index.json`), 'utf8'));
  const bin = readFileSync(path.join(embDir, `${name}.f16.bin`));
  const rows = expectedIds[name].length;
  check(idx.rows === rows && idx.ids.length === rows, `rows ${idx.rows} == expected ${rows}`);
  check(bin.length === rows * 1280 * 2 && idx.bytes === bin.length, `file size ${bin.length} == rows*1280*2`);
  check(idx.dim === 1280 && idx.dtype === 'float16' && idx.littleEndian === true, 'dim 1280, float16, littleEndian');
  check(idx.modelSha256 === modelSha, 'modelSha256 matches model.sha256.json');
  check(createHash('sha256').update(bin).digest('hex') === idx.binSha256, 'binSha256 matches file');
  check(JSON.stringify(idx.ids) === JSON.stringify(expectedIds[name]), 'ids equal manifest-derived order');
  if (name === 'aug') {
    check(idx.views.every((v, i) => v === i % 2), 'views alternate 0,1 per image');
    check(idx.ids.every((id, i) => idx.seeds[i] === augSeed(id, idx.views[i])), 'seeds follow fnv1a32(id|view)');
  }
  const feats = decodeFloat16(bin);
  const bad = feats.reduce((n, v) => n + (Number.isFinite(v) ? 0 : 1), 0);
  check(bad === 0, 'no NaN/Inf');
  const m = feats.reduce((s, v) => s + v, 0) / feats.length;
  const sd = Math.sqrt(feats.reduce((s, v) => s + (v - m) * (v - m), 0) / feats.length);
  const nz = feats.reduce((n, v) => n + (v === 0 ? 1 : 0), 0) / feats.length;
  console.log(`     mean ${f(m)} std ${f(sd)} fraction exactly 0: ${f(nz, 3)}`);
  V[name] = { idx, feats };
}

// ---- sanity cosines
console.log('\n== cosine sanity (same image, different variant)');
const baseRow = new Map(V.base.idx.ids.map((id, i) => [id, i]));
const cosVs = (name, label) => {
  const { idx, feats } = V[name];
  const cs = idx.ids.map((id, i) => cosine(feats, V.base.feats, i * 1280, baseRow.get(id) * 1280, 1280));
  const s = stats(cs);
  console.log(`  ${label}: n=${cs.length} min ${f(s.min)} mean ${f(s.mean)} max ${f(s.max)}`);
  return s;
};
const sSt = cosVs('stamped', 'base vs stamped');
check(sSt.max < 1 && sSt.mean > 0.5, 'stamped differs from base (max < 1) but is related');
const sSs = cosVs('stress', 'base vs stress ');
check(sSs.max < 1, 'stress differs from base');
for (const view of [0, 1]) {
  const { idx, feats } = V.aug;
  const cs = [];
  idx.ids.forEach((id, i) => { if (idx.views[i] === view) cs.push(cosine(feats, V.base.feats, i * 1280, baseRow.get(id) * 1280, 1280)); });
  const s = stats(cs);
  console.log(`  base vs aug view ${view}: n=${cs.length} min ${f(s.min)} mean ${f(s.mean)} max ${f(s.max)}`);
}

// ---- parity with plain tfjs (browser code path)
console.log(`\n== parity vs plain @tensorflow/tfjs CPU (${SAMPLES} random rows per variant, min cosine must be >= ${MIN_COS})`);
const tmp = path.join(here, '..', '.cache', 'verify_embeddings');
mkdirSync(tmp, { recursive: true });
const jobRows = [];
const pixels = [];
for (const name of ['base', 'stamped', 'aug', 'stress']) {
  const { idx } = V[name];
  const picks = new Set();
  while (picks.size < Math.min(SAMPLES, idx.rows)) picks.add(Math.floor(rng() * idx.rows));
  for (const row of [...picks].sort((a, b) => a - b)) {
    const entry = byId.get(idx.ids[row]);
    const img = tf.node.decodeJpeg(new Uint8Array(readFileSync(path.join(dataDir, entry.path))), 3, 1, true, false, 1, 'INTEGER_ACCURATE');
    if (img.shape[0] !== IMG_SIZE || img.shape[1] !== IMG_SIZE) throw new Error(`unexpected size ${img.shape}`);
    pixels.push(Uint8Array.from(img.dataSync()));
    img.dispose();
    const params = name === 'aug' ? randomAugmentParams(mulberry32(augSeed(entry.id, idx.views[row]))) : undefined;
    jobRows.push({ variant: name, row, params });
  }
}
const pixelFile = path.join(tmp, 'pixels.u8');
writeFileSync(pixelFile, Buffer.concat(pixels.map((p) => Buffer.from(p))));
writeFileSync(path.join(tmp, 'job.json'), JSON.stringify({ pixels: pixelFile, rows: jobRows }));
const outFile = path.join(tmp, 'plain.f32');
const t0 = Date.now();
const r = spawnSync(process.execPath, [path.join(here, 'plain_tfjs_features.mjs'), path.join(tmp, 'job.json'), outFile], { encoding: 'utf8' });
if (r.status !== 0) { console.log(r.stdout, r.stderr); check(false, 'plain tfjs child process ran'); process.exit(1); }
console.log(`  ${r.stdout.trim()} in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
const plainBuf = readFileSync(outFile);
const plain = new Float32Array(plainBuf.buffer, plainBuf.byteOffset, plainBuf.length / 4);
for (const name of ['base', 'stamped', 'aug', 'stress']) {
  const cs = [];
  jobRows.forEach((jr, i) => { if (jr.variant === name) cs.push(cosine(V[name].feats, plain, jr.row * 1280, i * 1280, 1280)); });
  const s = stats(cs);
  console.log(`  ${name.padEnd(8)} n=${cs.length} min cos ${f(s.min, 6)} mean ${f(s.mean, 6)}`);
  check(s.min >= MIN_COS, `${name}: min cosine ${f(s.min, 6)} >= ${MIN_COS}`);
}

// ---- control: pixels decoded by PIL (what browsers' accurate decoders give) instead of tfjs-node
console.log('\n== control: PIL-decoded pixels -> plain tfjs -> compare with stored rows (min cosine must be >= 0.9999)');
const PY = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
if (!existsSync(PY)) {
  console.log(`  SKIP  ${PY} not found (create .venv to enable the PIL control)`);
} else {
  const names = ['base', 'stamped', 'aug', 'stress'];
  const ctl = [];
  for (let k = 0; k < 10; k++) {
    const name = names[k % 4];
    const idx = V[name].idx;
    const row = Math.floor(rng() * idx.rows);
    const entry = byId.get(idx.ids[row]);
    const params = name === 'aug' ? randomAugmentParams(mulberry32(augSeed(entry.id, idx.views[row]))) : undefined;
    ctl.push({ variant: name, row, params, file: path.join(dataDir, entry.path) });
  }
  const filesJson = path.join(tmp, 'pil_files.json');
  const pilPixels = path.join(tmp, 'pil_pixels.u8');
  writeFileSync(filesJson, JSON.stringify(ctl.map((c) => c.file)));
  const py = spawnSync(PY, ['-I', path.join(here, 'pil_dump_rgb.py'), filesJson, pilPixels], { encoding: 'utf8' });
  if (py.status !== 0) { console.log(py.stdout, py.stderr); check(false, 'PIL helper ran'); }
  else {
    const jobFile = path.join(tmp, 'job_pil.json');
    const out2 = path.join(tmp, 'plain_pil.f32');
    writeFileSync(jobFile, JSON.stringify({ pixels: pilPixels, rows: ctl.map(({ variant, params }) => ({ variant, params })) }));
    const r2 = spawnSync(process.execPath, [path.join(here, 'plain_tfjs_features.mjs'), jobFile, out2], { encoding: 'utf8' });
    if (r2.status !== 0) { console.log(r2.stdout, r2.stderr); check(false, 'plain tfjs child (PIL) ran'); }
    else {
      const b2 = readFileSync(out2);
      const pf = new Float32Array(b2.buffer, b2.byteOffset, b2.length / 4);
      const cs = ctl.map((c, i) => cosine(V[c.variant].feats, pf, c.row * 1280, i * 1280, 1280));
      const s = stats(cs);
      console.log(`  n=${cs.length} (variants cycled) min cos ${f(s.min, 6)} mean ${f(s.mean, 6)}`);
      check(s.min >= 0.9999, `PIL pixels vs stored features: min cosine ${f(s.min, 6)} >= 0.9999`);
    }
  }
}

console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
