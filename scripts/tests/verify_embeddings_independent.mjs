#!/usr/bin/env node
// Independent tester checks for the precomputed embeddings (not written by the builder).
//
//   node scripts/tests/verify_embeddings_independent.mjs [semantic] [reextract] [--seeds N]
//
//   semantic   Train a plain-JS logistic-regression head (Dense(1,sigmoid), Adam lr 1e-3, batch 32,
//              20 epochs, same recipe as the app) on the precomputed `base` features and report
//              accuracy / sensitivity / specificity on Kermany test, adult test, stress and the stamp
//              flip rate. Also reproduces the F6 shortcut experiment and the Day-2 fixes.
//   reextract  Recompute base / stamped / stress features of 6 random images with PLAIN
//              @tensorflow/tfjs (CPU backend, from app/node_modules) and the vendored model and
//              compare cosine to the stored rows. JPEG decoding runs in a child process with
//              tfjs-node (scripts/node_modules) because both TF.js builds must not share a process.
//
// With no mode argument both run. Exit code 1 if a hard check fails (reextract cosine < 0.999).

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const appDir = path.join(root, 'app');
const dataDir = path.join(appDir, 'public', 'data');
const modelDir = path.join(appDir, 'public', 'models', 'mobilenet_v2');
const DIM = 1280;

const args = process.argv.slice(2);

// ----------------------------------------------------------------------------------------------
// child mode: decode JPEGs with tfjs-node -> raw RGB bytes
// ----------------------------------------------------------------------------------------------
if (args[0] === '--decode-child') {
  process.env.TF_CPP_MIN_LOG_LEVEL = '2';
  const req = createRequire(path.join(root, 'scripts', 'package.json'));
  const tfn = req('@tensorflow/tfjs-node');
  const [, outDir, ...files] = args;
  files.forEach((f, i) => {
    const img = tfn.node.decodeJpeg(new Uint8Array(readFileSync(path.join(dataDir, f))), 3, 1, true, false, 1, 'INTEGER_ACCURATE');
    const [h, w] = img.shape;
    writeFileSync(path.join(outDir, `${i}.raw`), Buffer.from(img.dataSync()));
    writeFileSync(path.join(outDir, `${i}.json`), JSON.stringify({ h, w }));
    img.dispose();
  });
  process.exit(0);
}

// ----------------------------------------------------------------------------------------------
// shared data loading (own float16 decode, written from the spec; does not import the builder's)
// ----------------------------------------------------------------------------------------------
function half(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 31;
  const f = h & 1023;
  if (e === 0) return s * (f / 1024) * 2 ** -14;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}
const TABLE = Float32Array.from({ length: 65536 }, (_, i) => half(i));
function loadVariant(v) {
  const buf = readFileSync(path.join(dataDir, 'embeddings', `${v}.f16.bin`));
  const idx = JSON.parse(readFileSync(path.join(dataDir, 'embeddings', `${v}.index.json`), 'utf8'));
  const out = new Float32Array(buf.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = TABLE[buf[2 * i] | (buf[2 * i + 1] << 8)];
  return { idx, f: out };
}
const manifest = JSON.parse(readFileSync(path.join(dataDir, 'manifest.json'), 'utf8'));
const byId = new Map(manifest.map((e) => [e.id, e]));
const cosine = (a, b) => {
  let d = 0, x = 0, y = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; }
  return d / Math.sqrt(x * y);
};

// ----------------------------------------------------------------------------------------------
// semantic sanity
// ----------------------------------------------------------------------------------------------
function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Logistic regression head. X: array of Float32Array rows; y: 0/1; weights per sample. */
function trainHead(X, y, { seed = 42, epochs = 20, lr = 1e-3, batch = 32, balanced = false } = {}) {
  const rnd = mulberry(seed);
  const n = X.length;
  const lim = Math.sqrt(6 / (DIM + 1)); // glorot uniform like Keras Dense
  const w = new Float64Array(DIM).map(() => (rnd() * 2 - 1) * lim);
  let b = 0;
  const m = new Float64Array(DIM + 1), v = new Float64Array(DIM + 1);
  const n1 = y.reduce((p, c) => p + c, 0);
  const cw = balanced ? [n / (2 * (n - n1)), n / (2 * n1)] : [1, 1];
  const order = Array.from({ length: n }, (_, i) => i);
  let t = 0;
  const b1 = 0.9, b2 = 0.999, eps = 1e-7;
  for (let ep = 0; ep < epochs; ep++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    for (let s = 0; s < n; s += batch) {
      const bi = order.slice(s, s + batch);
      const gw = new Float64Array(DIM + 1);
      let wsum = 0;
      for (const k of bi) {
        const x = X[k];
        let z = b;
        for (let d = 0; d < DIM; d++) z += w[d] * x[d];
        const p = 1 / (1 + Math.exp(-z));
        const sw = cw[y[k]];
        const g = (p - y[k]) * sw;
        for (let d = 0; d < DIM; d++) gw[d] += g * x[d];
        gw[DIM] += g;
        wsum += 1; // Keras averages the weighted loss over the batch size
      }
      t++;
      for (let d = 0; d <= DIM; d++) {
        const g = gw[d] / wsum;
        m[d] = b1 * m[d] + (1 - b1) * g;
        v[d] = b2 * v[d] + (1 - b2) * g * g;
        const step = (lr * Math.sqrt(1 - b2 ** t)) / (1 - b1 ** t);
        const upd = (step * m[d]) / (Math.sqrt(v[d]) + eps);
        if (d < DIM) w[d] -= upd; else b -= upd;
      }
    }
  }
  return (x) => {
    let z = b;
    for (let d = 0; d < DIM; d++) z += w[d] * x[d];
    return 1 / (1 + Math.exp(-z));
  };
}

function evalSet(predict, rows) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const { x, y } of rows) {
    const p = predict(x) >= 0.5 ? 1 : 0;
    if (y === 1) (p === 1 ? tp++ : fn++); else (p === 0 ? tn++ : fp++);
  }
  const n = rows.length;
  const pct = (a, b) => (b === 0 ? NaN : (100 * a) / b);
  return { n, acc: pct(tp + tn, n), sens: pct(tp, tp + fn), spec: pct(tn, tn + fp), flipToPneu: pct(fp, tn + fp) };
}
const fmt = (r) => `n=${String(r.n).padStart(3)} acc ${r.acc.toFixed(1).padStart(5)}%  sens ${r.sens.toFixed(1).padStart(5)}%  spec ${r.spec.toFixed(1).padStart(5)}%`;
const mean = (a) => a.reduce((p, c) => p + c, 0) / a.length;

function runSemantic(nSeeds) {
  console.log('\n=== semantic sanity (plain-JS logistic head on precomputed features) ===');
  const base = loadVariant('base');
  const stamped = loadVariant('stamped');
  const stress = loadVariant('stress');
  const aug = loadVariant('aug');
  const baseRow = (id) => base.f.subarray(base.idx.ids.indexOf(id) * DIM, (base.idx.ids.indexOf(id) + 1) * DIM);
  const rowsOf = (variantObj, pred) => {
    const rows = [];
    variantObj.idx.ids.forEach((id, i) => {
      const e = byId.get(id);
      if (pred(e)) rows.push({ id, x: variantObj.f.subarray(i * DIM, (i + 1) * DIM), y: e.label === 'pneumonia' ? 1 : 0, view: variantObj.idx.views?.[i] });
    });
    return rows;
  };
  const K = (split) => (e) => e.source === 'kermany' && e.split === split;
  const A = (split) => (e) => e.source === 'nih' && e.split === split;
  const trainK = rowsOf(base, K('train'));
  const trainA = rowsOf(base, A('train_adult'));
  const valK = rowsOf(base, K('val'));
  const valOff = rowsOf(base, K('val_official'));
  const testK = rowsOf(base, K('test'));
  const testA = rowsOf(base, A('test_adult'));
  const stressRows = rowsOf(stress, () => true);
  const stampedNormalTest = rowsOf(stamped, (e) => K('test')(e) && e.label === 'normal');
  const stampedPneuTest = rowsOf(stamped, (e) => K('test')(e) && e.label === 'pneumonia');
  const stampedTrain = rowsOf(stamped, K('train'));
  const augTrain = rowsOf(aug, () => true);
  const stampedAdultTest = rowsOf(stamped, (e) => A('test_adult')(e));
  console.log(`train kermany ${trainK.length} (pneumonia ${trainK.filter((r) => r.y).length}), train_adult ${trainA.length}, val ${valK.length}, val_official ${valOff.length}, test ${testK.length}, test_adult ${testA.length}, stress ${stressRows.length}`);
  console.log(`always-pneumonia baseline on Kermany test: ${(100 * testK.filter((r) => r.y).length / testK.length).toFixed(1)}%`);

  const configs = {
    'A realistic  (base, Kermany train, unweighted)': () => ({ X: trainK.map((r) => r.x), y: trainK.map((r) => r.y), balanced: false }),
    'B balanced   (base, Kermany train, class-weighted)': () => ({ X: trainK.map((r) => r.x), y: trainK.map((r) => r.y), balanced: true }),
    'C +adult mix (base, Kermany+adult train, balanced)': () => ({ X: [...trainK, ...trainA].map((r) => r.x), y: [...trainK, ...trainA].map((r) => r.y), balanced: true }),
    'D +aug       (base+aug views of train, balanced)': () => ({ X: [...trainK, ...trainA, ...augTrain].map((r) => r.x), y: [...trainK, ...trainA, ...augTrain].map((r) => r.y), balanced: true }),
    'E stamp-shortcut (pneumonia train stamped, normal not; unweighted)': () => {
      const rows = trainK.map((r) => (r.y === 1 ? stampedTrain.find((s) => s.id === r.id) : r));
      return { X: rows.map((r) => r.x), y: rows.map((r) => r.y), balanced: false };
    },
  };

  const summary = {};
  for (const [name, mk] of Object.entries(configs)) {
    console.log(`\n--- ${name} ---`);
    const acc = { val: [], testK: [], testA: [], stress: [], flipStamp: [], flipBase: [], stampPneuAcc: [] };
    for (let s = 0; s < nSeeds; s++) {
      const { X, y, balanced } = mk();
      const predict = trainHead(X, y, { seed: 42 + s, balanced });
      const rVal = evalSet(predict, valK);
      const rK = evalSet(predict, testK);
      const rA = evalSet(predict, testA);
      const rS = evalSet(predict, stressRows);
      const rStampN = evalSet(predict, stampedNormalTest);
      const rBaseN = evalSet(predict, testK.filter((r) => r.y === 0));
      const rStampP = evalSet(predict, stampedPneuTest);
      const rOff = evalSet(predict, valOff);
      if (s === 0) {
        console.log(` seed 42:`);
        console.log(`   val (180)        ${fmt(rVal)}`);
        console.log(`   val_official (16)${fmt(rOff)}`);
        console.log(`   Kermany test     ${fmt(rK)}`);
        console.log(`   adult test       ${fmt(rA)}`);
        console.log(`   stress           ${fmt(rS)}`);
        console.log(`   normal test, base    -> flagged pneumonia: ${rBaseN.flipToPneu.toFixed(1)}%  (= 100 - specificity)`);
        console.log(`   normal test, STAMPED -> flagged pneumonia: ${rStampN.flipToPneu.toFixed(1)}%  (flip rate)`);
        console.log(`   pneumonia test, STAMPED sens: ${rStampP.sens.toFixed(1)}%`);
        console.log(`   adult test stamped(all): ${fmt(evalSet(predict, stampedAdultTest))}`);
      }
      acc.val.push(rVal.acc); acc.testK.push(rK.acc); acc.testA.push(rA.acc); acc.stress.push(rS.acc);
      acc.flipStamp.push(rStampN.flipToPneu); acc.flipBase.push(rBaseN.flipToPneu); acc.stampPneuAcc.push(rStampP.sens);
    }
    const f = (a) => `${mean(a).toFixed(1)} [${Math.min(...a).toFixed(1)}..${Math.max(...a).toFixed(1)}]`;
    console.log(` over ${nSeeds} seeds: val acc ${f(acc.val)} | Kermany test acc ${f(acc.testK)} | adult acc ${f(acc.testA)} | stress acc ${f(acc.stress)} | stamp flip ${f(acc.flipStamp)} vs unstamped FP ${f(acc.flipBase)}`);
    summary[name] = acc;
  }
  void baseRow;
  return summary;
}

// ----------------------------------------------------------------------------------------------
// independent re-extraction
// ----------------------------------------------------------------------------------------------
async function runReextract() {
  console.log('\n=== independent re-extraction (plain @tensorflow/tfjs, CPU) ===');
  const reqApp = createRequire(path.join(appDir, 'package.json'));
  const tf = reqApp('@tensorflow/tfjs');
  await tf.setBackend('cpu');
  await tf.ready();
  const px = await import(pathToFileURL(path.join(appDir, 'src', 'shared', 'pixelOps.js')).href);

  const base = loadVariant('base');
  const stamped = loadVariant('stamped');
  const stress = loadVariant('stress');
  const aug = loadVariant('aug');

  // choose 6 images pseudo-randomly (seed from clock so repeated runs vary; printed for reproduction)
  const seed = Number(process.env.SEED ?? Date.now() % 100000);
  const rnd = mulberry(seed);
  const pickFrom = (pred) => {
    const c = manifest.filter(pred);
    return c[Math.floor(rnd() * c.length)];
  };
  const picks = [
    pickFrom((e) => e.source === 'kermany' && e.split === 'test'),
    pickFrom((e) => e.source === 'kermany' && e.split === 'test'),
    pickFrom((e) => e.source === 'nih' && e.split === 'test_adult'),
    pickFrom((e) => e.source === 'nih' && e.split === 'test_adult'),
    pickFrom((e) => e.source === 'kermany' && e.split === 'train'),
    pickFrom((e) => e.source === 'kermany' && e.split === 'train'),
  ];
  console.log(`seed ${seed}; images: ${picks.map((p) => p.id).join(', ')}`);

  // decode in child (tfjs-node)
  const tmp = path.join(os.tmpdir(), `reextract_${process.pid}`);
  mkdirSync(tmp, { recursive: true });
  const r = spawnSync(process.execPath, [fileURLToPath(import.meta.url), '--decode-child', tmp, ...picks.map((p) => p.path)], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error('decode child failed');

  // model, loaded the same way as the app (model.json + shards via tf.io.fromMemory)
  const mj = readFileSync(path.join(modelDir, 'model.json'));
  const json = JSON.parse(mj.toString('utf8'));
  const shards = json.weightsManifest.flatMap((g) => g.paths).map((p) => readFileSync(path.join(modelDir, p)));
  const hash = createHash('sha256');
  for (const b of [mj, ...shards]) hash.update(b);
  const combined = hash.digest('hex');
  const expected = JSON.parse(readFileSync(path.join(modelDir, 'model.sha256.json'), 'utf8')).combined;
  console.log(`model sha256 ${combined === expected ? 'OK' : 'MISMATCH'}`);
  const model = await tf.loadGraphModel(
    tf.io.fromMemory({
      modelTopology: json.modelTopology,
      weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
      weightData: tf.io.concatenateArrayBuffers(shards.map((b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))),
      format: json.format,
      generatedBy: json.generatedBy,
    }),
  );
  const features = (img01) => tf.tidy(() => {
    const x = img01.rank === 3 ? img01.expandDims(0) : img01;
    return tf.reshape(model.execute({ images: x }, px.FEATURE_OUTPUT_NODE), [-1]);
  });

  let worst = 1;
  const lines = [];
  for (let i = 0; i < picks.length; i++) {
    const e = picks[i];
    const { h, w } = JSON.parse(readFileSync(path.join(tmp, `${i}.json`), 'utf8'));
    const raw = readFileSync(path.join(tmp, `${i}.raw`));
    const rgb = tf.tensor3d(Int32Array.from(raw), [h, w, 3], 'int32');
    const img01 = px.toModelInput(tf, rgb, { scale255: true });
    const res = [];
    const compare = (name, v, rowIdx, tensorFn) => {
      if (rowIdx < 0) return;
      const t = tensorFn();
      const f = features(t).dataSync();
      const stored = v.f.subarray(rowIdx * DIM, (rowIdx + 1) * DIM);
      const c = cosine(f, stored);
      worst = Math.min(worst, c);
      res.push(`${name} cos ${c.toFixed(7)} (1-cos=${(1 - c).toExponential(1)})`);
      t.dispose();
    };
    compare('base', base, base.idx.ids.indexOf(e.id), () => img01.clone());
    compare('stamped', stamped, stamped.idx.ids.indexOf(e.id), () => px.applyStamp(tf, img01, px.DEFAULT_STAMP));
    compare('stress', stress, stress.idx.ids.indexOf(e.id), () => px.applyPerturbation(tf, img01, px.STRESS_PRESET));
    // aug (both views, seeds taken from the independent FNV recomputation, not from the index)
    for (const view of [0, 1]) {
      const rowIdx = aug.idx.ids.findIndex((id, k) => id === e.id && aug.idx.views[k] === view);
      if (rowIdx < 0) continue;
      let hsh = 2166136261;
      for (const bt of Buffer.from(`${e.id}|${view}`, 'utf8')) hsh = Math.imul((hsh ^ bt) >>> 0, 16777619) >>> 0;
      compare(`aug${view}`, aug, rowIdx, () => px.applyPerturbation(tf, img01, px.randomAugmentParams(px.mulberry32(hsh))));
    }
    console.log(`${e.id.padEnd(32)} ${res.join(' | ')}`);
    lines.push(res.length);
    rgb.dispose();
    img01.dispose();
  }
  console.log(`worst cosine over all comparisons: ${worst.toFixed(6)} (threshold 0.999)`);
  return worst >= 0.999;
}

// ----------------------------------------------------------------------------------------------
const modes = args.filter((a) => !a.startsWith('--'));
const run = (m) => modes.length === 0 || modes.includes(m);
const si = args.indexOf('--seeds');
const nSeeds = si >= 0 ? Number(args[si + 1]) : 5;
let ok = true;
if (run('semantic')) runSemantic(nSeeds);
if (run('reextract')) ok = (await runReextract()) && ok;
process.exit(ok ? 0 : 1);
