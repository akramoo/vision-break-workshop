#!/usr/bin/env node
// D3: precompute MobileNet v2 features (1280-d) for the workshop images and write them
// as float16 so students' laptops only train the tiny head.
//
//   cd scripts && npm install && npm run precompute
//   node precompute_embeddings.mjs [--variant base,stamped,aug,stress] [--limit N] [--out DIR] [--batch N]
//
// Variants (written to app/public/data/embeddings/<variant>.f16.bin + <variant>.index.json):
//   base     every manifest entry, in manifest order
//   stamped  every manifest entry with applyStamp(DEFAULT_STAMP)
//   aug      for every 'train' / 'train_adult' entry: view 0 and view 1 (random perturbation,
//            seeded per (id, view)); row order = manifest order, view 0 then view 1
//   stress   Kermany 'test' entries with applyPerturbation(STRESS_PRESET)
//
// The image maths comes from app/src/shared/pixelOps.js, the same file the browser app uses.
// (app/package.json has "type": "module", so Node loads that file as ESM directly.)
//
// --limit N   only the first N rows of each variant (quick test); nothing is written to the
//             real embeddings folder unless --out DIR is given as well.
// Re-running is idempotent: if a variant's output would be byte-identical, files are left alone.

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { encodeFloat16ToBytes } from '../app/src/shared/float16.js';
import * as px from '../app/src/shared/pixelOps.js';
import { AUG_SEED_SCHEME, augSeed } from '../app/src/shared/augSeed.js';
import { ensureTfjsNode } from './ensure_tfjs_node.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const dataDir = path.join(root, 'app', 'public', 'data');
const modelDir = path.join(root, 'app', 'public', 'models', 'mobilenet_v2');
const defaultOutDir = path.join(dataDir, 'embeddings');
const ALL_VARIANTS = ['base', 'stamped', 'aug', 'stress'];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const opts = { variants: ALL_VARIANTS, limit: null, out: null, batch: 32 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} needs a value`);
      return v;
    };
    if (a === '--variant' || a === '--variants') {
      opts.variants = val().split(',').map((s) => s.trim()).filter(Boolean);
      for (const v of opts.variants) if (!ALL_VARIANTS.includes(v)) throw new Error(`unknown variant "${v}" (use ${ALL_VARIANTS.join(',')})`);
    } else if (a === '--limit') {
      opts.limit = Number(val());
      if (!Number.isInteger(opts.limit) || opts.limit < 1) throw new Error('--limit must be a positive integer');
    } else if (a === '--out') {
      opts.out = path.resolve(val());
    } else if (a === '--batch') {
      opts.batch = Number(val());
      if (!Number.isInteger(opts.batch) || opts.batch < 1) throw new Error('--batch must be a positive integer');
    } else if (a === '--help' || a === '-h') {
      console.log('usage: node precompute_embeddings.mjs [--variant base,stamped,aug,stress] [--limit N] [--out DIR] [--batch N]');
      process.exit(0);
    } else {
      throw new Error(`unknown argument ${a}`);
    }
  }
  return opts;
}

// ---------------------------------------------------------------------------
// tfjs-node setup
// ---------------------------------------------------------------------------

process.env.TF_CPP_MIN_LOG_LEVEL ??= '2'; // silence the oneDNN banner

ensureTfjsNode();
const require = createRequire(import.meta.url);
const tf = require('@tensorflow/tfjs-node');

/**
 * tfjs-node has no native 'Transform' kernel (used for the rotation in applyPerturbation).
 * Wrap `tf` so that tf.image.transform runs this single op with the pure-JS CPU kernel from
 * the same tfjs version (identical to the CPU backend the browser falls back to) and
 * hands the result back as a normal tensor of the active backend. Everything else is native.
 */
function withCpuTransform(tfNs) {
  const cpu = tfNs.findBackend('cpu');
  if (!cpu) throw new Error('the pure-JS "cpu" backend is not registered; cannot run Transform');
  const kernel = tfNs.getKernel('Transform', 'cpu');
  if (!kernel) throw new Error('no Transform kernel on the cpu backend');
  const transform = (image, transforms, interpolation = 'nearest', fillMode = 'constant', fillValue = 0, outputShape) => {
    const img = image; // pixelOps always passes float32 tensors
    const tr = transforms;
    const shape = outputShape ?? [img.shape[1], img.shape[2]];
    const imgInfo = cpu.makeTensorInfo(img.shape, 'float32', img.dataSync());
    const trInfo = cpu.makeTensorInfo(tr.shape, 'float32', tr.dataSync());
    let outInfo;
    try {
      outInfo = kernel.kernelFunc({
        inputs: { image: imgInfo, transforms: trInfo },
        attrs: { interpolation, fillMode, fillValue, outputShape: shape },
        backend: cpu,
      });
      return tfNs.tensor(cpu.readSync(outInfo.dataId), outInfo.shape, 'float32');
    } finally {
      cpu.disposeIntermediateTensorInfo(imgInfo);
      cpu.disposeIntermediateTensorInfo(trInfo);
      if (outInfo) cpu.disposeIntermediateTensorInfo(outInfo);
    }
  };
  const image = new Proxy(tfNs.image, { get: (t, p) => (p === 'transform' ? transform : t[p]) });
  return new Proxy(tfNs, { get: (t, p) => (p === 'image' ? image : t[p]) });
}

const tfx = withCpuTransform(tf);

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/** Same loading path as app/src/ml/extractor.ts: model.json + shards -> tf.io.fromMemory. */
async function loadModel() {
  const modelJsonBuf = readFileSync(path.join(modelDir, 'model.json'));
  const json = JSON.parse(modelJsonBuf.toString('utf8'));
  const shardPaths = json.weightsManifest.flatMap((g) => g.paths);
  const shards = shardPaths.map((p) => readFileSync(path.join(modelDir, p)));

  const hash = createHash('sha256');
  for (const b of [modelJsonBuf, ...shards]) hash.update(b);
  const combined = hash.digest('hex');
  const expected = JSON.parse(readFileSync(path.join(modelDir, 'model.sha256.json'), 'utf8')).combined;
  if (combined !== expected) {
    throw new Error(`model files do not match model.sha256.json (computed ${combined}, expected ${expected}); run scripts/fetch_model.mjs`);
  }

  const model = await tf.loadGraphModel(
    tf.io.fromMemory({
      modelTopology: json.modelTopology,
      weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
      weightData: tf.io.concatenateArrayBuffers(shards.map((b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))),
      format: json.format,
      generatedBy: json.generatedBy,
    }),
  );
  return { model, modelSha256: combined };
}

function extract(model, batch01) {
  return tf.tidy(() => {
    const out = model.execute({ images: batch01 }, px.FEATURE_OUTPUT_NODE);
    return tf.reshape(out, [batch01.shape[0], px.FEATURE_DIM]);
  });
}

// ---------------------------------------------------------------------------
// Row plans
// ---------------------------------------------------------------------------

/** A row = {entry, view?, params?}. */
function planRows(variant, manifest, limit) {
  let rows;
  if (variant === 'base' || variant === 'stamped') {
    rows = manifest.map((entry) => ({ entry }));
  } else if (variant === 'stress') {
    rows = manifest.filter((e) => e.source === 'kermany' && e.split === 'test').map((entry) => ({ entry }));
  } else {
    rows = [];
    for (const entry of manifest.filter((e) => e.split === 'train' || e.split === 'train_adult')) {
      for (const view of [0, 1]) {
        const seed = augSeed(entry.id, view);
        rows.push({ entry, view, seed, params: px.randomAugmentParams(px.mulberry32(seed)) });
      }
    }
  }
  return limit === null ? rows : rows.slice(0, limit);
}

/** Decode -> variant-specific pixel ops -> [n,224,224,3] float32 in [0,1]. Disposed by the caller's tidy. */
function buildBatch(variant, rows) {
  const inputs = rows.map((r) => {
    const buf = readFileSync(path.join(dataDir, r.entry.path));
    // INTEGER_ACCURATE = the accurate IDCT browsers and PIL use (the default INTEGER_FAST differs slightly)
    const img = tf.node.decodeJpeg(new Uint8Array(buf), 3, 1, true, false, 1, 'INTEGER_ACCURATE');
    if (img.shape[0] !== px.IMG_SIZE || img.shape[1] !== px.IMG_SIZE) {
      throw new Error(`${r.entry.path}: expected ${px.IMG_SIZE}x${px.IMG_SIZE}, got ${img.shape[0]}x${img.shape[1]}`);
    }
    return px.toModelInput(tfx, img, { scale255: true });
  });
  if (variant === 'aug') {
    return tf.stack(inputs.map((x, i) => px.applyPerturbation(tfx, x, rows[i].params)));
  }
  const batch = tf.stack(inputs);
  if (variant === 'stamped') return px.applyStamp(tfx, batch, px.DEFAULT_STAMP);
  if (variant === 'stress') return px.applyPerturbation(tfx, batch, px.STRESS_PRESET);
  return batch;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

function writeAtomic(file, data) {
  const tmp = `${file}.tmp`;
  writeFileSync(tmp, data);
  renameSync(tmp, file);
}

function describe(variant) {
  const common = 'decodeJpeg(3 ch, INTEGER_ACCURATE) -> toModelInput(scale255:true): float32 /255, [224,224,3] in [0,1]';
  switch (variant) {
    case 'base':
      return { preprocess: common };
    case 'stamped':
      return { preprocess: `${common} -> applyStamp(DEFAULT_STAMP)`, stamp: { ...px.DEFAULT_STAMP } };
    case 'stress':
      return { preprocess: `${common} -> applyPerturbation(STRESS_PRESET)`, stressPreset: { ...px.STRESS_PRESET } };
    default:
      return {
        preprocess: `${common} -> applyPerturbation(randomAugmentParams(mulberry32(seed)))`,
        augParams: { brightness: px.AUG_PARAMS.brightness, contrast: [...px.AUG_PARAMS.contrast], rotationDeg: px.AUG_PARAMS.rotationDeg, cropFrac: [...px.AUG_PARAMS.cropFrac] },
        seedScheme: AUG_SEED_SCHEME,
      };
  }
}

function fmtTime(sec) {
  sec = Math.max(0, Math.round(sec));
  return `${Math.floor(sec / 60)}m${String(sec % 60).padStart(2, '0')}s`;
}

async function runVariant(variant, manifest, model, ctx) {
  const rows = planRows(variant, manifest, ctx.opts.limit);
  const n = rows.length;
  const features = new Float32Array(n * px.FEATURE_DIM);
  const t0 = Date.now();
  console.log(`\n[${variant}] ${n} rows, batch ${ctx.opts.batch}`);
  const isTty = process.stdout.isTTY;
  let lastLog = 0;
  for (let start = 0; start < n; start += ctx.opts.batch) {
    const chunk = rows.slice(start, start + ctx.opts.batch);
    const feats = tf.tidy(() => extract(model, buildBatch(variant, chunk)));
    features.set(feats.dataSync(), start * px.FEATURE_DIM);
    feats.dispose();
    const done = start + chunk.length;
    const now = Date.now();
    if (done === n || now - lastLog > (isTty ? 500 : 5000)) {
      lastLog = now;
      const el = (now - t0) / 1000;
      const eta = (el / done) * (n - done);
      const line = `  [${variant}] ${done}/${n} (${((100 * done) / n).toFixed(0)}%) elapsed ${fmtTime(el)} eta ${fmtTime(eta)}`;
      if (isTty) process.stdout.write(`\r${line}${done === n ? '\n' : '   '}`);
      else console.log(line);
    }
  }

  // Quality gates before anything is written.
  let sum = 0;
  let sumSq = 0;
  let nonFinite = 0;
  for (let i = 0; i < features.length; i++) {
    const v = features[i];
    if (!Number.isFinite(v)) nonFinite++;
    sum += v;
    sumSq += v * v;
  }
  if (nonFinite > 0) throw new Error(`[${variant}] ${nonFinite} non-finite feature values; refusing to write`);
  const mean = sum / features.length;
  const std = Math.sqrt(Math.max(0, sumSq / features.length - mean * mean));
  console.log(`  features: mean ${mean.toFixed(5)} std ${std.toFixed(5)}`);

  const bytes = encodeFloat16ToBytes(features);
  if (bytes.length !== n * px.FEATURE_DIM * 2) throw new Error('internal error: wrong byte length');
  const index = {
    variant,
    dim: px.FEATURE_DIM,
    dtype: 'float16',
    littleEndian: true,
    rows: n,
    bytes: bytes.length,
    binSha256: sha256(bytes),
    modelSha256: ctx.modelSha256,
    ...describe(variant),
    tfVersion: { 'tfjs-node': tf.version['tfjs-node'], 'tfjs-core': tf.version['tfjs-core'], tfjs: tf.version.tfjs },
    backend: tf.getBackend(),
    kernelFallbacks: variant === 'aug' || variant === 'stress' ? { Transform: 'cpu (pure JS; tfjs-node has no native kernel)' } : {},
    ids: rows.map((r) => r.entry.id),
    ...(variant === 'aug' ? { views: rows.map((r) => r.view), seeds: rows.map((r) => r.seed) } : {}),
  };

  const outDir = ctx.outDir;
  mkdirSync(outDir, { recursive: true });
  const binPath = path.join(outDir, `${variant}.f16.bin`);
  const idxPath = path.join(outDir, `${variant}.index.json`);
  let unchanged = false;
  if (existsSync(binPath) && existsSync(idxPath)) {
    try {
      const old = JSON.parse(readFileSync(idxPath, 'utf8'));
      const { createdAt, ...rest } = old;
      unchanged = JSON.stringify(rest) === JSON.stringify(index) && sha256(readFileSync(binPath)) === index.binSha256;
    } catch {
      unchanged = false;
    }
  }
  if (unchanged) {
    console.log(`  unchanged: ${path.relative(root, binPath)} (byte-identical output, files left alone)`);
  } else {
    writeAtomic(binPath, bytes);
    writeAtomic(idxPath, JSON.stringify({ ...index, createdAt: new Date().toISOString() }) + '\n');
    console.log(`  wrote ${path.relative(root, binPath)} (${bytes.length} bytes, sha256 ${index.binSha256.slice(0, 16)}...) + index.json`);
  }
  return { variant, rows: n, bytes: bytes.length, seconds: (Date.now() - t0) / 1000, mean, std };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const manifest = JSON.parse(readFileSync(path.join(dataDir, 'manifest.json'), 'utf8'));
  const dryRun = opts.limit !== null && opts.out === null;
  const ctx = { opts, outDir: opts.out ?? defaultOutDir, modelSha256: null };
  console.log(`tfjs-node ${tf.version['tfjs-node']} (backend: ${tf.getBackend()}), manifest ${manifest.length} entries`);
  if (dryRun) console.log('--limit without --out: dry run, nothing will be written');

  const { model, modelSha256 } = await loadModel();
  ctx.modelSha256 = modelSha256;
  console.log(`model loaded, sha256 ${modelSha256.slice(0, 16)}... verified against model.sha256.json`);
  const baseline = tf.memory().numTensors;

  const results = [];
  const t0 = Date.now();
  for (const variant of opts.variants) {
    if (dryRun) {
      // Same code path, but write into a throwaway temp dir.
      const tmp = path.join(here, '.cache', 'precompute_dryrun');
      results.push(await runVariant(variant, manifest, model, { ...ctx, outDir: tmp }));
    } else {
      results.push(await runVariant(variant, manifest, model, ctx));
    }
  }

  const leaked = tf.memory().numTensors - baseline;
  console.log('\nsummary');
  for (const r of results) {
    console.log(`  ${r.variant.padEnd(8)} rows ${String(r.rows).padStart(5)}  bytes ${String(r.bytes).padStart(9)}  mean ${r.mean.toFixed(4)} std ${r.std.toFixed(4)}  ${fmtTime(r.seconds)}`);
  }
  console.log(`total ${fmtTime((Date.now() - t0) / 1000)}; tensors beyond the model's: ${leaked}`);
  if (leaked !== 0) console.warn('WARNING: tensor leak detected');
}

main().catch((e) => {
  console.error(`\nFAILED: ${e instanceof Error ? e.stack : e}`);
  process.exit(1);
});
