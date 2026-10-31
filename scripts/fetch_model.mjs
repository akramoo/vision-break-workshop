#!/usr/bin/env node
// D0: download the TF Hub MobileNetV2 (1.0, 224) feature-vector tfjs graph model
// into app/public/models/mobilenet_v2/ and record SHA-256 hashes.
// Zero npm deps (Node >= 18: built-in fetch + crypto). Idempotent; use --force to re-download.

import { createHash } from 'node:crypto';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE =
  'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1/';
const QUERY = '?tfjs-format=file';
const MODEL_URL = `${BASE}model.json${QUERY}`;
const SHARDS = [1, 2, 3].map((i) => `group1-shard${i}of3.bin`);

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(here, '..', 'app', 'public', 'models', 'mobilenet_v2');
const force = process.argv.includes('--force');

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');
const sizeOf = async (p) => (await stat(p).catch(() => null))?.size ?? null;

// fetch() follows the tfhub.dev -> kaggle.com -> storage.googleapis.com redirects by default.
async function download(url) {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const expected = Number(res.headers.get('content-length'));
  if (expected && buf.length !== expected) {
    throw new Error(`${url}: got ${buf.length} bytes, expected ${expected}`);
  }
  return buf;
}

let downloaded = false;

// expectedSize is optional (unknown for model.json: a truncated copy fails JSON.parse below).
async function fetchFile(name, url, expectedSize) {
  const dest = path.join(outDir, name);
  const existing = await sizeOf(dest);
  if (!force && existing !== null && (expectedSize === undefined || existing === expectedSize)) {
    console.log(`skip     ${name} (${existing} bytes, exists)`);
    return await readFile(dest);
  }
  console.log(`download ${name} ...`);
  const buf = await download(url);
  await writeFile(dest, buf);
  downloaded = true;
  console.log(`  saved  ${buf.length} bytes`);
  return buf;
}

await mkdir(outDir, { recursive: true });

const modelBuf = await fetchFile('model.json', MODEL_URL);
const manifest = JSON.parse(modelBuf.toString('utf8'));
const listed = manifest.weightsManifest.flatMap((g) => g.paths);
if (JSON.stringify(listed) !== JSON.stringify(SHARDS)) {
  throw new Error(`Unexpected shard list in model.json: ${listed.join(', ')}`);
}

// Total weight bytes expected from the manifest (all float32 here) -> validate shard sizes.
const DTYPE_BYTES = { float32: 4, int32: 4, uint8: 1, int8: 1, uint16: 2, float16: 2 };
const expectedTotal = manifest.weightsManifest
  .flatMap((g) => g.weights)
  .reduce((n, w) => n + w.shape.reduce((a, b) => a * b, 1) * (DTYPE_BYTES[w.quantization?.dtype ?? w.dtype]), 0);

// tfjs shards are 4 MiB chunks and the last holds the remainder; the expected sizes
// let a re-run detect (and replace) truncated files.
const SHARD_CHUNK = 4 * 1024 * 1024;
const shardBufs = [];
for (const [i, name] of SHARDS.entries()) {
  const size = i < SHARDS.length - 1 ? SHARD_CHUNK : expectedTotal - SHARD_CHUNK * i;
  // Shards resolve relative to the tfhub.dev URL, keeping ?tfjs-format=file.
  shardBufs.push(await fetchFile(name, `${BASE}${name}${QUERY}`, size));
}
const totalShardBytes = shardBufs.reduce((n, b) => n + b.length, 0);
if (totalShardBytes !== expectedTotal) {
  throw new Error(`Shard bytes (${totalShardBytes}) != weights in model.json (${expectedTotal})`);
}

// Hashes: per file + combined (sha256 over model.json bytes then each shard in order).
const files = [
  { name: 'model.json', buf: modelBuf },
  ...SHARDS.map((name, i) => ({ name, buf: shardBufs[i] })),
].map(({ name, buf }) => ({ name, bytes: buf.length, sha256: sha256(buf) }));

const combinedHash = createHash('sha256');
for (const buf of [modelBuf, ...shardBufs]) combinedHash.update(buf);
const combined = combinedHash.digest('hex');

await writeFile(path.join(outDir, 'model.sha256.json'), JSON.stringify({ combined }, null, 2) + '\n');

// Keep SOURCE.md byte-stable on no-op re-runs (it embeds the download date).
if (!downloaded && (await sizeOf(path.join(outDir, 'SOURCE.md'))) !== null) {
  console.log('nothing downloaded; SOURCE.md kept');
  console.log(`combined sha256: ${combined}`);
  process.exit(0);
}

const date = new Date().toISOString().slice(0, 10);
const rows = files.map((f) => `| \`${f.name}\` | ${f.bytes} | \`${f.sha256}\` |`).join('\n');
await writeFile(
  path.join(outDir, 'SOURCE.md'),
  `# MobileNetV2 (1.0, 224) feature vector - tfjs graph model

- **Source:** ${MODEL_URL}
  (TF Hub \`google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1\`; redirects to Kaggle storage)
- **Downloaded:** ${date} by \`scripts/fetch_model.mjs\`
- **License:** Apache-2.0 (Google, MobileNetV2; https://www.apache.org/licenses/LICENSE-2.0)
- **Format:** TensorFlow.js graph model (\`tf.loadGraphModel\`), weights in 3 shards
- **Input:** \`[-1, 224, 224, 3]\` float32 RGB, values in **[0, 1]**
- **Output node:** \`module_apply_default/hub_output/feature_vector/SpatialSqueeze\` -> **1280-d** feature vector (shape \`[N, 1280]\`)

## SHA-256

| File | Bytes | SHA-256 |
|---|---|---|
${rows}

**Combined** (SHA-256 over \`model.json\` bytes followed by the three shards in order):

\`${combined}\`

The same value is stored in \`model.sha256.json\` for the app and the precompute script.
`,
);

for (const f of files) console.log(`${f.sha256}  ${f.bytes.toString().padStart(8)}  ${f.name}`);
console.log(`combined sha256: ${combined}`);
console.log(`wrote SOURCE.md and model.sha256.json in ${outDir}`);
