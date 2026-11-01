import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { ACTIVATION_NODE, extractWithActivationsFrom } from '../../src/ml/extractor';
import { blendHeatmap, computeCam, jetColor, peakLocation, predictedCamClass, upsampleCam } from '../../src/ml/gradcam';
import { FEATURE_DIM, FEATURE_OUTPUT_NODE, IMG_SIZE, mulberry32 } from '../../src/shared/pixelOps.js';

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

// 2 x 2 grid, 2 channels. Rows are cells (row-major), columns are channels.
const ACTS = Float32Array.from([1, 0, /**/ 0, 2, /**/ 1, 1, /**/ 3, 0]);
const W = Float32Array.from([2, -1]);

describe('computeCam', () => {
  it('matches a hand-computed example (pneumonia)', () => {
    // evidence per cell = w . a = [2, -2, 1, 6]; ReLU -> [2, 0, 1, 6]; divide by max -> [1/3, 0, 1/6, 1]
    const cam = computeCam(ACTS, W, 'pneumonia');
    expect(cam).toHaveLength(4);
    expect(cam[0]).toBeCloseTo(1 / 3, 6);
    expect(cam[1]).toBe(0);
    expect(cam[2]).toBeCloseTo(1 / 6, 6);
    expect(cam[3]).toBeCloseTo(1, 6);
  });

  it('flips the sign of the weights for the normal class', () => {
    // -w . a = [-2, 2, -1, -6]; ReLU -> [0, 2, 0, 0]; normalised -> [0, 1, 0, 0]
    expect(Array.from(computeCam(ACTS, W, 'normal'))).toEqual([0, 1, 0, 0]);
  });

  it('is exactly the negative-weight map for the other class', () => {
    const neg = Float32Array.from(W, (v) => -v);
    expect(Array.from(computeCam(ACTS, neg, 'pneumonia'))).toEqual(Array.from(computeCam(ACTS, W, 'normal')));
  });

  it('normalises to a maximum of 1 and is invariant to the scale of the weights', () => {
    const a = computeCam(ACTS, W, 'pneumonia');
    const b = computeCam(ACTS, Float32Array.from(W, (v) => v * 37), 'pneumonia');
    expect(Math.max(...a)).toBeCloseTo(1, 6);
    a.forEach((v, i) => expect(b[i]).toBeCloseTo(v, 5));
  });

  it('removes negative evidence (ReLU) and returns all zeros when there is none', () => {
    const acts = Float32Array.from([1, 1, 2, 2]); // 4 cells, 1 channel
    expect(Array.from(computeCam(acts, Float32Array.from([-1]), 'pneumonia'))).toEqual([0, 0, 0, 0]);
    expect(Array.from(computeCam(acts, Float32Array.from([0]), 'pneumonia'))).toEqual([0, 0, 0, 0]);
    expect(Array.from(computeCam(acts, Float32Array.from([-1]), 'normal'))).toEqual([0.5, 0.5, 1, 1]);
  });

  it('takes the channel count from the weights and rejects mismatched sizes', () => {
    expect(computeCam(new Float32Array(9 * 5).fill(1), new Float32Array(5).fill(1), 'pneumonia')).toHaveLength(9);
    expect(() => computeCam(new Float32Array(7), new Float32Array(2), 'pneumonia')).toThrow();
    expect(() => computeCam(new Float32Array(4), new Float32Array(0), 'pneumonia')).toThrow();
  });
});

describe('upsampleCam', () => {
  const cam = Float32Array.from([0, 0.25, 0.5, 0.75, 1, 0.1, 0.3, 0.6, 0.9]); // 3 x 3

  it('keeps the four corner values exactly', () => {
    const up = upsampleCam(cam, 12);
    expect(up).toHaveLength(144);
    expect(up[0]).toBe(0);
    expect(up[11]).toBeCloseTo(0.5, 6);
    expect(up[11 * 12]).toBeCloseTo(0.3, 6);
    expect(up[143]).toBeCloseTo(0.9, 6);
  });

  it('is monotonic along a ramp and stays within the input range', () => {
    const ramp = Float32Array.from({ length: 49 }, (_, i) => (i % 7) / 6); // increases left to right
    const up = upsampleCam(ramp, 224);
    for (let y = 0; y < 224; y += 37) {
      for (let x = 1; x < 224; x++) expect(up[y * 224 + x] as number).toBeGreaterThanOrEqual(up[y * 224 + x - 1] as number);
    }
    expect(Math.min(...up)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...up)).toBeLessThanOrEqual(1 + 1e-6);
    expect(up[0]).toBe(0);
    expect(up[223]).toBeCloseTo(1, 6);
  });

  it('keeps a constant map constant and rejects non-square input', () => {
    const up = upsampleCam(new Float32Array(49).fill(0.4), 28);
    up.forEach((v) => expect(v).toBeCloseTo(0.4, 6));
    expect(() => upsampleCam(new Float32Array(10), 8)).toThrow();
  });
});

describe('peakLocation', () => {
  it('finds the row, column and value of the maximum (first one on ties)', () => {
    const cam = new Float32Array(49);
    cam[3 * 7 + 5] = 0.75;
    expect(peakLocation(cam)).toEqual({ row: 3, col: 5, value: 0.75 });
    cam[1 * 7 + 1] = 0.75;
    expect(peakLocation(cam)).toEqual({ row: 1, col: 1, value: 0.75 });
  });
});

describe('jetColor', () => {
  it('has dark blue at 0, dark red at 1 and clamps outside [0, 1]', () => {
    expect(jetColor(0)).toEqual([0, 0, 128]);
    expect(jetColor(1)).toEqual([128, 0, 0]);
    expect(jetColor(-3)).toEqual(jetColor(0));
    expect(jetColor(9)).toEqual(jetColor(1));
  });

  it('goes through green in the middle and gets warmer with t', () => {
    const [r, g, b] = jetColor(0.5);
    expect(g).toBe(255);
    expect(r).toBeGreaterThan(100);
    expect(b).toBeGreaterThan(100);
    expect(jetColor(0.9)[0]).toBeGreaterThan(jetColor(0.3)[0]);
    expect(jetColor(0.9)[2]).toBeLessThan(jetColor(0.3)[2]);
  });
});

describe('blendHeatmap', () => {
  it('shows the plain image where the map is 0 and the opacity-weighted colour where it is 1', () => {
    const pixels = Float32Array.from([1, 0, 0, 0, 1, 0]); // two pixels: red, green
    const out = blendHeatmap(pixels, Float32Array.from([0, 1]), 0.5);
    expect(Array.from(out.slice(0, 4))).toEqual([255, 0, 0, 255]);
    const [r, g, b] = jetColor(1);
    expect(out[4]).toBeCloseTo((0 * 0.5 + r * 0.5), 0);
    expect(Math.abs((out[5] as number) - (255 * 0.5 + g * 0.5))).toBeLessThanOrEqual(0.5);
    expect(out[6]).toBeCloseTo(b * 0.5, 0);
    expect(blendHeatmap(pixels, Float32Array.from([1, 1]), 0).slice(4, 8)).toEqual(Uint8ClampedArray.from([0, 255, 0, 255]));
  });

  it('rejects mismatched sizes', () => {
    expect(() => blendHeatmap(new Float32Array(5), new Float32Array(2), 0.5)).toThrow();
  });
});

// ---- real MobileNet in node (tfjs CPU backend) ----
const here = path.dirname(fileURLToPath(import.meta.url));
const modelDir = path.resolve(here, '../../public/models/mobilenet_v2');
const haveModel = existsSync(path.join(modelDir, 'model.json')) && existsSync(path.join(modelDir, 'group1-shard1of3.bin'));

async function loadModelFromDisk(): Promise<tf.GraphModel> {
  const json = JSON.parse(readFileSync(path.join(modelDir, 'model.json'), 'utf8')) as {
    modelTopology: tf.io.ModelArtifacts['modelTopology'];
    format?: string;
    generatedBy?: string;
    weightsManifest: { paths: string[]; weights: tf.io.WeightsManifestEntry[] }[];
  };
  const shards = json.weightsManifest
    .flatMap((g) => g.paths)
    .map((p) => {
      const b = readFileSync(path.join(modelDir, p));
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
    });
  return tf.loadGraphModel(
    tf.io.fromMemory({
      modelTopology: json.modelTopology,
      weightSpecs: json.weightsManifest.flatMap((g) => g.weights),
      weightData: tf.io.concatenateArrayBuffers(shards),
      ...(json.format !== undefined ? { format: json.format } : {}),
      ...(json.generatedBy !== undefined ? { generatedBy: json.generatedBy } : {}),
    }),
  );
}

/** A smooth-ish deterministic test image in [0, 1]. */
function testImage(seed: number): tf.Tensor4D {
  const rng = mulberry32(seed);
  const data = new Float32Array(IMG_SIZE * IMG_SIZE * 3);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const base = 0.5 + 0.4 * Math.sin((x + 3 * seed) / 17) * Math.cos(y / 23);
      for (let c = 0; c < 3; c++) data[(y * IMG_SIZE + x) * 3 + c] = Math.min(1, Math.max(0, base + (rng() - 0.5) * 0.1));
    }
  }
  return tf.tensor4d(data, [1, IMG_SIZE, IMG_SIZE, 3]);
}

describe.skipIf(!haveModel)('real MobileNet activations', () => {
  let model: tf.GraphModel;
  beforeAll(async () => {
    model = await loadModelFromDisk();
  }, 120_000);

  it('returns [1,7,7,1280] activations whose 7x7 mean equals the SpatialSqueeze features', async () => {
    const img = testImage(1);
    try {
      const { features, activations } = await extractWithActivationsFrom(model, img);
      expect(features).toHaveLength(FEATURE_DIM);
      expect(activations).toHaveLength(7 * 7 * FEATURE_DIM);

      // Shape check straight from the graph.
      const act = tf.tidy(() => model.execute({ images: img }, ACTIVATION_NODE) as tf.Tensor);
      expect(act.shape).toEqual([1, 7, 7, FEATURE_DIM]);
      act.dispose();

      let worst = 0;
      for (let k = 0; k < FEATURE_DIM; k++) {
        let s = 0;
        for (let cell = 0; cell < 49; cell++) s += activations[cell * FEATURE_DIM + k] as number;
        worst = Math.max(worst, Math.abs(s / 49 - (features[k] as number)));
      }
      expect(worst).toBeLessThan(1e-4);

      // Same features as the path the app already uses for predictions.
      const plain = tf.tidy(() => tf.reshape(model.execute({ images: img }, FEATURE_OUTPUT_NODE) as tf.Tensor, [1, FEATURE_DIM]));
      const plainData = (await plain.data()) as Float32Array;
      plain.dispose();
      let diff = 0;
      for (let k = 0; k < FEATURE_DIM; k++) diff = Math.max(diff, Math.abs((plainData[k] as number) - (features[k] as number)));
      expect(diff).toBeLessThan(1e-5);
    } finally {
      img.dispose();
    }
  }, 60_000);

  it('Grad-CAM sums to the logit (minus bias) and does not leak tensors', async () => {
    const img = testImage(2);
    try {
      const { features, activations } = await extractWithActivationsFrom(model, img);
      const rng = mulberry32(7);
      const w = Float32Array.from({ length: FEATURE_DIM }, () => (rng() - 0.5) * 0.2);

      // Exactness: sum over cells of (w . A_cell) / 49 is w . pooled_features, i.e. the logit without the bias.
      let cellSum = 0;
      for (let cell = 0; cell < 49; cell++) {
        for (let k = 0; k < FEATURE_DIM; k++) cellSum += ((w[k] as number) * (activations[cell * FEATURE_DIM + k] as number)) / 49;
      }
      let logit = 0;
      for (let k = 0; k < FEATURE_DIM; k++) logit += (w[k] as number) * (features[k] as number);
      expect(cellSum).toBeCloseTo(logit, 3);

      const before = tf.memory().numTensors;
      for (let i = 0; i < 50; i++) {
        const cam = computeCam(activations, w, i % 2 === 0 ? 'pneumonia' : 'normal');
        expect(cam).toHaveLength(49);
        cam.forEach((v) => {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(1);
        });
        upsampleCam(cam, 224);
      }
      // Also the forward pass itself must release everything it creates.
      await extractWithActivationsFrom(model, img);
      await extractWithActivationsFrom(model, img);
      expect(tf.memory().numTensors).toBe(before);
    } finally {
      img.dispose();
    }
  }, 60_000);

  it('rejects a wrongly shaped input', async () => {
    const bad = tf.zeros([1, 100, 100, 3]) as tf.Tensor4D;
    try {
      await expect(extractWithActivationsFrom(model, bad)).rejects.toThrow(/expects/);
    } finally {
      bad.dispose();
    }
  });
});

describe('predictedCamClass', () => {
  it('maps P(pneumonia) to the class the model answered (p >= 0.5 is pneumonia)', () => {
    expect(predictedCamClass(0.9)).toBe('pneumonia');
    expect(predictedCamClass(0.5)).toBe('pneumonia');
    expect(predictedCamClass(0.49)).toBe('normal');
    expect(predictedCamClass(0)).toBe('normal');
  });

  it('is undefined when there is no prediction yet', () => {
    expect(predictedCamClass(null)).toBeUndefined();
    expect(predictedCamClass(undefined)).toBeUndefined();
    expect(predictedCamClass(Number.NaN)).toBeUndefined();
  });
});
