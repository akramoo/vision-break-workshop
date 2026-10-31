import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  DEFAULT_STAMP,
  IMG_SIZE,
  STRESS_PRESET,
  applyPerturbation,
  applyStamp,
  mulberry32,
  randomAugmentParams,
  stampBox,
  toModelInput,
} from '../../src/shared/pixelOps.js';

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

/** Deterministic pseudo-image in [0, 1]. */
function testImage(): tf.Tensor3D {
  const rng = mulberry32(7);
  const data = new Float32Array(IMG_SIZE * IMG_SIZE * 3).map(() => 0.1 + 0.8 * rng());
  return tf.tensor3d(data, [IMG_SIZE, IMG_SIZE, 3]);
}

describe('pixelOps smoke', () => {
  it('identity perturbation equals the input', async () => {
    const img = testImage();
    const out = applyPerturbation(tf, img, {});
    const diff = tf.max(tf.abs(tf.sub(out, img)));
    expect((await diff.data())[0]).toBeLessThan(1e-6);
    tf.dispose([img, out, diff]);
  });

  it('stamp changes pixels only inside the box', async () => {
    const img = testImage();
    const out = applyStamp(tf, img, DEFAULT_STAMP);
    const a = (await img.data()) as Float32Array;
    const b = (await out.data()) as Float32Array;
    const { x0, y0, x1, y1 } = stampBox(DEFAULT_STAMP);
    let changedInside = 0;
    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        const inside = x >= x0 && x < x1 && y >= y0 && y < y1;
        for (let c = 0; c < 3; c++) {
          const i = (y * IMG_SIZE + x) * 3 + c;
          if (inside) {
            if (a[i] !== b[i]) changedInside++;
          } else {
            expect(b[i]).toBe(a[i]);
          }
        }
      }
    }
    expect(changedInside).toBeGreaterThan(0);
    tf.dispose([img, out]);
  });

  it('stress preset output stays in [0, 1], keeps its shape and is seed-reproducible', async () => {
    const img = testImage();
    const out = applyPerturbation(tf, img, STRESS_PRESET);
    expect(out.shape).toEqual([IMG_SIZE, IMG_SIZE, 3]);
    const [min, max] = await Promise.all([tf.min(out).data(), tf.max(out).data()]);
    expect(min[0]).toBeGreaterThanOrEqual(0);
    expect(max[0]).toBeLessThanOrEqual(1);
    expect(randomAugmentParams(mulberry32(1))).toEqual(randomAugmentParams(mulberry32(1)));
    tf.dispose([img, out]);
  });

  it('toModelInput requires scale255 and scales / expands correctly', async () => {
    const gray = tf.fill([IMG_SIZE, IMG_SIZE, 1], 255, 'int32') as tf.Tensor3D;
    expect(() => toModelInput(tf, gray, {} as { scale255: boolean })).toThrow();
    const out = toModelInput(tf, gray, { scale255: true });
    expect(out.shape).toEqual([IMG_SIZE, IMG_SIZE, 3]);
    expect(out.dtype).toBe('float32');
    expect((await tf.mean(out).data())[0]).toBeCloseTo(1, 6);
    tf.dispose([gray, out]);
  });

  it('positive rotation is counter-clockwise (right edge moves to the top)', async () => {
    const data = new Float32Array(IMG_SIZE * IMG_SIZE * 3);
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) data.fill(1, ((112 + dy) * IMG_SIZE + 200 + dx) * 3, ((112 + dy) * IMG_SIZE + 200 + dx) * 3 + 3);
    }
    const img = tf.tensor3d(data, [IMG_SIZE, IMG_SIZE, 3]);
    const out = applyPerturbation(tf, img, { rotationDeg: 90 });
    const gray = (await tf.mean(out, 2).data()) as Float32Array;
    const peak = gray.indexOf(Math.max(...gray));
    expect(Math.floor(peak / IMG_SIZE)).toBeLessThan(30); // moved near the top row
    tf.dispose([img, out]);
  });
});
