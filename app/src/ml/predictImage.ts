// Run image tensors through MobileNet + the trained head. tidy-safe: inputs are never disposed,
// every temporary tensor is released before returning.
import * as tf from '@tensorflow/tfjs';
import type { Extractor } from './extractor';
import type { Head } from './head';

/** P(pneumonia) for one image [224, 224, 3] in [0, 1]. */
export async function predictImage(extractor: Extractor, head: Head, img01: tf.Tensor3D): Promise<number> {
  const feats = await extractor.extractFromTensor3D(img01);
  return head.predictProba(feats)[0] as number;
}

/** P(pneumonia) for a batch [n, 224, 224, 3] in [0, 1]. */
export async function predictBatch(extractor: Extractor, head: Head, imgs01: tf.Tensor4D): Promise<Float32Array> {
  return head.predictProba(await extractor.extractBatch(imgs01));
}

/** Probabilities for many images given as flat float32 pixel arrays (224*224*3 each), processed in batches. */
export async function predictPixelArrays(
  extractor: Extractor,
  head: Head,
  pixels: readonly Float32Array[],
  batchSize = 16,
  signal?: AbortSignal,
): Promise<Float32Array> {
  const out = new Float32Array(pixels.length);
  for (let start = 0; start < pixels.length; start += batchSize) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    const chunk = pixels.slice(start, start + batchSize);
    const batch = pixelsToBatch(chunk);
    try {
      out.set(await predictBatch(extractor, head, batch), start);
    } finally {
      batch.dispose();
    }
  }
  return out;
}

export const PIXELS_PER_IMAGE = 224 * 224 * 3;

/** Stack flat float32 images into one tensor [n, 224, 224, 3]. The caller disposes it. */
export function pixelsToBatch(pixels: readonly Float32Array[]): tf.Tensor4D {
  const flat = new Float32Array(pixels.length * PIXELS_PER_IMAGE);
  pixels.forEach((p, i) => flat.set(p, i * PIXELS_PER_IMAGE));
  return tf.tensor4d(flat, [pixels.length, 224, 224, 3]);
}
