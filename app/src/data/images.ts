// Image URLs and decoding of the shipped X-rays into model-ready tensors.
import * as tf from '@tensorflow/tfjs';
import { IMG_SIZE, toModelInput } from '../shared/pixelOps.js';
import { dataBaseUrl, type ManifestEntry } from './manifest';

/** URL of an X-ray from its manifest path. Plain URL for <img loading="lazy"> thumbnails. */
export function imageUrl(path: string): string {
  return `${dataBaseUrl()}${path}`;
}

/** Decode an <img> element from a URL; rejects with a readable error. */
export function loadHtmlImage(url: string, signal?: AbortSignal): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load image ${url}`));
    img.src = url;
  });
}

/**
 * Image -> float32 [224, 224, 3] in [0, 1], via the shared toModelInput so that live
 * extraction matches the precomputed features. The caller owns (and must dispose) the tensor.
 */
export function imageToTensor(img: HTMLImageElement): tf.Tensor3D {
  return tf.tidy(() => {
    const px = tf.browser.fromPixels(img, 3);
    return toModelInput(tf, px, { scale255: true }) as tf.Tensor3D;
  });
}

/** Load one manifest image as a model-input tensor [224, 224, 3] in [0, 1]. Caller disposes. */
export async function loadImageTensor(entry: Pick<ManifestEntry, 'path'>, signal?: AbortSignal): Promise<tf.Tensor3D> {
  const img = await loadHtmlImage(imageUrl(entry.path), signal);
  try {
    const t = imageToTensor(img);
    const [h, w] = t.shape;
    if (h !== IMG_SIZE || w !== IMG_SIZE) {
      t.dispose();
      throw new Error(`Unexpected image size for ${entry.path}`);
    }
    return t;
  } finally {
    img.src = ''; // release the decoded bitmap sooner
  }
}
