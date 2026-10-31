// Pixels of a shipped X-ray exactly as the model sees them, for failure-log thumbnails.
import * as tf from '@tensorflow/tfjs';
import { loadImageTensor } from '../data/images';
import type { ManifestEntry } from '../data/manifest';
import { STRESS_PRESET, applyPerturbation } from '../shared/pixelOps.js';

/** Float32 [224*224*3] in [0, 1]. For the stress set the fixed stress preset is applied, as the stress features were. */
export async function entryPixels(entry: ManifestEntry, stress: boolean): Promise<Float32Array> {
  const t = await loadImageTensor(entry);
  try {
    if (!stress) return (await t.data()) as Float32Array;
    const out = tf.tidy(() => applyPerturbation(tf, t, STRESS_PRESET));
    try {
      return (await out.data()) as Float32Array;
    } finally {
      out.dispose();
    }
  } finally {
    t.dispose();
  }
}
