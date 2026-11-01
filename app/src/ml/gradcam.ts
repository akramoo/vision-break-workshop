// Grad-CAM heatmaps for the logistic-regression head.
//
// The head is linear on the globally average-pooled last conv layer A (7 x 7 x K, K = 1280):
//     pooled_k = (1/49) * sum_ij A_k[i,j]          logit = sum_k w_k * pooled_k + b
// Grad-CAM weights each channel by alpha_k = mean_ij d logit / d A_k[i,j] = w_k / 49 (the gradient of the
// logit w.r.t. pooled_k is exactly w_k), so no backprop through MobileNet is needed and the result is exact:
//     cam[i,j] = ReLU( sum_k alpha_k * A_k[i,j] )
// For the "normal" class the logit is the negative one, so alpha_k = -w_k / 49. The map is then divided by
// its maximum (all zeros stay zero). Summing cam over the 49 cells gives the (ReLU-free) logit minus the bias.
// The map is only 7 x 7, so it is a coarse hint of where the evidence came from, not a segmentation.
import type { Extractor } from './extractor';
import { pixelsToBatch } from './predictImage';

export type CamClass = 'pneumonia' | 'normal';

/** The class a model predicts from P(pneumonia): same rule as the "Model says" labels (p >= 0.5 is pneumonia). */
export function predictedCamClass(p: number | null | undefined): CamClass | undefined {
  return p === null || p === undefined || Number.isNaN(p) ? undefined : p >= 0.5 ? 'pneumonia' : 'normal';
}

/** Cells per side of the activation grid. */
export const CAM_SIDE = 7;

/**
 * Normalised 7 x 7 (side x side) class activation map, row-major, values in [0, 1].
 * `activations` is [side, side, K] NHWC row-major for ONE image; K is taken from `weights.length`.
 */
export function computeCam(activations: ArrayLike<number>, weights: ArrayLike<number>, cls: CamClass): Float32Array {
  const k = weights.length;
  if (k === 0 || activations.length % k !== 0) throw new Error('computeCam: activations do not match the number of weights.');
  const cells = activations.length / k;
  const sign = cls === 'pneumonia' ? 1 : -1;
  const scale = sign / cells; // alpha_k = +-w_k / (number of cells)
  const cam = new Float32Array(cells);
  let max = 0;
  for (let c = 0; c < cells; c++) {
    let s = 0;
    const off = c * k;
    for (let j = 0; j < k; j++) s += (weights[j] as number) * (activations[off + j] as number);
    const v = Math.max(0, s * scale);
    cam[c] = v;
    if (v > max) max = v;
  }
  if (max > 0) for (let c = 0; c < cells; c++) cam[c] = (cam[c] as number) / max;
  return cam;
}

/** Side length of a square map stored as a flat array. */
function sideOf(len: number): number {
  const s = Math.round(Math.sqrt(len));
  if (s * s !== len) throw new Error(`Expected a square map, got ${len} values.`);
  return s;
}

/**
 * Bilinear upsample of a square map (pixel-centre convention, edges clamped) to outSide x outSide.
 * The four corners keep the original corner values exactly.
 */
export function upsampleCam(cam: ArrayLike<number>, outSide = 224): Float32Array {
  const side = sideOf(cam.length);
  const out = new Float32Array(outSide * outSide);
  const ratio = side / outSide;
  const lo = new Int32Array(outSide);
  const frac = new Float32Array(outSide);
  for (let o = 0; o < outSide; o++) {
    const p = Math.min(side - 1, Math.max(0, (o + 0.5) * ratio - 0.5));
    lo[o] = Math.min(side - 1, Math.floor(p));
    frac[o] = p - (lo[o] as number);
  }
  for (let y = 0; y < outSide; y++) {
    const y0 = lo[y] as number;
    const y1 = Math.min(side - 1, y0 + 1);
    const fy = frac[y] as number;
    for (let x = 0; x < outSide; x++) {
      const x0 = lo[x] as number;
      const x1 = Math.min(side - 1, x0 + 1);
      const fx = frac[x] as number;
      const top = (cam[y0 * side + x0] as number) * (1 - fx) + (cam[y0 * side + x1] as number) * fx;
      const bottom = (cam[y1 * side + x0] as number) * (1 - fx) + (cam[y1 * side + x1] as number) * fx;
      out[y * outSide + x] = top * (1 - fy) + bottom * fy;
    }
  }
  return out;
}

export interface CamPeak {
  row: number;
  col: number;
  value: number;
}

/** Cell with the highest value (first one in row-major order on ties). */
export function peakLocation(cam: ArrayLike<number>): CamPeak {
  const side = sideOf(cam.length);
  let best = 0;
  for (let i = 1; i < cam.length; i++) if ((cam[i] as number) > (cam[best] as number)) best = i;
  return { row: Math.floor(best / side), col: best % side, value: cam[best] as number };
}

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v));

/** "Jet" colormap: t in [0, 1] -> [r, g, b] in 0..255 (dark blue, cyan, yellow, dark red). */
export function jetColor(t: number): [number, number, number] {
  const x = clamp01(t);
  const r = clamp01(1.5 - Math.abs(4 * x - 3));
  const g = clamp01(1.5 - Math.abs(4 * x - 2));
  const b = clamp01(1.5 - Math.abs(4 * x - 1));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * RGBA bytes of the image with the heatmap blended on top. The overlay strength at a pixel is
 * opacity * map value, so places without evidence show the plain X-ray.
 * `pixels` is flat [side*side*3] in [0, 1]; `map` is [side*side] in [0, 1].
 */
export function blendHeatmap(pixels: ArrayLike<number>, map: ArrayLike<number>, opacity: number): Uint8ClampedArray<ArrayBuffer> {
  const n = map.length;
  if (pixels.length !== n * 3) throw new Error('blendHeatmap: pixels and map sizes do not match.');
  const out = new Uint8ClampedArray(n * 4);
  const op = clamp01(opacity);
  for (let i = 0; i < n; i++) {
    const t = clamp01(map[i] as number);
    const a = op * t;
    const [r, g, b] = jetColor(t);
    out[i * 4] = (pixels[i * 3] as number) * 255 * (1 - a) + r * a;
    out[i * 4 + 1] = (pixels[i * 3 + 1] as number) * 255 * (1 - a) + g * a;
    out[i * 4 + 2] = (pixels[i * 3 + 2] as number) * 255 * (1 - a) + b * a;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/**
 * One forward pass for one image (flat [224*224*3] in [0, 1]): pooled features plus last-conv activations.
 * Tidy-safe: all tensors created here are disposed before returning.
 */
export async function activationsForImage(
  extractor: Extractor,
  pixels: Float32Array,
): Promise<{ features: Float32Array; activations: Float32Array }> {
  const batch = pixelsToBatch([pixels]);
  try {
    return await extractor.extractWithActivations(batch);
  } finally {
    batch.dispose();
  }
}
