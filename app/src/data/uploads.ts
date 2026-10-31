// User-uploaded images: validation, decoding (center-crop to a square, resize to 224) and an in-memory store.
// Images never leave the browser. Pixel data lives here (not in React state) as plain Float32Arrays.
import * as tf from '@tensorflow/tfjs';
import { IMG_SIZE, toModelInput } from '../shared/pixelOps.js';

export const MAX_UPLOADS = 50;
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
export const ACCEPTED_TYPES: readonly string[] = ['image/jpeg', 'image/png', 'image/webp'];
export const THUMB_SIZE = 96;
export const UPLOAD_NOTICE = "Do not upload your own or family members' medical images. Images never leave this browser.";

export type UploadLabel = 'unlabeled' | 'normal' | 'pneumonia';

export interface UploadMeta {
  id: string;
  name: string;
  /** 96 px JPEG data URL of the cropped square. */
  thumb: string;
  label: UploadLabel;
}

/** 1 = pneumonia, 0 = normal, null = unlabeled. */
export function uploadLabelValue(label: UploadLabel): 0 | 1 | null {
  return label === 'pneumonia' ? 1 : label === 'normal' ? 0 : null;
}

/** Largest centered square of a w x h image: top-left corner and side length (integer px). */
export function centerCropRect(width: number, height: number): { sx: number; sy: number; side: number } {
  const side = Math.max(1, Math.floor(Math.min(width, height)));
  return { sx: Math.floor((width - side) / 2), sy: Math.floor((height - side) / 2), side };
}

/** Returns a readable problem, or null when the file can be added. */
export function validateUploadFile(file: { name: string; type: string; size: number }, currentCount: number): string | null {
  if (currentCount >= MAX_UPLOADS) return `${file.name}: at most ${MAX_UPLOADS} images can be uploaded. Remove some first.`;
  if (!ACCEPTED_TYPES.includes(file.type)) return `${file.name}: only JPEG, PNG and WebP images are accepted.`;
  if (file.size > MAX_UPLOAD_BYTES) return `${file.name}: larger than 10 MB.`;
  if (file.size === 0) return `${file.name}: the file is empty.`;
  return null;
}

// ---- pixel store ----
const pixels = new Map<string, Float32Array>();
const featureCache = new Map<string, Float32Array>();
let counter = 0;

export function newUploadId(): string {
  return `upload-${Date.now().toString(36)}-${++counter}`;
}

export function getUploadPixels(id: string): Float32Array | undefined {
  return pixels.get(id);
}
export function putUpload(id: string, data: Float32Array): void {
  pixels.set(id, data);
}
export function deleteUpload(id: string): void {
  pixels.delete(id);
  featureCache.delete(id);
}
export function clearUploadStore(): void {
  pixels.clear();
  featureCache.clear();
}
/** MobileNet features per upload id (1280 floats), so a new head can be scored without re-extracting. */
export function uploadFeatureCache(): Map<string, Float32Array> {
  return featureCache;
}

type Drawable = ImageBitmap | HTMLImageElement;

async function decodeToDrawable(file: Blob): Promise<{ src: Drawable; width: number; height: number; release: () => void }> {
  if (typeof createImageBitmap === 'function') {
    try {
      const bmp = await createImageBitmap(file);
      return { src: bmp, width: bmp.width, height: bmp.height, release: () => bmp.close() };
    } catch {
      // fall through to the <img> path
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('This file could not be read as an image.'));
      el.src = url;
    });
    return { src: img, width: img.naturalWidth, height: img.naturalHeight, release: () => URL.revokeObjectURL(url) };
  } catch (e) {
    URL.revokeObjectURL(url);
    throw e;
  }
}

/**
 * Draw any drawable (image, bitmap, video frame) center-cropped to a square onto a new size x size canvas.
 * Transparent pixels become black.
 */
export function drawSquare(src: CanvasImageSource, width: number, height: number, size: number): HTMLCanvasElement {
  const { sx, sy, side } = centerCropRect(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas is not available in this browser.');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(src, sx, sy, side, side, 0, 0, size, size);
  return canvas;
}

/** Canvas (any size) -> float32 [224, 224, 3] in [0, 1] via the shared toModelInput (RGB, alpha dropped). */
export async function canvasToPixels(canvas: HTMLCanvasElement): Promise<Float32Array> {
  const t = tf.tidy(() => toModelInput(tf, tf.browser.fromPixels(canvas, 3), { scale255: true }) as tf.Tensor3D);
  try {
    return (await t.data()) as Float32Array;
  } finally {
    t.dispose();
  }
}

export interface DecodedUpload {
  data: Float32Array;
  thumb: string;
}

/** Decode a validated file: center-crop to square, resize to 224 for the model, 96 px thumbnail for the UI. */
export async function decodeUpload(file: File): Promise<DecodedUpload> {
  const dec = await decodeToDrawable(file);
  try {
    if (dec.width < 1 || dec.height < 1) throw new Error('The image has no pixels.');
    const model = drawSquare(dec.src, dec.width, dec.height, IMG_SIZE);
    const thumbCanvas = drawSquare(dec.src, dec.width, dec.height, THUMB_SIZE);
    return { data: await canvasToPixels(model), thumb: thumbCanvas.toDataURL('image/jpeg', 0.85) };
  } finally {
    dec.release();
  }
}

export interface AddResult {
  added: UploadMeta[];
  errors: string[];
}

/** Validate and decode as many files as fit. Pixel data goes into the store; the caller keeps the metadata. */
export async function addUploadFiles(files: readonly File[], currentCount: number): Promise<AddResult> {
  const out: AddResult = { added: [], errors: [] };
  let count = currentCount;
  for (const file of files) {
    const problem = validateUploadFile(file, count);
    if (problem) {
      out.errors.push(problem);
      continue;
    }
    try {
      const dec = await decodeUpload(file);
      const id = newUploadId();
      putUpload(id, dec.data);
      out.added.push({ id, name: file.name, thumb: dec.thumb, label: 'unlabeled' });
      count++;
    } catch (e) {
      out.errors.push(`${file.name}: ${e instanceof Error ? e.message : 'could not be decoded.'}`);
    }
  }
  return out;
}
