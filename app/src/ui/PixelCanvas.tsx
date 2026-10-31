import { useEffect, useRef } from 'react';
import { MAX_THUMB_CHARS } from '../state/failureLog';

const SIDE = 224;

/** Flat float32 [224*224*3] in [0, 1] -> RGBA bytes. */
export function pixelsToRgba(data: Float32Array): Uint8ClampedArray<ArrayBuffer> {
  const n = SIDE * SIDE;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = (data[i * 3] as number) * 255;
    out[i * 4 + 1] = (data[i * 3 + 1] as number) * 255;
    out[i * 4 + 2] = (data[i * 3 + 2] as number) * 255;
    out[i * 4 + 3] = 255;
  }
  return out;
}

function toCanvas(data: Float32Array): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = SIDE;
  c.height = SIDE;
  c.getContext('2d')?.putImageData(new ImageData(pixelsToRgba(data), SIDE, SIDE), 0, 0);
  return c;
}

/**
 * 96 px JPEG data URL (quality 0.7) of the exact pixels, used for failure-log thumbnails. Lowers the quality
 * step by step in the rare case the result is longer than `maxChars`, so 100 entries stay under 1 MB.
 */
export function pixelsToDataUrl(data: Float32Array, size = 96, maxChars = MAX_THUMB_CHARS): string {
  const src = toCanvas(data);
  const out = document.createElement('canvas');
  out.width = size;
  out.height = size;
  const ctx = out.getContext('2d');
  if (!ctx) return '';
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, size, size);
  let url = '';
  for (const q of [0.7, 0.55, 0.4, 0.25]) {
    url = out.toDataURL('image/jpeg', q);
    if (url.length <= maxChars) break;
  }
  return url;
}

interface Props {
  data: Float32Array | null;
  /** CSS size in px. */
  size?: number;
  label: string;
  testId?: string;
}

/** Shows model-input pixels (what the network actually sees). */
export function PixelCanvas({ data, size = 224, label, testId }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    if (data) ctx.putImageData(new ImageData(pixelsToRgba(data), SIDE, SIDE), 0, 0);
    else ctx.clearRect(0, 0, SIDE, SIDE);
  }, [data]);
  return (
    <canvas
      ref={ref}
      width={SIDE}
      height={SIDE}
      className="pixel-canvas"
      style={{ width: size, height: size }}
      role="img"
      aria-label={label}
      {...(testId ? { 'data-testid': testId } : {})}
    />
  );
}
