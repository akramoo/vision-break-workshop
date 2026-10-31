// Webcam access with readable errors, and frame -> model-input conversion.
import * as tf from '@tensorflow/tfjs';
import { toModelInput } from '../shared/pixelOps.js';
import { centerCropRect } from './uploads';

export type CameraErrorKind = 'insecure' | 'denied' | 'not-found' | 'busy' | 'other';

export interface CameraErrorInfo {
  kind: CameraErrorKind;
  title: string;
  message: string;
}

/**
 * Map whatever getUserMedia threw (or a missing API) to a readable explanation.
 * `secure` is window.isSecureContext. Pure, so it can be unit-tested.
 */
export function describeCameraError(err: unknown, secure: boolean): CameraErrorInfo {
  if (!secure) {
    return {
      kind: 'insecure',
      title: 'The camera needs a secure page',
      message:
        'Browsers only allow camera access on https:// pages or on http://localhost. Open this app from its https address, or from http://localhost / http://127.0.0.1 on this computer.',
    };
  }
  const name = typeof err === 'object' && err !== null ? String((err as { name?: unknown }).name ?? '') : '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return {
        kind: 'denied',
        title: 'Camera permission was denied',
        message:
          'Click the camera or lock icon in the browser address bar, set Camera to Allow for this site, then reload the page and press Start camera again.',
      };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return {
        kind: 'not-found',
        title: 'No camera found',
        message: 'This computer reports no usable camera. Plug one in (or enable it in your system settings) and try again.',
      };
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return {
        kind: 'busy',
        title: 'The camera is busy',
        message: 'Another app or browser tab is using the camera. Close it (video calls, other tabs), then press Start camera again.',
      };
    default:
      return {
        kind: 'other',
        title: 'The camera could not be started',
        message: `Something went wrong while opening the camera${
          err instanceof Error && err.message ? ` (${err.message})` : ''
        }. You can still use uploaded images.`,
      };
  }
}

// Every stream we open, so tests (and pagehide) can verify the camera light really goes off.
const openStreams = new Set<MediaStream>();

/** Number of media tracks this app has started and not yet stopped. */
export function liveTrackCount(): number {
  let n = 0;
  for (const s of openStreams) for (const t of s.getTracks()) if (t.readyState === 'live') n++;
  return n;
}

export class CameraError extends Error {
  readonly info: CameraErrorInfo;
  constructor(info: CameraErrorInfo) {
    super(info.message);
    this.name = 'CameraError';
    this.info = info;
  }
}

/** Open the default camera. Throws CameraError with a readable explanation. */
export async function startCamera(): Promise<MediaStream> {
  const secure = typeof window !== 'undefined' && window.isSecureContext === true;
  if (!secure) throw new CameraError(describeCameraError(null, false));
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
    throw new CameraError(describeCameraError(new Error('this browser has no camera support'), true));
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    openStreams.add(stream);
    return stream;
  } catch (e) {
    throw new CameraError(describeCameraError(e, true));
  }
}

/** Stop every track of a stream (the camera light goes off). Safe to call twice. */
export function stopStream(stream: MediaStream | null | undefined): void {
  if (!stream) return;
  stream.getTracks().forEach((t) => t.stop());
  openStreams.delete(stream);
}

/** Stop everything this app has opened. */
export function stopAllStreams(): void {
  for (const s of [...openStreams]) stopStream(s);
}

/** Current video frame -> float32 [224, 224, 3] in [0, 1], center-cropped to a square. Caller disposes. */
export function videoFrameToTensor(video: HTMLVideoElement): tf.Tensor3D {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w < 1 || h < 1) throw new Error('The camera has not produced a frame yet.');
  const { sx, sy, side } = centerCropRect(w, h);
  return tf.tidy(() => {
    const px = tf.browser.fromPixels(video, 3);
    const square = tf.slice(px, [sy, sx, 0], [side, side, 3]);
    return toModelInput(tf, square, { scale255: true }) as tf.Tensor3D;
  });
}
