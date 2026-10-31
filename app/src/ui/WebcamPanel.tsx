import { useCallback, useEffect, useRef, useState } from 'react';
import type * as tf from '@tensorflow/tfjs';
import { drawSquare } from '../data/uploads';
import {
  CameraError,
  describeCameraError,
  startCamera,
  stopStream,
  videoFrameToTensor,
  type CameraErrorInfo,
} from '../data/webcam';
import { predictImage } from '../ml/predictImage';
import { labelName } from '../state/failureLog';
import { useSession } from '../state/session';
import { EntryEditor, useAddLog, useLogConfig } from './LogThis';
import { pixelsToDataUrl } from './PixelCanvas';
import { PredictionBadge } from './PredictionBadge';

const FRAME_INTERVAL_MS = 250; // about 4 fps
type Status = 'idle' | 'starting' | 'live';

interface Frozen {
  src: string;
  p: number;
}

/** Live webcam classification. Every track is stopped on Stop, when the panel unmounts and on pagehide. */
export function WebcamPanel() {
  const { state } = useSession();
  const logConfig = useLogConfig();
  const addLog = useAddLog();
  const [lastLoggedId, setLastLoggedId] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<CameraErrorInfo | null>(() =>
    typeof window !== 'undefined' && !window.isSecureContext ? describeCameraError(null, false) : null,
  );
  const [pred, setPred] = useState<{ p: number; frames: number } | null>(null);
  const [frozen, setFrozen] = useState<Frozen | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const liveRef = useRef(false);
  const mountedRef = useRef(true);
  const framesRef = useRef(0);

  const headRef = useRef(state.head);
  headRef.current = state.head;
  const extractorRef = useRef(state.extractor);
  extractorRef.current = state.extractor;

  const stop = useCallback(() => {
    liveRef.current = false;
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    stopStream(streamRef.current);
    streamRef.current = null;
    const v = videoRef.current;
    if (v) {
      v.pause();
      v.srcObject = null;
    }
    if (mountedRef.current) {
      setStatus('idle');
      setPred(null);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    const onPageHide = () => stop();
    window.addEventListener('pagehide', onPageHide);
    return () => {
      mountedRef.current = false;
      window.removeEventListener('pagehide', onPageHide);
      stop();
    };
  }, [stop]);

  // Without a trained model the camera has nothing to show; stop it if the model disappears (Reset).
  const hasHead = state.head !== null;
  useEffect(() => {
    if (!hasHead && liveRef.current) stop();
  }, [hasHead, stop]);

  const loop = useCallback(() => {
    if (!liveRef.current) return;
    const v = videoRef.current;
    const head = headRef.current;
    const extractor = extractorRef.current;
    const schedule = () => {
      if (liveRef.current) timerRef.current = window.setTimeout(loop, FRAME_INTERVAL_MS);
    };
    if (document.hidden || !v || v.readyState < 2 || v.videoWidth === 0 || !head || !extractor) {
      schedule();
      return;
    }
    let frame: tf.Tensor3D | null = null;
    void (async () => {
      try {
        frame = videoFrameToTensor(v);
        const p = await predictImage(extractor, head, frame);
        if (liveRef.current && mountedRef.current) {
          framesRef.current += 1;
          setPred({ p, frames: framesRef.current });
        }
      } catch {
        // a dropped frame is not worth an error banner
      } finally {
        frame?.dispose();
        schedule();
      }
    })();
  }, []);

  async function start() {
    if (status !== 'idle') return;
    setError(null);
    setFrozen(null);
    setStatus('starting');
    try {
      const stream = await startCamera();
      if (!mountedRef.current) {
        stopStream(stream);
        return;
      }
      streamRef.current = stream;
      const v = videoRef.current;
      if (v) {
        v.srcObject = stream;
        await v.play().catch(() => undefined);
      }
      liveRef.current = true;
      framesRef.current = 0;
      setStatus('live');
      loop();
    } catch (e) {
      stopStream(streamRef.current);
      streamRef.current = null;
      if (mountedRef.current) {
        setStatus('idle');
        setError(e instanceof CameraError ? e.info : describeCameraError(e, window.isSecureContext));
      }
    }
  }

  async function freeze() {
    const v = videoRef.current;
    const head = headRef.current;
    const extractor = extractorRef.current;
    if (!v || !head || !extractor || v.videoWidth === 0) return;
    let frame: tf.Tensor3D | null = null;
    try {
      frame = videoFrameToTensor(v);
      const p = await predictImage(extractor, head, frame);
      const big = drawSquare(v, v.videoWidth, v.videoHeight, 224).toDataURL('image/jpeg', 0.9);
      // The thumbnail is made from the exact pixels the model saw (center-cropped, 224 px).
      const thumbnail = pixelsToDataUrl((await frame.data()) as Float32Array);
      setFrozen({ src: big, p });
      // One click logs it with the tag "other"; the editor below lets the student pick the real tag and add a hypothesis.
      const entry = addLog(
        {
          thumbnail,
          setName: 'webcam',
          trueLabel: null,
          predictedLabel: labelName(p >= 0.5),
          confidence: Math.max(p, 1 - p),
          probPneumonia: p,
          referred: Math.max(p, 1 - p) < state.abstainTau,
          config: logConfig(),
        },
        'other',
        '',
      );
      setLastLoggedId(entry ? entry.id : null);
    } finally {
      frame?.dispose();
    }
  }

  const live = status === 'live';
  const canStart = status === 'idle' && state.model.state === 'ready' && hasHead && !(error?.kind === 'insecure');
  const savedCount = state.failureLog.filter((f) => f.setName === 'webcam').length;
  const lastEntry = lastLoggedId ? state.failureLog.find((f) => f.id === lastLoggedId) : undefined;

  return (
    <section className="card" aria-labelledby="webcam-title" data-testid="webcam-panel">
      <h3 id="webcam-title">Live webcam</h3>
      <p className="hint" data-testid="webcam-hint">
        Try pointing it at a cup or your hand - the model must still pick a class. It was only ever taught "pneumonia" and "normal"; it has no
        "none of the above".
      </p>
      {!hasHead ? <p className="muted">Train a model first (Train tab), then start the camera.</p> : null}
      <div className="actions">
        <button type="button" className="btn btn--primary" onClick={() => void start()} disabled={!canStart} data-testid="webcam-start">
          {status === 'starting' ? 'Starting...' : 'Start camera'}
        </button>
        <button type="button" className="btn" onClick={stop} disabled={status === 'idle'} data-testid="webcam-stop">
          Stop camera
        </button>
        <button type="button" className="btn" onClick={() => void freeze()} disabled={!live || pred === null} data-testid="webcam-freeze">
          Freeze &amp; log
        </button>
        <span className="muted" data-testid="webcam-status">
          {live ? 'Camera on' : status === 'starting' ? 'Waiting for the camera...' : 'Camera off'}
        </span>
      </div>
      {error ? (
        <div className="error-banner" role="alert" data-testid="webcam-error" data-kind={error.kind}>
          <strong className="error-banner__title">{error.title}</strong>
          <p>{error.message}</p>
        </div>
      ) : null}

      <div className="webcam">
        <div className="webcam__view" hidden={!live}>
          <video ref={videoRef} muted playsInline autoPlay className="webcam__video" data-testid="webcam-video" />
          {pred ? (
            <div className="webcam__overlay" data-testid="webcam-prediction" data-frames={pred.frames}>
              <PredictionBadge p={pred.p} tau={state.abstainTau} />
            </div>
          ) : (
            <div className="webcam__overlay">Waiting for the first frame...</div>
          )}
        </div>
        {frozen ? (
          <div className="webcam__frozen" data-testid="webcam-frozen">
            <img src={frozen.src} alt="Frozen camera frame" width={224} height={224} />
            <PredictionBadge p={frozen.p} tau={state.abstainTau} testId="webcam-frozen-prediction" />
            <button type="button" className="btn btn--small" onClick={() => setFrozen(null)}>
              Dismiss
            </button>
            {lastEntry ? (
              <div className="log-after" data-testid="webcam-log-editor">
                <p className="muted">Saved to the failure log as "{lastEntry.tag}". Say why you think it failed:</p>
                <EntryEditor key={lastEntry.id + lastEntry.tag} entry={lastEntry} onDone={() => setLastLoggedId(null)} doneLabel="Update tag and hypothesis" />
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
      <p className="muted" data-testid="webcam-saved">
        Frozen frames saved for the failure log: {savedCount}
      </p>
    </section>
  );
}
