// Grad-CAM-style heatmap on top of an X-ray (stretch S1). Lazy: nothing is computed until "Show heatmap" is
// pressed (or `autoShow` is set), then one MobileNet pass gives the last-conv activations and every head is
// turned into a map with the head's own weights (see ml/gradcam.ts for the maths).
import { useEffect, useMemo, useRef, useState } from 'react';
import { CAM_SIDE, activationsForImage, blendHeatmap, computeCam, jetColor, peakLocation, upsampleCam, type CamClass } from '../ml/gradcam';
import type { Head } from '../ml/head';
import { useSession } from '../state/session';

const SIDE = 224;

export interface HeatmapHead {
  /** Stable key, also used in test ids. */
  id: string;
  label: string;
  head: Head;
}

interface Props {
  /** Flat float32 [224*224*3] in [0, 1]: exactly what the model sees. */
  pixels: Float32Array;
  heads: readonly HeatmapHead[];
  /** CSS size of each map in px. */
  size?: number;
  /** Start computing right away (used when the user already pressed a row button). */
  autoShow?: boolean;
  testId?: string;
  initialOpacity?: number;
}

function kernelOf(head: Head): Float32Array | null {
  try {
    return head.getWeights().kernel;
  } catch {
    return null; // disposed head
  }
}

const LEGEND_GRADIENT = `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1]
  .map((t) => `rgb(${jetColor(t).join(',')}) ${t * 100}%`)
  .join(', ')})`;

export function HeatmapOverlay({ pixels, heads, size = 224, autoShow = false, testId = 'heatmap', initialOpacity = 0.5 }: Props) {
  const { state } = useSession();
  const extractor = state.extractor;
  const [shown, setShown] = useState(autoShow);
  const [cls, setCls] = useState<CamClass>('pneumonia');
  const [opacity, setOpacity] = useState(initialOpacity);
  const [acts, setActs] = useState<{ pixels: Float32Array; data: Float32Array } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // One forward pass per displayed image; recomputed when the image changes while the heatmap is open.
  useEffect(() => {
    if (!shown || !extractor) return;
    let cancelled = false;
    setError(null);
    void activationsForImage(extractor, pixels).then(
      (r) => {
        if (!cancelled) setActs({ pixels, data: r.activations });
      },
      (e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [shown, extractor, pixels]);

  const current = acts && acts.pixels === pixels ? acts.data : null;
  const headIds = heads.map((h) => h.head.id).join(',');

  const maps = useMemo(() => {
    if (!current) return [];
    return heads.map((h) => {
      const kernel = kernelOf(h.head);
      if (!kernel) return { id: h.id, label: h.label, map: null, peak: null };
      const cam = computeCam(current, kernel, cls);
      return { id: h.id, label: h.label, map: upsampleCam(cam, SIDE), peak: peakLocation(cam) };
    });
    // `heads` is identified by head ids; labels never change for the same id.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, cls, headIds]);

  return (
    <div className="heatmap" data-testid={testId}>
      {!shown ? (
        <button type="button" className="btn btn--small" onClick={() => setShown(true)} disabled={!extractor} data-testid={`${testId}-show`}>
          Show heatmap
        </button>
      ) : (
        <>
          <div className="heatmap__controls toolbar">
            <fieldset className="choice heatmap__class">
              <legend>Evidence for</legend>
              {(['pneumonia', 'normal'] as const).map((c) => (
                <label key={c} className="choice__option">
                  <input
                    type="radio"
                    name={`${testId}-class`}
                    value={c}
                    checked={cls === c}
                    onChange={() => setCls(c)}
                    data-testid={`${testId}-class-${c}`}
                  />
                  <span>{c}</span>
                </label>
              ))}
            </fieldset>
            <label className="heatmap__opacity">
              Heatmap strength{' '}
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                data-testid={`${testId}-opacity`}
              />{' '}
              <output data-testid={`${testId}-opacity-value`}>{Math.round(opacity * 100)}%</output>
            </label>
            <button type="button" className="btn btn--small" onClick={() => setShown(false)} data-testid={`${testId}-hide`}>
              Hide heatmap
            </button>
          </div>
          {error ? (
            <p className="inline-warn" role="alert" data-testid={`${testId}-error`}>
              Could not compute the heatmap: {error}
            </p>
          ) : null}
          {!current && !error ? (
            <p role="status" className="muted" data-testid={`${testId}-status`}>
              Computing heatmap...
            </p>
          ) : null}
          <div className="before-after heatmap__maps">
            {maps.map((m) => (
              <figure key={m.id} data-testid={`${testId}-fig-${m.id}`}>
                {m.map ? (
                  <HeatCanvas
                    pixels={pixels}
                    map={m.map}
                    opacity={opacity}
                    size={size}
                    label={`${m.label}: evidence for ${cls} (the brighter the colour, the stronger)`}
                    testId={`${testId}-canvas-${m.id}`}
                    peak={m.peak ? `${m.peak.row},${m.peak.col}` : ''}
                  />
                ) : (
                  <p className="muted">This model is no longer available.</p>
                )}
                <figcaption>{m.label}</figcaption>
              </figure>
            ))}
          </div>
          <div className="heatmap__legend" aria-hidden="true">
            <span>low</span>
            <span className="heatmap__bar" style={{ background: LEGEND_GRADIENT }} />
            <span>high</span>
          </div>
          <p className="muted heatmap__caption" data-testid={`${testId}-caption`}>
            Warm colours show where the model found evidence for the chosen class. It is a hint, not proof. (The map is only {CAM_SIDE} x{' '}
            {CAM_SIDE} cells, so it is blurry.)
          </p>
        </>
      )}
    </div>
  );
}

function HeatCanvas({
  pixels,
  map,
  opacity,
  size,
  label,
  testId,
  peak,
}: {
  pixels: Float32Array;
  map: Float32Array;
  opacity: number;
  size: number;
  label: string;
  testId: string;
  peak: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    ctx.putImageData(new ImageData(blendHeatmap(pixels, map, opacity), SIDE, SIDE), 0, 0);
  }, [pixels, map, opacity]);
  return (
    <canvas
      ref={ref}
      width={SIDE}
      height={SIDE}
      className="pixel-canvas"
      style={{ width: size, height: size }}
      role="img"
      aria-label={label}
      data-testid={testId}
      data-peak={peak}
      data-opacity={opacity}
    />
  );
}
