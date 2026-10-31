// Shared pixel operations: the single source of truth for every image step that
// must be identical in the browser app (@tensorflow/tfjs) and in the offline
// precompute script (@tensorflow/tfjs-node).
//
// Rules for everything exported here:
//   * plain ES module, no imports; the `tf` namespace is always the FIRST argument
//   * every function is tf.tidy-safe, returns NEW tensors and never disposes its inputs
//   * images are float32 in [0, 1], layout [h, w, 3] (or [n, h, w, 3] where noted)
//   * deterministic: no canvas, no Math.random (randomness comes from a seeded rng)

/** Side length (px) of the square image MobileNet v2 expects. */
export const IMG_SIZE = 224;

/** Name of the graph output node holding the 1280-d feature vector. */
export const FEATURE_OUTPUT_NODE = 'module_apply_default/hub_output/feature_vector/SpatialSqueeze';

/** Length of the feature vector produced by MobileNet v2 (Conv_1 -> average pool). */
export const FEATURE_DIM = 1280;

// ---------------------------------------------------------------------------
// Seeded RNG
// ---------------------------------------------------------------------------

/**
 * mulberry32: small, fast, seedable PRNG shared by scripts and app.
 * @param {number} seed any integer (coerced to uint32)
 * @returns {() => number} function returning floats in [0, 1)
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Model input
// ---------------------------------------------------------------------------

/**
 * Convert an image tensor into the model input: float32 [224, 224, 3] in [0, 1].
 *
 * The caller must say how the values are scaled, because guessing from the data
 * (for example "max > 1.5 means 0..255") is fragile:
 *   scale255 = true  -> values are 0..255 (uint8 / int32 / float32 from a decoded JPEG)
 *   scale255 = false -> values are already 0..1
 * Grayscale [h, w, 1] is replicated to 3 channels; RGBA [h, w, 4] drops alpha.
 * Resizing (bilinear) only happens when the input is not already 224 x 224.
 *
 * @param {object} tf TensorFlow.js namespace
 * @param {import('@tensorflow/tfjs-core').Tensor3D} img [h, w, c] with c in {1, 3, 4}
 * @param {{scale255: boolean}} opts
 * @returns {import('@tensorflow/tfjs-core').Tensor3D} new float32 [224, 224, 3]
 */
export function toModelInput(tf, img, opts) {
  if (!opts || typeof opts.scale255 !== 'boolean') {
    throw new Error('toModelInput: opts.scale255 (boolean) is required');
  }
  if (img.rank !== 3) throw new Error(`toModelInput: expected a 3D tensor, got rank ${img.rank}`);
  const [h, w, c] = img.shape;
  if (c !== 1 && c !== 3 && c !== 4) throw new Error(`toModelInput: unsupported channel count ${c}`);
  return tf.tidy(() => {
    let x = tf.cast(img, 'float32');
    if (opts.scale255) x = tf.div(x, 255);
    if (c === 1) x = tf.tile(x, [1, 1, 3]);
    else if (c === 4) x = tf.slice(x, [0, 0, 0], [h, w, 3]);
    if (h !== IMG_SIZE || w !== IMG_SIZE) {
      x = tf.image.resizeBilinear(x, [IMG_SIZE, IMG_SIZE]);
    }
    return tf.clipByValue(x, 0, 1);
  });
}

// ---------------------------------------------------------------------------
// The "R" stamp (shortcut lab)
// ---------------------------------------------------------------------------

/** 5 columns x 7 rows bitmap of the letter R ('#' = ink). */
export const R_GLYPH = Object.freeze([
  '####.',
  '#...#',
  '#...#',
  '####.',
  '#.#..',
  '#..#.',
  '#...#',
]);

const GLYPH_COLS = 5;
const GLYPH_ROWS = 7;
// Glyph plus a 1-cell margin on every side.
const GRID_COLS = GLYPH_COLS + 2;
const GRID_ROWS = GLYPH_ROWS + 2;

export const STAMP_POSITIONS = Object.freeze(['top-left', 'top-right', 'bottom-left', 'bottom-right']);

/** Default stamp used by the precomputed `stamped` embeddings and the stamp lab. */
export const DEFAULT_STAMP = Object.freeze({
  size: 48,
  position: 'top-left',
  offsetX: 8,
  offsetY: 8,
  opacity: 1,
});

/**
 * Fill in defaults and clamp a (possibly partial) stamp config.
 * size is rounded to an integer in [8, 96]; opacity is clamped to [0, 1];
 * offsets are the inset in px from the chosen corner (default 8).
 * @param {Partial<import('./pixelOps').StampConfig>} [cfg]
 * @returns {import('./pixelOps').StampConfig}
 */
export function resolveStampConfig(cfg = {}) {
  const num = (v, d) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
  const position = STAMP_POSITIONS.includes(cfg.position) ? cfg.position : DEFAULT_STAMP.position;
  return {
    size: Math.min(96, Math.max(8, Math.round(num(cfg.size, DEFAULT_STAMP.size)))),
    position,
    offsetX: Math.round(num(cfg.offsetX, DEFAULT_STAMP.offsetX)),
    offsetY: Math.round(num(cfg.offsetY, DEFAULT_STAMP.offsetY)),
    opacity: Math.min(1, Math.max(0, num(cfg.opacity, DEFAULT_STAMP.opacity))),
  };
}

/**
 * Pixel rectangle of the stamp box, in image coordinates. The rectangle may
 * extend past the image; callers clip. x1/y1 are exclusive.
 * @param {Partial<import('./pixelOps').StampConfig>} cfg
 * @param {number} [height]
 * @param {number} [width]
 * @returns {{x0: number, y0: number, x1: number, y1: number}}
 */
export function stampBox(cfg, height = IMG_SIZE, width = IMG_SIZE) {
  const c = resolveStampConfig(cfg);
  const x0 = c.position.endsWith('left') ? c.offsetX : width - c.size - c.offsetX;
  const y0 = c.position.startsWith('top') ? c.offsetY : height - c.size - c.offsetY;
  return { x0, y0, x1: x0 + c.size, y1: y0 + c.size };
}

/**
 * Build the stamp layers for an image of the given size.
 *
 * Design choice: a WHITE filled square containing a BLACK "R". Chest X-ray
 * corners are mostly dark, so a white box is the most visible marker, and the
 * black glyph keeps it recognisable on bright backgrounds as well. The glyph is
 * the 5x7 bitmap scaled with nearest-neighbour sampling to fit a 9-row grid
 * (7 glyph rows + 1 cell of margin above and below), centred horizontally.
 *
 * Returns two [h, w, 1] tensors: `alpha` (opacity inside the box, 0 elsewhere)
 * and `color` (0 on glyph pixels, 1 on the white box). The box is clipped at
 * the image borders.
 *
 * @param {object} tf
 * @param {Partial<import('./pixelOps').StampConfig>} [cfg]
 * @param {number} [height]
 * @param {number} [width]
 * @returns {{alpha: import('@tensorflow/tfjs-core').Tensor3D, color: import('@tensorflow/tfjs-core').Tensor3D}}
 */
export function makeStampMask(tf, cfg = {}, height = IMG_SIZE, width = IMG_SIZE) {
  const c = resolveStampConfig(cfg);
  const { x0, y0, x1, y1 } = stampBox(c, height, width);
  const alpha = new Float32Array(height * width);
  const color = new Float32Array(height * width);
  const cell = c.size / GRID_ROWS;
  const xInset = (c.size - GRID_COLS * cell) / 2;
  for (let y = Math.max(0, y0); y < Math.min(height, y1); y++) {
    const gy = Math.floor((y - y0 + 0.5) / cell) - 1;
    const row = gy >= 0 && gy < GLYPH_ROWS ? R_GLYPH[gy] : undefined;
    for (let x = Math.max(0, x0); x < Math.min(width, x1); x++) {
      const gx = Math.floor((x - x0 + 0.5 - xInset) / cell) - 1;
      const ink = row !== undefined && gx >= 0 && gx < GLYPH_COLS && row[gx] === '#';
      const i = y * width + x;
      alpha[i] = c.opacity;
      color[i] = ink ? 0 : 1;
    }
  }
  return {
    alpha: tf.tensor3d(alpha, [height, width, 1]),
    color: tf.tensor3d(color, [height, width, 1]),
  };
}

/**
 * Stamp an "R" marker onto an image. Pixels outside the box are returned
 * bit-identical. Works on [h, w, 3] and batches [n, h, w, 3].
 *
 * @param {object} tf
 * @param {import('@tensorflow/tfjs-core').Tensor3D | import('@tensorflow/tfjs-core').Tensor4D} img01 float32 in [0, 1]
 * @param {Partial<import('./pixelOps').StampConfig>} [cfg] defaults to DEFAULT_STAMP
 * @returns {import('@tensorflow/tfjs-core').Tensor3D | import('@tensorflow/tfjs-core').Tensor4D} new tensor, same shape
 */
export function applyStamp(tf, img01, cfg = DEFAULT_STAMP) {
  const rank = img01.rank;
  if (rank !== 3 && rank !== 4) throw new Error(`applyStamp: expected rank 3 or 4, got ${rank}`);
  const h = img01.shape[rank - 3];
  const w = img01.shape[rank - 2];
  return tf.tidy(() => {
    const { alpha, color } = makeStampMask(tf, cfg, h, w);
    return tf.add(tf.mul(img01, tf.sub(1, alpha)), tf.mul(color, alpha));
  });
}

// ---------------------------------------------------------------------------
// Perturbations (Break lab) and augmentation
// ---------------------------------------------------------------------------

/** Parameters that leave an image unchanged. */
export const IDENTITY_PERTURBATION = Object.freeze({
  brightness: 0,
  contrast: 1,
  rotationDeg: 0,
  cropFrac: 1,
});

/** Slider ranges exposed by the UI (inclusive). */
export const PERTURBATION_RANGES = Object.freeze({
  brightness: Object.freeze([-0.5, 0.5]),
  contrast: Object.freeze([0.5, 1.5]),
  rotationDeg: Object.freeze([-30, 30]),
  cropFrac: Object.freeze([0.6, 1]),
});

/** Fixed degradation behind the "Kermany stress" test set. */
export const STRESS_PRESET = Object.freeze({
  brightness: -0.25,
  contrast: 0.7,
  rotationDeg: 10,
  cropFrac: 0.85,
});

/** Random augmentation ranges: brightness +-0.3, contrast [0.7, 1.3], rotation +-15 deg, crop [0.8, 1.0]. */
export const AUG_PARAMS = Object.freeze({
  brightness: 0.3,
  contrast: Object.freeze([0.7, 1.3]),
  rotationDeg: 15,
  cropFrac: Object.freeze([0.8, 1.0]),
});

/**
 * Draw one set of random perturbation params from AUG_PARAMS.
 * Always consumes exactly 4 values from `rng`, in a fixed order, so scripts and
 * app generate the same views from the same seed.
 * @param {() => number} rng returns floats in [0, 1) (see mulberry32)
 * @returns {import('./pixelOps').PerturbationParams}
 */
export function randomAugmentParams(rng) {
  const sym = (m) => (rng() * 2 - 1) * m;
  const between = (lo, hi) => lo + rng() * (hi - lo);
  const brightness = sym(AUG_PARAMS.brightness);
  const contrast = between(AUG_PARAMS.contrast[0], AUG_PARAMS.contrast[1]);
  const rotationDeg = sym(AUG_PARAMS.rotationDeg);
  const cropFrac = between(AUG_PARAMS.cropFrac[0], AUG_PARAMS.cropFrac[1]);
  return { brightness, contrast, rotationDeg, cropFrac };
}

/**
 * Apply brightness / contrast / center-crop / rotation to an image in [0, 1].
 *
 * Order (documented because it matters): contrast about the image mean, then
 * additive brightness, clip to [0, 1], then center crop resized back to 224,
 * then rotation about the image center. Photometric steps come first so the
 * black fill exposed by rotation stays black (0).
 *
 *   brightness  additive, -0.5..0.5
 *   contrast    multiplicative around the per-image mean, 0.5..1.5
 *   rotationDeg counter-clockwise, -30..30, bilinear, fill 0
 *   cropFrac    0.6..1.0 of the side length, centered, resized back with bilinear
 *
 * Rotation uses tf.image.transform (bilinear, constant fill) rather than
 * tf.image.rotateWithOffset, which only does nearest-neighbour sampling.
 * Steps whose parameter is the identity are skipped, so identity params return
 * the input values exactly. Works on [h, w, 3] and batches [n, h, w, 3].
 *
 * @param {object} tf
 * @param {import('@tensorflow/tfjs-core').Tensor3D | import('@tensorflow/tfjs-core').Tensor4D} img01 float32 in [0, 1]
 * @param {Partial<import('./pixelOps').PerturbationParams>} [params]
 * @returns {import('@tensorflow/tfjs-core').Tensor3D | import('@tensorflow/tfjs-core').Tensor4D} new float32 tensor in [0, 1], same shape
 */
export function applyPerturbation(tf, img01, params = {}) {
  const rank = img01.rank;
  if (rank !== 3 && rank !== 4) throw new Error(`applyPerturbation: expected rank 3 or 4, got ${rank}`);
  const p = { ...IDENTITY_PERTURBATION, ...params };
  return tf.tidy(() => {
    let x = rank === 3 ? tf.expandDims(img01, 0) : img01; // [n, h, w, c]
    const [, h, w] = x.shape;

    if (p.contrast !== 1) {
      const mean = tf.mean(x, [1, 2, 3], true);
      x = tf.add(tf.mul(tf.sub(x, mean), p.contrast), mean);
    }
    if (p.brightness !== 0) x = tf.add(x, p.brightness);
    x = tf.clipByValue(x, 0, 1);

    if (p.cropFrac < 1) {
      const f = Math.min(1, Math.max(0.05, p.cropFrac));
      const lo = (1 - f) / 2;
      const hi = lo + f;
      const n = x.shape[0];
      x = tf.image.cropAndResize(
        x,
        tf.tensor2d(Array.from({ length: n }, () => [lo, lo, hi, hi]), [n, 4]),
        tf.tensor1d(Array.from({ length: n }, (_, i) => i), 'int32'),
        [h, w],
        'bilinear',
      );
    }

    if (p.rotationDeg !== 0) {
      const th = (p.rotationDeg * Math.PI) / 180;
      const cos = Math.cos(th);
      const sin = Math.sin(th);
      const cx = (w - 1) / 2;
      const cy = (h - 1) / 2;
      // Output pixel (x, y) samples the input at R(-theta) * ((x, y) - c) + c;
      // with image y pointing down this rotates the picture counter-clockwise.
      const a2 = cx - cos * cx + sin * cy;
      const b2 = cy - sin * cx - cos * cy;
      const t = tf.tensor2d([[cos, -sin, a2, sin, cos, b2, 0, 0]], [1, 8]);
      const rotate = (img4) => tf.image.transform(img4, t, 'bilinear', 'constant', 0);
      // The CPU backend's Transform kernel only computes batch element 0, so rotate
      // each image on its own (batch of 1) whenever there is more than one.
      x = x.shape[0] === 1 ? rotate(x) : tf.concat(tf.unstack(x).map((img) => rotate(tf.expandDims(img, 0))), 0);
    }

    x = tf.clipByValue(x, 0, 1);
    return rank === 3 ? tf.squeeze(x, [0]) : x;
  });
}
