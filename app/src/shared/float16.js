// IEEE 754 binary16 ("half precision") codec, pure JS, no dependencies.
//
// Used to store the precomputed 1280-d MobileNet features at half the size
// (scripts/precompute_embeddings.mjs writes them, the app reads them back).
//
// Format of a binary16 value (16 bits): 1 sign | 5 exponent (bias 15) | 10 mantissa.
// Encoding rounds to nearest, ties to even. Float32 values too large for binary16
// (|x| >= 65520) become +-Infinity, tiny values become binary16 subnormals or +-0,
// NaN stays NaN (always a quiet NaN), +-0 keeps its sign.
//
// Binary files are little-endian: element i occupies bytes 2i (low) and 2i+1 (high).

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/**
 * Convert one number to its binary16 bit pattern (round to nearest even).
 * @param {number} x
 * @returns {number} integer in [0, 65535]
 */
export function float32ToFloat16Bits(x) {
  f32[0] = x;
  return bitsToHalf(u32[0]);
}

/** float32 bit pattern -> binary16 bit pattern. */
function bitsToHalf(b) {
  const sign = (b >>> 16) & 0x8000;
  const exp = (b >>> 23) & 0xff;
  const mant = b & 0x7fffff;
  if (exp === 0xff) {
    // Infinity keeps mantissa 0; NaN gets the quiet bit so it can never collapse to Infinity.
    return mant === 0 ? sign | 0x7c00 : sign | 0x7e00 | (mant >>> 13);
  }
  const e = exp - 112; // exp - 127 + 15 = target exponent field
  if (e >= 31) return sign | 0x7c00; // overflow -> Infinity
  if (e <= 0) {
    // Subnormal in binary16 (or zero). Includes float32 zero / denormals (exp === 0).
    const shift = 14 - e; // bits to drop from the 24-bit significand
    if (shift > 24) return sign; // below half of the smallest subnormal -> +-0
    const m = exp === 0 ? 0 : mant | 0x800000; // float32 denormals are far below 2^-25
    let half = m >>> shift;
    const rem = m & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1) === 1)) half++;
    return sign | half; // a carry to 0x400 correctly becomes the smallest normal
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1) === 1)) half++; // carry may bump exponent / reach Infinity
  return sign | half;
}

/**
 * Convert a binary16 bit pattern to a number.
 * @param {number} h integer in [0, 65535]
 * @returns {number}
 */
export function float16BitsToFloat32(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 5.960464477539063e-8; // mant * 2^-24 (exact), covers +-0
  if (exp === 31) return mant === 0 ? sign * Infinity : NaN;
  return sign * (1 + mant / 1024) * Math.pow(2, exp - 15);
}

/** @type {Float32Array | null} */
let decodeTable = null;

/** 65536-entry lookup table (256 KiB), built on first use. */
function getDecodeTable() {
  if (decodeTable === null) {
    const t = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) t[i] = float16BitsToFloat32(i);
    decodeTable = t;
  }
  return decodeTable;
}

/**
 * Encode float32 values as binary16 bit patterns.
 * The returned Uint16Array holds the numeric bit patterns; on little-endian
 * machines (all browsers / Node targets we support) its `.buffer` is already the
 * little-endian file layout. Use encodeFloat16ToBytes for a byte array that is
 * little-endian on every platform.
 * @param {Float32Array | ArrayLike<number>} values
 * @returns {Uint16Array}
 */
export function encodeFloat16(values) {
  const n = values.length;
  const out = new Uint16Array(n);
  const src = values instanceof Float32Array ? values : Float32Array.from(values);
  const bits = new Uint32Array(src.buffer, src.byteOffset, n);
  for (let i = 0; i < n; i++) out[i] = bitsToHalf(bits[i]);
  return out;
}

/**
 * Encode float32 values as little-endian binary16 bytes (2 bytes per value),
 * independent of the host byte order. This is the on-disk format.
 * @param {Float32Array | ArrayLike<number>} values
 * @returns {Uint8Array}
 */
export function encodeFloat16ToBytes(values) {
  const bits = encodeFloat16(values);
  const out = new Uint8Array(bits.length * 2);
  for (let i = 0, j = 0; i < bits.length; i++, j += 2) {
    const h = bits[i];
    out[j] = h & 0xff;
    out[j + 1] = h >>> 8;
  }
  return out;
}

/**
 * Decode binary16 data to float32.
 * Accepts a Uint16Array (numeric bit patterns), or raw little-endian bytes as an
 * ArrayBuffer or any ArrayBuffer view (Uint8Array, DataView, Buffer, ...).
 * Byte input must have an even length.
 * @param {Uint16Array | ArrayBuffer | ArrayBufferView} data
 * @returns {Float32Array}
 */
export function decodeFloat16(data) {
  const table = getDecodeTable();
  if (data instanceof Uint16Array) {
    const n = data.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = table[data[i]];
    return out;
  }
  const bytes =
    data instanceof ArrayBuffer
      ? new Uint8Array(data)
      : ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : null;
  if (bytes === null) throw new TypeError('decodeFloat16: expected a Uint16Array, ArrayBuffer or ArrayBuffer view');
  if (bytes.length % 2 !== 0) throw new RangeError(`decodeFloat16: byte length ${bytes.length} is not a multiple of 2`);
  const n = bytes.length >>> 1;
  const out = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 2) out[i] = table[bytes[j] | (bytes[j + 1] << 8)];
  return out;
}
