// Independent tests for src/shared/float16.js.
// The reference decoder / encoder below are written from the IEEE 754 binary16 definition
// (1 sign, 5 exponent bits with bias 15, 10 fraction bits) and share no code with the implementation.
import { describe, expect, it } from 'vitest';
import {
  decodeFloat16,
  encodeFloat16,
  encodeFloat16ToBytes,
  float16BitsToFloat32,
  float32ToFloat16Bits,
} from '../../src/shared/float16.js';

// ---------------------------------------------------------------- reference model

/** Value of a binary16 pattern straight from the spec. NaN for exponent 31 / fraction != 0. */
function refDecode(h: number): number {
  const s = (h >> 15) & 1;
  const e = (h >> 10) & 31;
  const f = h & 1023;
  const sign = s ? -1 : 1;
  if (e === 0) return sign * (f / 1024) * 2 ** -14; // subnormal / zero
  if (e === 31) return f === 0 ? sign * Infinity : NaN;
  return sign * (1 + f / 1024) * 2 ** (e - 15);
}

/** Positive finite halves 0x0000..0x7bff; bit pattern order == value order. */
const POS: number[] = [];
for (let h = 0; h <= 0x7bff; h++) POS.push(refDecode(h));

/**
 * Reference round-to-nearest-even of a (float32-representable) number, by search over the
 * list of all representable halves. Returns the bit pattern (NaN -> 0x7e00).
 */
function refEncode(x: number): number {
  if (Number.isNaN(x)) return 0x7e00;
  const sign = x < 0 || Object.is(x, -0) ? 0x8000 : 0;
  const a = Math.abs(x);
  if (a === Infinity) return sign | 0x7c00;
  // largest index with POS[i] <= a
  let lo = 0;
  let hi = POS.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (POS[mid]! <= a) lo = mid;
    else hi = mid - 1;
  }
  const i = lo;
  if (i === POS.length - 1) {
    // a >= 65504. Next "value" would be 65536 (the infinity position); halfway is 65520.
    // Tie at 65520 rounds to even: 0x7bff is odd, so it goes up to infinity.
    return sign | (a >= 65520 ? 0x7c00 : 0x7bff);
  }
  const dLow = a - POS[i]!;
  const dHigh = POS[i + 1]! - a;
  let pick: number;
  if (dLow < dHigh) pick = i;
  else if (dHigh < dLow) pick = i + 1;
  else pick = i % 2 === 0 ? i : i + 1;
  return sign | pick;
}

// deterministic PRNG for test data (xorshift32)
function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

// ---------------------------------------------------------------- decode

describe('float16 decode: every bit pattern', () => {
  it('float16BitsToFloat32 matches the reference decoder for all 65,536 patterns', () => {
    const bad: string[] = [];
    for (let h = 0; h < 65536; h++) {
      const got = float16BitsToFloat32(h);
      const want = refDecode(h);
      const same = Number.isNaN(want) ? Number.isNaN(got) : Object.is(got, want);
      if (!same) bad.push(`0x${h.toString(16)}: got ${got}, want ${want}`);
    }
    expect(bad.slice(0, 5)).toEqual([]);
    expect(bad.length).toBe(0);
  });

  it('decodeFloat16(Uint16Array of all patterns) matches the reference (via Float32 rounding, exact for halves)', () => {
    const all = new Uint16Array(65536);
    for (let i = 0; i < 65536; i++) all[i] = i;
    const out = decodeFloat16(all);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(65536);
    let bad = 0;
    for (let h = 0; h < 65536; h++) {
      const want = refDecode(h);
      if (Number.isNaN(want) ? !Number.isNaN(out[h]) : !Object.is(out[h], want)) bad++;
    }
    expect(bad).toBe(0);
  });

  it('known landmark values', () => {
    expect(float16BitsToFloat32(0x3c00)).toBe(1);
    expect(float16BitsToFloat32(0xc000)).toBe(-2);
    expect(float16BitsToFloat32(0x7bff)).toBe(65504);
    expect(float16BitsToFloat32(0x0400)).toBe(2 ** -14);
    expect(float16BitsToFloat32(0x0001)).toBe(2 ** -24); // 5.96e-8
    expect(float16BitsToFloat32(0x0001)).toBeCloseTo(5.96e-8, 10);
    expect(float16BitsToFloat32(0x3555)).toBeCloseTo(1 / 3, 3);
    expect(Object.is(float16BitsToFloat32(0x8000), -0)).toBe(true);
    expect(Object.is(float16BitsToFloat32(0x0000), 0)).toBe(true);
    expect(float16BitsToFloat32(0x7c00)).toBe(Infinity);
    expect(float16BitsToFloat32(0xfc00)).toBe(-Infinity);
    expect(Number.isNaN(float16BitsToFloat32(0x7e00))).toBe(true);
    expect(Number.isNaN(float16BitsToFloat32(0x7c01))).toBe(true);
    expect(Number.isNaN(float16BitsToFloat32(0xffff))).toBe(true);
  });
});

// ---------------------------------------------------------------- encode

describe('float16 encode', () => {
  it('encode(decode(h)) == h for every non-NaN pattern', () => {
    const bad: number[] = [];
    for (let h = 0; h < 65536; h++) {
      const v = float16BitsToFloat32(h);
      if (Number.isNaN(v)) continue;
      if (float32ToFloat16Bits(v) !== h) bad.push(h);
    }
    expect(bad.slice(0, 5)).toEqual([]);
    expect(bad.length).toBe(0);
  });

  it('NaN patterns stay NaN after a roundtrip (and never become Infinity)', () => {
    let nanCount = 0;
    for (let h = 0; h < 65536; h++) {
      if ((h & 0x7c00) === 0x7c00 && (h & 0x3ff) !== 0) {
        nanCount++;
        const back = float32ToFloat16Bits(float16BitsToFloat32(h));
        expect(back & 0x7c00).toBe(0x7c00);
        expect(back & 0x3ff).not.toBe(0);
      }
    }
    expect(nanCount).toBe(2046);
    for (const v of [NaN, -NaN, Number.NaN]) {
      const b = float32ToFloat16Bits(v);
      expect(b & 0x7c00).toBe(0x7c00);
      expect(b & 0x3ff).not.toBe(0);
    }
  });

  it('matches the reference encoder on 400,000 random float32 values of mixed magnitudes', () => {
    const r = rng(12345);
    const f32 = new Float32Array(1);
    const bad: string[] = [];
    for (let k = 0; k < 400_000; k++) {
      // log-uniform magnitude from 2^-30 to 2^17 plus random sign; covers subnormal, normal, overflow
      const mag = 2 ** (-30 + r() * 47);
      f32[0] = (r() < 0.5 ? -1 : 1) * mag;
      const x = f32[0];
      const got = float32ToFloat16Bits(x);
      const want = refEncode(x);
      if (got !== want) bad.push(`${x}: got 0x${got.toString(16)} want 0x${want.toString(16)}`);
    }
    expect(bad.slice(0, 5)).toEqual([]);
    expect(bad.length).toBe(0);
  });

  it('matches the reference encoder on exact midpoints and their float32 neighbours for every adjacent pair of halves', () => {
    const f32 = new Float32Array(1);
    const bad: string[] = [];
    for (let i = 0; i < POS.length - 1; i++) {
      const mid = (POS[i]! + POS[i + 1]!) / 2; // exact in float64; also exact in float32 for these magnitudes
      f32[0] = mid;
      if (f32[0] !== mid) continue; // (cannot happen for halves, but be safe)
      for (const x of [mid, -mid]) {
        const got = float32ToFloat16Bits(x);
        const want = refEncode(x);
        if (got !== want) bad.push(`tie ${x}: got 0x${got.toString(16)} want 0x${want.toString(16)}`);
      }
      // nudge by one float32 ulp each side
      const bits = new Uint32Array(f32.buffer);
      f32[0] = mid;
      bits[0]! += 1;
      const up = f32[0];
      f32[0] = mid;
      bits[0]! -= 1;
      const down = f32[0];
      for (const x of [up, down, -up, -down]) {
        const got = float32ToFloat16Bits(x);
        const want = refEncode(x);
        if (got !== want) bad.push(`near-tie ${x}: got 0x${got.toString(16)} want 0x${want.toString(16)}`);
      }
    }
    expect(bad.slice(0, 5)).toEqual([]);
    expect(bad.length).toBe(0);
  });

  it('rounds ties to even at hand-picked cases', () => {
    // 1 + 2^-11 is exactly between 1.0 (0x3c00, even) and 1+2^-10 (0x3c01, odd) -> 0x3c00
    expect(float32ToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    // 1 + 3*2^-11 is between 0x3c01 (odd) and 0x3c02 (even) -> 0x3c02
    expect(float32ToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
    // just above/below the tie
    expect(float32ToFloat16Bits(1 + 2 ** -11 + 2 ** -20)).toBe(0x3c01);
    expect(float32ToFloat16Bits(1 + 2 ** -11 - 2 ** -20)).toBe(0x3c00);
    // negative symmetric
    expect(float32ToFloat16Bits(-(1 + 2 ** -11))).toBe(0xbc00);
    // half of the smallest subnormal ties to even (0)
    expect(float32ToFloat16Bits(2 ** -25)).toBe(0x0000);
    // 1.5 * smallest subnormal ties to even (2 * 2^-24)
    expect(float32ToFloat16Bits(1.5 * 2 ** -24)).toBe(0x0002);
    // slightly more than half of smallest subnormal rounds up
    expect(float32ToFloat16Bits(2 ** -25 + 2 ** -40)).toBe(0x0001);
    // largest subnormal/smallest normal tie: (0x03ff + 0x0400)/2 -> even = 0x0400
    expect(float32ToFloat16Bits((1023 / 1024 + 1) / 2 * 2 ** -14)).toBe(0x0400);
  });

  it('overflow: 65504 is exact, 65519.99 stays finite, 65520 and above become infinity', () => {
    expect(float32ToFloat16Bits(65504)).toBe(0x7bff);
    expect(float32ToFloat16Bits(65519.99)).toBe(0x7bff);
    expect(float32ToFloat16Bits(Math.fround(65519.996))).toBe(0x7bff); // largest float32 < 65520
    expect(float32ToFloat16Bits(65520)).toBe(0x7c00); // tie rounds to even => infinity
    expect(float32ToFloat16Bits(65521)).toBe(0x7c00);
    expect(float32ToFloat16Bits(1e6)).toBe(0x7c00);
    expect(float32ToFloat16Bits(3.4e38)).toBe(0x7c00);
    expect(float32ToFloat16Bits(-65504)).toBe(0xfbff);
    expect(float32ToFloat16Bits(-65520)).toBe(0xfc00);
    expect(float32ToFloat16Bits(-1e9)).toBe(0xfc00);
    expect(float32ToFloat16Bits(Infinity)).toBe(0x7c00);
    expect(float32ToFloat16Bits(-Infinity)).toBe(0xfc00);
  });

  it('denormals, underflow and signed zero', () => {
    expect(float32ToFloat16Bits(5.960464477539063e-8)).toBe(0x0001);
    expect(float32ToFloat16Bits(5.96e-8)).toBe(0x0001);
    expect(float32ToFloat16Bits(-5.96e-8)).toBe(0x8001);
    expect(float32ToFloat16Bits(2 ** -14)).toBe(0x0400);
    expect(float32ToFloat16Bits(2 ** -14 - 2 ** -24)).toBe(0x03ff);
    expect(float32ToFloat16Bits(1e-9)).toBe(0x0000);
    expect(float32ToFloat16Bits(-1e-9)).toBe(0x8000);
    expect(float32ToFloat16Bits(1e-30)).toBe(0x0000);
    expect(float32ToFloat16Bits(1e-45)).toBe(0x0000); // float32 denormal
    expect(float32ToFloat16Bits(0)).toBe(0x0000);
    expect(float32ToFloat16Bits(-0)).toBe(0x8000);
    expect(Object.is(decodeFloat16(encodeFloat16([-0]))[0], -0)).toBe(true);
  });

  it('encodeFloat16 accepts Float32Array and plain arrays and returns Uint16Array of the same length', () => {
    const arr = [0, 1, -2, 0.5, 65504, 1e6, NaN];
    const a = encodeFloat16(arr);
    const b = encodeFloat16(Float32Array.from(arr));
    expect(a).toBeInstanceOf(Uint16Array);
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(Array.from(a.slice(0, 6))).toEqual([0x0000, 0x3c00, 0xc000, 0x3800, 0x7bff, 0x7c00]);
    expect(encodeFloat16([]).length).toBe(0);
  });

  it('encodeFloat16 handles a Float32Array view with non-zero byteOffset (subarray)', () => {
    const base = new Float32Array([9, 9, 1, 2, 3, 9]);
    const sub = base.subarray(2, 5);
    expect(Array.from(encodeFloat16(sub))).toEqual([0x3c00, 0x4000, 0x4200]);
    expect(Array.from(encodeFloat16ToBytes(sub))).toEqual([0x00, 0x3c, 0x00, 0x40, 0x00, 0x42]);
  });
});

// ---------------------------------------------------------------- bytes / endianness / views

describe('float16 byte-level API', () => {
  it('encodeFloat16ToBytes is little-endian: 1.0 -> [0x00, 0x3c]', () => {
    const b = encodeFloat16ToBytes([1, -2, 65504]);
    expect(b).toBeInstanceOf(Uint8Array);
    expect(Array.from(b)).toEqual([0x00, 0x3c, 0x00, 0xc0, 0xff, 0x7b]);
  });

  it('encodeFloat16ToBytes output equals a DataView little-endian write of the reference encoder (host independent)', () => {
    const r = rng(777);
    const vals = new Float32Array(5000);
    for (let i = 0; i < vals.length; i++) vals[i] = (r() - 0.3) * 8;
    const dv = new DataView(new ArrayBuffer(vals.length * 2));
    for (let i = 0; i < vals.length; i++) dv.setUint16(i * 2, refEncode(vals[i] as number), true);
    expect(Array.from(encodeFloat16ToBytes(vals))).toEqual(Array.from(new Uint8Array(dv.buffer)));
  });

  it('roundtrip bytes -> decodeFloat16(Uint8Array) is exact for halves', () => {
    const all: number[] = [];
    for (let h = 0; h < 65536; h++) {
      const v = refDecode(h);
      if (!Number.isNaN(v)) all.push(v);
    }
    const bytes = encodeFloat16ToBytes(all);
    const back = decodeFloat16(bytes);
    expect(back.length).toBe(all.length);
    let bad = 0;
    for (let i = 0; i < all.length; i++) if (!Object.is(back[i], all[i])) bad++;
    expect(bad).toBe(0);
  });

  it('decodeFloat16 accepts ArrayBuffer, Uint8Array, Buffer and DataView', () => {
    const vals = [0.25, -1.5, 3, 100.5];
    const bytes = encodeFloat16ToBytes(vals);
    const want = vals.map((v) => Math.fround(v));
    expect(Array.from(decodeFloat16(bytes.buffer as ArrayBuffer))).toEqual(want);
    expect(Array.from(decodeFloat16(bytes))).toEqual(want);
    expect(Array.from(decodeFloat16(Buffer.from(bytes)))).toEqual(want);
    expect(Array.from(decodeFloat16(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)))).toEqual(want);
    expect(Array.from(decodeFloat16(Uint16Array.from(encodeFloat16(vals))))).toEqual(want);
  });

  it('decodeFloat16 handles Uint8Array views at odd (misaligned) byteOffsets into a larger buffer', () => {
    const vals = [1, 2, 3, 4, -0.5, 0.125, 65504];
    const payload = encodeFloat16ToBytes(vals);
    for (const off of [0, 1, 2, 3, 5, 7]) {
      const big = new ArrayBuffer(off + payload.length + 9);
      new Uint8Array(big).fill(0xee);
      new Uint8Array(big, off, payload.length).set(payload);
      const view = new Uint8Array(big, off, payload.length);
      expect(view.byteOffset).toBe(off);
      expect(Array.from(decodeFloat16(view))).toEqual(vals.map((v) => Math.fround(v)));
    }
    // Node Buffer slices from the shared pool are also commonly misaligned
    const pooled = Buffer.concat([Buffer.from([0xaa]), Buffer.from(payload)]).subarray(1);
    expect(Array.from(decodeFloat16(pooled))).toEqual(vals.map((v) => Math.fround(v)));
  });

  it('decodeFloat16 on a Uint16Array subarray (non-zero byteOffset) decodes only that window', () => {
    const base = Uint16Array.from([0xffff, 0x3c00, 0x4000, 0xffff]);
    expect(Array.from(decodeFloat16(base.subarray(1, 3)))).toEqual([1, 2]);
  });

  it('decodeFloat16 on a Uint8Array view bounded within a larger ArrayBuffer ignores bytes outside the view', () => {
    const big = new Uint8Array([0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44]);
    expect(Array.from(decodeFloat16(big.subarray(2, 6)))).toEqual([2, 3]);
  });

  it('decodeFloat16 throws on odd byte length', () => {
    expect(() => decodeFloat16(new Uint8Array(3))).toThrow();
    expect(() => decodeFloat16(new ArrayBuffer(5))).toThrow();
    expect(() => decodeFloat16(new Uint8Array(new ArrayBuffer(9), 1, 3))).toThrow();
    expect(() => decodeFloat16(Buffer.from([1, 2, 3]))).toThrow();
  });

  it('decodeFloat16 on empty inputs returns an empty Float32Array', () => {
    expect(decodeFloat16(new Uint8Array(0)).length).toBe(0);
    expect(decodeFloat16(new ArrayBuffer(0)).length).toBe(0);
    expect(decodeFloat16(new Uint16Array(0)).length).toBe(0);
  });

  it('decodeFloat16 rejects non-buffer input', () => {
    expect(() => decodeFloat16([1, 2] as unknown as Uint16Array)).toThrow();
    expect(() => decodeFloat16(null as unknown as Uint16Array)).toThrow();
  });

  it('decodeFloat16 returns a fresh buffer (does not alias the input)', () => {
    const u16 = Uint16Array.from([0x3c00]);
    const out = decodeFloat16(u16);
    expect(out.buffer).not.toBe(u16.buffer);
  });
});

// ---------------------------------------------------------------- performance

describe('float16 performance', () => {
  it('encodes and decodes 3M values in well under 1 s each (after warm-up)', () => {
    const n = 3_000_000;
    const r = rng(99);
    const src = new Float32Array(n);
    for (let i = 0; i < n; i++) src[i] = r() * 6;
    // warm up JIT / lookup table
    decodeFloat16(encodeFloat16ToBytes(src.subarray(0, 10000)));

    const t0 = performance.now();
    const bytes = encodeFloat16ToBytes(src);
    const t1 = performance.now();
    const back = decodeFloat16(bytes);
    const t2 = performance.now();
    // eslint-disable-next-line no-console
    console.log(`float16 perf: encode ${(t1 - t0).toFixed(0)} ms, decode ${(t2 - t1).toFixed(0)} ms for ${n} values`);
    expect(t1 - t0).toBeLessThan(1000);
    expect(t2 - t1).toBeLessThan(1000);
    // accuracy: relative error of half rounding <= 2^-11 for normals in [6e-5, 6]
    let worst = 0;
    for (let i = 0; i < n; i += 997) {
      const x = src[i] as number;
      if (x > 6.2e-5) worst = Math.max(worst, Math.abs((back[i] as number) - x) / x);
    }
    expect(worst).toBeLessThanOrEqual(2 ** -11 + 1e-9);
  });
});
