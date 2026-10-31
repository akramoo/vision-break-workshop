/** Convert one number to its IEEE 754 binary16 bit pattern (round to nearest even). */
export function float32ToFloat16Bits(x: number): number;
/** Convert a binary16 bit pattern (0..65535) to a number. */
export function float16BitsToFloat32(h: number): number;
/** Encode float32 values as binary16 bit patterns (round to nearest even). */
export function encodeFloat16(values: Float32Array | ArrayLike<number>): Uint16Array;
/** Encode float32 values as little-endian binary16 bytes (2 per value), independent of host byte order. */
export function encodeFloat16ToBytes(values: Float32Array | ArrayLike<number>): Uint8Array;
/** Decode binary16 data (Uint16Array bit patterns, or little-endian bytes in an ArrayBuffer / view) to float32. */
export function decodeFloat16(data: Uint16Array | ArrayBuffer | ArrayBufferView): Float32Array;
