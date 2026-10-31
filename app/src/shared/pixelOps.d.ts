import type { Tensor3D, Tensor4D } from '@tensorflow/tfjs-core';

/** The TensorFlow.js namespace (browser `@tensorflow/tfjs` or `@tensorflow/tfjs-node`). Typed loosely on purpose. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TF = any;

export type StampPosition = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';

export interface StampConfig {
  /** Side of the white box in px, 8..96. */
  size: number;
  position: StampPosition;
  /** Inset in px from the chosen corner (horizontal). */
  offsetX: number;
  /** Inset in px from the chosen corner (vertical). */
  offsetY: number;
  /** 0..1 */
  opacity: number;
}

export interface PerturbationParams {
  /** Additive, -0.5..0.5 */
  brightness: number;
  /** Multiplicative around the image mean, 0.5..1.5 */
  contrast: number;
  /** Counter-clockwise degrees, -30..30; exposed corners are black. */
  rotationDeg: number;
  /** Center crop fraction 0.6..1.0, resized back to the original size. */
  cropFrac: number;
}

export const IMG_SIZE: 224;
export const FEATURE_OUTPUT_NODE: string;
export const FEATURE_DIM: 1280;

export function mulberry32(seed: number): () => number;

export function toModelInput(tf: TF, img: Tensor3D, opts: { scale255: boolean }): Tensor3D;

export const R_GLYPH: readonly string[];
export const STAMP_POSITIONS: readonly StampPosition[];
export const DEFAULT_STAMP: Readonly<StampConfig>;
export function resolveStampConfig(cfg?: Partial<StampConfig>): StampConfig;
export function stampBox(
  cfg?: Partial<StampConfig>,
  height?: number,
  width?: number,
): { x0: number; y0: number; x1: number; y1: number };
export function makeStampMask(
  tf: TF,
  cfg?: Partial<StampConfig>,
  height?: number,
  width?: number,
): { alpha: Tensor3D; color: Tensor3D };
export function applyStamp(tf: TF, img01: Tensor3D, cfg?: Partial<StampConfig>): Tensor3D;
export function applyStamp(tf: TF, img01: Tensor4D, cfg?: Partial<StampConfig>): Tensor4D;

export const IDENTITY_PERTURBATION: Readonly<PerturbationParams>;
export const PERTURBATION_RANGES: Readonly<{
  brightness: readonly [number, number];
  contrast: readonly [number, number];
  rotationDeg: readonly [number, number];
  cropFrac: readonly [number, number];
}>;
export const STRESS_PRESET: Readonly<PerturbationParams>;
export const AUG_PARAMS: Readonly<{
  brightness: number;
  contrast: readonly [number, number];
  rotationDeg: number;
  cropFrac: readonly [number, number];
}>;
export function randomAugmentParams(rng: () => number): PerturbationParams;
export function applyPerturbation(tf: TF, img01: Tensor3D, params?: Partial<PerturbationParams>): Tensor3D;
export function applyPerturbation(tf: TF, img01: Tensor4D, params?: Partial<PerturbationParams>): Tensor4D;
