import { describe, expect, it } from 'vitest';
import { backendFromQuery, defaultMsPerImage, isSoftwareRenderer } from '../../src/ml/backend';

// backendFromQuery takes the query string as a parameter, so no DOM / window mock is needed.
// (initBackend() itself needs a real tf backend and is covered by the Playwright specs.)
describe('backendFromQuery', () => {
  it('accepts webgl, wasm and cpu', () => {
    expect(backendFromQuery('?backend=webgl')).toBe('webgl');
    expect(backendFromQuery('?backend=wasm')).toBe('wasm');
    expect(backendFromQuery('?backend=cpu')).toBe('cpu');
    expect(backendFromQuery('backend=cpu')).toBe('cpu');
  });

  it('works alongside other params, in any position', () => {
    expect(backendFromQuery('?debug=1&backend=wasm')).toBe('wasm');
    expect(backendFromQuery('?backend=cpu&debug=1')).toBe('cpu');
  });

  it('ignores unknown, empty, mis-cased or injected values', () => {
    for (const q of [
      '',
      '?',
      '?backend=',
      '?backend=gpu',
      '?backend=WASM',
      '?backend=wasm%20',
      '?backend=wasm,cpu',
      '?backend=<script>',
      '?Backend=wasm',
      '?debug=1',
      '?backend[]=wasm',
    ]) {
      expect(backendFromQuery(q), `query ${JSON.stringify(q)}`).toBeNull();
    }
  });

  it('with several backend params the first wins (URLSearchParams.get semantics)', () => {
    expect(backendFromQuery('?backend=cpu&backend=wasm')).toBe('cpu');
  });

  it('default argument does not throw when there is no window/location (node)', () => {
    expect(() => backendFromQuery()).not.toThrow();
    expect(backendFromQuery()).toBeNull();
  });
});

describe('isSoftwareRenderer', () => {
  it('recognises CPU rasterisers', () => {
    for (const r of [
      'ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)',
      'Mesa llvmpipe (LLVM 15.0.7, 256 bits)',
      'Microsoft Basic Render Driver',
      'Software Rasterizer',
    ]) {
      expect(isSoftwareRenderer(r), r).toBe(true);
    }
  });

  it('does not flag real GPUs or an unknown renderer', () => {
    expect(isSoftwareRenderer('ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0, D3D11)')).toBe(false);
    expect(isSoftwareRenderer('Apple M2')).toBe(false);
    expect(isSoftwareRenderer(undefined)).toBe(false);
    expect(isSoftwareRenderer('')).toBe(false);
  });
});

describe('defaultMsPerImage', () => {
  it('uses ~25 ms for hardware WebGL and 120 ms for software WebGL', () => {
    expect(defaultMsPerImage({ backend: 'webgl', softwareRenderer: false })).toBe(25);
    expect(defaultMsPerImage({ backend: 'webgl', softwareRenderer: true })).toBe(120);
  });

  it('keeps the WASM and CPU figures, and is safe before the backend is known', () => {
    expect(defaultMsPerImage({ backend: 'wasm', softwareRenderer: false })).toBe(350);
    expect(defaultMsPerImage({ backend: 'cpu', softwareRenderer: false })).toBe(900);
    expect(defaultMsPerImage({ backend: 'none', softwareRenderer: false })).toBe(900);
    expect(defaultMsPerImage(null)).toBe(900);
    expect(defaultMsPerImage(undefined)).toBe(900);
  });
});
