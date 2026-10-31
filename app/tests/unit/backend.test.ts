import { describe, expect, it } from 'vitest';
import { backendFromQuery } from '../../src/ml/backend';

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
