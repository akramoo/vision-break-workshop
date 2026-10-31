// Records which WebGL renderer each browser project really uses, so timings can be read with the right context.
// Playwright's bundled headless chromium and Firefox use software rendering; the chromium-gpu project must report the real GPU.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { test, expect } from './fixtures';
import { waitForReady } from './helpers';
import { dryRunFile } from './measure.helpers';

test('WebGL renderer behind this browser project', async ({ page, browserName }, info) => {
  await page.goto('/?debug=1');
  await waitForReady(page);
  const gl = await page.evaluate(() => {
    const c = document.createElement('canvas');
    const g = (c.getContext('webgl2') || c.getContext('webgl')) as WebGLRenderingContext | null;
    if (!g) return { vendor: 'none', renderer: 'none', version: '' };
    const ext = g.getExtension('WEBGL_debug_renderer_info');
    return {
      vendor: ext ? String(g.getParameter(ext.UNMASKED_VENDOR_WEBGL)) : 'masked',
      renderer: ext ? String(g.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : 'masked',
      version: String(g.getParameter(g.VERSION)),
    };
  });
  // The context tfjs itself computes on (it may differ from a fresh canvas).
  const tfGl = await page.evaluate(() => {
    const b = window.__vbw?.tf.backend() as unknown as { gpgpu?: { gl?: WebGLRenderingContext } } | undefined;
    const g = b?.gpgpu?.gl;
    if (!g) return 'n/a (not the webgl backend)';
    const ext = g.getExtension('WEBGL_debug_renderer_info');
    return ext ? String(g.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : 'masked';
  });
  const badge = (await page.getByTestId('backend-badge').textContent())?.trim();
  const version = page.context().browser()?.version() ?? '';
  const row = { project: info.project.name, browserName, browserVersion: version, ...gl, tfRenderer: tfGl, badge };
  console.log(`[gpu-proof] ${JSON.stringify(row)}`);
  const file = dryRunFile(`gpu-proof-${info.project.name}.json`);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(row, null, 2));
  if (info.project.name === 'chromium-gpu') {
    expect(gl.renderer).toMatch(/NVIDIA|RTX 3070/i);
    expect(gl.renderer).not.toMatch(/SwiftShader|llvmpipe|Software/i);
    expect(tfGl).toMatch(/NVIDIA|RTX 3070/i);
  }
  expect(badge).toBeTruthy();
});
