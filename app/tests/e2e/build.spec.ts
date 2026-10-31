import { spawn, type ChildProcess } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { request } from 'node:http';
import { join } from 'node:path';
import { test, expect } from './fixtures';
import { DIST_DIR, collectPageErrors, freePort, servePrefixed, waitForReady } from './helpers';

/** Raw HTTP request: Node's client does NOT normalise "../" or "%2e%2e", unlike fetch() / page.goto(). */
function rawGet(
  port: number,
  path: string,
  method = 'GET',
): Promise<{ status: number; type: string; body: string }> {
  return new Promise((resolve, reject) => {
    const req = request({ host: '127.0.0.1', port, path, method }, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c: Buffer) => chunks.push(c));
      res.on('end', () =>
        resolve({
          status: res.statusCode ?? 0,
          type: String(res.headers['content-type'] ?? ''),
          body: Buffer.concat(chunks).toString('utf8'),
        }),
      );
    });
    req.on('error', reject);
    req.end();
  });
}

async function startServeOffline(): Promise<{ port: number; proc: ChildProcess }> {
  const port = await freePort();
  const proc = spawn(process.execPath, [join(DIST_DIR, 'serve-offline.mjs'), '--port', String(port)], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let out = '';
  proc.stdout?.on('data', (d: Buffer) => (out += d.toString()));
  proc.stderr?.on('data', (d: Buffer) => (out += d.toString()));
  const deadline = Date.now() + 15_000;
  while (!out.includes('Open') && Date.now() < deadline) {
    if (proc.exitCode !== null) throw new Error(`serve-offline exited early (${proc.exitCode}): ${out}`);
    await new Promise((r) => setTimeout(r, 100));
  }
  if (!out.includes('Open')) throw new Error(`serve-offline did not start: ${out}`);
  return { port, proc };
}

test.describe('production build: relative paths', () => {
  test('dist/index.html has no absolute /assets (or any root-absolute) src/href', async ({ request: api }) => {
    const res = await api.get('/index.html');
    expect(res.ok()).toBe(true);
    const served = await res.text();
    const onDisk = readFileSync(join(DIST_DIR, 'index.html'), 'utf8');
    for (const html of [served, onDisk]) {
      expect(html).not.toMatch(/(?:src|href)\s*=\s*["']\/(?!\/)/); // "/assets/..." style
      expect(html).not.toMatch(/(?:src|href)\s*=\s*["']https?:/); // no CDN
      expect(html).toMatch(/src="\.\/assets\/[^"]+\.js"/);
      expect(html).toMatch(/href="\.\/assets\/[^"]+\.css"/);
    }
  });

  test('built JS / CSS contain no root-absolute asset, model or data URLs', () => {
    const assets = join(DIST_DIR, 'assets');
    const files = readdirSync(assets).filter((f) => /\.(js|css)$/.test(f));
    expect(files.length).toBeGreaterThan(0);
    for (const f of files) {
      const text = readFileSync(join(assets, f), 'utf8');
      // quote or url( followed by /assets, /models, /data  -> would break under a sub-path
      expect(text, f).not.toMatch(/["'`(]\/(assets|models|data)\//);
      expect(text, f).not.toMatch(/https?:\/\/(?!www\.w3\.org|127\.0\.0\.1|localhost)[a-z0-9.-]+\.(com|net|org|io|dev)\b[^"'`\s]*\.(js|css|wasm|json|woff2?)/i);
    }
  });

  test('wasm binaries are vendored into dist/assets (no CDN needed)', () => {
    const wasm = readdirSync(join(DIST_DIR, 'assets')).filter((f) => f.endsWith('.wasm'));
    expect(wasm.length).toBeGreaterThanOrEqual(1);
  });

  test('model + serve-offline.mjs are copied into dist', () => {
    const files = readdirSync(DIST_DIR);
    expect(files).toContain('serve-offline.mjs');
    expect(files).toContain('models');
    expect(readdirSync(join(DIST_DIR, 'models', 'mobilenet_v2'))).toContain('model.json');
  });
});

test.describe('production build under a sub-path (GitHub Pages style)', () => {
  test.setTimeout(150_000);

  for (const backend of ['', '?backend=wasm']) {
    test(`loads and reaches Ready under /vision-break-workshop/${backend}`, async ({ page }) => {
      const prefix = '/vision-break-workshop/';
      const srv = await servePrefixed(prefix);
      try {
        const errors = collectPageErrors(page);
        await page.goto(`http://127.0.0.1:${srv.port}${prefix}${backend}`);
        await waitForReady(page);
        await expect(page.getByTestId('error-banner')).toHaveCount(0);
        await expect(page.locator('footer')).toContainText('Educational only. Not for clinical use.');
        expect(srv.failures, 'asset requests that 404 under the sub-path').toEqual([]);
        // everything the page asked for must live under the prefix
        expect(srv.requests.filter((p) => !p.startsWith(prefix))).toEqual([]);
        expect(srv.requests.some((p) => p.endsWith('model.json'))).toBe(true);
        expect(srv.requests.some((p) => p.endsWith('.bin'))).toBe(true);
        expect(errors).toEqual([]);
        if (backend) {
          expect(srv.requests.some((p) => p.endsWith('.wasm'))).toBe(true);
          await expect(page.getByTestId('backend-badge')).toHaveText('WASM');
        }
      } finally {
        await srv.close();
      }
    });
  }
});

test.describe('serve-offline.mjs', () => {
  test.setTimeout(150_000);
  let proc: ChildProcess | undefined;
  let port = 0;

  test.beforeEach(async () => {
    ({ port, proc } = await startServeOffline());
  });
  test.afterEach(() => {
    proc?.kill();
  });

  test('serves the app, and the page reaches Ready from it', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.goto(`http://127.0.0.1:${port}/`);
    await waitForReady(page);
    await expect(page.getByTestId('error-banner')).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  test('wasm backend works from serve-offline (application/wasm MIME)', async ({ page }) => {
    await page.goto(`http://127.0.0.1:${port}/?backend=wasm`);
    await waitForReady(page);
    await expect(page.getByTestId('backend-badge')).toHaveText('WASM');
  });

  test('.wasm is served as application/wasm, json/js/html with sensible types', async () => {
    const wasm = readdirSync(join(DIST_DIR, 'assets')).find((f) => f.endsWith('.wasm'));
    expect(wasm).toBeTruthy();
    const w = await rawGet(port, `/assets/${wasm}`);
    expect(w.status).toBe(200);
    expect(w.type).toBe('application/wasm');

    const j = await rawGet(port, '/models/mobilenet_v2/model.json');
    expect(j.status).toBe(200);
    expect(j.type).toMatch(/application\/json/);
    expect(() => JSON.parse(j.body)).not.toThrow();

    const root = await rawGet(port, '/');
    expect(root.status).toBe(200);
    expect(root.type).toMatch(/text\/html/);
    expect(root.body).toContain('<div id="root">');

    const js = readdirSync(join(DIST_DIR, 'assets')).find((f) => f.endsWith('.js'));
    expect((await rawGet(port, `/assets/${js}`)).type).toMatch(/javascript/);
  });

  const traversal = [
    '/../package.json',
    '/../../package.json',
    '/assets/../../package.json',
    '/%2e%2e/package.json',
    '/%2E%2E/package.json',
    '/%2e%2e%2fpackage.json',
    '/%2e%2e%2f%2e%2e%2fpackage.json',
    '/assets/%2e%2e/%2e%2e/package.json',
    '/..%2fpackage.json',
    '/..%5cpackage.json',
    '/..\\package.json',
    '/%2e%2e%5cpackage.json',
    '/.%2e/package.json',
    '/%2e./package.json',
    '/models/../../package.json',
    '/%2e%2e/%2e%2e/%2e%2e/Windows/win.ini',
    '/../serve-offline.mjs/../../package.json',
  ];
  for (const path of traversal) {
    test(`rejects path traversal ${path}`, async () => {
      const r = await rawGet(port, path);
      expect([403, 404], `status for ${path}`).toContain(r.status);
      // the canary: app/package.json must never be leaked
      expect(r.body).not.toContain('vision-break-workshop-app');
      expect(r.body).not.toContain('"devDependencies"');
    });
  }

  test('rejects NUL bytes and malformed percent-encoding', async () => {
    for (const path of ['/index.html%00.png', '/%00', '/%E0%A4%A', '/%']) {
      const r = await rawGet(port, path);
      expect([400, 403, 404], path).toContain(r.status);
    }
  });

  test('double-encoded dots are not decoded twice (still no leak)', async () => {
    const r = await rawGet(port, '/%252e%252e/package.json');
    expect([403, 404]).toContain(r.status);
    expect(r.body).not.toContain('vision-break-workshop-app');
  });

  test('unknown files 404, POST is 405, HEAD works', async () => {
    expect((await rawGet(port, '/nope.txt')).status).toBe(404);
    expect((await rawGet(port, '/', 'POST')).status).toBe(405);
    const head = await rawGet(port, '/', 'HEAD');
    expect(head.status).toBe(200);
    expect(head.body).toBe('');
  });
});
