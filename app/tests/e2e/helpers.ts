import { createServer, type Server } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { expect, type Page } from '@playwright/test';

export const DISCLAIMER = 'Educational only. Not for clinical use.';
export const TAB_NAMES = ['Data', 'Train', 'Evaluate', 'Break it', 'Failure log', 'Results'] as const;
export const DIST_DIR = resolve(process.cwd(), 'dist');
export const SCREENSHOT_DIR = resolve(process.cwd(), '..', 'docs', 'screenshots');

/** Wait until the header shows the model as ready (MobileNet loaded and warmed up). */
export async function waitForReady(page: Page, timeout = 90_000): Promise<void> {
  await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout });
}

/** Collect uncaught page errors (never expected in a healthy app). */
export function collectPageErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`${e.name}: ${e.message}`));
  return errors;
}

/** Ask the OS for a free TCP port on 127.0.0.1. */
export function freePort(): Promise<number> {
  return new Promise((res, rej) => {
    const s = createNetServer();
    s.once('error', rej);
    s.listen(0, '127.0.0.1', () => {
      const addr = s.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      s.close(() => res(port));
    });
  });
}

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
};

export interface PrefixServer {
  port: number;
  /** Requests answered with a status >= 400, as "<status> <path>". */
  failures: string[];
  /** Every path requested. */
  requests: string[];
  close(): Promise<void>;
}

/** Serve `dist/` ONLY under `prefix` (e.g. "/vision-break-workshop/"), like GitHub Pages does. Anything else is a 404. */
export async function servePrefixed(prefix: string): Promise<PrefixServer> {
  const failures: string[] = [];
  const requests: string[] = [];
  const server: Server = createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://x');
    requests.push(url.pathname);
    const fail = (code: number) => {
      failures.push(`${code} ${url.pathname}`);
      res.writeHead(code).end();
    };
    if (!url.pathname.startsWith(prefix)) return fail(404);
    let rel = decodeURIComponent(url.pathname.slice(prefix.length));
    if (rel === '' || rel.endsWith('/')) rel += 'index.html';
    const file = normalize(join(DIST_DIR, rel));
    if (!file.startsWith(DIST_DIR + sep)) return fail(403);
    stat(file)
      .then((info) => {
        if (!info.isFile()) return fail(404);
        res.writeHead(200, {
          'Content-Type': MIME[extname(file)] ?? 'application/octet-stream',
          'Content-Length': info.size,
        });
        createReadStream(file).pipe(res);
      })
      .catch(() => fail(404));
  });
  const port = await freePort();
  await new Promise<void>((ok) => server.listen(port, '127.0.0.1', ok));
  return {
    port,
    failures,
    requests,
    close: () => new Promise((ok) => server.close(() => ok())),
  };
}

/**
 * Timing budget in ms: the given value on Chromium (the tight desktop budget), three times as much elsewhere.
 * Firefox (software WebGL/CPU backend, different scheduler) is verified for behaviour; its timings are logged.
 */
export function budget(browserName: string, ms: number): number {
  return browserName === 'chromium' ? ms : ms * 3;
}
