// Helpers shared by the dry-run specs (measure.spec.ts, measure-compare.spec.ts, gpu-proof.spec.ts).
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { Page } from '@playwright/test';
import { expect } from './fixtures';
import { SCREENSHOT_DIR, waitForReady } from './helpers';
import { openTab } from './f123t.helpers';

/** Scratch folder for dry-run result fragments (inside node_modules, so git ignores it and Playwright does not wipe it). */
export const DRYRUN_DIR = resolve(process.cwd(), 'node_modules', '.cache', 'dryrun');
export const dryRunFile = (name: string) => resolve(DRYRUN_DIR, name);

export function record(name: string, data: unknown): void {
  const file = dryRunFile(`${name}.json`);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(data, null, 2));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function readRecord<T = Record<string, any>>(name: string): T | null {
  const file = dryRunFile(`${name}.json`);
  return existsSync(file) ? (JSON.parse(readFileSync(file, 'utf8')) as T) : null;
}

/** Viewport screenshot (small) kept in test-results/screenshots. */
export async function dryShot(page: Page, name: string): Promise<void> {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({ path: `${SCREENSHOT_DIR}/dryrun-${name}.png` });
}

export const subTab = (page: Page, id: 'perturb' | 'stamp' | 'images') => page.getByTestId(`break-subtab-${id}`).click();
export const txt = async (page: Page, id: string) => ((await page.getByTestId(id).textContent()) ?? '').trim();
export const pctNum = (s: string) => parseFloat(s.replace('%', ''));

/** Load the app and wait for Ready. Returns the wall time in ms from navigation start to "Ready". */
export async function boot(page: Page, query = '?debug=1'): Promise<number> {
  const t0 = Date.now();
  await page.goto(`/${query}`);
  await waitForReady(page, 120_000);
  return Date.now() - t0;
}

export async function tensorsInfo(page: Page) {
  return page.evaluate(() => {
    const m = window.__vbw!.tf.memory() as { numTensors: number; numBytes: number; numBytesInGPU?: number };
    const perf = performance as unknown as { memory?: { usedJSHeapSize: number; totalJSHeapSize: number } };
    return {
      numTensors: m.numTensors,
      numBytes: m.numBytes,
      numBytesInGPU: m.numBytesInGPU ?? null,
      heapUsed: perf.memory?.usedJSHeapSize ?? null,
    };
  });
}

export function stats(xs: number[]) {
  const n = xs.length;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
  return { n, min: Math.min(...xs), max: Math.max(...xs), mean, sd, spread: Math.max(...xs) - Math.min(...xs) };
}

// ------------------------------------------------------------------ perturb helpers

export interface PerturbRead {
  balancedPerturbed: number;
  balancedClean: number;
  accuracyPerturbed: number;
  accuracyClean: number;
  flipped: number;
  n: number;
  meanAbsDelta: number;
  appSeconds: number;
  wallMs: number;
}

const perturbCounters = (page: Page) => page.evaluate(() => ({ ...window.__perturbStats! }));

/** Run `act` (a slider change), wait for the recomputation it triggers to complete, and read the result panel. */
export async function perturbAndRead(page: Page, act: () => Promise<void>): Promise<PerturbRead> {
  const before = await perturbCounters(page);
  const t0 = Date.now();
  await act();
  await page.waitForFunction(
    (b) => {
      const s = window.__perturbStats;
      return !!s && s.completed > b.completed;
    },
    before,
    { timeout: 120_000, polling: 25 },
  );
  const wallMs = Date.now() - t0;
  await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-running', 'false', { timeout: 30_000 });
  await page.waitForTimeout(80);
  const flipsText = await txt(page, 'perturb-flips'); // "12 of 100 predictions changed"
  const m = /(\d+) of (\d+)/.exec(flipsText)!;
  const dur = await txt(page, 'perturb-duration'); // "Computed in 1.2 s for 100 images."
  const secs = /Computed in ([\d.]+) s/.exec(dur);
  const mins = /Computed in (\d+) min (\d+) s/.exec(dur);
  return {
    balancedPerturbed: pctNum(await txt(page, 'perturbed-balanced')),
    balancedClean: pctNum(await txt(page, 'clean-balanced')),
    accuracyPerturbed: pctNum(await txt(page, 'perturbed-accuracy')),
    accuracyClean: pctNum(await txt(page, 'clean-accuracy')),
    flipped: Number(m[1]),
    n: Number(m[2]),
    meanAbsDelta: pctNum(await txt(page, 'perturb-delta')),
    appSeconds: secs ? Number(secs[1]) : mins ? Number(mins[1]) * 60 + Number(mins[2]) : NaN,
    wallMs,
  };
}

export async function trainWithSeed(page: Page, seed: number, timeout = 90_000): Promise<number> {
  await openTab(page, 'Train');
  const advanced = page.getByTestId('advanced');
  if (!(await advanced.evaluate((el) => (el as HTMLDetailsElement).open))) await advanced.locator('summary').click();
  await page.getByTestId('seed-input').fill(String(seed));
  await expect(page.getByTestId('train-button')).toBeEnabled();
  const t0 = Date.now();
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished \d+ epochs/, { timeout });
  return Date.now() - t0;
}

// ------------------------------------------------------------------ generated pictures (no X-rays)

export interface GenImage {
  name: string;
  note: string;
  extra: boolean;
  buffer: Buffer;
}

/** Synthetic non-X-ray pictures, drawn in the browser (canvas) so no image library or fixture files are needed. */
export async function genImages(page: Page): Promise<GenImage[]> {
  const items = await page.evaluate(() => {
    function rng(seed: number) {
      let a = seed >>> 0;
      return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }
    const make = (w: number, h: number, draw: (c: CanvasRenderingContext2D, w: number, h: number) => void) => {
      const cv = document.createElement('canvas');
      cv.width = w;
      cv.height = h;
      const c = cv.getContext('2d') as CanvasRenderingContext2D;
      draw(c, w, h);
      return cv.toDataURL('image/png').split(',')[1] as string;
    };
    const clamp = (v: number) => Math.max(0, Math.min(255, v));
    const out: { name: string; note: string; extra: boolean; b64: string }[] = [];
    out.push({
      name: 'photo-scene',
      note: 'synthetic landscape: sky gradient, sun, hills, a house, trees, film grain',
      extra: false,
      b64: make(512, 384, (c, w, h) => {
        const sky = c.createLinearGradient(0, 0, 0, h * 0.65);
        sky.addColorStop(0, '#4f93e0');
        sky.addColorStop(1, '#d8ecff');
        c.fillStyle = sky;
        c.fillRect(0, 0, w, h);
        c.fillStyle = '#ffe36b';
        c.beginPath();
        c.arc(w * 0.8, h * 0.2, 40, 0, Math.PI * 2);
        c.fill();
        c.fillStyle = '#3e8a3a';
        c.beginPath();
        c.ellipse(w * 0.3, h * 0.85, w * 0.55, h * 0.3, 0, 0, Math.PI * 2);
        c.fill();
        c.fillStyle = '#2f7030';
        c.beginPath();
        c.ellipse(w * 0.85, h * 0.9, w * 0.45, h * 0.25, 0, 0, Math.PI * 2);
        c.fill();
        c.fillStyle = '#b5703a';
        c.fillRect(w * 0.28, h * 0.5, 120, 80);
        c.fillStyle = '#a02828';
        c.beginPath();
        c.moveTo(w * 0.28 - 10, h * 0.5);
        c.lineTo(w * 0.28 + 60, h * 0.5 - 55);
        c.lineTo(w * 0.28 + 130, h * 0.5);
        c.fill();
        for (const [tx, ty] of [
          [0.1, 0.62],
          [0.55, 0.7],
          [0.68, 0.6],
        ] as const) {
          c.fillStyle = '#5a3a1a';
          c.fillRect(w * tx, h * ty, 10, 40);
          c.fillStyle = '#1f6a2a';
          c.beginPath();
          c.arc(w * tx + 5, h * ty - 5, 30, 0, Math.PI * 2);
          c.fill();
        }
        const r = rng(7);
        const img = c.getImageData(0, 0, w, h);
        for (let i = 0; i < img.data.length; i += 4) {
          const n = (r() - 0.5) * 24;
          img.data[i] = clamp((img.data[i] as number) + n);
          img.data[i + 1] = clamp((img.data[i + 1] as number) + n);
          img.data[i + 2] = clamp((img.data[i + 2] as number) + n);
        }
        c.putImageData(img, 0, 0);
      }),
    });
    out.push({
      name: 'colour-gradient',
      note: 'diagonal rainbow gradient',
      extra: false,
      b64: make(400, 400, (c, w, h) => {
        const g = c.createLinearGradient(0, 0, w, h);
        ['#ff0000', '#ffcc00', '#22cc44', '#2255ff', '#aa22cc'].forEach((col, i, a) => g.addColorStop(i / (a.length - 1), col));
        c.fillStyle = g;
        c.fillRect(0, 0, w, h);
      }),
    });
    out.push({
      name: 'random-noise',
      note: 'uniform random RGB noise (seeded)',
      extra: false,
      b64: make(256, 256, (c, w, h) => {
        const r = rng(99);
        const img = c.createImageData(w, h);
        for (let i = 0; i < img.data.length; i += 4) {
          img.data[i] = Math.floor(r() * 256);
          img.data[i + 1] = Math.floor(r() * 256);
          img.data[i + 2] = Math.floor(r() * 256);
          img.data[i + 3] = 255;
        }
        c.putImageData(img, 0, 0);
      }),
    });
    out.push({
      name: 'drawn-cup',
      note: 'a white coffee cup with a handle on a beige table',
      extra: false,
      b64: make(400, 400, (c, w, h) => {
        c.fillStyle = '#d9c7a3';
        c.fillRect(0, 0, w, h);
        c.fillStyle = 'rgba(0,0,0,0.18)';
        c.beginPath();
        c.ellipse(w / 2, h * 0.82, 120, 22, 0, 0, Math.PI * 2);
        c.fill();
        c.fillStyle = '#fafafa';
        c.strokeStyle = '#888';
        c.lineWidth = 4;
        c.beginPath();
        c.moveTo(110, 130);
        c.lineTo(290, 130);
        c.lineTo(262, 310);
        c.quadraticCurveTo(200, 335, 138, 310);
        c.closePath();
        c.fill();
        c.stroke();
        c.beginPath();
        c.ellipse(290, 190, 45, 55, 0, -Math.PI / 2, Math.PI / 2);
        c.lineWidth = 14;
        c.strokeStyle = '#fafafa';
        c.stroke();
        c.fillStyle = '#4a2c17';
        c.beginPath();
        c.ellipse(200, 132, 90, 16, 0, 0, Math.PI * 2);
        c.fill();
      }),
    });
    out.push({
      name: 'solid-red',
      note: 'one flat red colour',
      extra: false,
      b64: make(300, 300, (c, w, h) => {
        c.fillStyle = '#d02020';
        c.fillRect(0, 0, w, h);
      }),
    });
    out.push({
      name: 'text-page',
      note: 'black text on white',
      extra: false,
      b64: make(480, 360, (c, w, h) => {
        c.fillStyle = '#ffffff';
        c.fillRect(0, 0, w, h);
        c.fillStyle = '#111';
        c.font = 'bold 44px sans-serif';
        c.fillText('HELLO', 40, 90);
        c.font = '28px serif';
        ['The quick brown fox', 'jumps over the lazy dog.', 'Not a chest X-ray.'].forEach((line, i) => c.fillText(line, 40, 160 + i * 48));
      }),
    });
    out.push({
      name: 'solid-black',
      note: 'extra: flat black',
      extra: true,
      b64: make(300, 300, (c, w, h) => {
        c.fillStyle = '#000';
        c.fillRect(0, 0, w, h);
      }),
    });
    out.push({
      name: 'solid-white',
      note: 'extra: flat white',
      extra: true,
      b64: make(300, 300, (c, w, h) => {
        c.fillStyle = '#fff';
        c.fillRect(0, 0, w, h);
      }),
    });
    return out;
  });
  return items.map((i) => ({ name: i.name, note: i.note, extra: i.extra, buffer: Buffer.from(i.b64, 'base64') }));
}
