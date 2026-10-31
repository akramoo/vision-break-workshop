// End-to-end scenario for F4 (uploads, webcam), F5 (perturb), F6 (stamp) and F7 (abstain) in the production build.
// Runs in Chromium and Firefox (both use a fake camera, see playwright.config.ts); timing budgets are tighter on Chromium.
import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { deflateSync, crc32 } from 'node:zlib';
import { test, expect } from './fixtures';
import { SCREENSHOT_DIR, budget, collectPageErrors, waitForReady } from './helpers';
import type { Page } from '@playwright/test';

test.setTimeout(240_000);

const DATA = resolve(process.cwd(), 'public', 'data');

async function shot(page: Page, name: string) {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({ path: `${SCREENSHOT_DIR}/f4567-${name}.png`, fullPage: true });
}

const numTensors = (page: Page) => page.evaluate(() => window.__vbw?.numTensors() ?? -1);
const openTab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true }).click();
const subTab = (page: Page, id: 'perturb' | 'stamp' | 'images') => page.getByTestId(`break-subtab-${id}`).click();

async function setRange(page: Page, testId: string, value: number) {
  await page.getByTestId(testId).fill(String(value));
}

async function trainHead(page: Page) {
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled({ timeout: 90_000 });
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });
}

async function start(page: Page, url = '/?debug=1') {
  await page.goto(url);
  await waitForReady(page);
  await trainHead(page);
}

/** A grayscale PNG with a smooth gradient and a bright blob, written without any image library. */
function makePng(w: number, h: number, seed: number): Buffer {
  const raw = Buffer.alloc((w + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w + 1)] = 0;
    for (let x = 0; x < w; x++) {
      const d = Math.hypot(x - w * (0.3 + 0.1 * seed), y - h / 2);
      raw[y * (w + 1) + 1 + x] = Math.min(255, Math.round(40 + (x / w) * 120 + Math.max(0, 90 - d)));
    }
  }
  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body) >>> 0);
    return Buffer.concat([len, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 0; // grayscale
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** A JPEG made by the browser's own encoder (non-square on purpose, to exercise the center crop). */
async function makeJpeg(page: Page, w: number, h: number): Promise<Buffer> {
  const b64 = await page.evaluate(
    ([cw, ch]) => {
      const c = document.createElement('canvas');
      c.width = cw as number;
      c.height = ch as number;
      const ctx = c.getContext('2d') as CanvasRenderingContext2D;
      const g = ctx.createLinearGradient(0, 0, cw as number, ch as number);
      g.addColorStop(0, '#202020');
      g.addColorStop(1, '#e0e0e0');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, cw as number, ch as number);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect((cw as number) / 2 - 30, (ch as number) / 2 - 30, 60, 60);
      return c.toDataURL('image/jpeg', 0.9).split(',')[1] as string;
    },
    [w, h],
  );
  return Buffer.from(b64, 'base64');
}

test('F4 uploads: 4 images, one labeled, predictions and scores on labeled only; bad files are refused', async ({ page }) => {
  const errors = collectPageErrors(page);
  await start(page);
  await openTab(page, 'Evaluate');
  await page.getByTestId('eval-set-uploads').check();
  await expect(page.getByTestId('upload-notice')).toContainText("Do not upload your own or family members' medical images. Images never leave this browser.");
  await expect(page.getByTestId('eval-results')).toHaveCount(0);

  const normalXray = readFileSync(resolve(DATA, 'kermany/test/normal/kermany_test_normal_0001.jpg'));
  const jpeg = await makeJpeg(page, 300, 200);
  await page.getByTestId('upload-input').setInputFiles([
    { name: 'xray-normal.jpg', mimeType: 'image/jpeg', buffer: normalXray },
    { name: 'gradient-a.png', mimeType: 'image/png', buffer: makePng(256, 256, 1) },
    { name: 'wide.jpg', mimeType: 'image/jpeg', buffer: jpeg },
    { name: 'gradient-b.png', mimeType: 'image/png', buffer: makePng(320, 180, 2) },
  ]);
  await expect(page.getByTestId('upload-count')).toHaveText('4 images');
  await expect(page.getByTestId('upload-predictions').locator('tbody tr')).toHaveCount(4);
  // every row ends up with a prediction badge (closed world: it must pick a class)
  await expect(page.getByTestId('upload-predictions').locator('.pred-badge')).toHaveCount(4, { timeout: 30_000 });
  // nothing labeled yet: no scores, a hint instead
  await expect(page.getByTestId('upload-metrics')).toHaveCount(0);
  await expect(page.getByTestId('upload-metrics-hint')).toBeVisible();

  await page.getByTestId('upload-label-0').selectOption('normal');
  await expect(page.getByTestId('upload-metrics')).toBeVisible();
  await expect(page.getByTestId('upload-metrics-note')).toContainText('1 image you labeled');
  const acc = (await page.getByTestId('upload-model-accuracy').textContent())?.trim();
  expect(['0.0%', '100.0%']).toContain(acc);
  await page.getByTestId('upload-label-1').selectOption('pneumonia');
  await expect(page.getByTestId('upload-metrics-note')).toContainText('2 images you labeled');
  await expect(page.getByTestId('upload-baseline-accuracy')).toHaveText('50.0%');
  await shot(page, 'uploads-evaluate');

  // the same list shows in Break it; bad files are refused with a reason
  await openTab(page, 'Break it');
  await subTab(page, 'images');
  await expect(page.getByTestId('upload-count')).toHaveText('4 images');
  await page.getByTestId('upload-input').setInputFiles([
    { name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('hello') },
    { name: 'anim.gif', mimeType: 'image/gif', buffer: Buffer.from('GIF89a') },
  ]);
  await expect(page.getByTestId('upload-errors')).toContainText('notes.txt');
  await expect(page.getByTestId('upload-errors')).toContainText('JPEG, PNG and WebP');
  await expect(page.getByTestId('upload-count')).toHaveText('4 images');

  // remove one, then clear all
  await page.getByTestId('upload-remove-3').click();
  await expect(page.getByTestId('upload-count')).toHaveText('3 images');
  await page.getByTestId('upload-clear').click();
  await expect(page.getByTestId('upload-list')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('F4 uploads: drag and drop adds images', async ({ page }) => {
  await start(page);
  await openTab(page, 'Break it');
  await subTab(page, 'images');
  const dt = await page.evaluateHandle(async () => {
    const c = document.createElement('canvas');
    c.width = 120;
    c.height = 90;
    (c.getContext('2d') as CanvasRenderingContext2D).fillRect(10, 10, 50, 50);
    const blob = await new Promise<Blob>((res) => c.toBlob((b) => res(b as Blob), 'image/png'));
    const d = new DataTransfer();
    d.items.add(new File([blob], 'dropped.png', { type: 'image/png' }));
    return d;
  });
  await page.getByTestId('upload-dropzone').dispatchEvent('drop', { dataTransfer: dt });
  await expect(page.getByTestId('upload-count')).toHaveText('1 image');
});

test('F4 webcam: fake camera updates the overlay at about 4 fps; every track stops on Stop, sub-tab change, tab change', async ({ page }) => {
  const errors = collectPageErrors(page);
  await page.addInitScript(() => {
    // Record every stream so the test can check that the tracks really ended.
    const w = window as unknown as { __streams: MediaStream[] };
    w.__streams = [];
    const md = navigator.mediaDevices;
    const orig = md.getUserMedia.bind(md);
    md.getUserMedia = async (c) => {
      const s = await orig(c);
      w.__streams.push(s);
      return s;
    };
  });
  await start(page);
  await openTab(page, 'Break it');
  await subTab(page, 'images');
  await expect(page.getByTestId('webcam-hint')).toContainText('Try pointing it at a cup or your hand - the model must still pick a class.');
  const baseline = await numTensors(page);

  await page.getByTestId('webcam-start').click();
  await expect(page.getByTestId('webcam-status')).toHaveText('Camera on');
  await expect(page.getByTestId('webcam-video')).toBeVisible();
  const overlay = page.getByTestId('webcam-prediction');
  await expect(overlay).toBeVisible({ timeout: 30_000 });
  const frames = async () => Number((await overlay.getAttribute('data-frames')) ?? '0');
  const f0 = await frames();
  const t0 = Date.now();
  await page.waitForTimeout(3000);
  const f1 = await frames();
  const fps = ((f1 - f0) / (Date.now() - t0)) * 1000;
  console.log(`[f4567] WEBCAM overlay updates: ${f1 - f0} frames in 3 s = ${fps.toFixed(1)} fps`);
  expect(f1).toBeGreaterThan(f0 + 2);
  expect(fps).toBeLessThanOrEqual(4.5);
  expect(await page.evaluate(() => window.__vbw?.liveCameraTracks())).toBeGreaterThan(0);
  await shot(page, 'webcam-live');

  // Freeze & log keeps a failure-log entry with a thumbnail
  await page.getByTestId('webcam-freeze').click();
  await expect(page.getByTestId('webcam-frozen')).toBeVisible();
  const entry = await page.evaluate(() => window.__vbw?.getState().failureLog[0]);
  expect(entry?.setName).toBe('webcam');
  expect(entry?.thumbnail.startsWith('data:image/jpeg')).toBe(true);
  expect(typeof entry?.probPneumonia).toBe('number');
  expect(typeof entry?.createdAt).toBe('number');
  expect(entry?.tag).toBe('other');
  await expect(page.getByTestId('webcam-saved')).toContainText('1');

  // Stop: tracks ended, video detached
  await page.getByTestId('webcam-stop').click();
  await expect(page.getByTestId('webcam-status')).toHaveText('Camera off');
  expect(await page.evaluate(() => window.__vbw?.liveCameraTracks())).toBe(0);
  expect(await page.evaluate(() => (document.querySelector('video') as HTMLVideoElement).srcObject)).toBeNull();
  expect(await page.evaluate(() => (window as unknown as { __streams: MediaStream[] }).__streams.every((s) => s.getTracks().every((t) => t.readyState === 'ended')))).toBe(true);

  // Leaving the sub-tab stops the camera
  await page.getByTestId('webcam-start').click();
  await expect(page.getByTestId('webcam-status')).toHaveText('Camera on');
  expect(await page.evaluate(() => window.__vbw?.liveCameraTracks())).toBeGreaterThan(0);
  await subTab(page, 'perturb');
  expect(await page.evaluate(() => window.__vbw?.liveCameraTracks())).toBe(0);

  // Leaving the Break it tab stops the camera too
  await subTab(page, 'images');
  await page.getByTestId('webcam-start').click();
  await expect(page.getByTestId('webcam-status')).toHaveText('Camera on');
  await openTab(page, 'Evaluate');
  expect(await page.evaluate(() => window.__vbw?.liveCameraTracks())).toBe(0);
  expect(await page.evaluate(() => (window as unknown as { __streams: MediaStream[] }).__streams.every((s) => s.getTracks().every((t) => t.readyState === 'ended')))).toBe(true);
  expect(await numTensors(page)).toBe(baseline);
  expect(errors).toEqual([]);
});

for (const [name, message, kind] of [
  ['NotAllowedError', 'address bar', 'denied'],
  ['NotFoundError', 'No camera found', 'not-found'],
  ['NotReadableError', 'Another app', 'busy'],
] as const) {
  test(`F4 webcam error: ${name} shows an explanation`, async ({ page }) => {
    await page.addInitScript((errName) => {
      navigator.mediaDevices.getUserMedia = () => Promise.reject(Object.assign(new Error('blocked'), { name: errName }));
    }, name);
    await start(page);
    await openTab(page, 'Break it');
    await subTab(page, 'images');
    await page.getByTestId('webcam-start').click();
    const box = page.getByTestId('webcam-error');
    await expect(box).toBeVisible();
    await expect(box).toHaveAttribute('data-kind', kind);
    await expect(box).toContainText(message);
    await expect(page.getByTestId('webcam-status')).toHaveText('Camera off');
    if (name === 'NotAllowedError') await shot(page, 'webcam-denied');
  });
}

test('F5 perturb: identity reproduces the original, a brightness change moves the metrics, resets work', async ({ page }) => {
  const errors = collectPageErrors(page);
  await start(page);
  const baselineTensors = await numTensors(page);
  await openTab(page, 'Break it');
  await subTab(page, 'perturb');
  await expect(page.getByTestId('perturb-running')).toBeVisible({ timeout: 30_000 });
  const results = page.getByTestId('perturb-results');
  await expect(results).toHaveAttribute('data-running', 'false', { timeout: 120_000 });
  await expect(results).toHaveAttribute('data-n', '100');
  const txt = async (id: string) => (await page.getByTestId(id).textContent())?.trim() ?? '';
  // identity: exactly the unperturbed predictions
  await expect(page.getByTestId('perturb-flips')).toHaveText('0 of 100 predictions changed');
  expect(await txt('perturb-delta')).toBe('0.0%');
  expect(await txt('perturbed-accuracy')).toBe(await txt('clean-accuracy'));
  expect(await txt('baseline-accuracy')).toBe('63.0%'); // stratified 100-image sample of Kermany test: 63 pneumonia
  console.log(`[f4567] PERTURB first run (loads 100 images, base + perturbed): ${await txt('perturb-duration')}`);

  // a brightness change moves the metrics and flips some predictions
  await setRange(page, 'perturb-brightness', -0.4);
  await expect(results).toHaveAttribute('data-running', 'true', { timeout: 10_000 }).catch(() => undefined);
  await expect(page.getByTestId('perturb-brightness-value')).toHaveText('-0.40');
  await expect(page.getByTestId('perturb-flips')).not.toHaveText('0 of 100 predictions changed', { timeout: 60_000 });
  await expect(results).toHaveAttribute('data-running', 'false', { timeout: 60_000 });
  const dur = await txt('perturb-duration');
  const flips = await txt('perturb-flips');
  console.log(
    `[f4567] PERTURB brightness -0.4: ${flips}; mean |dp| ${await txt('perturb-delta')}; accuracy ${await txt('perturbed-accuracy')} vs clean ${await txt('clean-accuracy')}; ${dur}`,
  );
  expect(await txt('perturbed-accuracy')).not.toBe(await txt('clean-accuracy'));
  await expect(page.getByTestId('perturb-before')).toBeVisible();
  await expect(page.getByTestId('perturb-before-pred')).toBeVisible();
  await expect(page.getByTestId('perturb-after-pred')).toBeVisible();
  await shot(page, 'perturb');

  // each slider has its own Reset; Reset all returns to the identity and to the original predictions
  await setRange(page, 'perturb-rotationDeg', 20);
  await expect(page.getByTestId('perturb-rotationDeg-value')).toHaveText('20°');
  await page.getByTestId('perturb-reset-rotationDeg').click();
  await expect(page.getByTestId('perturb-rotationDeg-value')).toHaveText('0°');
  await page.getByTestId('perturb-reset-all').click();
  await expect(page.getByTestId('perturb-brightness-value')).toHaveText('0.00');
  await expect(page.getByTestId('perturb-flips')).toHaveText('0 of 100 predictions changed', { timeout: 60_000 });

  // the tab leaves no tensors behind
  await openTab(page, 'Evaluate');
  await page.waitForTimeout(300);
  expect(await numTensors(page)).toBe(baselineTensors);
  expect(errors).toEqual([]);
});

test('F5 perturb: a newer slider move aborts the run in flight; 20 rapid changes leak no tensors', async ({ page }) => {
  const errors = collectPageErrors(page);
  await start(page);
  const baselineTensors = await numTensors(page);
  await openTab(page, 'Break it');
  await subTab(page, 'perturb');
  await page.getByTestId('perturb-size').selectOption('50'); // keeps this test short on slow software WebGL
  const results = page.getByTestId('perturb-results');
  await expect(results).toHaveAttribute('data-n', '50', { timeout: 60_000 });
  await expect(results).toHaveAttribute('data-running', 'false', { timeout: 60_000 });
  // abort: a newer move cancels the run in flight
  const stats = () => page.evaluate(() => ({ ...window.__perturbStats! }));
  const before = await stats();
  await setRange(page, 'perturb-contrast', 0.6);
  await expect(page.getByTestId('perturb-running')).toBeVisible({ timeout: 5000 }); // debounce passed, run in flight
  await setRange(page, 'perturb-contrast', 1.4);
  await expect(results).toHaveAttribute('data-running', 'false', { timeout: 60_000 });
  await expect(page.getByTestId('perturb-contrast-value')).toHaveText('1.40');
  await expect.poll(async () => (await stats()).aborted, { timeout: 10_000 }).toBeGreaterThan(before.aborted);

  // 20 rapid slider changes leak no tensors
  await page.getByTestId('perturb-reset-all').click();
  await expect(results).toHaveAttribute('data-running', 'false', { timeout: 60_000 });
  for (let i = 0; i < 20; i++) await setRange(page, 'perturb-brightness', Math.round((i % 2 === 0 ? -1 : 1) * ((i % 6) + 1) * 5) / 100);
  await setRange(page, 'perturb-brightness', 0.2);
  await expect(page.getByTestId('perturb-brightness-value')).toHaveText('0.20');
  await expect(page.getByTestId('perturb-running')).toBeVisible({ timeout: 10_000 });
  await expect(results).toHaveAttribute('data-running', 'false', { timeout: 60_000 });
  await expect(page.getByTestId('perturb-running')).toHaveCount(0);
  await page.waitForTimeout(500);
  const after = await stats();
  console.log(`[f4567] PERTURB runs started=${after.started} completed=${after.completed} aborted=${after.aborted}`);
  expect(await numTensors(page)).toBe(baselineTensors);

  expect(errors).toEqual([]);
});

test('F5 perturb: uploads can be perturbed; a sample selector offers 50 / 100 / all', async ({ page }) => {
  await start(page);
  await openTab(page, 'Break it');
  await subTab(page, 'images');
  await page.getByTestId('upload-input').setInputFiles([
    { name: 'a.png', mimeType: 'image/png', buffer: makePng(224, 224, 1) },
    { name: 'b.png', mimeType: 'image/png', buffer: makePng(300, 200, 2) },
  ]);
  await expect(page.getByTestId('upload-count')).toHaveText('2 images');
  await subTab(page, 'perturb');
  await expect(page.getByTestId('perturb-size').locator('option')).toHaveText(['50 images', '100 images', 'all images (slow)']);
  await page.getByTestId('perturb-set').selectOption('uploads');
  await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-n', '2', { timeout: 60_000 });
  await expect(page.getByTestId('perturb-unlabeled')).toBeVisible();
  await page.getByTestId('perturb-set').selectOption('kermany_test');
  await page.getByTestId('perturb-size').selectOption('50');
  await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-n', '50', { timeout: 90_000 });
});

test('F6 stamp: stamp-train + flip rate (precomputed) in under 3 s; custom stamp extracts live with progress, cancel works', async ({ page, browserName }) => {
  const errors = collectPageErrors(page);
  await start(page);
  const baselineTensors = await numTensors(page);
  await openTab(page, 'Break it');
  await subTab(page, 'stamp');
  await expect(page.getByTestId('stamp-challenge')).toContainText('smallest stamp that still fools the model');
  await expect(page.getByTestId('stamp-teaching-note')).toBeVisible();
  await expect(page.getByTestId('stamp-preview')).toBeVisible();
  await expect(page.getByTestId('stamp-mode')).toContainText('instant');
  // "before" flip rate for the clean model is computed automatically (instant path)
  await expect(page.getByTestId('flip-before-normal')).toBeVisible({ timeout: 30_000 });
  await shot(page, 'stamp-config');

  // default config: train the stamped model; time from click to the flip table with an "after" row
  const t0 = Date.now();
  await page.getByTestId('stamp-train').click();
  await expect(page.getByTestId('stamp-train-msg')).toContainText(/Trained 20 epochs/, { timeout: 60_000 });
  await expect(page.getByTestId('flip-after-normal')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('flip-results')).toHaveAttribute('data-stale', 'false');
  const wallMs = Date.now() - t0;
  const txt = async (id: string) => (await page.getByTestId(id).textContent())?.trim() ?? '';
  const numbers = {
    beforeNormal: { unstamped: await txt('flip-before-normal-clean'), stamped: await txt('flip-before-normal-stamped'), flip: await txt('flip-before-normal') },
    afterNormal: { unstamped: await txt('flip-after-normal-clean'), stamped: await txt('flip-after-normal-stamped'), flip: await txt('flip-after-normal') },
    beforePneumonia: { unstamped: await txt('flip-before-pneumonia-clean'), stamped: await txt('flip-before-pneumonia-stamped'), flip: await txt('flip-before-pneumonia') },
    afterPneumonia: { unstamped: await txt('flip-after-pneumonia-clean'), stamped: await txt('flip-after-pneumonia-stamped'), flip: await txt('flip-after-pneumonia') },
  };
  console.log(`[f4567] STAMP stamp-train + flip wall=${wallMs} ms; ${await txt('flip-duration')}`);
  console.log(`[f4567] STAMP FLIP RATES (Kermany test, NORMAL images stamped) ${JSON.stringify(numbers, null, 1)}`);
  expect(wallMs).toBeLessThan(budget(browserName, 15_000)); // training itself (20 epochs) is part of this
  // the flip computation alone (precomputed variants already loaded): re-run it and time it
  const tFlip = Date.now();
  await page.getByTestId('flip-compute').click();
  await expect(page.getByTestId('flip-duration')).toContainText('Computed in');
  const flipMs = Date.now() - tFlip;
  console.log(`[f4567] STAMP flip recompute (precomputed) wall=${flipMs} ms`);
  expect(flipMs).toBeLessThan(budget(browserName, 3000));
  // the stamp-trained model reacts to the stamp far more than the clean one
  const rate = (s: string) => Number.parseFloat(s);
  expect(rate(numbers.afterNormal.flip)).toBeGreaterThan(rate(numbers.beforeNormal.flip));
  await expect(page.getByTestId('stamp-metrics')).toBeVisible();
  await shot(page, 'stamp-flip');

  // single image toggle
  await page.getByTestId('stamp-img-0').click();
  await expect(page.getByTestId('stamp-detail-state')).toHaveText('ON');
  await expect(page.getByTestId('detail-before')).toBeVisible();
  await expect(page.getByTestId('detail-after')).toBeVisible();
  const pAfterStamped = await page.getByTestId('detail-after').getAttribute('data-decision');
  await page.getByTestId('stamp-img-0').click();
  await expect(page.getByTestId('stamp-detail-state')).toHaveText('off');
  const pAfterClean = await page.getByTestId('detail-after').getAttribute('data-decision');
  console.log(`[f4567] STAMP first normal test image: stamp-trained model says ${pAfterClean} clean, ${pAfterStamped} stamped`);
  await page.getByTestId('stamp-img-1').click();
  await shot(page, 'stamp-single');
  await page.getByTestId('stamp-log').click();
  const logged = await page.evaluate(() => window.__vbw?.getState().failureLog.filter((f) => f.setName === 'stamp'));
  expect(logged).toHaveLength(1);
  expect(logged?.[0]?.thumbnail.startsWith('data:image/jpeg')).toBe(true);
  expect(logged?.[0]?.config).toMatchObject({ stamped: true });

  // custom config: live extraction with progress, and Cancel
  await setRange(page, 'stamp-size', 24);
  await expect(page.getByTestId('stamp-mode')).toContainText('Custom stamp');
  await page.getByTestId('flip-set').selectOption('adult_test');
  await page.getByTestId('flip-compute').click();
  await expect(page.getByTestId('flip-running')).toBeVisible();
  await page.getByTestId('flip-cancel').click();
  await expect(page.getByTestId('flip-running')).toHaveCount(0);
  const tLive = Date.now();
  await page.getByTestId('flip-compute').click();
  await expect(page.getByTestId('flip-running')).toBeVisible();
  await expect(page.getByTestId('flip-running')).toHaveCount(0, { timeout: 180_000 });
  await expect(page.getByTestId('flip-results')).toHaveAttribute('data-set', 'adult_test');
  console.log(`[f4567] STAMP custom 24 px live flip on 100 adult normals: ${Date.now() - tLive} ms; ${await txt('flip-duration')}`);
  console.log(`[f4567] STAMP custom flip before=${await txt('flip-before-normal')} after=${await txt('flip-after-normal')}`);

  // cancel a custom retrain: nothing changes
  const headBefore = await page.evaluate(() => window.__vbw?.getState().stampedHead?.trainedAt);
  await page.getByTestId('stamp-train').click();
  await expect(page.getByTestId('stamp-training')).toBeVisible();
  await page.getByTestId('stamp-train-cancel').click();
  await expect(page.getByTestId('stamp-train-msg')).toContainText(/Cancelled/, { timeout: 30_000 });
  expect(await page.evaluate(() => window.__vbw?.getState().stampedHead?.trainedAt)).toBe(headBefore);

  // leaving the tab keeps both models and leaves the tensor count flat
  await openTab(page, 'Evaluate');
  await page.waitForTimeout(500);
  expect(await numTensors(page)).toBe(baselineTensors);
  expect(errors).toEqual([]);
});

test('F7 abstain: tau 0.5 never refers; 0.9 refers and adds Referred/Coverage columns on Evaluate, Perturb and Stamp', async ({ page }) => {
  const errors = collectPageErrors(page);
  await start(page);
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toBeVisible();
  const slider = page.getByTestId('tau-slider');
  await expect(slider).toHaveValue('0.5');
  await expect(page.getByTestId('tau-control')).toContainText('Refer to a doctor when the model is less than 50% sure');
  await expect(page.getByTestId('tau-explain')).toContainText('referred cases get half credit in the worst-case score');
  await expect(page.getByTestId('model-referred')).toHaveCount(0);
  await expect(page.getByTestId('prediction-list')).not.toContainText('Refer to a doctor');

  await slider.fill('0.9');
  await expect(page.getByTestId('tau-value')).toHaveText('90%');
  await expect(page.getByTestId('tau-control')).toContainText('less than 90% sure');
  const referred = page.getByTestId('model-referred');
  await expect(referred).toBeVisible();
  const n = Number.parseInt((await referred.textContent()) ?? '0', 10);
  console.log(`[f4567] ABSTAIN tau=0.9 on Kermany test: referred "${(await referred.textContent())?.trim()}", coverage ${(await page.getByTestId('model-coverage').textContent())?.trim()}`);
  expect(n).toBeGreaterThan(0);
  await expect(page.getByTestId('model-coverage')).not.toHaveText('100.0%');
  await expect(page.getByTestId('baseline-referred')).toHaveText('-');
  await expect(page.getByTestId('abstain-note')).toBeVisible();
  await shot(page, 'abstain-evaluate');

  // tau is shared and persisted: same value in Break it, columns in Perturb and Stamp
  await openTab(page, 'Break it');
  await expect(page.getByTestId('tau-slider')).toHaveValue('0.9');
  await subTab(page, 'perturb');
  await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-running', 'false', { timeout: 120_000 });
  await expect(page.getByTestId('perturbed-referred')).toBeVisible();
  await expect(page.getByTestId('perturbed-coverage')).toBeVisible();
  await subTab(page, 'stamp');
  await expect(page.getByTestId('stamp-before-referred')).toBeVisible({ timeout: 30_000 });
  await page.reload();
  await waitForReady(page);
  await openTab(page, 'Break it');
  await expect(page.getByTestId('tau-slider')).toHaveValue('0.9');

  // back to 0.5: no referrals, no extra columns
  await page.getByTestId('tau-slider').fill('0.5');
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('model-referred')).toHaveCount(0);
  expect(errors).toEqual([]);
});
