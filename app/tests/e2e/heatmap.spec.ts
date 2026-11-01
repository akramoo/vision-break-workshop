// End-to-end checks for the Grad-CAM heatmap (stretch S1) in the production build: Evaluate rows, the Stamp tab's
// before/after maps and Uploads. Chromium and Firefox, one worker, precomputed features.
import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { deflateSync, crc32 } from 'node:zlib';
import { test, expect } from './fixtures';
import { SCREENSHOT_DIR, collectPageErrors, waitForReady } from './helpers';
import type { Page } from '@playwright/test';

test.setTimeout(90_000);

const DATA = resolve(process.cwd(), 'public', 'data');

const numTensors = (page: Page) => page.evaluate(() => window.__vbw?.numTensors() ?? -1);
const openTab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true }).click();

async function shot(page: Page, name: string) {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({ path: `${SCREENSHOT_DIR}/${name}.png` }); // viewport only
}

/** Console errors (not warnings) and uncaught exceptions. */
function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console.error: ${m.text()}`);
  });
  return errors;
}

async function trainHead(page: Page) {
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled({ timeout: 60_000 });
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });
}

type Cls = 'pneumonia' | 'normal';

/** Which "Evidence for" radio is checked for a heatmap with this test-id prefix. */
async function checkedClass(page: Page, prefix: string): Promise<Cls> {
  const pneu = await page.getByTestId(`${prefix}-class-pneumonia`).isChecked();
  const norm = await page.getByTestId(`${prefix}-class-normal`).isChecked();
  expect(pneu !== norm, `exactly one class radio is checked for ${prefix}`).toBe(true);
  return pneu ? 'pneumonia' : 'normal';
}
const other = (c: Cls): Cls => (c === 'pneumonia' ? 'normal' : 'pneumonia');

async function start(page: Page) {
  await page.goto('/?debug=1');
  await waitForReady(page);
  await trainHead(page);
}

interface CanvasInfo {
  hash: string;
  distinct: number;
  std: number;
  w: number;
  h: number;
}

/** Fingerprint of a canvas: FNV-1a hash of its pixels, number of distinct colours, and the standard deviation of the luma. */
async function canvasInfo(page: Page, testId: string): Promise<CanvasInfo> {
  return page.getByTestId(testId).evaluate((el) => {
    const c = el as HTMLCanvasElement;
    const ctx = c.getContext('2d') as CanvasRenderingContext2D;
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let h = 2166136261;
    const colours = new Set<number>();
    let sum = 0;
    let sum2 = 0;
    const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i] as number;
      const g = d[i + 1] as number;
      const b = d[i + 2] as number;
      h = Math.imul(h ^ r, 16777619);
      h = Math.imul(h ^ g, 16777619);
      h = Math.imul(h ^ b, 16777619);
      colours.add((r << 16) | (g << 8) | b);
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      sum += l;
      sum2 += l * l;
    }
    const mean = sum / n;
    return { hash: (h >>> 0).toString(16), distinct: colours.size, std: Math.sqrt(Math.max(0, sum2 / n - mean * mean)), w: c.width, h: c.height };
  });
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
  ihdr[8] = 8;
  ihdr[9] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const parsePeak = (s: string | null): [number, number] => {
  const m = /^(\d+),(\d+)$/.exec(s ?? '');
  if (!m) throw new Error(`bad data-peak ${JSON.stringify(s)}`);
  return [Number(m[1]), Number(m[2])];
};

test('Evaluate: a row heatmap renders, the class and strength controls change it, 10 open/close cycles leak no tensors', async ({ page, browserName }) => {
  const pageErrors = collectPageErrors(page);
  const consoleErrors = collectConsoleErrors(page);
  await start(page);
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toBeVisible({ timeout: 30_000 });
  const buttons = page.getByTestId('pred-heatmap-btn');
  await expect(buttons.first()).toBeVisible();
  const baseline = await numTensors(page);

  // open the first row
  await buttons.first().click();
  const row = page.getByTestId('pred-heatmap-row');
  await expect(row).toBeVisible();
  const main = page.getByTestId('pred-heatmap-canvas-main');
  await expect(main).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('pred-heatmap-status')).toHaveCount(0);
  await expect(page.getByTestId('pred-heatmap-error')).toHaveCount(0);
  await expect(page.getByTestId('pred-heatmap-caption')).toContainText('hint, not proof');
  // The control opens on the class the model PREDICTED for this row (intentional change: it used to always open on
  // 'pneumonia'). With 'Mistakes first' the first row can be either a false alarm or a miss.
  const saysFirst = ((await page.getByTestId('pred-row').first().locator('td').nth(2).textContent()) ?? '').trim();
  const firstCls: Cls = saysFirst.startsWith('pneumonia') ? 'pneumonia' : 'normal';
  expect(await checkedClass(page, 'pred-heatmap')).toBe(firstCls);
  const secondCls = other(firstCls);

  // not a blank or uniform canvas: 224 x 224, many colours, real spread of brightness
  const pneu = await canvasInfo(page, 'pred-heatmap-canvas-main');
  console.log(`[heatmap:${browserName}] EVAL canvas ${pneu.w}x${pneu.h}, ${pneu.distinct} colours, luma std ${pneu.std.toFixed(1)}, peak ${await main.getAttribute('data-peak')}`);
  expect([pneu.w, pneu.h]).toEqual([224, 224]);
  expect(pneu.distinct).toBeGreaterThan(50);
  expect(pneu.std).toBeGreaterThan(10);
  expect(await main.getAttribute('data-peak')).toMatch(/^[0-6],[0-6]$/);
  expect(await main.getAttribute('data-opacity')).toBe('0.5');
  await expect(page.getByTestId('pred-heatmap-opacity-value')).toHaveText('50%');

  // evidence for the other class draws a different image
  await page.getByTestId(`pred-heatmap-class-${secondCls}`).check();
  await expect.poll(async () => (await canvasInfo(page, 'pred-heatmap-canvas-main')).hash).not.toBe(pneu.hash);
  const normal = await canvasInfo(page, 'pred-heatmap-canvas-main');
  expect(normal.distinct).toBeGreaterThan(50);
  console.log(`[heatmap:${browserName}] EVAL normal map peak ${await main.getAttribute('data-peak')}, ${normal.distinct} colours`);

  // Back to the predicted class for the strength checks: for a very confident row the OTHER class can have no evidence at
  // all (an all-zero map), and then the strength slider has nothing to change.
  await page.getByTestId(`pred-heatmap-class-${firstCls}`).check();
  await expect.poll(async () => (await canvasInfo(page, 'pred-heatmap-canvas-main')).hash).toBe(pneu.hash);
  const atHalf = pneu;

  // strength slider: attribute and pixels follow; 0% shows the plain X-ray (no colour overlay)
  await page.getByTestId('pred-heatmap-opacity').fill('1');
  await expect(main).toHaveAttribute('data-opacity', '1');
  await expect(page.getByTestId('pred-heatmap-opacity-value')).toHaveText('100%');
  const strong = await canvasInfo(page, 'pred-heatmap-canvas-main');
  expect(strong.hash).not.toBe(atHalf.hash);
  await page.getByTestId('pred-heatmap-opacity').fill('0');
  await expect(main).toHaveAttribute('data-opacity', '0');
  const plain = await canvasInfo(page, 'pred-heatmap-canvas-main');
  expect(plain.hash).not.toBe(strong.hash);
  expect(plain.distinct).toBeGreaterThan(20); // still the X-ray itself
  await page.getByTestId('pred-heatmap-opacity').fill('0.5');
  await expect(main).toHaveAttribute('data-opacity', '0.5');

  // hide, then show again
  await page.getByTestId('pred-heatmap-hide').click();
  await expect(main).toHaveCount(0);
  await page.getByTestId('pred-heatmap-show').click();
  await expect(main).toBeVisible({ timeout: 30_000 });
  await page.getByTestId(`pred-heatmap-class-${secondCls}`).check();
  await page.getByTestId(`pred-heatmap-class-${firstCls}`).check();

  await row.scrollIntoViewIfNeeded();
  await shot(page, 'heatmap-evaluate');

  // close, then 10 more open / close cycles on different rows: the tensor count returns to the baseline
  await buttons.first().click();
  await expect(row).toHaveCount(0);
  for (let i = 0; i < 10; i++) {
    await buttons.nth(i).click();
    await expect(page.getByTestId('pred-heatmap-canvas-main')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId('pred-heatmap-status')).toHaveCount(0);
    await buttons.nth(i).click();
    await expect(row).toHaveCount(0);
  }
  await page.waitForTimeout(300);
  const after = await numTensors(page);
  console.log(`[heatmap:${browserName}] EVAL tensors baseline=${baseline} after 11 open/close cycles=${after}`);
  expect(after).toBe(baseline);

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test('Stamp: the stamp-trained model puts its evidence on the stamp (peak in cells 0..1 of the 7x7 grid) for at least 8 of 10 normal images', async ({ page, browserName }) => {
  const pageErrors = collectPageErrors(page);
  const consoleErrors = collectConsoleErrors(page);
  await start(page);
  await openTab(page, 'Break it');
  await page.getByTestId('break-subtab-stamp').click();
  await expect(page.getByTestId('flip-before-normal')).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('stamp-train').click();
  await expect(page.getByTestId('stamp-train-msg')).toContainText(/Trained 20 epochs/, { timeout: 60_000 });
  await expect(page.getByTestId('flip-after-normal')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('browse-class')).toHaveValue('normal');

  const detail = page.getByTestId('stamp-detail');
  const show = page.getByTestId('stamp-heatmap-show');
  const before = page.getByTestId('stamp-heatmap-canvas-before');
  const after = page.getByTestId('stamp-heatmap-canvas-after');
  let prevHash = '';
  const rows: { i: number; before: string; after: string; inStamp: boolean; pBefore: string | null; pAfter: string | null }[] = [];

  for (let i = 0; i < 10; i++) {
    await page.getByTestId(`stamp-img-${i}`).click(); // first click on an image: select it and switch its stamp ON
    await expect(page.getByTestId('stamp-detail-state')).toHaveText('ON');
    // the detail canvas now holds a new image (stamped); wait for it, then for the maps computed for exactly that image
    await expect
      .poll(async () => (await canvasInfo(page, 'stamp-detail-canvas')).hash, { timeout: 30_000 })
      .not.toBe(prevHash);
    prevHash = (await canvasInfo(page, 'stamp-detail-canvas')).hash;
    if (i === 0) {
      await expect(detail).toBeVisible();
      await show.click();
      await expect(page.getByTestId('stamp-heatmap-caption')).toContainText('hint, not proof');
    }
    await expect(after).toBeVisible({ timeout: 30_000 });
    await expect(before).toBeVisible();
    await expect(page.getByTestId('stamp-heatmap-status')).toHaveCount(0);
    const pb = await before.getAttribute('data-peak');
    const pa = await after.getAttribute('data-peak');
    const [ar, ac] = parsePeak(pa);
    const pBefore = await page.getByTestId('detail-before').textContent();
    const pAfter = await page.getByTestId('detail-after').textContent();
    rows.push({ i, before: pb ?? '', after: pa ?? '', inStamp: ar <= 1 && ac <= 1, pBefore: pBefore?.trim() ?? null, pAfter: pAfter?.trim() ?? null });
    if (i === 3) {
      // both maps are real pictures, and "normal" evidence differs from "pneumonia" evidence
      const a1 = await canvasInfo(page, 'stamp-heatmap-canvas-after');
      const b1 = await canvasInfo(page, 'stamp-heatmap-canvas-before');
      expect(a1.distinct).toBeGreaterThan(50);
      expect(b1.distinct).toBeGreaterThan(50);
      expect(a1.hash).not.toBe(b1.hash);
      // the control opens on the class the stamp-trained (after) model predicted for this image
      const c0 = await checkedClass(page, 'stamp-heatmap');
      await page.getByTestId(`stamp-heatmap-class-${other(c0)}`).check();
      await expect.poll(async () => (await canvasInfo(page, 'stamp-heatmap-canvas-after')).hash).not.toBe(a1.hash);
      await page.getByTestId(`stamp-heatmap-class-${c0}`).check();
      await expect.poll(async () => (await canvasInfo(page, 'stamp-heatmap-canvas-after')).hash).toBe(a1.hash);
    }
    if (i === 1) {
      await page.getByTestId('stamp-heatmap-caption').scrollIntoViewIfNeeded();
      await shot(page, 'heatmap-stamp');
    }
  }

  const hits = rows.filter((r) => r.inStamp).length;
  const dist = (key: 'before' | 'after') => {
    const m = new Map<string, number>();
    for (const r of rows) m.set(r[key], (m.get(r[key]) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}:${v}`).join(' ');
  };
  const beforeInStamp = rows.filter((r) => {
    const [br, bc] = parsePeak(r.before);
    return br <= 1 && bc <= 1;
  }).length;
  console.log(`[heatmap:${browserName}] STAMP stamped normal images 0..9, stamp-trained (AFTER) peak inside stamp cells (row,col <= 1): ${hits}/10`);
  console.log(`[heatmap:${browserName}] STAMP AFTER peaks: ${rows.map((r) => r.after).join(' | ')}  => ${dist('after')}`);
  console.log(`[heatmap:${browserName}] STAMP BEFORE (normal model) peaks: ${rows.map((r) => r.before).join(' | ')}  => ${dist('before')}; inside stamp cells: ${beforeInStamp}/10`);
  console.log(`[heatmap:${browserName}] STAMP says (before -> after): ${rows.map((r) => `${r.pBefore} -> ${r.pAfter}`).join(' ; ')}`);
  expect(hits, `AFTER peaks ${rows.map((r) => r.after).join(' ')}`).toBeGreaterThanOrEqual(8);

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test('Uploads: an uploaded image gets a heatmap', async ({ page, browserName }) => {
  const pageErrors = collectPageErrors(page);
  const consoleErrors = collectConsoleErrors(page);
  await start(page);
  await openTab(page, 'Evaluate');
  await page.getByTestId('eval-set-uploads').check();
  await page.getByTestId('upload-input').setInputFiles([
    { name: 'generated.png', mimeType: 'image/png', buffer: makePng(256, 256, 1) },
  ]);
  await expect(page.getByTestId('upload-count')).toHaveText('1 image');
  await expect(page.getByTestId('upload-predictions').locator('.pred-badge')).toHaveCount(1, { timeout: 30_000 });
  await page.getByTestId('upload-heatmap-btn-0').click();
  const main = page.getByTestId('upload-heatmap-0-canvas-main');
  await expect(main).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('upload-heatmap-0-error')).toHaveCount(0);
  await expect(page.getByTestId('upload-heatmap-0-caption')).toContainText('hint, not proof');
  const info = await canvasInfo(page, 'upload-heatmap-0-canvas-main');
  console.log(`[heatmap:${browserName}] UPLOAD canvas ${info.w}x${info.h}, ${info.distinct} colours, std ${info.std.toFixed(1)}, peak ${await main.getAttribute('data-peak')}`);
  expect(info.distinct).toBeGreaterThan(50);
  expect(info.std).toBeGreaterThan(10);
  const upCls = await checkedClass(page, 'upload-heatmap-0'); // the class the model predicted for this upload
  await page.getByTestId(`upload-heatmap-0-class-${other(upCls)}`).check();
  await expect.poll(async () => (await canvasInfo(page, 'upload-heatmap-0-canvas-main')).hash).not.toBe(info.hash);

  // closing the row removes it
  await page.getByTestId('upload-heatmap-btn-0').click();
  await expect(page.getByTestId('upload-heatmap-row-0')).toHaveCount(0);

  // a real shipped X-ray as an upload works too
  const xray = readFileSync(resolve(DATA, 'kermany/test/normal/kermany_test_normal_0001.jpg'));
  await page.getByTestId('upload-input').setInputFiles([{ name: 'xray.jpg', mimeType: 'image/jpeg', buffer: xray }]);
  await expect(page.getByTestId('upload-count')).toHaveText('2 images');
  await page.getByTestId('upload-heatmap-btn-1').click();
  await expect(page.getByTestId('upload-heatmap-1-canvas-main')).toBeVisible({ timeout: 30_000 });

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test('Evaluate: the heatmap opens on the class the model predicted, so a missed sick child starts on "normal" and can be switched', async ({ page }) => {
  const pageErrors = collectPageErrors(page);
  const consoleErrors = collectConsoleErrors(page);
  await start(page);
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('pred-mode-only-mistakes').check();

  // false alarms (true normal, model said pneumonia) open on 'pneumonia'; misses (true pneumonia, model said normal) on 'normal'
  for (const [trueClass, expected] of [['negative', 'pneumonia'], ['pneumonia', 'normal']] as const) {
    await page.getByTestId(`pred-class-${trueClass}`).check();
    const rows = page.getByTestId('pred-row');
    const n = Math.min(await rows.count(), 3);
    for (let i = 0; i < n; i++) {
      await rows.nth(i).getByTestId('pred-heatmap-btn').click();
      await expect(page.getByTestId('pred-heatmap-canvas-main')).toBeVisible({ timeout: 30_000 });
      expect(await checkedClass(page, 'pred-heatmap'), `${trueClass} mistake ${i}`).toBe(expected);
      if (i === 0) {
        // the student can still switch
        await page.getByTestId(`pred-heatmap-class-${other(expected)}`).check();
        await expect(page.getByTestId(`pred-heatmap-class-${other(expected)}`)).toBeChecked();
      }
      await rows.nth(i).getByTestId('pred-heatmap-btn').click();
      await expect(page.getByTestId('pred-heatmap-row')).toHaveCount(0);
    }
  }

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
});
