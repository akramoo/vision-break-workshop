// Scripted, captioned screen recording of the whole app (about two minutes). Not a test: it is skipped unless DEMO=1.
//
//   DEMO=1 npx playwright test --project=chromium-gpu --workers=1 demo.spec.ts
//
// Playwright's video has no mouse cursor and no captions, so an init script draws both inside the page: a caption bar at the
// bottom, a cursor dot with a click ripple, and full-screen title / closing cards. The run records a 1280x720 .webm and copies it
// to private/demo/vision-demo.webm (private/ is git-ignored), plus a few PNG frames read back from the finished video.
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { test, expect, type Locator, type Page } from '@playwright/test';

test.skip(!process.env['DEMO'], 'set DEMO=1 to record the demo video');
test.setTimeout(420_000);

const OUT_DIR = resolve(process.cwd(), '..', 'private', 'demo');
const RAW_DIR = resolve(process.cwd(), 'test-results', 'demo-video');
const OUT_VIDEO = resolve(OUT_DIR, 'vision-demo.webm');
const SIZE = { width: 1280, height: 720 };

// ---------------------------------------------------------------------------------------------------------------------
// Overlay injected into every page: captions, cursor, cards.
// ---------------------------------------------------------------------------------------------------------------------
function overlayScript() {
  if (window.top !== window) return;
  const NAVY = '#0F1E3A';
  const PAPER = '#F6F2E9';
  const AMBER = '#F2A93B';
  const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
  const MAX_Z = '2147483647';

  const style = document.createElement('style');
  style.textContent = `
    html body { padding-bottom: 140px !important; }
    #demo-caption { position: fixed; left: 0; right: 0; bottom: 0; z-index: ${MAX_Z}; box-sizing: border-box; min-height: 92px; padding: 14px 56px;
      display: flex; align-items: center; justify-content: center; text-align: center; background: rgba(15,30,58,0.94); color: ${PAPER};
      font: 600 28px/1.25 ${FONT}; pointer-events: none; opacity: 0; transition: opacity 0.35s ease; border-top: 3px solid ${AMBER}; }
    #demo-cursor { position: fixed; left: 0; top: 0; z-index: ${MAX_Z}; width: 26px; height: 26px; margin: -13px 0 0 -13px; border-radius: 50%;
      box-sizing: border-box; border: 4px solid ${AMBER}; background: rgba(242,169,59,0.35); box-shadow: 0 0 0 2px rgba(15,30,58,0.55);
      pointer-events: none; opacity: 0; transition: opacity 0.2s ease; }
    .demo-ripple { position: fixed; z-index: ${MAX_Z}; width: 26px; height: 26px; margin: -13px 0 0 -13px; border-radius: 50%; box-sizing: border-box;
      border: 4px solid ${AMBER}; pointer-events: none; animation: demo-ripple 0.65s ease-out forwards; }
    @keyframes demo-ripple { from { transform: scale(0.6); opacity: 0.95; } to { transform: scale(3.2); opacity: 0; } }
    #demo-card { position: fixed; inset: 0; z-index: ${MAX_Z}; box-sizing: border-box; padding: 0 110px; display: flex; flex-direction: column;
      align-items: center; justify-content: center; text-align: center; background: ${NAVY}; color: ${PAPER}; font-family: ${FONT};
      pointer-events: none; opacity: 1; transition: opacity 0.9s ease; }
    #demo-card h1 { margin: 0 0 26px; font-size: 62px; line-height: 1.12; font-weight: 800; letter-spacing: -0.5px; }
    #demo-card p { margin: 0 0 16px; font-size: 30px; line-height: 1.35; font-weight: 500; }
    #demo-card p.small { margin-top: 30px; font-size: 26px; color: ${AMBER}; font-weight: 700; }
    #demo-card p.url { font-size: 36px; font-weight: 700; color: ${AMBER}; }
    #demo-card p.url span { color: ${PAPER}; font-weight: 500; font-size: 26px; display: block; margin-bottom: 4px; }
    .demo-hl { outline: 4px solid ${AMBER} !important; outline-offset: 4px !important; border-radius: 6px; }
  `;

  const caption = document.createElement('div');
  caption.id = 'demo-caption';
  const cursor = document.createElement('div');
  cursor.id = 'demo-cursor';
  const card = document.createElement('div');
  card.id = 'demo-card';
  card.innerHTML =
    '<h1>Build a Vision Model, Then Break It</h1>' +
    '<p>Train a pneumonia detector in your browser, then find out how it fails.</p>' +
    '<p class="small">Educational only. Not for clinical use.</p>';

  const mount = () => {
    document.documentElement.append(style, caption, cursor, card);
  };
  if (document.documentElement) mount();
  else document.addEventListener('DOMContentLoaded', mount, { once: true });

  let timer: number | undefined;
  const w = window as unknown as Record<string, unknown>;
  w['__demoCaption'] = (text: string) => {
    window.clearTimeout(timer);
    if (!text) {
      caption.style.opacity = '0';
      return;
    }
    caption.style.opacity = '0';
    timer = window.setTimeout(() => {
      caption.textContent = text;
      caption.style.opacity = '1';
    }, 300);
  };
  w['__demoCard'] = (html: string | null) => {
    if (html === null) {
      card.style.opacity = '0';
      return;
    }
    card.innerHTML = html;
    card.style.opacity = '1';
  };
  document.addEventListener(
    'mousemove',
    (e) => {
      cursor.style.opacity = '1';
      cursor.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    },
    true,
  );
  document.addEventListener(
    'mousedown',
    (e) => {
      const r = document.createElement('div');
      r.className = 'demo-ripple';
      r.style.left = `${e.clientX}px`;
      r.style.top = `${e.clientY}px`;
      document.documentElement.append(r);
      window.setTimeout(() => r.remove(), 800);
    },
    true,
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------------------------------
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Show a caption (it fades in after the old one fades out) and give the viewer `readMs` to read it. */
async function caption(page: Page, text: string, readMs = 1500) {
  await page.evaluate((t) => (window as unknown as { __demoCaption: (s: string) => void }).__demoCaption(t), text);
  await sleep(400 + readMs);
}

async function card(page: Page, html: string | null) {
  await page.evaluate((h) => (window as unknown as { __demoCard: (s: string | null) => void }).__demoCard(h), html);
}

/** Slow wheel scroll by `dy` pixels in small increments so the page glides. */
async function glideScroll(page: Page, dy: number) {
  const steps = Math.min(30, Math.max(1, Math.ceil(Math.abs(dy) / 40)));
  for (let i = 0; i < steps; i++) {
    await page.mouse.wheel(0, dy / steps);
    await sleep(12);
  }
  await sleep(180);
}

/** Native smooth scroll to the top or bottom of the page (long distances). */
async function smoothTo(page: Page, where: 'top' | 'bottom') {
  await page.evaluate((w) => window.scrollTo({ top: w === 'top' ? 0 : document.body.scrollHeight, behavior: 'smooth' }), where);
  await sleep(1100);
}

/** Scroll until the element sits comfortably between the header and the caption bar (above y = 590). */
async function reveal(page: Page, loc: Locator) {
  for (let i = 0; i < 5; i++) {
    const b = await loc.boundingBox();
    if (!b) {
      await loc.scrollIntoViewIfNeeded();
      await sleep(200);
      continue;
    }
    const tall = b.height > 380;
    if (b.y >= 90 && (tall ? b.y <= 200 : b.y + b.height <= 585)) return b;
    const wanted = tall ? 130 : Math.max(110, 330 - b.height / 2);
    await glideScroll(page, b.y - wanted);
  }
  return loc.boundingBox();
}

async function glideTo(page: Page, x: number, y: number) {
  await page.mouse.move(x, y, { steps: 25 });
  await sleep(120);
}

/** Visible click target: an input that is visually hidden inside a label falls back to the label. */
async function clickable(loc: Locator): Promise<Locator> {
  const b = await loc.boundingBox();
  if (b && b.width >= 8 && b.height >= 8) return loc;
  return loc.locator('xpath=..');
}

async function clickWithCursor(page: Page, target: Locator, after = 600) {
  const loc = await clickable(target);
  const b = await reveal(page, loc);
  if (!b) throw new Error('clickWithCursor: element has no box');
  await glideTo(page, b.x + b.width / 2, b.y + b.height / 2);
  await page.mouse.down();
  await sleep(70);
  await page.mouse.up();
  await sleep(after);
}

/** Move the cursor onto an element, outline it in amber and keep it there for `holdMs`. */
async function pointAt(page: Page, target: Locator, holdMs = 1800) {
  const b = await reveal(page, target);
  if (!b) throw new Error('pointAt: element has no box');
  await target.evaluate((el) => el.classList.add('demo-hl'));
  await glideTo(page, b.x + Math.min(b.width / 2, 120), b.y + b.height / 2);
  await sleep(holdMs);
  await target.evaluate((el) => el.classList.remove('demo-hl'));
}

/** Drag a range input's thumb with the mouse (then make sure the exact value landed). */
async function dragRange(page: Page, input: Locator, value: number) {
  const b = await reveal(page, input);
  if (!b) throw new Error('dragRange: no box');
  const min = Number(await input.getAttribute('min'));
  const max = Number(await input.getAttribute('max'));
  const cur = Number(await input.inputValue());
  const xOf = (v: number) => b.x + 8 + ((v - min) / (max - min)) * (b.width - 16);
  const y = b.y + b.height / 2;
  await glideTo(page, xOf(cur), y);
  await page.mouse.down();
  await sleep(120);
  await page.mouse.move(xOf(value), y, { steps: 30 });
  await sleep(120);
  await page.mouse.up();
  await sleep(300);
  const got = Number(await input.inputValue());
  if (Math.abs(got - value) > (max - min) * 0.02) await input.fill(String(value));
}

async function selectWithCursor(page: Page, sel: Locator, value: string) {
  const b = await reveal(page, sel);
  if (b) await glideTo(page, b.x + b.width / 2, b.y + b.height / 2);
  await sleep(500);
  await sel.selectOption(value);
  await sleep(500);
}

async function typeWithCursor(page: Page, field: Locator, text: string) {
  await clickWithCursor(page, field, 200);
  await page.keyboard.type(text, { delay: 35 });
  await sleep(500);
}

const openTab = (page: Page, name: string) => clickWithCursor(page, page.getByRole('tab', { name, exact: true }), 900);
const txt = async (page: Page, id: string) => ((await page.getByTestId(id).textContent()) ?? '').trim();

// ---------------------------------------------------------------------------------------------------------------------
// The recording
// ---------------------------------------------------------------------------------------------------------------------
test('record the demo video', async ({ browser }) => {
  mkdirSync(OUT_DIR, { recursive: true });
  mkdirSync(RAW_DIR, { recursive: true });

  // A brand-new context every run: no stored head, snapshots or log carried over from earlier sessions.
  const context = await browser.newContext({
    baseURL: 'http://127.0.0.1:4173',
    viewport: SIZE,
    screen: SIZE,
    deviceScaleFactor: 1,
    recordVideo: { dir: RAW_DIR, size: SIZE },
  });
  await context.addInitScript(overlayScript);
  const page = await context.newPage();
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`${e.name}: ${e.message}`));
  const t0 = Date.now();
  const stamp = () => `${((Date.now() - t0) / 1000).toFixed(1)}s`;
  const shot = (name: string) => page.screenshot({ path: resolve(OUT_DIR, `shot-${name}.png`) });
  const log = (m: string) => console.log(`[demo ${stamp()}] ${m}`);

  // 0. Title card: it is on screen from the first frame while the model loads behind it.
  await page.goto('/');
  await sleep(1800);
  await shot('0-title');
  await expect(page.getByTestId('model-status')).toHaveText('Ready', { timeout: 120_000 });
  while (Date.now() - t0 < 3600) await sleep(100);
  await card(page, null);
  await sleep(800);

  // 1. Open the app.
  log('scene 1');
  await caption(page, 'Everything runs in the browser. No server, no upload.', 800);
  await pointAt(page, page.getByTestId('backend-badge'), 1300);
  await shot('1-ready');

  // 2. Data tab.
  log('scene 2');
  await caption(page, 'Three pneumonia images for every normal one.', 500);
  await pointAt(page, page.getByTestId('counts-table'), 1700);
  await caption(page, 'The recipe: class balance and the fixes we can try later.', 400);
  await pointAt(page, page.getByTestId('recipe'), 1700);
  await shot('2-data-recipe');
  await reveal(page, page.getByTestId('thumb-grid'));
  await sleep(1500);
  await smoothTo(page, 'top');

  // 3. Train tab.
  log('scene 3');
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled({ timeout: 90_000 });
  await caption(page, 'A frozen MobileNet turns each X-ray into 1,280 numbers. We only train a small layer on top: a few seconds.', 400);
  await clickWithCursor(page, page.getByTestId('train-button'), 300);
  await sleep(800);
  await shot('3-training');
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 90_000 });
  const strip = await txt(page, 'model-strip-text');
  log(`model strip: ${strip}`);
  await reveal(page, page.getByTestId('chart-accuracy'));
  await caption(page, 'Validation accuracy about 99%. Looks great... but is it?', 2200);
  await shot('3-trained');

  // 4. Evaluate on the Kermany test set.
  log('scene 4');
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toBeVisible({ timeout: 30_000 });
  await smoothTo(page, 'top');
  await caption(page, 'On the test set: about 82% accuracy. But always saying "pneumonia" already scores 62.5%.', 500);
  log(`kermany model acc ${await txt(page, 'model-accuracy')}, baseline ${await txt(page, 'baseline-accuracy')}`);
  await pointAt(page, page.getByTestId('model-accuracy'), 900);
  await pointAt(page, page.getByTestId('baseline-accuracy'), 1500);
  await shot('4-evaluate-baseline');
  await caption(page, 'Sensitivity: sick children caught. Specificity: healthy children cleared.', 300);
  await pointAt(page, page.getByTestId('model-sensitivity'), 1000);
  await pointAt(page, page.getByTestId('model-specificity'), 1100);
  await caption(page, 'The confusion matrix counts every kind of mistake.', 300);
  await pointAt(page, page.getByTestId('confusion-matrix'), 1300);
  await shot('4-confusion');
  await caption(page, 'Mistakes come first in the list. Open the heatmap on the first one.', 300);
  await reveal(page, page.getByTestId('prediction-list'));
  await sleep(500);
  await clickWithCursor(page, page.getByTestId('pred-heatmap-btn').first(), 300);
  await expect(page.getByTestId('pred-heatmap-canvas-main')).toBeVisible({ timeout: 30_000 });
  await reveal(page, page.getByTestId('pred-heatmap-row'));
  await caption(page, 'Warm colours show where the model found evidence. A hint, not proof.', 2000);
  await shot('4-heatmap');
  // close it again so the list stays tidy for the failure-log scene
  await clickWithCursor(page, page.getByTestId('pred-heatmap-btn').first(), 300);

  // 5. Adult test set.
  log('scene 5');
  await smoothTo(page, 'top');
  await clickWithCursor(page, page.getByTestId('eval-set-adult_test'), 700);
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'adult_test');
  await caption(page, 'Same model on adult X-rays: about 52%. A different population breaks it.', 400);
  log(`adult model acc ${await txt(page, 'model-accuracy')}`);
  await pointAt(page, page.getByTestId('model-accuracy'), 1600);
  await shot('5-adult');

  // 6. Break it > Perturb.
  log('scene 6');
  await openTab(page, 'Break it');
  await clickWithCursor(page, page.getByTestId('break-subtab-perturb'), 500);
  await caption(page, 'Image quality: rotate or darken the picture and watch the answers change.', 400);
  const results = page.getByTestId('perturb-results');
  await expect(results).toHaveAttribute('data-running', 'false', { timeout: 120_000 });
  await dragRange(page, page.getByTestId('perturb-rotationDeg'), 20);
  await sleep(300);
  await dragRange(page, page.getByTestId('perturb-brightness'), -0.3);
  await expect(page.getByTestId('perturb-rotationDeg-value')).toHaveText('20°');
  await sleep(500);
  await expect(results).toHaveAttribute('data-running', 'false', { timeout: 120_000 });
  log(`perturb: ${await txt(page, 'perturb-flips')}`);
  await pointAt(page, page.getByTestId('perturb-flips'), 1800);
  await shot('6-perturb');

  // 7. Break it > Stamp.
  log('scene 7');
  await clickWithCursor(page, page.getByTestId('break-subtab-stamp'), 500);
  await caption(page, 'Shortcut learning: stamp a white R on pneumonia training images only.', 600);
  await expect(page.getByTestId('flip-before-normal')).toBeVisible({ timeout: 60_000 });
  await pointAt(page, page.getByTestId('stamp-preview'), 1200).catch(() => undefined);
  await clickWithCursor(page, page.getByTestId('stamp-train'), 300);
  await expect(page.getByTestId('stamp-train-msg')).toContainText(/Trained 20 epochs/, { timeout: 90_000 });
  await expect(page.getByTestId('flip-after-normal')).toBeVisible({ timeout: 60_000 });
  await caption(page, 'About 94% of normal X-rays now flip to pneumonia.', 300);
  log(`flip before ${await txt(page, 'flip-before-normal')} after ${await txt(page, 'flip-after-normal')}`);
  await pointAt(page, page.getByTestId('flip-table'), 1900);
  await shot('7-flip');
  await clickWithCursor(page, page.getByTestId('stamp-img-0'), 500);
  await expect(page.getByTestId('stamp-detail-state')).toHaveText('ON');
  await reveal(page, page.getByTestId('stamp-detail'));
  await sleep(600);
  await clickWithCursor(page, page.getByTestId('stamp-heatmap-show'), 300);
  await expect(page.getByTestId('stamp-heatmap-canvas-after')).toBeVisible({ timeout: 30_000 });
  await reveal(page, page.getByTestId('stamp-heatmap-canvas-after'));
  await caption(page, 'The stamp-trained model looks at the R, not the lungs.', 2200);
  await shot('7-stamp-heatmap');

  // 8. Refer threshold.
  log('scene 8');
  await openTab(page, 'Evaluate');
  await smoothTo(page, 'top');
  await clickWithCursor(page, page.getByTestId('eval-set-kermany_test'), 700);
  await caption(page, 'Let the model say "Refer to a doctor" when it is unsure.', 500);
  await dragRange(page, page.getByTestId('tau-slider'), 0.9);
  await expect(page.getByTestId('tau-value')).toHaveText('90%');
  await sleep(600);
  await pointAt(page, page.getByTestId('model-referred'), 1200);
  await pointAt(page, page.getByTestId('model-coverage'), 1800);
  await shot('8-refer');

  // 9. Failure log.
  log('scene 9');
  await caption(page, 'Every failure gets a tag and a hypothesis.', 300);
  await reveal(page, page.getByTestId('prediction-list'));
  await sleep(300);
  await clickWithCursor(page, page.getByTestId('pred-log').first(), 500);
  await expect(page.getByTestId('log-form')).toBeVisible();
  await reveal(page, page.getByTestId('log-form'));
  await selectWithCursor(page, page.getByTestId('log-tag-select'), 'shortcut');
  await typeWithCursor(page, page.getByTestId('log-hypothesis'), 'It leans on image markers, not the lungs');
  await shot('9-log-form');
  await clickWithCursor(page, page.getByTestId('log-save'), 900);
  await openTab(page, 'Failure log');
  await expect(page.getByTestId('log-entry')).toHaveCount(1);
  await reveal(page, page.getByTestId('log-entry').first());
  await sleep(1700);
  await shot('9-failure-log');

  // 10. Results.
  log('scene 10');
  await openTab(page, 'Results');
  const evalAll = page.getByTestId('results-eval-all');
  await expect(evalAll).toBeEnabled();
  await clickWithCursor(page, evalAll, 300);
  await expect(page.getByTestId('leaderboard-value')).toBeVisible({ timeout: 90_000 });
  await smoothTo(page, 'top');
  await caption(page, 'The worst case across three test sets is the score that counts.', 400);
  log(`leaderboard ${await txt(page, 'leaderboard-value')} on ${await txt(page, 'leaderboard-set')}`);
  await pointAt(page, page.getByTestId('leaderboard-value'), 2000);
  await shot('10-results');
  await caption(page, 'Fix, re-measure, compare.', 300);
  await clickWithCursor(page, page.getByTestId('snapshot-save'), 900);
  await expect(page.getByTestId('snap-name-0')).toBeVisible();
  await reveal(page, page.getByTestId('snapshot-table'));
  await sleep(900);

  // The footer, with the disclaimer, above the caption bar.
  await caption(page, 'Educational only. Not for clinical use. Everything stays on your computer.', 300);
  await smoothTo(page, 'bottom');
  await sleep(1000);
  await shot('10-footer');

  // Closing card.
  await caption(page, '', 100);
  await card(
    page,
    '<h1>Open source</h1>' +
      '<p class="url"><span>Code</span>github.com/akramoo/vision-break-workshop</p>' +
      '<p class="url"><span>Try it in your browser</span>akramoo.github.io/vision-break-workshop</p>' +
      '<p class="small">Educational only. Not for clinical use.</p>',
  );
  await sleep(3200);
  await shot('11-closing');

  const wallSeconds = (Date.now() - t0) / 1000;
  log(`done, wall time ${wallSeconds.toFixed(1)} s`);
  const videoPath = await page.video()!.path();
  await page.close();
  await context.close();
  expect(existsSync(videoPath)).toBe(true);
  copyFileSync(videoPath, OUT_VIDEO);
  const mb = statSync(OUT_VIDEO).size / 1048576;
  log(`video copied: ${OUT_VIDEO} (${mb.toFixed(2)} MB), recorded wall time about ${wallSeconds.toFixed(0)} s`);

  // Read the finished file back: real duration, and a few frames straight from the encoded video.
  const viewer = await browser.newContext({ viewport: SIZE, deviceScaleFactor: 1 });
  const vp = await viewer.newPage();
  await vp.goto(pathToFileURL(OUT_VIDEO).href);
  const info = await vp.evaluate(async () => {
    const v = document.querySelector('video') as HTMLVideoElement;
    v.controls = false;
    await new Promise<void>((ok) => (v.readyState >= 1 ? ok() : v.addEventListener('loadedmetadata', () => ok(), { once: true })));
    if (!Number.isFinite(v.duration)) {
      v.currentTime = 1e6;
      await new Promise<void>((ok) => v.addEventListener('durationchange', () => ok(), { once: true }));
      await new Promise((ok) => setTimeout(ok, 500));
    }
    return { duration: v.duration, w: v.videoWidth, h: v.videoHeight };
  });
  log(`video file: ${info.w}x${info.h}, duration ${info.duration.toFixed(1)} s`);
  if (Number.isFinite(info.duration)) {
    for (const frac of [0.02, 0.17, 0.32, 0.5, 0.64, 0.78, 0.9, 0.985]) {
      const at = info.duration * frac;
      await vp.evaluate(
        (t) =>
          new Promise<void>((ok) => {
            const v = document.querySelector('video') as HTMLVideoElement;
            v.pause();
            v.addEventListener('seeked', () => setTimeout(ok, 250), { once: true });
            v.currentTime = t;
          }),
        at,
      );
      await vp.screenshot({ path: resolve(OUT_DIR, `frame-${String(Math.round(frac * 100)).padStart(3, '0')}.png`) });
    }
  }
  await viewer.close();
  log(`page errors: ${errors.length ? errors.join(' | ') : 'none'}`);
  expect(errors).toEqual([]);
});
