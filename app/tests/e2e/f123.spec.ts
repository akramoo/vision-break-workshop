// End-to-end scenario for F1 (Data), F2 (Train) and F3 (Evaluate) in the production build.
// Chromium only: the timing budgets assume a desktop browser; the numbers are printed for the report.
import { mkdirSync } from 'node:fs';
import { test, expect } from './fixtures';
import { SCREENSHOT_DIR, collectPageErrors, waitForReady } from './helpers';
import type { Page } from '@playwright/test';

test.skip(({ browserName }) => browserName !== 'chromium', 'timing budgets are checked in Chromium');
test.describe.configure({ mode: 'serial' });

const numTensors = (page: Page) => page.evaluate(() => window.__vbw?.numTensors() ?? -1);

async function shot(page: Page, name: string) {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({ path: `${SCREENSHOT_DIR}/f123-${name}.png`, fullPage: true });
}

async function openTab(page: Page, name: string) {
  await page.getByRole('tab', { name, exact: true }).click();
}

async function metric(page: Page, row: 'model' | 'baseline', col: 'accuracy' | 'sensitivity' | 'specificity' | 'balanced') {
  return (await page.getByTestId(`${row}-${col}`).textContent())?.trim() ?? '';
}

async function pickSet(page: Page, id: 'kermany_test' | 'adult_test' | 'stress') {
  await page.getByTestId(`eval-set-${id}`).check();
}

test('Data tab: counts, weights, toggles, paged thumbnails, load status', async ({ page }) => {
  const errors = collectPageErrors(page);
  await page.goto('/?debug=1');
  await waitForReady(page);

  await expect(page.getByTestId('count-kermany-train')).toContainText('255');
  await expect(page.getByTestId('count-kermany-train')).toContainText('765');
  await expect(page.getByTestId('count-kermany-train').getByTestId('count-total')).toHaveText('1020');
  await expect(page.getByTestId('count-kermany-val').getByTestId('count-total')).toHaveText('180');
  await expect(page.getByTestId('count-kermany-val_official').getByTestId('count-total')).toHaveText('16');
  await expect(page.getByTestId('count-kermany-test').getByTestId('count-total')).toHaveText('624');
  await expect(page.getByTestId('count-nih-train_adult').getByTestId('count-total')).toHaveText('200');
  await expect(page.getByTestId('count-nih-test_adult').getByTestId('count-total')).toHaveText('200');

  await expect(page.getByTestId('data-source-status')).toContainText('Using precomputed features');
  await expect(page.getByTestId('embeddings-banner')).toHaveCount(0);

  await expect(page.getByTestId('class-weights')).toContainText('normal 1.00 / pneumonia 1.00');
  await page.getByRole('radio', { name: /Balanced/ }).check();
  await expect(page.getByTestId('class-weights')).toContainText('normal 2.00 / pneumonia 0.67');
  await page.getByRole('checkbox', { name: /Mix in adult X-rays \(\+200\)/ }).check();
  await expect(page.getByTestId('training-summary')).toContainText('1220 rows');
  await expect(page.getByTestId('class-weights')).toContainText('normal 1.72 / pneumonia 0.71');
  await page.getByRole('checkbox', { name: /Augmentation \(\+2 views per image\)/ }).check();
  await expect(page.getByTestId('training-summary')).toContainText('3660 rows');
  await page.getByRole('checkbox', { name: /Mix in adult/ }).uncheck();
  await page.getByRole('checkbox', { name: /Augmentation/ }).uncheck();
  await page.getByRole('radio', { name: /Realistic/ }).check();

  // gallery: 48 per page, lazy images, filters
  const thumbs = page.getByTestId('thumb-grid').locator('img');
  await expect(thumbs).toHaveCount(48);
  await expect(thumbs.first()).toHaveAttribute('loading', 'lazy');
  await expect(page.getByTestId('page-label')).toHaveText('Page 1 of 47');
  await page.getByTestId('page-next').click();
  await expect(page.getByTestId('page-label')).toHaveText('Page 2 of 47');
  await page.getByTestId('filter-split').selectOption('nih/test_adult');
  await page.getByTestId('filter-label').selectOption('normal');
  await expect(page.getByTestId('gallery-count')).toHaveText('100 images');
  await expect(page.getByTestId('thumb-grid').locator('.chip').first()).toHaveText('no finding');
  await page.getByTestId('filter-split').selectOption('kermany/test');
  await expect(page.getByTestId('thumb-grid').locator('.chip').first()).toHaveText('normal');
  await page.getByTestId('filter-split').selectOption('all');
  await page.getByTestId('filter-label').selectOption('all');
  await shot(page, 'data');
  expect(errors).toEqual([]);
});

test('Data tab: a broken embeddings file shows a banner and the app falls back to live', async ({ page }) => {
  await page.route('**/data/embeddings/base.index.json', (r) => r.fulfill({ status: 404, body: 'gone' }));
  await page.goto('/');
  await waitForReady(page);
  await expect(page.getByTestId('embeddings-banner')).toBeVisible();
  await expect(page.getByTestId('data-source-status')).toContainText('Live extraction');
  await expect(page.getByTestId('live-estimate')).toBeVisible();
  await shot(page, 'data-fallback');
  // the rest of the app still works
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled();
});

test('Train, Evaluate, Repeat x5, reload restores the head, live parity', async ({ page }) => {
  test.setTimeout(240_000);
  const errors = collectPageErrors(page);
  await page.goto('/?debug=1');
  await waitForReady(page);
  await expect(page.getByTestId('data-source-status')).toContainText('Using precomputed features');

  // ---- Evaluate before training: friendly empty state ----
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('evaluate-empty')).toBeVisible();
  await shot(page, 'evaluate-empty');

  // ---- Train ----
  await openTab(page, 'Train');
  const baseline = await numTensors(page);
  await expect(page.getByTestId('train-button')).toBeEnabled();
  const t0 = Date.now();
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('cancel-button')).toBeEnabled();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });
  const wallMs = Date.now() - t0;
  const hist = await page.evaluate(() => window.__vbw?.getState().train.history ?? []);
  const last = hist[hist.length - 1];
  const info = await page.evaluate(() => window.__vbw?.getState().headInfo);
  const afterTrain = await numTensors(page);
  console.log(
    `[f123] TRAIN epochs=${hist.length} valAcc=${last?.valAcc.toFixed(4)} trainAcc=${last?.trainAcc.toFixed(4)} ` +
      `valLoss=${last?.valLoss.toFixed(4)} trainMs=${Math.round(info?.durationMs ?? -1)} wallMs=${wallMs} ` +
      `tensors before=${baseline} after=${afterTrain}`,
  );
  expect(hist).toHaveLength(20);
  expect(last?.valAcc ?? 0).toBeGreaterThan(0.85);
  expect(wallMs).toBeLessThan(15_000);
  expect(afterTrain).toBe(baseline);
  await expect(page.getByTestId('chart-loss')).toBeVisible();
  await expect(page.getByTestId('chart-accuracy').locator('svg').first()).toHaveAttribute('role', 'img');
  await shot(page, 'train');

  // ---- Evaluate ----
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'kermany_test');
  expect(await metric(page, 'baseline', 'accuracy')).toBe('62.5%');
  expect(await metric(page, 'baseline', 'sensitivity')).toBe('100.0%');
  expect(await metric(page, 'baseline', 'specificity')).toBe('0.0%');
  const kermany = {
    acc: await metric(page, 'model', 'accuracy'),
    sens: await metric(page, 'model', 'sensitivity'),
    spec: await metric(page, 'model', 'specificity'),
    bal: await metric(page, 'model', 'balanced'),
  };
  // confusion matrix sums to n
  const cm = async () => {
    const v = await Promise.all(['tn', 'fp', 'fn', 'tp'].map((k) => page.getByTestId(`cm-${k}`).locator('.cm__count').textContent()));
    return v.map(Number);
  };
  const k = await cm();
  expect(k.reduce((a, b) => a + b, 0)).toBe(624);
  expect(k[0]! + k[1]!).toBe(234);
  expect(k[2]! + k[3]!).toBe(390);
  await expect(page.getByTestId('cost-note')).toContainText('missed sick child');
  await expect(page.getByTestId('prediction-list').locator('tbody tr')).toHaveCount(40);
  await shot(page, 'evaluate-kermany');

  await pickSet(page, 'adult_test');
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'adult_test');
  expect(await metric(page, 'baseline', 'accuracy')).toBe('50.0%');
  const adult = {
    acc: await metric(page, 'model', 'accuracy'),
    sens: await metric(page, 'model', 'sensitivity'),
    spec: await metric(page, 'model', 'specificity'),
    bal: await metric(page, 'model', 'balanced'),
  };
  const a = await cm();
  expect(a.reduce((x, y) => x + y, 0)).toBe(200);
  await expect(page.getByTestId('confusion-matrix')).toContainText('no finding');
  await shot(page, 'evaluate-adult');

  await pickSet(page, 'stress');
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'stress');
  const stress = {
    acc: await metric(page, 'model', 'accuracy'),
    sens: await metric(page, 'model', 'sensitivity'),
    spec: await metric(page, 'model', 'specificity'),
    bal: await metric(page, 'model', 'balanced'),
  };
  expect((await cm()).reduce((x, y) => x + y, 0)).toBe(624);
  await shot(page, 'evaluate-stress');
  console.log(`[f123] EVAL kermany=${JSON.stringify(kermany)} adult=${JSON.stringify(adult)} stress=${JSON.stringify(stress)}`);

  // evaluate everything again: results are cached, tensors stay at the baseline
  await pickSet(page, 'kermany_test');
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'kermany_test');
  expect(await numTensors(page)).toBe(baseline);

  // ---- Retrain with the same seed: identical numbers; tensors still flat ----
  await openTab(page, 'Train');
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });
  const hist2 = await page.evaluate(() => window.__vbw?.getState().train.history ?? []);
  expect(hist2[19]?.valAcc).toBeCloseTo(last?.valAcc ?? -1, 6);
  expect(hist2[19]?.trainLoss).toBeCloseTo(last?.trainLoss ?? -1, 4);
  expect(await numTensors(page)).toBe(baseline);
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toBeVisible();
  expect(await metric(page, 'model', 'accuracy')).toBe(kermany.acc);
  expect(await numTensors(page)).toBe(baseline);

  // ---- Cancel stops within an epoch ----
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  await page.getByTestId('epochs-input').fill('100');
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Training: epoch/, { timeout: 30_000 });
  await page.getByTestId('cancel-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Stopped after epoch/, { timeout: 10_000 });
  const stopped = await page.evaluate(() => window.__vbw?.getState().train.history.length ?? 0);
  expect(stopped).toBeLessThan(100);
  await page.getByTestId('epochs-input').fill('20');
  expect(await numTensors(page)).toBe(baseline);
  // train the original model again (same seed, so it is identical) so the saved head matches kermany below
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });

  // ---- Repeat x5 ----
  const r0 = Date.now();
  await page.getByTestId('repeat-button').click();
  await expect(page.getByTestId('repeat-status')).toContainText('Finished 5 runs', { timeout: 120_000 });
  const repeatMs = Date.now() - r0;
  const runs = await page.evaluate(() => window.__vbw?.getState().repeat.runs ?? []);
  expect(runs).toHaveLength(5);
  const spread = (xs: (number | null)[]) => {
    const v = xs.filter((x): x is number => x !== null);
    return Math.max(...v) - Math.min(...v);
  };
  const s16 = spread(runs.map((r) => r.val16));
  const s180 = spread(runs.map((r) => r.val180));
  const sTest = spread(runs.map((r) => r.test));
  console.log(
    `[f123] REPEAT x5 ms=${repeatMs} spread val16=${(s16 * 100).toFixed(1)}pt val180=${(s180 * 100).toFixed(1)}pt ` +
      `test=${(sTest * 100).toFixed(1)}pt runs=${JSON.stringify(runs)}`,
  );
  await expect(page.getByTestId('repeat-table')).toBeVisible();
  await expect(page.getByTestId('spread-val16')).toContainText('points');
  expect(await numTensors(page)).toBe(baseline);
  await shot(page, 'train-repeat');

  // ---- Reload: the head is restored from IndexedDB ----
  await page.reload();
  await waitForReady(page);
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toBeVisible({ timeout: 30_000 });
  expect(await metric(page, 'model', 'accuracy')).toBe(kermany.acc);
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-status')).toContainText('restored');

  // ---- Live mode parity: precomputed OFF, adult test within 2 points ----
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('switch', { name: /Use precomputed features/ }).uncheck();
  await page.keyboard.press('Escape');
  await openTab(page, 'Evaluate');
  await pickSet(page, 'adult_test');
  const live0 = Date.now();
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'adult_test', { timeout: 180_000 });
  const liveMs = Date.now() - live0;
  const liveAcc = await metric(page, 'model', 'accuracy');
  console.log(`[f123] LIVE adult acc=${liveAcc} (precomputed ${adult.acc}) in ${liveMs} ms`);
  expect(Math.abs(parseFloat(liveAcc) - parseFloat(adult.acc))).toBeLessThanOrEqual(2);

  expect(errors).toEqual([]);
});
