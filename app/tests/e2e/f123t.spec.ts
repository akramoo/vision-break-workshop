// Tester-written acceptance tests for F1 (Data), F2 (Train), F3 (Evaluate), derived from the acceptance criteria, run in
// Chromium AND Firefox. Heavy/slow scenarios (live extraction, strategy comparison) live in f123t-live.spec.ts.
import { test, expect } from './fixtures';
import { waitForReady } from './helpers';
import {
  dispatchAction,
  evaluate,
  numTensors,
  openTab,
  pct,
  readConfusion,
  readMetrics,
  setRecipe,
  shot,
  trainAndWait,
  trainHistory,
  watchConsole,
} from './f123t.helpers';

test.describe.configure({ mode: 'parallel' });

const MAX_TRAIN_MS = 20_000;

test('F1 Data: counts, class weights, toggles, lazy loading, gallery paging and filters', async ({ page }, info) => {
  const c = watchConsole(page);
  const requests: string[] = [];
  page.on('request', (r) => requests.push(new URL(r.url()).pathname));
  await page.goto('/?debug=1');
  await waitForReady(page);
  await expect(page.getByTestId('data-source-status')).toContainText('Using precomputed features');

  // ---- counts per set: normal / pneumonia / total ----
  const expected: Record<string, [number, number, number]> = {
    'kermany-train': [255, 765, 1020],
    'kermany-val': [45, 135, 180],
    'kermany-val_official': [8, 8, 16],
    'kermany-test': [234, 390, 624],
    'nih-train_adult': [100, 100, 200],
    'nih-test_adult': [100, 100, 200],
  };
  for (const [key, [n, p, t]] of Object.entries(expected)) {
    const row = page.getByTestId(`count-${key}`);
    await expect(row.getByTestId('count-normal'), key).toContainText(String(n));
    await expect(row.getByTestId('count-pneumonia'), key).toHaveText(String(p));
    await expect(row.getByTestId('count-total'), key).toHaveText(String(t));
  }
  // adult negatives are called "no finding" in the table
  await expect(page.getByTestId('count-nih-test_adult').getByTestId('count-normal')).toContainText('no finding');
  await expect(page.getByTestId('count-kermany-test').getByTestId('count-normal')).not.toContainText('no finding');

  // ---- class weights: realistic (1.00/1.00, balanced would use 2.00/0.67) ----
  const weights = page.getByTestId('class-weights');
  const summary = page.getByTestId('training-summary');
  await expect(weights).toContainText('normal 1.00 / pneumonia 1.00');
  await expect(weights).toContainText('normal 2.00 / pneumonia 0.67');
  await expect(summary).toContainText('1020 rows');
  await expect(summary).toContainText('255 normal and 765 pneumonia');

  await page.getByRole('radio', { name: /Balanced/ }).check();
  await expect(weights).toContainText('normal 2.00 / pneumonia 0.67');
  await expect(weights).not.toContainText('normal 1.00 / pneumonia 1.00');

  // adult mix: +200 rows, weights recomputed from 355 / 865
  await page.getByRole('checkbox', { name: /Mix in adult/ }).check();
  await expect(summary).toContainText('1220 rows');
  await expect(summary).toContainText('355 normal and 865 pneumonia');
  await expect(weights).toContainText('normal 1.72 / pneumonia 0.71');
  // augmentation on top: x3
  await page.getByRole('checkbox', { name: /Augmentation/ }).check();
  await expect(summary).toContainText('3660 rows');
  await expect(summary).toContainText('1065 normal and 2595 pneumonia');
  await expect(weights).toContainText('normal 1.72 / pneumonia 0.71'); // ratio unchanged by x3
  // augmentation alone
  await page.getByRole('checkbox', { name: /Mix in adult/ }).uncheck();
  await expect(summary).toContainText('3060 rows');
  await expect(weights).toContainText('normal 2.00 / pneumonia 0.67');
  // live-mode warning must NOT show while precomputed features are healthy
  await expect(page.getByTestId('live-estimate')).toHaveCount(0);
  await page.getByRole('checkbox', { name: /Augmentation/ }).uncheck();
  await expect(summary).toContainText('1020 rows');
  // adult mix alone, realistic: weights row says 1.00/1.00 and "balanced would use 1.72/0.71"
  await page.getByRole('radio', { name: /Realistic/ }).check();
  await page.getByRole('checkbox', { name: /Mix in adult/ }).check();
  await expect(weights).toContainText('normal 1.00 / pneumonia 1.00');
  await expect(weights).toContainText('normal 1.72 / pneumonia 0.71');
  await page.getByRole('checkbox', { name: /Mix in adult/ }).uncheck();

  // ---- gallery ----
  const grid = page.getByTestId('thumb-grid');
  await expect(grid.locator('img')).toHaveCount(48);
  await grid.scrollIntoViewIfNeeded();
  // wait until the lazy images near the viewport have really loaded instead of sleeping (off-screen lazy images are
  // deliberately never fetched: Firefox keeps them incomplete, Chromium loads them within its look-ahead distance)
  await expect
    .poll(() =>
      grid.locator('img').evaluateAll((imgs) => {
        const near = imgs.filter((i) => i.getBoundingClientRect().top < window.innerHeight) as HTMLImageElement[];
        return near.length > 0 && near.every((i) => i.complete);
      }),
    )
    .toBe(true);
  const firstPageJpgs = requests.filter((p) => p.endsWith('.jpg')).length;
  expect(firstPageJpgs, 'jpg requests on first load (lazy; page size 48)').toBeLessThanOrEqual(48);
  await expect(page.getByTestId('gallery-count')).toHaveText('2240 images');
  await expect(page.getByTestId('page-label')).toHaveText('Page 1 of 47');
  await expect(page.getByTestId('page-prev')).toBeDisabled();
  // the first thumbnails really decode (not broken links)
  await expect
    .poll(() => grid.locator('img').first().evaluate((i: HTMLImageElement) => i.complete && i.naturalWidth > 0))
    .toBe(true);
  for (const img of await grid.locator('img').all()) {
    expect(await img.getAttribute('alt')).toMatch(/Chest X-ray .*labelled/);
  }
  // official validation set: 16 images, one page, both pager buttons disabled
  await page.getByTestId('filter-split').selectOption('kermany/val_official');
  await expect(page.getByTestId('gallery-count')).toHaveText('16 images');
  await expect(grid.locator('img')).toHaveCount(16);
  await expect(page.getByTestId('page-label')).toHaveText('Page 1 of 1');
  await expect(page.getByTestId('page-next')).toBeDisabled();
  await expect(page.getByTestId('page-prev')).toBeDisabled();
  // Kermany test pneumonia: 390 images -> 9 pages; last page has 390 - 8*48 = 6 thumbs; prev goes back
  await page.getByTestId('filter-split').selectOption('kermany/test');
  await page.getByTestId('filter-label').selectOption('pneumonia');
  await expect(page.getByTestId('gallery-count')).toHaveText('390 images');
  await expect(page.getByTestId('page-label')).toHaveText('Page 1 of 9');
  for (let i = 0; i < 8; i++) await page.getByTestId('page-next').click();
  await expect(page.getByTestId('page-label')).toHaveText('Page 9 of 9');
  await expect(grid.locator('img')).toHaveCount(6);
  await expect(page.getByTestId('page-next')).toBeDisabled();
  await page.getByTestId('page-prev').click();
  await expect(page.getByTestId('page-label')).toHaveText('Page 8 of 9');
  await expect(grid.locator('img')).toHaveCount(48);
  // changing a filter resets to page 1
  await page.getByTestId('filter-label').selectOption('normal');
  await expect(page.getByTestId('page-label')).toHaveText('Page 1 of 5'); // 234 / 48
  await expect(page.getByTestId('gallery-count')).toHaveText('234 images');
  for (const chip of await grid.locator('.chip').all()) await expect(chip).toHaveText('normal');
  // adult negatives are "no finding"
  await page.getByTestId('filter-split').selectOption('nih/train_adult');
  await expect(page.getByTestId('gallery-count')).toHaveText('100 images');
  for (const chip of await grid.locator('.chip').all()) await expect(chip).toHaveText('no finding');
  await page.getByTestId('filter-label').selectOption('pneumonia');
  for (const chip of await grid.locator('.chip').first().all()) await expect(chip).toHaveText('pneumonia');
  await page.getByTestId('filter-split').selectOption('all');
  await page.getByTestId('filter-label').selectOption('all');
  await shot(page, info, 'data');

  // ---- F1: with precomputed ON only `base` (plus visible thumbnails) loads; aug / stamped / stress stay unloaded ----
  expect(requests.filter((p) => p.includes('/data/embeddings/')).sort()).toEqual([
    '/data/embeddings/base.f16.bin',
    '/data/embeddings/base.index.json',
  ]);
  expect(c.errors).toEqual([]);
  expect(c.warnings).toEqual([]);
});

test('F1 Data: a failing embeddings file shows a banner, live-mode estimate, and "Try again" retries', async ({ page }, info) => {
  const c = watchConsole(page);
  let fail = true;
  await page.route('**/data/embeddings/base.f16.bin', (r) => (fail ? r.abort() : r.continue()));
  await page.goto('/');
  await waitForReady(page);
  await expect(page.getByTestId('embeddings-banner')).toBeVisible();
  await expect(page.getByTestId('embeddings-banner')).toContainText('could not be loaded');
  await expect(page.getByTestId('data-source-status')).toContainText('Live extraction');
  await expect(page.getByTestId('live-estimate')).toContainText('MobileNet');
  await shot(page, info, 'data-fallback');
  // the rest of the app is still usable
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled();
  await openTab(page, 'Data');
  // heal the network, press Try again
  fail = false;
  await page.getByRole('button', { name: 'Try again' }).click();
  await expect(page.getByTestId('data-source-status')).toContainText('Using precomputed features');
  await expect(page.getByTestId('embeddings-banner')).toHaveCount(0);
  await expect(page.getByTestId('live-estimate')).toHaveCount(0);
  // aborted request produces a browser console error for the failed resource; that is expected here, nothing else is
  expect(c.errors.filter((e) => !/base\.f16\.bin|net::ERR|NetworkError|Failed to load resource/i.test(e))).toEqual([]);
});

test('F2 Train: advanced is collapsed, 20 epochs finish fast with a growing chart, cancel, same seed = same result', async ({ page }, info) => {
  test.setTimeout(180_000);
  const c = watchConsole(page);
  await page.goto('/?debug=1');
  await waitForReady(page);
  await openTab(page, 'Train');

  // Advanced is collapsed by default and contains epochs / learning rate / seed
  const adv = page.getByTestId('advanced');
  await expect(adv).toBeVisible();
  expect(await adv.evaluate((d: HTMLDetailsElement) => d.open)).toBe(false);
  await expect(page.getByTestId('epochs-input')).toBeHidden();
  await expect(page.getByTestId('seed-input')).toBeHidden();
  await page.getByText('Advanced settings').click();
  await expect(page.getByLabel(/Epochs/)).toHaveValue('20');
  await expect(page.getByLabel(/Learning rate/)).toBeVisible();
  await expect(page.getByTestId('lr-value')).toHaveText('1.0e-3');
  await expect(page.getByLabel(/Seed/)).toHaveValue('42');
  await expect(page.getByTestId('new-seed')).toBeVisible();

  // initial state
  await expect(page.getByTestId('train-status')).toHaveText('Not trained yet. Click Train.');
  await expect(page.getByTestId('cancel-button')).toBeDisabled();

  const baseline = await numTensors(page);
  const ms = await trainAndWait(page);
  const hist = await trainHistory(page);
  const last = hist[hist.length - 1]!;
  console.log(`[f123t] ${info.project.name} TRAIN 20 epochs wall=${ms}ms valAcc=${last.valAcc.toFixed(4)} trainAcc=${last.trainAcc.toFixed(4)} valLoss=${last.valLoss.toFixed(4)}`);
  expect(ms).toBeLessThan(MAX_TRAIN_MS);
  expect(hist.map((h) => h.epoch)).toEqual(Array.from({ length: 20 }, (_, i) => i + 1));
  expect(last.valAcc).toBeGreaterThan(0.85);
  expect(hist[19]!.trainLoss).toBeLessThan(hist[0]!.trainLoss);
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/);
  await expect(page.getByTestId('train-status')).toContainText(/Validation accuracy \d+\.\d%/);
  expect(await numTensors(page)).toBe(baseline);

  // the chart got one point per epoch for both series
  for (const id of ['chart-loss', 'chart-accuracy']) {
    const chart = page.getByTestId(id);
    await expect(chart.locator('svg[role=img] desc')).toContainText('after epoch 20');
    const paths = await chart.locator('svg[role=img] path').evaluateAll((ps) => ps.map((p) => p.getAttribute('d') ?? ''));
    expect(paths.length).toBe(2);
    for (const d of paths) {
      expect((d.match(/[ML]/g) ?? []).length, `${id} path points`).toBe(20);
      expect(d).not.toMatch(/NaN|Infinity/);
    }
  }
  await shot(page, info, 'train');

  // ---- same seed twice -> identical numbers; a different seed -> different ----
  await trainAndWait(page);
  const again = await trainHistory(page);
  expect(again[19]!.valAcc).toBe(last.valAcc);
  expect(again[19]!.trainLoss).toBeCloseTo(last.trainLoss, 5);
  expect(again[19]!.valLoss).toBeCloseTo(last.valLoss, 5);
  await page.getByTestId('seed-input').fill('7');
  await trainAndWait(page);
  const other = await trainHistory(page);
  expect(other[19]!.trainLoss).not.toBeCloseTo(last.trainLoss, 6);
  // "New seed" button changes the seed
  await page.getByTestId('new-seed').click();
  expect(await page.getByTestId('seed-input').inputValue()).not.toBe('7');
  await page.getByTestId('seed-input').fill('42');

  // ---- Cancel: stops within one epoch, keeps the model trained so far ----
  await page.getByTestId('epochs-input').fill('100');
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Training: epoch \d+ of 100/, { timeout: 30_000 });
  await expect(page.getByTestId('cancel-button')).toBeEnabled();
  await expect(page.getByTestId('train-button')).toBeDisabled();
  const lenAtClick = await page.evaluate(() => window.__vbw!.getState().train.history.length);
  await page.getByTestId('cancel-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Stopped after epoch \d+ of 100/, { timeout: 10_000 });
  const stopped = await trainHistory(page);
  console.log(`[f123t] ${info.project.name} CANCEL clicked at ~epoch ${lenAtClick}, stopped with ${stopped.length} epochs`);
  expect(stopped.length).toBeGreaterThanOrEqual(1);
  expect(stopped.length).toBeLessThan(100);
  expect(stopped.length - lenAtClick).toBeLessThanOrEqual(3); // click -> a bounded number of epochs, not "runs to the end"
  await expect(page.getByTestId('cancel-button')).toBeDisabled();
  await expect(page.getByTestId('train-button')).toBeEnabled();
  // wait a moment: no further epochs arrive after "Stopped"
  await page.waitForTimeout(1500);
  expect((await trainHistory(page)).length).toBe(stopped.length);
  // the partly trained model is usable
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toBeVisible({ timeout: 30_000 });
  expect(await numTensors(page)).toBe(baseline);
  expect(c.errors).toEqual([]);
  expect(c.warnings).toEqual([]);
});

test('F3 Evaluate: empty state, exact 62.5/100/0 baseline, confusion sums, adult "no finding", stress', async ({ page }, info) => {
  test.setTimeout(180_000);
  const c = watchConsole(page);
  await page.goto('/?debug=1');
  await waitForReady(page);

  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('evaluate-empty')).toBeVisible();
  await expect(page.getByTestId('evaluate-empty')).toContainText('No trained model yet');
  await expect(page.getByTestId('eval-set-picker')).toHaveCount(0);
  await expect(page.locator('body')).not.toContainText('NaN');
  await shot(page, info, 'evaluate-empty');
  // the call to action leads to Train
  await page.getByRole('button', { name: 'Go to Train' }).click();
  await expect(page.getByRole('tab', { name: 'Train', exact: true })).toHaveAttribute('aria-selected', 'true');

  await trainAndWait(page);

  // Kermany test
  const k = await evaluate(page, 'kermany_test');
  const base = await readMetrics(page, 'baseline');
  expect(await page.getByTestId('baseline-accuracy').textContent()).toBe('62.5%');
  expect(await page.getByTestId('baseline-sensitivity').textContent()).toBe('100.0%');
  expect(await page.getByTestId('baseline-specificity').textContent()).toBe('0.0%');
  expect(base.bal).toBe(50);
  const cmK = await readConfusion(page);
  expect(cmK.tn + cmK.fp + cmK.fn + cmK.tp).toBe(624);
  expect(cmK.tn + cmK.fp).toBe(234);
  expect(cmK.fn + cmK.tp).toBe(390);
  // metrics follow from the confusion matrix by first principles
  const r1 = (x: number) => Math.round(x * 1000) / 10;
  expect(k.acc).toBeCloseTo(r1((cmK.tp + cmK.tn) / 624), 5);
  expect(k.sens).toBeCloseTo(r1(cmK.tp / 390), 5);
  expect(k.spec).toBeCloseTo(r1(cmK.tn / 234), 5);
  expect(k.bal).toBeCloseTo((r1(cmK.tp / 390) + r1(cmK.tn / 234)) / 2, 0);
  await expect(page.getByTestId('confusion-matrix')).toContainText('Confusion matrix: 624 X-rays');
  await expect(page.getByTestId('confusion-matrix')).not.toContainText('no finding');
  await expect(page.getByTestId('metrics-table')).not.toContainText('Referred'); // tau = 0.5: no abstain columns
  await expect(page.getByTestId('prediction-list').locator('tbody tr')).toHaveCount(40);
  await expect(page.locator('body')).not.toContainText('NaN');
  await shot(page, info, 'evaluate-kermany');

  // Adult test
  const a = await evaluate(page, 'adult_test');
  expect(await page.getByTestId('baseline-accuracy').textContent()).toBe('50.0%');
  const cmA = await readConfusion(page);
  expect(cmA.tn + cmA.fp + cmA.fn + cmA.tp).toBe(200);
  expect(cmA.tn + cmA.fp).toBe(100);
  await expect(page.getByTestId('confusion-matrix')).toContainText('no finding');
  await expect(page.getByTestId('prediction-list')).toContainText(/no finding|pneumonia/);
  // wording: adult negatives are never called "normal" in the matrix / list
  const adultTexts = await page.getByTestId('confusion-matrix').innerText();
  expect(adultTexts).not.toMatch(/\bnormal\b/i);
  const listText = await page.getByTestId('prediction-list').locator('tbody').innerText();
  expect(listText).not.toMatch(/\bnormal\b/);
  await shot(page, info, 'evaluate-adult');

  // Stress
  const s = await evaluate(page, 'stress');
  const cmS = await readConfusion(page);
  expect(cmS.tn + cmS.fp + cmS.fn + cmS.tp).toBe(624);
  expect(await page.getByTestId('baseline-accuracy').textContent()).toBe('62.5%');
  await shot(page, info, 'evaluate-stress');
  console.log(`[f123t] ${info.project.name} EVAL default recipe: kermany=${JSON.stringify(k)} adult=${JSON.stringify(a)} stress=${JSON.stringify(s)}`);

  // Re-selecting an evaluated set is instant (cached) and unchanged
  const k2 = await evaluate(page, 'kermany_test');
  expect(k2).toEqual(k);
  expect(c.errors).toEqual([]);
  expect(c.warnings).toEqual([]);
});

test('F2/F3 memory: tensor count and bytes return to baseline after train + 2x evaluate + retrain + cancelled run', async ({ page }) => {
  test.setTimeout(240_000);
  const c = watchConsole(page);
  await page.goto('/?debug=1');
  await waitForReady(page);
  const mem = () => page.evaluate(() => ({ t: window.__vbw!.tf.memory().numTensors, b: window.__vbw!.tf.memory().numBytes }));
  await openTab(page, 'Train');
  const base = await mem();

  await trainAndWait(page);
  expect(await mem()).toEqual(base);
  for (const id of ['kermany_test', 'adult_test', 'stress', 'kermany_test', 'adult_test', 'stress'] as const) {
    await evaluate(page, id);
    expect(await mem(), `after evaluating ${id}`).toEqual(base);
  }
  await trainAndWait(page); // retrain: old head replaced, results invalidated
  expect(await mem()).toEqual(base);
  await evaluate(page, 'kermany_test');
  expect(await mem()).toEqual(base);

  // cancelled run
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  await page.getByTestId('epochs-input').fill('100');
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Training: epoch/, { timeout: 30_000 });
  await page.getByTestId('cancel-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Stopped after/, { timeout: 10_000 });
  expect(await mem()).toEqual(base);
  // cancel immediately while the features are still being prepared
  await page.getByTestId('train-button').click();
  await page.getByTestId('cancel-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Stopped after|cancelled|Finished/, { timeout: 15_000 });
  await expect(page.getByTestId('train-button')).toBeEnabled();
  expect(await mem()).toEqual(base);

  // balanced + adult + augmentation training also leaves nothing behind
  await setRecipe(page, { mode: 'balanced', adultMix: true, augmentation: true });
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click(); // the <details> is closed again after a tab switch
  await page.getByTestId('epochs-input').fill('5');
  await trainAndWait(page);
  await evaluate(page, 'stress');
  expect(await mem()).toEqual(base);
  expect(c.errors).toEqual([]);
  expect(c.warnings).toEqual([]);
});

test('F2 Repeat x5: five seeds, val16 / val180 / test per run, spread shown, no leak', async ({ page }, info) => {
  test.setTimeout(240_000);
  const c = watchConsole(page);
  await page.goto('/?debug=1');
  await waitForReady(page);
  await openTab(page, 'Train');
  const base = await numTensors(page);
  const t0 = Date.now();
  await page.getByTestId('repeat-button').click();
  await expect(page.getByTestId('repeat-cancel')).toBeEnabled();
  await expect(page.getByTestId('repeat-status')).toContainText('Finished 5 runs', { timeout: 150_000 });
  const ms = Date.now() - t0;
  const runs = await page.evaluate(() => window.__vbw!.getState().repeat.runs);
  expect(runs.map((r) => r.seed)).toEqual([1, 2, 3, 4, 5]);
  for (const r of runs) for (const v of [r.val16, r.val180, r.test]) expect(v).not.toBeNull();
  // val16 only takes multiples of 1/16
  for (const r of runs) expect(Math.round((r.val16 as number) * 16)).toBeCloseTo((r.val16 as number) * 16, 6);
  const spread = (xs: (number | null)[]) => Math.max(...(xs as number[])) - Math.min(...(xs as number[]));
  console.log(
    `[f123t] ${info.project.name} REPEAT x5 ${ms}ms spread val16=${(spread(runs.map((r) => r.val16)) * 100).toFixed(1)}pt ` +
      `val180=${(spread(runs.map((r) => r.val180)) * 100).toFixed(1)}pt test=${(spread(runs.map((r) => r.test)) * 100).toFixed(1)}pt`,
  );
  await expect(page.getByTestId('repeat-table').locator('tbody tr')).toHaveCount(5);
  await expect(page.getByTestId('spread-val16')).toContainText('points');
  expect(await numTensors(page)).toBe(base);
  // Repeat does not replace the user's head
  expect(await page.evaluate(() => window.__vbw!.getState().head)).toBeNull();
  await shot(page, info, 'train-repeat');
  expect(c.errors).toEqual([]);
  expect(c.warnings).toEqual([]);
});

test('persistence: reload restores the trained head; Reset session clears it (state, IndexedDB, localStorage)', async ({ page }) => {
  test.setTimeout(180_000);
  const c = watchConsole(page);
  page.on('dialog', (d) => void d.accept());
  const idbHead = () =>
    page.evaluate(
      () =>
        new Promise<'none' | 'present' | 'error'>((resolve) => {
          const req = indexedDB.open('vbw', 1);
          req.onerror = () => resolve('error');
          req.onsuccess = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains('head')) {
              db.close();
              resolve('none');
              return;
            }
            const g = db.transaction('head', 'readonly').objectStore('head').get('current');
            g.onsuccess = () => {
              db.close();
              resolve(g.result ? 'present' : 'none');
            };
            g.onerror = () => {
              db.close();
              resolve('error');
            };
          };
        }),
    );

  await page.goto('/?debug=1');
  await waitForReady(page);
  await trainAndWait(page);
  const acc = (await evaluate(page, 'kermany_test')).acc;
  await expect.poll(idbHead).toBe('present');

  // ---- reload: head restored without retraining ----
  await page.reload();
  await waitForReady(page);
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-status')).toContainText('restored');
  const info = await page.evaluate(() => window.__vbw!.getState().headInfo);
  expect(info?.restored).toBe(true);
  expect(info?.seed).toBe(42);
  expect((await evaluate(page, 'kermany_test')).acc).toBe(acc);
  // and the restored loss curves are shown on the Train tab
  await openTab(page, 'Train');
  await expect(page.getByTestId('chart-loss').locator('svg[role=img] desc')).toContainText('after epoch 20');

  // ---- Reset session ----
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('button', { name: 'Reset session' }).click();
  await expect.poll(() => page.evaluate(() => window.__vbw!.getState().head)).toBeNull();
  await expect.poll(idbHead, { message: 'IndexedDB head record after Reset session' }).toBe('none');
  const ls = await page.evaluate(() => Object.keys(localStorage));
  console.log(`[f123t] localStorage keys after Reset: ${JSON.stringify(ls)}`);
  await page.keyboard.press('Escape');
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('evaluate-empty')).toBeVisible();
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-status')).toHaveText('Not trained yet. Click Train.');
  await expect(page.getByTestId('chart-loss').locator('svg[role=img] desc')).toContainText('no data yet');

  // a reload after Reset must NOT bring the old head back
  await page.reload();
  await waitForReady(page);
  await page.waitForTimeout(1000);
  expect(await page.evaluate(() => window.__vbw!.getState().head)).toBeNull();
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('evaluate-empty')).toBeVisible();

  // after Reset, training still works and tensors are at baseline
  const base = await numTensors(page);
  await trainAndWait(page);
  expect(await numTensors(page)).toBe(base);
  expect(c.errors).toEqual([]);
});

test('settings survive a reload, and Reset restores defaults (precomputed switch, recipe, seed)', async ({ page }) => {
  page.on('dialog', (d) => void d.accept());
  await page.goto('/?debug=1');
  await waitForReady(page);
  await setRecipe(page, { mode: 'balanced', adultMix: true, augmentation: false });
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  await page.getByTestId('seed-input').fill('123');
  await page.reload();
  await waitForReady(page);
  await expect(page.getByRole('radio', { name: /Balanced/ })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: /Mix in adult/ })).toBeChecked();
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  await expect(page.getByTestId('seed-input')).toHaveValue('123');
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('button', { name: 'Reset session' }).click();
  await page.keyboard.press('Escape');
  await openTab(page, 'Data');
  await expect(page.getByRole('radio', { name: /Realistic/ })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: /Mix in adult/ })).not.toBeChecked();
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  await expect(page.getByTestId('seed-input')).toHaveValue('42');
});

test('F7-path in Evaluate: tau > 0.5 adds Referred / Coverage, matches an independent count, never decreases', async ({ page }, info) => {
  test.setTimeout(180_000);
  const c = watchConsole(page);
  await page.goto('/?debug=1');
  await waitForReady(page);
  await trainAndWait(page);
  await evaluate(page, 'kermany_test');
  const nAt = async (id: string) => {
    const t = (await page.getByTestId(id).textContent()) ?? '';
    return t;
  };
  await expect(page.getByTestId('model-referred')).toHaveCount(0);

  // independent referral count computed in the page from the raw probabilities
  const expectedReferred = (tau: number) =>
    page.evaluate((t) => {
      const r = window.__vbw!.getState().evalResults['kermany_test']!;
      let n = 0;
      for (const p of r.probs) if (Math.max(p, 1 - p) < t) n++;
      return n;
    }, tau);

  let prevRef = 0;
  for (const tau of [0.6, 0.8, 0.95, 0.99]) {
    expect(await dispatchAction(page, { type: 'setTau', tau })).toBe(true);
    await expect(page.getByTestId('model-referred')).toBeVisible();
    await expect(page.getByTestId('abstain-note')).toBeVisible();
    const refTxt = await nAt('model-referred'); // "N (x.x%)"
    const m = /^(\d+) \((\d+\.\d)%\)$/.exec(refTxt.trim());
    expect(m, `referred text "${refTxt}"`).not.toBeNull();
    const nRef = Number(m![1]);
    expect(nRef).toBe(await expectedReferred(tau));
    expect(nRef).toBeGreaterThanOrEqual(prevRef);
    prevRef = nRef;
    expect(Number(m![2])).toBeCloseTo((nRef / 624) * 100, 1);
    const cov = pct(await nAt('model-coverage'));
    expect(cov).toBeCloseTo(100 - (nRef / 624) * 100, 1);
    const cm = await readConfusion(page);
    expect(cm.tn + cm.fp + cm.fn + cm.tp + nRef).toBe(624);
    expect(await nAt('baseline-referred')).toBe('-');
    if (nRef > 0) await expect(page.getByTestId('confusion-matrix')).toContainText(`${nRef} referred, not counted`);
    await expect(page.locator('body')).not.toContainText('NaN');
  }
  // tau 0.8: some X-rays are referred in the prediction list
  await dispatchAction(page, { type: 'setTau', tau: 0.8 });
  await shot(page, info, 'evaluate-tau-0.8');
  // UX finding: with "Show mistakes first" ON, the first 40 rows are all mistakes, so referred cases are not visible at tau 0.8.
  const hiddenAtDefault = !(await page.getByTestId('prediction-list').innerText()).includes('Refer to a doctor');
  console.log(`[f123t] ${info.project.name} referred rows visible in default (mistakes-first) list at tau 0.8: ${!hiddenAtDefault}`);
  await page.getByTestId('pred-mode-dataset').check(); // "Dataset order" (replaced the old "Show mistakes first" checkbox)
  await dispatchAction(page, { type: 'setTau', tau: 0.99 });
  await expect(page.getByTestId('prediction-list')).toContainText('Refer to a doctor');
  await dispatchAction(page, { type: 'setTau', tau: 0.8 });
  // out-of-range values clamp to [0.5, 0.99]; 0.5 hides the columns again
  await dispatchAction(page, { type: 'setTau', tau: 7 });
  expect(await page.evaluate(() => window.__vbw!.getState().abstainTau)).toBe(0.99);
  await dispatchAction(page, { type: 'setTau', tau: 0.1 });
  expect(await page.evaluate(() => window.__vbw!.getState().abstainTau)).toBe(0.5);
  await expect(page.getByTestId('model-referred')).toHaveCount(0);
  // switching set at tau 0.9 also shows the columns
  await dispatchAction(page, { type: 'setTau', tau: 0.9 });
  await evaluate(page, 'adult_test');
  await expect(page.getByTestId('model-referred')).toBeVisible();
  const cm = await readConfusion(page);
  const refA = Number(/^(\d+)/.exec((await nAt('model-referred')).trim())![1]);
  expect(cm.tn + cm.fp + cm.fn + cm.tp + refA).toBe(200);
  expect(c.errors).toEqual([]);
});

test('accessibility: chart title + desc, labelled inputs, aria-live status, readable text size, keyboard reachable', async ({ page }, info) => {
  test.setTimeout(180_000);
  const c = watchConsole(page);
  await page.goto('/?debug=1');
  await waitForReady(page);

  const unlabeled = () =>
    page.evaluate(() => {
      const bad: string[] = [];
      const visible = (el: Element) => {
        const r = (el as HTMLElement).getBoundingClientRect();
        const s = getComputedStyle(el);
        return r.width > 0 && r.height > 0 && s.visibility !== 'hidden';
      };
      for (const el of Array.from(document.querySelectorAll('input, select, textarea'))) {
        if (!visible(el)) continue;
        const e = el as HTMLInputElement;
        const labelledBy = e.getAttribute('aria-labelledby');
        const byIds = labelledBy ? labelledBy.split(/\s+/).map((i) => document.getElementById(i)?.textContent ?? '').join('').trim() : '';
        const name = (e.getAttribute('aria-label') ?? '').trim() || byIds || Array.from(e.labels ?? []).map((l) => l.textContent ?? '').join('').trim();
        if (!name) bad.push(`${e.tagName.toLowerCase()}[type=${e.type}] ${e.outerHTML.slice(0, 100)}`);
      }
      for (const b of Array.from(document.querySelectorAll('button'))) {
        if (!visible(b)) continue;
        if (!((b.textContent ?? '').trim() || b.getAttribute('aria-label'))) bad.push(`button without name ${b.outerHTML.slice(0, 80)}`);
      }
      for (const i of Array.from(document.querySelectorAll('img'))) {
        if (i.getAttribute('alt') === null) bad.push(`img without alt ${i.src}`);
      }
      return bad;
    });

  const smallText = () =>
    page.evaluate(() => {
      const out: { size: number; text: string }[] = [];
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      let node: Node | null;
      while ((node = walker.nextNode())) {
        const t = (node.textContent ?? '').trim();
        if (!t) continue;
        const el = node.parentElement;
        if (!el) continue;
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        const size = parseFloat(getComputedStyle(el).fontSize);
        // SVG text is scaled by the viewBox: measure rendered height instead
        const rendered = el instanceof SVGElement ? r.height : size;
        out.push({ size: rendered, text: t.slice(0, 40) });
      }
      return out.filter((o) => o.size < 11).slice(0, 12);
    });

  // Data tab
  expect(await unlabeled(), 'Data tab').toEqual([]);
  // Train tab (advanced open)
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  expect(await unlabeled(), 'Train tab').toEqual([]);
  const status = page.getByTestId('train-status');
  expect(await status.getAttribute('role')).toBe('status');
  expect(await status.getAttribute('aria-live')).toBe('polite');
  expect(await page.getByTestId('repeat-status').getAttribute('aria-live')).toBe('polite');
  await trainAndWait(page);
  for (const id of ['chart-loss', 'chart-accuracy']) {
    const svg = page.getByTestId(id).locator('svg[role=img]');
    const ids = ((await svg.getAttribute('aria-labelledby')) ?? '').split(/\s+/);
    expect(ids).toHaveLength(2);
    const [title, desc] = await Promise.all(
      ids.map((i) => page.evaluate((x) => document.getElementById(x)?.textContent ?? '', i)),
    );
    expect(title!.length, `${id} title`).toBeGreaterThan(3);
    expect(desc!, `${id} desc`).toMatch(/Training: .* after epoch 20; Validation: .* after epoch 20/);
  }
  console.log(`[f123t] ${info.project.name} chart desc: ${await page.getByTestId('chart-accuracy').locator('desc').textContent()}`);
  // Evaluate tab with results
  await evaluate(page, 'kermany_test');
  expect(await unlabeled(), 'Evaluate tab').toEqual([]);
  await expect(page.getByTestId('eval-set-picker')).toContainText('Test on');
  // Settings dialog
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  expect(await unlabeled(), 'Settings drawer').toEqual([]);
  await expect(page.getByRole('dialog')).toHaveAttribute('aria-modal', 'true');
  await page.keyboard.press('Escape');

  // text legibility (projector): nothing below 11px; report the smallest
  const small = await smallText();
  console.log(`[f123t] ${info.project.name} text below 11px: ${JSON.stringify(small)}`);
  expect(small).toEqual([]);

  // keyboard: radio set can be reached and changed with the keyboard alone
  await page.getByTestId('eval-set-kermany_test').focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'adult_test', { timeout: 30_000 });
  await shot(page, info, 'a11y-evaluate');
  expect(c.errors).toEqual([]);
  expect(c.warnings).toEqual([]);
});
