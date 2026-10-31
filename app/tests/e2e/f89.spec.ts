// End-to-end scenario for F8 (failure log), F9 (results card, snapshots, print), Reset session and the Steps checklist.
// Chromium only, one worker, precomputed features (no live 1,020-image extraction). Each test stays well under 2 minutes.
import { mkdirSync, readFileSync } from 'node:fs';
import { test, expect } from './fixtures';
import { DISCLAIMER, SCREENSHOT_DIR, collectPageErrors, waitForReady } from './helpers';
import type { Page } from '@playwright/test';

test.skip(({ browserName }) => browserName !== 'chromium', 'chromium only (fake camera / pdf / timing budgets)');
test.setTimeout(120_000);

const SET_LABELS = { kermany_test: 'Kermany test', adult_test: 'Adult test', stress: 'Kermany stress' } as const;
type SetId = keyof typeof SET_LABELS;
const SETS = Object.keys(SET_LABELS) as SetId[];

async function shot(page: Page, name: string) {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({ path: `${SCREENSHOT_DIR}/f89-${name}.png`, fullPage: true });
}

const numTensors = (page: Page) => page.evaluate(() => window.__vbw?.numTensors() ?? -1);
const openTab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true }).click();
const text = async (page: Page, id: string) => ((await page.getByTestId(id).textContent()) ?? '').trim();

async function trainHead(page: Page) {
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled({ timeout: 90_000 });
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });
}

async function start(page: Page) {
  await page.goto('/?debug=1');
  await waitForReady(page);
}

async function waitForHead(page: Page) {
  await expect.poll(() => page.evaluate(() => window.__vbw?.getState().head !== null), { timeout: 20_000 }).toBe(true);
}

/** Independent re-implementation of the half-credit score (mean over classes of (correct + 0.5 * referred) / n_class). */
function halfCredit(labels: number[], probs: number[], tau: number): number {
  const n = [0, 0];
  const got = [0, 0];
  labels.forEach((y, i) => {
    const p = probs[i] as number;
    n[y] = (n[y] as number) + 1;
    if (Math.max(p, 1 - p) < tau) got[y] = (got[y] as number) + 0.5;
    else if ((p >= 0.5 ? 1 : 0) === y) got[y] = (got[y] as number) + 1;
  });
  return ((got[0] as number) / (n[0] as number) + (got[1] as number) / (n[1] as number)) / 2;
}

interface StoredEval {
  labels: number[];
  probs: number[];
  halfCredit: number;
  usedPrecomputed: boolean;
}
const readEvals = (page: Page) =>
  page.evaluate(() => {
    const r = window.__vbw?.getState().evalResults ?? {};
    return Object.fromEntries(
      Object.entries(r).map(([k, v]) => [
        k,
        { labels: [...(v?.labels ?? [])], probs: Array.from(v?.probs ?? []), halfCredit: v?.metrics.halfCreditScore ?? Number.NaN, usedPrecomputed: v?.usedPrecomputed ?? true },
      ]),
    );
  }) as Promise<Record<SetId, StoredEval>>;

const vbwKeys = (page: Page) => page.evaluate(() => Object.keys(localStorage).filter((k) => k.startsWith('vbw:')));

async function evaluateAllAndWait(page: Page) {
  await openTab(page, 'Results');
  await page.getByTestId('results-eval-all').click();
  await expect(page.getByTestId('leaderboard-value')).toBeVisible({ timeout: 60_000 });
}

async function makeJpeg(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(() => {
    const c = document.createElement('canvas');
    c.width = 300;
    c.height = 200;
    const ctx = c.getContext('2d') as CanvasRenderingContext2D;
    const g = ctx.createLinearGradient(0, 0, 300, 200);
    g.addColorStop(0, '#202020');
    g.addColorStop(1, '#e0e0e0');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 300, 200);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(120, 70, 60, 60);
    return c.toDataURL('image/jpeg', 0.9).split(',')[1] as string;
  });
  return Buffer.from(b64, 'base64');
}

test('F8: log from Evaluate, Perturb and uploads; tags, edit, delete, filter, export, persistence, quota notice', async ({ page }) => {
  const errors = collectPageErrors(page);
  await start(page);
  await trainHead(page);

  // ---- 1. Evaluate row -> inline form (a tag is required) ----
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('prediction-list')).toBeVisible({ timeout: 30_000 });
  await page.getByTestId('pred-log').first().click();
  await expect(page.getByTestId('log-form')).toBeVisible();
  await expect(page.getByTestId('log-save')).toBeDisabled();
  const options = page.getByTestId('log-tag-select').locator('option');
  await expect(options).toHaveCount(8); // placeholder + 7 tags
  const optionTexts = await options.allTextContents();
  for (const tag of ['imbalance', 'shortcut', 'shift', 'quality', 'closed-world', 'evaluation', 'other']) {
    expect(optionTexts.some((t) => t.startsWith(`${tag}: `) && t.length > tag.length + 25), `tag ${tag} has a one-line explanation`).toBe(true);
  }
  await page.getByTestId('log-tag-select').selectOption('imbalance');
  await expect(page.getByTestId('log-tag-explain')).toContainText('One class is much more common');
  await page.getByTestId('log-hypothesis').fill('Three times more pneumonia in training, so it guesses pneumonia.');
  await page.getByTestId('log-save').click();
  await expect(page.getByTestId('log-form')).toHaveCount(0);

  const e1 = await page.evaluate(() => window.__vbw?.getState().failureLog[0]);
  expect(e1?.setName).toBe('kermany_test');
  expect(e1?.tag).toBe('imbalance');
  expect(e1?.hypothesis).toContain('Three times more pneumonia');
  expect(e1?.thumbnail.startsWith('data:image/jpeg;base64,')).toBe(true);
  expect(e1?.thumbnail.length).toBeLessThan(7000);
  expect(e1?.trueLabel === 'pneumonia' || e1?.trueLabel === 'normal').toBe(true);
  expect(e1?.predictedLabel).not.toBe(e1?.trueLabel); // "mistakes first" is on by default
  expect(e1?.referred).toBe(false);
  expect(e1?.config).toMatchObject({ mode: 'realistic', adultMix: false, augmentation: false, epochs: 20, seed: 42, tau: 0.5, usePrecomputed: true });
  expect(typeof e1?.confidence).toBe('number');
  expect(typeof e1?.probPneumonia).toBe('number');
  expect(typeof e1?.createdAt).toBe('number');
  const dims = await page.evaluate(
    (src) =>
      new Promise<number[]>((ok) => {
        const im = new Image();
        im.onload = () => ok([im.naturalWidth, im.naturalHeight]);
        im.src = src;
      }),
    e1?.thumbnail ?? '',
  );
  expect(dims).toEqual([96, 96]);

  // ---- 2. Perturb: log the CHANGED image ----
  await openTab(page, 'Break it');
  await page.getByTestId('perturb-size').selectOption('50');
  await page.getByTestId('perturb-brightness').fill('-0.4');
  await expect(page.getByTestId('perturb-after-pred')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('perturb-log')).toBeEnabled();
  await page.getByTestId('perturb-log').click();
  await page.getByTestId('log-tag-select').selectOption('quality');
  await page.getByTestId('log-hypothesis').fill('Darker pictures look different to the network.');
  await page.getByTestId('log-save').click();
  await expect(page.getByTestId('perturb-log-saved')).toBeVisible();
  const e2 = await page.evaluate(() => window.__vbw?.getState().failureLog[1]);
  expect(e2?.setName).toBe('perturb (kermany_test)');
  expect(e2?.tag).toBe('quality');
  expect(e2?.config).toMatchObject({ perturb: { brightness: -0.4, contrast: 1, rotationDeg: 0, cropFrac: 1 } });
  expect(e2?.thumbnail).not.toBe(e1?.thumbnail);
  await shot(page, 'perturb-log-form');

  // ---- 3. Uploads row ----
  await page.getByTestId('break-subtab-images').click();
  await page.getByTestId('upload-input').setInputFiles([{ name: 'wide.jpg', mimeType: 'image/jpeg', buffer: await makeJpeg(page) }]);
  await expect(page.getByTestId('upload-predictions').locator('.pred-badge')).toHaveCount(1, { timeout: 30_000 });
  await page.getByTestId('upload-log-0').click();
  await page.getByTestId('log-tag-select').selectOption('closed-world');
  await page.getByTestId('log-save').click();
  await expect(page.getByTestId('upload-log-0-saved')).toBeVisible();
  const e3 = await page.evaluate(() => window.__vbw?.getState().failureLog[2]);
  expect(e3).toMatchObject({ setName: 'uploads', tag: 'closed-world', trueLabel: null, imageId: 'wide.jpg' });

  // ---- 4. Failure log tab: counts, filter, edit, delete ----
  await openTab(page, 'Failure log');
  await expect(page.getByTestId('log-entry')).toHaveCount(3);
  await expect(page.getByTestId('log-count-all')).toHaveText('3');
  const expectedCounts: Record<string, string> = { imbalance: '1', shortcut: '0', shift: '0', quality: '1', 'closed-world': '1', evaluation: '0', other: '0' };
  for (const [tag, n] of Object.entries(expectedCounts)) await expect(page.getByTestId(`log-count-${tag}`)).toHaveText(n);
  await shot(page, 'failure-log');

  await page.getByTestId('log-filter-quality').click();
  await expect(page.getByTestId('log-entry')).toHaveCount(1);
  await expect(page.getByTestId('log-entry')).toHaveAttribute('data-tag', 'quality');
  await expect(page.getByTestId('log-entry-hypothesis')).toContainText('Darker pictures');
  await page.getByTestId('log-filter-quality').click(); // toggles back to all
  await expect(page.getByTestId('log-entry')).toHaveCount(3);

  // edit the Evaluate entry (newest first, so it is the last card)
  await page.getByTestId('log-entry').last().getByTestId('log-entry-edit').click();
  await page.getByTestId('log-tag-select').selectOption('evaluation');
  await page.getByTestId('log-edit-hypothesis').fill('Maybe the test set is simply too easy.');
  await page.getByTestId('log-edit-save').click();
  await expect(page.getByTestId('log-count-imbalance')).toHaveText('0');
  await expect(page.getByTestId('log-count-evaluation')).toHaveText('1');
  await expect(page.getByTestId('log-entry').last()).toContainText('Maybe the test set is simply too easy.');

  // ---- 5. Export JSON ----
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('log-export').click()]);
  expect(download.suggestedFilename()).toMatch(/^vision-failures-\d{8}-\d{4}\.json$/);
  const exported = JSON.parse(readFileSync((await download.path()) as string, 'utf8')) as Record<string, unknown>;
  expect(Object.keys(exported).sort()).toEqual(['appVersion', 'disclaimer', 'entries', 'exportedAt']);
  expect(exported['disclaimer']).toBe(DISCLAIMER);
  expect(exported['appVersion']).toMatch(/^\d+\.\d+\.\d+/);
  expect(new Date(exported['exportedAt'] as string).toString()).not.toBe('Invalid Date');
  const entries = exported['entries'] as Record<string, unknown>[];
  expect(entries).toHaveLength(3);
  for (const e of entries) {
    expect(Object.keys(e).sort()).toEqual(
      ['config', 'confidence', 'createdAt', 'hypothesis', 'id', 'imageId', 'predictedLabel', 'probPneumonia', 'referred', 'setName', 'tag', 'thumbnail', 'trueLabel'].sort(),
    );
  }
  expect(entries.map((e) => e['tag']).sort()).toEqual(['closed-world', 'evaluation', 'quality']);

  // ---- 6. delete, then persistence across a reload ----
  await page.getByTestId('log-entry').first().getByTestId('log-entry-delete').click(); // the newest = the upload
  await expect(page.getByTestId('log-entry')).toHaveCount(2);
  await expect(page.getByTestId('log-count-closed-world')).toHaveText('0');
  const stored = await page.evaluate(() => localStorage.getItem('vbw:v1:log') ?? '');
  expect(stored.length).toBeGreaterThan(1000);
  expect(stored.length).toBeLessThan(1_000_000);

  await start(page);
  await openTab(page, 'Failure log');
  await expect(page.getByTestId('log-entry')).toHaveCount(2);
  await expect(page.getByTestId('log-count-evaluation')).toHaveText('1');
  await expect(page.getByTestId('log-count-quality')).toHaveText('1');
  await expect(page.getByTestId('log-entry').first().locator('img')).toBeVisible();
  expect(await page.evaluate(() => window.__vbw?.getState().failureLog.map((e) => e.tag))).toEqual(['evaluation', 'quality']);

  // ---- 7. storage full: keep the entries in memory and say so ----
  await page.evaluate(() => {
    const orig = Storage.prototype.setItem;
    Storage.prototype.setItem = function (this: Storage, k: string, v: string) {
      if (k === 'vbw:v1:log') throw new DOMException('The quota has been exceeded.', 'QuotaExceededError');
      return orig.call(this, k, v);
    };
  });
  await page.getByTestId('log-entry').first().getByTestId('log-entry-delete').click();
  await expect(page.getByTestId('notice')).toContainText('failure log could not be saved');
  await expect(page.getByTestId('log-entry')).toHaveCount(1); // still works in memory
  expect(await page.evaluate(() => window.__vbw?.getState().failureLog.length)).toBe(1);

  expect(errors).toEqual([]);
});

test('F9: worst-case leaderboard value, per-set table, snapshots with deltas, persistence', async ({ page }) => {
  const errors = collectPageErrors(page);
  await start(page);
  await openTab(page, 'Results');
  await expect(page.getByTestId('results-empty')).toBeVisible(); // untrained: empty state
  await expect(page.getByTestId('results-disclaimer')).toHaveText(DISCLAIMER);

  await trainHead(page);
  await openTab(page, 'Results');
  await expect(page.getByTestId('leaderboard-incomplete')).toBeVisible();
  await expect(page.getByTestId('snapshot-save')).toBeDisabled();
  await expect(page.getByTestId('results-eval-all')).toBeEnabled();
  await page.getByTestId('results-eval-all').click();
  await expect(page.getByTestId('leaderboard-value')).toBeVisible({ timeout: 60_000 });

  // leaderboard value = min over the three half-credit scores, and it names that set
  const evals = await readEvals(page);
  const scores = SETS.map((id) => halfCredit(evals[id].labels, evals[id].probs, 0.5));
  for (const [i, id] of SETS.entries()) {
    expect(scores[i]).toBeCloseTo(evals[id].halfCredit, 10); // independent formula agrees with the app's metrics
    expect(await text(page, `res-${id}-model-score`)).toBe(`${((scores[i] as number) * 100).toFixed(1)}%`);
  }
  const min = Math.min(...scores);
  const worst = SETS[scores.indexOf(min)] as SetId;
  expect(await text(page, 'leaderboard-value')).toBe(`${(min * 100).toFixed(1)}%`);
  expect(await text(page, 'leaderboard-set')).toBe(SET_LABELS[worst]);
  expect(await text(page, 'leaderboard-tau')).toBe('50%');
  expect(await text(page, 'leaderboard-coverage')).toBe('100.0%');
  await expect(page.getByTestId(`res-${worst}`)).toHaveAttribute('data-worst', 'true');
  await expect(page.getByTestId('leaderboard-explain')).toContainText('Referring everything scores exactly 50');
  console.log(`[f89] DEFAULT realistic model: half-credit per set ${SETS.map((id, i) => `${id}=${((scores[i] as number) * 100).toFixed(1)}`).join(' ')}; worst-case ${(min * 100).toFixed(1)} (${worst})`);

  // table: baseline rows (Kermany 62.5 / adult 50 / stress 62.5 accuracy; 50% half credit everywhere), config, no stale marker
  expect(await text(page, 'res-kermany_test-baseline-accuracy')).toBe('62.5%');
  expect(await text(page, 'res-adult_test-baseline-accuracy')).toBe('50.0%');
  expect(await text(page, 'res-stress-baseline-accuracy')).toBe('62.5%');
  for (const id of SETS) {
    expect(await text(page, `res-${id}-baseline-score`)).toBe('50.0%');
    expect(await text(page, `res-${id}-baseline-sensitivity`)).toBe('100.0%');
    expect(await text(page, `res-${id}-baseline-specificity`)).toBe('0.0%');
    expect(await text(page, `res-${id}-model-referred`)).toBe('0 (0.0%)');
    expect(await text(page, `res-${id}-model-mode`)).toBe('precomputed');
  }
  expect(await text(page, 'res-kermany_test-model-balanced')).toBe(await text(page, 'res-kermany_test-model-score')); // tau 0.5: half credit = balanced accuracy
  await expect(page.getByTestId('results-config')).toContainText('realistic');
  expect(await text(page, 'cfg-epochs')).toBe('20');
  expect(await text(page, 'cfg-seed')).toBe('42');
  expect(await text(page, 'cfg-stamp')).toBe('no');
  await expect(page.getByTestId('results-recomputed')).toHaveCount(0);
  await expect(page.getByTestId('results-mode-note')).toHaveCount(0);

  // snapshot 1 (default name), then a real fix: adult X-rays mixed in + balanced classes, retrain
  await expect(page.getByTestId('snapshot-name')).toHaveAttribute('placeholder', 'Before fix');
  await page.getByTestId('snapshot-save').click();
  await expect(page.getByTestId('snap-name-0')).toHaveText('Before fix');
  await expect(page.getByTestId('snapshot-name')).toHaveAttribute('placeholder', 'After fix 1');
  await openTab(page, 'Data');
  await page.getByRole('radio', { name: /Balanced/ }).check();
  await page.getByRole('checkbox', { name: /Mix in adult X-rays/ }).check();
  await trainHead(page);

  // new head -> the old evaluations are gone (not shown as if they were current)
  await openTab(page, 'Results');
  await expect(page.getByTestId('leaderboard-incomplete')).toBeVisible();
  await expect(page.getByTestId('leaderboard-value')).toHaveCount(0);
  await expect(page.getByTestId(`res-${worst}-missing`)).toBeVisible();
  await expect(page.getByTestId('snapshot-save')).toBeDisabled();
  await expect(page.getByTestId('snap-name-0')).toHaveText('Before fix'); // the snapshot stays
  await page.getByTestId('results-eval-all').click();
  await expect(page.getByTestId('leaderboard-value')).toBeVisible({ timeout: 60_000 });
  expect(await text(page, 'cfg-mode')).toBe('balanced');
  expect(await text(page, 'cfg-adultMix')).toBe('yes');
  const evals2 = await readEvals(page);
  const scores2 = SETS.map((id) => halfCredit(evals2[id].labels, evals2[id].probs, 0.5));
  const min2 = Math.min(...scores2);
  expect(await text(page, 'leaderboard-value')).toBe(`${(min2 * 100).toFixed(1)}%`);
  expect(await text(page, 'leaderboard-set')).toBe(SET_LABELS[SETS[scores2.indexOf(min2)] as SetId]);
  console.log(`[f89] AFTER FIX (balanced + adult mix): half-credit per set ${SETS.map((id, i) => `${id}=${((scores2[i] as number) * 100).toFixed(1)}`).join(' ')}; worst-case ${(min2 * 100).toFixed(1)}`);

  await page.getByTestId('snapshot-save').click();
  await expect(page.getByTestId('snap-name-1')).toHaveText('After fix 1');
  await expect(page.getByTestId('snapshot-count')).toContainText('2 of 8');
  // deltas in points, column 2 vs column 1
  const lbDelta = await text(page, 'snap-1-leaderboard-delta');
  expect(lbDelta).toBe(`${(min2 - min) * 100 >= 0.05 ? '+' : (min2 - min) * 100 <= -0.05 ? '-' : ''}${Math.abs((min2 - min) * 100).toFixed(1)} pts`);
  for (const [i, id] of SETS.entries()) {
    const d = ((scores2[i] as number) - (scores[i] as number)) * 100;
    const shown = await text(page, `snap-1-${id}-halfCreditScore-delta`);
    expect(shown).toMatch(/^(\+|-)?\d+\.\d pts$/);
    expect(Math.abs(parseFloat(shown) - d)).toBeLessThanOrEqual(0.051);
  }
  await expect(page.getByTestId('snap-0-adult_test-accuracy-delta')).toHaveCount(0); // first column has no delta
  await expect(page.getByTestId('snap-1-kermany_test-sensitivity-delta')).toHaveText(/pts$/);
  await expect(page.getByTestId('snap-1-config')).toContainText('balanced, adult mix on');
  await expect(page.getByTestId('snap-0-config')).toContainText('realistic, adult mix off');
  await shot(page, 'results');

  // steps checklist
  await page.getByTestId('steps-summary').click();
  await expect(page.getByTestId('steps-day-1').locator('input[type=checkbox]')).toHaveCount(6);
  await expect(page.getByTestId('steps-day-2').locator('input[type=checkbox]')).toHaveCount(5);
  await page.getByTestId('step-d1-train').check();
  await page.getByTestId('step-d2-post').check();
  await expect(page.getByTestId('steps-progress')).toHaveText('(2 of 11 done)');
  await shot(page, 'steps');

  // ---- reload: snapshots, steps, model persist; evaluations are recomputed on demand ----
  await start(page);
  await waitForHead(page);
  expect(await page.evaluate(() => window.__vbw?.getState().snapshots.map((s) => s.name))).toEqual(['Before fix', 'After fix 1']);
  await openTab(page, 'Results');
  await expect(page.getByTestId('snap-name-0')).toHaveText('Before fix');
  await expect(page.getByTestId('snap-name-1')).toHaveText('After fix 1');
  await expect(page.getByTestId('snap-1-leaderboard')).toContainText('%');
  await expect(page.getByTestId('leaderboard-incomplete')).toBeVisible();
  await page.getByTestId('steps-summary').click();
  await expect(page.getByTestId('step-d1-train')).toBeChecked();
  await expect(page.getByTestId('step-d2-post')).toBeChecked();
  await expect(page.getByTestId('step-d1-data')).not.toBeChecked();
  await expect(page.getByTestId('steps-progress')).toHaveText('(2 of 11 done)');
  await page.getByTestId('results-eval-all').click();
  await expect(page.getByTestId('leaderboard-value')).toBeVisible({ timeout: 60_000 });
  expect(await text(page, 'leaderboard-value')).toBe(`${(min2 * 100).toFixed(1)}%`); // same head, same numbers after a reload

  // delete a snapshot
  await page.getByTestId('snap-delete-0').click();
  await expect(page.getByTestId('snap-col-0')).toContainText('After fix 1');
  await expect(page.getByTestId('snap-col-1')).toHaveCount(0);
  await expect(page.getByTestId('snapshot-name')).toHaveAttribute('placeholder', 'After fix 2'); // "After fix 1" is taken

  expect(errors).toEqual([]);
});

test('F9: tau recomputes the card, mode switch is flagged, results JSON, print CSS; Reset session returns to baseline', async ({ page }) => {
  const errors = collectPageErrors(page);
  await start(page);
  const baseline = await numTensors(page);
  expect(baseline).toBeGreaterThan(0);
  await trainHead(page);
  expect(await numTensors(page)).toBeGreaterThanOrEqual(baseline); // the head may hold a few tensors
  await evaluateAllAndWait(page);

  const evals = await readEvals(page);
  const at50 = SETS.map((id) => halfCredit(evals[id].labels, evals[id].probs, 0.5));
  const lb50 = await text(page, 'leaderboard-value');
  expect(lb50).toBe(`${(Math.min(...at50) * 100).toFixed(1)}%`);

  // ---- tau 0.9: the card is recomputed (and says so), never stale ----
  await openTab(page, 'Evaluate');
  await page.getByTestId('tau-slider').fill('0.9');
  await expect(page.getByTestId('tau-value')).toHaveText('90%');
  await openTab(page, 'Results');
  const at90 = SETS.map((id) => halfCredit(evals[id].labels, evals[id].probs, 0.9));
  const min90 = Math.min(...at90);
  await expect(page.getByTestId('results-card')).toHaveAttribute('data-tau', '0.9');
  await expect(page.getByTestId('results-card')).toHaveAttribute('data-recomputed', 'true');
  await expect(page.getByTestId('results-recomputed')).toContainText('Recomputed for the current threshold (tau = 90%)');
  expect(await text(page, 'leaderboard-tau')).toBe('90%');
  expect(await text(page, 'leaderboard-value')).toBe(`${(min90 * 100).toFixed(1)}%`);
  expect(await text(page, 'leaderboard-set')).toBe(SET_LABELS[SETS[at90.indexOf(min90)] as SetId]);
  for (const [i, id] of SETS.entries()) {
    expect(await text(page, `res-${id}-model-score`)).toBe(`${((at90[i] as number) * 100).toFixed(1)}%`);
    const referred = Number((await text(page, `res-${id}-model-referred`)).split(' ')[0]);
    expect(referred).toBeGreaterThan(0);
  }
  console.log(`[f89] tau 0.5 -> 0.9: leaderboard ${lb50} -> ${await text(page, 'leaderboard-value')} (${await text(page, 'leaderboard-set')}), coverage ${await text(page, 'leaderboard-coverage')}`);
  await shot(page, 'results-tau90');

  // ---- precomputed switch toggled mid-session: each evaluation says which mode made it ----
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('switch', { name: /Use precomputed features/ }).uncheck();
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('results-mode-note')).toContainText('evaluated with precomputed');
  await expect(page.getByTestId('results-mode-note')).toContainText('the switch is now live extraction');
  expect(await text(page, 'res-adult_test-model-mode')).toBe('precomputed');
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('switch', { name: /Use precomputed features/ }).check();
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('results-mode-note')).toHaveCount(0);

  // ---- failure counts + results JSON ----
  await openTab(page, 'Evaluate');
  await page.getByTestId('pred-log').first().click();
  await page.getByTestId('log-tag-select').selectOption('shift');
  await page.getByTestId('log-save').click();
  await openTab(page, 'Results');
  await expect(page.getByTestId('res-tag-shift')).toContainText('1');
  await expect(page.getByTestId('res-tag-imbalance')).toContainText('0');
  await page.getByTestId('snapshot-save').click();
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('results-download').click()]);
  expect(download.suggestedFilename()).toMatch(/^vision-results-\d{8}-\d{4}\.json$/);
  const out = JSON.parse(readFileSync((await download.path()) as string, 'utf8')) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  expect(Object.keys(out).sort()).toEqual(
    ['appVersion', 'config', 'disclaimer', 'exportedAt', 'explanation', 'failureCounts', 'leaderboard', 'perSet', 'snapshots', 'tau'].sort(),
  );
  expect(out['disclaimer']).toBe(DISCLAIMER);
  expect(out['config']).toMatchObject({ mode: 'realistic', adultMix: false, augmentation: false, epochs: 20, seed: 42, tau: 0.9, stampTrained: false });
  expect(out['tau']).toBe(0.9);
  expect(out['leaderboard'].score).toBeCloseTo(min90, 10);
  expect(out['leaderboard'].set).toBe(SETS[at90.indexOf(min90)]);
  expect(Object.keys(out['perSet'])).toEqual(SETS);
  expect(out['perSet'].adult_test.baseline.accuracy).toBe(0.5);
  expect(out['perSet'].kermany_test.scores.n).toBe(624);
  expect(out['snapshots']).toHaveLength(1);
  expect(out['snapshots'][0].config.tau).toBe(0.9);
  expect(out['failureCounts']).toMatchObject({ shift: 1, imbalance: 0 });
  expect(out['explanation']).toContain('Referring everything scores exactly 50');

  // ---- print CSS: tabs/controls hidden, title + disclaimer kept, black on white ----
  await page.emulateMedia({ media: 'print' });
  await expect(page.locator('.tabbar')).toBeHidden();
  await expect(page.getByTestId('results-controls')).toBeHidden();
  await expect(page.getByTestId('steps-panel')).toBeHidden();
  await expect(page.locator('footer')).toBeHidden();
  await expect(page.getByTestId('backend-badge')).toBeHidden();
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await expect(page.getByTestId('disclaimer-header')).toBeVisible();
  await expect(page.getByTestId('results-disclaimer')).toBeVisible();
  await expect(page.getByTestId('results-disclaimer')).toHaveText(DISCLAIMER);
  await expect(page.getByTestId('leaderboard-value')).toBeVisible();
  await expect(page.getByTestId('results-table')).toBeVisible();
  await expect(page.getByTestId('snapshot-table')).toBeVisible();
  const colors = await page.evaluate(() => ({
    header: getComputedStyle(document.querySelector('.app-header') as Element).backgroundColor,
    title: getComputedStyle(document.querySelector('.app-title') as Element).color,
    body: getComputedStyle(document.body).backgroundColor,
    value: getComputedStyle(document.querySelector('[data-testid=leaderboard-value]') as Element).color,
  }));
  expect(colors).toEqual({ header: 'rgb(255, 255, 255)', title: 'rgb(0, 0, 0)', body: 'rgb(255, 255, 255)', value: 'rgb(0, 0, 0)' });
  await shot(page, 'print');
  const pdf = await page.pdf({ preferCSSPageSize: true, printBackground: true });
  const pdfText = pdf.toString('latin1');
  const pages = (pdfText.match(/\/Type\s*\/Page[^s]/g) ?? []).length;
  console.log(`[f89] PRINT: pdf pages = ${pages} (${pdf.length} bytes)`);
  expect(pages).toBe(1);
  await page.emulateMedia({ media: 'screen' });
  await expect(page.locator('.tabbar')).toBeVisible();

  // ---- Reset session: head + stamped head disposed, tensors back to the post-load baseline, storage empty ----
  await page.getByTestId('steps-summary').click();
  await page.getByTestId('step-d1-train').check();
  expect(await vbwKeys(page)).toEqual(expect.arrayContaining(['vbw:v1:log', 'vbw:v1:snapshots', 'vbw:v1:steps', 'vbw:v1:tau']));
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  page.once('dialog', (d) => void d.accept());
  await page.getByTestId('reset-session').click();
  await expect.poll(() => page.evaluate(() => window.__vbw?.getState().head), { timeout: 10_000 }).toBeNull();
  await expect.poll(() => numTensors(page), { timeout: 10_000 }).toBe(baseline);
  await expect.poll(() => vbwKeys(page), { timeout: 10_000 }).toEqual([]);
  const idb = await page.evaluate(
    () =>
      new Promise<unknown>((ok) => {
        const req = indexedDB.open('vbw', 1);
        req.onsuccess = () => {
          const get = req.result.transaction('head', 'readonly').objectStore('head').get('current');
          get.onsuccess = () => ok(get.result ?? null);
          get.onerror = () => ok('error');
        };
        req.onerror = () => ok('error');
      }),
  );
  expect(idb).toBeNull();
  const s = await page.evaluate(() => {
    const st = window.__vbw?.getState();
    return st && { log: st.failureLog.length, snaps: st.snapshots.length, steps: st.stepsDone.length, uploads: st.uploads.length, tau: st.abstainTau, evals: Object.keys(st.evalResults).length, pre: st.settings.usePrecomputed, extractor: st.extractor !== null };
  });
  expect(s).toEqual({ log: 0, snaps: 0, steps: 0, uploads: 0, tau: 0.5, evals: 0, pre: true, extractor: true });
  expect(await page.evaluate(() => window.__vbw?.liveCameraTracks())).toBe(0);
  await openTab(page, 'Results');
  await expect(page.getByTestId('results-empty')).toBeVisible();
  await openTab(page, 'Failure log');
  await expect(page.getByTestId('log-empty')).toBeVisible();
  console.log(`[f89] RESET: tensors baseline=${baseline}, after reset=${await numTensors(page)}, vbw keys=${(await vbwKeys(page)).length}`);

  // the app still works after a reset: train again without reloading
  await trainHead(page);
  expect(await vbwKeys(page)).toEqual([]); // defaults are not written

  // ---- cancelling a later run keeps the previous model (state and IndexedDB untouched) ----
  const trainedAt = await page.evaluate(() => window.__vbw?.getState().headInfo?.trainedAt);
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  await page.getByTestId('epochs-input').fill('100');
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Training: epoch/, { timeout: 30_000 });
  await page.getByTestId('cancel-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Stopped after epoch \d+ of 100\. Training cancelled; your previous model is kept\./, { timeout: 15_000 });
  expect(await page.evaluate(() => window.__vbw?.getState().headInfo?.trainedAt)).toBe(trainedAt);
  expect(await page.evaluate(() => window.__vbw?.getState().headInfo?.completedEpochs)).toBe(20);

  // ---- Reset while a run is in flight: the finishing run must not bring a head back (state, IndexedDB, storage) ----
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Training: epoch/, { timeout: 30_000 });
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  page.once('dialog', (d) => void d.accept());
  await page.getByTestId('reset-session').click();
  await expect.poll(() => page.evaluate(() => window.__vbw?.getState().head), { timeout: 10_000 }).toBeNull();
  await page.waitForTimeout(2_000); // give a (wrongly) surviving run time to publish / save
  expect(await page.evaluate(() => window.__vbw?.getState().head)).toBeNull();
  expect(await page.evaluate(() => window.__vbw?.getState().train.status.state)).toBe('idle');
  expect(await page.evaluate(() => window.__vbw?.getState().train.history.length)).toBe(0);
  expect(await numTensors(page)).toBe(baseline);
  expect(await vbwKeys(page)).toEqual([]);
  const idbAfter = await page.evaluate(
    () =>
      new Promise<unknown>((ok) => {
        const req = indexedDB.open('vbw', 1);
        req.onsuccess = () => {
          const get = req.result.transaction('head', 'readonly').objectStore('head').get('current');
          get.onsuccess = () => ok(get.result ?? null);
          get.onerror = () => ok('error');
        };
        req.onerror = () => ok('error');
      }),
  );
  expect(idbAfter).toBeNull();
  expect(errors).toEqual([]);
});
