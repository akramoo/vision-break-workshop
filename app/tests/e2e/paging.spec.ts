// Evaluate tab: the prediction list can be paged and filtered, so every image of the evaluated set can be browsed.
// Chromium only, one worker, precomputed features.
import { mkdirSync } from 'node:fs';
import { test, expect } from './fixtures';
import { SCREENSHOT_DIR, collectPageErrors, waitForReady } from './helpers';
import { dispatchAction } from './f123t.helpers';
import type { Page } from '@playwright/test';

test.setTimeout(120_000);

const openTab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true }).click();

const label = (page: Page) => page.getByTestId('pred-label');
const rows = (page: Page) => page.getByTestId('pred-row');

async function cm(page: Page) {
  const v = await Promise.all(['tn', 'fp', 'fn', 'tp'].map((k) => page.getByTestId(`cm-${k}`).locator('.cm__count').textContent()));
  const [tn, fp, fn, tp] = v.map(Number) as [number, number, number, number];
  return { tn, fp, fn, tp };
}

/** The DOM must never hold more than one page of rows, and every thumbnail must be lazy. */
async function expectAtMost(page: Page, size: number) {
  const n = await rows(page).count();
  expect(n).toBeLessThanOrEqual(size);
  expect(await page.getByTestId('prediction-list').locator('tbody > tr[data-testid="pred-row"]').count()).toBe(n);
  const lazy = await page.getByTestId('prediction-list').locator('img[loading="lazy"]').count();
  const imgs = await page.getByTestId('prediction-list').locator('img').count();
  expect(lazy).toBe(imgs);
  return n;
}

test('Evaluate: page through all 624 predictions, filter mistakes and classes, change the page size', async ({ page }) => {
  const errors = collectPageErrors(page);
  await page.goto('/?debug=1');
  await waitForReady(page);
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled({ timeout: 60_000 });
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });

  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'kermany_test', { timeout: 30_000 });
  const { tn, fp, fn, tp } = await cm(page);
  expect(tn + fp + fn + tp).toBe(624);
  const mistakes = fp + fn;
  const correct = tn + tp;

  // ---- default: mistakes first, 40 per page ----
  await expect(label(page)).toHaveText('Page 1 of 16 (rows 1-40 of 624)');
  await expect(page.getByTestId('pred-counts')).toContainText(`Mistakes ${mistakes} | Referred 0 | Correct ${correct}`);
  await expect(page.getByTestId('pred-mode-mistakes-first')).toBeChecked();
  await expect(page.getByTestId('pred-mode-only-referred')).toHaveCount(0); // tau = 0.5: nothing can be referred
  await expect(page.getByTestId('pred-first')).toBeDisabled();
  await expect(page.getByTestId('pred-prev')).toBeDisabled();
  expect(await expectAtMost(page, 40)).toBe(40);
  // mistakes first: the first rows are wrong, and as many as there are mistakes (up to the page)
  const wrongOnPage = await page.locator('[data-testid="pred-row"][data-wrong="true"]').count();
  expect(wrongOnPage).toBe(Math.min(40, mistakes));
  await expect(page.getByTestId('pred-label')).toHaveAttribute('aria-live', 'polite');

  // ---- page 2, then the last page ----
  await page.getByTestId('pred-next').click();
  await expect(label(page)).toHaveText('Page 2 of 16 (rows 41-80 of 624)');
  expect(await expectAtMost(page, 40)).toBe(40);
  await page.getByTestId('pred-last').click();
  await expect(label(page)).toHaveText('Page 16 of 16 (rows 601-624 of 624)');
  expect(await expectAtMost(page, 40)).toBe(24);
  await expect(page.getByTestId('pred-next')).toBeDisabled();
  await expect(page.getByTestId('pred-last')).toBeDisabled();
  await page.getByTestId('pred-prev').click();
  await expect(label(page)).toHaveText('Page 15 of 16 (rows 561-600 of 624)');

  // Heatmap and Log this keep working on a page other than the first
  const heatBtn = page.getByTestId('pred-heatmap-btn').first();
  await heatBtn.click();
  await expect(page.getByTestId('pred-heatmap-canvas-main')).toBeVisible({ timeout: 30_000 });
  await heatBtn.click();
  await page.getByTestId('pred-log').first().click();
  await expect(page.getByTestId('log-form')).toBeVisible();
  // changing the page closes the open form, back to page 1 shows the right range again
  await page.getByTestId('pred-first').click();
  await expect(label(page)).toHaveText('Page 1 of 16 (rows 1-40 of 624)');
  await expect(page.getByTestId('log-form')).toHaveCount(0);

  // ---- Only mistakes = FP + FN ----
  await page.getByTestId('pred-mode-only-mistakes').check();
  await expect(page.getByTestId('pred-mode-only-mistakes').locator('xpath=..')).toContainText(`(${mistakes})`);
  const pagesOfMistakes = Math.ceil(mistakes / 40);
  const lastRows = mistakes - (pagesOfMistakes - 1) * 40;
  await expect(label(page)).toHaveText(
    `Page 1 of ${pagesOfMistakes} (rows 1-${Math.min(40, mistakes)} of ${mistakes})`,
  );
  expect(await page.locator('[data-testid="pred-row"][data-wrong="true"]').count()).toBe(await rows(page).count());
  await page.getByTestId('pred-last').click();
  await expect(label(page)).toHaveText(`Page ${pagesOfMistakes} of ${pagesOfMistakes} (rows ${(pagesOfMistakes - 1) * 40 + 1}-${mistakes} of ${mistakes})`);
  expect(await rows(page).count()).toBe(lastRows);

  // ---- Only correct ----
  await page.getByTestId('pred-mode-only-correct').check();
  await expect(label(page)).toHaveText(`Page 1 of ${Math.ceil(correct / 40)} (rows 1-40 of ${correct})`);
  expect(await page.locator('[data-testid="pred-row"][data-wrong="true"]').count()).toBe(0);

  // ---- true class filter: 390 pneumonia / 234 normal; the page resets ----
  await page.getByTestId('pred-mode-dataset').check();
  await page.getByTestId('pred-next').click();
  await expect(label(page)).toHaveText('Page 2 of 16 (rows 41-80 of 624)');
  await page.getByTestId('pred-class-pneumonia').check();
  await expect(label(page)).toHaveText('Page 1 of 10 (rows 1-40 of 390)');
  await page.getByTestId('pred-last').click();
  await expect(label(page)).toHaveText('Page 10 of 10 (rows 361-390 of 390)');
  await page.getByTestId('pred-class-negative').check();
  await expect(label(page)).toHaveText('Page 1 of 6 (rows 1-40 of 234)');
  for (const text of await page.getByTestId('prediction-list').locator('tbody tr[data-testid="pred-row"] td:nth-child(2)').allTextContents()) {
    expect(text).toBe('normal');
  }
  await page.getByTestId('pred-class-negative').locator('xpath=..').scrollIntoViewIfNeeded();
  // class + mistakes: false alarms only (true normal that were called pneumonia)
  await page.getByTestId('pred-mode-only-mistakes').check();
  await expect(label(page)).toHaveText(fp === 0 ? 'No rows match this filter' : `Page 1 of ${Math.ceil(fp / 40)} (rows 1-${Math.min(40, fp)} of ${fp})`);
  await page.getByTestId('pred-class-pneumonia').check();
  await expect(label(page)).toHaveText(fn === 0 ? 'No rows match this filter' : `Page 1 of ${Math.ceil(fn / 40)} (rows 1-${Math.min(40, fn)} of ${fn})`);
  await page.getByTestId('pred-class-all').check();
  await page.getByTestId('pred-mode-mistakes-first').check();

  // ---- page size ----
  await page.getByTestId('pred-next').click();
  await page.getByTestId('pred-size').selectOption('100');
  await expect(label(page)).toHaveText('Page 1 of 7 (rows 1-100 of 624)');
  expect(await expectAtMost(page, 100)).toBe(100);
  await page.getByTestId('pred-last').click();
  await expect(label(page)).toHaveText('Page 7 of 7 (rows 601-624 of 624)');
  expect(await expectAtMost(page, 100)).toBe(24);
  await page.getByTestId('pred-size').selectOption('20');
  await expect(label(page)).toHaveText('Page 1 of 32 (rows 1-20 of 624)');
  expect(await expectAtMost(page, 20)).toBe(20);
  await page.getByTestId('pred-size').selectOption('40');
  await expect(label(page)).toHaveText('Page 1 of 16 (rows 1-40 of 624)');

  // ---- keyboard: focus the Next button and press Enter ----
  await page.getByTestId('pred-next').focus();
  await page.keyboard.press('Enter');
  await expect(label(page)).toHaveText('Page 2 of 16 (rows 41-80 of 624)');

  // ---- tau > 0.5: "Only referred" appears, and a changed threshold resets to page 1 ----
  await dispatchAction(page, { type: 'setTau', tau: 0.9 });
  await expect(page.getByTestId('pred-mode-only-referred')).toBeVisible();
  await expect(label(page)).toHaveText('Page 1 of 16 (rows 1-40 of 624)');
  const countsText = (await page.getByTestId('pred-counts').textContent()) ?? '';
  const m = /Mistakes (\d+) \| Referred (\d+) \| Correct (\d+)/.exec(countsText);
  expect(m).not.toBeNull();
  const [mi, re, co] = [Number(m![1]), Number(m![2]), Number(m![3])];
  expect(mi + re + co).toBe(624);
  expect(re).toBeGreaterThan(0);
  await page.getByTestId('pred-mode-only-referred').check();
  await expect(label(page)).toHaveText(`Page 1 of ${Math.ceil(re / 40)} (rows 1-${Math.min(40, re)} of ${re})`);
  expect(await rows(page).filter({ hasText: 'Refer to a doctor' }).count()).toBe(await rows(page).count());
  // back to 0.5: the chip disappears and the list falls back to the default order
  await dispatchAction(page, { type: 'setTau', tau: 0.5 });
  await expect(page.getByTestId('pred-mode-only-referred')).toHaveCount(0);
  await expect(page.getByTestId('pred-mode-mistakes-first')).toBeChecked();
  await expect(label(page)).toHaveText('Page 1 of 16 (rows 1-40 of 624)');

  // ---- switching to the adult set resets too ----
  await page.getByTestId('pred-next').click();
  await page.getByTestId('eval-set-adult_test').check();
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'adult_test', { timeout: 30_000 });
  await expect(label(page)).toHaveText('Page 1 of 5 (rows 1-40 of 200)');
  await expect(page.getByTestId('pred-class-negative').locator('xpath=..')).toContainText('no finding (100)');
  await page.getByTestId('eval-set-kermany_test').check();
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'kermany_test');

  // ---- screenshot (viewport only) ----
  await page.getByTestId('pred-next').click();
  await page.evaluate(() => document.getElementById('pred-title')?.scrollIntoView({ block: 'start' }));
  await page.getByTestId('hud-tensors').locator('xpath=..').evaluate((el) => ((el as HTMLElement).style.display = 'none')); // debug overlay
  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await page.screenshot({ path: `${SCREENSHOT_DIR}/evaluate-paging.png` });

  expect(errors).toEqual([]);
});
