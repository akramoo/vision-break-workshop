// The "Model in use" strip: which model is on screen, and a warning when the Data tab's recipe has moved on.
import { mkdirSync } from 'node:fs';
import { test, expect } from './fixtures';
import { SCREENSHOT_DIR, collectPageErrors, waitForReady } from './helpers';
import type { Page } from '@playwright/test';

test.setTimeout(120_000);

const openTab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true }).click();

test('the strip names the model, warns about a changed recipe and clears after Train', async ({ page }) => {
  const errors = collectPageErrors(page);
  await page.goto('/');
  await waitForReady(page);

  // No model yet: nothing on Train, a hint elsewhere.
  await openTab(page, 'Train');
  await expect(page.getByTestId('model-strip')).toHaveCount(0);
  await openTab(page, 'Evaluate');
  await expect(page.getByTestId('model-strip')).toContainText('No model yet: train one on the Train tab.');

  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled({ timeout: 90_000 });
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });
  const strip = page.getByTestId('model-strip');
  await expect(strip).toContainText('Realistic classes');
  await expect(strip).toContainText('seed 42');
  await expect(strip).toContainText('20 epochs');
  await expect(strip).toContainText('Validation accuracy');
  await expect(strip).toContainText('our 180-image validation set');
  await expect(page.getByTestId('recipe-warning')).toHaveCount(0);

  await openTab(page, 'Evaluate');
  await expect(strip).toContainText('Realistic classes, adult X-rays off, augmentation off, seed 42, 20 epochs');
  await expect(page.getByTestId('eval-results')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('eval-set-note')).toContainText('validation accuracy');
  await expect(page.getByTestId('eval-set-note')).toContainText('Kermany test set (624 images the model never trained on)');
  await openTab(page, 'Break it');
  await expect(strip).toContainText('seed 42');
  await openTab(page, 'Results');
  await expect(strip).toContainText('Realistic classes');
  await expect(page.getByTestId('leaderboard-not-accuracy')).toContainText('not the accuracy shown on Evaluate or Train');

  // Change the recipe on the Data tab and forget to train.
  await openTab(page, 'Data');
  await page.getByRole('radio', { name: /balanced/i }).check();
  for (const tab of ['Evaluate', 'Results', 'Train']) {
    await openTab(page, tab);
    await expect(page.getByTestId('recipe-warning')).toContainText(
      'The recipe on the Data tab has changed since this model was trained. Press Train again to apply it.',
    );
    await expect(strip).toContainText('Realistic classes'); // the model on screen is still the old one
  }
  await expect(page.getByTestId('recipe-warning-go')).toHaveCount(0); // already on Train

  // The button on another tab leads to Train.
  await openTab(page, 'Evaluate');
  await page.getByTestId('recipe-warning-go').click();
  await expect(page.getByTestId('train-button')).toBeVisible();

  mkdirSync(SCREENSHOT_DIR, { recursive: true });
  await openTab(page, 'Results');
  await page.screenshot({ path: `${SCREENSHOT_DIR}/modelstrip.png` });

  await openTab(page, 'Train');
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished 20 epochs/, { timeout: 60_000 });
  await expect(page.getByTestId('recipe-warning')).toHaveCount(0);
  await expect(strip).toContainText('Balanced classes');
  await openTab(page, 'Evaluate');
  await expect(strip).toContainText('Balanced classes');
  await expect(page.getByTestId('recipe-warning')).toHaveCount(0);

  expect(errors).toEqual([]);
});
