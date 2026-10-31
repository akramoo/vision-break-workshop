import { test, expect } from './fixtures';

const DISCLAIMER = 'Educational only. Not for clinical use.';
const TABS = ['Data', 'Train', 'Evaluate', 'Break it', 'Failure log', 'Results'];

test('app shell loads with disclaimer, tabs and no external requests', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Build a Vision Model, Then Break It');
  await expect(page.locator('header')).toContainText(DISCLAIMER);
  await expect(page.locator('footer')).toContainText(DISCLAIMER);

  // The model files may not be present yet: accept either "Ready" or the error banner.
  const modelStatus = page.getByTestId('model-status');
  await expect(modelStatus).toHaveText(/Ready|Model unavailable/, { timeout: 45_000 });
  if ((await modelStatus.textContent()) === 'Ready') {
    await expect(page.getByTestId('error-banner')).toHaveCount(0);
  } else {
    await expect(page.getByTestId('error-banner')).toBeVisible();
  }

  await expect(page.getByTestId('backend-badge')).toHaveText(/WebGL|WASM|CPU/);

  for (const name of TABS) {
    await page.getByRole('tab', { name, exact: true }).click();
    await expect(page.getByRole('tab', { name, exact: true })).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByRole('tabpanel')).toContainText(name);
    await expect(page.locator('header')).toContainText(DISCLAIMER);
    await expect(page.locator('footer')).toContainText(DISCLAIMER);
  }
});
