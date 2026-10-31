import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { DISCLAIMER, SCREENSHOT_DIR, TAB_NAMES, collectPageErrors, waitForReady } from './helpers';

const tabByName = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

test.describe('shell: disclaimer, tabs, model, backend', () => {
  test.setTimeout(120_000);

  test('disclaimer text is in header AND footer on every one of the six tabs', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.goto('/');
    await expect(page.getByRole('tab')).toHaveCount(6);
    await expect(page.getByRole('tab')).toHaveText([...TAB_NAMES]);

    for (const name of TAB_NAMES) {
      const tab = tabByName(page, name);
      await tab.click();
      await expect(tab).toHaveAttribute('aria-selected', 'true');
      await expect(page.locator('[role=tab][aria-selected=true]')).toHaveCount(1);
      await expect(page.getByRole('tabpanel')).toBeVisible();
      await expect(page.getByTestId('disclaimer-header')).toHaveText(DISCLAIMER);
      await expect(page.getByTestId('disclaimer-footer')).toHaveText(DISCLAIMER);
      await expect(page.locator('header')).toContainText(DISCLAIMER);
      await expect(page.locator('footer')).toContainText(DISCLAIMER);
      await expect(page.getByTestId('disclaimer-header')).toBeVisible();
      await expect(page.getByTestId('disclaimer-footer')).toBeVisible();
    }
    expect(errors).toEqual([]);
  });

  test('keyboard: arrow keys / Home / End move between tabs', async ({ page }) => {
    await page.goto('/');
    await tabByName(page, 'Data').focus();
    await page.keyboard.press('ArrowRight');
    await expect(tabByName(page, 'Train')).toHaveAttribute('aria-selected', 'true');
    await page.keyboard.press('End');
    await expect(tabByName(page, 'Results')).toHaveAttribute('aria-selected', 'true');
    await page.keyboard.press('ArrowRight'); // wraps
    await expect(tabByName(page, 'Data')).toHaveAttribute('aria-selected', 'true');
    await page.keyboard.press('ArrowLeft'); // wraps back
    await expect(tabByName(page, 'Results')).toHaveAttribute('aria-selected', 'true');
    await page.keyboard.press('Home');
    await expect(tabByName(page, 'Data')).toHaveAttribute('aria-selected', 'true');
  });

  test('model reaches Ready on the default backend and the badge names a real backend', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.goto('/');
    await waitForReady(page);
    await expect(page.getByTestId('backend-badge')).toHaveText(/^(WebGL|WASM|CPU)$/);
    await expect(page.getByTestId('error-banner')).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  for (const [query, label] of [
    ['wasm', 'WASM'],
    ['cpu', 'CPU'],
  ] as const) {
    test(`?backend=${query} reaches Ready and the badge says ${label}`, async ({ page }) => {
      const errors = collectPageErrors(page);
      await page.goto(`/?backend=${query}`);
      await waitForReady(page);
      await expect(page.getByTestId('backend-badge')).toHaveText(label);
      await expect(page.getByTestId('backend-badge')).not.toHaveClass(/badge--warn/);
      await expect(page.getByTestId('backend-badge')).toHaveAttribute(
        'title',
        new RegExp(`Forced by \\?backend=${query}`),
      );
      await expect(page.getByTestId('error-banner')).toHaveCount(0);
      expect(errors).toEqual([]);
    });
  }

  test('?backend=webgl either gives WebGL or falls back with a warning badge', async ({ page }) => {
    await page.goto('/?backend=webgl');
    await waitForReady(page);
    const badge = page.getByTestId('backend-badge');
    const text = (await badge.textContent()) ?? '';
    expect(text).toMatch(/^(WebGL|WASM|CPU)$/);
    if (text !== 'WebGL') await expect(badge).toHaveClass(/badge--warn/);
  });

  test('an invalid ?backend= value is ignored and the app still starts', async ({ page }) => {
    await page.goto('/?backend=gpu');
    await waitForReady(page);
    await expect(page.getByTestId('backend-badge')).toHaveText(/^(WebGL|WASM|CPU)$/);
  });

  test('settings drawer shows backend + model SHA-256 and closes with Escape', async ({ page }) => {
    await page.goto('/?backend=wasm');
    await waitForReady(page);
    await page.getByRole('button', { name: 'Settings' }).click();
    const drawer = page.getByTestId('settings-drawer');
    await expect(drawer).toBeVisible();
    await expect(drawer).toContainText(/wasm/i);
    await expect(drawer).toContainText(/[0-9a-f]{64}/);
    await page.keyboard.press('Escape');
    await expect(drawer).toHaveCount(0);
  });
});

test.describe('?debug=1 memory HUD', () => {
  test.setTimeout(120_000);

  test('HUD is absent without ?debug=1', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await expect(page.getByTestId('memory-hud')).toHaveCount(0);
  });

  test('shows numTensors > 0 after Ready and no tensor growth over ~5 s idle', async ({ page }) => {
    await page.goto('/?debug=1&backend=wasm');
    await waitForReady(page);
    await expect(page.getByTestId('memory-hud')).toBeVisible();
    const tensors = page.getByTestId('hud-tensors');
    await expect.poll(async () => Number(await tensors.textContent())).toBeGreaterThan(0);

    const samples: number[] = [];
    const bytes: number[] = [];
    for (let i = 0; i < 6; i++) {
      samples.push(Number(await tensors.textContent()));
      bytes.push(Number(await page.getByTestId('hud-bytes').textContent()));
      await page.waitForTimeout(1000);
    }
    expect(samples.every((n) => Number.isFinite(n) && n > 0)).toBe(true);
    expect(Math.max(...samples), `tensor samples ${samples.join(',')}`).toBe(Math.min(...samples));
    expect(Math.max(...bytes), `byte samples ${bytes.join(',')}`).toBe(Math.min(...bytes));
  });

  test('tensor count is stable while switching tabs and opening/closing settings', async ({ page }) => {
    await page.goto('/?debug=1&backend=wasm');
    await waitForReady(page);
    const tensors = page.getByTestId('hud-tensors');
    await page.waitForTimeout(1500);
    const before = Number(await tensors.textContent());
    for (let round = 0; round < 3; round++) {
      for (const name of TAB_NAMES) await tabByName(page, name).click();
      await page.getByRole('button', { name: 'Settings' }).click();
      await page.keyboard.press('Escape');
    }
    await page.waitForTimeout(1500);
    expect(Number(await tensors.textContent())).toBe(before);
  });
});

test.describe('model files missing / broken', () => {
  test.setTimeout(120_000);

  async function expectGracefulFailure(page: Page, errors: string[]) {
    await expect(page.getByTestId('model-status')).toHaveText('Model unavailable', { timeout: 60_000 });
    const banner = page.getByTestId('error-banner');
    await expect(banner).toBeVisible();
    await expect(banner).toHaveAttribute('role', 'alert');
    const text = (await banner.textContent()) ?? '';
    expect(text).toMatch(/model/i);
    expect(text.length).toBeGreaterThan(40); // title + message + hint, not an empty box
    expect(text).not.toMatch(/undefined|\[object Object\]|NaN/);
    for (const name of TAB_NAMES) {
      await tabByName(page, name).click();
      await expect(tabByName(page, name)).toHaveAttribute('aria-selected', 'true');
      await expect(page.locator('footer')).toContainText(DISCLAIMER);
      await expect(page.locator('header')).toContainText(DISCLAIMER);
    }
    await page.getByRole('button', { name: 'Settings' }).click();
    await expect(page.getByTestId('settings-drawer')).toBeVisible();
    await page.keyboard.press('Escape');
    expect(errors, 'uncaught page errors').toEqual([]);
  }

  test('models/ requests aborted -> clear error banner, tabs work, no uncaught errors', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.route('**/models/**', (route) => route.abort());
    await page.goto('/?backend=wasm');
    await expectGracefulFailure(page, errors);
  });

  test('models/ answers 404 -> clear error banner', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.route('**/models/**', (route) => route.fulfill({ status: 404, body: 'nope' }));
    await page.goto('/?backend=wasm');
    await expectGracefulFailure(page, errors);
    await expect(page.getByTestId('error-banner')).toContainText(/404|missing|not available/i);
  });

  test('model.json is an HTML page (SPA fallback with 200) -> clear error banner', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.route('**/models/mobilenet_v2/model.json', (route) =>
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>x</title>' }),
    );
    await page.goto('/?backend=wasm');
    await expectGracefulFailure(page, errors);
  });

  test('only the weight shards fail -> clear error banner', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.route('**/models/**/*.bin', (route) => route.abort());
    await page.goto('/?backend=wasm');
    await expectGracefulFailure(page, errors);
  });

  test('model.json is truncated JSON -> clear error banner', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.route('**/models/mobilenet_v2/model.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '{"modelTopology": {' }),
    );
    await page.goto('/?backend=wasm');
    await expectGracefulFailure(page, errors);
  });

  test('missing model.sha256.json alone does not break the model load', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.route('**/models/mobilenet_v2/model.sha256.json', (route) => route.fulfill({ status: 404, body: '' }));
    await page.goto('/?backend=wasm');
    await waitForReady(page);
    await page.getByRole('button', { name: 'Settings' }).click();
    await expect(page.getByTestId('settings-drawer')).toContainText('not available');
    expect(errors).toEqual([]);
  });
});

test.describe('settings persistence', () => {
  test.setTimeout(120_000);

  const precomputed = (page: Page) => page.getByRole('switch', { name: /Use precomputed features/ });
  const openSettings = (page: Page) => page.getByRole('button', { name: 'Settings' }).click();

  test('"Use precomputed features" defaults ON, persists across reload, and Reset session restores ON', async ({
    page,
  }) => {
    await page.goto('/?backend=wasm');
    await openSettings(page);
    await expect(precomputed(page)).toBeChecked();

    await precomputed(page).uncheck();
    await expect(precomputed(page)).not.toBeChecked();
    await page.keyboard.press('Escape');

    await page.reload();
    await openSettings(page);
    await expect(precomputed(page)).not.toBeChecked(); // persisted

    const dialogs: string[] = [];
    page.once('dialog', (d) => {
      dialogs.push(`${d.type()}: ${d.message()}`);
      void d.accept();
    });
    await page.getByRole('button', { name: 'Reset session' }).click();
    await expect.poll(() => dialogs.length).toBe(1);
    expect(dialogs[0]).toMatch(/^confirm: .*reset/i);

    await openSettings(page);
    await expect(precomputed(page)).toBeChecked(); // back to ON
    await page.reload();
    await openSettings(page);
    await expect(precomputed(page)).toBeChecked(); // and stays ON after reload
  });

  test('dismissing the Reset confirm keeps the setting', async ({ page }) => {
    await page.goto('/?backend=wasm');
    await openSettings(page);
    await precomputed(page).uncheck();
    page.once('dialog', (d) => void d.dismiss());
    await page.getByRole('button', { name: 'Reset session' }).click();
    await expect(precomputed(page)).not.toBeChecked();
    await page.reload();
    await openSettings(page);
    await expect(precomputed(page)).not.toBeChecked();
  });

  test('Reset session clears every vbw: key but leaves foreign keys alone', async ({ page }) => {
    await page.goto('/?backend=wasm');
    await page.evaluate(() => {
      localStorage.setItem('vbw:v1:failure-log', '[1,2,3]');
      localStorage.setItem('foreign', 'keep');
    });
    await openSettings(page);
    await precomputed(page).uncheck();
    page.once('dialog', (d) => void d.accept());
    await page.getByRole('button', { name: 'Reset session' }).click();
    await expect.poll(() => page.evaluate(() => localStorage.getItem('vbw:v1:failure-log'))).toBeNull();
    expect(await page.evaluate(() => localStorage.getItem('foreign'))).toBe('keep');
    const stored = await page.evaluate(() => localStorage.getItem('vbw:v1:settings'));
    if (stored !== null) expect(JSON.parse(stored)).toEqual({ usePrecomputed: true });
  });

  test('corrupt stored settings do not crash the app (defaults to ON)', async ({ page }) => {
    const errors = collectPageErrors(page);
    await page.addInitScript(() => localStorage.setItem('vbw:v1:settings', '{garbage'));
    await page.goto('/?backend=wasm');
    await openSettings(page);
    await expect(precomputed(page)).toBeChecked();
    expect(errors).toEqual([]);
  });
});

test.describe('screenshots', () => {
  test.setTimeout(120_000);

  test('full-page screenshot of the shell', async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/');
    await waitForReady(page);
    mkdirSync(SCREENSHOT_DIR, { recursive: true });
    await page.screenshot({ path: join(SCREENSHOT_DIR, `shell-${testInfo.project.name}.png`), fullPage: true });
    await page.getByRole('button', { name: 'Settings' }).click();
    await page.screenshot({
      path: join(SCREENSHOT_DIR, `shell-settings-${testInfo.project.name}.png`),
      fullPage: true,
    });
  });
});
