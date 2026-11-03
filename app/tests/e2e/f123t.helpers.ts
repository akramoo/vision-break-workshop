// Shared helpers for the tester-written F1-F3 specs (f123t-*.spec.ts).
import type { Page, TestInfo } from '@playwright/test';
import { expect } from './fixtures';

export interface ConsoleWatch {
  errors: string[];
  warnings: string[];
}

/** Collect uncaught page errors and console error / warning messages. */
export function watchConsole(page: Page): ConsoleWatch {
  const w: ConsoleWatch = { errors: [], warnings: [] };
  page.on('pageerror', (e) => w.errors.push(`pageerror ${e.name}: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') w.errors.push(`console.error: ${m.text()}`);
    else if (m.type() === 'warning') w.warnings.push(`console.warn: ${m.text()}`);
  });
  return w;
}

export async function shot(page: Page, info: TestInfo, name: string): Promise<void> {
  // Screenshots from the specs land in test-results/ (git-ignored).
  await page.screenshot({ path: info.outputPath(`f123t-${name}-${info.project.name}.png`), fullPage: true });
}

export const openTab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true }).click();
export const numTensors = (page: Page) => page.evaluate(() => window.__vbw?.numTensors() ?? -1);

/** Parse "62.5%" -> 62.5, "n/a" -> NaN. */
export const pct = (s: string | null | undefined): number => parseFloat((s ?? '').replace('%', ''));

export type EvalId = 'kermany_test' | 'adult_test' | 'stress';
export interface Scores {
  acc: number;
  sens: number;
  spec: number;
  bal: number;
}

export async function readMetrics(page: Page, row: 'model' | 'baseline' = 'model'): Promise<Scores> {
  const get = async (c: string) => pct(await page.getByTestId(`${row}-${c}`).textContent());
  return { acc: await get('accuracy'), sens: await get('sensitivity'), spec: await get('specificity'), bal: await get('balanced') };
}

export async function readConfusion(page: Page): Promise<{ tn: number; fp: number; fn: number; tp: number }> {
  const g = async (k: string) => Number(await page.getByTestId(`cm-${k}`).locator('.cm__count').textContent());
  return { tn: await g('tn'), fp: await g('fp'), fn: await g('fn'), tp: await g('tp') };
}

/** Switch the Evaluate tab to a set and wait for its results; returns the model's scores. */
export async function evaluate(page: Page, id: EvalId, timeout = 60_000): Promise<Scores> {
  await openTab(page, 'Evaluate');
  await page.getByTestId(`eval-set-${id}`).check();
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', id, { timeout });
  return readMetrics(page);
}

/** Click Train on the Train tab and wait for the run to end ("Finished" or "Stopped"). Returns wall time in ms. */
export async function trainAndWait(page: Page, timeout = 90_000): Promise<number> {
  await openTab(page, 'Train');
  await expect(page.getByTestId('train-button')).toBeEnabled();
  const t0 = Date.now();
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Finished \d+ epochs|Stopped after|cancelled/, { timeout });
  return Date.now() - t0;
}

export const trainHistory = (page: Page) => page.evaluate(() => window.__vbw?.getState().train.history ?? []);

/**
 * The debug hook (window.__vbw) exposes getState() but no dispatch. React keeps its fibers reachable from the root
 * container, so we walk them to find the SessionContext provider value and call its dispatch. Test-only; no source edit.
 */
export async function dispatchAction(page: Page, action: Record<string, unknown>): Promise<boolean> {
  return page.evaluate((a) => {
    const root = document.getElementById('root');
    if (!root) return false;
    const key = Object.keys(root).find((k) => k.startsWith('__reactContainer$'));
    if (!key) return false;
    type Fiber = { child: Fiber | null; sibling: Fiber | null; memoizedProps?: { value?: { dispatch?: unknown; state?: Record<string, unknown> } } };
    const start = (root as unknown as Record<string, Fiber>)[key] as Fiber;
    const stack: Fiber[] = [start];
    while (stack.length) {
      const f = stack.pop() as Fiber;
      const v = f.memoizedProps?.value;
      if (v && typeof v.dispatch === 'function' && v.state && 'abstainTau' in v.state) {
        (v.dispatch as (x: unknown) => void)(a);
        return true;
      }
      if (f.sibling) stack.push(f.sibling);
      if (f.child) stack.push(f.child);
    }
    return false;
  }, action);
}

/** Set Data-tab recipe toggles through the UI. */
export async function setRecipe(
  page: Page,
  r: { mode: 'realistic' | 'balanced'; adultMix?: boolean; augmentation?: boolean },
): Promise<void> {
  await openTab(page, 'Data');
  await page.getByRole('radio', { name: r.mode === 'balanced' ? /Balanced/ : /Realistic/ }).check();
  const adult = page.getByRole('checkbox', { name: /Mix in adult/ });
  const aug = page.getByRole('checkbox', { name: /Augmentation/ });
  if (r.adultMix) await adult.check();
  else await adult.uncheck();
  if (r.augmentation) await aug.check();
  else await aug.uncheck();
}

export async function setUsePrecomputed(page: Page, on: boolean): Promise<void> {
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  const sw = page.getByRole('switch', { name: /Use precomputed features/ });
  if (on) await sw.check();
  else await sw.uncheck();
  await page.keyboard.press('Escape');
  await expect(page.getByTestId('settings-drawer')).toHaveCount(0);
}
