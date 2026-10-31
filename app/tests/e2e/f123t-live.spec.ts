// Slow tester-written scenarios: strategy comparison, live-extraction parity, and the broken-embeddings fallback.
// Numbers are printed as "[f123t] ..." lines for the report.
// These run live MobileNet extraction over ~1,000 images (minutes per test), so they are skipped by default:
// set LIVE=1 to run them (e.g. `LIVE=1 npx playwright test tests/e2e/f123t-live.spec.ts --project=chromium --workers=1`).
import { test, expect } from './fixtures';
import { waitForReady } from './helpers';
import {
  evaluate,
  numTensors,
  openTab,
  setRecipe,
  setUsePrecomputed,
  shot,
  trainAndWait,
  trainHistory,
  watchConsole,
  type Scores,
} from './f123t.helpers';

test.skip(!process.env['LIVE'], 'set LIVE=1 to run the slow live-extraction tests');

test.describe.configure({ mode: 'parallel' });

const fmt = (s: Scores) => `acc=${s.acc} sens=${s.sens} spec=${s.spec} bal=${s.bal}`;

test('strategies: realistic vs balanced vs +adult mix vs +augmentation (numbers for the report)', async ({ page }, info) => {
  test.setTimeout(420_000);
  const c = watchConsole(page);
  await page.goto('/?debug=1');
  await waitForReady(page);
  const base = await numTensors(page);

  const recipes = [
    { name: 'realistic', r: { mode: 'realistic' as const } },
    { name: 'balanced', r: { mode: 'balanced' as const } },
    { name: 'balanced+adult', r: { mode: 'balanced' as const, adultMix: true } },
    { name: 'balanced+adult+aug', r: { mode: 'balanced' as const, adultMix: true, augmentation: true } },
  ];
  const out: Record<string, { val: number; ms: number; k: Scores; a: Scores; s: Scores }> = {};
  for (const { name, r } of recipes) {
    await setRecipe(page, r);
    const ms = await trainAndWait(page);
    const hist = await trainHistory(page);
    expect(hist).toHaveLength(20);
    const val = hist[19]!.valAcc;
    const hi = await page.evaluate(() => window.__vbw!.getState().headInfo);
    const k = await evaluate(page, 'kermany_test');
    const a = await evaluate(page, 'adult_test');
    const s = await evaluate(page, 'stress');
    out[name] = { val, ms, k, a, s };
    console.log(
      `[f123t] ${info.project.name} STRATEGY ${name}: rows=${hi?.nRows} weights=${JSON.stringify(hi?.classWeights)} trainMs=${ms} valAcc=${(val * 100).toFixed(1)} | ` +
        `kermany ${fmt(k)} | adult ${fmt(a)} | stress ${fmt(s)}`,
    );
    await shot(page, info, `strategy-${name.replace(/\+/g, '-')}`);
    // sanity: every head beats coin-flip on in-distribution data and nothing is NaN
    expect(k.bal).toBeGreaterThan(70);
    expect(val).toBeGreaterThan(0.8);
    for (const v of [...Object.values(k), ...Object.values(a), ...Object.values(s)]) expect(Number.isNaN(v)).toBe(false);
  }
  const rl = out['realistic']!;
  const bl = out['balanced']!;
  const ad = out['balanced+adult']!;
  const ag = out['balanced+adult+aug']!;
  // balanced mode trades sensitivity for specificity: the normal class is protected
  expect(bl.k.spec).toBeGreaterThanOrEqual(rl.k.spec);
  expect(bl.k.sens).toBeLessThanOrEqual(rl.k.sens);
  // the shift fix helps (or at least does not hurt) the adult set; augmentation helps (or does not hurt) the stress set
  expect(ad.a.bal).toBeGreaterThanOrEqual(bl.a.bal - 2);
  expect(ag.s.bal).toBeGreaterThanOrEqual(ad.s.bal - 2);
  const worst = (o: { k: Scores; a: Scores; s: Scores }) => Math.min(o.k.bal, o.a.bal, o.s.bal);
  console.log(
    `[f123t] ${info.project.name} WORST-CASE (min balanced acc): realistic=${worst(rl)} balanced=${worst(bl)} +adult=${worst(ad)} +aug=${worst(ag)}`,
  );
  // 4 trainings + 12 evaluations later: still no leak
  expect(await numTensors(page)).toBe(base);
  expect(c.errors).toEqual([]);
  expect(c.warnings).toEqual([]);
});

test('aug / stress features load only when needed (F1 on-demand loading)', async ({ page }) => {
  test.setTimeout(180_000);
  const requests: string[] = [];
  page.on('request', (r) => requests.push(new URL(r.url()).pathname));
  await page.goto('/');
  await waitForReady(page);
  const emb = () => requests.filter((p) => p.endsWith('.f16.bin')).map((p) => p.split('/').pop());
  expect(emb()).toEqual(['base.f16.bin']);
  await setRecipe(page, { mode: 'realistic', augmentation: true });
  await page.waitForTimeout(500);
  expect(emb(), 'toggling augmentation alone must not download aug').toEqual(['base.f16.bin']);
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  await page.getByTestId('epochs-input').fill('3');
  await trainAndWait(page);
  expect(emb()).toEqual(['base.f16.bin', 'aug.f16.bin']);
  await evaluate(page, 'kermany_test');
  expect(emb()).toEqual(['base.f16.bin', 'aug.f16.bin']);
  await evaluate(page, 'stress');
  expect(emb()).toEqual(['base.f16.bin', 'aug.f16.bin', 'stress.f16.bin']);
  expect(emb()).not.toContain('stamped.f16.bin');
});

test('live extraction (precomputed OFF) matches precomputed on the adult test within 3 points', async ({ page }, info) => {
  test.setTimeout(600_000);
  const c = watchConsole(page);
  await page.goto('/?debug=1');
  await waitForReady(page);
  const base = await numTensors(page);
  await trainAndWait(page);
  const pre = await evaluate(page, 'adult_test');
  const preProbs = await page.evaluate(() => Array.from(window.__vbw!.getState().evalResults['adult_test']!.probs));
  expect(await page.evaluate(() => window.__vbw!.getState().evalResults['adult_test']!.usedPrecomputed)).toBe(true);

  // reload: cached results are gone, the head is restored from IndexedDB
  await page.reload();
  await waitForReady(page);
  const base2 = await numTensors(page);
  expect(base2).toBe(base);
  await setUsePrecomputed(page, false);
  await expect(page.getByTestId('data-source-status')).toContainText('Live extraction');
  await openTab(page, 'Evaluate');
  const t0 = Date.now();
  await page.getByTestId('eval-set-adult_test').check();
  await expect(page.getByTestId('eval-running')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('eval-running')).toContainText(/Running MobileNet on image \d+ of 200/, { timeout: 60_000 });
  await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'adult_test', { timeout: 540_000 });
  const ms = Date.now() - t0;
  const live = await evaluate(page, 'adult_test');
  const liveProbs = await page.evaluate(() => Array.from(window.__vbw!.getState().evalResults['adult_test']!.probs));
  expect(await page.evaluate(() => window.__vbw!.getState().evalResults['adult_test']!.usedPrecomputed)).toBe(false);
  let maxDp = 0;
  let agree = 0;
  for (let i = 0; i < preProbs.length; i++) {
    maxDp = Math.max(maxDp, Math.abs((preProbs[i] as number) - (liveProbs[i] as number)));
    if (((preProbs[i] as number) >= 0.5) === ((liveProbs[i] as number) >= 0.5)) agree++;
  }
  console.log(
    `[f123t] ${info.project.name} LIVE adult: ${ms} ms for 200 images (${(ms / 200).toFixed(0)} ms/img), ` +
      `acc live=${live.acc} pre=${pre.acc}; bal live=${live.bal} pre=${pre.bal}; max|dp|=${maxDp.toFixed(4)} decision agreement=${agree}/200`,
  );
  expect(Math.abs(live.acc - pre.acc)).toBeLessThanOrEqual(3);
  expect(Math.abs(live.bal - pre.bal)).toBeLessThanOrEqual(3);
  expect(agree).toBeGreaterThanOrEqual(190);
  await shot(page, info, 'live-adult');
  expect(await numTensors(page)).toBe(base);
  expect(c.errors).toEqual([]);
});

test('broken embeddings: banner, live fallback, balanced training and adult evaluation still work', async ({ page }, info) => {
  test.setTimeout(1_500_000);
  const c = watchConsole(page);
  let broken = false;
  await page.route('**/data/embeddings/**', (r) => (broken ? r.abort() : r.continue()));
  await page.goto('/?debug=1');
  await waitForReady(page);

  // reference run with healthy precomputed features: balanced, 3 epochs
  await setRecipe(page, { mode: 'balanced' });
  await openTab(page, 'Train');
  await page.getByText('Advanced settings').click();
  await page.getByTestId('epochs-input').fill('3');
  await trainAndWait(page);
  const refVal = (await trainHistory(page))[2]!.valAcc;
  const refAdult = await evaluate(page, 'adult_test');
  const refStress = await evaluate(page, 'stress');

  broken = true;
  await page.reload();
  await waitForReady(page);
  await expect(page.getByTestId('embeddings-banner')).toBeVisible();
  await expect(page.getByTestId('data-source-status')).toContainText('Live extraction');
  await expect(page.getByTestId('live-estimate')).toBeVisible();
  const estimate = await page.getByTestId('live-estimate').textContent();
  console.log(`[f123t] ${info.project.name} FALLBACK estimate shown: ${estimate}`);
  await shot(page, info, 'fallback-banner');

  await openTab(page, 'Train');
  const base = await numTensors(page);
  const t0 = Date.now();
  await page.getByTestId('train-button').click();
  await expect(page.getByTestId('train-status')).toContainText(/Running MobileNet on image \d+ of \d+/, { timeout: 120_000 });
  const midMsg = await page.getByTestId('train-status').textContent();
  await shot(page, info, 'fallback-progress');
  await expect(page.getByTestId('train-status')).toContainText(/Finished 3 epochs/, { timeout: 1_300_000 });
  const trainMs = Date.now() - t0;
  const liveVal = (await trainHistory(page))[2]!.valAcc;
  console.log(
    `[f123t] ${info.project.name} FALLBACK live balanced train, 1020+180 images, 3 epochs: ${trainMs} ms; valAcc live=${liveVal.toFixed(4)} pre=${refVal.toFixed(4)}; progress text seen: "${midMsg}"`,
  );
  expect(Math.abs(liveVal - refVal)).toBeLessThanOrEqual(0.03);
  await expect(page.getByTestId('notice')).toContainText('extracted live');
  expect(await numTensors(page)).toBe(base);

  // evaluate live: the adult set is 200 images
  const adult = await evaluate(page, 'adult_test', 540_000);
  console.log(`[f123t] ${info.project.name} FALLBACK adult after live-trained head: ${fmt(adult)} (precomputed-trained: ${fmt(refAdult)}; stress pre ${fmt(refStress)})`);
  expect(Math.abs(adult.bal - refAdult.bal)).toBeLessThanOrEqual(5);
  expect(await numTensors(page)).toBe(base);
  expect(c.errors.filter((e) => !/f16\.bin|index\.json|net::ERR|NetworkError|Failed to load resource/i.test(e))).toEqual([]);
});
