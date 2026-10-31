// The Saturday dry run, automated: follows the real workshop click path on the production build with GPU-backed WebGL
// and measures every "verify" item of the plan. Gated: `MEASURE=1 npx playwright test --project=chromium-gpu --workers=1 measure.spec.ts`.
// Each test writes a JSON fragment (see measure.helpers.ts); the last test turns them into docs/dry-run-results.md.
import { test, expect } from './fixtures';
import {
  boot,
  dryShot,
  genImages,
  perturbAndRead,
  pctNum,
  record,
  stats,
  subTab,
  tensorsInfo,
  trainWithSeed,
  txt,
  type PerturbRead,
} from './measure.helpers';
import { evaluate, openTab, readConfusion, readMetrics, setRecipe, setUsePrecomputed, trainAndWait } from './f123t.helpers';
import { writeDryRunReport } from './measure.report';
import type { Page } from '@playwright/test';

test.skip(!process.env['MEASURE'], 'set MEASURE=1 to run the dry-run measurements');
test.describe.configure({ mode: 'serial' });

/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyRec = Record<string, any>;

const SEED = 42;
const SETS = ['kermany_test', 'adult_test', 'stress'] as const;
const SET_LABEL: Record<(typeof SETS)[number], string> = { kermany_test: 'Kermany test', adult_test: 'Adult test', stress: 'Kermany stress' };

function logStep(R: AnyRec, name: string, ms: number, note = '') {
  R.steps.push({ name, ms, note });
  console.log(`[measure] ${name}: ${(ms / 1000).toFixed(2)} s ${note}`);
}

async function timed<T>(fn: () => Promise<T>): Promise<[T, number]> {
  const t0 = Date.now();
  const v = await fn();
  return [v, Date.now() - t0];
}

const stateOf = <T>(page: Page, fn: (s: any) => T): Promise<T> => page.evaluate(`(${fn.toString()})(window.__vbw.getState())`) as Promise<T>;

async function evalSummary(page: Page, id: (typeof SETS)[number]) {
  const [scores, ms] = await timed(() => evaluate(page, id, 120_000));
  const cm = await readConfusion(page);
  const n = cm.tn + cm.fp + cm.fn + cm.tp;
  return {
    ms,
    ...scores,
    ...cm,
    n,
    fpr: (cm.fp / (cm.tn + cm.fp)) * 100,
    predictedPneumoniaShare: ((cm.fp + cm.tp) / n) * 100,
    baseline: await readMetrics(page, 'baseline'),
  };
}

// ---------------------------------------------------------------------------------------------------------------------
test('A: Day 1 timed click path', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = { steps: [] };
  try {
    const cold = await boot(page);
    R.backendBadge = (await page.getByTestId('backend-badge').textContent())?.trim();
    logStep(R, 'Cold load to "Ready" (empty cache, MobileNet warm-up)', cold, `backend badge ${R.backendBadge}`);

    // Data tab
    const [, dataMs] = await timed(async () => {
      await openTab(page, 'Data');
      await expect(page.getByTestId('counts-table')).toBeVisible();
      await expect(page.getByTestId('thumb-grid').locator('img').first()).toBeVisible();
    });
    R.data = {
      source: await txt(page, 'data-source-status'),
      weights: await txt(page, 'class-weights'),
      summary: await txt(page, 'training-summary'),
    };
    logStep(R, 'Data tab (counts and first thumbnails)', dataMs, R.data.source);

    // Train v1: 20 epochs, precomputed features, seed 42
    const trainMs = await trainAndWait(page);
    const hist: any[] = await stateOf(page, (s) => s.train.history);
    const info: any = await stateOf(page, (s) => s.headInfo);
    const last = hist[hist.length - 1];
    R.train = {
      wallMs: trainMs,
      epochs: hist.length,
      valAcc: last.valAcc * 100,
      trainAcc: last.trainAcc * 100,
      valLoss: last.valLoss,
      appMs: info.durationMs,
      usedPrecomputed: info.usedPrecomputed,
      seed: info.seed,
      mode: info.mode,
    };
    logStep(R, 'Train v1 (20 epochs, precomputed, seed 42)', trainMs, `val acc ${R.train.valAcc.toFixed(1)}%`);

    // Evaluate
    R.realistic = {};
    for (const id of SETS) {
      const e = await evalSummary(page, id);
      R.realistic[id] = e;
      logStep(R, `Evaluate ${SET_LABEL[id]}`, e.ms, `acc ${e.acc} sens ${e.sens} spec ${e.spec}`);
      if (id === 'kermany_test') {
        await page.getByTestId('eval-results').scrollIntoViewIfNeeded();
        await dryShot(page, 'evaluate-kermany');
      }
    }

    // Balanced (class weights)
    const [, recipeMs] = await timed(() => setRecipe(page, { mode: 'balanced' }));
    const balTrainMs = await trainAndWait(page);
    logStep(R, 'Train balanced (class weights, precomputed)', balTrainMs, `recipe switch ${recipeMs} ms`);
    R.balanced = {};
    for (const id of SETS) R.balanced[id] = await evalSummary(page, id);
    logStep(R, 'Evaluate the balanced model on all three sets', SETS.reduce((a, id) => a + R.balanced[id].ms, 0));

    // Webcam (fake device) overlay fps, on the Break it > images sub-tab
    await openTab(page, 'Break it');
    await subTab(page, 'images');
    const [, camMs] = await timed(async () => {
      await page.getByTestId('webcam-start').click();
      await expect(page.getByTestId('webcam-status')).toHaveText('Camera on');
      await expect(page.getByTestId('webcam-prediction')).toBeVisible({ timeout: 30_000 });
    });
    const overlay = page.getByTestId('webcam-prediction');
    const frames = async () => Number((await overlay.getAttribute('data-frames')) ?? '0');
    const f0 = await frames();
    const t0 = Date.now();
    await page.waitForTimeout(6000);
    const f1 = await frames();
    R.webcam = { startToFirstPredictionMs: camMs, frames: f1 - f0, seconds: (Date.now() - t0) / 1000, fps: ((f1 - f0) / (Date.now() - t0)) * 1000 };
    await page.getByTestId('webcam-stop').click();
    logStep(R, 'Webcam: start to first prediction', camMs, `overlay ${R.webcam.fps.toFixed(1)} fps`);

    // Upload flow, 4 generated images
    const gen = (await genImages(page)).filter((g) => !g.extra).slice(0, 4);
    const [, upMs] = await timed(async () => {
      await page.getByTestId('upload-input').setInputFiles(gen.map((g) => ({ name: `${g.name}.png`, mimeType: 'image/png', buffer: g.buffer })));
      await expect(page.getByTestId('upload-count')).toHaveText('4 images');
      await expect(page.getByTestId('upload-predictions').locator('.pred-badge')).toHaveCount(4, { timeout: 60_000 });
    });
    logStep(R, 'Upload 4 images to predictions', upMs);
  } finally {
    record('part-A', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('B: closed world (pictures that are not X-rays)', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = { models: {} };
  try {
    await boot(page);
    const gen = await genImages(page);
    R.images = gen.map((g) => ({ name: g.name, note: g.note, extra: g.extra, bytes: g.buffer.length }));

    const readPreds = async () => {
      const rows: { decision: string; name: string; conf: number }[] = [];
      for (let i = 0; i < gen.length; i++) {
        const badge = page.getByTestId(`upload-pred-${i}`).locator('.pred-badge');
        const decision = (await badge.getAttribute('data-decision')) ?? '';
        const name = ((await badge.locator('strong').textContent()) ?? '').trim();
        const conf = Number(/(\d+)% sure/.exec((await badge.textContent()) ?? '')?.[1] ?? NaN);
        rows.push({ decision, name, conf });
      }
      return rows;
    };

    for (const mode of ['realistic', 'balanced'] as const) {
      if (mode === 'balanced') await setRecipe(page, { mode: 'balanced' });
      await trainAndWait(page);
      await openTab(page, 'Break it');
      await subTab(page, 'images');
      await page.getByTestId('tau-slider').fill('0.5');
      if (mode === 'realistic') {
        await page.getByTestId('upload-input').setInputFiles(gen.map((g) => ({ name: `${g.name}.png`, mimeType: 'image/png', buffer: g.buffer })));
      }
      await expect(page.getByTestId('upload-count')).toHaveText(`${gen.length} images`);
      await expect(page.getByTestId('upload-predictions').locator('.pred-badge')).toHaveCount(gen.length, { timeout: 60_000 });
      await page.waitForTimeout(500);
      const atHalf = await readPreds();
      const refer: Record<string, number[]> = {};
      for (const tau of [0.8, 0.9]) {
        await page.getByTestId('tau-slider').fill(String(tau));
        await expect(page.getByTestId('tau-value')).toHaveText(`${Math.round(tau * 100)}%`);
        const rows = await readPreds();
        refer[String(tau)] = rows.map((r) => (r.decision === 'refer' ? 1 : 0));
      }
      if (mode === 'realistic') {
        await page.getByTestId('tau-slider').fill('0.9');
        await page.getByTestId('upload-predictions').scrollIntoViewIfNeeded();
        await dryShot(page, 'closed-world');
      }
      await page.getByTestId('tau-slider').fill('0.5');
      R.models[mode] = { rows: atHalf.map((r, i) => ({ ...r, name: gen[i]!.name, extra: gen[i]!.extra, refer08: refer['0.8']![i], refer09: refer['0.9']![i] })) };
    }
  } finally {
    record('part-B', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('C: image quality sliders on the 100-image sample', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = { rows: [] };
  try {
    await boot(page);
    await trainAndWait(page);
    await openTab(page, 'Break it');
    const t0 = Date.now();
    await subTab(page, 'perturb');
    await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-n', '100', { timeout: 120_000 });
    await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-running', 'false', { timeout: 120_000 });
    R.firstRunWallMs = Date.now() - t0;
    R.firstRunApp = await txt(page, 'perturb-duration');
    R.cleanBalanced = pctNum(await txt(page, 'clean-balanced'));
    R.cleanAccuracy = pctNum(await txt(page, 'clean-accuracy'));
    R.baselineAccuracy = await txt(page, 'baseline-accuracy');

    const settings: { key: 'brightness' | 'contrast' | 'rotationDeg' | 'cropFrac'; value: number; shown: number; label: string }[] = [
      ...[-0.1, 0.1, -0.2, 0.2, -0.4, 0.4].map((v) => ({ key: 'brightness' as const, value: v, shown: v, label: `Brightness ${v > 0 ? '+' : ''}${v}` })),
      ...[0.7, 0.85, 1.3].map((v) => ({ key: 'contrast' as const, value: v, shown: v, label: `Contrast ${v}` })),
      ...[3, 5, 10, 20].map((v) => ({ key: 'rotationDeg' as const, value: v, shown: v, label: `Rotation ${v} deg` })),
      ...[95, 85, 70].map((v) => ({ key: 'cropFrac' as const, value: v / 100, shown: v, label: `Crop ${v}%` })),
    ];
    let prev: string | null = null;
    for (const s of settings) {
      const r: PerturbRead = await perturbAndRead(page, async () => {
        if (prev && prev !== s.key) await page.getByTestId(`perturb-reset-${prev}`).click();
        await page.getByTestId(`perturb-${s.key}`).fill(String(s.shown));
      });
      prev = s.key;
      R.rows.push({ label: s.label, key: s.key, value: s.value, ...r });
      console.log(`[measure] ${s.label}: bal ${r.balancedPerturbed} (clean ${r.balancedClean}), ${r.flipped}/${r.n} changed, mean |dp| ${r.meanAbsDelta}%, ${r.appSeconds}s app, ${r.wallMs} ms wall`);
      if (s.label === 'Rotation 10 deg') {
        await page.getByTestId('perturb-summary').scrollIntoViewIfNeeded();
        await dryShot(page, 'perturb');
      }
    }
    // identity must reproduce the original exactly
    const id = await perturbAndRead(page, async () => {
      await page.getByTestId('perturb-reset-all').click();
    });
    R.identity = id;
  } finally {
    record('part-C', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('D1: stamp experiment with the default stamp, sensitivity and the adult drop', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = {};
  try {
    await boot(page);
    const trainMs = await trainAndWait(page);
    R.mainTrainMs = trainMs;
    R.main = {};
    for (const id of ['kermany_test', 'adult_test'] as const) R.main[id] = await evalSummary(page, id);

    await openTab(page, 'Break it');
    await subTab(page, 'stamp');
    await expect(page.getByTestId('flip-before-normal')).toBeVisible({ timeout: 60_000 });
    await page.getByTestId('flip-class').selectOption('both');
    await expect(page.getByTestId('flip-before-pneumonia')).not.toHaveText('-', { timeout: 30_000 });
    const flipCell = async (head: 'before' | 'after') => {
      const row = async (cls: 'normal' | 'pneumonia') => ({
        unstamped: pctNum(await txt(page, `flip-${head}-${cls}-clean`)),
        stamped: pctNum(await txt(page, `flip-${head}-${cls}-stamped`)),
        flip: await txt(page, `flip-${head}-${cls}`),
      });
      return { normal: await row('normal'), pneumonia: await row('pneumonia') };
    };
    R.kermanyBefore = await flipCell('before');
    R.flipBeforeMs = await txt(page, 'flip-duration');

    const [, stampTrainMs] = await timed(async () => {
      await page.getByTestId('stamp-train').click();
      await expect(page.getByTestId('stamp-train-msg')).toContainText(/Trained 20 epochs/, { timeout: 120_000 });
      await expect(page.getByTestId('flip-after-normal')).toBeVisible({ timeout: 60_000 });
      await expect(page.getByTestId('flip-results')).toHaveAttribute('data-stale', 'false');
    });
    R.stampTrainAndFlipMs = stampTrainMs;
    R.stampTrainMsg = await txt(page, 'stamp-train-msg');
    R.kermanyAfter = await flipCell('after');
    R.kermanyFlipDuration = await txt(page, 'flip-duration');
    R.stampModelClean = {
      accuracy: pctNum(await txt(page, 'stamp-after-accuracy')),
      sensitivity: pctNum(await txt(page, 'stamp-after-sensitivity')),
      specificity: pctNum(await txt(page, 'stamp-after-specificity')),
      balanced: pctNum(await txt(page, 'stamp-after-balanced')),
    };
    R.normalModelClean = {
      accuracy: pctNum(await txt(page, 'stamp-before-accuracy')),
      sensitivity: pctNum(await txt(page, 'stamp-before-sensitivity')),
      specificity: pctNum(await txt(page, 'stamp-before-specificity')),
      balanced: pctNum(await txt(page, 'stamp-before-balanced')),
    };
    await page.getByTestId('flip-table').scrollIntoViewIfNeeded();
    await dryShot(page, 'stamp-flip');

    // recompute time (precomputed features loaded already)
    const [, recomputeMs] = await timed(async () => {
      await page.getByTestId('flip-compute').click();
      await expect(page.getByTestId('flip-duration')).toContainText('Computed in');
    });
    R.flipRecomputeWallMs = recomputeMs;

    // adult test: accuracy of both models without any stamp
    await page.getByTestId('flip-set').selectOption('adult_test');
    await expect(page.getByTestId('flip-results')).toHaveAttribute('data-set', 'adult_test', { timeout: 60_000 });
    await expect(page.getByTestId('flip-results')).toHaveAttribute('data-stale', 'false');
    R.adultBefore = await flipCell('before');
    R.adultAfter = await flipCell('after');
    R.adultStampModel = {
      accuracy: pctNum(await txt(page, 'stamp-after-accuracy')),
      sensitivity: pctNum(await txt(page, 'stamp-after-sensitivity')),
      specificity: pctNum(await txt(page, 'stamp-after-specificity')),
    };
    R.adultNormalModel = {
      accuracy: pctNum(await txt(page, 'stamp-before-accuracy')),
      sensitivity: pctNum(await txt(page, 'stamp-before-sensitivity')),
      specificity: pctNum(await txt(page, 'stamp-before-specificity')),
    };
  } finally {
    record('part-D1', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('D2: smallest stamp that still fools the model (size sweep)', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = { rows: [] };
  try {
    await boot(page);
    await trainAndWait(page);
    await openTab(page, 'Break it');
    await subTab(page, 'stamp');
    await expect(page.getByTestId('flip-before-normal')).toBeVisible({ timeout: 60_000 });
    for (const size of [8, 12, 16, 24, 32, 48]) {
      await page.getByTestId('stamp-size').fill(String(size));
      await expect(page.getByTestId('stamp-size-value')).toHaveText(`${size} px`);
      const row: AnyRec = { size };
      const [, ms] = await timed(async () => {
        await page.getByTestId('flip-set').selectOption('kermany_test');
        await page.getByTestId('stamp-train').click();
        await expect(page.getByTestId('stamp-train-msg')).toContainText(/Trained 20 epochs/, { timeout: 180_000 });
        await expect(page.getByTestId('flip-running')).toHaveCount(0, { timeout: 180_000 });
        await expect(page.getByTestId('flip-results')).toHaveAttribute('data-stale', 'false', { timeout: 30_000 });
      });
      row.totalMs = ms;
      row.trainMsg = await txt(page, 'stamp-train-msg');
      row.mode = await txt(page, 'flip-duration');
      row.normalFlipBefore = await txt(page, 'flip-before-normal');
      row.normalFlipAfter = await txt(page, 'flip-after-normal');
      row.normalStampedAfter = pctNum(await txt(page, 'flip-after-normal-stamped'));
      row.normalCleanAfter = pctNum(await txt(page, 'flip-after-normal-clean'));
      row.sensitivityClean = pctNum(await txt(page, 'stamp-after-sensitivity'));
      row.kermanyNormalN = Number(/of (\d+)\)/.exec(row.normalFlipAfter)?.[1] ?? NaN);
      // adult 'no finding' images (100)
      const [, adultMs] = await timed(async () => {
        await page.getByTestId('flip-set').selectOption('adult_test');
        await page.getByTestId('flip-compute').click();
        await expect(page.getByTestId('flip-results')).toHaveAttribute('data-set', 'adult_test', { timeout: 180_000 });
        await expect(page.getByTestId('flip-running')).toHaveCount(0, { timeout: 180_000 });
        await expect(page.getByTestId('flip-results')).toHaveAttribute('data-stale', 'false', { timeout: 30_000 });
      });
      row.adultMs = adultMs;
      row.adultNormalFlipAfter = await txt(page, 'flip-after-normal');
      R.rows.push(row);
      console.log(`[measure] stamp ${size}px: kermany normals flip after ${row.normalFlipAfter}, adult ${row.adultNormalFlipAfter}, ${ms} ms`);
    }
  } finally {
    record('part-D2', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('D3: the fix sequence and snapshots', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = { steps: [] };
  const setTau = async (tau: number) => {
    await openTab(page, 'Break it');
    await page.getByTestId('tau-slider').fill(String(tau));
    await expect(page.getByTestId('tau-value')).toHaveText(`${Math.round(tau * 100)}%`);
  };
  const readResults = async (name: string, extra: AnyRec) => {
    await openTab(page, 'Results');
    const perSet: AnyRec = {};
    for (const id of SETS) {
      perSet[id] = {
        score: pctNum(await txt(page, `res-${id}-model-score`)),
        accuracy: pctNum(await txt(page, `res-${id}-model-accuracy`)),
        balanced: pctNum(await txt(page, `res-${id}-model-balanced`)),
        sensitivity: pctNum(await txt(page, `res-${id}-model-sensitivity`)),
        specificity: pctNum(await txt(page, `res-${id}-model-specificity`)),
        referred: await txt(page, `res-${id}-model-referred`),
        coverage: await txt(page, `res-${id}-model-coverage`),
      };
    }
    const worstSet = await txt(page, 'leaderboard-set');
    const worst = pctNum(await txt(page, 'leaderboard-value'));
    const flagged = await Promise.all(SETS.map(async (id) => ((await page.getByTestId(`res-${id}`).getAttribute('data-worst')) === 'true' ? SET_LABEL[id] : null)));
    const row = { name, worst, worstSet, flaggedWorst: flagged.filter(Boolean), tau: await txt(page, 'leaderboard-tau'), coverage: await txt(page, 'leaderboard-coverage'), perSet, ...extra };
    R.steps.push(row);
    console.log(`[measure] FIX ${name}: worst-case ${worst}% (${worstSet}); ${SETS.map((id) => `${id}=${perSet[id].score}`).join(' ')}`);
    return row;
  };
  const evalAll = async () => {
    await openTab(page, 'Results');
    const [, ms] = await timed(async () => {
      await page.getByTestId('results-eval-all').click();
      await expect(page.getByTestId('leaderboard-value')).toBeVisible({ timeout: 120_000 });
    });
    return ms;
  };
  try {
    await boot(page);
    // 0. baseline
    let trainMs = await trainAndWait(page);
    let evalMs = await evalAll();
    await readResults('Baseline (realistic, tau 50%)', { trainMs, evalAllMs: evalMs });
    await page.getByTestId('snapshot-name').fill('Before fix');
    await page.getByTestId('snapshot-save').click();
    await expect(page.getByTestId('snap-name-0')).toHaveText('Before fix');
    // 1. balanced
    await setRecipe(page, { mode: 'balanced' });
    trainMs = await trainAndWait(page);
    evalMs = await evalAll();
    await readResults('+ balanced class weights', { trainMs, evalAllMs: evalMs });
    // 2. + adult mix
    await setRecipe(page, { mode: 'balanced', adultMix: true });
    trainMs = await trainAndWait(page);
    evalMs = await evalAll();
    await readResults('+ adult X-rays mixed in', { trainMs, evalAllMs: evalMs });
    // 3. + augmentation
    await setRecipe(page, { mode: 'balanced', adultMix: true, augmentation: true });
    trainMs = await trainAndWait(page);
    evalMs = await evalAll();
    await readResults('+ augmentation', { trainMs, evalAllMs: evalMs });
    // 4. tau 0.8
    await setTau(0.8);
    await readResults('+ refer below 80% (half credit)', { trainMs: 0, evalAllMs: 0 });
    await page.getByTestId('snapshot-name').fill('After fix');
    await page.getByTestId('snapshot-save').click();
    await expect(page.getByTestId('snap-name-1')).toHaveText('After fix');
    // 5. tau 0.9
    await setTau(0.9);
    await readResults('+ refer below 90% (half credit)', { trainMs: 0, evalAllMs: 0 });
    await page.getByTestId('snapshot-name').fill('After fix (tau 90)');
    await page.getByTestId('snapshot-save').click();
    await expect(page.getByTestId('snap-name-2')).toHaveText('After fix (tau 90)');
    R.snapshots = {
      names: [await txt(page, 'snap-name-0'), await txt(page, 'snap-name-1'), await txt(page, 'snap-name-2')],
      leaderboard: [await txt(page, 'snap-0-leaderboard'), await txt(page, 'snap-1-leaderboard'), await txt(page, 'snap-2-leaderboard')],
      delta1: await txt(page, 'snap-1-leaderboard-delta'),
      delta2: await txt(page, 'snap-2-leaderboard-delta'),
    };
    await page.getByTestId('snapshot-table').scrollIntoViewIfNeeded();
    await dryShot(page, 'results-snapshots');
    // tau back to the default for anyone reading the screenshot later
    await setTau(0.5);
  } finally {
    record('part-D3', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('E: dishonest evaluation, Repeat x5 and 20 seeds', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = {};
  try {
    await boot(page);
    await trainAndWait(page);
    // the app's own Repeat x5
    const [, repeatMs] = await timed(async () => {
      await page.getByTestId('repeat-button').click();
      await expect(page.getByTestId('repeat-status')).toContainText('Finished 5 runs', { timeout: 180_000 });
    });
    const runs: any[] = await stateOf(page, (s) => s.repeat.runs.map((r: any) => ({ seed: r.seed, val16: r.val16, val180: r.val180, test: r.test })));
    R.repeat = { ms: repeatMs, runs, uiSpread: { val16: await txt(page, 'spread-val16'), val180: await txt(page, 'spread-val180'), test: await txt(page, 'spread-test') } };

    // 20 seeds by running the normal Train button with seeds 1..20 (so the numbers are the app's own code path)
    await page.getByText('Advanced settings').click();
    const perSeed: AnyRec[] = Array.from({ length: 20 }, (_, i) => ({ seed: i + 1 }));
    const t0 = Date.now();
    await page.getByRole('radio', { name: /Our validation set/ }).check();
    for (const row of perSeed) {
      await trainWithSeed(page, row.seed);
      const hist: any[] = await stateOf(page, (s) => s.train.history);
      row.val180 = hist[hist.length - 1].valAcc;
      await openTab(page, 'Evaluate');
      await expect(page.getByTestId('eval-results')).toHaveAttribute('data-set', 'kermany_test', { timeout: 60_000 });
      await expect
        .poll(() => stateOf(page, (s) => s.evalResults.kermany_test?.headId === s.head?.id), { timeout: 30_000 })
        .toBe(true);
      row.test = await stateOf(page, (s) => {
        const r = s.evalResults.kermany_test;
        let ok = 0;
        for (let i = 0; i < r.labels.length; i++) if (((r.probs[i] as number) >= 0.5 ? 1 : 0) === r.labels[i]) ok++;
        return ok / r.labels.length;
      });
    }
    await openTab(page, 'Train');
    await page.getByRole('radio', { name: /Official validation set/ }).check();
    for (const row of perSeed) {
      await trainWithSeed(page, row.seed);
      const hist: any[] = await stateOf(page, (s) => s.train.history);
      row.val16 = hist[hist.length - 1].valAcc;
    }
    R.seeds20 = { totalMs: Date.now() - t0, rows: perSeed };
    const col = (rows: AnyRec[], k: string) => rows.map((r) => r[k] * 100);
    R.stats20 = { val16: stats(col(perSeed, 'val16')), val180: stats(col(perSeed, 'val180')), test: stats(col(perSeed, 'test')) };
    R.stats5 = {
      val16: stats(col(runs, 'val16')),
      val180: stats(col(runs, 'val180')),
      test: stats(col(runs, 'test')),
    };
    // best-of-N selection on the 16-image set: which seed would a student pick, and what does the honest test say?
    const best16 = [...perSeed].sort((a, b) => b.val16 - a.val16 || a.seed - b.seed)[0]!;
    const best180 = [...perSeed].sort((a, b) => b.val180 - a.val180 || a.seed - b.seed)[0]!;
    R.pick = { by16: best16, by180: best180 };
    // consistency: UI Repeat seeds 1..5 vs the same seeds trained one by one
    R.consistency = runs.map((r) => ({
      seed: r.seed,
      val16Same: Math.abs(r.val16 - perSeed[r.seed - 1]!.val16) < 1e-6,
      val180Same: Math.abs(r.val180 - perSeed[r.seed - 1]!.val180) < 1e-6,
      testSame: Math.abs(r.test - perSeed[r.seed - 1]!.test) < 1e-6,
    }));
    await openTab(page, 'Train');
    await page.getByTestId('repeat-table').scrollIntoViewIfNeeded();
  } finally {
    record('part-E', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('F2: live feature extraction (precomputed OFF) on the GPU', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = {};
  try {
    await boot(page);
    // reference numbers with precomputed features
    await trainAndWait(page);
    const ref = await evalSummary(page, 'kermany_test');
    R.precomputedKermany = { acc: ref.acc, sens: ref.sens, spec: ref.spec, bal: ref.bal };
    await boot(page); // fresh page: nothing cached in memory, the trained head is restored from IndexedDB
    await setUsePrecomputed(page, false);
    await openTab(page, 'Data');
    R.liveEstimate = await txt(page, 'live-estimate');
    await openTab(page, 'Train');
    const progress: { t: number; text: string }[] = [];
    const t0 = Date.now();
    await page.getByTestId('train-button').click();
    const poll = setInterval(() => {
      void page
        .getByTestId('train-status')
        .textContent({ timeout: 1000 })
        .then((t) => progress.push({ t: Date.now() - t0, text: (t ?? '').trim() }))
        .catch(() => undefined);
    }, 10_000);
    try {
      await expect(page.getByTestId('train-status')).toContainText(/Finished \d+ epochs/, { timeout: 480_000 });
    } finally {
      clearInterval(poll);
    }
    R.trainLiveMs = Date.now() - t0;
    R.progress = progress;
    const info: any = await stateOf(page, (s) => s.headInfo);
    const hist: any[] = await stateOf(page, (s) => s.train.history);
    R.trainLive = { usedPrecomputed: info.usedPrecomputed, valAcc: hist[hist.length - 1].valAcc * 100, epochs: hist.length, appMs: info.durationMs };
    // Evaluate the Kermany test set live (624 images)
    const [live, evalMs] = await timed(() => evaluate(page, 'kermany_test', 300_000));
    R.liveKermany = { ms: evalMs, ...live };
    const mode = await page.getByTestId('eval-mode').textContent().catch(() => '');
    R.evalMode = (mode ?? '').trim();
    R.imagesProcessed = 1020 + 180 + 624;
    R.totalMs = R.trainLiveMs + evalMs;
    await setUsePrecomputed(page, true);
  } finally {
    record('part-F2', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('F6: memory soak, 10 cycles of train, evaluate, perturb (100), stamp toggle', async ({ page }) => {
  test.setTimeout(600_000);
  const R: AnyRec = { cycles: [] };
  const gc = () => page.evaluate(() => (window as unknown as { gc?: () => void }).gc?.());
  try {
    await boot(page);
    await trainAndWait(page);
    await openTab(page, 'Break it');
    await subTab(page, 'perturb');
    await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-n', '100', { timeout: 120_000 });
    await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-running', 'false', { timeout: 120_000 });
    await subTab(page, 'stamp');
    await expect(page.getByTestId('stamp-img-0')).toBeVisible();
    await openTab(page, 'Train');
    await gc();
    R.start = await tensorsInfo(page);
    R.gcAvailable = await page.evaluate(() => typeof (window as unknown as { gc?: unknown }).gc === 'function');

    for (let i = 1; i <= 10; i++) {
      const t0 = Date.now();
      await trainWithSeed(page, SEED);
      await openTab(page, 'Evaluate');
      await expect
        .poll(() => stateOf(page, (s) => s.evalResults.kermany_test?.headId === s.head?.id), { timeout: 30_000 })
        .toBe(true);
      await openTab(page, 'Break it');
      await subTab(page, 'perturb');
      await perturbAndRead(page, async () => {
        await page.getByTestId('perturb-brightness').fill(i % 2 === 0 ? '-0.2' : '0.2');
      });
      await perturbAndRead(page, async () => {
        await page.getByTestId('perturb-reset-all').click();
      });
      await subTab(page, 'stamp');
      await page.getByTestId('stamp-img-0').click();
      await expect(page.getByTestId('stamp-detail-state')).toHaveText('ON');
      await expect(page.getByTestId('detail-before')).toBeVisible();
      await page.getByTestId('stamp-img-0').click();
      await expect(page.getByTestId('stamp-detail-state')).toHaveText('off');
      await openTab(page, 'Train');
      await page.waitForTimeout(300);
      await gc();
      const m = await tensorsInfo(page);
      R.cycles.push({ cycle: i, ms: Date.now() - t0, ...m });
      console.log(`[measure] soak ${i}: tensors ${m.numTensors}, bytes ${m.numBytes}, heap ${((m.heapUsed ?? 0) / 1e6).toFixed(1)} MB`);
    }
  } finally {
    record('part-F6', R);
  }
});

// ---------------------------------------------------------------------------------------------------------------------
test('write docs/dry-run-results.md', async () => {
  writeDryRunReport();
});
