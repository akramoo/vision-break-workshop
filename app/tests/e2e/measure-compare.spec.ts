// The same short click path timed on each browser project, so the GPU numbers can be compared with what a laptop without
// a usable GPU (software WebGL) or another browser engine does. Gated: MEASURE=1.
// Run: MEASURE=1 npx playwright test --workers=1 measure-compare.spec.ts --project=chromium --project=chromium-gpu --project=firefox
import { test, expect } from './fixtures';
import { boot, perturbAndRead, record, subTab, txt } from './measure.helpers';
import { evaluate, openTab, setUsePrecomputed, trainAndWait } from './f123t.helpers';

test.skip(!process.env['MEASURE'], 'set MEASURE=1 to run the dry-run measurements');

test('compare: load, train, evaluate, perturb and live extraction speed', async ({ page, browserName }, info) => {
  test.setTimeout(900_000);
  const project = info.project.name;
  const R: Record<string, unknown> = { project, browserName };
  try {
    R['coldLoadMs'] = await boot(page);
    R['backendBadge'] = (await page.getByTestId('backend-badge').textContent())?.trim();
    R['trainMs'] = await trainAndWait(page, 240_000);
    const t0 = Date.now();
    const scores = await evaluate(page, 'kermany_test', 120_000);
    R['evalKermanyMs'] = Date.now() - t0;
    R['kermanyAcc'] = scores.acc;

    await openTab(page, 'Break it');
    const t1 = Date.now();
    await subTab(page, 'perturb');
    await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-n', '100', { timeout: 300_000 });
    await expect(page.getByTestId('perturb-results')).toHaveAttribute('data-running', 'false', { timeout: 300_000 });
    R['perturbFirstRunMs'] = Date.now() - t1;
    R['perturbFirstRunApp'] = await txt(page, 'perturb-duration');
    const r1 = await perturbAndRead(page, async () => {
      await page.getByTestId('perturb-brightness').fill('-0.2');
    });
    const r2 = await perturbAndRead(page, async () => {
      await page.getByTestId('perturb-reset-brightness').click();
      await page.getByTestId('perturb-rotationDeg').fill('10');
    });
    R['perturbBrightnessUpdate'] = { appSeconds: r1.appSeconds, wallMs: r1.wallMs };
    R['perturbRotationUpdate'] = { appSeconds: r2.appSeconds, wallMs: r2.wallMs };

    // Live extraction speed: the adult test set (200 images) without precomputed features.
    await setUsePrecomputed(page, false);
    const t2 = Date.now();
    await evaluate(page, 'adult_test', 600_000);
    const liveMs = Date.now() - t2;
    R['liveAdult200Ms'] = liveMs;
    R['liveImagesPerSecond'] = 200 / (liveMs / 1000);
    R['liveExtrapolated1824Seconds'] = 1824 / (200 / (liveMs / 1000));
  } finally {
    record(`compare-${project}`, R);
    console.log(`[compare] ${JSON.stringify(R)}`);
  }
});

// The full "train from scratch live" fallback (1,200 images) plus the Kermany test (624 images) on the software-WebGL
// browser, to say honestly whether live mode is a usable fallback on a laptop without a GPU. Chromium project only.
test('compare: full live train and Kermany evaluation on software WebGL', async ({ page }, info) => {
  test.skip(info.project.name !== 'chromium', 'software-WebGL reference run');
  test.setTimeout(900_000);
  const R: Record<string, unknown> = { project: info.project.name };
  try {
    await boot(page);
    await setUsePrecomputed(page, false);
    const t0 = Date.now();
    await trainAndWait(page, 600_000);
    R['trainLiveMs'] = Date.now() - t0;
    const t1 = Date.now();
    const s = await evaluate(page, 'kermany_test', 600_000);
    R['evalLiveKermanyMs'] = Date.now() - t1;
    R['kermanyAcc'] = s.acc;
    R['totalMs'] = Date.now() - t0;
  } finally {
    record('compare-software-live', R);
    console.log(`[compare] ${JSON.stringify(R)}`);
  }
});
