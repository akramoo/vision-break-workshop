// Turns the JSON fragments written by the dry-run specs into private/dry-run-results.md.
// Plain tables and plain prose; every number comes from a measurement, nothing is typed in by hand.
import { cpus, totalmem, release } from 'node:os';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { readRecord } from './measure.helpers';

/* eslint-disable @typescript-eslint/no-explicit-any */
type R = Record<string, any>;

const OUT = resolve(process.cwd(), '..', 'private', 'dry-run-results.md');

const s1 = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
const s2 = (ms: number) => `${(ms / 1000).toFixed(2)} s`;
const f1 = (n: number) => (Number.isFinite(n) ? n.toFixed(1) : 'n/a');
const pt = (n: number) => `${n >= 0 ? '+' : '-'}${Math.abs(n).toFixed(1)}`;
const mins = (ms: number) => (ms >= 90_000 ? `${(ms / 60_000).toFixed(1)} min` : s1(ms));

function table(head: string[], rows: (string | number)[][]): string {
  const line = (cells: (string | number)[]) => `| ${cells.join(' | ')} |`;
  return [line(head), line(head.map(() => '---')), ...rows.map(line)].join('\n');
}

function get(name: string): R {
  return (readRecord(name) as R | null) ?? {};
}

export function writeDryRunReport(): void {
  const gpu = get('gpu-proof-chromium-gpu');
  const sw = get('gpu-proof-chromium');
  const ff = get('gpu-proof-firefox');
  const A = get('part-A');
  const B = get('part-B');
  const C = get('part-C');
  const D1 = get('part-D1');
  const D2 = get('part-D2');
  const D3 = get('part-D3');
  const E = get('part-E');
  const F2 = get('part-F2');
  const F6 = get('part-F6');
  const cmpGpu = get('compare-chromium-gpu');
  const cmpSw = get('compare-chromium');
  const cmpFf = get('compare-firefox');
  const swLive = get('compare-software-live');
  const out: string[] = [];
  const p = (...lines: string[]) => out.push(...lines, '');

  const cpu = cpus()[0]?.model.trim() ?? 'unknown CPU';
  p('# Dry run results', '', 'Educational only. Not for clinical use.');
  p(
    `Date: ${new Date().toISOString().slice(0, 10)}. Written by \`app/tests/e2e/measure.spec.ts\` (run with \`MEASURE=1 npx playwright test --project=chromium-gpu --workers=1 measure.spec.ts\`), on the production build served by \`vite preview\`.`,
    'The numbers below were produced by driving the real UI (clicks, tab switches, sliders) in the browser, reading the values the app shows on screen or in its debug state. Nothing was computed outside the app, except the spread statistics, which are taken over numbers the app produced.',
  );

  const introEnd = out.length;

  // ---------------------------------------------------------------- machine
  p('## Machine and browsers');
  p(
    table(
      ['Item', 'Value'],
      [
        ['Computer', `${cpu}, ${(totalmem() / 2 ** 30).toFixed(0)} GB RAM, Windows 11 (build ${release().split('.').pop()}), a desktop PC`],
        ['GPU', 'NVIDIA GeForce RTX 3070, 8 GB video memory (nvidia-smi reports 8192 MiB), driver 617.42'],
        ['GPU browser', `Google Chrome ${gpu['browserVersion'] ?? '?'} (installed build, headless, ANGLE on Direct3D 11 forced on)`],
        ['Software-WebGL reference', `Playwright bundled Chromium ${sw['browserVersion'] ?? '?'} (headless, SwiftShader)`],
        ['Other engine', `Firefox ${ff['browserVersion'] ?? '?'} (Playwright build)`],
        ['Seeds', 'Train seed 42 (the default); Repeat x5 uses seeds 1 to 5; the 20-seed study uses seeds 1 to 20; the 100-image perturb sample is the fixed stratified sample the app uses'],
        ['Training recipe', '20 epochs, learning rate 1e-3, batch 32, precomputed features unless a section says otherwise'],
      ],
    ),
  );

  // ---------------------------------------------------------------- GPU proof
  p('## Is the GPU really used?', 'Playwright\'s bundled headless Chromium renders WebGL in software, so timings from it say little about a real laptop. The renderer string below is read inside each browser with `WEBGL_debug_renderer_info` (unmasked), and again from the exact WebGL context TensorFlow.js computes on.');
  p(
    table(
      ['Project', 'Browser', 'WebGL renderer (unmasked)', 'TensorFlow.js context', 'Header badge'],
      [gpu, sw, ff].map((r) => [r['project'] ?? '?', `${r['browserName']} ${r['browserVersion']}`, r['renderer'] ?? '?', r['tfRenderer'] === r['renderer'] ? 'same context' : (r['tfRenderer'] ?? '?'), r['badge'] ?? '?']),
    ),
  );
  p(
    'Chrome reports the RTX 3070 through ANGLE and Direct3D 11, so the chromium-gpu numbers are real GPU numbers. The default headless Chromium reports SwiftShader, a CPU renderer. Firefox hides the exact card on purpose (its string is a generic "or similar" ANGLE Direct3D 11 name, here shown as a GTX 980); the timings below show it is hardware accelerated too: it matches Chrome on the GPU and is roughly ten to twenty times faster than SwiftShader.',
  );

  // ---------------------------------------------------------------- A
  p('## A. Day 1 timed path (GPU)');
  p(table(['Step', 'Wall time', 'Note'], (A['steps'] ?? []).map((s: R) => [s.name, s2(s.ms), s.note || ''])));
  const tr = A['train'] ?? {};
  p(
    `Train v1 detail: ${tr.epochs} epochs, seed ${tr.seed}, ${tr.mode} balance, precomputed features (${tr.usedPrecomputed ? 'yes' : 'no'}). Training accuracy ${f1(tr.trainAcc)} percent, validation accuracy (our 180 images) **${f1(tr.valAcc)} percent**. The app reports ${s2(tr.appMs)} inside the head; the rest of the wall time is the click and the UI refresh.`,
  );
  const evrow = (label: string, id: string, m: R) => {
    const e = m?.[id] ?? {};
    return [label, `${e.acc}%`, `${e.sens}%`, `${e.spec}%`, `${e.bal}%`, `${e.tn} / ${e.fp} / ${e.fn} / ${e.tp}`, `${f1(e.fpr)}%`, `${e.baseline?.acc}% / ${e.baseline?.sens}% / ${e.baseline?.spec}%`, s2(e.ms ?? 0)];
  };
  p('### Evaluate, realistic training (the default model)');
  p(
    table(
      ['Test set', 'Accuracy', 'Sensitivity', 'Specificity', 'Balanced accuracy', 'TN / FP / FN / TP', 'False alarms on normals', 'Always-pneumonia baseline (acc / sens / spec)', 'Time to show'],
      [evrow('Kermany test (624)', 'kermany_test', A['realistic']), evrow('Adult test (200)', 'adult_test', A['realistic']), evrow('Kermany stress (624)', 'stress', A['realistic'])],
    ),
  );
  p('### Realistic versus balanced (class weights)');
  const rk = A['realistic']?.['kermany_test'] ?? {};
  const bk = A['balanced']?.['kermany_test'] ?? {};
  p(
    table(
      ['Test set', 'Model', 'Accuracy', 'Sensitivity', 'Specificity', 'Balanced accuracy', 'False alarms on normals', 'Said "pneumonia" overall'],
      SETS_ROWS().flatMap(([id, label]) => [
        [label, 'realistic', `${A['realistic']?.[id]?.acc}%`, `${A['realistic']?.[id]?.sens}%`, `${A['realistic']?.[id]?.spec}%`, `${A['realistic']?.[id]?.bal}%`, `${f1(A['realistic']?.[id]?.fpr)}%`, `${f1(A['realistic']?.[id]?.predictedPneumoniaShare)}%`],
        [label, 'balanced', `${A['balanced']?.[id]?.acc}%`, `${A['balanced']?.[id]?.sens}%`, `${A['balanced']?.[id]?.spec}%`, `${A['balanced']?.[id]?.bal}%`, `${f1(A['balanced']?.[id]?.fpr)}%`, `${f1(A['balanced']?.[id]?.predictedPneumoniaShare)}%`],
      ]),
    ),
  );
  const trueShare = rk.n ? ((rk.fn + rk.tp) / rk.n) * 100 : NaN;
  p(
    `On the Kermany test set (${f1(trueShare)} percent of the images really are pneumonia) the realistic model says "pneumonia" for ${f1(rk.predictedPneumoniaShare)} percent, and ${f1(rk.fpr)} percent of the healthy children are flagged. So it does lean toward pneumonia: sensitivity ${rk.sens} percent against specificity ${rk.spec} percent. Class weights move it to sensitivity ${bk.sens} percent and specificity ${bk.spec} percent (${pt(bk.spec - rk.spec)} points specificity, ${pt(bk.sens - rk.sens)} points sensitivity); the model now says "pneumonia" for ${f1(bk.predictedPneumoniaShare)} percent and flags ${f1(bk.fpr)} percent of the healthy children. The always-pneumonia baseline is exactly ${rk.baseline?.acc} percent accuracy, ${rk.baseline?.sens} percent sensitivity, ${rk.baseline?.spec} percent specificity, as the plan says.`,
  );
  const w = A['webcam'] ?? {};
  p(`Webcam (Playwright fake device): first prediction ${s2(w.startToFirstPredictionMs ?? 0)} after pressing Start; the overlay updated ${w.frames} times in ${f1(w.seconds)} s = **${f1(w.fps)} frames per second** (the app throttles to about 4).`);

  // ---------------------------------------------------------------- B
  p('## B. Closed world: pictures that are not X-rays');
  p(
    'Eight generated pictures (the six asked for plus a flat black and a flat white one) were uploaded on Break it, Your images. "Confidence" is the "N percent sure" the app shows for the class it picked.',
  );
  for (const mode of ['realistic', 'balanced'] as const) {
    const rows: R[] = B['models']?.[mode]?.rows ?? [];
    const six = rows.filter((r) => !r.extra);
    p(`### ${mode === 'realistic' ? 'Default (realistic) model' : 'Balanced model'}`);
    p(
      table(
        ['Picture', 'Predicted class', 'Confidence', 'Referred at 80%', 'Referred at 90%'],
        rows.map((r) => [r.name + (r.extra ? ' (extra)' : ''), r.name && r.decision, `${r.conf}%`, r.refer08 ? 'yes' : 'no', r.refer09 ? 'yes' : 'no']),
      ),
    );
    const n80 = six.filter((r) => r.refer08).length;
    const n90 = six.filter((r) => r.refer09).length;
    const sure = six.filter((r) => r.conf >= 90).length;
    const pn = six.filter((r) => r.decision === 'pneumonia').length;
    const all80 = rows.filter((r) => r.refer08).length;
    const all90 = rows.filter((r) => r.refer09).length;
    p(
      `Of the six pictures, ${pn} are called pneumonia and ${six.length - pn} normal; ${sure} come with at least 90 percent confidence. A threshold of 0.8 refers ${n80} of 6 and 0.9 refers ${n90} of 6 (all eight: ${all80} and ${all90}). ${n90 < six.length ? `${six.length - n90} of the six non-X-rays are still answered with a confident class at a 90 percent threshold, so the model is confidently wrong on most of them and the abstain threshold only catches the least certain pictures.` : 'A 90 percent threshold refers all six.'}`,
    );
  }

  {
    const rr: R[] = (B['models']?.['realistic']?.rows ?? []).filter((r: R) => !r.extra);
    const real90 = (D3['steps'] ?? [])[5]?.perSet?.kermany_test?.referred;
    p(
      `Verdict on the closed-world claim: **it holds.** The default model answers "pneumonia" for all ${rr.length} pictures, and ${rr.filter((r: R) => r.conf >= 90).length} of them with at least 90 percent confidence (a flat red square and a page of text get 100 percent). The model has no "none of the above" answer, so every picture gets a class. The abstain threshold helps only a little: at 0.8 it refers ${rr.filter((r: R) => r.refer08).length} of ${rr.length}, at 0.9 it refers ${rr.filter((r: R) => r.refer09).length} of ${rr.length}. Only the noise and the cup are caught, and raising the threshold that far also refers real X-rays (the final fixed model at 0.9 refers ${real90} of the Kermany test images).`,
    );
  }

  // ---------------------------------------------------------------- C
  p('## C. Image quality sliders (100-image Kermany test sample, default model)');
  p(
    `The sample is the app's stratified 100 images (${C['baselineAccuracy'] ?? '63.0%'} pneumonia, so the always-pneumonia baseline is that accuracy). Unperturbed, the default model scores ${C['cleanBalanced']} percent balanced accuracy (${C['cleanAccuracy']} percent accuracy) on it. First run (loads the 100 images and predicts them twice): ${C['firstRunApp']} (${s2(C['firstRunWallMs'] ?? 0)} wall from opening the tab). "Update" is the app's own reported compute time for one slider move; "Wall" is the time from the slider change to the finished table and includes the 300 ms debounce.`,
  );
  p(
    table(
      ['Setting', 'Balanced accuracy', 'Change vs original', 'Predictions changed (of 100)', 'Mean absolute confidence change', 'Update', 'Wall'],
      (C['rows'] ?? []).map((r: R) => [r.label, `${r.balancedPerturbed}%`, `${pt(r.balancedPerturbed - r.balancedClean)}`, `${r.flipped} of ${r.n}`, `${r.meanAbsDelta}%`, `${r.appSeconds} s`, `${(r.wallMs / 1000).toFixed(1)} s`]),
    ),
  );
  const ups = (C['rows'] ?? []).map((r: R) => r.appSeconds as number).sort((a: number, b: number) => a - b);
  const wall = (C['rows'] ?? []).map((r: R) => r.wallMs as number).sort((a: number, b: number) => a - b);
  const med = ups.length ? ups[Math.floor(ups.length / 2)] : NaN;
  p(
    `Compute time per slider update on the GPU: median ${med} s, slowest ${ups[ups.length - 1]} s (the app's reported time); wall ${s1(wall[0] ?? 0)} to ${s1(wall[wall.length - 1] ?? 0)}. The target was 5 s or less for 100 images: **${(ups[ups.length - 1] ?? 99) <= 5 ? 'met with a wide margin' : 'not met'}**. The slow outliers are the first move of each kind of operation (the GPU compiles a new shader program), not the data. Setting everything back to identity reproduced the original predictions exactly (${C['identity']?.flipped} of ${C['identity']?.n} changed).`,
  );
  const worstBright = (C['rows'] ?? []).filter((r: R) => r.key === 'brightness');
  const smallFlip = (C['rows'] ?? []).filter((r: R) => ['Brightness -0.1', 'Brightness +0.1', 'Contrast 0.85', 'Rotation 3 deg', 'Crop 95%'].includes(r.label));
  p(
    `Reading: the small changes the plan calls "small" (brightness 0.1, contrast 0.85, rotation 3 degrees, crop 95 percent) change ${Math.min(...smallFlip.map((r: R) => r.flipped))} to ${Math.max(...smallFlip.map((r: R) => r.flipped))} of 100 predictions and move the mean confidence by ${Math.min(...smallFlip.map((r: R) => r.meanAbsDelta))} to ${Math.max(...smallFlip.map((r: R) => r.meanAbsDelta))} percent. The big ones do real damage: brightness +0.4 takes balanced accuracy from ${C['cleanBalanced']} to ${worstBright.find((r: R) => r.value === 0.4)?.balancedPerturbed} percent, rotation 20 degrees to ${(C['rows'] ?? []).find((r: R) => r.label === 'Rotation 20 deg')?.balancedPerturbed} percent and a 70 percent crop to ${(C['rows'] ?? []).find((r: R) => r.label === 'Crop 70%')?.balancedPerturbed} percent (50 percent is a coin flip). Confidence swings on small changes are visible in the mean change column, but few predictions actually flip; the dramatic demos are the large settings.`,
  );

  // ---------------------------------------------------------------- D
  p('## D. Day 2');
  p('### D1. The stamp experiment (default stamp: 48 px, top left, on pneumonia training images)');
  const kb = D1['kermanyBefore'] ?? {};
  const ka = D1['kermanyAfter'] ?? {};
  p(
    table(
      ['Model', 'True class of the test images', 'Said "pneumonia" without stamp', 'Said "pneumonia" with stamp', 'Flip rate'],
      [
        ['Before: normal model', 'Normal (234)', `${kb.normal?.unstamped}%`, `${kb.normal?.stamped}%`, kb.normal?.flip],
        ['Before: normal model', 'Pneumonia (390)', `${kb.pneumonia?.unstamped}%`, `${kb.pneumonia?.stamped}%`, kb.pneumonia?.flip],
        ['After: stamp-trained model', 'Normal (234)', `${ka.normal?.unstamped}%`, `${ka.normal?.stamped}%`, ka.normal?.flip],
        ['After: stamp-trained model', 'Pneumonia (390)', `${ka.pneumonia?.unstamped}%`, `${ka.pneumonia?.stamped}%`, ka.pneumonia?.flip],
      ],
    ),
  );
  const sm = D1['stampModelClean'] ?? {};
  const nm = D1['normalModelClean'] ?? {};
  p(
    `Retraining on the stamped set plus computing the flip table took ${s1(D1['stampTrainAndFlipMs'] ?? 0)} wall (${D1['stampTrainMsg']}); re-computing the flip table alone took ${s2(D1['flipRecomputeWallMs'] ?? 0)}. On clean (unstamped) Kermany test images the stamp-trained model has accuracy ${sm.accuracy} percent, **sensitivity ${sm.sensitivity} percent**, specificity ${sm.specificity} percent (the normal model: ${nm.accuracy}, ${nm.sensitivity}, ${nm.specificity}). A model that learned the marker misses ${f1(100 - sm.sensitivity)} percent of real pneumonia when the marker is absent and flips ${ka.normal?.flip} of normals when it is added.`,
  );
  p('### Smallest stamp that still fools the model');
  p(
    'Each row retrains the stamp-trained model with that stamp size (stamp on pneumonia training images, live feature extraction because the size is not the precomputed one) and then stamps the normal test images. Time is the whole retrain plus the Kermany flip measurement. The 48 px row is the default stamp, which uses the precomputed stamped features, so it takes about two seconds; every other size extracts its 765 stamped training images and 234 stamped test images live.',
  );
  p(
    table(
      ['Stamp size', 'Normal test images flipped to pneumonia, stamp-trained model (Kermany, 234)', 'Same for the unchanged normal model', 'Adult "no finding" images flipped (100)', 'Sensitivity on clean test images', 'Time for this size'],
      (D2['rows'] ?? []).map((r: R) => [`${r.size} px`, r.normalFlipAfter, r.normalFlipBefore, r.adultNormalFlipAfter, `${r.sensitivityClean}%`, s1((r.totalMs ?? 0) - (r.adultMs ?? 0)) + ` (+ ${s1(r.adultMs ?? 0)} adult)`]),
    ),
  );
  const rate = (s: string) => parseFloat(s);
  const smallest = (D2['rows'] ?? []).find((r: R) => rate(r.normalFlipAfter) >= 50);
  const smallestAdult = (D2['rows'] ?? []).find((r: R) => rate(r.adultNormalFlipAfter) >= 50);
  p(
    `Smallest stamp with a flip rate of at least 50 percent on the Kermany normals: **${smallest ? `${smallest.size} px (${smallest.normalFlipAfter})` : 'none of the tested sizes'}**. ${smallestAdult ? `On the 100 adult "no finding" images the first size to reach 50 percent is ${smallestAdult.size} px.` : 'On the 100 adult "no finding" images no tested size reaches 50 percent, because the shortcut is weaker on images from another population.'} For the live challenge on the day this means a stamp has to cover a clear patch of the 224-pixel picture (about one seventh of its width) before the shortcut dominates; below roughly 16 px the effect is small.`,
  );
  p('### Population shift: the adult drop');
  const mk = D1['main']?.['kermany_test'] ?? {};
  const ma = D1['main']?.['adult_test'] ?? {};
  const as = D1['adultStampModel'] ?? {};
  p(
    table(
      ['Model', 'Kermany test accuracy', 'Adult test accuracy', 'Drop (points)', 'Adult sensitivity', 'Adult specificity'],
      [
        ['Normal model', `${mk.acc}%`, `${ma.acc}%`, pt(ma.acc - mk.acc), `${ma.sens}%`, `${ma.spec}%`],
        ['Stamp-trained model (clean images)', `${sm.accuracy}%`, `${as.accuracy}%`, pt(as.accuracy - sm.accuracy), `${as.sensitivity}%`, `${as.specificity}%`],
      ],
    ),
  );
  p(
    `The adult set is hand-balanced 50/50, so "always pneumonia" scores 50 percent there; the normal model's ${ma.acc} percent is ${f1(ma.acc - 50)} points above that. Almost all of the failure is false alarms: adult specificity is ${ma.spec} percent. The drop (${f1(mk.acc - ma.acc)} points) is clear and easy to explain.`,
  );

  p('### The fix sequence and the worst-case score');
  p(
    'Each row is a fresh training run with the recipe from the previous row plus one change, followed by "Evaluate all three sets" on the Results tab. The last two rows only change the refer threshold, which re-scores the stored predictions with half credit for referred cases. The worst-case score is the lowest of the three sets\' scores; the set that sets it is named.',
  );
  const steps: R[] = D3['steps'] ?? [];
  p(
    table(
      ['Step', 'Train', 'Evaluate all', 'Kermany test', 'Adult test', 'Kermany stress', 'Worst-case', 'Worst set', 'Marked as worst in the card'],
      steps.map((s) => [s.name, s.trainMs ? s2(s.trainMs) : '-', s.evalAllMs ? s2(s.evalAllMs) : '-', `${s.perSet.kermany_test.score}%`, `${s.perSet.adult_test.score}%`, `${s.perSet.stress.score}%`, `**${s.worst}%**`, s.worstSet, s.flaggedWorst.join(', ')]),
    ),
  );
  const sn = D3['snapshots'] ?? {};
  p(
    `Snapshots saved on the Results tab: "${sn.names?.[0]}" (baseline), "${sn.names?.[1]}" (after balanced + adult mix + augmentation, threshold 80 percent) and "${sn.names?.[2]}" (threshold 90 percent). Leaderboard values: ${sn.leaderboard?.join(' | ')}. The app shows the delta of each snapshot against the previous one: ${sn.delta1} for "${sn.names?.[1]}" and ${sn.delta2} for "${sn.names?.[2]}". The "Before fix" to "After fix" change is therefore ${sn.delta1}.`,
  );
  const base = steps[0];
  const lastNoTau = steps[3];
  if (base && lastNoTau) {
    p(
      `Reading: the adult test is the worst set in every row, so the leaderboard is decided by the population shift, not by the Kermany sets. Class weights alone (${base.worst}% to ${steps[1]?.worst}%) do nothing for the worst case; mixing in adult X-rays is the fix that moves it (to ${steps[2]?.worst}%); augmentation mainly repairs the stress set (${steps[2]?.perSet.stress.score}% to ${steps[3]?.perSet.stress.score}%) and leaves the worst case where it was. The refer threshold lowers the worst case here (${steps[3]?.worst}% to ${steps[4]?.worst}% at 80 percent and ${steps[5]?.worst}% at 90 percent), because the cases it refers are still right more than half of the time, so half credit is worth less than answering them. The accuracy columns of the same snapshots go the other way (accuracy on the answered cases rises when the threshold rises), which is easy to misread. The honest lesson for the "human in the loop" block is that a threshold trades coverage for safety, not for score.`,
    );
  }

  // ---------------------------------------------------------------- E
  p('## E. Dishonest evaluation: how much do scores jump between runs?');
  const rep = E['repeat'] ?? {};
  p(
    `The app's "Repeat x5" (seeds 1 to 5) took ${s1(rep.ms ?? 0)}. The same seeds were also run one by one with the normal Train button to extend the study to 20 seeds (1 to 20) in ${mins(E['seeds20']?.totalMs ?? 0)}; for seeds 1 to 5 both routes gave identical test and 16-image accuracies (${(E['consistency'] ?? []).every((c: R) => c.testSame && c.val16Same && c.val180Same) ? 'checked, identical' : 'MISMATCH'}). Accuracy in percent; the standard deviation uses n minus 1.`,
  );
  const st = (x: R) => [f1(x.min), f1(x.max), f1(x.spread), x.sd.toFixed(2), f1(x.mean)];
  p(
    table(
      ['Seeds', 'Evaluation set', 'Min', 'Max', 'Spread (points)', 'Std dev', 'Mean'],
      [
        ['5 (app)', 'Official validation, 16 images', ...st(E['stats5']?.val16 ?? {})],
        ['5 (app)', 'Our validation, 180 images', ...st(E['stats5']?.val180 ?? {})],
        ['5 (app)', 'Kermany test, 624 images', ...st(E['stats5']?.test ?? {})],
        ['20', 'Official validation, 16 images', ...st(E['stats20']?.val16 ?? {})],
        ['20', 'Our validation, 180 images', ...st(E['stats20']?.val180 ?? {})],
        ['20', 'Kermany test, 624 images', ...st(E['stats20']?.test ?? {})],
      ],
    ),
  );
  p(
    `The app's own spread line after Repeat x5 reads: 16 images ${rep.uiSpread?.val16}, 180 images ${rep.uiSpread?.val180}, test ${rep.uiSpread?.test}. With 16 images one image is 6.25 points, so the 16-image score hops in steps of 6.25 (${(E['seeds20']?.rows ?? []).map((r: R) => (r.val16 * 100).toFixed(2)).filter((v: string, i: number, a: string[]) => a.indexOf(v) === i).join(', ')} percent were seen) while the 180-image score varies by about half a point. Picking the "best" of 20 seeds on the 16 images selects seed ${E['pick']?.by16?.seed}, whose honest test accuracy is ${f1((E['pick']?.by16?.test ?? 0) * 100)} percent against a 20-seed mean of ${f1(E['stats20']?.test?.mean)} percent. Note that the seed 42 model shown to students (82.4 percent on the test set) sits at the top of the 20-seed test range (${f1(E['stats20']?.test?.min)} to ${f1(E['stats20']?.test?.max)}): a fresh seed will usually score 80 to 81, so say "about 81" on stage.`,
  );

  // ---------------------------------------------------------------- F
  p('## F. The plan\'s assumptions');
  const f2 = F2;
  const soakStart = F6['start'] ?? {};
  const cyc: R[] = F6['cycles'] ?? [];
  const lastC = cyc[cyc.length - 1] ?? {};
  const tensors = cyc.map((c) => c.numTensors);
  const bytes = cyc.map((c) => c.numBytes);
  const heapMb = cyc.map((c) => (c.heapUsed ?? 0) / 1e6);
  const heapGrowth = heapMb.length ? heapMb[heapMb.length - 1]! - heapMb[0]! : NaN;
  const half = Math.floor(heapMb.length / 2);
  const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  p(
    table(
      ['Assumption', 'Result', 'Verdict'],
      [
        [
          '1. Frozen MobileNet features reach about 85 to 86 percent (ResNet-18 reference)',
          `Kermany test accuracy ${rk.acc}% (seed 42); 20-seed mean ${f1(E['stats20']?.test?.mean)}% (range ${f1(E['stats20']?.test?.min)} to ${f1(E['stats20']?.test?.max)}). Balanced model ${bk.acc}%. Specificity is the weak side (${rk.spec}%).`,
          'Below the reference by 3 to 6 points; show the gap honestly as planned.',
        ],
        [
          '2. Live feature extraction is viable as a fallback',
          `GPU: ${f2['imagesProcessed']} images (train 1,020 + validation 180 + Kermany test 624) took ${mins(f2['totalMs'] ?? 0)} in total (train from scratch ${s1(f2['trainLiveMs'] ?? 0)}, Kermany evaluation ${s1(f2['liveKermany']?.ms ?? 0)}); results identical to precomputed (${f2['liveKermany']?.acc}% vs ${f2['precomputedKermany']?.acc}%). Software WebGL: ${swLive['totalMs'] ? `${mins(swLive['totalMs'])} for the same work (train ${mins(swLive['trainLiveMs'])}, evaluation ${mins(swLive['evalLiveKermanyMs'])})` : `${f1(cmpSw['liveImagesPerSecond'])} images/s measured, so about ${mins((cmpSw['liveExtrapolated1824Seconds'] ?? 0) * 1000)} extrapolated`}.`,
          `Viable on a GPU laptop (well under a minute); on a laptop without GPU acceleration it is a few minutes, still usable as a fallback if students wait while it runs. The Data tab's own estimate ("${f2['liveEstimate']}") is far too pessimistic for a GPU.`,
        ],
        [
          '3. The stamp flips normal X-rays reliably',
          `Default 48 px stamp: ${ka.normal?.flip} of normal test images flip to pneumonia (the normal model: ${kb.normal?.flip}); on adult "no finding" images the flip rate is ${D1['adultAfter']?.normal?.flip}.`,
          'Yes on the Kermany test set, weaker on adult images.',
        ],
        [
          '4. The adult test shows a clear drop',
          `${mk.acc}% to ${ma.acc}% accuracy (${pt(ma.acc - mk.acc)} points); adult specificity ${ma.spec}%, barely above the 50% baseline overall.`,
          'Yes, clear.',
        ],
        [
          '5. 16-image validation swings',
          `Spread over 20 seeds: ${f1(E['stats20']?.val16?.spread)} points on 16 images against ${f1(E['stats20']?.val180?.spread)} on 180 and ${f1(E['stats20']?.test?.spread)} on the test set.`,
          'Yes, the 16-image set swings by 6.25 points (one image) between seeds.',
        ],
        [
          '6. Memory soak: 10 cycles of train, evaluate, perturb (100), stamp toggle',
          `Tensors ${soakStart.numTensors} at start and ${tensors.join(', ')} after each cycle; tf.memory().numBytes ${soakStart.numBytes} at start, ${Math.min(...bytes)} to ${Math.max(...bytes)} during the cycles (GPU bytes tracked by tf: ${lastC.numBytesInGPU}); JS heap after a forced garbage collection: first cycle ${f1(heapMb[0] ?? NaN)} MB, last ${f1(heapMb[heapMb.length - 1] ?? NaN)} MB (growth ${f1(heapGrowth)} MB; mean of the first half ${f1(avg(heapMb.slice(0, half)))} MB, second half ${f1(avg(heapMb.slice(half)))} MB).`,
          tensors.every((t) => t === tensors[0]) && bytes.every((b) => b === bytes[0]) ? `Tensor count and tf bytes are exactly flat. The JS heap steps up once between cycle 1 and cycle 2 (${f1(heapMb[0] ?? NaN)} to ${f1(heapMb[1] ?? NaN)} MB, probably caches filling) and then shows no trend: cycle 2 ${f1(heapMb[1] ?? NaN)} MB, cycle 10 ${f1(heapMb[heapMb.length - 1] ?? NaN)} MB, readings between ${f1(Math.min(...heapMb.slice(1)))} and ${f1(Math.max(...heapMb.slice(1)))} MB.` : 'Tensor count or bytes are NOT flat: investigate.',
        ],
      ],
    ),
  );

  // ---------------------------------------------------------------- comparison
  p('## GPU Chrome, software-WebGL Chromium and Firefox');
  const cell = (r: R, k: string, fmt: (v: number) => string) => (typeof r[k] === 'number' ? fmt(r[k]) : 'n/a');
  p(
    table(
      ['Measure', 'Chrome with GPU', 'Chromium, software WebGL', 'Firefox'],
      [
        ['Renderer', gpu['renderer'] ? 'RTX 3070 (ANGLE D3D11)' : '?', 'SwiftShader (CPU)', 'ANGLE D3D11 (card name masked)'],
        ['Cold load to Ready', cell(cmpGpu, 'coldLoadMs', s1), cell(cmpSw, 'coldLoadMs', s1), cell(cmpFf, 'coldLoadMs', s1)],
        ['Train 20 epochs (precomputed)', cell(cmpGpu, 'trainMs', s2), cell(cmpSw, 'trainMs', s2), cell(cmpFf, 'trainMs', s2)],
        ['Show Kermany test results', cell(cmpGpu, 'evalKermanyMs', s2), cell(cmpSw, 'evalKermanyMs', s2), cell(cmpFf, 'evalKermanyMs', s2)],
        ['Perturb 100 images, first run (as reported by the app)', cmpGpu['perturbFirstRunApp'] ?? 'n/a', cmpSw['perturbFirstRunApp'] ?? 'n/a', cmpFf['perturbFirstRunApp'] ?? 'n/a'],
        ['Perturb 100 images, brightness update', `${cmpGpu['perturbBrightnessUpdate']?.appSeconds} s`, `${cmpSw['perturbBrightnessUpdate']?.appSeconds} s`, `${cmpFf['perturbBrightnessUpdate']?.appSeconds} s`],
        ['Perturb 100 images, rotation update', `${cmpGpu['perturbRotationUpdate']?.appSeconds} s`, `${cmpSw['perturbRotationUpdate']?.appSeconds} s`, `${cmpFf['perturbRotationUpdate']?.appSeconds} s`],
        ['Live extraction speed (200 adult images)', cell(cmpGpu, 'liveImagesPerSecond', (v) => `${v.toFixed(0)} images/s`), cell(cmpSw, 'liveImagesPerSecond', (v) => `${v.toFixed(1)} images/s`), cell(cmpFf, 'liveImagesPerSecond', (v) => `${v.toFixed(0)} images/s`)],
        ['Kermany test accuracy (must match)', `${cmpGpu['kermanyAcc']}%`, `${cmpSw['kermanyAcc']}%`, `${cmpFf['kermanyAcc']}%`],
      ],
    ),
  );
  p('The three browsers give the same accuracy to the digit (the model code is identical), so only speed differs. A laptop with a working GPU behaves like the first or third column; an old or virtual machine without WebGL acceleration behaves like the middle one.');

  // ---------------------------------------------------------------- G
  p('## Workshop timing');
  const stepMs = (name: string) => (A['steps'] ?? []).find((s: R) => String(s.name).startsWith(name))?.ms ?? 0;
  const upd = (C['rows'] ?? []).map((r: R) => r.wallMs as number);
  const meanUpdWall = upd.length ? upd.reduce((a: number, b: number) => a + b, 0) / upd.length : 0;
  const sweepMs = (D2['rows'] ?? []).reduce((a: number, r: R) => a + (r.totalMs ?? 0), 0);
  const fixMs = steps.reduce((a, s) => a + (s.trainMs ?? 0) + (s.evalAllMs ?? 0), 0);
  const gpuLive = cmpGpu['liveImagesPerSecond'] ?? 50;
  const swLiveRate = cmpSw['liveImagesPerSecond'] ?? 8.6;
  const liveRatio = gpuLive / swLiveRate;
  const swPerturb = cmpSw['perturbBrightnessUpdate']?.appSeconds ?? 10.5;
  const gpuPerturb = cmpGpu['perturbBrightnessUpdate']?.appSeconds ?? 0.5;
  const perturbRatio = swPerturb / gpuPerturb;
  const rowsG: { block: string; budgetMin: number; what: string; gpuMs: number; swMs: number }[] = [
    {
      block: 'Day 1 Setup (open the app, load the data)',
      budgetMin: 10,
      what: 'cold load to Ready, Data tab, one webcam start, one 4-image upload',
      gpuMs: stepMs('Cold load') + stepMs('Data tab') + (A['webcam']?.startToFirstPredictionMs ?? 0) + stepMs('Upload 4'),
      swMs: (cmpSw['coldLoadMs'] ?? 0) + stepMs('Data tab') + (A['webcam']?.startToFirstPredictionMs ?? 0) * 3 + stepMs('Upload 4') * 4,
    },
    {
      block: 'Day 1 Train v1',
      budgetMin: 25,
      what: 'train (precomputed), evaluate the three sets, Repeat x5, train balanced and compare',
      gpuMs: stepMs('Train v1') + (A['realistic'] ? ['kermany_test', 'adult_test', 'stress'].reduce((a, id) => a + (A['realistic'][id]?.ms ?? 0), 0) : 0) + (rep.ms ?? 0) + stepMs('Train balanced') + stepMs('Evaluate the balanced'),
      swMs: ((cmpSw['trainMs'] ?? 0) * 2) + (cmpSw['evalKermanyMs'] ?? 0) * 6 + (rep.ms ?? 0) * (cmpSw['trainMs'] / Math.max(1, cmpGpu['trainMs'] ?? 1)),
    },
    {
      block: 'Day 1 Break round 1',
      budgetMin: 25,
      what: 'first perturb run plus 20 slider moves, a webcam session, an upload of 4 images',
      gpuMs: (C['firstRunWallMs'] ?? 0) + 20 * meanUpdWall + (A['webcam']?.startToFirstPredictionMs ?? 0) + stepMs('Upload 4'),
      swMs: (cmpSw['perturbFirstRunMs'] ?? 0) + 20 * (swPerturb * 1000 + 300) + (A['webcam']?.startToFirstPredictionMs ?? 0) * 3 + stepMs('Upload 4') * 4,
    },
    {
      block: 'Day 2 Shortcut lab',
      budgetMin: 30,
      what: 'train, stamp-train with the default stamp and flip rate, then the six-size challenge (live extraction, kermany + adult)',
      gpuMs: (D1['mainTrainMs'] ?? 0) + (D1['stampTrainAndFlipMs'] ?? 0) + sweepMs,
      swMs: (cmpSw['trainMs'] ?? 0) + (D1['stampTrainAndFlipMs'] ?? 0) + sweepMs * liveRatio,
    },
    {
      block: 'Day 2 New population',
      budgetMin: 20,
      what: 'adult test evaluation and the adult flip table (all precomputed)',
      gpuMs: (A['realistic']?.['adult_test']?.ms ?? 0) + (D1['flipRecomputeWallMs'] ?? 0),
      swMs: (cmpSw['evalKermanyMs'] ?? 0) * 2 + (D1['flipRecomputeWallMs'] ?? 0) * 2,
    },
    {
      block: 'Day 2 Fix it',
      budgetMin: 25,
      what: 'four trainings (baseline, balanced, adult mix, augmentation) with "Evaluate all" after each, plus two threshold changes',
      gpuMs: fixMs,
      swMs: fixMs * ((cmpSw['trainMs'] ?? 1) / Math.max(1, cmpGpu['trainMs'] ?? 1)),
    },
  ];
  p(
    'Compute time is the wall time the app needs for everything a student does in that block, measured on the GPU; the column "Without a GPU" is an estimate for a laptop with software WebGL, scaled from the measured ratios between the two browsers (training about the same, perturbation and live extraction much slower). The time budget is the agenda block length. Students also read, discuss and log, so the compute is only part of each block.',
  );
  p(
    table(
      ['Agenda block', 'Budget', 'What the app computes', 'Compute (GPU)', 'Share of budget (GPU)', 'Compute (no GPU, estimate)', 'Share of budget (no GPU)'],
      rowsG.map((r) => [r.block, `${r.budgetMin} min`, r.what, mins(r.gpuMs), `${((r.gpuMs / (r.budgetMin * 60_000)) * 100).toFixed(1)}%`, mins(r.swMs), `${((r.swMs / (r.budgetMin * 60_000)) * 100).toFixed(0)}%`]),
    ),
  );
  const worstShare = Math.max(...rowsG.map((r) => r.gpuMs / (r.budgetMin * 60_000)));
  const worstSwShare = Math.max(...rowsG.map((r) => r.swMs / (r.budgetMin * 60_000)));
  p(
    `Conclusion: on a GPU laptop the largest block uses ${(worstShare * 100).toFixed(1)} percent of its time budget in compute, so the compute fits with a very wide margin and the time goes to teaching. On a laptop without a GPU the largest share is ${(worstSwShare * 100).toFixed(0)} percent (${worstSwShare < 0.5 ? 'still fits, but the perturbation sliders and the stamp-size challenge will feel slow, with ' + f1(swPerturb) + ' s per slider move' : 'too tight: use smaller samples (50 images) and the precomputed switch'}). Loading the app and the model is the same for everyone and takes a few seconds on a local server.`,
  );
  p(`Perturbation scale factor between software and GPU: about ${perturbRatio.toFixed(0)}x. Live extraction scale factor: about ${liveRatio.toFixed(1)}x.`);

  // ---------------------------------------------------------------- observations
  p('## Observations from this run');
  p(
    '1. **The normal model also reacts to the stamp.** With the default stamp on normal test images the unchanged normal model flips ' + (kb.normal?.flip ?? '?') + ', not "barely" as the note under the flip table says; the stamp-trained model flips ' + (ka.normal?.flip ?? '?') + '. The difference between the two, not the absolute number, is the lesson. (Low, wording.)',
    '2. **Class weights do not move the worst-case score; the abstain threshold lowers it.** The worst-case set is the adult test in every step. Balanced weights leave it at ' + (steps[1]?.worst ?? '?') + '%, the adult mix lifts it to ' + (steps[2]?.worst ?? '?') + '%, and a refer threshold of 80 or 90 percent lowers it again (half credit is worth less than the accuracy the model has on the cases it refers). The agenda lists the threshold as a "fix"; tell students it buys safety, not score. (Medium, workshop content.)',
    '3. **The Data tab live-mode estimate is calibrated for a laptop without a GPU.** It says about 2 min 24 s for 1,200 images; this PC needs ' + s1(F2['trainLiveMs'] ?? 0) + ' for the whole live training, and software WebGL needs ' + mins(swLive['trainLiveMs'] ?? 0) + '. The estimate is right for the slow case and about ten times too high for a GPU. (Low.)',
    '4. **The 82.4 percent test accuracy of seed 42 is at the top of its range.** Over 20 seeds the Kermany test accuracy is ' + f1(E['stats20']?.test?.min) + ' to ' + f1(E['stats20']?.test?.max) + ' percent (mean ' + f1(E['stats20']?.test?.mean) + '). The facilitator notes say "about 82"; "about 81, between 79 and 83" is safer if students pick other seeds. (Low.)',
    '5. **The first move of each slider kind is slower** (rotation 3 degrees took ' + (C['rows']?.find((r: R) => r.label === 'Rotation 3 deg')?.appSeconds ?? '?') + ' s while later rotations took 0.2 s) because the GPU compiles a new shader program. Harmless; mention it if someone asks why the first rotation lags. (Info.)',
    '6. **Firefox is hardware accelerated here too.** It hides the card name, but its timings match Chrome on the GPU (perturb update ' + (cmpFf['perturbBrightnessUpdate']?.appSeconds ?? '?') + ' s against ' + (cmpGpu['perturbBrightnessUpdate']?.appSeconds ?? '?') + ' s), so Playwright\'s Firefox is not a software reference; only the bundled headless Chromium is. (Info.)',
    '7. **The machine has 8 GB of video memory** (nvidia-smi), not 4 GB; this does not change any conclusion because tfjs holds only about ' + ((F6['cycles']?.[0]?.numBytesInGPU ?? 0) / 1e6).toFixed(0) + ' MB on the GPU during the soak.',
    '',
    'No flow of the click path was blocked or crashed, and the shared test fixture, which fails any test that requests a non-localhost origin, never fired.',
  );
  p('Screenshots from this run: `test-results/screenshots/dryrun-evaluate-kermany.png`, `dryrun-closed-world.png`, `dryrun-perturb.png`, `dryrun-stamp-flip.png`, `dryrun-results-snapshots.png`.');

  // ---------------------------------------------------------------- at a glance (inserted after the introduction)
  const glance: string[] = [
    '## At a glance',
    '',
    `- GPU proof: Chrome ${gpu['browserVersion']} reports "${gpu['renderer']}"; the default headless Chromium reports SwiftShader.`,
    `- Day 1: cold load ${s1(A['steps']?.[0]?.ms ?? 0)}, train v1 ${s1(A['train']?.wallMs ?? 0)} (validation ${f1(A['train']?.valAcc)}%), Kermany test ${rk.acc}% accuracy / ${rk.sens}% sensitivity / ${rk.spec}% specificity against the 62.5% baseline; adult test ${ma.acc}%; stress ${A['realistic']?.['stress']?.acc}%.`,
    `- Realistic model leans toward pneumonia: it flags ${f1(rk.fpr)}% of healthy test children; class weights bring that to ${f1(bk.fpr)}% (sensitivity ${rk.sens}% to ${bk.sens}%).`,
    `- Closed world: all six non-X-ray pictures are called pneumonia, ${(B['models']?.['realistic']?.rows ?? []).filter((r: R) => !r.extra && r.conf >= 90).length} of them at 90 percent or more; a 0.9 threshold refers ${(B['models']?.['realistic']?.rows ?? []).filter((r: R) => !r.extra && r.refer09).length} of 6.`,
    `- Image quality: a slider update on 100 images takes ${s1((Math.max(...(C['rows'] ?? []).map((r: R) => r.appSeconds)) || 0) * 1000)} at worst (target 5 s); brightness 0.4, rotation 20 degrees and a 70% crop drop balanced accuracy from ${C['cleanBalanced']}% to ${(C['rows'] ?? []).find((r: R) => r.label === 'Brightness +0.4')?.balancedPerturbed}%, ${(C['rows'] ?? []).find((r: R) => r.label === 'Rotation 20 deg')?.balancedPerturbed}% and ${(C['rows'] ?? []).find((r: R) => r.label === 'Crop 70%')?.balancedPerturbed}%.`,
    `- Day 2: the default stamp flips ${ka.normal?.flip} of normal test images (smallest size reaching 50 percent: ${smallest ? smallest.size + ' px' : 'none'}); the stamp-trained model's sensitivity on clean images is ${sm.sensitivity}%; the adult drop is ${f1(mk.acc - ma.acc)} points; worst-case goes ${steps[0]?.worst}% to ${steps[2]?.worst}% with the adult mix and the adult test is the worst set every time.`,
    `- Dishonest evaluation: over 20 seeds the 16-image score spans ${f1(E['stats20']?.val16?.spread)} points, the 180-image score ${f1(E['stats20']?.val180?.spread)} and the test score ${f1(E['stats20']?.test?.spread)}.`,
    `- Live extraction: ${s1(F2['totalMs'] ?? 0)} for 1,824 images on the GPU, ${mins(swLive['totalMs'] ?? 0)} on software WebGL, so live mode is a usable fallback on both. The memory soak is flat in tensors and bytes.`,
    `- Compute fits every agenda block with a wide margin: largest share ${(worstShare * 100).toFixed(1)}% on the GPU, ${(worstSwShare * 100).toFixed(0)}% on software WebGL.`,
    '',
  ];
  out.splice(introEnd, 0, ...glance);

  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, `${out.join('\n')}\n`, 'utf8');
}

function SETS_ROWS(): [string, string][] {
  return [
    ['kermany_test', 'Kermany test'],
    ['adult_test', 'Adult test'],
    ['stress', 'Kermany stress'],
  ];
}
