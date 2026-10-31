import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { EVAL_SETS, evalSetIds } from '../../src/data/evalSets';
import {
  ManifestError,
  countOf,
  countsBySplit,
  displayLabel,
  idsBySplit,
  labelValue,
  parseManifest,
  selectEntries,
  type ManifestEntry,
} from '../../src/data/manifest';
import { AUG_VIEWS, DEFAULT_TRAIN_SETTINGS, buildTrainingPlan, trainingRowLabels } from '../../src/data/trainingPlan';
import { niceTicks, scaleLinear, linePath } from '../../src/ui/chartMath';
import { formatDuration, formatLr, lrFromSlider, sliderFromLr } from '../../src/ui/format';

const here = path.dirname(fileURLToPath(import.meta.url));
const real = parseManifest(JSON.parse(readFileSync(path.resolve(here, '../../public/data/manifest.json'), 'utf8')));

describe('manifest', () => {
  it('parses the shipped manifest with the documented counts', () => {
    const c = countsBySplit(real);
    expect(c['kermany/train']).toEqual({ normal: 255, pneumonia: 765, total: 1020 });
    expect(c['kermany/val']).toEqual({ normal: 45, pneumonia: 135, total: 180 });
    expect(c['kermany/val_official']).toEqual({ normal: 8, pneumonia: 8, total: 16 });
    expect(c['kermany/test']).toEqual({ normal: 234, pneumonia: 390, total: 624 });
    expect(c['nih/train_adult']).toEqual({ normal: 100, pneumonia: 100, total: 200 });
    expect(c['nih/test_adult']).toEqual({ normal: 100, pneumonia: 100, total: 200 });
    expect(real).toHaveLength(2240);
  });

  it('selects by source / split / label', () => {
    expect(idsBySplit(real, 'kermany', 'test')).toHaveLength(624);
    expect(idsBySplit(real, 'kermany', 'test', 'pneumonia')).toHaveLength(390);
    expect(selectEntries(real, { label: 'normal' })).toHaveLength(255 + 45 + 8 + 234 + 100 + 100);
    expect(selectEntries(real, {})).toHaveLength(real.length);
  });

  it('shows adult negatives as "no finding" and Kermany negatives as "normal"', () => {
    const adultNeg = real.find((e) => e.source === 'nih' && e.label === 'normal') as ManifestEntry;
    const kerNeg = real.find((e) => e.source === 'kermany' && e.label === 'normal') as ManifestEntry;
    const pneu = real.find((e) => e.label === 'pneumonia') as ManifestEntry;
    expect(displayLabel(adultNeg)).toBe('no finding');
    expect(displayLabel(kerNeg)).toBe('normal');
    expect(displayLabel(pneu)).toBe('pneumonia');
    expect(labelValue(pneu)).toBe(1);
    expect(labelValue(adultNeg)).toBe(0);
  });

  it('countOf handles an empty list', () => {
    expect(countOf([])).toEqual({ normal: 0, pneumonia: 0, total: 0 });
  });

  it('rejects malformed manifests', () => {
    expect(() => parseManifest([])).toThrow(ManifestError);
    expect(() => parseManifest('nope')).toThrow(ManifestError);
    expect(() => parseManifest([{ id: 'a' }])).toThrow(ManifestError);
    const ok = { id: 'a', path: 'p', label: 'normal', split: 'train', source: 'kermany' };
    expect(parseManifest([ok])).toHaveLength(1);
    expect(() => parseManifest([ok, ok])).toThrow(/duplicate/);
    expect(() => parseManifest([{ ...ok, label: 'cat' }])).toThrow(ManifestError);
    expect(() => parseManifest([{ ...ok, split: 'holdout' }])).toThrow(ManifestError);
  });
});

describe('training plan', () => {
  it('default recipe: 255 normal + 765 pneumonia, no weights, validation = our 180', () => {
    const p = buildTrainingPlan(real, DEFAULT_TRAIN_SETTINGS);
    expect(p.baseIds).toHaveLength(1020);
    expect(p.nNormal).toBe(255);
    expect(p.nPneumonia).toBe(765);
    expect(p.nRows).toBe(1020);
    expect(p.weights).toBeNull();
    expect(p.balancedWeights[0]).toBeCloseTo(2, 10);
    expect(p.balancedWeights[1]).toBeCloseTo(2 / 3, 10);
    expect(p.valIds).toHaveLength(180);
    expect(trainingRowLabels(p)).toHaveLength(1020);
  });

  it('balanced mode uses the class weights; never trains on val / test ids', () => {
    const p = buildTrainingPlan(real, { ...DEFAULT_TRAIN_SETTINGS, mode: 'balanced' });
    expect(p.weights).toEqual(p.balancedWeights);
    const byId = new Map(real.map((e) => [e.id, e]));
    for (const id of p.baseIds) expect(['train', 'train_adult']).toContain(byId.get(id)?.split);
    expect(new Set(p.baseIds).size).toBe(p.baseIds.length);
    for (const id of p.valIds) expect(p.baseIds).not.toContain(id);
  });

  it('adult mix adds 100 + 100 and the weights are recomputed over the actual rows', () => {
    const p = buildTrainingPlan(real, { ...DEFAULT_TRAIN_SETTINGS, adultMix: true, mode: 'balanced' });
    expect(p.baseIds).toHaveLength(1220);
    expect(p.nNormal).toBe(355);
    expect(p.nPneumonia).toBe(865);
    expect(p.weights?.[0]).toBeCloseTo(1220 / (2 * 355), 10);
    expect(p.weights?.[1]).toBeCloseTo(1220 / (2 * 865), 10);
  });

  it('augmentation triples the rows and keeps the class ratio', () => {
    const p = buildTrainingPlan(real, { ...DEFAULT_TRAIN_SETTINGS, augmentation: true });
    expect(AUG_VIEWS).toEqual([0, 1]);
    expect(p.nRows).toBe(3060);
    expect(p.nNormal).toBe(255 * 3);
    expect(p.nPneumonia).toBe(765 * 3);
    const labels = trainingRowLabels(p);
    expect(labels).toHaveLength(3060);
    expect(labels.slice(0, 1020)).toEqual(p.baseLabels);
    // augmented rows are id-major: both views of image 0 first
    expect(labels.slice(1020, 1022)).toEqual([p.baseLabels[0], p.baseLabels[0]]);
    expect(labels.reduce((a, b) => a + b, 0)).toBe(p.nPneumonia);
  });

  it('official validation set is the 16 images', () => {
    const p = buildTrainingPlan(real, { ...DEFAULT_TRAIN_SETTINGS, valSet: 'val_official' });
    expect(p.valIds).toHaveLength(16);
    expect(p.valLabels.reduce((a, b) => a + b, 0)).toBe(8);
  });
});

describe('evaluation sets', () => {
  it('map to the right ids', () => {
    const [kermany, adult, stress] = EVAL_SETS;
    expect(kermany?.id).toBe('kermany_test');
    expect(evalSetIds(real, kermany!)).toHaveLength(624);
    expect(evalSetIds(real, adult!)).toHaveLength(200);
    expect(adult?.negativeName).toBe('no finding');
    expect(stress?.variant).toBe('stress');
    expect(evalSetIds(real, stress!)).toEqual(evalSetIds(real, kermany!));
    expect(kermany?.variant).toBe('base');
  });
});

describe('chart maths and formatting', () => {
  it('niceTicks cover the range with round steps', () => {
    expect(niceTicks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    const t = niceTicks(0.013, 0.69, 5);
    expect(t[0]).toBeLessThanOrEqual(0.013);
    expect(t[t.length - 1]).toBeGreaterThanOrEqual(0.69);
    expect(niceTicks(5, 5).length).toBeGreaterThan(1); // degenerate range still gives an axis
    expect(niceTicks(Number.NaN, 3)).toEqual(niceTicks(0, 1));
    expect(niceTicks(0, 20, 5)).toEqual([0, 5, 10, 15, 20]);
  });

  it('scaleLinear maps and survives an empty domain', () => {
    expect(scaleLinear(5, 0, 10, 0, 100)).toBe(50);
    expect(scaleLinear(1, 2, 2, 7, 9)).toBe(7);
  });

  it('linePath skips non-finite values', () => {
    expect(linePath([1, 2], (i) => i * 10, (v) => v * 5)).toBe('M10.0 5.0 L20.0 10.0');
    expect(linePath([Number.NaN, 2], (i) => i * 10, (v) => v)).toBe('M20.0 2.0');
    expect(linePath([], (i) => i, (v) => v)).toBe('');
  });

  it('learning rate slider is log-scaled and round-trips', () => {
    expect(lrFromSlider(-3)).toBe(0.001);
    expect(lrFromSlider(-4)).toBe(0.0001);
    expect(lrFromSlider(-1)).toBe(0.1);
    expect(sliderFromLr(0.001)).toBeCloseTo(-3, 10);
    expect(formatLr(0.001)).toBe('1.0e-3');
    expect(formatLr(0.0003)).toBe('3.0e-4');
  });

  it('formats durations', () => {
    expect(formatDuration(3400)).toBe('3.4 s');
    expect(formatDuration(125000)).toBe('2 min 5 s');
    expect(formatDuration(-1)).toBe('-');
  });
});
