// F8: the failure log. Pure data model + validation + persistence (no React, no DOM), so it can be unit-tested.
import type { ModelConfig } from '../ml/modelConfig';
import { DISCLAIMER } from '../ui/disclaimer';
import { APP_VERSION } from '../version';
import { LOG_KEY, readStored, safeParse, writeStored } from './persistence';

/** The seven reasons a model can fail, in the order the UI shows them. */
export const LOG_TAGS = ['imbalance', 'shortcut', 'shift', 'quality', 'closed-world', 'evaluation', 'other'] as const;
export type LogTag = (typeof LOG_TAGS)[number];

/** One-line explanation of each tag (shown next to the dropdown and in the log tab). */
export const TAG_INFO: Record<LogTag, string> = {
  imbalance: 'One class is much more common in the training data, so the model leans towards it.',
  shortcut: 'The model reacts to a marker or artefact (a stamp, a label, a corner) instead of the lungs.',
  shift: 'The image comes from a different population or machine than the training images (children vs adults).',
  quality: 'The picture is darker, lower contrast, tilted or badly framed compared with the clean training images.',
  'closed-world': 'The model must answer pneumonia or normal even when the picture shows something else (a cup, a hand).',
  evaluation: 'The score itself misleads: too few test images, the wrong metric, or test data that leaked into training.',
  other: 'None of the above, or not sure yet. Say what you think in the hypothesis.',
};

export const TRUE_LABELS = ['pneumonia', 'normal', 'no finding'] as const;
export type LabelName = (typeof TRUE_LABELS)[number];

/** Hard limits that keep 100 entries under 1 MB of localStorage. */
export const MAX_LOG_ENTRIES = 100;
export const MAX_THUMB_CHARS = 7000;
export const MAX_HYPOTHESIS_CHARS = 600;
const MAX_SET_NAME = 60;

/** Model + session settings in force when the failure was recorded. Extra keys (perturb, stamp, ...) are allowed. */
export interface LogConfig {
  mode: 'realistic' | 'balanced';
  adultMix: boolean;
  augmentation: boolean;
  epochs: number;
  lr: number;
  seed: number;
  /** Refer-to-a-doctor threshold. */
  tau: number;
  usePrecomputed: boolean;
  [extra: string]: unknown;
}

export interface LogEntry {
  id: string;
  /** Epoch milliseconds. */
  createdAt: number;
  /** 96 px JPEG data URL of the exact pixels the model saw (including stamp / perturbation). */
  thumbnail: string;
  /** Where the image came from: kermany_test, adult_test, stress, uploads, webcam, stamp, perturb (...). */
  setName: string;
  imageId?: string;
  trueLabel: LabelName | null;
  /** The hard call at p >= 0.5 (even when the app referred the case). */
  predictedLabel: LabelName;
  /** max(p, 1 - p) */
  confidence: number;
  probPneumonia: number;
  /** True if the confidence was below tau, so the app said "Refer to a doctor". */
  referred: boolean;
  config: LogConfig;
  tag: LogTag;
  hypothesis: string;
}

/** Settings snapshot for a log entry: the model config + tau + feature mode, plus context-specific extras. */
export function buildLogConfig(model: ModelConfig, tau: number, usePrecomputed: boolean, extra: Record<string, unknown> = {}): LogConfig {
  return { ...extra, ...model, tau, usePrecomputed };
}

/** What a "Log this" button supplies; the form adds the tag and hypothesis. */
export type LogDraftBase = Omit<LogEntry, 'id' | 'createdAt' | 'tag' | 'hypothesis'>;
export type LogDraft = LogDraftBase & { tag: LogTag; hypothesis: string };

export function isLogTag(v: unknown): v is LogTag {
  return typeof v === 'string' && (LOG_TAGS as readonly string[]).includes(v);
}

function isLabelName(v: unknown): v is LabelName {
  return typeof v === 'string' && (TRUE_LABELS as readonly string[]).includes(v);
}

/** Name of a class for display / export. */
export function labelName(isPneumonia: boolean, negativeName: 'normal' | 'no finding' | string = 'normal'): LabelName {
  return isPneumonia ? 'pneumonia' : negativeName === 'no finding' ? 'no finding' : 'normal';
}

let idCounter = 0;
function newId(now: number): string {
  idCounter += 1;
  return `log-${now.toString(36)}-${idCounter.toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

export function trimHypothesis(text: string): string {
  return text.trim().slice(0, MAX_HYPOTHESIS_CHARS);
}

/** Turn a draft into a full entry (new id, timestamp). */
export function makeLogEntry(draft: LogDraft, now = Date.now()): LogEntry {
  return { ...draft, id: newId(now), createdAt: now, hypothesis: trimHypothesis(draft.hypothesis) };
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function validConfig(raw: unknown): LogConfig | null {
  if (!isObject(raw)) return null;
  const { mode, adultMix, augmentation, epochs, lr, seed, tau, usePrecomputed } = raw;
  const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
  if (mode !== 'realistic' && mode !== 'balanced') return null;
  if (typeof adultMix !== 'boolean' || typeof augmentation !== 'boolean' || typeof usePrecomputed !== 'boolean') return null;
  if (!num(epochs) || !num(lr) || !num(seed) || !num(tau)) return null;
  return { ...raw, mode, adultMix, augmentation, epochs, lr, seed, tau, usePrecomputed };
}

/** Check one stored / imported entry. Returns a clean copy, or null if it is malformed. */
export function validateEntry(raw: unknown): LogEntry | null {
  if (!isObject(raw)) return null;
  const { id, createdAt, thumbnail, setName, imageId, trueLabel, predictedLabel, confidence, probPneumonia, referred, tag, hypothesis } = raw;
  if (typeof id !== 'string' || id.length === 0 || id.length > 80) return null;
  if (typeof createdAt !== 'number' || !Number.isFinite(createdAt)) return null;
  if (typeof thumbnail !== 'string' || !/^data:image\/(jpeg|png);base64,/.test(thumbnail) || thumbnail.length > MAX_THUMB_CHARS) return null;
  if (typeof setName !== 'string' || setName.length === 0 || setName.length > MAX_SET_NAME) return null;
  if (trueLabel !== null && trueLabel !== undefined && !isLabelName(trueLabel)) return null;
  if (!isLabelName(predictedLabel)) return null;
  if (typeof probPneumonia !== 'number' || !Number.isFinite(probPneumonia) || probPneumonia < 0 || probPneumonia > 1) return null;
  if (typeof confidence !== 'number' || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) return null;
  if (typeof referred !== 'boolean') return null;
  if (!isLogTag(tag)) return null;
  if (typeof hypothesis !== 'string') return null;
  const config = validConfig(raw['config']);
  if (!config) return null;
  return {
    id,
    createdAt,
    thumbnail,
    setName,
    ...(typeof imageId === 'string' && imageId.length > 0 && imageId.length <= 200 ? { imageId } : {}),
    trueLabel: trueLabel ?? null,
    predictedLabel,
    confidence,
    probPneumonia,
    referred,
    config,
    tag,
    hypothesis: trimHypothesis(hypothesis),
  };
}

/** Keep only the well-formed entries of a parsed list (duplicates by id dropped, at most MAX_LOG_ENTRIES). */
export function validateEntries(raw: unknown): LogEntry[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: LogEntry[] = [];
  for (const item of raw) {
    const e = validateEntry(item);
    if (!e || seen.has(e.id)) continue;
    seen.add(e.id);
    out.push(e);
    if (out.length >= MAX_LOG_ENTRIES) break;
  }
  return out;
}

export function tagCounts(entries: readonly { tag: LogTag }[]): Record<LogTag, number> {
  const counts = Object.fromEntries(LOG_TAGS.map((t) => [t, 0])) as Record<LogTag, number>;
  for (const e of entries) counts[e.tag] += 1;
  return counts;
}

// ---- export ----

export interface LogExport {
  disclaimer: string;
  exportedAt: string;
  appVersion: string;
  entries: LogEntry[];
}

export function buildLogExport(entries: readonly LogEntry[], now: Date = new Date(), version: string = APP_VERSION): LogExport {
  return { disclaimer: DISCLAIMER, exportedAt: now.toISOString(), appVersion: version, entries: [...entries] };
}

/** "vision-failures-20261031-1542.json" (local time). */
export function logExportFileName(now: Date = new Date()): string {
  const p = (n: number, w = 2) => String(n).padStart(w, '0');
  return `vision-failures-${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}-${p(now.getHours())}${p(now.getMinutes())}.json`;
}

// ---- persistence ----

export function loadLog(): LogEntry[] {
  return validateEntries(safeParse<unknown>(readStored(LOG_KEY), null));
}

/** Serialized size of the log in characters (all ASCII, so also bytes). */
export function estimateLogSize(entries: readonly LogEntry[]): number {
  return JSON.stringify(entries).length;
}

/** Store the log. An empty log removes the key. Returns false if the browser refused (quota exceeded, storage blocked). */
export function saveLog(entries: readonly LogEntry[]): boolean {
  return writeStored(LOG_KEY, entries.length === 0 ? null : JSON.stringify(entries));
}
