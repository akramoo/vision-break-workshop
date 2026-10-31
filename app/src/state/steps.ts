// The "Workshop steps" checklist (Day 1 / Day 2). Checkboxes only; nothing ticks automatically.
import { STEPS_KEY, readStored, safeParse, writeStored } from './persistence';

export interface Step {
  id: string;
  text: string;
}

export interface StepGroup {
  day: 1 | 2;
  title: string;
  steps: readonly Step[];
}

export const STEP_GROUPS: readonly StepGroup[] = [
  {
    day: 1,
    title: 'Day 1',
    steps: [
      { id: 'd1-data', text: 'Load data and read the class counts' },
      { id: 'd1-train', text: 'Train your first model' },
      { id: 'd1-baseline', text: "Compare with 'always say pneumonia'" },
      { id: 'd1-adult', text: 'Evaluate on the adult X-rays' },
      { id: 'd1-break', text: 'Break it three ways (sliders, upload, webcam on a cup)' },
      { id: 'd1-log', text: 'Log each failure with a tag' },
    ],
  },
  {
    day: 2,
    title: 'Day 2',
    steps: [
      { id: 'd2-stamp', text: 'Run the stamp experiment and record the flip rate' },
      { id: 'd2-adult', text: 'Test on the adult set and note the drop' },
      { id: 'd2-fixes', text: 'Apply two fixes (balance, adult mix, augmentation, refer threshold)' },
      { id: 'd2-remeasure', text: 'Re-measure all three sets and save a snapshot' },
      { id: 'd2-post', text: 'Post your worst-case score' },
    ],
  },
];

export const ALL_STEP_IDS: readonly string[] = STEP_GROUPS.flatMap((g) => g.steps.map((s) => s.id));

/** Keep only known step ids (each once). */
export function sanitizeSteps(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  const known = new Set(ALL_STEP_IDS);
  return [...new Set(raw.filter((v): v is string => typeof v === 'string' && known.has(v)))];
}

export function loadSteps(): string[] {
  return sanitizeSteps(safeParse<unknown>(readStored(STEPS_KEY), null));
}

/** Store the ticked ids; none removes the key. */
export function saveSteps(done: readonly string[]): boolean {
  return writeStored(STEPS_KEY, done.length === 0 ? null : JSON.stringify(done));
}

export function toggleStepId(done: readonly string[], id: string): string[] {
  if (!ALL_STEP_IDS.includes(id)) return [...done];
  return done.includes(id) ? done.filter((d) => d !== id) : [...done, id];
}
