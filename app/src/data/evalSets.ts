// The test sets the Evaluate tab can score. More sets (uploads, webcam, ...) are added to this list later.
import type { Variant } from './embeddings';
import { idsBySplit, type ManifestEntry, type Source, type Split } from './manifest';

export type EvalSetId = 'kermany_test' | 'adult_test' | 'stress';

/** What the Evaluate tab can be pointed at: a shipped set or the student's own uploads. */
export type SetChoice = EvalSetId | 'uploads';
export const UPLOADS_LABEL = 'Your uploaded images';

export interface EvalSetDef {
  id: EvalSetId;
  label: string;
  description: string;
  source: Source;
  split: Split;
  /** Which feature variant to score. */
  variant: Variant;
  /** What the negative class is called for this set. */
  negativeName: 'normal' | 'no finding';
}

export const EVAL_SETS: readonly EvalSetDef[] = [
  {
    id: 'kermany_test',
    label: 'Kermany test',
    description: 'Children, same hospital as the training images (624 X-rays, 62.5% pneumonia).',
    source: 'kermany',
    split: 'test',
    variant: 'base',
    negativeName: 'normal',
  },
  {
    id: 'adult_test',
    label: 'Adult test',
    description: 'Adults from another hospital (200 X-rays, 50% pneumonia).',
    source: 'nih',
    split: 'test_adult',
    variant: 'base',
    negativeName: 'no finding',
  },
  {
    id: 'stress',
    label: 'Kermany stress',
    description: 'The same 624 test X-rays, darker, lower contrast, rotated and cropped.',
    source: 'kermany',
    split: 'test',
    variant: 'stress',
    negativeName: 'normal',
  },
];

export function evalSetDef(id: EvalSetId): EvalSetDef {
  return EVAL_SETS.find((s) => s.id === id) ?? (EVAL_SETS[0] as EvalSetDef);
}

export function evalSetIds(entries: readonly ManifestEntry[], def: EvalSetDef): string[] {
  return idsBySplit(entries, def.source, def.split);
}
