import type { EvalSetId } from '../data/evalSets';

/** Which set the numbers on the Evaluate tab are measured on, and why they differ from the Train tab's validation accuracy. */
export function evalSetNote(id: EvalSetId, n: number): string {
  const name =
    id === 'stress' ? 'degraded copies of the Kermany test images' : id === 'adult_test' ? 'Adult test set' : 'Kermany test set';
  return `These numbers are on the ${name} (${n} images the model never trained on). The validation accuracy on the Train tab is measured on a different, easier set, so the two numbers differ.`;
}
