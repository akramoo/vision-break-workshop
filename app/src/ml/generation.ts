// Session generation: bumped by "Reset session". Long-running jobs (training, stamp training, saving the head)
// remember the generation they started in and drop their results if it changed, so a reset can never be undone
// by a run that was still finishing.
let generation = 0;

export function currentGeneration(): number {
  return generation;
}

/** Called by Reset session. Returns the new generation. */
export function bumpGeneration(): number {
  generation += 1;
  return generation;
}
