import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

interface Snapshot {
  numTensors: number;
  numBytes: number;
}

function read(): Snapshot {
  const { numTensors, numBytes } = tf.memory();
  return { numTensors, numBytes };
}

/** Shown with ?debug=1. Polls tf.memory() once per second. */
export function MemoryHud() {
  const [snap, setSnap] = useState<Snapshot>(read);
  useEffect(() => {
    const id = window.setInterval(() => setSnap(read()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return (
    <div className="memory-hud" data-testid="memory-hud">
      tensors: <span data-testid="hud-tensors">{snap.numTensors}</span> | bytes:{' '}
      <span data-testid="hud-bytes">{snap.numBytes}</span>
    </div>
  );
}
