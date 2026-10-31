// Windows quirk: `npm install` can leave tensorflow.dll in tfjs-node/deps/lib/ without copying it
// next to tfjs_binding.node, so the addon fails with "The specified module could not be found".
// This copies it when (and only when) it is missing. win32 only; no-op everywhere else.
// Used as the npm "postinstall" script and imported by the scripts that load tfjs-node.
import { copyFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export function ensureTfjsNode() {
  if (process.platform !== 'win32') return false;
  const pkgDir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'node_modules', '@tensorflow', 'tfjs-node');
  const src = path.join(pkgDir, 'deps', 'lib', 'tensorflow.dll');
  const dst = path.join(pkgDir, 'lib', 'napi-v8', 'tensorflow.dll');
  if (existsSync(dst) || !existsSync(src) || !existsSync(path.dirname(dst))) return false;
  copyFileSync(src, dst);
  console.log('ensure_tfjs_node: copied tensorflow.dll next to tfjs_binding.node');
  return true;
}

// Run directly (npm postinstall): `node ensure_tfjs_node.mjs`
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) ensureTfjsNode();
