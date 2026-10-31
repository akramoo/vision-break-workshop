// Publish app/dist to the gh-pages branch and ask GitHub to rebuild Pages.
// Used instead of the Actions workflow while Actions is unavailable on the account.
//   node scripts/deploy_pages.mjs            build, then publish
//   node scripts/deploy_pages.mjs --no-build publish the existing app/dist
import { execFileSync } from 'node:child_process';
import { cpSync, mkdtempSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const appDir = join(root, 'app');
const dist = join(appDir, 'dist');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';

const run = (cmd, args, opts = {}) =>
  execFileSync(cmd, args, { stdio: 'inherit', shell: cmd === npm && process.platform === 'win32', ...opts });
const out = (cmd, args, opts = {}) => execFileSync(cmd, args, { encoding: 'utf8', ...opts }).trim();

const remote = out('git', ['remote', 'get-url', 'origin'], { cwd: root });
const slug = remote.replace(/^.*github\.com[:/]/, '').replace(/\.git$/, '');
const head = out('git', ['rev-parse', '--short', 'HEAD'], { cwd: root });

if (!process.argv.includes('--no-build')) run(npm, ['run', 'build'], { cwd: appDir });
if (!existsSync(join(dist, 'index.html'))) throw new Error('app/dist/index.html not found; build first');

const tmp = mkdtempSync(join(tmpdir(), 'pages-'));
try {
  cpSync(dist, tmp, { recursive: true });
  writeFileSync(join(tmp, '.nojekyll'), '');
  const git = (...args) => run('git', ['-c', 'core.autocrlf=false', ...args], { cwd: tmp });
  git('init', '-q', '-b', 'gh-pages');
  git('add', '-A');
  git('-c', `user.name=${out('git', ['config', 'user.name'], { cwd: root })}`,
      '-c', `user.email=${out('git', ['config', 'user.email'], { cwd: root })}`,
      'commit', '-q', '-m', `Deploy ${head}`);
  git('push', '-q', '--force', remote, 'gh-pages');
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

run('gh', ['api', '-X', 'POST', `repos/${slug}/pages/builds`, '--silent']);
console.log(`Published ${head}. It goes live at https://${slug.split('/')[0]}.github.io/${slug.split('/')[1]}/ in about a minute.`);
