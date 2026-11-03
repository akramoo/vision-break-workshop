import { defineConfig, devices } from '@playwright/test';

const PORT = 4173;
// 127.0.0.1 (a secure context, like localhost) avoids IPv4/IPv6 resolution races between browsers and the server.
const BASE_URL = `http://127.0.0.1:${PORT}`;

// Tests run against the production build served by `vite preview`, the same
// artefact that ships to GitHub Pages. The webServer command rebuilds dist/ first, so
// tests never run against a stale build (an already-running preview server is reused locally).
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: true,
  // Every worker runs a browser with software-rendered WebGL and a MobileNet warm-up; more than a few
  // in parallel starves the CPU and model loading stalls.
  workers: process.env['CI'] ? 2 : 3,
  forbidOnly: !!process.env['CI'],
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: `npm run build && npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: BASE_URL,
    reuseExistingServer: !process.env['CI'],
    timeout: 180_000,
  },
  projects: [
    {
      name: 'chromium',
      // measure.spec.ts is the GPU dry run: it only runs on chromium-gpu. measure-compare.spec.ts is the same click path
      // timed on the software-WebGL browsers. Both are gated behind MEASURE=1 anyway.
      testIgnore: /measure\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
        },
      },
    },
    {
      // The user's installed Google Chrome with the GPU forced on (ANGLE on Direct3D 11). Playwright's bundled headless
      // chromium renders WebGL in software (SwiftShader), which says nothing about a real laptop. Only the GPU proof and
      // the dry-run measurements run here: `MEASURE=1 npx playwright test --project=chromium-gpu --workers=1 measure`.
      name: 'chromium-gpu',
      // The demo recording (tests/demo/demo.spec.ts, gated behind DEMO=1) lives outside tests/e2e, so this project looks in tests/.
      testDir: './tests',
      testMatch: /(gpu-proof|measure.*|demo)\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        launchOptions: {
          args: [
            '--use-angle=d3d11',
            '--enable-gpu',
            '--ignore-gpu-blocklist',
            '--enable-webgl',
            '--enable-gpu-rasterization',
            '--use-fake-device-for-media-stream',
            '--use-fake-ui-for-media-stream',
            // The memory soak reads performance.memory and forces a garbage collection before each sample.
            '--enable-precise-memory-info',
            '--js-flags=--expose-gc',
          ],
        },
      },
    },
    {
      name: 'firefox',
      testIgnore: /measure\.spec\.ts/,
      use: {
        ...devices['Desktop Firefox'],
        launchOptions: {
          firefoxUserPrefs: {
            'media.navigator.streams.fake': true,
            'media.navigator.permission.disabled': true,
          },
        },
      },
    },
  ],
});
