import { test as base, expect } from '@playwright/test';

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);

/** True for URLs the browser handles itself (no network): data:, blob:, about:. */
function isLocalRequest(url: string): boolean {
  try {
    const u = new URL(url);
    if (u.protocol === 'data:' || u.protocol === 'blob:' || u.protocol === 'about:') return true;
    return LOCAL_HOSTS.has(u.hostname);
  } catch {
    return false;
  }
}

/**
 * Shared fixture: every test fails if the page requests anything from a
 * non-localhost origin (no CDNs, no analytics, no remote fonts).
 * Blocked requests are aborted so they cannot slow the test, then reported.
 */
export const test = base.extend<{ externalRequests: string[] }>({
  externalRequests: [
    async ({ page }, use) => {
      const external: string[] = [];
      await page.route(
        (url) => !isLocalRequest(url.toString()),
        (route) => {
          external.push(route.request().url());
          return route.abort();
        },
      );
      page.on('request', (req) => {
        if (!isLocalRequest(req.url()) && !external.includes(req.url())) external.push(req.url());
      });
      await use(external);
      expect(external, 'requests to non-localhost origins').toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
