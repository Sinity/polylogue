import { expect, test } from '@playwright/test';

import {
  startWebuiV2DemoServer,
  stopWebuiV2DemoServer,
  type WebuiV2DemoServer,
} from './support/webui-v2-demo-server';

// Exercises the shipped webui-v2 verticals (list/read/search) against the
// REAL production daemon serving a demo archive (`polylogue demo seed`
// equivalent, via `webui_v2_demo_server.py`) — never the live archive, and
// never the synthetic in-process fixture used by design-system.spec.ts.
// The demo corpus alone has no session large enough to exercise pagination,
// so the server additionally seeds one 90-message synthetic session; that
// session is what the pagination/deep-link journeys below drive (polylogue-07g6).

let server: WebuiV2DemoServer;

test.beforeAll(async () => {
  server = await startWebuiV2DemoServer();
  expect(server.demoSessionIds.length).toBeGreaterThan(0);
});

test.afterAll(async () => {
  await stopWebuiV2DemoServer(server);
});

const LARGE_SESSION_TITLE = 'webui-v2 e2e pagination fixture';

test('list renders real demo sessions served by the daemon', async ({ page }) => {
  await page.goto(`${server.baseUrl}/app/sessions`, { waitUntil: 'networkidle' });
  await expect(page.getByRole('heading', { name: 'Sessions', level: 1 })).toBeVisible();
  await expect(page.getByRole('link', { name: LARGE_SESSION_TITLE })).toBeVisible();
  const sessionLinks = page.locator('.activity-row h3 a');
  expect(await sessionLinks.count()).toBeGreaterThan(0);
});

test('opening a large session composes only the first bounded page (polylogue-07g6)', async ({ page }) => {
  await page.goto(`${server.baseUrl}/app/sessions`, { waitUntil: 'networkidle' });
  await page.getByRole('link', { name: LARGE_SESSION_TITLE }).click();

  await expect(page.getByRole('heading', { name: LARGE_SESSION_TITLE, level: 1 })).toBeVisible();
  await expect(page.getByText(`Showing 30 of ${server.largeSessionMessageCount} messages.`)).toBeVisible();
  await expect(page.locator(`[id$="webui-v2-e2e-large-session:m0"]`)).toBeVisible();
  await expect(page.locator(`[id$="webui-v2-e2e-large-session:m29"]`)).toBeVisible();
  await expect(page.locator(`[id$="webui-v2-e2e-large-session:m30"]`)).toHaveCount(0);
});

test('load more pages through the full bounded transcript', async ({ page }) => {
  await page.goto(`${server.baseUrl}/app/sessions`, { waitUntil: 'networkidle' });
  await page.getByRole('link', { name: LARGE_SESSION_TITLE }).click();
  await expect(page.locator(`[id$="webui-v2-e2e-large-session:m29"]`)).toBeVisible();

  const loadMore = page.getByRole('button', { name: 'Load more messages' });
  await loadMore.click();
  await expect(page.locator(`[id$="webui-v2-e2e-large-session:m30"]`)).toBeVisible();
  await expect(page.locator(`[id$="webui-v2-e2e-large-session:m59"]`)).toBeVisible();
  await expect(page.locator(`[id$="webui-v2-e2e-large-session:m60"]`)).toHaveCount(0);

  await loadMore.click();
  await expect(page.locator(`[id$="webui-v2-e2e-large-session:m89"]`)).toBeVisible();
  await expect(page.getByRole('button', { name: 'All messages loaded' })).toBeDisabled();
});

test('a deep link beyond the first page auto-pages to the target message', async ({ page }) => {
  const targetId = `${server.largeSessionId}:m75`;
  await page.goto(`${server.baseUrl}/app/sessions/${encodeURIComponent(server.largeSessionId)}#msg-${targetId}`, {
    waitUntil: 'domcontentloaded',
  });
  await expect(page.locator(`[id="msg-${targetId}"]`)).toBeVisible();
});

test('search finds the large session and deep-links into a specific message', async ({ page }) => {
  await page.goto(`${server.baseUrl}/app/search?q=webui-v2`, { waitUntil: 'networkidle' });
  await expect(page.locator('.search-panel')).toHaveAttribute('data-search-state', 'ok');

  const hit = page.locator(`.search-hit[data-session-id="${server.largeSessionId}"]`).first();
  await expect(hit).toBeVisible();
  const href = await hit.locator('h3 a').getAttribute('href');
  expect(href).toContain(`/app/sessions/${encodeURIComponent(server.largeSessionId)}`);
  expect(href).toContain('#msg-');

  await hit.locator('h3 a').click();
  await expect(page.getByRole('heading', { name: LARGE_SESSION_TITLE, level: 1 })).toBeVisible();
  const targetId = decodeURIComponent(href!.split('#msg-')[1]!);
  await expect(page.locator(`[id="msg-${targetId}"]`)).toBeVisible();
});

for (const viewport of [
  { width: 375, height: 812, label: '375' },
  { width: 768, height: 1024, label: '768' },
  { width: 1440, height: 900, label: '1440' },
]) {
  test(`visual: session list at ${viewport.label}px`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.emulateMedia({ colorScheme: 'light', reducedMotion: 'reduce' });
    await page.goto(`${server.baseUrl}/app/sessions`, { waitUntil: 'networkidle' });
    await expect(page).toHaveScreenshot(`session-list-${viewport.label}.png`, {
      animations: 'disabled',
      caret: 'hide',
      fullPage: true,
    });
  });

  test(`visual: session read at ${viewport.label}px`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.emulateMedia({ colorScheme: 'light', reducedMotion: 'reduce' });
    await page.goto(`${server.baseUrl}/app/sessions/${encodeURIComponent(server.largeSessionId)}`, {
      waitUntil: 'networkidle',
    });
    await expect(page).toHaveScreenshot(`session-read-${viewport.label}.png`, {
      animations: 'disabled',
      caret: 'hide',
      fullPage: true,
    });
  });

  test(`visual: search at ${viewport.label}px`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.emulateMedia({ colorScheme: 'light', reducedMotion: 'reduce' });
    await page.goto(`${server.baseUrl}/app/search?q=webui-v2`, { waitUntil: 'networkidle' });
    await expect(page).toHaveScreenshot(`search-${viewport.label}.png`, {
      animations: 'disabled',
      caret: 'hide',
      fullPage: true,
    });
  });
}
