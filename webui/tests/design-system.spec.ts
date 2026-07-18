import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

import { startFixtureServer, stopFixtureServer, type FixtureServer } from './support/fixture-server';
import { expectVerticalContract } from './support/vertical-contract';

let fixture: FixtureServer;

test.beforeAll(async () => {
  fixture = await startFixtureServer();
});

test.afterAll(async () => {
  await stopFixtureServer(fixture);
});

test('list to reader to search uses the shared vertical contract', async ({ page }) => {
  const externalRequests: string[] = [];
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (url.origin !== fixture.baseUrl) externalRequests.push(request.url());
  });

  await page.goto(fixture.baseUrl, { waitUntil: 'networkidle' });
  await expectVerticalContract(page, 'webui-02', 'ready');
  await expect(page.getByRole('table', { name: /visible sanitized sessions/i })).toBeVisible();

  await page.getByRole('button', { name: 'Load final fixture row' }).click();
  await expect(page.getByRole('link', { name: 'Unknown source normalization' })).toBeVisible();
  await page.getByRole('link', { name: 'Deterministic release-note synthesis' }).click();

  await expectVerticalContract(page, 'webui-03', 'ready');
  await expect(page.getByRole('heading', { name: 'Transcript' })).toBeVisible();
  await expect(page.getByText('Four sanitized records were read from the in-memory fixture.')).toBeVisible();

  const search = page.getByRole('search', { name: 'Search from reader' });
  await search.getByRole('searchbox', { name: 'Search from reader' }).fill('accessibility');
  await search.getByRole('button', { name: 'Search' }).press('Enter');

  await expectVerticalContract(page, 'webui-04', 'ready');
  await expect(page.getByRole('link', { name: 'Accessibility audit handoff' })).toBeVisible();
  expect(externalRequests).toEqual([]);
});


const verticalRoutes = [
  { route: '/', id: 'webui-02', state: 'ready' },
  { route: '/sessions/sanitized-alpha', id: 'webui-03', state: 'ready' },
  { route: '/search?q=evidence', id: 'webui-04', state: 'ready' },
  { route: '/evidence', id: 'webui-05', state: 'degraded' },
  { route: '/timeline', id: 'webui-06', state: 'ready' },
] as const;

test('every vertical exposes the stable harness slot contract', async ({ page }) => {
  for (const vertical of verticalRoutes) {
    await page.goto(`${fixture.baseUrl}${vertical.route}`);
    await expectVerticalContract(page, vertical.id, vertical.state);
  }
});

test('keyboard-only journey reaches facets, table rows, and reader', async ({ page }) => {
  await page.goto(fixture.baseUrl);

  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to main content' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).toBeFocused();

  await page.keyboard.press('Tab');
  await expect(page.getByRole('searchbox', { name: 'Search sessions' })).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Search' })).toBeFocused();
  await page.keyboard.press('Tab');
  const exactFacet = page.locator('[data-facet-chip="exact"]');
  const qualifiedFacet = page.locator('[data-facet-chip="qualified"]');
  await expect(exactFacet).toBeFocused();
  await page.keyboard.press('ArrowRight');
  await expect(qualifiedFacet).toBeFocused();
  await page.keyboard.press('Space');
  await expect(qualifiedFacet).toHaveAttribute('aria-pressed', 'true');

  await page.keyboard.press('Tab');
  await expect(page.getByRole('region', { name: /horizontally scrollable/i })).toBeFocused();
  await page.keyboard.press('Tab');
  const firstRow = page.locator('tbody tr[data-table-row]').first();
  await expect(firstRow).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).toHaveAttribute('data-webui-vertical', 'webui-03');
});

test('SSR journey remains navigable with JavaScript disabled', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(fixture.baseUrl);

  await expect(page.locator('main')).toHaveAttribute('data-webui-vertical', 'webui-02');
  await page.getByRole('link', { name: 'Search indexing regression' }).click();
  await expect(page.locator('main')).toHaveAttribute('data-webui-vertical', 'webui-03');

  await page.getByRole('searchbox', { name: 'Search from reader' }).fill('evidence');
  await page.getByRole('button', { name: 'Search' }).press('Enter');
  await expect(page.locator('main')).toHaveAttribute('data-webui-vertical', 'webui-04');
  await expect(page.getByRole('table', { name: /results for/i })).toBeVisible();
  await context.close();
});

for (const route of ['/', '/sessions/sanitized-alpha', '/evidence', '/timeline']) {
  test(`axe scan has no violations on ${route}`, async ({ page }) => {
    await page.goto(`${fixture.baseUrl}${route}`);
    const results = await new AxeBuilder({ page }).analyze();
    expect(results.violations).toEqual([]);
  });
}

test('light theme visual contract', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light', reducedMotion: 'reduce' });
  await page.goto(`${fixture.baseUrl}/evidence`, { waitUntil: 'networkidle' });
  await expect(page).toHaveScreenshot('evidence-light.png', {
    animations: 'disabled',
    caret: 'hide',
    fullPage: true,
  });
});

test('dark theme visual contract', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.goto(`${fixture.baseUrl}/evidence`, { waitUntil: 'networkidle' });
  await expect(page).toHaveScreenshot('evidence-dark.png', {
    animations: 'disabled',
    caret: 'hide',
    fullPage: true,
  });
});
