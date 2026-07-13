import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import readline from 'node:readline';

import { APIRequestContext, BrowserContext, Page, expect, request, test } from '@playwright/test';

const COOKIE_NAME = 'polylogue_web_credential';
const repoRoot = path.resolve(process.cwd(), '..');

type ReadyReceipt = {
  kind: 'ready';
  base_url: string;
  session_count: number;
  message_count: number;
  credential_ttl_s: number;
};

let server: ChildProcessWithoutNullStreams;
let receipt: ReadyReceipt;
let tempRoot = '';
const serverOutput: string[] = [];

async function startServer(): Promise<ReadyReceipt> {
  const scratchRoot = existsSync('/realm/tmp') ? '/realm/tmp' : tmpdir();
  tempRoot = await mkdtemp(path.join(scratchRoot, 'polylogue-web-auth-'));
  server = spawn('uv', ['run', 'python', 'tests/browser/web_auth_server.py'], {
    cwd: repoRoot,
    env: {
      ...process.env,
      POLYLOGUE_BROWSER_TEST_ARCHIVE_ROOT: path.join(tempRoot, 'archive'),
      POLYLOGUE_BROWSER_TEST_CREDENTIAL_TTL_S: '4',
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  server.stdout.setEncoding('utf8');
  server.stderr.setEncoding('utf8');
  server.stderr.on('data', (chunk: string) => serverOutput.push(chunk));

  return await new Promise<ReadyReceipt>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`browser fixture timed out: ${serverOutput.join('')}`)), 60_000);
    const lines = readline.createInterface({ input: server.stdout });
    lines.on('line', (line) => {
      serverOutput.push(line);
      let payload: unknown;
      try { payload = JSON.parse(line); } catch { return; }
      if (typeof payload === 'object' && payload !== null && (payload as { kind?: string }).kind === 'ready') {
        clearTimeout(timer);
        resolve(payload as ReadyReceipt);
      }
    });
    server.once('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`browser fixture exited ${code}: ${serverOutput.join('')}`));
    });
  });
}

async function stopServer(): Promise<void> {
  if (server && server.exitCode === null) {
    server.kill('SIGTERM');
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, 5_000);
      server.once('exit', () => { clearTimeout(timer); resolve(); });
    });
  }
  if (tempRoot) await rm(tempRoot, { recursive: true, force: true });
}

async function credential(context: BrowserContext): Promise<string> {
  const cookies = await context.cookies(receipt.base_url);
  const cookie = cookies.find((item) => item.name === COOKIE_NAME);
  expect(cookie, 'HttpOnly first-party credential cookie').toBeTruthy();
  expect(cookie?.httpOnly).toBe(true);
  expect(cookie?.sameSite).toBe('Strict');
  return cookie?.value ?? '';
}

async function directBrowserFetch(page: Page, route: string): Promise<{ status: number; body: Record<string, unknown> }> {
  return await page.evaluate(async (target) => {
    const response = await fetch(target, {
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'X-Polylogue-Web-Client': '1' },
    });
    return { status: response.status, body: await response.json() as Record<string, unknown> };
  }, route);
}

test.beforeAll(async () => {
  receipt = await startServer();
  expect(receipt.session_count).toBeGreaterThan(0);
  expect(receipt.message_count).toBeGreaterThan(0);
});

test.afterAll(stopServer);

test.describe.serial('first-party daemon credentials', () => {
  test('cockpit aggregates drive the bounded reader journey', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(receipt.base_url, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('html')).toHaveAttribute('data-web-auth-state', 'ready');

    const overview = await directBrowserFetch(page, '/api/overview');
    expect(overview).toMatchObject({ status: 200, body: { mode: 'cockpit-overview', recent_limit: 6 } });
    await expect(page.locator('#msg-list')).toContainText('Keep the receipts for AI work.');

    await page.locator('.conv-item').first().click();
    await expect(page.locator('#msg-list [data-msg-id]').first()).toBeVisible();
    const selectedId = await page.locator('.conv-item.selected').getAttribute('data-id');
    expect(selectedId).toBeTruthy();

    const evidence = await directBrowserFetch(page, `/api/sessions/${encodeURIComponent(selectedId ?? '')}/evidence-summary`);
    expect(evidence).toMatchObject({ status: 200, body: { mode: 'session-evidence-summary' } });
    await expect(page.locator('.evidence-strip')).toBeVisible();
    await context.close();
  });

  test('list, read, mutation, and SSE reconnect share a non-leaking credential', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    const consoleLines: string[] = [];
    const navigations: string[] = [];
    const networkMetadata: string[] = [];
    const issuedSecrets = new Set<string>();
    const responseProbes: Promise<void>[] = [];
    let eventRequests = 0;

    page.on('console', (message) => consoleLines.push(message.text()));
    page.on('framenavigated', (frame) => {
      if (frame === page.mainFrame()) navigations.push(frame.url());
    });
    page.on('request', (webRequest) => {
      const headers = webRequest.headers();
      const visibleHeaders = Object.entries(headers)
        .filter(([name]) => name.toLowerCase() !== 'cookie')
        .map(([name, value]) => `${name}:${value}`)
        .join('\n');
      networkMetadata.push([
        webRequest.method(),
        webRequest.resourceType(),
        webRequest.url(),
        headers.referer ?? '',
        visibleHeaders,
      ].join('\n'));
    });
    page.on('response', (response) => {
      if (!response.url().endsWith('/api/web-auth/session')) return;
      responseProbes.push((async () => {
        const setCookie = await response.headerValue('set-cookie');
        const match = setCookie?.match(new RegExp(`${COOKIE_NAME}=([^;]+)`));
        if (match?.[1]) issuedSecrets.add(match[1]);
      })());
    });
    await page.route('**/api/events?**', async (route) => {
      eventRequests += 1;
      if (eventRequests === 1) await route.abort('connectionfailed');
      else await route.continue();
    });

    await page.goto(receipt.base_url, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('html')).toHaveAttribute('data-web-auth-state', 'ready');
    await expect(page.locator('.conv-item').first()).toBeVisible();
    await page.locator('.conv-item').first().click();
    await expect(page.locator('#msg-list [data-msg-id]').first()).toBeVisible();
    await expect(page.locator('#conv-header h2')).not.toHaveText('Polylogue');

    const selectedId = await page.locator('.conv-item.selected').getAttribute('data-id');
    expect(selectedId).toBeTruthy();
    const star = page.locator('button[title="Toggle star"]');
    const wasStarred = (await star.getAttribute('class'))?.split(/\s+/).includes('active') ?? false;
    await star.click();
    if (wasStarred) await expect(star).not.toHaveClass(/active/);
    else await expect(star).toHaveClass(/active/);
    const marks = await directBrowserFetch(page, '/api/user/marks');
    expect(marks.status).toBe(200);
    expect((marks.body.items as Array<{ session_id: string; mark_type: string }>).some(
      (mark) => mark.session_id === selectedId && mark.mark_type === 'star',
    )).toBe(!wasStarred);

    await expect.poll(() => eventRequests, { timeout: 12_000 }).toBeGreaterThanOrEqual(2);
    await expect(page.locator('#status-live')).toHaveClass(/accent/);
    await page.goBack();
    await expect(page).toHaveURL(receipt.base_url + '/');
    await page.goForward();
    await expect(page.locator('#msg-list [data-msg-id]').first()).toBeVisible();

    issuedSecrets.add(await credential(context));
    await Promise.all(responseProbes);
    const browserState = await page.evaluate(() => JSON.stringify({
      url: location.href,
      history: history.state,
      dom: document.documentElement.outerHTML,
      resources: performance.getEntriesByType('resource').map((entry) => entry.name),
      navigation: performance.getEntriesByType('navigation').map((entry) => entry.name),
    }));
    const screenshot = await page.screenshot();
    const observableText = [
      browserState,
      consoleLines.join('\n'),
      navigations.join('\n'),
      networkMetadata.join('\n'),
      serverOutput.join('\n'),
    ].join('\n');
    for (const secret of issuedSecrets) {
      expect(secret.length).toBeGreaterThan(20);
      expect(observableText).not.toContain(secret);
      expect(screenshot.includes(Buffer.from(secret))).toBe(false);
    }
    expect(networkMetadata.join('\n')).not.toContain('access_token');
    await context.close();
  });

  test('missing, expired, revoked, and wrong-origin states are explicit', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(receipt.base_url, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('html')).toHaveAttribute('data-web-auth-state', 'ready');

    const initial = await credential(context);
    await context.clearCookies();
    const missing = await directBrowserFetch(page, '/api/sessions?limit=1');
    expect(missing).toMatchObject({ status: 401, body: { error: 'web_credential_missing' } });

    await context.addCookies([{ name: COOKIE_NAME, value: initial, url: receipt.base_url, httpOnly: true, sameSite: 'Strict' }]);
    const expiresAt = await page.evaluate(() => Date.parse(
      (window as unknown as { state: { webAuth: { expiresAt: string } } }).state.webAuth.expiresAt,
    ));
    await page.waitForTimeout(Math.max(0, expiresAt - Date.now() + 150));
    const expired = await directBrowserFetch(page, '/api/sessions?limit=1');
    expect(expired).toMatchObject({ status: 401, body: { error: 'web_credential_expired' } });

    await page.evaluate(async () => {
      const shell = window as unknown as {
        requestJSON: (route: string) => Promise<unknown>;
        state: { webAuth: { expiresAt: string } };
      };
      shell.state.webAuth.expiresAt = '';
      await shell.requestJSON('/api/sessions?limit=1');
    });
    await expect(page.locator('html')).toHaveAttribute('data-web-auth-state', 'ready');
    const active = await credential(context);
    const revoke = await page.evaluate(async () => {
      const response = await fetch('/api/web-auth/session', {
        method: 'DELETE', credentials: 'same-origin', headers: { 'X-Polylogue-Web-Client': '1' },
      });
      return { status: response.status, body: await response.json() as Record<string, unknown> };
    });
    expect(revoke).toMatchObject({ status: 200, body: { credential: { state: 'web_credential_revoked' } } });

    await context.addCookies([{ name: COOKIE_NAME, value: active, url: receipt.base_url, httpOnly: true, sameSite: 'Strict' }]);
    const revoked = await directBrowserFetch(page, '/api/sessions?limit=1');
    expect(revoked).toMatchObject({ status: 401, body: { error: 'web_credential_revoked' } });

    await page.evaluate(async () => {
      await (window as unknown as { bootstrapWebCredential: () => Promise<unknown> }).bootstrapWebCredential();
    });
    const valid = await credential(context);
    const list = await directBrowserFetch(page, '/api/sessions?limit=1');
    const sessions = (list.body.sessions ?? list.body.items ?? []) as Array<{ id: string }>;
    const target = sessions[0].id;
    let foreign: APIRequestContext | undefined;
    try {
      foreign = await request.newContext({
        extraHTTPHeaders: {
          Origin: 'http://127.0.0.1:1',
          Cookie: `${COOKIE_NAME}=${valid}`,
          'Content-Type': 'application/json',
          'X-Polylogue-Web-Client': '1',
        },
      });
      const response = await foreign.post(`${receipt.base_url}/api/user/marks`, {
        data: { session_id: target, mark_type: 'pin' },
      });
      expect(response.status()).toBe(403);
      expect(await response.json()).toMatchObject({ error: 'web_credential_wrong_origin' });
    } finally {
      await foreign?.dispose();
    }
    await context.close();
  });
});
