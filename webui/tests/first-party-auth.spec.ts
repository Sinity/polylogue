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
  const scratchRoot = existsSync('/realm/tmp/work') ? '/realm/tmp/work' : tmpdir();
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

async function issueCredential(page: Page): Promise<void> {
  const result = await page.evaluate(async () => {
    const response = await fetch('/api/web-auth/session', {
      method: 'POST',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'X-Polylogue-Web-Client': '1' },
    });
    return { status: response.status, body: await response.json() as Record<string, unknown> };
  });
  expect(result).toMatchObject({ status: 201, body: { credential: { state: 'ready' } } });
}

test.beforeAll(async () => {
  receipt = await startServer();
  expect(receipt.session_count).toBeGreaterThan(0);
  expect(receipt.message_count).toBeGreaterThan(0);
});

test.afterAll(stopServer);

test.describe.serial('first-party daemon credentials', () => {
  test('the typed overview bootstraps its credential before an island read', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    const bootstrap = page.waitForResponse((response) => response.url().endsWith('/api/web-auth/session'));

    await page.goto(receipt.base_url, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('#archive-activity-list')).toContainText('Keep the receipts for AI work.');
    await page.locator('#archive-overview-island .load-more').click();
    expect((await bootstrap).status()).toBe(201);
    await expect(page.locator('#archive-activity-more')).toBeVisible();
    await credential(context);
    await context.close();
  });

  test('missing, expired, revoked, and wrong-origin credentials are explicit', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    const issuedSecrets = new Set<string>();
    await page.goto(receipt.base_url, { waitUntil: 'domcontentloaded' });

    const missing = await directBrowserFetch(page, '/api/sessions?limit=1');
    expect(missing).toMatchObject({ status: 401, body: { error: 'web_credential_missing' } });

    await issueCredential(page);
    const initial = await credential(context);
    issuedSecrets.add(initial);
    const active = await directBrowserFetch(page, '/api/sessions?limit=1');
    expect(active.status).toBe(200);

    await page.waitForTimeout(receipt.credential_ttl_s * 1_000 + 150);
    const expired = await directBrowserFetch(page, '/api/sessions?limit=1');
    expect(expired).toMatchObject({ status: 401, body: { error: 'web_credential_expired' } });

    await issueCredential(page);
    const valid = await credential(context);
    issuedSecrets.add(valid);
    const revoke = await page.evaluate(async () => {
      const response = await fetch('/api/web-auth/session', {
        method: 'DELETE', credentials: 'same-origin', headers: { 'X-Polylogue-Web-Client': '1' },
      });
      return { status: response.status, body: await response.json() as Record<string, unknown> };
    });
    expect(revoke).toMatchObject({ status: 200, body: { credential: { state: 'web_credential_revoked' } } });

    await context.addCookies([{ name: COOKIE_NAME, value: valid, url: receipt.base_url, httpOnly: true, sameSite: 'Strict' }]);
    const revoked = await directBrowserFetch(page, '/api/sessions?limit=1');
    expect(revoked).toMatchObject({ status: 401, body: { error: 'web_credential_revoked' } });

    await issueCredential(page);
    const renewed = await credential(context);
    issuedSecrets.add(renewed);
    const list = await directBrowserFetch(page, '/api/sessions?limit=1');
    expect(list.status).toBe(200);
    const sessions = (list.body.sessions ?? list.body.items ?? []) as Array<{ id: string }>;
    expect(sessions).not.toHaveLength(0);

    let foreign: APIRequestContext | undefined;
    try {
      foreign = await request.newContext({
        extraHTTPHeaders: {
          Origin: 'http://127.0.0.1:1',
          Cookie: `${COOKIE_NAME}=${renewed}`,
          'Content-Type': 'application/json',
          'X-Polylogue-Web-Client': '1',
        },
      });
      const response = await foreign.post(`${receipt.base_url}/api/user/marks`, {
        data: { session_id: sessions[0]!.id, mark_type: 'pin' },
      });
      expect(response.status()).toBe(403);
      expect(await response.json()).toMatchObject({ error: 'web_credential_wrong_origin' });
    } finally {
      await foreign?.dispose();
    }

    const observableText = await page.evaluate(() => JSON.stringify({
      dom: document.documentElement.outerHTML,
      resources: performance.getEntriesByType('resource').map((entry) => entry.name),
      navigation: performance.getEntriesByType('navigation').map((entry) => entry.name),
    }));
    for (const secret of issuedSecrets) {
      expect(secret.length).toBeGreaterThan(20);
      expect(observableText).not.toContain(secret);
      expect(serverOutput.join('\n')).not.toContain(secret);
    }
    await context.close();
  });
});
