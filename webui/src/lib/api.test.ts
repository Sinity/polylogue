import { afterEach, describe, expect, it, vi } from 'vitest';

import { PolylogueClient } from '../api/generated';
import { fetchSessionListPage, fetchSessionMessagesPage } from './api';

afterEach(() => vi.restoreAllMocks());

describe('generated session request adapters', () => {
  it('passes list filters and offset to the declared client operation', async () => {
    vi.spyOn(PolylogueClient.prototype, 'bootstrapWebCredential').mockResolvedValue({
      ok: true,
      credential: { expires_at: '2099-09-26T00:00:00Z', scopes: ['read'] },
    } as never);
    const search = vi.spyOn(PolylogueClient.prototype, 'searchSessions').mockResolvedValue({
      items: [], total: 0, limit: 20, offset: 40,
    } as never);

    const page = await fetchSessionListPage({ origin: 'codex-session', repo: '/example' }, 40);

    expect(search).toHaveBeenCalledWith({ origin: 'codex-session', repo: '/example', limit: 20, offset: 40 });
    expect(page.offset).toBe(40);
  });

  it('sends a named message anchor without a competing offset', async () => {
    vi.spyOn(PolylogueClient.prototype, 'bootstrapWebCredential').mockResolvedValue({
      ok: true,
      credential: { expires_at: '2099-09-26T00:00:00Z', scopes: ['read'] },
    } as never);
    const read = vi.spyOn(PolylogueClient.prototype, 'readSessionView').mockResolvedValue({
      payload: { messages: [], total: 0, limit: 30, offset: 90 },
    } as never);

    const page = await fetchSessionMessagesPage('codex-session:example', { around: 'message:deep' });

    expect(read).toHaveBeenCalledWith({
      session_id: 'codex-session:example', view: 'messages', limit: 30, around: 'message:deep',
    });
    expect(page.offset).toBe(90);
  });

  it('renews the cached web credential before its declared expiry', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2030-01-01T00:00:00.000Z'));
    vi.resetModules();
    const [{ ensureWebCredential }, { PolylogueClient: FreshPolylogueClient }] = await Promise.all([
      import('./api'),
      import('../api/generated'),
    ]);
    const firstExpiry = new Date(Date.now() + 300_000).toISOString();
    const bootstrap = vi.spyOn(FreshPolylogueClient.prototype, 'bootstrapWebCredential')
      .mockResolvedValueOnce({ ok: true, credential: { expires_at: firstExpiry, scopes: ['read'] } } as never)
      .mockResolvedValueOnce({ ok: true, credential: { expires_at: new Date(Date.now() + 570_000).toISOString(), scopes: ['read'] } } as never);

    try {
      await ensureWebCredential();
      await ensureWebCredential();
      expect(bootstrap).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(270_001);
      await Promise.all([ensureWebCredential(), ensureWebCredential()]);
      expect(bootstrap).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('bootstraps again when a healthy daemon has forgotten the cached credential', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2030-01-01T00:00:00.000Z'));
    vi.resetModules();
    const [{ withWebCredential }, { PolylogueClient: FreshPolylogueClient }, { DaemonHttpError: FreshDaemonHttpError }] = await Promise.all([
      import('./api'),
      import('../api/generated'),
      import('../api/runtime'),
    ]);
    const bootstrap = vi.spyOn(FreshPolylogueClient.prototype, 'bootstrapWebCredential')
      .mockResolvedValue({
        ok: true,
        credential: { expires_at: new Date(Date.now() + 300_000).toISOString(), scopes: ['read'] },
      } as never);
    const request = vi.fn()
      .mockRejectedValueOnce(new FreshDaemonHttpError(
        new Response(null, { status: 401, statusText: 'Unauthorized' }),
        { error: 'web_credential_invalid' },
        { code: 'web_credential_invalid', detail: null, field: null },
      ))
      .mockResolvedValueOnce('recovered');

    try {
      await expect(withWebCredential(request)).resolves.toBe('recovered');
      expect(request).toHaveBeenCalledTimes(2);
      expect(bootstrap).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});
