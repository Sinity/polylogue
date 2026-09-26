import { afterEach, describe, expect, it, vi } from 'vitest';

import { PolylogueClient } from '../api/generated';
import { fetchSessionListPage, fetchSessionMessagesPage } from './api';

afterEach(() => vi.restoreAllMocks());

describe('generated session request adapters', () => {
  it('passes list filters and offset to the declared client operation', async () => {
    vi.spyOn(PolylogueClient.prototype, 'bootstrapWebCredential').mockResolvedValue({
      ok: true,
      credential: { expires_at: '2026-09-26T00:00:00Z', scopes: ['read'] },
    } as never);
    const search = vi.spyOn(PolylogueClient.prototype, 'searchSessions').mockResolvedValue({
      items: [], total: 0, limit: 20, offset: 40,
    } as never);

    const page = await fetchSessionListPage({ origin: 'codex-session', repo: '/example' }, 40);

    expect(search).toHaveBeenCalledWith({ origin: 'codex-session', repo: '/example', limit: 20, offset: 40 });
    expect(page.offset).toBe(40);
  });

  it('sends a named message anchor without a competing offset', async () => {
    vi.spyOn(PolylogueClient.prototype, 'bootstrapWebCredential').mockResolvedValue({} as never);
    const read = vi.spyOn(PolylogueClient.prototype, 'readSessionView').mockResolvedValue({
      payload: { messages: [], total: 0, limit: 30, offset: 90 },
    } as never);

    const page = await fetchSessionMessagesPage('codex-session:example', { around: 'message:deep' });

    expect(read).toHaveBeenCalledWith({
      session_id: 'codex-session:example', view: 'messages', limit: 30, around: 'message:deep',
    });
    expect(page.offset).toBe(90);
  });
});
