import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';
import { PolylogueClient, type QueryUnitEnvelope } from '../api/generated';
import type { ClientRequest, ClientTransport } from '../api/runtime';
import { ArchiveOverviewIsland, loadArchiveMessagePage } from './archive-overview';

const page: QueryUnitEnvelope = {
  mode: 'query-unit',
  unit: 'message',
  query: 'messages where words >= 0 | sort by time desc',
  items: [
    {
      unit: 'message',
      message_id: 'message:2',
      session_id: 'codex-session:session/2',
      origin: 'codex-session',
      title: 'Continuation contract wiring',
      role: 'assistant',
      message_type: 'message',
      material_origin: 'assistant_authored',
      occurred_at_ms: Date.UTC(2026, 6, 17, 12, 0, 0),
      position: 1,
      word_count: 42,
      text: 'The opaque continuation is replayed without reconstructing filters in the browser.',
    },
  ],
  total: 1,
  limit: 1,
  offset: 1,
  next_offset: null,
  query_ref: 'query:overview',
  result_ref: 'result:overview',
  continuation: null,
  outcome: { state: 'ok' },
};

describe('ArchiveOverviewIsland', () => {
  it('uses the generated operation with the daemon continuation unchanged', async () => {
    const requests: ClientRequest[] = [];
    const transport: ClientTransport = {
      async request<TResponse>(request: ClientRequest): Promise<TResponse> {
        requests.push(request);
        return page as TResponse;
      },
    };
    vi.spyOn(PolylogueClient.prototype, 'bootstrapWebCredential').mockResolvedValue({} as never);
    try {
      const client = new PolylogueClient(transport);
      const first = await loadArchiveMessagePage(undefined, client);
      const next = await loadArchiveMessagePage('q2.opaque/token', client);
      expect(first).toBe(page);
      expect(next).toBe(page);
      expect(requests[0]?.query).toMatchObject({
        expression: 'messages where words >= 0 | sort by time desc',
        limit: 6,
      });
      expect(requests[1]?.query).toMatchObject({ continuation: 'q2.opaque/token' });
      expect(requests[1]?.query?.expression).toBeUndefined();
    } finally {
      vi.restoreAllMocks();
    }
  });

  it('loads the next typed page using the opaque continuation', async () => {
    const loadPage = vi.fn(async () => page);
    render(<ArchiveOverviewIsland initialContinuation="q1.opaque-token" loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more activity' }));

    expect(await screen.findByRole('link', { name: 'Continuation contract wiring' })).toHaveAttribute(
      'href',
      '/sessions/codex-session%3Asession%2F2#msg-message%3A2',
    );
    expect(loadPage).toHaveBeenCalledTimes(1);
    expect(loadPage).toHaveBeenCalledWith('q1.opaque-token');
    expect(screen.getByRole('button', { name: 'All activity loaded' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Loaded 1 additional record.');
  });
});
