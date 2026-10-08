import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';
import { PolylogueClient, type SearchPage, type SessionSearchHitPayload } from '../api/generated';
import type { SessionMessageRow, SessionMessageWindow } from '../contracts/session-read';
import { SessionReadIsland } from './session-read';
import { SearchIsland } from './search';

const hit: SessionSearchHitPayload = {
  session: { id: 'codex-session:session/2', title: 'Continuation contract wiring', origin: 'codex-session' },
  match: {
    rank: 1,
    message_id: 'message:2',
    snippet: '...the [continuation] is replayed...',
    score_kind: 'bm25',
    matched_terms: ['continuation'],
    match_surface: 'message',
    retrieval_lane: 'dialogue',
  },
};

function page(
  items: readonly SessionSearchHitPayload[],
  cursor: string | null,
  coverage: SearchPage['coverage'] = { kind: 'exact', total: 12 },
): SearchPage {
  return {
    items,
    cursor,
    coverage,
    queryRef: 'query-ref',
    resultRef: 'result-ref',
    envelope: {
      hits: items,
      limit: 20,
      offset: 0,
      outcome: { state: 'ok' },
      query: 'continuation',
      retrieval_lane: 'dialogue',
      total: 12,
    },
  };
}

describe('SearchIsland', () => {
  it.each(['message:2', 'message%3Aliteral'])(
    'resolves the generated browser fragment for exact message ID %s',
    async (messageId) => {
      Element.prototype.scrollIntoView = vi.fn();
      const priorHash = window.location.hash;
      const linkedHit: SessionSearchHitPayload = { ...hit, match: { ...hit.match, message_id: messageId } };
      const searchLoader = vi.fn(async () => page([linkedHit], null));
      try {
        render(<SearchIsland query="continuation" initialCursor="next-page" loadPage={searchLoader} />);
        fireEvent.click(screen.getByRole('button', { name: 'Load more results' }));
        const href = await screen.findByRole('link', { name: 'Continuation contract wiring' }).then((link) =>
          link.getAttribute('href'),
        );
        expect(href).not.toBeNull();

        const browserHash = new URL(href!, window.location.href).hash;
        expect(browserHash).toBe(`#msg-${encodeURIComponent(messageId)}`);
        window.location.hash = browserHash;

        const message = {
          id: messageId,
          role: 'assistant',
          material_origin: 'assistant_authored',
          text: 'The exact linked message.',
          timestamp: null,
          has_tool_use: false,
          has_thinking: false,
          has_paste_evidence: false,
          semantic_entries: [],
          semantic_card_suppressed: false,
        } satisfies SessionMessageRow;
        const readLoader = vi.fn(async (_sessionId: string, _window: SessionMessageWindow) => ({
          messages: [message],
          total: 5000,
          offset: 1590,
        }));

        render(<SessionReadIsland sessionId="codex-session:session/2" initialNextOffset={30} loadPage={readLoader} />);
        await screen.findByText('The exact linked message.');

        expect(readLoader).toHaveBeenCalledTimes(1);
        expect(readLoader).toHaveBeenCalledWith('codex-session:session/2', { around: messageId });
        await vi.waitFor(() => expect(Element.prototype.scrollIntoView).toHaveBeenCalled());
      } finally {
        window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}${priorHash}`);
      }
    },
  );

  it('uses the generated search iterator with the server-provided initial cursor', async () => {
    const generatedPage = page([hit], null);
    const bootstrap = vi.spyOn(PolylogueClient.prototype, 'bootstrapWebCredential').mockResolvedValue({
      credential: { expires_at: '2099-01-01T00:00:00.000Z', scopes: ['read'] },
    });
    const search = vi.spyOn(PolylogueClient.prototype, 'search').mockReturnValue(
      (async function* () {
        yield generatedPage;
      })(),
    );
    render(<SearchIsland query="continuation" initialCursor="c1.opaque-token" />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more results' }));

    expect(await screen.findByRole('link', { name: 'Continuation contract wiring' })).toHaveAttribute(
      'href',
      '/sessions/codex-session%3Asession%2F2#msg-message%3A2',
    );
    expect(search).toHaveBeenCalledWith({ query: 'continuation', cursor: 'c1.opaque-token' });
    expect(screen.getByText('...the [continuation] is replayed...')).toBeInTheDocument();
    expect(screen.getByText('continuation', { selector: '.search-hit__term' })).toBeInTheDocument();
    expect(screen.getByText('Exact coverage: 12 matching results.')).toHaveAttribute('data-coverage-kind', 'exact');
    expect(screen.getByRole('status')).toHaveTextContent('Loaded 1 additional result.');
    search.mockRestore();
    bootstrap.mockRestore();
  });

  it('keeps qualified coverage and falls back from a missing title to the session ID', async () => {
    const { title: _title, ...sessionWithoutTitle } = hit.session;
    const untitled: SessionSearchHitPayload = {
      ...hit,
      session: sessionWithoutTitle,
    };
    const loadPage = vi.fn(async () =>
      page([untitled], null, { kind: 'qualified', total: null, qualification: 'capped' }),
    );
    render(<SearchIsland query="continuation" initialCursor="c1" loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more results' }));

    expect(await screen.findByRole('link', { name: 'codex-session:session/2' })).toBeInTheDocument();
    expect(screen.getByText('Qualified coverage: capped.')).toHaveAttribute('data-coverage-kind', 'qualified');
  });

  it('uses the untitled fallback when both title and session ID are empty', async () => {
    const { title: _title, ...sessionWithoutTitle } = hit.session;
    const untitled: SessionSearchHitPayload = {
      ...hit,
      session: { ...sessionWithoutTitle, id: '' },
    };
    const loadPage = vi.fn(async () => page([untitled], null));
    render(<SearchIsland query="continuation" initialCursor="c1" loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more results' }));

    expect(await screen.findByRole('link', { name: '[untitled session]' })).toBeInTheDocument();
  });

  it('rejects a cursor already requested before appending the repeated page', async () => {
    const duplicateHit: SessionSearchHitPayload = {
      ...hit,
      session: { ...hit.session, id: 'duplicate-session', title: 'Should not be appended' },
    };
    const loadPage = vi
      .fn<(_query: string, cursor: string) => Promise<SearchPage>>()
      .mockResolvedValueOnce(page([hit], 'c2'))
      .mockResolvedValueOnce(page([duplicateHit], 'c1'));
    render(<SearchIsland query="continuation" initialCursor="c1" loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more results' }));
    await screen.findByRole('link', { name: 'Continuation contract wiring' });
    fireEvent.click(screen.getByRole('button', { name: 'Load more results' }));

    expect(await screen.findByRole('status')).toHaveTextContent('Search continuation repeated');
    expect(loadPage).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole('link', { name: 'Should not be appended' })).not.toBeInTheDocument();
    expect(screen.getAllByRole('link')).toHaveLength(1);
  });

  it('shows loader failures to the user', async () => {
    const loadPage = vi.fn(async () => {
      throw new Error('Search request failed.');
    });
    render(<SearchIsland query="continuation" initialCursor="c1" loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more results' }));

    expect(await screen.findByRole('status')).toHaveTextContent('Search request failed.');
  });

  it('allows retrying a continuation after a transient loader failure', async () => {
    const loadPage = vi
      .fn<(_query: string, cursor: string) => Promise<SearchPage>>()
      .mockRejectedValueOnce(new Error('Search request failed.'))
      .mockResolvedValueOnce(page([hit], null));
    render(<SearchIsland query="continuation" initialCursor="c1" loadPage={loadPage} />);
    const button = screen.getByRole('button', { name: 'Load more results' });

    fireEvent.click(button);
    expect(await screen.findByRole('status')).toHaveTextContent('Search request failed.');
    fireEvent.click(button);

    expect(await screen.findByRole('link', { name: 'Continuation contract wiring' })).toBeInTheDocument();
    expect(loadPage).toHaveBeenCalledTimes(2);
    expect(loadPage).toHaveBeenLastCalledWith('continuation', 'c1');
  });

  it('renders no-more-pages state without a loader call when no cursor is set', () => {
    const loadPage = vi.fn();
    render(<SearchIsland query="anything" initialCursor={null} loadPage={loadPage} />);

    expect(screen.getByRole('button', { name: 'All matching results loaded' })).toBeDisabled();
    expect(loadPage).not.toHaveBeenCalled();
  });
});
