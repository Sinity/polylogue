import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';
import { PolylogueClient, type SearchPage, type SessionSearchHitPayload } from '../api/generated';
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
  it('uses the generated search iterator with the server-provided initial cursor', async () => {
    const generatedPage = page([hit], null);
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

  it('renders no-more-pages state without a loader call when no cursor is set', () => {
    const loadPage = vi.fn();
    render(<SearchIsland query="anything" initialCursor={null} loadPage={loadPage} />);

    expect(screen.getByRole('button', { name: 'All matching results loaded' })).toBeDisabled();
    expect(loadPage).not.toHaveBeenCalled();
  });
});
