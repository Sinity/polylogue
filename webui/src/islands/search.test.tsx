import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';
import type { SearchHit } from '../contracts/search';
import { SearchIsland } from './search';

const hit: SearchHit = {
  session: { id: 'codex-session:session/2', title: 'Continuation contract wiring', origin: 'codex-session' },
  match: {
    rank: 1,
    message_id: 'message:2',
    snippet: '...the [continuation] is replayed...',
    score_kind: 'bm25',
    matched_terms: ['continuation'],
  },
};

describe('SearchIsland', () => {
  it('loads the next page by opaque cursor and deep-links to the matching message', async () => {
    const loadPage = vi.fn(async () => ({ hits: [hit], next_cursor: null }));
    render(<SearchIsland query="continuation" initialCursor="c1.opaque-token" loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more results' }));

    expect(await screen.findByRole('link', { name: 'Continuation contract wiring' })).toHaveAttribute(
      'href',
      '/app/sessions/codex-session%3Asession%2F2#msg-message%3A2',
    );
    expect(loadPage).toHaveBeenCalledTimes(1);
    expect(loadPage).toHaveBeenCalledWith('continuation', 'c1.opaque-token');
    expect(screen.getByRole('button', { name: 'All matching results loaded' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Loaded 1 additional result.');
  });

  it('renders no-more-pages state without a loader call when no cursor is set', () => {
    const loadPage = vi.fn();
    render(<SearchIsland query="anything" initialCursor={null} loadPage={loadPage} />);

    expect(screen.getByRole('button', { name: 'All matching results loaded' })).toBeDisabled();
    expect(loadPage).not.toHaveBeenCalled();
  });
});
