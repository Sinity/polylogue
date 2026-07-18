import { useState } from 'preact/hooks';
import type { SearchHit } from '../contracts/search';
import { fetchSearchPage } from '../lib/api';

type PageLoader = (query: string, cursor: string) => Promise<{ hits: readonly SearchHit[]; next_cursor: string | null }>;

export interface SearchIslandProps {
  readonly query: string;
  readonly initialCursor?: string | null;
  readonly loadPage?: PageLoader;
}

function SearchHitRow({ hit }: { readonly hit: SearchHit }) {
  const href = hit.match.message_id
    ? `/app/sessions/${encodeURIComponent(hit.session.id)}#msg-${encodeURIComponent(hit.match.message_id)}`
    : `/app/sessions/${encodeURIComponent(hit.session.id)}`;
  return (
    <li class="search-hit" data-session-id={hit.session.id} data-score-kind={hit.match.score_kind ?? 'unknown'}>
      <div class="search-hit__meta">
        <span class="search-hit__origin">{hit.session.origin}</span>
        {hit.match.rank !== null ? <span>rank {hit.match.rank}</span> : null}
      </div>
      <h3>
        <a href={href}>{hit.session.title}</a>
      </h3>
      {hit.match.snippet ? <p class="search-hit__snippet">{hit.match.snippet}</p> : null}
      <div class="search-hit__terms">
        {hit.match.matched_terms.map((term) => (
          <span key={term} class="search-hit__term">
            {term}
          </span>
        ))}
      </div>
    </li>
  );
}

export function SearchIsland({ query, initialCursor, loadPage = fetchSearchPage }: SearchIslandProps) {
  const [cursor, setCursor] = useState<string | null | undefined>(initialCursor);
  const [hits, setHits] = useState<readonly SearchHit[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  const exhausted = cursor === null || cursor === undefined;
  const buttonLabel = exhausted ? 'All matching results loaded' : 'Load more results';

  async function loadNextPage(): Promise<void> {
    if (loading || exhausted) {
      return;
    }
    setLoading(true);
    setStatus('Loading results…');
    try {
      const page = await loadPage(query, cursor);
      setHits((current) => [...current, ...page.hits]);
      setCursor(page.next_cursor);
      setStatus(
        page.hits.length === 0
          ? 'No additional results found.'
          : `Loaded ${page.hits.length.toLocaleString()} additional ${page.hits.length === 1 ? 'result' : 'results'}.`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Search results could not be loaded.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <button
        class="load-more"
        type="button"
        disabled={loading || exhausted}
        aria-controls="search-results-more"
        aria-busy={loading}
        onClick={() => void loadNextPage()}
      >
        {loading ? 'Loading…' : buttonLabel}
      </button>
      <p class="island-status" role="status" aria-live="polite">
        {status}
      </p>
      <ol id="search-results-more" class="search-results search-results--continued" aria-label="Additional search results">
        {hits.map((hit) => (
          <SearchHitRow key={`${hit.session.id}:${hit.match.message_id ?? ''}`} hit={hit} />
        ))}
      </ol>
    </>
  );
}
