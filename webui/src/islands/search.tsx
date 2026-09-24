import { useRef, useState } from 'preact/hooks';
import { PolylogueClient, type SearchPage, type SessionSearchHitPayload } from '../api/generated';

type PageLoader = (query: string, cursor: string) => Promise<SearchPage>;

const client = new PolylogueClient();

async function loadSearchPage(query: string, cursor: string): Promise<SearchPage> {
  const pages = client.search({ query, cursor });
  const first = await pages[Symbol.asyncIterator]().next();
  if (first.done) {
    throw new TypeError('Search returned no page.');
  }
  return first.value;
}

export interface SearchIslandProps {
  readonly query: string;
  readonly initialCursor?: string | null;
  readonly loadPage?: PageLoader;
}

function SearchHitRow({ hit }: { readonly hit: SessionSearchHitPayload }) {
  const title = hit.session.title?.trim() || hit.session.id?.trim() || '[untitled session]';
  const href = hit.match.message_id
    ? `/sessions/${encodeURIComponent(hit.session.id)}#msg-${encodeURIComponent(hit.match.message_id)}`
    : `/sessions/${encodeURIComponent(hit.session.id)}`;
  return (
    <li class="search-hit" data-session-id={hit.session.id} data-score-kind={hit.match.score_kind ?? 'unknown'}>
      <div class="search-hit__meta">
        <span class="search-hit__origin">{hit.session.origin}</span>
        {hit.match.rank !== null ? <span>rank {hit.match.rank}</span> : null}
      </div>
      <h3>
        <a href={href}>{title}</a>
      </h3>
      {hit.match.snippet ? <p class="search-hit__snippet">{hit.match.snippet}</p> : null}
      <div class="search-hit__terms">
        {(hit.match.matched_terms ?? []).map((term) => (
          <span key={term} class="search-hit__term">
            {term}
          </span>
        ))}
      </div>
    </li>
  );
}

export function SearchIsland({ query, initialCursor, loadPage = loadSearchPage }: SearchIslandProps) {
  const [cursor, setCursor] = useState<string | null | undefined>(initialCursor);
  const [hits, setHits] = useState<readonly SessionSearchHitPayload[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [coverage, setCoverage] = useState<SearchPage['coverage'] | null>(null);
  const requestedCursors = useRef(new Set<string>());

  const exhausted = cursor === null || cursor === undefined;
  const buttonLabel = exhausted ? 'All matching results loaded' : 'Load more results';

  async function loadNextPage(): Promise<void> {
    if (loading || exhausted) {
      return;
    }
    if (requestedCursors.current.has(cursor)) {
      setCursor(null);
      setStatus('Search continuation repeated; no additional results were loaded.');
      return;
    }
    requestedCursors.current.add(cursor);
    setLoading(true);
    setStatus('Loading results…');
    try {
      const page = await loadPage(query, cursor);
      if (page.cursor !== null && requestedCursors.current.has(page.cursor)) {
        setCursor(null);
        setStatus('Search continuation repeated; no additional results were loaded.');
        return;
      }
      setHits((current) => [...current, ...page.items]);
      setCursor(page.cursor);
      setCoverage(page.coverage);
      setStatus(
        page.items.length === 0
          ? 'No additional results found.'
          : `Loaded ${page.items.length.toLocaleString()} additional ${page.items.length === 1 ? 'result' : 'results'}.`,
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
      {coverage !== null ? (
        <p class="search-coverage" data-coverage-kind={coverage.kind}>
          {coverage.kind === 'exact'
            ? `Exact coverage: ${coverage.total.toLocaleString()} matching results.`
            : `Qualified coverage: ${coverage.qualification}.`}
        </p>
      ) : null}
      <ol id="search-results-more" class="search-results search-results--continued" aria-label="Additional search results">
        {hits.map((hit) => (
          <SearchHitRow key={`${hit.session.id}:${hit.match.message_id ?? ''}`} hit={hit} />
        ))}
      </ol>
    </>
  );
}
