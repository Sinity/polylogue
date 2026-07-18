import { useState } from 'preact/hooks';
import type { SessionListFilters } from '../lib/api';
import type { SessionListRow } from '../contracts/session-list';
import { fetchSessionListPage } from '../lib/api';

type PageLoader = (filters: SessionListFilters, offset: number) => Promise<{
  items: readonly SessionListRow[];
  total: number;
}>;

export interface SessionListIslandProps {
  readonly filters: SessionListFilters;
  readonly initialNextOffset?: number | null;
  readonly loadPage?: PageLoader;
}

function sessionDate(date: string | null): string {
  if (date === null) {
    return 'Time unavailable';
  }
  return date;
}

function SessionCard({ row }: { readonly row: SessionListRow }) {
  return (
    <li class="activity-row" data-session-id={row.id}>
      <div class="activity-row__meta">
        <span class="activity-row__origin">{row.origin}</span>
        <span>{sessionDate(row.date)}</span>
      </div>
      <h3>
        <a href={`/app/sessions/${encodeURIComponent(row.id)}`}>{row.title || row.id}</a>
      </h3>
      <p class="activity-row__detail">
        {row.message_count.toLocaleString()} messages · {row.word_count.toLocaleString()} words ·{' '}
        {row.repo ?? 'repo unknown'}
      </p>
    </li>
  );
}

export function SessionListIsland({
  filters,
  initialNextOffset,
  loadPage = fetchSessionListPage,
}: SessionListIslandProps) {
  const [nextOffset, setNextOffset] = useState<number | null | undefined>(initialNextOffset);
  const [rows, setRows] = useState<readonly SessionListRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  const exhausted = nextOffset === null || nextOffset === undefined;
  const buttonLabel = exhausted ? 'All matching sessions loaded' : 'Load more sessions';

  async function loadNextPage(): Promise<void> {
    if (loading || exhausted) {
      return;
    }
    setLoading(true);
    setStatus('Loading sessions…');
    try {
      const page = await loadPage(filters, nextOffset);
      setRows((current) => [...current, ...page.items]);
      const reachedEnd = nextOffset + page.items.length >= page.total || page.items.length === 0;
      setNextOffset(reachedEnd ? null : nextOffset + page.items.length);
      setStatus(
        page.items.length === 0
          ? 'No additional sessions found.'
          : `Loaded ${page.items.length.toLocaleString()} additional ${page.items.length === 1 ? 'session' : 'sessions'}.`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Sessions could not be loaded.');
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
        aria-controls="session-list-more"
        aria-busy={loading}
        onClick={() => void loadNextPage()}
      >
        {loading ? 'Loading…' : buttonLabel}
      </button>
      <p class="island-status" role="status" aria-live="polite">
        {status}
      </p>
      <ol id="session-list-more" class="activity-list activity-list--continued" aria-label="Additional matching sessions">
        {rows.map((row) => (
          <SessionCard key={row.id} row={row} />
        ))}
      </ol>
    </>
  );
}
