import { useState } from 'preact/hooks';
import type { MessageQueryPage, MessageQueryRow } from '../contracts/query-units';
import { fetchArchiveMessagePage } from '../lib/api';

const PREVIEW_LIMIT = 180;

type PageLoader = (continuation?: string) => Promise<MessageQueryPage>;

export interface ArchiveOverviewIslandProps {
  readonly initialContinuation?: string | null;
  readonly loadPage?: PageLoader;
}

function compactPreview(text: string): string {
  const compact = text.replace(/\s+/g, ' ').trim();
  if (compact.length <= PREVIEW_LIMIT) {
    return compact;
  }
  return `${compact.slice(0, PREVIEW_LIMIT - 1).trimEnd()}…`;
}

function activityTimestamp(occurredAtMs: number | null): string {
  if (occurredAtMs === null) {
    return 'Time unavailable';
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(occurredAtMs));
}

function ActivityRow({ row }: { readonly row: MessageQueryRow }) {
  return (
    <li class="activity-row" data-message-id={row.message_id}>
      <div class="activity-row__meta">
        <span class="activity-row__origin">{row.origin}</span>
        {row.occurred_at_ms === null ? (
          <span>Time unavailable</span>
        ) : (
          <time dateTime={new Date(row.occurred_at_ms).toISOString()}>
            {activityTimestamp(row.occurred_at_ms)}
          </time>
        )}
      </div>
      <h3>
        <a href={`/app/sessions/${encodeURIComponent(row.session_id)}#msg-${encodeURIComponent(row.message_id)}`}>
          {row.title ?? row.session_id}
        </a>
      </h3>
      <p class="activity-row__preview">{compactPreview(row.text) || '[empty message]'}</p>
      <p class="activity-row__detail">
        {row.role} · {row.word_count.toLocaleString()} words
      </p>
    </li>
  );
}

export function ArchiveOverviewIsland({
  initialContinuation,
  loadPage = fetchArchiveMessagePage,
}: ArchiveOverviewIslandProps) {
  const [continuation, setContinuation] = useState<string | null | undefined>(initialContinuation);
  const [rows, setRows] = useState<readonly MessageQueryRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  const exhausted = continuation === null;
  const buttonLabel = exhausted
    ? 'All activity loaded'
    : continuation === undefined
      ? 'Load recent activity'
      : 'Load more activity';

  async function loadNextPage(): Promise<void> {
    if (loading || exhausted) {
      return;
    }
    setLoading(true);
    setStatus('Loading archive activity…');
    try {
      const page = await loadPage(continuation ?? undefined);
      setRows((current) => [...current, ...page.items]);
      setContinuation(page.continuation);
      setStatus(
        page.items.length === 0
          ? 'No additional activity found.'
          : `Loaded ${page.items.length.toLocaleString()} additional ${page.items.length === 1 ? 'record' : 'records'}.`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Archive activity could not be loaded.');
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
        aria-controls="archive-activity-more"
        aria-busy={loading}
        onClick={() => void loadNextPage()}
      >
        {loading ? 'Loading…' : buttonLabel}
      </button>
      <p class="island-status" role="status" aria-live="polite">
        {status}
      </p>
      <ol id="archive-activity-more" class="activity-list activity-list--continued" aria-label="Additional archive activity">
        {rows.map((row) => (
          <ActivityRow key={row.message_id} row={row} />
        ))}
      </ol>
    </>
  );
}
