import { useState } from 'preact/hooks';
import { PolylogueClient, type MessageQueryRowPayload, type QueryUnitEnvelope } from '../api/generated';
import { ensureWebCredential } from '../lib/api';

const PREVIEW_LIMIT = 180;
const ARCHIVE_OVERVIEW_EXPRESSION = 'messages where words >= 0 | sort by time desc';
const ARCHIVE_OVERVIEW_LIMIT = 6;
const client = new PolylogueClient();

type PageLoader = (continuation?: string) => Promise<QueryUnitEnvelope>;

export async function loadArchiveMessagePage(
  continuation?: string,
  queryClient: PolylogueClient = client,
): Promise<QueryUnitEnvelope> {
  await ensureWebCredential();
  const response = await queryClient.queryUnits(
    continuation === undefined
      ? { expression: ARCHIVE_OVERVIEW_EXPRESSION, limit: ARCHIVE_OVERVIEW_LIMIT }
      : { continuation },
  );
  if (response.mode !== 'query-unit' || response.unit !== 'message') {
    throw new TypeError('archive activity response is not a message page');
  }
  if (response.items.some((row) => row.unit !== 'message')) {
    throw new TypeError('archive activity page contains a non-message row');
  }
  return response;
}

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

function ActivityRow({ row }: { readonly row: MessageQueryRowPayload }) {
  const occurredAtMs = row.occurred_at_ms ?? null;
  return (
    <li class="activity-row" data-message-id={row.message_id}>
      <div class="activity-row__meta">
        <span class="activity-row__origin">{row.origin}</span>
        {occurredAtMs === null ? (
          <span>Time unavailable</span>
        ) : (
          <time dateTime={new Date(occurredAtMs).toISOString()}>
            {activityTimestamp(occurredAtMs)}
          </time>
        )}
      </div>
      <h3>
        <a href={`/sessions/${encodeURIComponent(row.session_id)}#msg-${encodeURIComponent(row.message_id)}`}>
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
  loadPage = loadArchiveMessagePage,
}: ArchiveOverviewIslandProps) {
  const [continuation, setContinuation] = useState<string | null | undefined>(initialContinuation);
  const [rows, setRows] = useState<readonly MessageQueryRowPayload[]>([]);
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
      if (page.unit !== 'message' || page.items.some((row) => row.unit !== 'message')) {
        throw new TypeError('archive activity page contains a non-message row');
      }
      if (page.outcome.state === 'error') {
        throw new Error(page.outcome.reason ?? 'Archive activity is unavailable.');
      }
      setRows((current) => [...current, ...(page.items as readonly MessageQueryRowPayload[])]);
      setContinuation(page.continuation);
      setStatus(
        page.outcome.state === 'degraded'
          ? (page.outcome.reason ?? 'Archive activity is incomplete.')
          : page.items.length === 0
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
