import {
  ARCHIVE_OVERVIEW_EXPRESSION,
  ARCHIVE_OVERVIEW_LIMIT,
  parseMessageQueryPage,
  type MessageQueryPage,
} from '../contracts/query-units';
import { SESSION_LIST_LIMIT, parseSessionListPage, type SessionListPage } from '../contracts/session-list';
import {
  SESSION_READ_MESSAGE_LIMIT,
  parseSessionMessagePage,
  type SessionMessagePage,
} from '../contracts/session-read';

const WEB_CLIENT_HEADERS = Object.freeze({
  'X-Polylogue-Web-Client': '1',
});

let credentialBootstrap: Promise<void> | null = null;

async function ensureWebCredential(): Promise<void> {
  if (credentialBootstrap !== null) {
    return credentialBootstrap;
  }
  credentialBootstrap = (async () => {
    const response = await fetch('/api/web-auth/session', {
      method: 'POST',
      credentials: 'same-origin',
      cache: 'no-store',
      referrerPolicy: 'no-referrer',
      headers: WEB_CLIENT_HEADERS,
    });
    if (!response.ok) {
      throw new Error(`web credential bootstrap failed (${response.status})`);
    }
  })();
  try {
    await credentialBootstrap;
  } catch (error) {
    credentialBootstrap = null;
    throw error;
  }
}

export async function requestJson(path: string): Promise<unknown> {
  await ensureWebCredential();
  const response = await fetch(path, {
    method: 'GET',
    credentials: 'same-origin',
    cache: 'no-store',
    referrerPolicy: 'no-referrer',
    headers: WEB_CLIENT_HEADERS,
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof payload === 'object' && payload !== null && 'error' in payload
        ? String((payload as { error: unknown }).error)
        : `HTTP ${response.status}`;
    throw new Error(`archive query failed: ${detail}`);
  }
  return payload;
}

export async function fetchArchiveMessagePage(
  continuation?: string,
): Promise<MessageQueryPage> {
  const params = new URLSearchParams();
  if (continuation !== undefined) {
    params.set('continuation', continuation);
  } else {
    params.set('expression', ARCHIVE_OVERVIEW_EXPRESSION);
    params.set('limit', String(ARCHIVE_OVERVIEW_LIMIT));
  }
  return parseMessageQueryPage(await requestJson(`/api/query-units?${params.toString()}`));
}

export interface SessionListFilters {
  readonly origin?: string | undefined;
  readonly since?: string | undefined;
  readonly repo?: string | undefined;
}

export async function fetchSessionListPage(
  filters: SessionListFilters,
  offset: number,
): Promise<SessionListPage> {
  const params = new URLSearchParams();
  if (filters.origin) params.set('origin', filters.origin);
  if (filters.since) params.set('since', filters.since);
  if (filters.repo) params.set('repo', filters.repo);
  params.set('limit', String(SESSION_LIST_LIMIT));
  params.set('offset', String(offset));
  return parseSessionListPage(await requestJson(`/api/sessions?${params.toString()}`));
}

export async function fetchSessionMessagesPage(
  sessionId: string,
  offset: number,
): Promise<SessionMessagePage> {
  const params = new URLSearchParams();
  params.set('view', 'messages');
  params.set('limit', String(SESSION_READ_MESSAGE_LIMIT));
  params.set('offset', String(offset));
  return parseSessionMessagePage(
    await requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/read?${params.toString()}`),
  );
}
