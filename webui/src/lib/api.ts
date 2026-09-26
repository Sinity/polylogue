import { PolylogueClient } from '../api/generated';
import { DaemonHttpError } from '../api/runtime';
import { SESSION_LIST_LIMIT, parseSessionListPage, type SessionListPage } from '../contracts/session-list';
import {
  SESSION_READ_MESSAGE_LIMIT,
  parseSessionMessagePage,
  type SessionMessagePage,
  type SessionMessageWindow,
} from '../contracts/session-read';

let credentialBootstrap: Promise<void> | null = null;
const credentialClient = new PolylogueClient();

export async function ensureWebCredential(): Promise<void> {
  if (credentialBootstrap !== null) {
    return credentialBootstrap;
  }
  credentialBootstrap = (async () => {
    await credentialClient.bootstrapWebCredential();
  })();
  try {
    await credentialBootstrap;
  } catch (error) {
    credentialBootstrap = null;
    throw error;
  }
}

/**
 * An archive request the daemon refused, carrying the daemon's own typed
 * refusal code. Callers that must distinguish refusals — a deep link naming a
 * message this session does not contain is not the same event as a transport
 * failure — read `code` instead of matching on message text.
 */
export class ArchiveRequestError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(code: string, status: number) {
    super(`archive query failed: ${code}`);
    this.name = 'ArchiveRequestError';
    this.code = code;
    this.status = status;
  }
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
  await ensureWebCredential();
  try {
    return parseSessionListPage(await credentialClient.searchSessions({
      ...(filters.origin ? { origin: filters.origin } : {}),
      ...(filters.since ? { since: filters.since } : {}),
      ...(filters.repo ? { repo: filters.repo } : {}),
      limit: SESSION_LIST_LIMIT,
      offset,
    }));
  } catch (error) {
    if (error instanceof DaemonHttpError) throw new ArchiveRequestError(error.code ?? `HTTP ${error.status}`, error.status);
    throw error;
  }
}

/**
 * Fetch exactly one message window.
 *
 * An `around` window names a message and lets the daemon resolve the offset of
 * the window that holds it, so honouring a deep link is one request whatever
 * the target's depth — the client never walks pages looking for it
 * (polylogue-i5vqc). The response reports the `offset` it actually served.
 */
export async function fetchSessionMessagesPage(
  sessionId: string,
  window: SessionMessageWindow,
): Promise<SessionMessagePage> {
  await ensureWebCredential();
  try {
    return parseSessionMessagePage(await credentialClient.readSessionView({
      session_id: sessionId,
      view: 'messages',
      limit: SESSION_READ_MESSAGE_LIMIT,
      ...('around' in window ? { around: window.around } : { offset: window.offset }),
    }));
  } catch (error) {
    if (error instanceof DaemonHttpError) throw new ArchiveRequestError(error.code ?? `HTTP ${error.status}`, error.status);
    throw error;
  }
}
