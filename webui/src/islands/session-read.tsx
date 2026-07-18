import { useEffect, useState } from 'preact/hooks';
import type { SessionMessageRow } from '../contracts/session-read';
import { SemanticEntries } from '../design-system/semantic-cards';
import { fetchSessionMessagesPage } from '../lib/api';

type PageLoader = (
  sessionId: string,
  offset: number,
) => Promise<{ messages: readonly SessionMessageRow[]; total: number }>;

export interface SessionReadIslandProps {
  readonly sessionId: string;
  readonly initialNextOffset?: number | null;
  readonly loadPage?: PageLoader;
  /** Overridable for tests; defaults to the real page's fragment. */
  readonly initialHash?: string;
}

/** Bounds the deep-link resolve loop below so a stale/bogus anchor cannot page forever. */
const MAX_DEEP_LINK_PAGES = 50;

function messageTimestamp(timestamp: string | null): string {
  return timestamp ?? 'Time unavailable';
}

function MessageFlags({ message }: { readonly message: SessionMessageRow }) {
  return (
    <div class="message-flow__flags">
      {message.has_tool_use ? (
        <span class="message-flag" data-flag="tool-use">
          tool use
        </span>
      ) : null}
      {message.has_thinking ? (
        <span class="message-flag" data-flag="thinking">
          thinking
        </span>
      ) : null}
      {message.has_paste_evidence ? (
        <span class="message-flag" data-flag="paste">
          paste
        </span>
      ) : null}
    </div>
  );
}

function MessageFlowItem({ message }: { readonly message: SessionMessageRow }) {
  if (message.semantic_card_suppressed) {
    return null;
  }
  return (
    <li
      class="message-flow__item"
      id={`msg-${message.id}`}
      data-role={message.role}
      data-material-origin={message.material_origin}
    >
      <div class="message-flow__meta">
        <span class="message-flow__role">{message.role}</span>
        <span class="message-flow__material-origin">{message.material_origin}</span>
        <span>{messageTimestamp(message.timestamp)}</span>
      </div>
      {message.semantic_entries.length > 0 ? (
        <SemanticEntries entries={message.semantic_entries} />
      ) : (
        <>
          <p class="message-flow__text">{message.text || '[empty message]'}</p>
          <MessageFlags message={message} />
        </>
      )}
    </li>
  );
}

export function SessionReadIsland({
  sessionId,
  initialNextOffset,
  loadPage = fetchSessionMessagesPage,
  initialHash = typeof window === 'undefined' ? '' : window.location.hash,
}: SessionReadIslandProps) {
  const [nextOffset, setNextOffset] = useState<number | null | undefined>(initialNextOffset);
  const [messages, setMessages] = useState<readonly SessionMessageRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');

  const exhausted = nextOffset === null || nextOffset === undefined;
  const buttonLabel = exhausted ? 'All messages loaded' : 'Load more messages';

  useEffect(() => {
    const targetId = initialHash.startsWith('#') ? initialHash.slice(1) : initialHash;
    if (!targetId || document.getElementById(targetId) !== null) {
      return;
    }
    const targetMessageId = targetId.startsWith('msg-') ? targetId.slice(4) : targetId;
    let cancelled = false;
    let offset = initialNextOffset;
    async function resolveDeepLink(): Promise<void> {
      setLoading(true);
      setStatus('Locating the linked message…');
      let found = false;
      for (let page = 0; page < MAX_DEEP_LINK_PAGES && !found; page += 1) {
        if (cancelled || offset === null || offset === undefined) {
          break;
        }
        const loadedAt = offset;
        const result = await loadPage(sessionId, loadedAt);
        if (cancelled) {
          return;
        }
        setMessages((current) => [...current, ...result.messages]);
        found = result.messages.some((message) => message.id === targetMessageId);
        const reachedEnd = loadedAt + result.messages.length >= result.total || result.messages.length === 0;
        offset = reachedEnd ? null : loadedAt + result.messages.length;
        setNextOffset(offset);
      }
      setLoading(false);
      if (!found) {
        setStatus('The linked message could not be located within the paged transcript.');
        return;
      }
      // Wait a tick past this render so the freshly appended message has
      // committed to the DOM before scrolling to its anchor.
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      if (cancelled) {
        return;
      }
      document.getElementById(targetId)?.scrollIntoView();
      setStatus('');
    }
    void resolveDeepLink().catch(() => {
      if (!cancelled) {
        setLoading(false);
        setStatus('The linked message could not be loaded.');
      }
    });
    return () => {
      cancelled = true;
    };
    // Deep-link resolution runs once per mount against the anchor present at load time.
  }, []);

  async function loadNextPage(): Promise<void> {
    if (loading || exhausted) {
      return;
    }
    setLoading(true);
    setStatus('Loading messages…');
    try {
      const page = await loadPage(sessionId, nextOffset);
      setMessages((current) => [...current, ...page.messages]);
      const reachedEnd = nextOffset + page.messages.length >= page.total || page.messages.length === 0;
      setNextOffset(reachedEnd ? null : nextOffset + page.messages.length);
      setStatus(
        page.messages.length === 0
          ? 'No additional messages found.'
          : `Loaded ${page.messages.length.toLocaleString()} additional ${page.messages.length === 1 ? 'message' : 'messages'}.`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Messages could not be loaded.');
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
        aria-controls="message-flow-more"
        aria-busy={loading}
        onClick={() => void loadNextPage()}
      >
        {loading ? 'Loading…' : buttonLabel}
      </button>
      <p class="island-status" role="status" aria-live="polite">
        {status}
      </p>
      <ol id="message-flow-more" class="message-flow message-flow--continued" aria-label="Additional messages">
        {messages.map((message) => (
          <MessageFlowItem key={message.id} message={message} />
        ))}
      </ol>
    </>
  );
}
