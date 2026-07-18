import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';
import type { SessionMessageRow } from '../contracts/session-read';
import type { SemanticEntry } from '../contracts/semantic-cards';
import { SessionReadIsland } from './session-read';

const toolMessage: SessionMessageRow = {
  id: 'message:2',
  role: 'assistant',
  material_origin: 'assistant_authored',
  text: 'Ran the build and inspected the failing tool result.',
  timestamp: '2026-07-17T12:00:00+00:00',
  has_tool_use: true,
  has_thinking: false,
  has_paste_evidence: false,
  semantic_entries: [],
  semantic_card_suppressed: false,
};

describe('SessionReadIsland', () => {
  it('loads the next message page by offset and renders a structural tool-use flag', async () => {
    const loadPage = vi.fn(async () => ({ messages: [toolMessage], total: 1 }));
    render(<SessionReadIsland sessionId="codex-session:session/2" initialNextOffset={30} loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more messages' }));

    await screen.findByText('Ran the build and inspected the failing tool result.');
    expect(loadPage).toHaveBeenCalledWith('codex-session:session/2', 30);
    expect(screen.getByText('tool use')).toBeInTheDocument();
    expect(screen.queryByText('thinking')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'All messages loaded' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Loaded 1 additional message.');
  });

  it('renders no-more-pages state without a loader call when no next offset is set', () => {
    const loadPage = vi.fn();
    render(<SessionReadIsland sessionId="s1" initialNextOffset={null} loadPage={loadPage} />);

    expect(screen.getByRole('button', { name: 'All messages loaded' })).toBeDisabled();
    expect(loadPage).not.toHaveBeenCalled();
  });

  it('auto-pages to resolve a deep-linked message beyond the SSR-rendered first page', async () => {
    Element.prototype.scrollIntoView = vi.fn();
    const laterMessage: SessionMessageRow = { ...toolMessage, id: 'message:99', text: 'The later, deep-linked message.' };
    const loadPage = vi
      .fn()
      .mockResolvedValueOnce({ messages: [toolMessage], total: 32 })
      .mockResolvedValueOnce({ messages: [laterMessage], total: 32 });
    render(
      <SessionReadIsland
        sessionId="codex-session:session/2"
        initialNextOffset={30}
        loadPage={loadPage}
        initialHash="#msg-message:99"
      />,
    );

    await screen.findByText('The later, deep-linked message.');
    await vi.waitFor(() => expect(Element.prototype.scrollIntoView).toHaveBeenCalled());

    expect(loadPage).toHaveBeenNthCalledWith(1, 'codex-session:session/2', 30);
    expect(loadPage).toHaveBeenNthCalledWith(2, 'codex-session:session/2', 31);
  });

  it('reports a bounded search when a deep-linked message never resolves', async () => {
    const loadPage = vi.fn(async () => ({ messages: [], total: 0 }));
    render(
      <SessionReadIsland
        sessionId="s1"
        initialNextOffset={30}
        loadPage={loadPage}
        initialHash="#msg-does-not-exist"
      />,
    );

    await screen.findByText('The linked message could not be located within the paged transcript.');
    expect(loadPage).toHaveBeenCalledTimes(1);
  });

  it('renders a semantic card family for a message carrying materialized entries', async () => {
    const shellCard: SemanticEntry = {
      entry_type: 'card',
      card: {
        kind: 'shell',
        title: 'Shell command',
        summary: null,
        source: { session_id: 's1', provider_family: 'codex' },
        outcome: { state: 'failed', is_error: true, exit_code: 1 },
        fields: [{ label: 'command', value: 'pytest -k flaky' }],
        previews: [],
        caveats: [],
      },
    };
    const cardMessage: SessionMessageRow = { ...toolMessage, id: 'message:3', semantic_entries: [shellCard] };
    const loadPage = vi.fn(async () => ({ messages: [cardMessage], total: 1 }));
    render(<SessionReadIsland sessionId="s1" initialNextOffset={30} loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more messages' }));

    expect(await screen.findByText('pytest -k flaky')).toBeInTheDocument();
    expect(screen.getByText('FAILED')).toBeInTheDocument();
    expect(screen.queryByText('Ran the build and inspected the failing tool result.')).not.toBeInTheDocument();
  });

  it('hyperlinks a card field value using the registry session: ref convention', async () => {
    const lineageCard: SemanticEntry = {
      entry_type: 'card',
      card: {
        kind: 'lineage',
        title: 'Lineage boundary · fork',
        summary: null,
        source: { session_id: 'child-1', provider_family: 'codex' },
        outcome: null,
        fields: [{ label: 'parent', value: 'session:codex-session:parent-1' }],
        previews: [],
        caveats: [],
      },
    };
    const cardMessage: SessionMessageRow = { ...toolMessage, id: 'message:5', semantic_entries: [lineageCard] };
    const loadPage = vi.fn(async () => ({ messages: [cardMessage], total: 1 }));
    render(<SessionReadIsland sessionId="s1" initialNextOffset={30} loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more messages' }));

    const link = await screen.findByRole('link', { name: 'session:codex-session:parent-1' });
    expect(link).toHaveAttribute('href', '/app/sessions/codex-session%3Aparent-1');
  });

  it('renders nothing for a message whose evidence was absorbed into a paired card', async () => {
    const suppressedMessage: SessionMessageRow = { ...toolMessage, id: 'message:4', semantic_card_suppressed: true };
    const loadPage = vi.fn(async () => ({ messages: [suppressedMessage], total: 1 }));
    render(<SessionReadIsland sessionId="s1" initialNextOffset={30} loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more messages' }));

    await vi.waitFor(() => expect(loadPage).toHaveBeenCalled());
    expect(document.getElementById('msg-message:4')).not.toBeInTheDocument();
  });
});
