import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';
import type { SessionMessageRow } from '../contracts/session-read';
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
});
