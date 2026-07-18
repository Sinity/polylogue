import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';
import type { SessionListRow } from '../contracts/session-list';
import { SessionListIsland } from './session-list';

const row: SessionListRow = {
  id: 'codex-session:session/2',
  title: 'Continuation contract wiring',
  origin: 'codex-session',
  date: '2026-07-17T12:00:00+00:00',
  message_count: 42,
  word_count: 1024,
  repo: 'polylogue',
};

describe('SessionListIsland', () => {
  it('loads the next page by offset and links each session to its read page', async () => {
    const loadPage = vi.fn(async () => ({ items: [row], total: 1 }));
    render(<SessionListIsland filters={{ origin: 'codex-session' }} initialNextOffset={20} loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more sessions' }));

    expect(await screen.findByRole('link', { name: 'Continuation contract wiring' })).toHaveAttribute(
      'href',
      '/app/sessions/codex-session%3Asession%2F2',
    );
    expect(loadPage).toHaveBeenCalledTimes(1);
    expect(loadPage).toHaveBeenCalledWith({ origin: 'codex-session' }, 20);
    expect(screen.getByRole('button', { name: 'All matching sessions loaded' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Loaded 1 additional session.');
  });

  it('renders no-more-pages state without a loader call when no next offset is set', () => {
    const loadPage = vi.fn();
    render(<SessionListIsland filters={{}} initialNextOffset={null} loadPage={loadPage} />);

    expect(screen.getByRole('button', { name: 'All matching sessions loaded' })).toBeDisabled();
    expect(loadPage).not.toHaveBeenCalled();
  });
});
