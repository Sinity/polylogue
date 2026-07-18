import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';
import type { MessageQueryPage } from '../contracts/query-units';
import { ArchiveOverviewIsland } from './archive-overview';

const page: MessageQueryPage = {
  mode: 'query-unit',
  unit: 'message',
  query: 'messages where words >= 0 | sort by time desc',
  items: [
    {
      unit: 'message',
      message_id: 'message:2',
      session_id: 'codex-session:session/2',
      origin: 'codex-session',
      title: 'Continuation contract wiring',
      role: 'assistant',
      message_type: 'message',
      material_origin: 'assistant_authored',
      occurred_at_ms: Date.UTC(2026, 6, 17, 12, 0, 0),
      position: 1,
      word_count: 42,
      text: 'The opaque continuation is replayed without reconstructing filters in the browser.',
    },
  ],
  total: 1,
  limit: 1,
  offset: 1,
  next_offset: null,
  query_ref: 'query:overview',
  result_ref: 'result:overview',
  continuation: null,
};

describe('ArchiveOverviewIsland', () => {
  it('loads the next typed page using the opaque continuation', async () => {
    const loadPage = vi.fn(async () => page);
    render(<ArchiveOverviewIsland initialContinuation="q1.opaque-token" loadPage={loadPage} />);

    fireEvent.click(screen.getByRole('button', { name: 'Load more activity' }));

    expect(await screen.findByRole('link', { name: 'Continuation contract wiring' })).toHaveAttribute(
      'href',
      '/s/codex-session%3Asession%2F2',
    );
    expect(loadPage).toHaveBeenCalledTimes(1);
    expect(loadPage).toHaveBeenCalledWith('q1.opaque-token');
    expect(screen.getByRole('button', { name: 'All activity loaded' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Loaded 1 additional record.');
  });
});
