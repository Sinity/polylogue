import { render, screen, within } from '@testing-library/preact';
import { describe, expect, it } from 'vitest';

import type { InsightPanel, ObservabilityPayload } from '../contracts/observability';
import { FreshnessLadder, InsightBrowser, ObservabilityIsland } from './observability';

const payload: ObservabilityPayload = {
  contract_version: 1,
  status: { adapter: 'status-component-snapshot', components: [{ name: 'fts', state: 'timed_out', detail: 'deadline exceeded', age_s: 42, last_good: { indexed: 12 } }] },
  insights: [{ name: 'session_profiles', display_name: 'Session Profiles', state: 'available', error: null, readiness: { state: 'fresh', reason: null }, items: [{ fields: [{ label: 'sessions', value: '12' }], json: {}, provenance: { materializer_version: 7 } }] }],
};

describe('ObservabilityIsland', () => {
  it('renders a descriptor injected into the registry projection without web-code changes', () => {
    const fakeDescriptorPanel: InsightPanel = {
      name: 'fake_descriptor', display_name: 'Fake descriptor', state: 'available', error: null,
      readiness: { state: 'fresh', reason: null },
      items: [{ fields: [{ label: 'proof', value: 'registry generated' }], json: { registry: 'authoritative' }, provenance: null }],
    };
    render(<InsightBrowser panels={[...payload.insights, fakeDescriptorPanel]} />);
    const card = screen.getByRole('heading', { name: 'Fake descriptor' }).closest('article');
    expect(card).not.toBeNull();
    expect(within(card as HTMLElement).getByText('registry generated')).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText('JSON evidence')).toBeInTheDocument();
    expect(card).toHaveTextContent('"registry": "authoritative"');
  });

  it('keeps a timed-out component’s last-good evidence beside healthy panels', () => {
    render(<ObservabilityIsland initial={payload} />);
    expect(screen.getByText('Last known good value')).toBeInTheDocument();
    expect(screen.getByText(/indexed/)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Session Profiles' })).toBeInTheDocument();
  });

  it('renders exact-source counts and attention states as a monotone stage ladder', () => {
    render(<FreshnessLadder value={{ stage: 'parsed-unindexed', operational_state: 'degraded', operational_reason: 'cursor-ahead', pending_bytes: 12, cursor_ahead_bytes: 4, cursor_age_ms: 23, fts_checked_at: '2026-07-18T12:00:00Z', projection_sha256: 'receipt' }} />);
    expect(screen.getByText('parsed-unindexed')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText('23 ms')).toBeInTheDocument();
    expect(screen.getByTestId('source-freshness')).toHaveAttribute('data-operational-reason', 'cursor-ahead');
  });
});
