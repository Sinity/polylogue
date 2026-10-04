import { act, fireEvent, render, screen, within } from '@testing-library/preact';
import { describe, expect, it, vi } from 'vitest';

import { PolylogueClient } from '../api/generated';
import type { ClientRequest, ClientTransport, RequestOptions } from '../api/runtime';
import type { InsightPanel, ObservabilityPayload } from '../contracts/observability';
import { FreshnessLadder, InsightBrowser, ObservabilityIsland } from './observability';

const payload: ObservabilityPayload = {
  contract_version: 1,
  status: {
    adapter: 'status-component-snapshot',
    components: [{ name: 'fts', state: 'timed_out', detail: 'deadline exceeded', age_s: 42, last_good: { indexed: 12 } }],
    snapshot: { state: 'fresh', age_s: 0, captured_at: '2026-09-27T10:00:00Z', frame: 'frame-a', current_frame: 'frame-a', frame_changed: false, refresh_error: null },
    catchup: { mode: 'idle', current_phase: 'idle' },
  },
  insights: [{ name: 'session_profiles', display_name: 'Session Profiles', state: 'available', error: null, readiness: { state: 'fresh', reason: null }, items: [{ fields: [{ label: 'sessions', value: '12' }], json: {}, provenance: { materializer_version: 7 } }] }],
  insights_loaded: true,
};

describe('ObservabilityIsland', () => {
  it('retains the current status on a not-modified response', async () => {
    const transport: ClientTransport = {
      request: <TResponse,>(): Promise<TResponse> => Promise.resolve(null as TResponse),
    };
    const ensureCredential = vi.fn(async () => undefined);
    render(<ObservabilityIsland initial={payload} client={new PolylogueClient(transport)} ensureCredential={ensureCredential} />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByText('Last known good value')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Session Profiles' })).toBeInTheDocument();
    expect(await screen.findByText('Build monitor online; phase idle.')).toBeInTheDocument();
    expect(screen.queryByText('status response is not an object')).not.toBeInTheDocument();
  });

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

  it('uses generated getStatus for status and only loads insights after an explicit action', async () => {
    const requests: Array<{ request: ClientRequest; options: RequestOptions | undefined }> = [];
    const transport: ClientTransport = {
      request: <TResponse,>(_request: ClientRequest, options?: RequestOptions): Promise<TResponse> => {
        requests.push({ request: _request, options });
        const response = _request.path === '/api/status'
          ? {
              status_snapshot: { state: 'fresh', age_s: 0.2, captured_at: '2026-09-27T10:00:00Z', frame: 'frame-a', current_frame: 'frame-a', frame_changed: false },
              status_components: [
                { component: 'archive_storage', state: 'fresh', age_s: 0.4, error: null },
              ],
              catchup: {
                mode: 'discovering', current_phase: 'discovering', current_source: 'codex',
                discovery_inspected_count: 18, discovery_accepted_count: 4, discovery_rejected_count: 14,
                planned_raw_revision_count: null, completed_raw_revision_count: null, eta_s: null,
              },
            }
          : {
              contract_version: 1,
              status: payload.status,
              insights: [{
                name: 'session_profiles', display_name: 'Session Profiles', state: 'available', error: null,
                readiness: { state: 'fresh', reason: null },
                items: [{ fields: [{ label: 'sessions', value: '12' }], json: {}, provenance: null }],
              }],
              insights_loaded: true,
            };
        return Promise.resolve(response as TResponse);
      },
    };
    const ensureCredential = vi.fn(async () => undefined);
    render(<ObservabilityIsland initial={{ ...payload, insights: [], insights_loaded: false }} client={new PolylogueClient(transport)} ensureCredential={ensureCredential} />);

    expect(await screen.findByText('Phase: discovering · source: codex')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'archive_storage' })).toBeInTheDocument();
    expect(screen.getByText('18')).toBeInTheDocument();
    expect(screen.getByText('Unknown / Unknown')).toBeInTheDocument();
    expect(screen.getByText(/completion is not inferred/)).toBeInTheDocument();
    expect(requests.map(({ request }) => request.path)).toEqual(['/api/status']);
    expect(requests[0]?.options?.timeoutMs).toBe(3_000);
    expect(requests[0]?.options?.signal).toBeInstanceOf(AbortSignal);

    fireEvent.click(screen.getByRole('button', { name: 'Load insights' }));
    expect(await screen.findByRole('heading', { name: 'Session Profiles' })).toBeInTheDocument();
    expect(requests.map(({ request }) => request.path)).toEqual(['/api/status', '/api/webui/observability']);
  });

  it('aborts hidden and unmounted status requests and ignores a late older frame', async () => {
    vi.useFakeTimers();
    let hidden = false;
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    const pending: Array<{
      readonly request: ClientRequest;
      readonly options: RequestOptions | undefined;
      readonly resolve: (value: unknown) => void;
    }> = [];
    const transport: ClientTransport = {
      request: <TResponse,>(request: ClientRequest, options?: RequestOptions): Promise<TResponse> => new Promise((resolve) => {
        pending.push({ request, options, resolve: (value) => resolve(value as TResponse) });
      }),
    };
    const view = render(<ObservabilityIsland initial={{ ...payload, insights: [], insights_loaded: false }} client={new PolylogueClient(transport)} ensureCredential={async () => undefined} />);
    await act(async () => { await Promise.resolve(); });
    expect(pending).toHaveLength(1);

    hidden = true;
    document.dispatchEvent(new Event('visibilitychange'));
    expect(pending[0]?.options?.signal?.aborted).toBe(true);

    hidden = false;
    document.dispatchEvent(new Event('visibilitychange'));
    await act(async () => { await Promise.resolve(); });
    expect(pending).toHaveLength(2);
    await act(async () => {
      pending[1]?.resolve({
        status_snapshot: { state: 'stale', age_s: 0, captured_at: 'now', frame: 'frame-b', current_frame: 'frame-c', frame_changed: true },
        status_components: [],
        catchup: { mode: 'catching_up', current_phase: 'parse', current_source: 'codex', planned_raw_revision_count: 10, completed_raw_revision_count: 4, eta_s: 8 },
      });
      await Promise.resolve();
    });
    expect(screen.getByText('Phase: parse · source: codex')).toBeInTheDocument();
    expect(screen.getByText('Connected; sample is stale')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Snapshot evidence'));
    expect(screen.getByText('frame-b')).toBeInTheDocument();
    expect(screen.getByText('frame-c')).toBeInTheDocument();
    expect(screen.getByText('true')).toBeInTheDocument();

    await act(async () => {
      pending[0]?.resolve({
        status_snapshot: { state: 'fresh', age_s: 0, captured_at: 'older', frame: 'frame-a', current_frame: 'frame-a', frame_changed: false },
        status_components: [],
        catchup: { mode: 'idle', current_phase: 'idle', planned_raw_revision_count: null, completed_raw_revision_count: null },
      });
      await Promise.resolve();
    });
    expect(screen.getByText('Phase: parse · source: codex')).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
      await Promise.resolve();
    });
    expect(pending).toHaveLength(3);

    view.unmount();
    expect(pending[2]?.options?.signal?.aborted).toBe(true);
    vi.useRealTimers();
  });

  it('keeps last-known status visibly stale after a generated status request fails', async () => {
    vi.useFakeTimers();
    let calls = 0;
    const transport: ClientTransport = {
      request: async <TResponse,>(request: ClientRequest): Promise<TResponse> => {
        if (request.path === '/api/webui/observability') throw new Error('unexpected insight request');
        calls += 1;
        if (calls > 1) throw new Error('daemon disconnected');
        return {
          status_snapshot: { state: 'fresh', age_s: 0.1, captured_at: '2026-09-27T10:00:00Z', frame: 'frame-a', current_frame: 'frame-a', frame_changed: false },
          status_components: [],
          catchup: { mode: 'catching_up', current_phase: 'parse', current_source: 'claude', discovery_inspected_count: 7 },
        } as TResponse;
      },
    };
    const view = render(<ObservabilityIsland initial={{ ...payload, insights: [], insights_loaded: false }} client={new PolylogueClient(transport)} ensureCredential={async () => undefined} />);
    // Drain the credential bootstrap and retry wrapper, not one microtask.
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText('Phase: parse · source: claude')).toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
      await Promise.resolve();
    });
    expect(screen.getByText(/Offline; retaining last-known values as stale/)).toBeInTheDocument();
    expect(screen.getByText('7')).toBeInTheDocument();
    expect(screen.getByText('daemon disconnected')).toBeInTheDocument();
    view.unmount();
    vi.useRealTimers();
  });

  it('shows cold-build preparation counts before the first intake page', async () => {
    const transport: ClientTransport = {
      request: async <TResponse,>(request: ClientRequest): Promise<TResponse> => {
        if (request.path === '/api/webui/observability') throw new Error('unexpected insight request');
        return {
          status_snapshot: { state: 'fresh', age_s: 0.1, captured_at: '2026-09-27T10:00:00Z', frame: 'frame-a', current_frame: 'frame-a', frame_changed: false },
          status_components: [],
          catchup: {
            mode: 'cold_build_preparing', current_phase: 'baseline_hash', current_source: null,
            preparation_inspected_count: 432, preparation_revision_count: 65, preparation_hashed_bytes: 9876,
            preparation_age_s: 12.5, last_advanced_age_s: 0.4, planned_raw_revision_count: null, eta_s: null,
          },
        } as TResponse;
      },
    };
    const view = render(<ObservabilityIsland initial={{ ...payload, insights: [], insights_loaded: false }} client={new PolylogueClient(transport)} ensureCredential={async () => undefined} />);
    expect(await screen.findByText('Phase: baseline_hash · source: Unknown')).toBeInTheDocument();
    expect(screen.getByText('432')).toHaveAttribute('data-preparation', 'inspected');
    expect(screen.getByText('65')).toHaveAttribute('data-preparation', 'revisions');
    view.unmount();
  });

  it('keeps only one polling timer chain after manual refresh', async () => {
    vi.useFakeTimers();
    const pending: Array<{ resolve: (value: unknown) => void; signal: AbortSignal | undefined }> = [];
    const transport: ClientTransport = {
      request: <TResponse,>(_request: ClientRequest, options?: RequestOptions): Promise<TResponse> => new Promise((resolve) => {
        pending.push({ resolve: (value) => resolve(value as TResponse), signal: options?.signal });
      }),
    };
    const view = render(<ObservabilityIsland initial={{ ...payload, insights: [], insights_loaded: false }} client={new PolylogueClient(transport)} ensureCredential={async () => undefined} />);
    const resolveFreshStatus = async (index: number): Promise<void> => {
      await act(async () => {
        pending[index]?.resolve({
          status_snapshot: { state: 'fresh', age_s: 0, captured_at: 'now', frame: 'frame-a', current_frame: 'frame-a', frame_changed: false },
          status_components: [],
          catchup: { mode: 'catching_up', current_phase: 'parse', planned_raw_revision_count: 10, completed_raw_revision_count: index },
        });
        await Promise.resolve();
      });
    };

    await act(async () => { await Promise.resolve(); });
    expect(pending).toHaveLength(1);
    await resolveFreshStatus(0);
    await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh status' }));
    await act(async () => { await Promise.resolve(); });
    expect(pending).toHaveLength(2);
    await resolveFreshStatus(1);

    await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });
    expect(pending).toHaveLength(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });
    expect(pending).toHaveLength(3);

    view.unmount();
    expect(pending[2]?.signal?.aborted).toBe(true);
    vi.useRealTimers();
  });
});
