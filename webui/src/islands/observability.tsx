import { useEffect, useRef, useState } from 'preact/hooks';

import { PolylogueClient } from '../api/generated';
import { parseObservabilityPayload, parseStatusObservation, type InsightPanel, type ObservabilityPayload, type ObservabilityStatus, type StatusComponentSnapshot } from '../contracts/observability';
import { ensureWebCredential, retryCredentialRejectedRequest } from '../lib/api';

const observabilityClient = new PolylogueClient();
const STATUS_POLL_MS = 2_000;
const STATUS_TIMEOUT_MS = 3_000;

type MonitorConnection = 'checking' | 'online' | 'offline' | 'paused';

interface MonitorObservation {
  readonly status: ObservabilityStatus;
  readonly connection: MonitorConnection;
  readonly receivedAt: number | null;
  readonly error: string | null;
}

export function InsightBrowser({ panels }: { readonly panels: readonly InsightPanel[] }) {
  return <div class="insight-grid">{panels.map((panel) => <article class="insight-card" data-insight-state={panel.state} key={panel.name}><h3>{panel.display_name}</h3><p class="state-label">{panel.state} · readiness {panel.readiness.state}</p>{panel.error ? <p>{panel.error}</p> : panel.items.length === 0 ? <p>No materialized rows are available for this bounded view.</p> : <ul>{panel.items.map((item, index) => <li key={index}><dl>{item.fields.map((field) => <div key={field.label}><dt>{field.label || 'value'}</dt><dd>{field.value}</dd></div>)}</dl>{item.provenance ? <details><summary>Provenance</summary><pre>{JSON.stringify(item.provenance, null, 2)}</pre></details> : null}<details><summary>JSON evidence</summary><pre>{JSON.stringify(item.json, null, 2)}</pre></details></li>)}</ul>}</article>)}</div>;
}

function ComponentGrid({ components }: { readonly components: readonly StatusComponentSnapshot[] }) {
  return <ul class="status-grid">{components.map((component) => <li class="status-card" data-status-state={component.state} key={component.name}><h3>{component.name}</h3><p class="state-label">{component.state}</p>{component.detail ? <p>{component.detail}</p> : null}{component.age_s !== null ? <p>Age: {component.age_s.toFixed(1)} s</p> : null}{component.state === 'timed_out' && component.last_good !== null ? <details open><summary>Last known good value</summary><pre>{JSON.stringify(component.last_good, null, 2)}</pre></details> : null}</li>)}</ul>;
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function countText(value: unknown): string {
  const number = finiteNumber(value);
  return number === null || number < 0 ? 'Unknown' : Math.trunc(number).toLocaleString();
}

function durationText(value: unknown): string {
  const number = finiteNumber(value);
  return number === null || number < 0 ? 'Unknown' : `${number.toFixed(1)} s`;
}

function progressText(catchup: Record<string, unknown>): string {
  const completed = finiteNumber(catchup.completed_raw_revision_count);
  const planned = finiteNumber(catchup.planned_raw_revision_count);
  if (completed === null || planned === null || completed < 0 || planned <= 0) {
    if (planned === 0 && completed === 0) return '0 / 0; ratio unavailable';
    return `${countText(catchup.completed_raw_revision_count)} / ${countText(catchup.planned_raw_revision_count)}`;
  }
  const boundedCompleted = Math.min(completed, planned);
  return `${Math.trunc(completed).toLocaleString()} / ${Math.trunc(planned).toLocaleString()} (${(boundedCompleted * 100 / planned).toFixed(1)}%)`;
}

function LiveBuildMonitor({ observation }: { readonly observation: MonitorObservation }) {
  const catchup = record(observation.status.catchup);
  const snapshot = observation.status.snapshot;
  const phase = typeof catchup.current_phase === 'string'
    ? catchup.current_phase
    : typeof catchup.mode === 'string' ? catchup.mode : 'unknown';
  const source = typeof catchup.current_source === 'string' && catchup.current_source
    ? catchup.current_source
    : 'Unknown';
  const stalled = observation.connection === 'offline' || snapshot.state === 'stale'
    || snapshot.state === 'unavailable' || snapshot.frame_changed === true;
  const sampleAge = observation.receivedAt === null
    ? 'not observed by this browser'
    : `${Math.max(0, (Date.now() - observation.receivedAt) / 1000).toFixed(1)} s since last successful browser observation`;
  const connectionLabel = observation.connection === 'online'
    ? stalled ? 'Connected; sample is stale' : 'Connected'
    : observation.connection === 'offline'
      ? 'Offline; retaining last-known values as stale'
      : observation.connection === 'paused'
        ? 'Paused while this page is hidden'
        : 'Checking connection';
  const lastAdvance = catchup.last_advanced_age_s ?? catchup.discovery_last_advanced_age_s;
  const haltedSources = Array.isArray(catchup.halted_sources) ? catchup.halted_sources : [];

  return <section class="observability-panel" aria-labelledby="build-monitor-title">
    <h2 id="build-monitor-title">Live build monitor</h2>
    <p data-monitor-connection={stalled ? 'stale' : observation.connection}>{connectionLabel}</p>
    <p data-monitor-phase={phase}>Phase: {phase} · source: {source}</p>
    {catchup.mode === 'cold_build_preparing' ? <dl class="monitor-facts" aria-label="Cold build preparation">
      <div><dt>Baseline paths inspected</dt><dd data-preparation="inspected">{countText(catchup.preparation_inspected_count)}</dd></div>
      <div><dt>Baseline revisions hashed</dt><dd data-preparation="revisions">{countText(catchup.preparation_revision_count)}</dd></div>
      <div><dt>Baseline bytes hashed</dt><dd>{countText(catchup.preparation_hashed_bytes)}</dd></div>
      <div><dt>Preparation elapsed</dt><dd>{durationText(catchup.preparation_age_s)}</dd></div>
    </dl> : null}
    <dl class="monitor-facts">
      <div><dt>Accepted raw revisions</dt><dd data-progress="revisions">{progressText(catchup)}</dd></div>
      <div><dt>Discovery inspected</dt><dd>{countText(catchup.discovery_inspected_count)}</dd></div>
      <div><dt>Discovery accepted</dt><dd>{countText(catchup.discovery_accepted_count)}</dd></div>
      <div><dt>Discovery rejected</dt><dd>{countText(catchup.discovery_rejected_count)}</dd></div>
      <div><dt>Last advancement</dt><dd>{durationText(lastAdvance)}</dd></div>
      <div><dt>Owner ETA</dt><dd>{durationText(catchup.eta_s)}</dd></div>
      <div><dt>Successful files</dt><dd>{countText(catchup.cumulative_succeeded_file_count)}</dd></div>
      <div><dt>Failed file attempts</dt><dd>{countText(catchup.cumulative_failed_file_attempts)}</dd></div>
      <div><dt>Refused files</dt><dd>{countText(catchup.cumulative_refused_file_count)}</dd></div>
      <div><dt>Sample state</dt><dd data-monitor-snapshot={snapshot.state}>{snapshot.state} · {durationText(snapshot.age_s)} old</dd></div>
      <div><dt>Browser sample age</dt><dd>{sampleAge}</dd></div>
    </dl>
    {observation.error ? <p data-monitor-error>{observation.error}</p> : null}
    {haltedSources.length > 0 ? <section aria-label="Halted sources"><h3>Halted sources</h3><ul>{haltedSources.map((raw, index) => {
      const halted = record(raw);
      const name = typeof halted.source_name === 'string' ? halted.source_name : 'Unknown source';
      const code = typeof halted.code === 'string' ? halted.code : 'unknown';
      const message = typeof halted.message === 'string' ? halted.message : '';
      return <li key={`${name}-${index}`}>{name}: {code}{message ? ` — ${message}` : ''}</li>;
    })}</ul></section> : null}
    <p>Idle means no active catch-up was reported; completion is not inferred from polling or missing counts.</p>
    <details><summary>Snapshot evidence</summary><dl>
      <div><dt>Captured at</dt><dd>{snapshot.captured_at ?? 'Unknown'}</dd></div>
      <div><dt>Observed frame</dt><dd>{snapshot.frame ?? 'Unknown'}</dd></div>
      <div><dt>Current frame</dt><dd>{snapshot.current_frame ?? 'Unknown'}</dd></div>
      <div><dt>Frame changed</dt><dd>{snapshot.frame_changed === null ? 'Unknown' : String(snapshot.frame_changed)}</dd></div>
      <div><dt>Refresh error</dt><dd>{snapshot.refresh_error ?? 'None reported'}</dd></div>
    </dl></details>
    <p class="island-status" role="status" aria-live="polite">Build monitor {observation.connection}; phase {phase}.</p>
  </section>;
}

const FRESHNESS_STAGES = ['unseen', 'acquired-unparsed', 'parsed-unindexed', 'indexed-unconverged', 'searchable'];

export function FreshnessLadder({ value }: { readonly value: unknown }) {
  if (typeof value !== 'object' || value === null || !('stage' in value)) return null;
  const record = value as Record<string, unknown>;
  const stage = typeof record.stage === 'string' ? record.stage : 'unseen';
  const current = FRESHNESS_STAGES.indexOf(stage);
  return <section data-source-freshness data-testid="source-freshness" data-operational-state={String(record.operational_state ?? 'unknown')} data-operational-reason={String(record.operational_reason ?? 'unknown')} aria-live="polite"><h3>Source stage: {stage}</h3><ol class="freshness-ladder">{FRESHNESS_STAGES.map((candidate, index) => <li data-stage-state={index < current ? 'complete' : index === current ? 'current' : 'pending'} key={candidate}>{candidate}</li>)}</ol><dl><div><dt>Operational state</dt><dd>{String(record.operational_state ?? 'unknown')}</dd></div><div><dt>Reason</dt><dd>{String(record.operational_reason ?? 'unknown')}</dd></div><div><dt>Pending bytes</dt><dd>{String(record.pending_bytes ?? 'unknown')}</dd></div><div><dt>Cursor ahead bytes</dt><dd>{String(record.cursor_ahead_bytes ?? 'unknown')}</dd></div><div><dt>Cursor age</dt><dd>{String(record.cursor_age_ms ?? 'unknown')} ms</dd></div><div><dt>FTS checked</dt><dd>{String(record.fts_checked_at ?? 'unknown')}</dd></div><div><dt>Projection receipt</dt><dd>{String(record.projection_sha256 ?? 'unknown')}</dd></div></dl></section>;
}

export function ObservabilityIsland({
  initial,
  client = observabilityClient,
  ensureCredential = ensureWebCredential,
}: {
  readonly initial: ObservabilityPayload;
  readonly client?: PolylogueClient;
  readonly ensureCredential?: () => Promise<void>;
}) {
  const [observation, setObservation] = useState<MonitorObservation>({ status: initial.status, connection: 'checking', receivedAt: null, error: null });
  const [insights, setInsights] = useState(initial.insights);
  const [insightsLoaded, setInsightsLoaded] = useState(initial.insights_loaded);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [source, setSource] = useState('');
  const [sourceResult, setSourceResult] = useState<unknown>(null);
  const [actionStatus, setActionStatus] = useState('');
  const refreshStatusRef = useRef<() => void>(() => undefined);

  useEffect(() => {
    let stopped = false;
    let generation = 0;
    let timer: number | null = null;
    let controller: AbortController | null = null;
    let inFlight = false;

    const requestStatus = (): void => {
      if (stopped || document.hidden) return;
      if (timer !== null) window.clearTimeout(timer);
      timer = null;
      if (inFlight) return;
      const requestGeneration = ++generation;
      const requestController = new AbortController();
      controller = requestController;
      inFlight = true;
      void (async () => {
        try {
          await ensureCredential();
          if (stopped || requestController.signal.aborted || requestGeneration !== generation) return;
          const raw = await retryCredentialRejectedRequest(() => client.getStatus({}, { signal: requestController.signal, timeoutMs: STATUS_TIMEOUT_MS }));
          if (raw === null) {
            if (!stopped && !requestController.signal.aborted && requestGeneration === generation) {
              setObservation((previous) => ({ ...previous, connection: 'online', receivedAt: Date.now(), error: null }));
            }
            return;
          }
          const status = parseStatusObservation(raw);
          if (!stopped && !requestController.signal.aborted && requestGeneration === generation) {
            setObservation({ status, connection: 'online', receivedAt: Date.now(), error: null });
          }
        } catch (error) {
          if (!stopped && !requestController.signal.aborted && requestGeneration === generation) {
            setObservation((previous) => ({
              ...previous,
              connection: 'offline',
              error: error instanceof Error ? error.message : 'Status refresh failed.',
            }));
          }
        } finally {
          if (requestGeneration === generation) inFlight = false;
          if (controller === requestController) controller = null;
          if (!stopped && requestGeneration === generation && !document.hidden) {
            timer = window.setTimeout(requestStatus, STATUS_POLL_MS);
          }
        }
      })();
    };

    const onVisibilityChange = (): void => {
      if (document.hidden) {
        generation += 1;
        if (timer !== null) window.clearTimeout(timer);
        timer = null;
        controller?.abort();
        controller = null;
        inFlight = false;
        setObservation((previous) => ({ ...previous, connection: 'paused' }));
      } else {
        setObservation((previous) => ({ ...previous, connection: 'checking' }));
        requestStatus();
      }
    };

    refreshStatusRef.current = requestStatus;
    document.addEventListener('visibilitychange', onVisibilityChange);
    requestStatus();
    return () => {
      stopped = true;
      generation += 1;
      document.removeEventListener('visibilitychange', onVisibilityChange);
      if (timer !== null) window.clearTimeout(timer);
      controller?.abort();
      controller = null;
      inFlight = false;
      refreshStatusRef.current = () => undefined;
    };
  }, [client, ensureCredential]);

  async function loadInsights() {
    setActionStatus(insightsLoaded ? 'Refreshing insights…' : 'Loading insights…');
    setInsightsLoading(true);
    try {
      await ensureCredential();
      const response = await retryCredentialRejectedRequest(() => client.getWebuiObservability());
      const parsed = parseObservabilityPayload(response);
      setInsights(parsed.insights);
      setInsightsLoaded(true);
      setActionStatus('Insights loaded.');
    } catch (error) {
      setActionStatus(error instanceof Error ? error.message : 'Insight loading failed.');
    } finally {
      setInsightsLoading(false);
    }
  }

  async function inspectSource(event: Event) {
    event.preventDefault();
    if (!source.trim()) return;
    setActionStatus('Inspecting exact source…');
    try {
      await ensureCredential();
      setSourceResult(await retryCredentialRejectedRequest(() => client.getWebuiFreshness({ source })));
      setActionStatus('Source freshness loaded.');
    } catch (error) {
      setActionStatus(error instanceof Error ? error.message : 'Source freshness failed.');
    }
  }

  return <>
    <LiveBuildMonitor observation={observation} />
    <section class="observability-panel" aria-labelledby="status-title"><h2 id="status-title">Component status</h2><button type="button" onClick={() => refreshStatusRef.current()}>Refresh status</button><ComponentGrid components={observation.status.components} /></section>
    <section class="observability-panel" aria-labelledby="freshness-title"><h2 id="freshness-title">Named-source freshness</h2><form class="source-lookup" onSubmit={(event) => void inspectSource(event)}><label for="source-path">Exact source path</label><input id="source-path" value={source} onInput={(event) => setSource(event.currentTarget.value)} /><button type="submit">Inspect source</button></form><FreshnessLadder value={sourceResult} /></section>
    <section class="observability-panel" aria-labelledby="insights-title"><h2 id="insights-title">Insights</h2><button type="button" disabled={insightsLoading} onClick={() => void loadInsights()}>{insightsLoading ? 'Loading insights…' : insightsLoaded ? 'Refresh insights' : 'Load insights'}</button>{insightsLoaded ? <InsightBrowser panels={insights} /> : <p data-insights-state="not-loaded">Insights have not been loaded.</p>}</section>
    <p class="island-status" role="status" aria-live="polite">{actionStatus}</p>
  </>;
}
