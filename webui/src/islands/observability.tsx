import { useState } from 'preact/hooks';

import { parseObservabilityPayload, type InsightPanel, type ObservabilityPayload, type StatusComponentSnapshot } from '../contracts/observability';
import { requestJson } from '../lib/api';

export function InsightBrowser({ panels }: { readonly panels: readonly InsightPanel[] }) {
  return <div class="insight-grid">{panels.map((panel) => <article class="insight-card" data-insight-state={panel.state} key={panel.name}><h3>{panel.display_name}</h3><p class="state-label">{panel.state} · readiness {panel.readiness.state}</p>{panel.error ? <p>{panel.error}</p> : panel.items.length === 0 ? <p>No materialized rows are available for this bounded view.</p> : <ul>{panel.items.map((item, index) => <li key={index}><dl>{item.fields.map((field) => <div key={field.label}><dt>{field.label || 'value'}</dt><dd>{field.value}</dd></div>)}</dl>{item.provenance ? <details><summary>Provenance</summary><pre>{JSON.stringify(item.provenance, null, 2)}</pre></details> : null}</li>)}</ul>}</article>)}</div>;
}

function ComponentGrid({ components }: { readonly components: readonly StatusComponentSnapshot[] }) {
  return <ul class="status-grid">{components.map((component) => <li class="status-card" data-status-state={component.state} key={component.name}><h3>{component.name}</h3><p class="state-label">{component.state}</p>{component.detail ? <p>{component.detail}</p> : null}{component.age_s !== null ? <p>Age: {component.age_s.toFixed(1)} s</p> : null}{component.state === 'timed_out' && component.last_good !== null ? <details open><summary>Last known good value</summary><pre>{JSON.stringify(component.last_good, null, 2)}</pre></details> : null}</li>)}</ul>;
}

const FRESHNESS_STAGES = ['unseen', 'acquired-unparsed', 'parsed-unindexed', 'indexed-unconverged', 'searchable'];

export function FreshnessLadder({ value }: { readonly value: unknown }) {
  if (typeof value !== 'object' || value === null || !('stage' in value)) return null;
  const record = value as Record<string, unknown>;
  const stage = typeof record.stage === 'string' ? record.stage : 'unseen';
  const current = FRESHNESS_STAGES.indexOf(stage);
  return <section data-source-freshness aria-live="polite"><h3>Source stage: {stage}</h3><ol class="freshness-ladder">{FRESHNESS_STAGES.map((candidate, index) => <li data-stage-state={index < current ? 'complete' : index === current ? 'current' : 'pending'} key={candidate}>{candidate}</li>)}</ol><dl><div><dt>Operational state</dt><dd>{String(record.operational_state ?? 'unknown')}</dd></div><div><dt>Reason</dt><dd>{String(record.operational_reason ?? 'unknown')}</dd></div><div><dt>Pending bytes</dt><dd>{String(record.pending_bytes ?? 'unknown')}</dd></div><div><dt>Cursor ahead bytes</dt><dd>{String(record.cursor_ahead_bytes ?? 'unknown')}</dd></div><div><dt>Cursor age</dt><dd>{String(record.cursor_age_ms ?? 'unknown')} ms</dd></div><div><dt>FTS checked</dt><dd>{String(record.fts_checked_at ?? 'unknown')}</dd></div><div><dt>Projection receipt</dt><dd>{String(record.projection_sha256 ?? 'unknown')}</dd></div></dl></section>;
}

export function ObservabilityIsland({
  initial,
  load = async () => parseObservabilityPayload(await requestJson('/api/webui/observability')),
}: {
  readonly initial: ObservabilityPayload;
  readonly load?: () => Promise<ObservabilityPayload>;
}) {
  const [payload, setPayload] = useState(initial);
  const [source, setSource] = useState('');
  const [sourceResult, setSourceResult] = useState<unknown>(null);
  const [status, setStatus] = useState('');
  async function refresh() { setStatus('Refreshing observability…'); try { setPayload(await load()); setStatus('Observability refreshed.'); } catch (error) { setStatus(error instanceof Error ? error.message : 'Observability refresh failed.'); } }
  async function inspectSource(event: Event) { event.preventDefault(); if (!source.trim()) return; setStatus('Inspecting exact source…'); try { setSourceResult(await requestJson(`/api/webui/freshness?source=${encodeURIComponent(source)}`)); setStatus('Source freshness loaded.'); } catch (error) { setStatus(error instanceof Error ? error.message : 'Source freshness failed.'); } }
  return <><section class="observability-panel" aria-labelledby="status-title"><h2 id="status-title">Component status</h2><button type="button" onClick={() => void refresh()}>Refresh status</button><ComponentGrid components={payload.status.components} /></section><section class="observability-panel" aria-labelledby="freshness-title"><h2 id="freshness-title">Named-source freshness</h2><form class="source-lookup" onSubmit={(event) => void inspectSource(event)}><label for="source-path">Exact source path</label><input id="source-path" value={source} onInput={(event) => setSource(event.currentTarget.value)} /><button type="submit">Inspect source</button></form><FreshnessLadder value={sourceResult} /></section><section class="observability-panel" aria-labelledby="insights-title"><h2 id="insights-title">Insights</h2><InsightBrowser panels={payload.insights} /></section><p class="island-status" role="status" aria-live="polite">{status}</p></>;
}
