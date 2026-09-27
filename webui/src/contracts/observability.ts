import { isRecord, nullableNumber, nullableString, requiredString } from './runtime';

export type StatusComponentState = 'fresh' | 'stale' | 'refreshing' | 'timed_out' | 'unavailable' | 'degraded';

/** Integration contract expected from polylogue-20d.17's status backend. */
export interface StatusComponentSnapshot {
  readonly name: string;
  readonly state: StatusComponentState;
  readonly detail: string | null;
  readonly age_s: number | null;
  readonly last_good: unknown | null;
}

export interface InsightPanel {
  readonly name: string;
  readonly display_name: string;
  readonly state: string;
  readonly error: string | null;
  readonly readiness: { readonly state: string; readonly reason: string | null };
  readonly items: readonly { readonly fields: readonly { readonly label: string; readonly value: string }[]; readonly json: unknown; readonly provenance: unknown }[];
}

export interface StatusSnapshotEvidence {
  readonly state: string;
  readonly age_s: number | null;
  readonly captured_at: string | null;
  readonly frame: string | null;
  readonly current_frame: string | null;
  readonly frame_changed: boolean | null;
  readonly refresh_error: string | null;
}

export interface ObservabilityStatus {
  readonly adapter: string;
  readonly components: readonly StatusComponentSnapshot[];
  readonly snapshot: StatusSnapshotEvidence;
  readonly catchup: Readonly<Record<string, unknown>>;
}

export interface ObservabilityPayload {
  readonly contract_version: number;
  readonly status: ObservabilityStatus;
  readonly insights: readonly InsightPanel[];
  readonly insights_loaded: boolean;
}

const COMPONENT_STATES = new Set<StatusComponentState>(['fresh', 'stale', 'refreshing', 'timed_out', 'unavailable', 'degraded']);

function parseComponent(value: unknown, index: number): StatusComponentSnapshot {
  if (!isRecord(value)) throw new TypeError(`status component ${index} is not an object`);
  const state = requiredString(value, 'state') as StatusComponentState;
  if (!COMPONENT_STATES.has(state)) throw new TypeError(`status component ${index} has unsupported state ${state}`);
  return {
    name: typeof value.name === 'string' ? value.name : requiredString(value, 'component'),
    state,
    detail: nullableString({ ...value, detail: value.detail ?? value.error ?? value.reason ?? null }, 'detail'),
    age_s: nullableNumber(value, 'age_s'),
    last_good: value.last_good ?? null,
  };
}

function parseSnapshot(value: unknown): StatusSnapshotEvidence {
  const snapshot = isRecord(value) ? value : {};
  return {
    state: typeof snapshot.state === 'string' ? snapshot.state : 'unavailable',
    age_s: nullableNumber({ age_s: snapshot.age_s ?? null }, 'age_s'),
    captured_at: nullableString({ captured_at: snapshot.captured_at ?? null }, 'captured_at'),
    frame: nullableString({ frame: snapshot.frame ?? null }, 'frame'),
    current_frame: nullableString({ current_frame: snapshot.current_frame ?? null }, 'current_frame'),
    frame_changed: typeof snapshot.frame_changed === 'boolean' ? snapshot.frame_changed : null,
    refresh_error: nullableString({ refresh_error: snapshot.refresh_error ?? null }, 'refresh_error'),
  };
}

function parseStatusPanel(value: unknown, fallbackAdapter = 'unavailable'): ObservabilityStatus {
  if (!isRecord(value)) throw new TypeError('status projection is not an object');
  const components = Array.isArray(value.components) ? value.components.map(parseComponent) : [];
  return {
    adapter: typeof value.adapter === 'string' ? value.adapter : fallbackAdapter,
    components,
    snapshot: parseSnapshot(value.snapshot),
    catchup: isRecord(value.catchup) ? value.catchup : {},
  };
}

/** Parse the generated getStatus response into the page's compact observation contract. */
export function parseStatusObservation(value: unknown): ObservabilityStatus {
  if (!isRecord(value)) throw new TypeError('status response is not an object');
  const statusSnapshot = parseSnapshot(value.status_snapshot);
  const rawComponents = Array.isArray(value.status_components) ? value.status_components : [];
  return {
    adapter: 'status-snapshot',
    components: rawComponents.map(parseComponent),
    snapshot: statusSnapshot,
    catchup: isRecord(value.catchup) ? value.catchup : {},
  };
}

export function parseObservabilityPayload(value: unknown): ObservabilityPayload {
  if (!isRecord(value) || !isRecord(value.status) || !Array.isArray(value.status.components) || !Array.isArray(value.insights)) throw new TypeError('response is not an observability payload');
  return {
    contract_version: typeof value.contract_version === 'number' ? value.contract_version : 1,
    status: parseStatusPanel(value.status),
    insights_loaded: value.insights_loaded === true,
    insights: value.insights.map((raw, index) => {
      if (!isRecord(raw) || !Array.isArray(raw.items) || !isRecord(raw.readiness)) throw new TypeError(`insight panel ${index} is invalid`);
      return { name: requiredString(raw, 'name'), display_name: requiredString(raw, 'display_name'), state: requiredString(raw, 'state'), error: nullableString(raw, 'error'), readiness: { state: requiredString(raw.readiness, 'state'), reason: nullableString(raw.readiness, 'reason') }, items: raw.items.map((item, itemIndex) => {
        if (!isRecord(item) || !Array.isArray(item.fields)) throw new TypeError(`insight item ${index}:${itemIndex} is invalid`);
        return { fields: item.fields.map((field, fieldIndex) => { if (!isRecord(field)) throw new TypeError(`insight field ${index}:${itemIndex}:${fieldIndex} is invalid`); return { label: requiredString(field, 'label'), value: requiredString(field, 'value') }; }), json: item.json, provenance: item.provenance };
      }) };
    }),
  };
}
