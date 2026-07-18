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

export interface ObservabilityPayload {
  readonly contract_version: number;
  readonly status: { readonly adapter: string; readonly components: readonly StatusComponentSnapshot[] };
  readonly insights: readonly InsightPanel[];
}

const COMPONENT_STATES = new Set<StatusComponentState>(['fresh', 'stale', 'refreshing', 'timed_out', 'unavailable', 'degraded']);

function parseComponent(value: unknown, index: number): StatusComponentSnapshot {
  if (!isRecord(value)) throw new TypeError(`status component ${index} is not an object`);
  const state = requiredString(value, 'state') as StatusComponentState;
  if (!COMPONENT_STATES.has(state)) throw new TypeError(`status component ${index} has unsupported state ${state}`);
  return { name: requiredString(value, 'name'), state, detail: nullableString(value, 'detail'), age_s: nullableNumber(value, 'age_s'), last_good: value.last_good ?? null };
}

export function parseObservabilityPayload(value: unknown): ObservabilityPayload {
  if (!isRecord(value) || !isRecord(value.status) || !Array.isArray(value.status.components) || !Array.isArray(value.insights)) throw new TypeError('response is not an observability payload');
  return {
    contract_version: typeof value.contract_version === 'number' ? value.contract_version : 1,
    status: { adapter: requiredString(value.status, 'adapter'), components: value.status.components.map(parseComponent) },
    insights: value.insights.map((raw, index) => {
      if (!isRecord(raw) || !Array.isArray(raw.items) || !isRecord(raw.readiness)) throw new TypeError(`insight panel ${index} is invalid`);
      return { name: requiredString(raw, 'name'), display_name: requiredString(raw, 'display_name'), state: requiredString(raw, 'state'), error: nullableString(raw, 'error'), readiness: { state: requiredString(raw.readiness, 'state'), reason: nullableString(raw.readiness, 'reason') }, items: raw.items.map((item, itemIndex) => {
        if (!isRecord(item) || !Array.isArray(item.fields)) throw new TypeError(`insight item ${index}:${itemIndex} is invalid`);
        return { fields: item.fields.map((field, fieldIndex) => { if (!isRecord(field)) throw new TypeError(`insight field ${index}:${itemIndex}:${fieldIndex} is invalid`); return { label: requiredString(field, 'label'), value: requiredString(field, 'value') }; }), json: item.json, provenance: item.provenance };
      }) };
    }),
  };
}
