import { useState } from 'preact/hooks';

const LANES = [
  { key: 'provider_reported_usd', label: 'Provider-reported' },
  { key: 'api_equivalent_usd', label: 'API-equivalent' },
  { key: 'subscription_equivalent_usd', label: 'Subscription-equivalent' },
  { key: 'catalog_priced_usd', label: 'Catalog-priced' },
  { key: 'tool_surcharge_usd', label: 'Tool surcharge' },
] as const;

/**
 * Pure client-side focus toggle over the SSR-rendered lane columns already
 * present in `[data-lane-table]` tables - it never fetches or recomputes a
 * figure, only highlights one already-rendered basis column at a time so a
 * reader can compare lanes without the server collapsing them into one number.
 */
export function CostLaneToggle() {
  const [focused, setFocused] = useState<string | null>(null);

  function applyFocus(lane: string | null): void {
    setFocused(lane);
    for (const table of document.querySelectorAll('[data-lane-table]')) {
      if (lane) {
        table.setAttribute('data-lane-focus', lane);
      } else {
        table.removeAttribute('data-lane-focus');
      }
    }
  }

  return (
    <div class="cost-lane-toggle" role="group" aria-label="Focus one cost lane">
      <button type="button" class={focused === null ? 'is-active' : ''} onClick={() => applyFocus(null)}>
        All lanes
      </button>
      {LANES.map((lane) => (
        <button
          key={lane.key}
          type="button"
          class={focused === lane.key ? 'is-active' : ''}
          onClick={() => applyFocus(lane.key)}
        >
          {lane.label}
        </button>
      ))}
    </div>
  );
}
