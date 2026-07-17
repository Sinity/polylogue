import { useId } from 'preact/hooks';

import { EvidenceStateBadge } from './badges';
import type { EvidenceState } from '../generated/contracts';

export interface TimelineItem {
  id: string;
  at: string;
  label: string;
  description?: string;
  state?: EvidenceState;
}

export function Timeline({ items, label = 'Timeline' }: { items: ReadonlyArray<TimelineItem>; label?: string }) {
  const headingId = useId();
  return (
    <section class="pl-timeline" aria-labelledby={headingId}>
      <h2 id={headingId}>{label}</h2>
      <ol>
        {items.map((item) => (
          <li key={item.id}>
            <div class="pl-timeline__marker" aria-hidden="true" />
            <div class="pl-timeline__content">
              <header>
                <time dateTime={item.at}>{item.at}</time>
                {item.state ? <EvidenceStateBadge state={item.state} /> : null}
              </header>
              <h3>{item.label}</h3>
              {item.description ? <p>{item.description}</p> : null}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function points(values: ReadonlyArray<number>, width: number, height: number): string {
  if (values.length === 0) return '';
  const low = Math.min(...values);
  const high = Math.max(...values);
  const spread = high - low || 1;
  return values
    .map((value, index) => {
      const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
      const y = height - ((value - low) / spread) * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
}

export function Sparkline({
  values,
  label,
  width = 180,
  height = 44,
}: {
  values: ReadonlyArray<number>;
  label: string;
  width?: number;
  height?: number;
}) {
  const id = useId();
  const description = values.length === 0 ? 'No measurements' : `Values: ${values.join(', ')}`;
  return (
    <svg class="pl-sparkline" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
      <title id={`${id}-title`}>{label}</title>
      <desc id={`${id}-desc`}>{description}</desc>
      {values.length === 0 ? null : <polyline points={points(values, width, height)} vector-effect="non-scaling-stroke" />}
    </svg>
  );
}
