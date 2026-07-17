import type { ComponentChildren } from 'preact';

import { Button, Stack } from './layout';
import type { HonestStateKind } from './types';

const DEFAULT_TITLES: Readonly<Record<HonestStateKind, string>> = {
  loading: 'Loading',
  empty: 'No matching records',
  unknown: 'State is not known',
  degraded: 'Showing partial evidence',
  error: 'Unable to render this view',
};

export function Skeleton({ lines = 3, label = 'Loading content' }: { lines?: number; label?: string }) {
  return (
    <div class="pl-skeleton-wrap" role="status" aria-live="polite">
      <span class="pl-sr-only">{label}</span>
      <div class="pl-skeleton" aria-hidden="true">
        {Array.from({ length: lines }, (_, index) => (
          <span key={index} style={{ '--pl-skeleton-width': `${92 - index * 13}%` }} />
        ))}
      </div>
    </div>
  );
}

export function HonestState({
  kind,
  title = DEFAULT_TITLES[kind],
  description,
  action,
}: {
  kind: HonestStateKind;
  title?: string;
  description: string;
  action?: ComponentChildren;
}) {
  const role = kind === 'error' ? 'alert' : 'status';
  return (
    <section class="pl-honest-state" data-honest-state={kind} role={role} aria-live="polite">
      <Stack space={2}>
        <p class="pl-honest-state__kind">{kind}</p>
        <h2>{title}</h2>
        <p>{description}</p>
        {action ? <div>{action}</div> : null}
      </Stack>
    </section>
  );
}

export function LoadingState({ description = 'Retrieving the requested view.' }: { description?: string }) {
  return (
    <Stack space={3}>
      <HonestState kind="loading" description={description} />
      <Skeleton />
    </Stack>
  );
}

export function EmptyState({ description }: { description: string }) {
  return <HonestState kind="empty" description={description} />;
}

export function UnknownState({ description }: { description: string }) {
  return <HonestState kind="unknown" description={description} />;
}

export function DegradedState({ description }: { description: string }) {
  return <HonestState kind="degraded" description={description} />;
}

export function RetryState({ description, onRetry }: { description: string; onRetry: () => void }) {
  return (
    <HonestState
      kind="error"
      description={description}
      action={<Button onClick={onRetry}>Retry</Button>}
    />
  );
}
