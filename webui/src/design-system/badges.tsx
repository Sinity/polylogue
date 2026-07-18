import {
  EVIDENCE_STATE_LABELS,
  ORIGIN_LABELS,
  type EvidenceState,
  type OriginToken,
} from '../generated/contracts';
import { VisuallyHidden } from './layout';

const EVIDENCE_SYMBOLS: Readonly<Record<EvidenceState, string>> = {
  exact: '✓',
  qualified: '≈',
  stale: '◷',
  unknown: '?',
  degraded: '!',
};

export function OriginBadge({ origin }: { origin: OriginToken }) {
  return (
    <span class="pl-badge pl-origin-badge" data-origin={origin}>
      {ORIGIN_LABELS[origin]}
    </span>
  );
}

export function UnknownOriginBadge() {
  return (
    <span class="pl-badge pl-origin-unknown" data-origin-state="unknown">
      <span aria-hidden="true">?</span> Unknown origin
    </span>
  );
}

export function EvidenceStateBadge({ state, qualifiedBy }: { state: EvidenceState; qualifiedBy?: string }) {
  const label = EVIDENCE_STATE_LABELS[state];
  return (
    <span class="pl-badge pl-evidence-badge" data-evidence-state={state} title={qualifiedBy}>
      <span aria-hidden="true">{EVIDENCE_SYMBOLS[state]}</span>
      <span>{label}</span>
      {qualifiedBy ? <VisuallyHidden>{`: ${qualifiedBy}`}</VisuallyHidden> : null}
      <VisuallyHidden>{` evidence state`}</VisuallyHidden>
    </span>
  );
}
