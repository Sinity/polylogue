export const CAPTURE_FRESHNESS_QUEUE_VERSION = 1;
export const CAPTURE_FRESHNESS_MAX_ENTRIES = 500;

function entryKey(provider, nativeId) {
  return `${provider}:${nativeId}`;
}

export function normalizeFreshnessQueue(value) {
  const entries = value?.version === CAPTURE_FRESHNESS_QUEUE_VERSION
    && value.entries && typeof value.entries === "object"
    ? value.entries
    : {};
  return {
    version: CAPTURE_FRESHNESS_QUEUE_VERSION,
    entries: { ...entries },
    dropped_count: Number(value?.dropped_count) || 0,
    sweep_partition: Number(value?.sweep_partition) || 0,
    sweep_not_before_ms: Number(value?.sweep_not_before_ms) || 0,
    last_sweep_at: value?.last_sweep_at || null,
    last_sweep_error: value?.last_sweep_error || null,
  };
}

export function scheduleFreshnessHint(queueValue, {
  provider,
  nativeId,
  reason,
  nowMs,
  delayMs = 0,
  providerUpdatedAt = null,
}) {
  const queue = normalizeFreshnessQueue(queueValue);
  const key = entryKey(provider, nativeId);
  const previous = queue.entries[key] || null;
  const requestedAt = nowMs + Math.max(0, delayMs);
  const entry = {
    key,
    provider,
    native_id: nativeId,
    reasons: [...new Set([...(previous?.reasons || []), reason].filter(Boolean))].slice(-8),
    provider_updated_at: providerUpdatedAt || previous?.provider_updated_at || null,
    generation: (previous?.generation || 0) + 1,
    hinted_at: new Date(nowMs).toISOString(),
    first_hinted_at: previous?.first_hinted_at || new Date(nowMs).toISOString(),
    next_attempt_at_ms: previous
      ? Math.min(previous.next_attempt_at_ms || requestedAt, requestedAt)
      : requestedAt,
    attempt_count: previous?.attempt_count || 0,
    running_poll_count: previous?.running_poll_count || 0,
    lease_owner: null,
    lease_expires_at_ms: null,
    last_error: previous?.last_error || null,
  };
  const entries = { ...queue.entries, [key]: entry };
  const keys = Object.keys(entries);
  let dropped = queue.dropped_count;
  if (keys.length > CAPTURE_FRESHNESS_MAX_ENTRIES) {
    keys
      .filter((candidate) => candidate !== key)
      .sort((left, right) => Date.parse(entries[left].hinted_at) - Date.parse(entries[right].hinted_at))
      .slice(0, keys.length - CAPTURE_FRESHNESS_MAX_ENTRIES)
      .forEach((candidate) => {
        delete entries[candidate];
        dropped += 1;
      });
  }
  return { ...queue, entries, dropped_count: dropped };
}

export function claimDueFreshness(queueValue, { nowMs, owner, leaseMs }) {
  const queue = normalizeFreshnessQueue(queueValue);
  const due = Object.values(queue.entries)
    .filter((entry) => (
      (entry.lease_owner === null || (entry.lease_expires_at_ms || 0) <= nowMs)
      && (entry.next_attempt_at_ms || 0) <= nowMs
    ))
    .sort((left, right) => (
      (left.next_attempt_at_ms || 0) - (right.next_attempt_at_ms || 0)
      || Date.parse(left.first_hinted_at) - Date.parse(right.first_hinted_at)
    ))[0];
  if (!due) return { queue, claim: null };
  const claimed = {
    ...due,
    lease_owner: owner,
    lease_expires_at_ms: nowMs + leaseMs,
  };
  return {
    queue: { ...queue, entries: { ...queue.entries, [claimed.key]: claimed } },
    claim: claimed,
  };
}

export function completeFreshnessClaim(queueValue, claim, {
  nowMs,
  needsFollowUp,
  retryDelayMs = 0,
  error = null,
}) {
  const queue = normalizeFreshnessQueue(queueValue);
  const current = queue.entries[claim.key];
  if (!current || current.generation !== claim.generation) return queue;
  const entries = { ...queue.entries };
  if (!needsFollowUp && !error) {
    delete entries[claim.key];
    return { ...queue, entries };
  }
  entries[claim.key] = {
    ...current,
    next_attempt_at_ms: nowMs + Math.max(1_000, retryDelayMs),
    attempt_count: error ? (current.attempt_count || 0) + 1 : current.attempt_count || 0,
    running_poll_count: needsFollowUp ? (current.running_poll_count || 0) + 1 : current.running_poll_count || 0,
    lease_owner: null,
    lease_expires_at_ms: null,
    last_error: error ? String(error) : null,
  };
  return { ...queue, entries };
}

export function chatGptCaptureNeedsFollowUp(envelope) {
  const payload = envelope?.raw_provider_payload;
  const mapping = payload?.mapping;
  const current = mapping && payload?.current_node ? mapping[payload.current_node]?.message : null;
  if (!current) return true;
  const role = current.author?.role;
  const status = current.status;
  if (role !== "assistant") return true;
  return !["finished_successfully", "finished", "complete", "completed"].includes(status);
}

export function runningPollDelayMs(pollCount) {
  if (pollCount <= 0) return 30_000;
  if (pollCount === 1) return 60_000;
  if (pollCount === 2) return 2 * 60_000;
  return 5 * 60_000;
}

export function failureRetryDelayMs(attemptCount, outcome, retryAfterSeconds = null) {
  if (retryAfterSeconds) return Math.max(1_000, retryAfterSeconds * 1_000);
  if (["rate_limited", "provider_warning", "safety_locked"].includes(outcome)) return 15 * 60_000;
  if (outcome === "auth_challenge") return 60 * 60_000;
  return Math.min(15 * 60_000, 15_000 * 2 ** Math.min(6, Math.max(0, attemptCount)));
}
