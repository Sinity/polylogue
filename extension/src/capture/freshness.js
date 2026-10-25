import { requireProviderCooldownMs } from "./provider_cooldown.js";

export const CAPTURE_FRESHNESS_QUEUE_VERSION = 1;

function entryKey(provider, nativeId) {
  return `${provider}:${nativeId}`;
}

function isNewerProviderRevision(previous, candidate) {
  if (!candidate) return false;
  if (!previous) return true;
  const previousMs = Date.parse(previous);
  const candidateMs = Date.parse(candidate);
  if (Number.isFinite(previousMs) && Number.isFinite(candidateMs)) return candidateMs > previousMs;
  return candidate !== previous;
}

export function normalizeFreshnessQueue(value) {
  const entries = value?.version === CAPTURE_FRESHNESS_QUEUE_VERSION
    && value.entries && typeof value.entries === "object"
    ? value.entries
    : {};
  return {
    version: CAPTURE_FRESHNESS_QUEUE_VERSION,
    entries: { ...entries },
    provider_cooldowns: {
      ...(value?.provider_cooldowns && typeof value.provider_cooldowns === "object"
        ? value.provider_cooldowns
        : {}),
    },
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
  generationObservations = [],
}) {
  const queue = normalizeFreshnessQueue(queueValue);
  const key = entryKey(provider, nativeId);
  const previous = queue.entries[key] || null;
  const requestedAt = nowMs + Math.max(0, delayMs);
  const previousObservations = new Set(
    (previous?.generation_observations || []).map((observation) => observation?.observation_id).filter(Boolean),
  );
  const hasNewObservation = generationObservations.some((observation) => (
    observation?.observation_id && !previousObservations.has(observation.observation_id)
  ));
  const hasNewRevision = isNewerProviderRevision(previous?.provider_updated_at, providerUpdatedAt);
  if (previous && reason === "provider_native_observed" && !hasNewObservation && !hasNewRevision) return queue;
  const observationsById = new Map(
    [...(previous?.generation_observations || []), ...generationObservations]
      .filter((observation) => observation?.observation_id)
      .map((observation) => [observation.observation_id, observation]),
  );
  const entry = {
    key,
    provider,
    native_id: nativeId,
    reasons: [...new Set([...(previous?.reasons || []), reason].filter(Boolean))],
    provider_updated_at: hasNewRevision ? providerUpdatedAt : previous?.provider_updated_at || providerUpdatedAt || null,
    generation_observations: [...observationsById.values()],
    generation: (previous?.generation || 0) + 1,
    hinted_at: new Date(nowMs).toISOString(),
    first_hinted_at: previous?.first_hinted_at || new Date(nowMs).toISOString(),
    next_attempt_at_ms: previous
      ? Math.min(previous.next_attempt_at_ms || requestedAt, requestedAt)
      : requestedAt,
    attempt_count: previous?.attempt_count || 0,
    running_poll_count: previous?.running_poll_count || 0,
    // New evidence belongs to the pending generation; it does not release
    // the capture already holding this identity's claim.
    claimed_by: previous?.claimed_by ?? null,
    claim_expires_ms: previous?.claim_expires_ms ?? null,
    last_error: previous?.last_error || null,
  };
  const entries = { ...queue.entries, [key]: entry };
  return { ...queue, entries };
}

export function extendProviderCooldown(queueValue, { provider, untilMs, nowMs = Date.now() }) {
  const queue = normalizeFreshnessQueue(queueValue);
  const current = Number(queue.provider_cooldowns[provider]) || 0;
  const requested = Number(untilMs) || 0;
  requireProviderCooldownMs(Math.max(0, requested - nowMs), nowMs);
  const deadline = Math.max(current, requested);
  if (deadline === current) return queue;
  return { ...queue, provider_cooldowns: { ...queue.provider_cooldowns, [provider]: deadline } };

}

export function claimDueFreshness(queueValue, { nowMs, owner, claimMs }) {
  const queue = normalizeFreshnessQueue(queueValue);
  const due = Object.values(queue.entries)
    .filter((entry) => (
      (queue.provider_cooldowns[entry.provider] || 0) <= nowMs
      &&
      (entry.claimed_by === null || (entry.claim_expires_ms || 0) <= nowMs)
      && (entry.next_attempt_at_ms || 0) <= nowMs
    ))
    .sort((left, right) => (
      (left.next_attempt_at_ms || 0) - (right.next_attempt_at_ms || 0)
      || Date.parse(left.first_hinted_at) - Date.parse(right.first_hinted_at)
    ))[0];
  if (!due) return { queue, claim: null };
  const claimed = {
    ...due,
    claimed_by: owner,
    claim_expires_ms: nowMs + claimMs,
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
  if (!current || !claim.claimed_by
    || current.claimed_by !== claim.claimed_by
    || current.claim_expires_ms !== claim.claim_expires_ms) return queue;
  const entries = { ...queue.entries };
  if (current.generation === claim.generation && !needsFollowUp && !error) {
    delete entries[claim.key];
    return { ...queue, entries };
  }
  entries[claim.key] = {
    ...current,
    next_attempt_at_ms: needsFollowUp || error
      ? Math.max(current.next_attempt_at_ms || 0, nowMs + Math.max(1_000, retryDelayMs))
      : current.next_attempt_at_ms,
    attempt_count: error ? (current.attempt_count || 0) + 1 : current.attempt_count || 0,
    running_poll_count: needsFollowUp ? (current.running_poll_count || 0) + 1 : current.running_poll_count || 0,
    claimed_by: null,
    claim_expires_ms: null,
    last_error: error ? String(error) : null,
  };
  return { ...queue, entries };
}

export function chatGptCaptureNeedsFollowUp(envelope) {
  return envelope?.capture_summary?.needsFollowUp !== false;
}

export function runningPollDelayMs(pollCount) {
  if (pollCount <= 0) return 30_000;
  if (pollCount === 1) return 60_000;
  if (pollCount === 2) return 2 * 60_000;
  return 5 * 60_000;
}

export function failureRetryDelayMs(attemptCount, outcome, retryAfterSeconds = null) {
  if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0) {
    return Math.max(1_000, retryAfterSeconds * 1_000);
  }
  if (["rate_limited", "provider_warning", "safety_locked"].includes(outcome)) return 15 * 60_000;
  if (outcome === "auth_challenge") return 60 * 60_000;
  return Math.min(15 * 60_000, 15_000 * 2 ** Math.min(6, Math.max(0, attemptCount)));
}
