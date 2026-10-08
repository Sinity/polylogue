import { requireProviderCooldownMs } from "../capture/provider_cooldown.js";

export const BACKFILL_ALARM = "polylogueBackfillWake";
export const BACKFILL_DB_NAME = "polylogue-browser-backfill";
export const BACKFILL_DB_VERSION = 4;
export const BACKFILL_RECOVERY_CHECKPOINT_VERSION = 1;
export const DURABLE_RECEIVER_ACK_FIELDS = Object.freeze(["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"]);

export const DEFAULT_BACKFILL_POLICY = Object.freeze({
  maxCapturesPerWake: 5,
  maxDailyRequests: 250,
  leaseMs: 180000,
  baseCadenceMs: 15000,
  maxCadenceMs: 15 * 60 * 1000,
});

export const TERMINAL_QUEUE_STATES = new Set(["complete", "unchanged", "superseded", "no_turns", "auth_required", "failed", "cancelled"]);

export function backfillAlarmName(jobId) {
  return `${BACKFILL_ALARM}:${jobId}`;
}

export function receiverAckContractError(receipt, expectedContentHash) {
  const missing = DURABLE_RECEIVER_ACK_FIELDS.filter((field) => {
    const value = receipt?.[field];
    return typeof value !== "string" || !value;
  });
  let detail = missing.length ? `missing_${missing.join("_")}` : null;
  if (!detail && !["accepted", "noop", "superseded"].includes(receipt.outcome)) detail = "outcome_invalid";
  if (!detail && receipt.submitted_content_hash !== expectedContentHash) detail = "submitted_content_hash_mismatch";
  if (!detail && receipt.outcome === "accepted" && receipt.content_hash !== expectedContentHash) detail = "content_hash_mismatch";
  if (!detail) return null;
  const error = new Error(`receiver_contract_incompatible:${detail}`);
  error.code = "receiver_contract_incompatible";
  return error;
}

export function retryAfterMs(headers, nowMs) {
  const value = headers?.get?.("Retry-After");
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return requireProviderCooldownMs(Math.ceil(seconds * 1000), nowMs);
  const deadline = Date.parse(value);
  return Number.isFinite(deadline) ? requireProviderCooldownMs(Math.max(0, deadline - nowMs), nowMs) : null;
}

export function fullJitterDelay(attempt, baseMs, maxMs, random = Math.random) {
  const ceiling = Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt));
  return Math.floor(random() * ceiling);
}

export function dayKey(nowMs) {
  return new Date(nowMs).toISOString().slice(0, 10);
}
