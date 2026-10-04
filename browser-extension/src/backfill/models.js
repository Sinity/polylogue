export const BACKFILL_ALARM = "polylogueBackfillWake";
export const BACKFILL_DB_NAME = "polylogue-browser-backfill";
export const BACKFILL_DB_VERSION = 3;
export const BACKFILL_RECOVERY_CHECKPOINT_VERSION = 1;
export const PROVIDER_REQUEST_TIMEOUT_MS = 60000;
export const DURABLE_RECEIVER_ACK_FIELDS = Object.freeze(["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"]);

export const DEFAULT_BACKFILL_POLICY = Object.freeze({
  maxQueueSize: 10000,
  maxCapturesPerWake: 5,
  maxDailyRequests: 250,
  leaseMs: 180000,
  baseCadenceMs: 15000,
  maxCadenceMs: 15 * 60 * 1000,
  maxTransportAttempts: 5,
  maxReceiverAttempts: 20,
  maxStoredBytes: 100 * 1024 * 1024,
  breakerThreshold: 2,
});

export const TERMINAL_QUEUE_STATES = new Set(["complete", "unchanged", "superseded", "no_turns", "auth_required", "bridge_oversize", "failed", "cancelled"]);

export function backfillAlarmName(jobId) {
  return `${BACKFILL_ALARM}:${jobId}`;
}

export function serializedJson(value) {
  return JSON.stringify(value);
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

export async function serializedContentHash(serialized) {
  const bytes = new TextEncoder().encode(serialized);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function retryAfterMs(headers, nowMs) {
  const value = headers?.get?.("Retry-After");
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return Math.ceil(seconds * 1000);
  const deadline = Date.parse(value);
  return Number.isFinite(deadline) ? Math.max(0, deadline - nowMs) : null;
}

export function fullJitterDelay(attempt, baseMs, maxMs, random = Math.random) {
  const ceiling = Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt));
  return Math.floor(random() * ceiling);
}

export function dayKey(nowMs) {
  return new Date(nowMs).toISOString().slice(0, 10);
}
