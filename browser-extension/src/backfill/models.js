export const BACKFILL_ALARM = "polylogueBackfillWake";
export const BACKFILL_DB_NAME = "polylogue-browser-backfill";
export const BACKFILL_DB_VERSION = 2;
export const BACKFILL_RECOVERY_CHECKPOINT_VERSION = 1;
export const PROVIDER_REQUEST_TIMEOUT_MS = 60000;
export const DURABLE_RECEIVER_ACK_FIELDS = Object.freeze(["receiver_request_id", "content_hash"]);

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

export const TERMINAL_QUEUE_STATES = new Set(["complete", "unchanged", "no_turns", "auth_required", "failed", "cancelled"]);

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
  if (!missing.length && receipt.content_hash === expectedContentHash) return null;
  const detail = missing.length ? `missing_${missing.join("_")}` : "content_hash_mismatch";
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
