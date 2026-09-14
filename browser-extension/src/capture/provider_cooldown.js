/**
 * Single source of truth for the maximum provider cooldown the extension will
 * ever honour.
 *
 * A provider-supplied `Retry-After` reaches this code from attacker-influenced
 * ground: the ChatGPT MAIN world is page script, and any page script can post a
 * `polylogue.chatgpt.nativeFetchResponse` message carrying `{status: 429,
 * retryAfter: "315360000"}`. That number flows content script -> background ->
 * `extendProviderCooldown`, which is monotonic (`Math.max`) and persisted in
 * `chrome.storage.local`, so a single forged value can disable manual capture,
 * automatic capture, freshness convergence, backfill and queued browser actions
 * across browser restarts, recoverable only by clearing extension storage.
 *
 * Ceiling: 24 hours. Real provider back-off is minutes to hours -- the longest
 * genuine ChatGPT/Claude rate-limit windows are the 3-hour message caps and
 * 24-hour daily caps, so a full day is the largest value that can still be an
 * honest provider instruction. Anything beyond it is either a provider bug or a
 * forgery, and in both cases retrying a day later is the correct behaviour: the
 * clamp costs at most one wasted attempt per day, while the unclamped path costs
 * the entire capture pipeline indefinitely.
 *
 * Content scripts (`src/content/chatgpt.js`) are classic IIFE scripts in the
 * manifest, not ES modules, so they cannot import this file; that one mirrors
 * the literal and `tests/content/chatgpt.test.js` asserts the two agree.
 */
export const MAX_PROVIDER_COOLDOWN_MS = 24 * 60 * 60 * 1000;

/**
 * Clamp a provider-supplied cooldown duration.
 *
 * Returns `{ valueMs, clamped, requestedMs }`. `clamped` is deliberately part of
 * the result so every call site can surface the fact that a provider-supplied
 * number was overridden rather than silently rewriting it.
 */
export function clampProviderCooldownMs(valueMs) {
  const numeric = Number(valueMs);
  if (!Number.isFinite(numeric)) return { valueMs: numeric, clamped: false, requestedMs: numeric };
  if (numeric <= MAX_PROVIDER_COOLDOWN_MS) return { valueMs: numeric, clamped: false, requestedMs: numeric };
  return { valueMs: MAX_PROVIDER_COOLDOWN_MS, clamped: true, requestedMs: numeric };
}
