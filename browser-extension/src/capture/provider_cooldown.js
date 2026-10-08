/** Validate the physical JavaScript timestamp representation, without changing a provider's deadline. */
export function requireProviderCooldownMs(delayMs, nowMs = Date.now()) {
  const delay = Number(delayMs);
  if (!Number.isFinite(delay) || delay < 0 || !Number.isFinite(new Date(nowMs + delay).getTime())) {
    const error = new Error("provider_retry_after_unrepresentable");
    error.code = "provider_retry_after_unrepresentable";
    throw error;
  }
  return delay;
}
