import { describe, expect, it } from "vitest";

import {
  CAPTURE_FRESHNESS_MAX_ENTRIES,
  chatGptCaptureNeedsFollowUp,
  claimDueFreshness,
  completeFreshnessClaim,
  failureRetryDelayMs,
  normalizeFreshnessQueue,
  runningPollDelayMs,
  scheduleFreshnessHint,
} from "../src/capture/freshness.js";

function hint(queue, nativeId, nowMs = 1000, patch = {}) {
  return scheduleFreshnessHint(queue, {
    provider: "chatgpt",
    nativeId,
    reason: "provider_dom_changed",
    nowMs,
    delayMs: 5000,
    ...patch,
  });
}

describe("capture freshness queue", () => {
  it("coalesces hints by provider-native identity without postponing earlier work", () => {
    const first = hint(null, "conversation-1", 1000);
    const repeated = hint(first, "conversation-1", 2000, {
      reason: "provider_push_new_message",
      delayMs: 10_000,
      providerUpdatedAt: "2026-07-16T00:00:00Z",
    });
    const entry = repeated.entries["chatgpt:conversation-1"];

    expect(Object.keys(repeated.entries)).toHaveLength(1);
    expect(entry.generation).toBe(2);
    expect(entry.next_attempt_at_ms).toBe(6000);
    expect(entry.reasons).toEqual(["provider_dom_changed", "provider_push_new_message"]);
    expect(entry.provider_updated_at).toBe("2026-07-16T00:00:00Z");
  });

  it("retains deduplicated lifecycle observations until the exact capture claim completes", () => {
    const started = {
      observation_id: "conversation-1:turn-2:started:1000",
      state: "started",
      observed_at: "2026-07-16T00:00:00Z",
    };
    const completed = {
      observation_id: "conversation-1:turn-2:completed:worked-for",
      state: "completed",
      observed_at: "2026-07-16T01:26:30Z",
    };
    const first = hint(null, "conversation-1", 1000, { generationObservations: [started] });
    const repeated = hint(first, "conversation-1", 2000, {
      delayMs: 0,
      generationObservations: [started, completed],
    });

    expect(repeated.entries["chatgpt:conversation-1"].generation_observations).toEqual([
      started,
      completed,
    ]);
    expect(claimDueFreshness(repeated, { nowMs: 2000, owner: "one", leaseMs: 5000 }).claim)
      .toMatchObject({ generation_observations: [started, completed] });
  });

  it("leases one due identity and recovers an expired lease", () => {
    let queue = hint(null, "later", 1000, { delayMs: 10_000 });
    queue = hint(queue, "due", 1000, { delayMs: 0 });
    const first = claimDueFreshness(queue, { nowMs: 1000, owner: "one", leaseMs: 5000 });
    expect(first.claim.native_id).toBe("due");
    expect(claimDueFreshness(first.queue, { nowMs: 2000, owner: "two", leaseMs: 5000 }).claim).toBeNull();
    expect(claimDueFreshness(first.queue, { nowMs: 6001, owner: "two", leaseMs: 5000 }).claim.native_id).toBe("due");
  });

  it("does not erase a newer hint when an older claim completes", () => {
    const initial = hint(null, "conversation-1", 1000, { delayMs: 0 });
    const { queue: leased, claim } = claimDueFreshness(initial, { nowMs: 1000, owner: "one", leaseMs: 5000 });
    const updated = hint(leased, "conversation-1", 1500, { reason: "provider_native_observed" });
    const completed = completeFreshnessClaim(updated, claim, {
      nowMs: 2000,
      needsFollowUp: false,
    });
    expect(completed.entries["chatgpt:conversation-1"].generation).toBe(2);
  });

  it("removes terminal captures and adaptively reschedules running replies", () => {
    const initial = hint(null, "conversation-1", 1000, { delayMs: 0 });
    const { queue: leased, claim } = claimDueFreshness(initial, { nowMs: 1000, owner: "one", leaseMs: 5000 });
    const running = completeFreshnessClaim(leased, claim, {
      nowMs: 2000,
      needsFollowUp: true,
      retryDelayMs: runningPollDelayMs(0),
    });
    expect(running.entries[claim.key].next_attempt_at_ms).toBe(32_000);
    expect(running.entries[claim.key].running_poll_count).toBe(1);

    const reclaimed = claimDueFreshness(running, { nowMs: 32_000, owner: "one", leaseMs: 5000 });
    const complete = completeFreshnessClaim(reclaimed.queue, reclaimed.claim, {
      nowMs: 33_000,
      needsFollowUp: false,
    });
    expect(complete.entries[claim.key]).toBeUndefined();
  });

  it("models terminal assistant heads and conservative non-terminal states", () => {
    const envelope = (role, status) => ({
      raw_provider_payload: {
        current_node: "head",
        mapping: { head: { message: { author: { role }, status } } },
      },
    });
    expect(chatGptCaptureNeedsFollowUp(envelope("assistant", "finished_successfully"))).toBe(false);
    expect(chatGptCaptureNeedsFollowUp(envelope("assistant", "in_progress"))).toBe(true);
    expect(chatGptCaptureNeedsFollowUp(envelope("user", "finished_successfully"))).toBe(true);
    expect(chatGptCaptureNeedsFollowUp({})).toBe(true);
  });

  it("uses typed backoff and bounds the durable hint set", () => {
    expect(failureRetryDelayMs(0, "rate_limited")).toBe(15 * 60_000);
    expect(failureRetryDelayMs(0, "auth_challenge")).toBe(60 * 60_000);
    expect(failureRetryDelayMs(2, "network_error")).toBe(60_000);
    expect(failureRetryDelayMs(0, "rate_limited", 7)).toBe(7000);

    let queue = normalizeFreshnessQueue(null);
    for (let index = 0; index < CAPTURE_FRESHNESS_MAX_ENTRIES + 2; index += 1) {
      queue = hint(queue, `conversation-${index}`, 1000 + index);
    }
    expect(Object.keys(queue.entries)).toHaveLength(CAPTURE_FRESHNESS_MAX_ENTRIES);
    expect(queue.dropped_count).toBe(2);
  });
});
