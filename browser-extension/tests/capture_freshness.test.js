import { describe, expect, it } from "vitest";

import {
  chatGptCaptureNeedsFollowUp,
  claimDueFreshness,
  completeFreshnessClaim,
  extendProviderCooldown,
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

  it("settles a terminal claim after an unchanged native observation", () => {
    const initial = hint(null, "conversation-1", 1000, {
      delayMs: 0,
      providerUpdatedAt: "2026-07-16T00:00:00Z",
    });
    const { queue: leased, claim } = claimDueFreshness(initial, { nowMs: 1000, owner: "one", leaseMs: 5000 });
    const updated = hint(leased, "conversation-1", 1500, {
      reason: "provider_native_observed",
      providerUpdatedAt: "2026-07-16T00:00:00Z",
    });
    const completed = completeFreshnessClaim(updated, claim, {
      nowMs: 2000,
      needsFollowUp: false,
    });
    expect(completed.entries["chatgpt:conversation-1"]).toBeUndefined();
  });

  it("retains a newer revision or observation that arrives during a claim", () => {
    const initial = hint(null, "conversation-1", 1000, {
      delayMs: 0,
      providerUpdatedAt: "2026-07-16T00:00:00Z",
    });
    const { queue: leased, claim } = claimDueFreshness(initial, { nowMs: 1000, owner: "one", leaseMs: 5000 });
    const updated = hint(leased, "conversation-1", 1500, {
      providerUpdatedAt: "2026-07-16T00:01:00Z",
      generationObservations: [{ observation_id: "conversation-1:turn-2:completed", state: "completed" }],
    });
    const completed = completeFreshnessClaim(updated, claim, {
      nowMs: 2000,
      needsFollowUp: false,
    });
    expect(completed.entries["chatgpt:conversation-1"]).toMatchObject({
      generation: 2,
      provider_updated_at: "2026-07-16T00:01:00Z",
      generation_observations: [{ observation_id: "conversation-1:turn-2:completed" }],
    });
  });

  it("keeps newer hints leased until the original capture settles", () => {
    const initial = hint(null, "conversation-1", 1000, { delayMs: 0 });
    const first = claimDueFreshness(initial, { nowMs: 1000, owner: "one", leaseMs: 5000 });
    const updated = hint(first.queue, "conversation-1", 1500, {
      delayMs: 0,
      generationObservations: [{ observation_id: "new-completion", state: "completed" }],
    });
    expect(updated.entries[first.claim.key]).toMatchObject({
      generation: 2, lease_owner: "one", lease_expires_at_ms: 6000,
    });
    expect(claimDueFreshness(updated, { nowMs: 2000, owner: "two", leaseMs: 5000 }).claim).toBeNull();
    const settled = completeFreshnessClaim(updated, first.claim, { nowMs: 2000, needsFollowUp: false });
    const next = claimDueFreshness(settled, { nowMs: 2000, owner: "two", leaseMs: 5000 });
    expect(next.claim).toMatchObject({ generation: 2, generation_observations: [{ observation_id: "new-completion" }] });
    expect(completeFreshnessClaim(next.queue, next.claim, { nowMs: 2500, needsFollowUp: false }).entries).toEqual({});
  });

  it.each(["one", "two"])("fences a late expired completion from the reclaimed %s lease", (owner) => {
    const initial = hint(null, "conversation-1", 1000, { delayMs: 0 });
    const first = claimDueFreshness(initial, { nowMs: 1000, owner: "one", leaseMs: 5000 });
    const next = claimDueFreshness(first.queue, { nowMs: 6001, owner, leaseMs: 5000 });
    expect(next.claim.generation).toBe(first.claim.generation);
    expect(next.claim.lease_expires_at_ms).not.toBe(first.claim.lease_expires_at_ms);
    for (const result of [{ needsFollowUp: false }, { needsFollowUp: true }, { error: "network_error" }]) {
      expect(completeFreshnessClaim(next.queue, first.claim, { nowMs: 6500, ...result })).toEqual(next.queue);
    }
    expect(completeFreshnessClaim(next.queue, next.claim, { nowMs: 6500, needsFollowUp: false }).entries).toEqual({});
  });

  it("retains typed retry and provider cooldown when newer evidence arrives during failure", () => {
    const initial = hint(null, "conversation-1", 1000, { delayMs: 0 });
    const first = claimDueFreshness(initial, { nowMs: 1000, owner: "one", leaseMs: 5000 });
    const updated = hint(first.queue, "conversation-1", 1500, { delayMs: 0, providerUpdatedAt: "2026-07-16T00:01:00Z" });
    const cooled = extendProviderCooldown(updated, { provider: "chatgpt", untilMs: 61000, nowMs: 2000 });
    const settled = completeFreshnessClaim(cooled, first.claim, { nowMs: 2000, needsFollowUp: false, error: "rate_limited", retryDelayMs: 7000 });
    expect(settled.entries[first.claim.key]).toMatchObject({ generation: 2, lease_owner: null, next_attempt_at_ms: 9000, attempt_count: 1, last_error: "rate_limited" });
    expect(claimDueFreshness(settled, { nowMs: 60999, owner: "two", leaseMs: 5000 }).claim).toBeNull();
    expect(claimDueFreshness(settled, { nowMs: 61000, owner: "two", leaseMs: 5000 }).claim.provider_updated_at).toBe("2026-07-16T00:01:00Z");
  });

  it("holds every conversation for a provider until its throttle deadline", () => {
    let queue = hint(null, "conversation-1", 1000, { delayMs: 0 });
    queue = hint(queue, "conversation-2", 1000, { delayMs: 0 });
    queue = extendProviderCooldown(queue, { provider: "chatgpt", untilMs: 61_000 });
    queue = hint(queue, "conversation-3", 2000, { delayMs: 0 });

    expect(claimDueFreshness(queue, { nowMs: 60_999, owner: "one", leaseMs: 5000 }).claim).toBeNull();
    expect(claimDueFreshness(queue, { nowMs: 61_000, owner: "one", leaseMs: 5000 }).claim?.native_id)
      .toBe("conversation-1");
  });

  it("preserves a forty-eight hour provider deadline and rejects physically unrepresentable timestamps", () => {
    const deadline = 1000 + 48 * 60 * 60 * 1000;
    const queue = extendProviderCooldown(null, { provider: "chatgpt", untilMs: deadline, nowMs: 1000 });
    expect(queue.provider_cooldowns.chatgpt).toBe(deadline);
    expect(() => extendProviderCooldown(queue, { provider: "chatgpt", untilMs: 1e307, nowMs: 1000 })).toThrow("provider_retry_after_unrepresentable");
    expect(queue.provider_cooldowns.chatgpt).toBe(deadline);
    const due = hint(queue, "later-conversation", 1000);
    expect(claimDueFreshness(due, { nowMs: deadline - 1, owner: "one", leaseMs: 5000 }).claim).toBeNull();
    expect(claimDueFreshness(due, { nowMs: deadline, owner: "one", leaseMs: 5000 }).claim).not.toBeNull();
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

  it("keeps freshness pending until canonical receiver preparation reports a terminal head", () => {
    expect(chatGptCaptureNeedsFollowUp({ capture_summary: { needsFollowUp: false } })).toBe(false);
    expect(chatGptCaptureNeedsFollowUp({ capture_summary: { needsFollowUp: true } })).toBe(true);
    expect(chatGptCaptureNeedsFollowUp({})).toBe(true);
  });

  it("uses typed backoff and preserves every durable hint past the former count limit", () => {
    expect(failureRetryDelayMs(0, "rate_limited")).toBe(15 * 60_000);
    expect(failureRetryDelayMs(0, "auth_challenge")).toBe(60 * 60_000);
    expect(failureRetryDelayMs(2, "network_error")).toBe(60_000);
    expect(failureRetryDelayMs(0, "rate_limited", 7)).toBe(7000);

    let queue = normalizeFreshnessQueue(null);
    for (let index = 0; index < 502; index += 1) {
      queue = hint(queue, `conversation-${index}`, 1000 + index);
    }
    expect(Object.keys(queue.entries)).toHaveLength(502);
    expect(queue.dropped_count).toBe(0);
    expect(queue.entries["chatgpt:conversation-0"]).toBeDefined();
    expect(queue.entries["chatgpt:conversation-501"]).toBeDefined();
  });
});
