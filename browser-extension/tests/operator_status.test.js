import { beforeEach, describe, expect, it, vi } from "vitest";

async function loadApi() {
  delete globalThis.PolylogueOperatorStatus;
  vi.resetModules();
  await import("../src/operator_status.js");
  return globalThis.PolylogueOperatorStatus;
}

describe("shared operator status vocabulary", () => {
  let api;

  beforeEach(async () => {
    api = await loadApi();
  });

  it("maps archive, receiver, provider, and fidelity states to the canonical vocabulary", () => {
    const cases = [
      [{ online: true, archive_state: { state: "archived" } }, "safe_current", "Safe / current"],
      [{ online: true, archive_state: { state: "archived" }, capture_freshness: { next_attempt_at_ms: 1 } }, "catching_up", "Catching up"],
      [{ online: true, archive_state: { state: "spooled_only" } }, "catching_up", "Catching up"],
      [{ online: true, archive_state: { state: "ingest_pending" } }, "catching_up", "Catching up"],
      [{ online: true, archive_state: { state: "stale" } }, "catching_up", "Catching up"],
      [{ online: true, archive_state: { state: "missing" } }, "catching_up", "Catching up"],
      [{ online: true, archive_state: { state: "failed", latest_failure: "index rejected" } }, "failed", "Failed"],
      [{ online: false }, "receiver_offline", "Receiver offline"],
      [{ online: true, cooldown_reason: "rate_limited" }, "provider_warning", "Provider warning"],
    ];

    for (const [state, code, label] of cases) {
      expect(api.operatorStatusForState(state)).toMatchObject({ code, label });
    }

    const partial = api.operatorStatusForState({
      online: true,
      archive_state: { state: "archived" },
      capture_mode: "dom_degraded",
    });
    expect(partial).toMatchObject({ code: "safe_current", partialFidelity: true });
    expect(partial.fidelity).toMatchObject({ code: "partial_fidelity", label: "Partial fidelity" });
    expect(api.operatorStatusLabel({
      online: true,
      archive_state: { state: "archived" },
      capture_mode: "dom_degraded",
    })).toBe("Safe / current · Partial fidelity");
  });

  it("treats ordinary and provider landing pages as non-conversations", () => {
    for (const activePageState of ["unsupported", "supported_no_session"]) {
      const status = api.operatorStatusForState({
        online: false,
        error: "receiver_pairing_mismatch",
        active_page_state: activePageState,
        capture_mode: "dom_degraded",
      });
      expect(status).toMatchObject({
        code: "not_conversation",
        label: "No conversation",
        tone: "neutral",
        partialFidelity: false,
        fidelity: null,
      });
    }

    expect(api.operatorPresentationForState({ active_page_state: "unsupported" })).toMatchObject({
      badge: ["neutral", "idle"],
      archive: "Not applicable",
      headline: "Ordinary webpage",
    });
    expect(api.operatorPresentationForState({ active_page_state: "supported_no_session" })).toMatchObject({
      badge: ["neutral", "idle"],
      archive: "Not applicable",
      headline: "No conversation selected",
    });
  });

  it("normalizes capture retry and history backfill into one readable contract", () => {
    vi.setSystemTime(new Date("2026-07-16T12:00:00Z"));
    const items = api.normalizeWorkItems({
      receiverOnline: true,
      ownerInstanceId: "executor-this",
      captureQueue: {
        entries: [{
          id: "capture-1",
          enqueued_at: "2026-07-16T11:58:00Z",
          next_attempt_at: "2026-07-16T12:01:00Z",
          last_error: "receiver unavailable",
          envelope: {
            session: {
              provider: "chatgpt",
              provider_session_id: "conversation-1",
              title: "Architecture review",
            },
          },
        }],
      },
      freshnessQueue: {
        entries: {
          "chatgpt:conversation-2": {
            key: "chatgpt:conversation-2",
            provider: "chatgpt",
            native_id: "conversation-2",
            hinted_at: "2026-07-16T11:59:30Z",
            next_attempt_at_ms: Date.parse("2026-07-16T12:02:00Z"),
            reasons: ["provider_push_new_message"],
          },
        },
      },
      backfillJobs: [{
        id: "backfill-1",
        provider: "claude-ai",
        cutoff: "2026-04-01T00:00:00Z",
        status: "running",
        phase: "conversation_capture",
        learned_cadence_ms: 15000,
        lease_owner: "extension-backfill",
        updated_at: "2026-07-16T11:59:00Z",
        last_ack: { receiver_request_id: "ack-22" },
      }],
    });

    expect(items).toHaveLength(3);
    expect(items.find((item) => item.kind === "capture_retry")).toMatchObject({
      title: "Architecture review",
      status: { code: "queued", label: "Queued" },
      phase: "Waiting to deliver capture",
      cadence: "Due in 1m",
      owner: "This extension",
      receipt: "Receiver acknowledgement pending",
    });
    expect(items.find((item) => item.kind === "backfill")).toMatchObject({
      title: "Claude.ai history backfill since 2026-04-01",
      status: { code: "running", label: "Running" },
      phase: "conversation_capture",
      cadence: "15s cadence",
      owner: "extension-backfill",
      receipt: "Last receiver ACK ack-22",
    });
    expect(items.find((item) => item.kind === "freshness")).toMatchObject({
      title: "ChatGPT conversation conversation-2",
      status: { code: "queued", label: "Queued" },
      phase: "Waiting for freshness check",
      cadence: "Due in 2m",
      receipt: "provider_push_new_message",
    });
    vi.useRealTimers();
  });

  it("separates automatic waits, provider circuits, and operator-blocked work", () => {
    const items = api.normalizeWorkItems({
      receiverOnline: true,
      backfillJobs: [
        { id: "transport", provider: "chatgpt", status: "running", cooldown_reason: "transport_backoff" },
        { id: "rate", provider: "chatgpt", status: "running", cooldown_reason: "provider_rate_limited" },
        { id: "auth", provider: "claude-ai", status: "paused", cooldown_reason: "provider_auth_or_challenge" },
      ],
    });

    expect(items.find((item) => item.id === "transport").status).toMatchObject({ code: "queued", label: "Queued" });
    expect(items.find((item) => item.id === "rate").status).toMatchObject({ code: "provider_warning", label: "Provider warning" });
    expect(items.find((item) => item.id === "auth").status).toMatchObject({ code: "needs_attention", label: "Needs attention" });
  });

  it("marks unfinished capture and backfill work receiver-offline", () => {
    const items = api.normalizeWorkItems({
      receiverOnline: false,
      captureQueue: { entries: [{ id: "capture", envelope: { session: { provider: "chatgpt", provider_session_id: "c1" } } }] },
      backfillJobs: [{ id: "backfill", provider: "chatgpt", status: "running" }],
    });

    expect(items.find((item) => item.id === "capture").status.code).toBe("receiver_offline");
    expect(items.find((item) => item.id === "backfill").status.code).toBe("receiver_offline");
  });

  it("presents explicit no-op and recovery events instead of hiding them", () => {
    expect(api.eventPresentation({ event: "observed_no_action", detail: "already_safe" })).toEqual({
      label: "Observed; no action needed",
      detail: "Archive was already current, so Polylogue did not create a duplicate capture.",
    });
    expect(api.eventPresentation({ event: "observed_no_action", detail: "receiver_already_processing" })).toEqual({
      label: "Observed; no action needed",
      detail: "Durable evidence already existed and the receiver was still catching up.",
    });
    expect(api.eventPresentation({ event: "receiver_recovered", detail: "old -> canonical" })).toEqual({
      label: "Receiver connection recovered",
      detail: "old -> canonical",
    });
  });

  it("distinguishes paired, offline, recovered, unauthorized, and mismatched receivers", () => {
    const paired = api.receiverPairingPresentation({
      pairing: {
        state: "online",
        receiver_id: "rx-123",
        api_schema: "polylogue-browser-capture/v1",
        endpoint: "http://127.0.0.1:8765",
      },
      health: { status: "ok" },
    });
    expect(paired).toMatchObject({
      status: { code: "safe_current" },
      headline: "Paired receiver rx-123",
    });

    expect(api.receiverPairingPresentation({
      pairing: { state: "offline", receiver_id: "rx-123" },
      health: { status: "unreachable" },
    }).status.code).toBe("receiver_offline");
    expect(api.receiverPairingPresentation({ health: { status: "recovered", endpoint: "http://127.0.0.1:8765" } }).headline)
      .toBe("Receiver recovered");
    expect(api.receiverPairingPresentation({ health: { status: "unauthorized" } }).headline)
      .toBe("Pairing token required");
    expect(api.receiverPairingPresentation({
      pairing: { state: "mismatch" },
      health: { status: "pairing_mismatch" },
    })).toMatchObject({
      status: { code: "needs_attention" },
      headline: "Receiver identity changed",
    });
  });
});
