"""Browser-capture receiver route contracts.

The browser-capture receiver is a small local ingest boundary, separate from
the daemon web/API routes. Keeping its route metadata beside the receiver makes
the auth and DTO posture explicit without teaching callers to inspect handler
branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BrowserCaptureAuthPolicy = Literal[
    "extension_origin",
    "bearer_if_web_origin",
    "bearer_if_configured",
    "bearer_required",
    # No bearer token accepted or required -- the route's whole purpose is
    # minting/exchanging that token for an unpaired caller. Origin is still
    # enforced (extension-only); the short-lived single-use code carried in
    # the request body is this route's credential (polylogue-gnie).
    "unauthenticated_pairing_exchange",
    # No bearer token accepted or required -- a client must authenticate the
    # receiver before it sends the bearer anywhere. The answer is an HMAC,
    # keyed by the bearer, over the client's fresh challenge; it proves
    # possession without revealing the bearer.
    "unauthenticated_receiver_attestation",
]
BrowserCaptureRouteKind = Literal[
    "capabilities",
    "status",
    "archive_state",
    "mission_control",
    "capture_ingest",
    "assertion_candidate_capture",
    "browser_action_capabilities",
    "browser_action_enqueue",
    "browser_action_list_claim",
    "browser_action_read",
    "browser_action_attachment",
    "browser_action_attachment_upload",
    "browser_action_update",
    "browser_action_reconcile",
    "browser_action_approval",
    "capture_job_capabilities",
    "capture_job_create",
    "capture_job_discover",
    "capture_job_read",
    "capture_job_orphan_list",
    "capture_job_orphan_inspect",
    "capture_job_adopt",
    "capture_job_update",
    "capture_job_checkpoint",
    "capture_job_checkpoint_artifact",
    "capture_job_event_append",
    "capture_job_event_read",
    "capture_job_native_begin",
    "capture_job_native_cancel",
    "capture_job_native_member",
    "capture_job_native_prepare",
    "capture_job_native_plan",
    "capture_job_native_asset",
    "capture_job_native_finalize",
    "capture_job_native_publish",
    "pairing_redeem",
    "receiver_attestation",
    "capture_health_report",
    "capture_health_list",
]


@dataclass(frozen=True, slots=True)
class BrowserCaptureRouteContract:
    """Machine-readable contract for one browser-capture receiver route."""

    method: Literal["GET", "POST", "PUT"]
    pattern: str
    kind: BrowserCaptureRouteKind
    auth_policy: BrowserCaptureAuthPolicy
    request_contract: str | None
    response_contract: str
    notes: str = ""


BROWSER_CAPTURE_ROUTE_CONTRACTS: tuple[BrowserCaptureRouteContract, ...] = (
    BrowserCaptureRouteContract(
        "GET",
        "/v1/browser-captures/capabilities",
        "capabilities",
        "bearer_if_configured",
        None,
        "BrowserCaptureCapabilitiesPayload",
        "Declares the durable acknowledgement fields required by browser backfill.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/status",
        "status",
        "bearer_if_configured",
        None,
        "BrowserCaptureReceiverStatusPayload",
        "Reports receiver readiness and whether bearer auth is required.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/archive-state",
        "archive_state",
        "bearer_if_configured",
        "provider + provider_session_id query parameters",
        "BrowserCaptureArchiveStatePayload",
        "Reports whether a provider session is spooled, using a receiver-local artifact ref.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/mission-control",
        "mission_control",
        "bearer_if_configured",
        "provider + provider_session_id query parameters",
        "typed Layer 2 intelligence projection",
        "Reads canonical archive, usage provenance, and judged assertions; never inspects provider DOM.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/browser-captures",
        "capture_ingest",
        "bearer_if_web_origin",
        "BrowserCaptureEnvelope",
        "BrowserCaptureAcceptedPayload | BrowserCaptureErrorPayload",
        "Accepts captures and returns a receiver-local artifact ref; extra web origins require bearer auth.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/assertion-candidates",
        "assertion_candidate_capture",
        "bearer_if_web_origin",
        "candidate body, kind, exact evidence_refs, source_observation, idempotency_key",
        "candidate assertion envelope | BrowserCaptureErrorPayload",
        "Writes a user-tier candidate with inject=false; native message evidence is required.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/browser-actions/capabilities",
        "browser_action_capabilities",
        "bearer_if_configured",
        None,
        "BrowserActionCapabilitiesPayload",
        "Declares exact provider operations, presentation choices, project routing, and attachment transfer units.",
    ),
    BrowserCaptureRouteContract(
        "PUT",
        "/v1/browser-action-attachments",
        "browser_action_attachment_upload",
        "bearer_if_configured",
        "streamed attachment bytes",
        "attachment_ref, size_bytes | BrowserCaptureErrorPayload",
        "Durably retains an immutable original input before returning its SHA-256 reference for enqueue.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/browser-actions",
        "browser_action_enqueue",
        "bearer_if_web_origin",
        "BrowserActionRequest",
        "BrowserActionPayload | BrowserCaptureErrorPayload",
        "Copies hash-pinned input attachments and durably enqueues one idempotent provider action.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/browser-actions",
        "browser_action_list_claim",
        "bearer_if_configured",
        "optional claim_by extension-instance query parameter",
        "BrowserActionListPayload | BrowserCaptureErrorPayload",
        "Lists actions or atomically leases one action to a replaceable extension instance.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/browser-actions/{action_id}",
        "browser_action_read",
        "bearer_if_configured",
        None,
        "BrowserActionPayload | BrowserCaptureErrorPayload",
        "Reads one durable action and its exact provider receipt.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/browser-actions/{action_id}/attachments/{attachment_id}",
        "browser_action_attachment",
        "bearer_if_configured",
        None,
        "immutable attachment bytes | BrowserCaptureErrorPayload",
        "Returns receiver-owned bytes after size and SHA-256 integrity verification.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/browser-actions/{action_id}/events",
        "browser_action_update",
        "bearer_if_web_origin",
        "BrowserActionUpdateRequest",
        "BrowserActionPayload | BrowserCaptureErrorPayload",
        "Renews a lease or records a typed draft, submit, uncertainty, rate, auth, or drift outcome.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/browser-actions/{action_id}/reconcile",
        "browser_action_reconcile",
        "bearer_if_web_origin",
        "BrowserActionReconcileRequest",
        "BrowserActionPayload | BrowserCaptureErrorPayload",
        "Explicitly binds provider evidence to an outcome_unknown action; never retries it implicitly.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/browser-actions/{action_id}/approval",
        "browser_action_approval",
        "bearer_if_web_origin",
        "BrowserActionApprovalDecisionRequest",
        "BrowserActionPayload | BrowserCaptureErrorPayload",
        "Records the operator's explicit approve/decline decision on an awaiting_approval action; "
        "approve makes it claimable for the first time, decline is terminal.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-jobs/capabilities",
        "capture_job_capabilities",
        "bearer_if_configured",
        None,
        "CaptureJob capabilities v1 | BrowserCaptureErrorPayload",
        "Publishes protocol bounds and the bearer-rotation-stable scope namespace.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs",
        "capture_job_create",
        "bearer_if_web_origin",
        "CaptureJob create request v1",
        "CaptureJob create response v1 | BrowserCaptureErrorPayload",
        "Creates or idempotently returns a receiver-authoritative exact-scope capture job.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/discover",
        "capture_job_discover",
        "bearer_if_web_origin",
        "CaptureJob scope query v1 + optional created_at/job_id cursor",
        "CaptureJob discovery response v1 | BrowserCaptureErrorPayload",
        "Pages jobs in the keyed provider/account scope by immutable creation identity.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-jobs/orphans",
        "capture_job_orphan_list",
        "bearer_if_configured",
        "client_protocol + optional source_digest cursor query parameters",
        "CaptureJob typed orphan census v1 | BrowserCaptureErrorPayload",
        "Pages global retained checkpoint fingerprints only on an explicit operator route.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-jobs/{job_id}",
        "capture_job_read",
        "bearer_if_configured",
        "provider + JSON scope + client_protocol query parameters",
        "CaptureJob detail and receipts v1 | BrowserCaptureErrorPayload",
        "Reads one exact-scope job and its durable checkpoint/update receipts.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/adopt",
        "capture_job_adopt",
        "bearer_if_web_origin",
        "CaptureJob adoption request v1",
        "CaptureJob lease response v1 | BrowserCaptureErrorPayload",
        "Acquires or resumes a replaceable expiring client lease under revision CAS.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/update",
        "capture_job_update",
        "bearer_if_web_origin",
        "CaptureJob update request v1",
        "CaptureJob update receipt v1 | BrowserCaptureErrorPayload",
        "CAS-updates retry/hold state or renews the current proven lease.",
    ),
    BrowserCaptureRouteContract(
        "PUT",
        "/v1/capture-jobs/{job_id}/checkpoint",
        "capture_job_checkpoint",
        "bearer_required",
        "CAPTURE canonical JSON bytes + X-Polylogue-Checkpoint scalar descriptor",
        "CaptureJob checkpoint receipt v1 | BrowserCaptureErrorPayload",
        "Stages exact canonical bytes and acknowledges immutable custody under lease proof and revision CAS.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-jobs/{job_id}/checkpoint-artifacts/{digest}",
        "capture_job_checkpoint_artifact",
        "bearer_required",
        "X-Polylogue-Checkpoint scalar scope and current lease descriptor",
        "Exact CAPTURE canonical JSON artifact bytes | BrowserCaptureErrorPayload",
        "Streams only the exact digest owned by this scoped job and its receipts while the current lease is proven.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/events",
        "capture_job_event_append",
        "bearer_if_web_origin",
        "CaptureJobEvent request v1",
        "CaptureJobEvent response v1 | BrowserCaptureErrorPayload",
        "Appends one scoped, idempotent event under lease proof and revision CAS.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-jobs/{job_id}/events",
        "capture_job_event_read",
        "bearer_if_configured",
        "provider + JSON scope + client_protocol + optional limit + optional before_revision",
        "CaptureJobEvent page v1 | BrowserCaptureErrorPayload",
        "Reads the newest bounded page of the receiver-ordered event projection, oldest-ward by cursor.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/native/begin",
        "capture_job_native_begin",
        "bearer_required",
        "Native scoped acquisition descriptor v2",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Retains an immutable raw-member binding under the existing scoped lease.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/native/cancel",
        "capture_job_native_cancel",
        "bearer_required",
        "Native scoped acquisition descriptor v2",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Fences one native acquisition without removing committed raw, prefix, plan or asset custody.",
    ),
    BrowserCaptureRouteContract(
        "PUT",
        "/v1/capture-jobs/{job_id}/native/member",
        "capture_job_native_member",
        "bearer_required",
        "Native scoped acquisition descriptor v2 + streamed bytes in X-Polylogue-Native",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Streams one declared raw member into retained registry custody.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/native/prepare",
        "capture_job_native_prepare",
        "bearer_required",
        "Native scoped acquisition descriptor v2",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Runs the canonical provider parser through scratch storage and retains the unfinished envelope prefix and exact asset plan.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/native/plan",
        "capture_job_native_plan",
        "bearer_required",
        "Native scoped acquisition descriptor v2",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Pages canonical attachment occurrences and their immutable acquisition receipts.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/native/asset",
        "capture_job_native_asset",
        "bearer_required",
        "Native scoped acquisition descriptor v2",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Records a terminal failed acquisition for exactly one declared attachment occurrence.",
    ),
    BrowserCaptureRouteContract(
        "PUT",
        "/v1/capture-jobs/{job_id}/native/asset",
        "capture_job_native_asset",
        "bearer_required",
        "Native scoped acquisition descriptor v2 + streamed bytes in X-Polylogue-Native",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Streams verified asset bytes and records their exact occurrence receipt.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/native/finalize",
        "capture_job_native_finalize",
        "bearer_required",
        "Native scoped acquisition descriptor v2",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Seals the complete envelope after every declared occurrence has a receipt.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-jobs/{job_id}/native/publish",
        "capture_job_native_publish",
        "bearer_required",
        "Native scoped acquisition descriptor v2",
        "Native acquisition response v2 | BrowserCaptureErrorPayload",
        "Admits the exact sealed artifact through the shared capture route and retains its durable acknowledgement.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-jobs/orphans/{source_digest}/payload",
        "capture_job_orphan_inspect",
        "bearer_required",
        "exact source_digest + client_protocol",
        "Original retained JSON bytes | BrowserCaptureErrorPayload",
        "Inspects exact retained custody without assigning account scope or consuming evidence.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/pairing/redeem",
        "pairing_redeem",
        "unauthenticated_pairing_exchange",
        "BrowserCapturePairingRedeemRequest",
        "BrowserCapturePairingRedeemPayload | BrowserCaptureErrorPayload",
        (
            "Exchanges a short-lived single-use pairing code (minted out-of-band by "
            "`polylogued browser-capture pairing start`) for the receiver's current bearer "
            "token, so a fresh install never requires the operator to view/copy/paste the "
            "token itself (polylogue-gnie). Wrong codes and reuse are rejected; 5 wrong "
            "guesses or expiry invalidate the pending code."
        ),
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/receiver/attest",
        "receiver_attestation",
        "unauthenticated_receiver_attestation",
        "BrowserCaptureReceiverAttestationRequest",
        "BrowserCaptureReceiverAttestationPayload | BrowserCaptureErrorPayload",
        (
            "Answers a client's fresh 32-byte challenge with HMAC-SHA256 keyed by the receiver "
            "bearer over the receiver identity and the challenge, so the native pairing host "
            "releases the bearer only to the receiver that holds it, never to another process "
            "listening on the loopback port. Refused with receiver_auth_disabled when the "
            "receiver runs without a bearer."
        ),
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/capture-health",
        "capture_health_report",
        "bearer_if_configured",
        "BrowserCaptureHealthEventRequest",
        "BrowserCaptureHealthEventAcceptedPayload | BrowserCaptureErrorPayload",
        (
            "Extension-reported capture-health telemetry (gap/error/spool-backlog/"
            "provider-auth-broken), stored atomically in ops.db capture_health_history and announced under kind "
            "'browser_capture_health' so silent capture incompleteness becomes queryable "
            "history rather than only-visible-in-the-popup-at-that-moment (polylogue-3v1). Schema skew returns 409 schema_skew; storage faults use the same typed classification as GET."
        ),
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-health",
        "capture_health_list",
        "bearer_if_configured",
        "optional positive page_size and opaque cursor query parameters",
        "{ok: true, events: [...], next_cursor: string | null} | BrowserCaptureErrorPayload",
        "Newest-first snapshot pages of at most 100 reports; larger page_size requests continue through next_cursor without losing reports. Appends stay outside a continued snapshot; ops reset refuses history_cursor_reset. Schema skew returns 409 schema_skew; transient storage faults return 503 capture_history_unavailable; deterministic storage faults return 500 capture_history_storage_failed.",
    ),
)


def browser_capture_route_contract_for(method: str, path: str) -> BrowserCaptureRouteContract | None:
    """Return the receiver route contract for a method/path pair."""

    method_upper = method.upper()
    for contract in BROWSER_CAPTURE_ROUTE_CONTRACTS:
        if contract.method == method_upper and _route_pattern_matches(contract.pattern, path):
            return contract
    return None


def _route_pattern_matches(pattern: str, path: str) -> bool:
    pattern_parts = pattern.strip("/").split("/")
    path_parts = path.strip("/").split("/")
    if len(pattern_parts) != len(path_parts):
        return False
    for pattern_part, path_part in zip(pattern_parts, path_parts, strict=True):
        if pattern_part.startswith("{") and pattern_part.endswith("}"):
            if not path_part:
                return False
            continue
        if pattern_part != path_part:
            return False
    return True


__all__ = [
    "BROWSER_CAPTURE_ROUTE_CONTRACTS",
    "BrowserCaptureRouteContract",
    "browser_capture_route_contract_for",
]
