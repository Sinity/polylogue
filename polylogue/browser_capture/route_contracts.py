"""Browser-capture receiver route contracts.

The browser-capture receiver is a small local ingest boundary, separate from
the daemon web/API routes. Keeping its route metadata beside the receiver makes
the auth and DTO posture explicit without teaching callers to inspect handler
branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BrowserCaptureAuthPolicy = Literal["extension_origin", "bearer_if_web_origin", "bearer_if_configured"]
BrowserCaptureRouteKind = Literal[
    "capabilities",
    "status",
    "archive_state",
    "capture_ingest",
    "browser_action_capabilities",
    "browser_action_enqueue",
    "browser_action_list_claim",
    "browser_action_read",
    "browser_action_attachment",
    "browser_action_update",
    "browser_action_reconcile",
    "capture_job_create",
    "capture_job_discover",
    "capture_job_read",
    "capture_job_orphan_list",
    "capture_job_adopt",
    "capture_job_update",
    "capture_job_checkpoint",
    "backfill_checkpoint_store",
    "backfill_checkpoint_read",
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
        "POST",
        "/v1/browser-captures",
        "capture_ingest",
        "bearer_if_web_origin",
        "BrowserCaptureEnvelope",
        "BrowserCaptureAcceptedPayload | BrowserCaptureErrorPayload",
        "Accepts captures and returns a receiver-local artifact ref; extra web origins require bearer auth.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/browser-actions/capabilities",
        "browser_action_capabilities",
        "bearer_if_configured",
        None,
        "BrowserActionCapabilitiesPayload",
        "Declares exact provider operations, presentation choices, project routing, and attachment limits.",
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
        "CaptureJob scope query v1",
        "CaptureJob discovery response v1 | BrowserCaptureErrorPayload",
        "Lists only jobs in the keyed provider/account scope.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-jobs/orphans",
        "capture_job_orphan_list",
        "bearer_if_configured",
        "client_protocol query parameter",
        "CaptureJob typed orphan census v1 | BrowserCaptureErrorPayload",
        "Lists global legacy checkpoint fingerprints only on an explicit operator route.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/capture-jobs/{job_id}",
        "capture_job_read",
        "bearer_if_configured",
        "provider + account_scope + client_protocol query parameters",
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
        "bearer_if_web_origin",
        "CaptureJob checkpoint request v1",
        "CaptureJob checkpoint receipt v1 | BrowserCaptureErrorPayload",
        "Acknowledges a monotonic checkpoint under lease proof and revision CAS.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/backfill-checkpoint",
        "backfill_checkpoint_store",
        "bearer_if_web_origin",
        "BrowserBackfillCheckpointRequest",
        "BrowserBackfillCheckpointAcceptedPayload | BrowserCaptureErrorPayload",
        (
            "Mirrors the extension's sanitized backfill-ledger checkpoint (polylogue-06zm); "
            "IndexedDB remains the fast primary source, this is the profile-loss fallback. "
            "Overwrites any prior checkpoint for the same extension_instance_id."
        ),
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/backfill-checkpoint",
        "backfill_checkpoint_read",
        "bearer_if_configured",
        "extension_instance_id query parameter",
        "BrowserBackfillCheckpointPayload | BrowserCaptureErrorPayload",
        "Returns the mirrored checkpoint for an extension instance, or 404 if none was stored.",
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
