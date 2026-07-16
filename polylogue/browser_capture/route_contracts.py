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
    "capture_job_adopt",
    "capture_job_checkpoint",
    "backfill_checkpoint_store",
    "backfill_checkpoint_read",
]


@dataclass(frozen=True, slots=True)
class BrowserCaptureRouteContract:
    """Machine-readable contract for one browser-capture receiver route."""

    method: Literal["GET", "POST"]
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
