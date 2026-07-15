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
    "post_command_enqueue",
    "post_command_poll",
    "post_command_ack",
    "backfill_checkpoint_store",
    "backfill_checkpoint_read",
    "launch_job_enqueue",
    "launch_job_list_claim",
    "launch_job_attachment",
    "launch_job_update",
    "launch_job_control",
    "launch_job_handoff",
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
        "POST",
        "/v1/post-commands",
        "post_command_enqueue",
        "bearer_if_web_origin",
        "BrowserPostCommandRequest",
        "BrowserPostEnqueuedPayload | BrowserCaptureErrorPayload",
        (
            "Enqueues an outbound post command. Refused with 403 unless "
            "POLYLOGUE_BROWSER_POST_ENABLED=1 (default OFF safety guard)."
        ),
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/post-commands",
        "post_command_poll",
        "bearer_if_configured",
        "optional provider query parameter",
        "BrowserPostCommandListPayload",
        (
            "Extension polls for pending post commands and claims them "
            "(pending -> dispatched). Returns an empty command list when posting is disabled."
        ),
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/post-commands/{command_id}/ack",
        "post_command_ack",
        "bearer_if_web_origin",
        "BrowserPostCommandAckRequest",
        "BrowserPostAckPayload | BrowserCaptureErrorPayload",
        "Extension reports the post result (submitted|failed); updates the queued command.",
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
    BrowserCaptureRouteContract(
        "POST",
        "/v1/launch-jobs",
        "launch_job_enqueue",
        "bearer_if_web_origin",
        "BrowserLaunchJobRequest",
        "BrowserLaunchJobEnqueuedPayload | BrowserCaptureErrorPayload",
        "Copies explicit inputs into receiver-owned immutable job storage; enqueue is operator authorization.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/launch-jobs",
        "launch_job_list_claim",
        "bearer_if_configured",
        "optional claim_by extension-instance query parameter",
        "BrowserLaunchJobListPayload | BrowserCaptureErrorPayload",
        "Lists jobs or atomically leases one due job; receiver enforces concurrency one and cadence.",
    ),
    BrowserCaptureRouteContract(
        "GET",
        "/v1/launch-jobs/{job_id}/attachments/{attachment_id}",
        "launch_job_attachment",
        "bearer_if_configured",
        None,
        "immutable attachment bytes | BrowserCaptureErrorPayload",
        "Returns receiver-copied bytes after size and SHA-256 integrity verification.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/launch-jobs/{job_id}/events",
        "launch_job_update",
        "bearer_if_web_origin",
        "BrowserLaunchJobUpdateRequest",
        "BrowserLaunchJobEnqueuedPayload | BrowserCaptureErrorPayload",
        "Renews the lease and projects progress, submit, cooldown, pause, completion, or failure.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/launch-jobs/{job_id}/control",
        "launch_job_control",
        "bearer_if_web_origin",
        "BrowserLaunchJobControlRequest",
        "BrowserLaunchJobEnqueuedPayload | BrowserCaptureErrorPayload",
        "Operator pause/resume/cancel/retry and ambiguous-submit reconciliation; terminal completion cannot be reopened.",
    ),
    BrowserCaptureRouteContract(
        "POST",
        "/v1/launch-jobs/{job_id}/handoff",
        "launch_job_handoff",
        "bearer_if_web_origin",
        "BrowserLaunchHandoffRequest",
        "BrowserLaunchHandoffAcceptedPayload | BrowserCaptureErrorPayload",
        "Stores and checksum-validates the cohesive ZIP before marking the job complete.",
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
