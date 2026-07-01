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
    "status",
    "archive_state",
    "capture_ingest",
    "post_command_enqueue",
    "post_command_poll",
    "post_command_ack",
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
