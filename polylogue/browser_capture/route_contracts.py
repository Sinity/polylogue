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
BrowserCaptureRouteKind = Literal["status", "archive_state", "capture_ingest"]


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
)


def browser_capture_route_contract_for(method: str, path: str) -> BrowserCaptureRouteContract | None:
    """Return the receiver route contract for an exact method/path pair."""

    method_upper = method.upper()
    for contract in BROWSER_CAPTURE_ROUTE_CONTRACTS:
        if contract.method == method_upper and contract.pattern == path:
            return contract
    return None


__all__ = [
    "BROWSER_CAPTURE_ROUTE_CONTRACTS",
    "BrowserCaptureRouteContract",
    "browser_capture_route_contract_for",
]
