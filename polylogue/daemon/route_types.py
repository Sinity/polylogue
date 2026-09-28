"""Shared HTTP route projection types, independent of registry construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.declarations import DeclarationSpec

RouteMethod = Literal["GET", "POST", "DELETE"]
RouteKind = Literal[
    "browser_shell",
    "operational",
    "read_query",
    "read_detail",
    "user_overlay",
    "workspace",
    "maintenance",
    "capture",
    "observability",
]
RouteStability = Literal["stable", "shell_supported", "operational", "private"]
AuthPolicy = Literal[
    "unauthenticated_loopback",
    "credential_if_configured",
    "credential_and_same_origin",
    "bearer_if_configured_and_same_origin",
    "first_party_same_origin",
    "observability_flag_then_loopback_or_bearer",
]
NonReplayableReason = Literal[
    # The request mutates durable or credential state; replay would write.
    "mutation",
    # The only faithful example addresses an item a prior mutation minted.
    "requires-prior-mutation",
    # A read whose request is a POST body rather than a query string.
    "request-body",
    # The status code is the verdict (503 while any alert is present), so a
    # replay measures archive health rather than the request shape.
    "status-verdict",
]


@dataclass(frozen=True, slots=True)
class NonReplayable:
    """Why a route's declared examples cannot be replayed as a JSON GET."""

    reason: NonReplayableReason
    detail: str


@dataclass(frozen=True)
class RouteContract:
    """Machine-readable contract for one daemon HTTP route pattern."""

    method: RouteMethod
    pattern: str
    kind: RouteKind
    stability: RouteStability
    auth_policy: AuthPolicy
    response_contract: str
    notes: str = ""
    domain_operation: str | None = None
    handler_bound: bool = False

    @property
    def metadata_only_reason(self) -> str | None:
        """Name legacy route metadata that has no executable binding."""

        if self.domain_operation is not None or self.handler_bound:
            return None
        if self.notes:
            return self.notes
        return f"legacy {self.kind} adapter retained until its declaration migration"


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """HTTP projection of one shared declaration-kernel record.

    The kernel owns identity, handler ownership, and output/schema edges. The
    HTTP projection adds the transport vocabulary that the route adapter and
    OpenAPI renderer need. Keeping this projection beside the kernel record
    makes a route declaration executable without teaching the shared kernel
    about HTTP.

    Every declared example is a real request shape. An example argument named
    after a path placeholder (``name`` for ``:name``) fills that segment; the
    remaining arguments are the query string. ``non_replayable`` names why a
    route's examples cannot be replayed as a simple JSON GET; a non-GET route
    must carry one, and a GET route without one is replayed by
    ``TestDeclaredRouteExamples`` and must answer 2xx JSON.
    """

    kernel: DeclarationSpec
    method: RouteMethod
    path: str
    request_contract: str
    response_contract: str
    auth_policy: AuthPolicy
    domain_operation: str | None
    passes_params: bool = True
    passes_path: bool = False
    auth_scope: Literal["read", "events", "user_state"] = "read"
    write_gate: bool = False
    migration_reason: str = ""
    kind: RouteKind | None = None
    stability: RouteStability | None = None
    non_replayable: NonReplayable | None = None

    def __post_init__(self) -> None:
        if self.method != "GET" and self.non_replayable is None:
            raise ValueError(f"{self.method} {self.path} must declare why its examples are not replayable")
