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
