"""MCP-owned declaration semantics layered over the shared kernel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from polylogue.declarations import DeclarationSpec

#: The privileged capability flags a declaration can require. ``None`` on a
#: declaration means it is a base read-only transaction, always available.
MCPCapabilityFlag = Literal["write", "judge", "maintenance"]


@dataclass(frozen=True, slots=True)
class MCPCapabilities:
    """Explicit config opt-ins for the MCP server's privileged dispatchers.

    Per polylogue-800m: the MCP role ladder is not a product concept. The
    server is read-only by default (every field ``False``); write capability
    (``write``/``run`` dispatchers), judge capability (``judge``), and
    maintenance capability (``maintenance``) are independent boolean
    opt-ins resolved from config (see ``polylogue.config.PolylogueConfig``:
    ``mcp_write_enabled``/``mcp_judge_enabled``/``mcp_maintenance_enabled``).
    There is no ordering between them -- enabling one does not imply another.
    """

    write: bool = False
    judge: bool = False
    maintenance: bool = False

    def allows(self, required: MCPCapabilityFlag | None) -> bool:
        """Return whether these capabilities satisfy ``required``.

        ``required=None`` marks a base read-only transaction, always allowed.
        """
        if required is None:
            return True
        return bool(getattr(self, required))


class MCPVerb(str, Enum):
    QUERY = "query"
    READ = "read"
    GET = "get"
    EXPLAIN = "explain"
    CONTEXT = "context"
    STATUS = "status"
    WRITE = "write"
    JUDGE = "judge"
    RUN = "run"
    MAINTENANCE = "maintenance"


class MCPResultSemantics(str, Enum):
    EXHAUSTIVE_PAGE = "exhaustive_page"
    TOP_K = "top_k"
    SAMPLE = "sample"
    AGGREGATE = "aggregate"
    SINGLE_OBJECT = "single_object"
    BOUNDED_CONTEXT = "bounded_context"
    RECURSIVE_GRAPH = "recursive_graph"
    MUTATION = "mutation"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True, slots=True)
class MCPHandlerBinding:
    """Where the live FastMCP handler is registered and implemented."""

    module: str
    symbol: str
    registrar: str


@dataclass(frozen=True, slots=True)
class MCPToolDeclaration:
    """Runtime registration contract for one MCP tool."""

    kernel: DeclarationSpec
    name: str
    description: str
    required_capability: MCPCapabilityFlag | None
    registration: MCPHandlerBinding

    def __post_init__(self) -> None:
        if self.name != self.kernel.public_name:
            raise ValueError(f"MCP declaration name {self.name!r} != kernel public name {self.kernel.public_name!r}")

    @property
    def declaration_id(self) -> str:
        return self.kernel.declaration_id


@dataclass(frozen=True, slots=True)
class MCPTransactionDeclaration:
    """One target protocol-native transaction in the bounded discovery algebra."""

    name: str
    verb: MCPVerb
    required_capability: MCPCapabilityFlag | None
    object_kinds: tuple[str, ...]
    result_semantics: tuple[MCPResultSemantics, ...]
    purpose: str


@dataclass(frozen=True, slots=True)
class MCPResourceDeclaration:
    """Target URI resource class for stable archive identities."""

    uri_template: str
    object_kinds: tuple[str, ...]
    required_capability: MCPCapabilityFlag | None
    authority: str


@dataclass(frozen=True, slots=True)
class MCPPromptDeclaration:
    """Target prompt class for parameterized workflows without authority."""

    name: str
    workflow: str
    required_capability: MCPCapabilityFlag | None
    mutation_authority: Literal["none"]


MCPDeclarationMap: TypeAlias = dict[str, MCPToolDeclaration]

__all__ = [
    "MCPCapabilities",
    "MCPCapabilityFlag",
    "MCPDeclarationMap",
    "MCPHandlerBinding",
    "MCPPromptDeclaration",
    "MCPResourceDeclaration",
    "MCPResultSemantics",
    "MCPToolDeclaration",
    "MCPTransactionDeclaration",
    "MCPVerb",
]
