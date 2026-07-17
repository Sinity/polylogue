"""MCP-owned declaration semantics layered over the shared kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Final, Literal, TypeAlias

from polylogue.declarations import DeclarationSpec, JSONValue

MCPRole = Literal["read", "write", "review", "admin"]
MCP_ROLE_ORDER: Final[dict[MCPRole, int]] = {"read": 0, "write": 1, "review": 2, "admin": 3}


def mcp_role_allows(role: MCPRole, required: MCPRole) -> bool:
    """Return whether ``role`` includes the required MCP capability."""

    return MCP_ROLE_ORDER[role] >= MCP_ROLE_ORDER[required]


ObservedUse = Literal["observed", "not_observed", "unknown"]


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


class MCPDeprecationState(str, Enum):
    RETAINED = "retained"
    COMPATIBILITY = "compatibility"
    TARGET_RESOURCE = "target_resource"
    TARGET_PROMPT = "target_prompt"


@dataclass(frozen=True, slots=True)
class MCPHandlerBinding:
    """Where the live FastMCP handler is registered and implemented."""

    module: str
    symbol: str
    registrar: str


@dataclass(frozen=True, slots=True)
class MCPInputContract:
    """Source of the FastMCP input schema and its compatibility invariant."""

    schema_source: str
    schema_mode: str
    required_arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MCPOutputContract:
    """Semantic output classification independent from Python return ``str``."""

    kind: str
    envelope_fields: tuple[str, ...] = ()
    schema_source: str = "tests/unit/mcp/test_envelope_contracts.py::TOOL_CONTRACT"


@dataclass(frozen=True, slots=True)
class MCPContinuationContract:
    """Logical-completeness and continuation behavior of one route."""

    mode: str
    continuation_ref: str | None
    exhaustive_route: str | None
    notes: str


@dataclass(frozen=True, slots=True)
class PythonParityExpectation:
    """Public Python binding or an explicitly governed intentional absence."""

    binding: str | None = None
    intentional_absence_authority: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        bound = self.binding is not None
        absent = self.intentional_absence_authority is not None and self.reason is not None
        if bound == absent:
            raise ValueError("Python parity must declare exactly one binding or intentional absence")


@dataclass(frozen=True, slots=True)
class MCPToolDeclaration:
    """Executable inventory row for one legacy MCP tool."""

    kernel: DeclarationSpec
    name: str
    description: str
    verb: MCPVerb
    object_kinds: tuple[str, ...]
    minimum_role: MCPRole
    capability: str
    result_semantics: MCPResultSemantics
    canonical_plan: str
    canonical_projection: str
    input_contract: MCPInputContract
    output_contract: MCPOutputContract
    minimal_arguments: tuple[tuple[str, JSONValue], ...]
    grammar_discovery: tuple[str, ...]
    field_discovery: tuple[str, ...]
    value_discovery: tuple[str, ...]
    continuation: MCPContinuationContract
    resource_alternatives: tuple[str, ...]
    prompt_alternatives: tuple[str, ...]
    compatibility_route: str
    workflow_coverage: tuple[str, ...]
    incident_coverage: tuple[str, ...]
    observed_use: ObservedUse
    telemetry_key: str
    deprecation_state: MCPDeprecationState
    retirement_owner: str
    registration: MCPHandlerBinding
    operation_owner: str
    python_parity: PythonParityExpectation

    def __post_init__(self) -> None:
        if self.name != self.kernel.public_name:
            raise ValueError(f"MCP declaration name {self.name!r} != kernel public name {self.kernel.public_name!r}")
        if self.telemetry_key != self.name:
            raise ValueError(f"MCP telemetry key must remain the discovery name for {self.name!r}")
        if not self.object_kinds:
            raise ValueError(f"MCP declaration {self.name!r} has no object/ref kind")
        if not self.workflow_coverage:
            raise ValueError(f"MCP declaration {self.name!r} has no workflow coverage")

    @property
    def declaration_id(self) -> str:
        return self.kernel.declaration_id

    def minimal_arguments_dict(self) -> dict[str, JSONValue]:
        return dict(self.minimal_arguments)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible projection for generated artifacts."""

        payload = asdict(self)
        payload["verb"] = self.verb.value
        payload["result_semantics"] = self.result_semantics.value
        payload["deprecation_state"] = self.deprecation_state.value
        payload["minimal_arguments"] = dict(self.minimal_arguments)
        return payload


@dataclass(frozen=True, slots=True)
class MCPTransactionDeclaration:
    """One target protocol-native transaction in the bounded discovery algebra."""

    name: str
    verb: MCPVerb
    minimum_role: MCPRole
    object_kinds: tuple[str, ...]
    result_semantics: tuple[MCPResultSemantics, ...]
    purpose: str
    migration_owner: str


@dataclass(frozen=True, slots=True)
class MCPResourceDeclaration:
    """Target URI resource class for stable archive identities."""

    uri_template: str
    object_kinds: tuple[str, ...]
    minimum_role: MCPRole
    authority: str
    migration_owner: str


@dataclass(frozen=True, slots=True)
class MCPPromptDeclaration:
    """Target prompt class for parameterized workflows without authority."""

    name: str
    workflow: str
    minimum_role: MCPRole
    mutation_authority: Literal["none"]
    migration_owner: str


MCPDeclarationMap: TypeAlias = dict[str, MCPToolDeclaration]

__all__ = [
    "MCPContinuationContract",
    "MCPDeclarationMap",
    "MCPDeprecationState",
    "MCPHandlerBinding",
    "MCPInputContract",
    "MCPOutputContract",
    "MCPPromptDeclaration",
    "MCPResourceDeclaration",
    "MCPResultSemantics",
    "MCPRole",
    "MCP_ROLE_ORDER",
    "MCPToolDeclaration",
    "MCPTransactionDeclaration",
    "MCPVerb",
    "ObservedUse",
    "PythonParityExpectation",
    "mcp_role_allows",
]
