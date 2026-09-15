"""Shared-kernel projection of the executable query declaration family.

``capability_catalog`` used to build the executable declaration rows served by
``explain(subject="capability")`` by walking ``QUERY_FIELD_DESCRIPTORS`` and
``query_unit_descriptors()`` and hand-assembling a ``declaration_id`` for each.
That projection was the only place the query family had anything resembling a
declaration, and nothing resolved it: a unit whose ``sql_query_method`` named a
method ``ArchiveStore`` no longer defines produced a runtime ``AttributeError``
deep inside terminal execution, and a field whose ``spec_attr`` named a missing
``SessionQuerySpec`` attribute failed just as late.

This module is now the single declaration site for that family.  Each field and
each terminal unit is projected into a :class:`~polylogue.declarations.
DeclarationSpec` together with the exact catalog payload the detail route
serves, so:

* ``devtools gate declaration-bindings`` resolves every declared executor
  symbol and owner path against the live checkout;
* :func:`query_binding_diagnostics` additionally resolves each declaration's
  *domain* binding -- the spec/plan attribute and the ``ArchiveStore`` executor
  method -- which the shared kernel deliberately knows nothing about;
* ``capability_catalog`` derives its rows instead of re-deriving identity.

Closure note: ``fields.py``, ``metadata.py``, ``expression.py`` and
``transaction.py`` are inside the derived-schema identity closure.  This module
imports them but is not imported *by* them, and nothing in the closure imports
it, so adopting the kernel here does not move the archive's schema identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polylogue.archive.query.fields import QUERY_FIELD_DESCRIPTORS, QueryFieldDescriptor
from polylogue.archive.query.metadata import QueryUnitDescriptor, query_unit_descriptors
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    Diagnostic,
    ExampleSpec,
    HandlerBinding,
    JSONValue,
    OutputSpec,
    validate_registry,
)

REPAIR_COMMAND: Final = "devtools test tests/unit/archive/query tests/unit/declarations"

#: The bounded detail route that serves these declarations.
CAPABILITY_CONSUMER: Final = "polylogue.archive.query.capability_catalog.capability_detail_page"

_SPEC_OWNER_PATH: Final = "polylogue/archive/query/spec.py"
_PLAN_OWNER_PATH: Final = "polylogue/archive/query/plan.py"
_FIELDS_OWNER_PATH: Final = "polylogue/archive/query/fields.py"
_METADATA_OWNER_PATH: Final = "polylogue/archive/query/metadata.py"
_ARCHIVE_OWNER_PATH: Final = "polylogue/storage/sqlite/archive_tiers/archive.py"

_FIELD_PRODUCER: Final = "polylogue.archive.query.fields.QUERY_FIELD_DESCRIPTORS"
_UNIT_PRODUCER: Final = "polylogue.archive.query.metadata.QUERY_UNIT_DESCRIPTORS"


@dataclass(frozen=True, slots=True)
class QueryCapabilityDeclaration:
    """One executable query declaration: shared kernel plus its served payload.

    ``payload`` is the domain projection the capability detail route serves.
    It stays beside the kernel rather than inside it so the shared kernel never
    learns the query family's vocabulary, and so the served row and the
    completeness record can never name different declarations.
    """

    kernel: DeclarationSpec
    payload: dict[str, JSONValue]
    #: ``"field"`` or ``"unit"``; the payload's own ``kind``.
    kind: str
    #: The live descriptor this declaration was projected from.
    descriptor: QueryFieldDescriptor | QueryUnitDescriptor

    @property
    def name(self) -> str:
        return str(self.payload["name"])


def _field_payload(field: QueryFieldDescriptor) -> dict[str, JSONValue]:
    return {
        "declaration_id": f"query.field.{field.name}",
        "kind": "field",
        "name": field.name,
        "meaning": field.name.replace("_", " "),
        "authority": field.authority,
        "applicability": field.applicability,
        "projection": list(field.projections),
        "binding": {
            "spec": field.spec_attr,
            "plan": field.plan_attr,
            "storage": field.storage_names,
            "mcp": field.mcp_names,
            "api": field.api_names,
        },
        "operators": list(field.operators),
        "value_type": field.value_type,
        "cardinality": field.cardinality,
        "cost": {
            "shape": field.cost_shape,
            "pushdown": field.pushdown,
            "stats_join": field.requires_stats_join,
            "post_filter": field.requires_post_filter,
            "content_loading": field.requires_content_loading,
        },
        "stable_order": field.stable_order,
        "examples": list(field.examples),
    }


def _field_kernel(field: QueryFieldDescriptor) -> DeclarationSpec:
    handlers = [
        HandlerBinding(
            surface="query-field-registry",
            owner_path=_FIELDS_OWNER_PATH,
            symbol="QUERY_FIELD_DESCRIPTORS",
            binding_key=f"field:{field.name}",
        )
    ]
    if field.spec_attr is not None:
        handlers.append(
            HandlerBinding(
                surface="query-spec",
                owner_path=_SPEC_OWNER_PATH,
                symbol="SessionQuerySpec",
                binding_key=f"spec_attr:{field.spec_attr}",
            )
        )
    if field.plan_attr is not None:
        handlers.append(
            HandlerBinding(
                surface="query-plan",
                owner_path=_PLAN_OWNER_PATH,
                symbol="SessionQueryPlan",
                binding_key=f"plan_attr:{field.plan_attr}",
            )
        )
    # Every example is a spelling the live contract already declares, never
    # invented prose: the field's own declared examples first, then its public
    # MCP parameter name, which ``tests/unit/archive/query/
    # test_query_declarations.py`` resolves against the live
    # ``mcp_query_field_names()`` boundary vocabulary.
    examples = tuple(
        ExampleSpec(
            name=f"example-{index}",
            summary=f"Filter sessions on {field.name}.",
            arguments=(("expression", example),),
        )
        for index, example in enumerate(field.examples)
    )
    if field.mcp_names:
        examples += (
            ExampleSpec(
                name="mcp-parameter",
                summary=f"Public MCP query parameter for {field.name}.",
                arguments=(("mcp_parameter", field.mcp_names[0]),),
            ),
        )
    if not examples:
        examples = (
            ExampleSpec(
                name="field-token",
                summary=f"Declared query field token for {field.name}.",
                arguments=(("field", field.name),),
            ),
        )
    return DeclarationSpec(
        declaration_id=f"query.field.{field.name}",
        family_id="query.field",
        public_name=f"query.field:{field.name}",
        owner_path=_FIELDS_OWNER_PATH,
        compatibility=CompatibilityKey(
            identity="query-field",
            lifecycle="stable",
            authority="query-declaration",
            access_result_shape="capability-row",
            durability="read-only",
        ),
        producer=_FIELD_PRODUCER,
        role_gate="archive.read",
        schema_ref=f"polylogue.archive.query.fields.QueryFieldDescriptor:{field.name}",
        discovery_text=f"{field.name.replace('_', ' ')} ({field.authority}, {field.applicability})",
        repair_command=REPAIR_COMMAND,
        handlers=tuple(handlers),
        outputs=(
            OutputSpec(
                name="capability-row",
                kind="json",
                schema_ref="polylogue.archive.query.capability_catalog.capability_detail_page",
                target_path=f"explain://capability/query.field.{field.name}",
            ),
        ),
        examples=examples,
        completeness_edges=(
            CompletenessEdge(
                producer=_FIELD_PRODUCER,
                consumer=CAPABILITY_CONSUMER,
                kind="capability-detail-row",
                owner_path="polylogue/archive/query/capability_catalog.py",
            ),
        ),
    )


def _unit_payload(descriptor: QueryUnitDescriptor, *, stable_order: int) -> dict[str, JSONValue]:
    return {
        "declaration_id": f"query.unit.{descriptor.unit}",
        "kind": "unit",
        "name": descriptor.unit,
        "meaning": descriptor.description,
        "authority": "derived",
        "applicability": "unit",
        "projection": ["dsl", "mcp", "api"],
        "binding": {
            "source": descriptor.plural_source,
            "singular_source": descriptor.singular_source,
            "lowerer": descriptor.lowerer_kind,
            "sql": descriptor.sql_query_method,
            "runtime": descriptor.runtime_query_method,
        },
        "operators": ["where", "exists"] if descriptor.exists_supported else ["where"],
        "value_type": "record",
        "cardinality": "many",
        "cost": {
            "shape": "indexed" if descriptor.lowerer_kind == "sql" else "post_filter",
            "pushdown": descriptor.lowerer_kind == "sql",
        },
        "stable_order": stable_order,
        "examples": [descriptor.terminal_example or descriptor.example],
    }


def _unit_kernel(descriptor: QueryUnitDescriptor) -> DeclarationSpec:
    handlers = [
        HandlerBinding(
            surface="query-unit-registry",
            owner_path=_METADATA_OWNER_PATH,
            symbol="QUERY_UNIT_DESCRIPTORS",
            binding_key=f"unit:{descriptor.unit}",
        )
    ]
    if descriptor.sql_query_method is not None:
        # The executor is reached by ``getattr(archive, method_name)`` in
        # ``unit_results``/``attached_units``. Declaring it makes a rename a
        # gate failure instead of a runtime AttributeError under a real query.
        handlers.append(
            HandlerBinding(
                surface="archive-sql-executor",
                owner_path=_ARCHIVE_OWNER_PATH,
                symbol=descriptor.sql_query_method,
                binding_key=f"ArchiveStore.{descriptor.sql_query_method}",
            )
        )
    example = descriptor.terminal_example or descriptor.example
    return DeclarationSpec(
        declaration_id=f"query.unit.{descriptor.unit}",
        family_id="query.unit",
        public_name=f"query.unit:{descriptor.unit}",
        owner_path=_METADATA_OWNER_PATH,
        compatibility=CompatibilityKey(
            identity="query-unit",
            lifecycle="stable",
            authority="query-declaration",
            access_result_shape="capability-row",
            durability="read-only",
        ),
        producer=_UNIT_PRODUCER,
        role_gate="archive.read",
        schema_ref=descriptor.payload_model,
        discovery_text=descriptor.description,
        repair_command=REPAIR_COMMAND,
        handlers=tuple(handlers),
        outputs=(
            OutputSpec(
                name="capability-row",
                kind="json",
                schema_ref="polylogue.archive.query.capability_catalog.capability_detail_page",
                target_path=f"explain://capability/query.unit.{descriptor.unit}",
            ),
        ),
        # The unit's own live terminal example, which the production parser
        # accepts; the declaration never invents a second spelling.
        examples=(
            ExampleSpec(
                name="terminal",
                summary=f"Terminal {descriptor.plural_source} query.",
                arguments=(("expression", example),),
            ),
        ),
        completeness_edges=(
            CompletenessEdge(
                producer=_UNIT_PRODUCER,
                consumer=CAPABILITY_CONSUMER,
                kind="capability-detail-row",
                owner_path="polylogue/archive/query/capability_catalog.py",
            ),
        ),
    )


def query_capability_declarations() -> tuple[QueryCapabilityDeclaration, ...]:
    """Project every query field and terminal unit, in stable catalog order."""

    declarations: list[QueryCapabilityDeclaration] = [
        QueryCapabilityDeclaration(
            kernel=_field_kernel(field),
            payload=_field_payload(field),
            kind="field",
            descriptor=field,
        )
        for field in sorted(QUERY_FIELD_DESCRIPTORS, key=lambda item: (item.stable_order, item.name))
    ]
    for descriptor in query_unit_descriptors(terminal_supported=True):
        declarations.append(
            QueryCapabilityDeclaration(
                kernel=_unit_kernel(descriptor),
                payload=_unit_payload(descriptor, stable_order=10000 + len(declarations)),
                kind="unit",
                descriptor=descriptor,
            )
        )
    return tuple(declarations)


def build_query_kernel_registry() -> DeclarationRegistry:
    """Register every projected query kernel into a fresh kernel registry."""

    registry = DeclarationRegistry()
    for declaration in query_capability_declarations():
        registry.register(declaration.kernel)
    return registry


QUERY_CAPABILITY_DECLARATIONS: Final[tuple[QueryCapabilityDeclaration, ...]] = query_capability_declarations()
QUERY_KERNEL_REGISTRY: Final[DeclarationRegistry] = build_query_kernel_registry()

_DIAGNOSTICS = validate_registry(QUERY_KERNEL_REGISTRY)
if _DIAGNOSTICS:  # pragma: no cover - a structurally incomplete projection must not import
    raise RuntimeError("incomplete query declaration registry: " + "; ".join(item.message for item in _DIAGNOSTICS))


def _diagnostic(declaration_id: str, owner_path: str, code: str, message: str) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=f"{declaration_id}: {message}",
        declaration_id=declaration_id,
        owner_path=owner_path,
        repair_command=REPAIR_COMMAND,
    )


def query_binding_diagnostics() -> tuple[Diagnostic, ...]:
    """Resolve the query family's *domain* bindings against the live code.

    The shared kernel resolves files and symbols; only the query domain knows
    that ``spec_attr`` must name a ``SessionQuerySpec`` field and that
    ``sql_query_method`` must name an ``ArchiveStore`` method reached by
    ``getattr``. Domain-owned validation stays here, and its output is the
    kernel's own :class:`~polylogue.declarations.Diagnostic`, so one gate
    reports both with the same actionable shape.
    """

    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    def _resolves(owner: type, attribute: str) -> bool:
        """A declared attribute may be a dataclass field or a derived property."""

        return attribute in getattr(owner, "__dataclass_fields__", {}) or hasattr(owner, attribute)

    diagnostics: list[Diagnostic] = []
    for declaration in QUERY_CAPABILITY_DECLARATIONS:
        kernel = declaration.kernel
        descriptor = declaration.descriptor
        if isinstance(descriptor, QueryFieldDescriptor):
            if descriptor.spec_attr is not None and not _resolves(SessionQuerySpec, descriptor.spec_attr):
                diagnostics.append(
                    _diagnostic(
                        kernel.declaration_id,
                        kernel.owner_path,
                        "unresolved_spec_attr",
                        f"spec_attr {descriptor.spec_attr!r} is not a SessionQuerySpec field or property",
                    )
                )
            if descriptor.plan_attr is not None and not _resolves(SessionQueryPlan, descriptor.plan_attr):
                diagnostics.append(
                    _diagnostic(
                        kernel.declaration_id,
                        kernel.owner_path,
                        "unresolved_plan_attr",
                        f"plan_attr {descriptor.plan_attr!r} is not a SessionQueryPlan field or property",
                    )
                )
            continue
        method = descriptor.sql_query_method
        if method is not None and not callable(getattr(ArchiveStore, method, None)):
            diagnostics.append(
                _diagnostic(
                    kernel.declaration_id,
                    kernel.owner_path,
                    "unresolved_unit_executor",
                    f"sql_query_method {method!r} is not an ArchiveStore method",
                )
            )
    return tuple(sorted(diagnostics, key=lambda item: (item.declaration_id, item.code)))


def capability_declaration_rows() -> tuple[dict[str, JSONValue], ...]:
    """Return the executable declaration rows the capability detail route serves."""

    return tuple(declaration.payload for declaration in QUERY_CAPABILITY_DECLARATIONS)


__all__ = [
    "CAPABILITY_CONSUMER",
    "QUERY_CAPABILITY_DECLARATIONS",
    "QUERY_KERNEL_REGISTRY",
    "REPAIR_COMMAND",
    "QueryCapabilityDeclaration",
    "build_query_kernel_registry",
    "capability_declaration_rows",
    "query_binding_diagnostics",
    "query_capability_declarations",
]
