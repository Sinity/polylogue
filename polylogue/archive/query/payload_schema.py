"""One declaration per query-AST node: its payload shape and its wire schema.

The query DSL's compiled AST is serialized by ``to_payload()`` on the typed
nodes in :mod:`polylogue.archive.query.predicate` and
:mod:`polylogue.archive.query.expression`, and published to external consumers
(``devtools render openapi``, the generated webui client, the MCP ``explain``
operation's documented result) as a JSON Schema.

Both artifacts come from the declarations here. A node states its payload keys
once -- their wire names, their JSON types, where each value is read from, and
when a key is omitted -- and :func:`render_payload` produces the dict while
:func:`payload_model` produces the Pydantic model. A hand-written schema that
mirrors the serializer therefore cannot exist: there is nothing to mirror.

The declarations describe the *wire* shape, not the dataclass. A node may read
a payload key from a differently named attribute, emit a constant the class
does not store, or omit a key whose value is absent; those are stated per
field rather than inferred, because the wire name is the contract and the
attribute name is an implementation detail.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

#: Sentinel for "this field declares no constant / no schema default".
MISSING: Any = object()

#: When a key is left out of the rendered payload.
#:
#: ``never`` always emits the key; ``none`` omits it when the value is
#: ``None``; ``falsy`` omits it when the value is empty or ``False`` (the
#: shape used for optional lists, maps and flags).
Omission = Literal["never", "none", "falsy"]

#: How a declared key's value is rendered.
#:
#: ``scalar`` emits the attribute unchanged; ``node`` emits a nested node's
#: own payload; ``node_list`` a list of them; ``scalar_list`` materializes a
#: tuple as a list; ``pair_map`` materializes a tuple of pairs as a dict;
#: ``passthrough`` emits an already-rendered payload dict; ``group`` renders a
#: nested object from the *same* node.
Shape = Literal["scalar", "node", "node_list", "scalar_list", "pair_map", "passthrough", "group"]


@dataclass(frozen=True, slots=True)
class PayloadField:
    """One declared key of a node's payload."""

    key: str
    #: Thunk returning the published annotation for this key. A thunk (rather
    #: than the annotation itself) lets recursive node references name a model
    #: that does not exist yet; the models are rebuilt once all are declared.
    type: Callable[[], Any]
    shape: Shape = "scalar"
    #: Attribute this key reads from. Defaults to the key itself.
    source: str | None = None
    #: Constant emitted for this key; the node stores nothing for it.
    const: Any = MISSING
    omit: Omission = "never"
    #: Extra omission rule evaluated against the owning node, for a key whose
    #: presence is not decided by its own value alone.
    omit_when: Callable[[Any], bool] | None = None
    #: Value the published model uses when the key is absent. Required for any
    #: field that can be omitted, so the model never declares a key that is
    #: both optional and undefaulted.
    default: Any = MISSING
    #: Nested declaration for a ``group`` field, rendered from the same node.
    group: PayloadSchema | None = None
    description: str = ""

    def attribute(self) -> str:
        return self.source or self.key


@dataclass(frozen=True, slots=True)
class PayloadSchema:
    """The complete declared payload of one AST node."""

    name: str
    fields: tuple[PayloadField, ...]
    description: str = ""

    def keys(self) -> tuple[str, ...]:
        return tuple(field.key for field in self.fields)


@dataclass(frozen=True, slots=True)
class PayloadUnion:
    """A tagged union over node declarations sharing one discriminator key."""

    name: str
    members: tuple[PayloadSchema, ...]
    discriminator: str = "kind"


def _render_value(value: object, shape: Shape) -> object:
    if shape in ("scalar", "passthrough"):
        return value
    if shape == "node":
        return None if value is None else cast(Any, value).to_payload()
    if shape == "node_list":
        return [cast(Any, item).to_payload() for item in cast(Sequence[Any], value)]
    if shape == "scalar_list":
        return list(cast(Sequence[Any], value))
    if shape == "pair_map":
        return dict(cast(Any, value))
    raise ValueError(f"unsupported payload shape: {shape!r}")


def render_payload(node: object, schema: PayloadSchema) -> dict[str, object]:
    """Render ``node``'s payload from its declaration.

    Key order follows the declaration, so the emitted dict is stable and the
    declaration reads as the wire document it produces.
    """

    payload: dict[str, object] = {}
    for field in schema.fields:
        if field.shape == "group":
            assert field.group is not None
            group = render_payload(node, field.group)
            if group:
                payload[field.key] = group
            continue
        if field.const is not MISSING:
            payload[field.key] = field.const
            continue
        value = getattr(node, field.attribute())
        if field.omit_when is not None and field.omit_when(node):
            continue
        if field.omit == "none" and value is None:
            continue
        if field.omit == "falsy" and not value:
            continue
        payload[field.key] = _render_value(value, field.shape)
    return payload


_STRICT = ConfigDict(extra="forbid", frozen=True)


def payload_model(schema: PayloadSchema, *, suffix: str = "") -> type[BaseModel]:
    """Build the published Pydantic model for one declaration."""

    definitions: dict[str, Any] = {}
    for field in schema.fields:
        annotation = field.type()
        if field.const is not MISSING:
            definitions[field.key] = (annotation, field.const)
            continue
        if field.omit == "never" and field.default is MISSING:
            definitions[field.key] = (annotation, ...)
            continue
        default = field.default
        if default is MISSING:
            raise ValueError(f"{schema.name}.{field.key} may be omitted but declares no schema default")
        if isinstance(default, (list, dict)):
            definitions[field.key] = (annotation, Field(default_factory=lambda value=default: type(value)(value)))
            continue
        definitions[field.key] = (annotation, default)
    model = create_model(
        f"{schema.name}{suffix}",
        __config__=_STRICT,
        **definitions,
    )
    model.__doc__ = schema.description or schema.name
    return cast("type[BaseModel]", model)


def union_annotation(union: PayloadUnion, models: dict[str, type[BaseModel]]) -> Any:
    """Return the discriminated-union annotation for a declared union."""

    members = tuple(models[member.name] for member in union.members)
    if len(members) == 1:
        return members[0]
    annotation: Any = members[0]
    for model in members[1:]:
        annotation = annotation | model
    return Annotated[annotation, Field(discriminator=union.discriminator)]


@dataclass
class PayloadModelRegistry:
    """Models built from declarations, plus the namespace recursion needs."""

    models: dict[str, type[BaseModel]] = dataclass_field(default_factory=dict)

    def build(self, schema: PayloadSchema, *, suffix: str = "") -> type[BaseModel]:
        model = payload_model(schema, suffix=suffix)
        self.models[schema.name] = model
        return model

    def rebuild(self, namespace: dict[str, Any]) -> None:
        for model in self.models.values():
            model.model_rebuild(_types_namespace=namespace)


__all__ = [
    "MISSING",
    "Omission",
    "PayloadField",
    "PayloadModelRegistry",
    "PayloadSchema",
    "PayloadUnion",
    "Shape",
    "payload_model",
    "render_payload",
    "union_annotation",
]
