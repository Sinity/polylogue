"""Wire format configuration for provider export formats.

Describes HOW each provider structures their export data — encoding type,
tree vs. linear vs. JSONL, and message location paths.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.archive.raw_payload.decode import JSONValue

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.models import SchemaRecord

WireEncoding: TypeAlias = Literal["json", "jsonl"]
WireCapabilityStatus: TypeAlias = Literal["supported", "unsupported"]


class UnsupportedSyntheticWireRouteError(ValueError):
    """Raised when synthetic selection reaches an explicit unsupported route."""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"Synthetic wire route unsupported for {provider}: {reason}")


@dataclass(frozen=True)
class TreeConfig:
    """Configuration for tree-structured message formats."""

    container_path: str | None = None  # Top-level key containing the tree dict
    key_field: str = "id"
    parent_field: str = "parent"
    children_field: str | None = None
    session_field: str | None = None


@dataclass(frozen=True)
class WireFormat:
    """Wire format configuration for a provider's export format."""

    encoding: WireEncoding
    tree: TreeConfig | None = None
    messages_path: str | None = None  # Dot-path to messages array


@dataclass(frozen=True)
class WireRoute:
    """Explicit synthetic capability for one catalog provider.

    ``PROVIDER_WIRE_FORMATS`` remains the executable adapter map used by the
    generator.  This wider route map is the authority for catalog coverage,
    including providers whose parser shape is deliberately not synthesized by
    this lane.
    """

    status: WireCapabilityStatus
    wire_format: WireFormat | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status == "supported" and self.wire_format is None:
            raise ValueError("supported wire routes require a WireFormat")
        if self.status == "unsupported" and (self.wire_format is not None or not self.reason):
            raise ValueError("unsupported wire routes require a reason and no WireFormat")


@dataclass(frozen=True)
class ConstructCoverage:
    """Deterministic schema-construct coverage for generated payloads."""

    schema_keywords: tuple[str, ...]
    exercised_keywords: tuple[str, ...]
    missing_keywords: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_keywords


@dataclass(frozen=True)
class WireSupportEntry:
    """One provider row in the support receipt."""

    provider: str
    status: WireCapabilityStatus
    reason: str | None
    package_version: str | None
    element_kind: str | None
    schema_valid: bool | None
    parsed_session_count: int
    parsed_message_count: int
    construct_coverage: ConstructCoverage | None
    validation_error: str | None = None

    @property
    def healthy(self) -> bool:
        return self.status == "unsupported" or (
            self.schema_valid is True
            and self.parsed_session_count > 0
            and self.parsed_message_count > 0
            and self.construct_coverage is not None
            and self.construct_coverage.complete
            and self.validation_error is None
        )


@dataclass(frozen=True)
class WireSupportReceipt:
    """Registry-derived, deterministic support and coverage receipt."""

    catalog_providers: tuple[str, ...]
    entries: tuple[WireSupportEntry, ...]
    missing_routes: tuple[str, ...]

    @property
    def supported_count(self) -> int:
        return sum(entry.status == "supported" for entry in self.entries)

    @property
    def unsupported_count(self) -> int:
        return sum(entry.status == "unsupported" for entry in self.entries)

    @property
    def validated_supported_count(self) -> int:
        return sum(entry.status == "supported" and entry.healthy for entry in self.entries)

    @property
    def complete(self) -> bool:
        return not self.missing_routes and all(entry.healthy for entry in self.entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_providers": list(self.catalog_providers),
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "validated_supported_count": self.validated_supported_count,
            "missing_routes": list(self.missing_routes),
            "complete": self.complete,
            "entries": [
                {
                    "provider": entry.provider,
                    "status": entry.status,
                    "reason": entry.reason,
                    "package_version": entry.package_version,
                    "element_kind": entry.element_kind,
                    "schema_valid": entry.schema_valid,
                    "parsed_session_count": entry.parsed_session_count,
                    "parsed_message_count": entry.parsed_message_count,
                    "validation_error": entry.validation_error,
                    "construct_coverage": (
                        {
                            "schema_keywords": list(entry.construct_coverage.schema_keywords),
                            "exercised_keywords": list(entry.construct_coverage.exercised_keywords),
                            "missing_keywords": list(entry.construct_coverage.missing_keywords),
                            "complete": entry.construct_coverage.complete,
                        }
                        if entry.construct_coverage is not None
                        else None
                    ),
                }
                for entry in self.entries
            ],
        }


# Per-provider wire format configs — the only manual piece (~50 lines).
# Describes HOW the format is structured, not WHAT sessions say.
PROVIDER_WIRE_FORMATS: dict[str, WireFormat] = {
    "chatgpt": WireFormat(
        encoding="json",
        tree=TreeConfig(
            container_path="mapping",
            key_field="id",
            parent_field="parent",
            children_field="children",
        ),
    ),
    "claude-code": WireFormat(
        encoding="jsonl",
        tree=TreeConfig(
            key_field="uuid",
            parent_field="parentUuid",
            session_field="sessionId",
        ),
    ),
    "claude-ai": WireFormat(
        encoding="json",
        messages_path="chat_messages",
    ),
    "codex": WireFormat(
        encoding="jsonl",
    ),
    "gemini": WireFormat(
        encoding="json",
        messages_path="chunkedPrompt.chunks",
    ),
}


# All catalog providers must have a route here.  The unsupported routes
# are explicit capabilities, not implicit generator fallbacks.
PROVIDER_WIRE_ROUTES: dict[str, WireRoute] = {
    **{
        provider: WireRoute(status="supported", wire_format=wire_format)
        for provider, wire_format in PROVIDER_WIRE_FORMATS.items()
    },
    "antigravity": WireRoute(
        status="unsupported",
        reason="antigravity requires the language-server .pb adapter and source-path semantics, which generic JSON generation does not exercise",
    ),
    "browser-capture": WireRoute(
        status="unsupported",
        reason="browser-capture is an envelope with provider-specific typed turns, not a generic export wire format",
    ),
    "gemini-cli": WireRoute(
        status="unsupported",
        reason="gemini-cli requires its checkpoint document parser semantics, which this generic adapter does not model",
    ),
    "hermes": WireRoute(
        status="unsupported",
        reason="hermes requires local-agent session-document semantics, which this generic adapter does not model",
    ),
}

# Descriptive alias for callers that care about capability status rather than
# the historical executable-format name.
PROVIDER_WIRE_CAPABILITIES = PROVIDER_WIRE_ROUTES

_STRUCTURAL_SCHEMA_KEYWORDS = frozenset(
    {"$ref", "additionalProperties", "anyOf", "items", "oneOf", "properties", "required", "type"}
)


def _json_type_name(value: JSONValue) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    return "object"


def _type_matches(expected: str, actual: str) -> bool:
    return expected == actual or (expected == "number" and actual == "integer")


def _coverage_key(keyword: str, path: str) -> str:
    return keyword if path == "$" else f"{keyword}@{path}"


def _schema_type_names(schema: Mapping[str, object]) -> tuple[str, ...]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return (schema_type,)
    if isinstance(schema_type, Sequence) and not isinstance(schema_type, (str, bytes)):
        return tuple(item for item in schema_type if isinstance(item, str))
    return ()


def _matching_variants(schema: Mapping[str, object], payload: JSONValue) -> list[Mapping[str, object]]:
    actual_type = _json_type_name(payload)
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        matches: list[Mapping[str, object]] = []
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            declared_types = _schema_type_names(variant)
            if declared_types and not any(_type_matches(expected, actual_type) for expected in declared_types):
                continue
            required = variant.get("required")
            required_names = (
                {item for item in required if isinstance(item, str)} if isinstance(required, list) else set()
            )
            if required_names and (not isinstance(payload, dict) or not required_names <= payload.keys()):
                continue
            matches.append(variant)
        if matches:
            return matches
        return [variant for variant in variants if isinstance(variant, Mapping)]
    return []


def _collect_payload_coverage(
    schema: object,
    payload: JSONValue,
    *,
    path: str,
    handlers: set[str],
    schema_keywords: set[str],
    exercised_keywords: set[str],
) -> None:
    if not isinstance(schema, Mapping):
        return

    declared_types = _schema_type_names(schema)
    actual_type = _json_type_name(payload)
    if declared_types:
        matched_types = tuple(expected for expected in declared_types if _type_matches(expected, actual_type))
        types_to_report = matched_types or declared_types
        for expected in types_to_report:
            key = _coverage_key(f"type:{expected}", path)
            schema_keywords.add(key)
            if expected in handlers and _type_matches(expected, actual_type):
                exercised_keywords.add(key)

    for keyword in ("anyOf", "oneOf"):
        if isinstance(schema.get(keyword), list):
            key = _coverage_key(keyword, path)
            schema_keywords.add(key)
            if keyword in handlers:
                exercised_keywords.add(key)
    for variant in _matching_variants(schema, payload):
        _collect_payload_coverage(
            variant,
            payload,
            path=path,
            handlers=handlers,
            schema_keywords=schema_keywords,
            exercised_keywords=exercised_keywords,
        )

    if isinstance(payload, dict):
        properties = schema.get("properties")
        property_names = set(properties) if isinstance(properties, Mapping) else set()
        if isinstance(properties, Mapping):
            key = _coverage_key("properties", path)
            schema_keywords.add(key)
            if property_names & payload.keys():
                exercised_keywords.add(key)

        if "required" in schema:
            key = _coverage_key("required", path)
            schema_keywords.add(key)
            required = schema.get("required")
            required_names = (
                {item for item in required if isinstance(item, str)} if isinstance(required, list) else set()
            )
            if required_names <= payload.keys():
                exercised_keywords.add(key)

        if "additionalProperties" in schema:
            extra_names = payload.keys() - property_names
            additional_schema = schema.get("additionalProperties")
            if additional_schema is False:
                key = _coverage_key("additionalProperties", path)
                schema_keywords.add(key)
                if not extra_names:
                    exercised_keywords.add(key)
            elif extra_names:
                key = _coverage_key("additionalProperties", path)
                schema_keywords.add(key)
                exercised_keywords.add(key)

        if isinstance(properties, Mapping):
            for name, child_schema in properties.items():
                if isinstance(name, str) and name in payload:
                    _collect_payload_coverage(
                        child_schema,
                        payload[name],
                        path=f"{path}.properties.{name}",
                        handlers=handlers,
                        schema_keywords=schema_keywords,
                        exercised_keywords=exercised_keywords,
                    )

        additional_schema = schema.get("additionalProperties")
        if isinstance(additional_schema, Mapping):
            for name, item in payload.items():
                if name not in property_names:
                    _collect_payload_coverage(
                        additional_schema,
                        item,
                        path=f"{path}.additionalProperties.{name}",
                        handlers=handlers,
                        schema_keywords=schema_keywords,
                        exercised_keywords=exercised_keywords,
                    )

    if isinstance(payload, list) and isinstance(schema.get("items"), Mapping):
        key = _coverage_key("items", path)
        schema_keywords.add(key)
        exercised_keywords.add(key)
        for index, item in enumerate(payload):
            _collect_payload_coverage(
                schema["items"],
                item,
                path=f"{path}.items[{index}]",
                handlers=handlers,
                schema_keywords=schema_keywords,
                exercised_keywords=exercised_keywords,
            )


def construct_coverage(
    schema: SchemaRecord,
    payloads: Sequence[JSONValue],
    *,
    handler_names: Collection[str] | None = None,
) -> ConstructCoverage:
    """Compare schema constructs with generated values and live handlers.

    Type constructs are reported individually (for example ``type:array``),
    which makes removal of one recursive runtime handler visible in the
    receipt even when another construct still generates valid JSON.
    """

    if handler_names is None:
        from polylogue.schemas.synthetic.runtime import SCHEMA_CONSTRUCT_HANDLERS

        handler_names = SCHEMA_CONSTRUCT_HANDLERS.keys()
    handlers = set(handler_names)
    schema_keywords: set[str] = set()
    exercised: set[str] = set()
    for payload in payloads:
        _collect_payload_coverage(
            schema,
            payload,
            path="$",
            handlers=handlers,
            schema_keywords=schema_keywords,
            exercised_keywords=exercised,
        )

    return ConstructCoverage(
        schema_keywords=tuple(sorted(schema_keywords)),
        exercised_keywords=tuple(sorted(exercised)),
        missing_keywords=tuple(sorted(schema_keywords - exercised)),
    )


def build_wire_support_receipt(*, registry: object | None = None, seed: int = 20260805) -> WireSupportReceipt:
    """Validate executable routes through the selected schema and parser.

    The provider set comes from the package registry.  The parser call is the
    public ``parse_payload`` entry point used by production dispatch, so a
    route is healthy only when its generated artifact validates and produces a
    non-empty parsed session.
    """

    if registry is None:
        from polylogue.schemas.runtime_registry import SchemaRegistry

        registry = SchemaRegistry()
    from polylogue.schemas.synthetic.core import SyntheticCorpus
    from polylogue.schemas.validator import validate_provider_export
    from polylogue.sources.dispatch import parse_payload

    catalog_providers = tuple(sorted(registry.list_providers()))  # type: ignore[attr-defined]
    entries: list[WireSupportEntry] = []
    missing_routes: list[str] = []
    for provider in catalog_providers:
        route = PROVIDER_WIRE_ROUTES.get(provider)
        package = registry.get_package(provider, version="default")  # type: ignore[attr-defined]
        package_version = package.version if package is not None else None
        element_kind = package.default_element_kind if package is not None else None
        if route is None:
            missing_routes.append(provider)
            entries.append(
                WireSupportEntry(
                    provider=provider,
                    status="unsupported",
                    reason="no explicit synthetic wire route",
                    package_version=package_version,
                    element_kind=element_kind,
                    schema_valid=None,
                    parsed_session_count=0,
                    parsed_message_count=0,
                    construct_coverage=None,
                    validation_error="missing route",
                )
            )
            continue
        if route.status == "unsupported":
            entries.append(
                WireSupportEntry(
                    provider=provider,
                    status=route.status,
                    reason=route.reason,
                    package_version=package_version,
                    element_kind=element_kind,
                    schema_valid=None,
                    parsed_session_count=0,
                    parsed_message_count=0,
                    construct_coverage=None,
                )
            )
            continue

        schema = registry.get_element_schema(  # type: ignore[attr-defined]
            provider,
            version="default",
            element_kind=element_kind,
        )
        if schema is None or package is None or route.wire_format is None:
            entries.append(
                WireSupportEntry(
                    provider=provider,
                    status=route.status,
                    reason=None,
                    package_version=package_version,
                    element_kind=element_kind,
                    schema_valid=False,
                    parsed_session_count=0,
                    parsed_message_count=0,
                    construct_coverage=None,
                    validation_error="selected package schema is unavailable",
                )
            )
            continue

        schema_valid = False
        parsed_sessions = []
        payloads: list[JSONValue] = []
        validation_error: str | None = None
        try:
            corpus = SyntheticCorpus.for_provider(provider, version=package.version, element_kind=element_kind)
            batch = corpus.generate_batch(count=1, messages_per_session=range(4, 5), seed=seed)
            raw = batch.raw_items[0]
            if route.wire_format.encoding == "jsonl":
                import json

                payload: JSONValue = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
                payloads = payload if isinstance(payload, list) else []
            else:
                import json

                payload = json.loads(raw)
                payloads = [payload] if isinstance(payload, (dict, list)) else []
            validation_results = [validate_provider_export(item, provider, strict=False) for item in payloads]
            schema_valid = bool(validation_results) and all(result.is_valid for result in validation_results)
            if not schema_valid:
                validation_error = "; ".join(error for result in validation_results for error in result.errors) or (
                    "selected package schema rejected generated payload"
                )
            parsed_sessions = parse_payload(provider, payload, f"synthetic-wire-receipt:{provider}")
        except Exception as exc:  # Receipt records route failures instead of hiding them.
            validation_error = f"{type(exc).__name__}: {exc}"

        coverage = construct_coverage(schema, payloads)
        entries.append(
            WireSupportEntry(
                provider=provider,
                status=route.status,
                reason=None,
                package_version=package_version,
                element_kind=element_kind,
                schema_valid=schema_valid,
                parsed_session_count=len(parsed_sessions),
                parsed_message_count=sum(len(session.messages) for session in parsed_sessions),
                construct_coverage=coverage,
                validation_error=validation_error,
            )
        )

    return WireSupportReceipt(
        catalog_providers=catalog_providers,
        entries=tuple(entries),
        missing_routes=tuple(sorted(missing_routes)),
    )


__all__ = [
    "ConstructCoverage",
    "PROVIDER_WIRE_FORMATS",
    "PROVIDER_WIRE_CAPABILITIES",
    "PROVIDER_WIRE_ROUTES",
    "TreeConfig",
    "WireCapabilityStatus",
    "WireEncoding",
    "WireFormat",
    "WireRoute",
    "WireSupportEntry",
    "WireSupportReceipt",
    "UnsupportedSyntheticWireRouteError",
    "build_wire_support_receipt",
    "construct_coverage",
]
