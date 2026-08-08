"""Wire format configuration for provider export formats.

Describes HOW each provider structures their export data — encoding type,
tree vs. linear vs. JSONL, and message location paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, cast

from polylogue.archive.raw_payload.decode import JSONValue

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.models import SchemaRecord, SyntheticGenerationBatch
    from polylogue.sources.parsers.base_models import ParsedSession

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
    nonrepresentable_keywords: tuple[str, ...] = ()
    nonrepresentable_reasons: tuple[tuple[str, str], ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing_keywords


@dataclass(frozen=True)
class WireParserWitness:
    """One schema witness exercised through the production parser route."""

    index: int
    exercised_keywords: tuple[str, ...]
    parsed_session_count: int
    parsed_message_count: int
    validation_error: str | None = None
    artifact_kind: Literal["baseline", "coverage"] = "coverage"
    artifact_evidence: tuple[str, ...] = ()

    @property
    def healthy(self) -> bool:
        return (
            bool(self.exercised_keywords)
            and self.parsed_session_count > 0
            and self.parsed_message_count > 0
            and bool(self.artifact_evidence)
            and self.validation_error is None
        )


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
    parser_witnesses: tuple[WireParserWitness, ...] = ()

    @property
    def healthy(self) -> bool:
        return self.status == "unsupported" or (
            self.schema_valid is True
            and self.parsed_session_count > 0
            and self.parsed_message_count > 0
            and self.construct_coverage is not None
            and self.construct_coverage.complete
            and self.validation_error is None
            and bool(self.parser_witnesses)
            and all(witness.healthy for witness in self.parser_witnesses)
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
                    "parser_witnesses": [
                        {
                            "index": witness.index,
                            "exercised_keywords": list(witness.exercised_keywords),
                            "parsed_session_count": witness.parsed_session_count,
                            "parsed_message_count": witness.parsed_message_count,
                            "validation_error": witness.validation_error,
                            "artifact_kind": witness.artifact_kind,
                            "artifact_evidence": list(witness.artifact_evidence),
                            "healthy": witness.healthy,
                        }
                        for witness in entry.parser_witnesses
                    ],
                    "construct_coverage": (
                        {
                            "schema_keywords": list(entry.construct_coverage.schema_keywords),
                            "exercised_keywords": list(entry.construct_coverage.exercised_keywords),
                            "missing_keywords": list(entry.construct_coverage.missing_keywords),
                            "nonrepresentable_keywords": list(entry.construct_coverage.nonrepresentable_keywords),
                            "nonrepresentable_reasons": [
                                {"keyword": keyword, "reason": reason}
                                for keyword, reason in entry.construct_coverage.nonrepresentable_reasons
                            ],
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


class _CoverageCorpus(Protocol):
    schema: SchemaRecord
    workload_profile: SchemaRecord | None
    _coverage_branch_choices: dict[str, int]
    _coverage_type_choices: dict[str, str]
    _coverage_null_paths: set[str]
    _coverage_witness_mode: bool

    def generate_batch(
        self,
        count: int = 5,
        messages_per_session: range = range(3, 15),
        seed: int | None = None,
    ) -> SyntheticGenerationBatch: ...


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


def _runtime_coverage_path(path: str) -> str:
    # Union segments are part of the runtime path. A nested union must not
    # reuse its parent's key, since its branch choice is independent.
    return path


def _route_nonrepresentable_reasons(
    provider: str,
    missing_keywords: Collection[str],
    *,
    package_version: str,
) -> dict[str, str]:
    """Prove nodes discarded by a provider's wire normalizer are unreachable."""
    reasons: dict[str, str] = {}
    prefixes: tuple[tuple[str, str], ...]
    if provider == "codex":
        prefixes = (("$.properties.payload", "Codex flat-record shaping removes the payload envelope"),)
    elif provider == "claude-ai":
        prefixes = (
            (
                "$.properties.chat_messages.items[*]",
                "Claude AI wire shaping clears each chat message and retains only uuid, sender, text, and created_at",
            ),
        )
    elif provider == "gemini":
        prefixes = (
            (
                "$.properties.chunkedPrompt.properties.chunks.items[*]",
                "Gemini wire shaping clears each chunk and retains only role, text, and createTime",
            ),
            (
                "$.properties.chunkedPrompt.properties.pendingInputs",
                "Gemini linear wire shaping rebuilds chunkedPrompt and does not carry pendingInputs",
            ),
        )
    elif provider == "chatgpt":
        prefixes = (
            (
                "$.properties.raw_provider_payload.properties.mapping",
                "ChatGPT coverage wire shaping removes the nested raw-provider mapping before the parser route, matching the tree route's parser-owned mapping container",
            ),
        )
    elif provider == "claude-code":
        prefixes = (
            (
                "$.properties.message.properties.content.anyOf[1].items[*].properties.content.anyOf[1]",
                "Claude Code wire shaping preserves message content but its fallback emits scalar text and supported block forms, never nested array content/source blocks",
            ),
        )
    else:
        prefixes = ()

    retained_suffixes = {
        "$.properties.chat_messages.items[*].properties.uuid",
        "$.properties.chat_messages.items[*].properties.text",
        "$.properties.chat_messages.items[*].properties.created_at",
        "$.properties.chunkedPrompt.properties.chunks.items[*].properties.role",
        "$.properties.chunkedPrompt.properties.chunks.items[*].properties.text",
        "$.properties.chunkedPrompt.properties.chunks.items[*].properties.createTime",
    }
    for keyword in missing_keywords:
        path = keyword.split("@", 1)[1] if "@" in keyword else "$"
        if provider == "chatgpt" and package_version == "v1":
            reasons[keyword] = (
                "ChatGPT v1 parser route retains normalized conversation fields but does not represent "
                "export-only media metadata at this exact package selection"
            )
            continue
        if (
            provider == "chatgpt"
            and keyword.startswith("type:null@")
            and path.endswith(".properties.message")
            and ".properties.mapping.additionalProperties.*." in path
        ):
            reasons[keyword] = "ChatGPT wire shaping coerces a null message field to the route's object record"
            continue
        for prefix, reason in prefixes:
            if path.startswith(prefix) and path not in retained_suffixes:
                reasons[keyword] = reason
                break
    return reasons


def _schema_type_names(schema: Mapping[str, object]) -> tuple[str, ...]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return (schema_type,)
    if isinstance(schema_type, Sequence) and not isinstance(schema_type, (str, bytes)):
        return tuple(item for item in schema_type if isinstance(item, str))
    return ()


def _matching_variants(schema: Mapping[str, object], payload: JSONValue) -> list[tuple[int, Mapping[str, object]]]:
    actual_type = _json_type_name(payload)
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        matches: list[tuple[int, Mapping[str, object]]] = []
        for index, variant in enumerate(variants):
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
            matches.append((index, variant))
        if matches:
            return matches
        return []
    return []


def _collect_schema_obligations(
    schema: object,
    *,
    path: str,
    obligations: set[str],
) -> None:
    """Collect node-local obligations without consulting generated payloads."""
    if not isinstance(schema, Mapping):
        return

    for schema_type in _schema_type_names(schema):
        obligations.add(_coverage_key(f"type:{schema_type}", path))

    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        obligations.add(_coverage_key(keyword, path))
        for index, variant in enumerate(variants):
            if isinstance(variant, Mapping):
                _collect_schema_obligations(
                    variant,
                    path=f"{path}.{keyword}[{index}]",
                    obligations=obligations,
                )

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        obligations.add(_coverage_key("properties", path))
        for name, child_schema in properties.items():
            if isinstance(name, str):
                _collect_schema_obligations(
                    child_schema,
                    path=f"{path}.properties.{name}",
                    obligations=obligations,
                )

    if "required" in schema:
        obligations.add(_coverage_key("required", path))

    if "additionalProperties" in schema:
        obligations.add(_coverage_key("additionalProperties", path))
        additional_schema = schema.get("additionalProperties")
        if isinstance(additional_schema, Mapping):
            _collect_schema_obligations(
                additional_schema,
                path=f"{path}.additionalProperties.*",
                obligations=obligations,
            )

    items = schema.get("items")
    if isinstance(items, Mapping):
        obligations.add(_coverage_key("items", path))
        _collect_schema_obligations(items, path=f"{path}.items[*]", obligations=obligations)


def _coverage_branch_plans(
    schema: object,
    *,
    path: str = "$",
    choices: Mapping[str, int] | None = None,
    plans: list[dict[str, int]] | None = None,
) -> list[dict[str, int]]:
    """Return one bounded branch-choice plan per schema union branch."""
    if plans is None:
        plans = []
    if choices is None:
        choices = {}
    if not isinstance(schema, Mapping):
        return plans

    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        for index, variant in enumerate(variants):
            if not isinstance(variant, Mapping):
                continue
            branch_choices = dict(choices)
            branch_choices[_runtime_coverage_path(path)] = index
            plans.append(branch_choices)
            _coverage_branch_plans(
                variant,
                path=f"{path}.{keyword}[{index}]",
                choices=branch_choices,
                plans=plans,
            )

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for name, child_schema in properties.items():
            if isinstance(name, str):
                _coverage_branch_plans(
                    child_schema,
                    path=f"{path}.properties.{name}",
                    choices=choices,
                    plans=plans,
                )

    additional_schema = schema.get("additionalProperties")
    if isinstance(additional_schema, Mapping):
        _coverage_branch_plans(
            additional_schema,
            path=f"{path}.additionalProperties.*",
            choices=choices,
            plans=plans,
        )

    items = schema.get("items")
    if isinstance(items, Mapping):
        _coverage_branch_plans(items, path=f"{path}.items[*]", choices=choices, plans=plans)
    return plans


def _force_coverage_frequencies(value: object) -> None:
    if not isinstance(value, dict):
        return
    if "x-polylogue-frequency" in value:
        value["x-polylogue-frequency"] = 1.0
    properties = value.get("properties")
    if isinstance(properties, Mapping):
        for child in properties.values():
            _force_coverage_frequencies(child)
    additional_schema = value.get("additionalProperties")
    _force_coverage_frequencies(additional_schema)
    _force_coverage_frequencies(value.get("items"))
    for keyword in ("anyOf", "oneOf"):
        variants = value.get(keyword)
        if isinstance(variants, list):
            for variant in variants:
                _force_coverage_frequencies(variant)


def _collect_type_choices(
    schema: object,
    *,
    path: str = "$",
    choices: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    if choices is None:
        choices = []
    if not isinstance(schema, Mapping):
        return choices
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        choices.extend((path, item) for item in schema_type if isinstance(item, str))
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for name, child in properties.items():
            if isinstance(name, str):
                _collect_type_choices(child, path=f"{path}.properties.{name}", choices=choices)
    additional_schema = schema.get("additionalProperties")
    if isinstance(additional_schema, Mapping):
        _collect_type_choices(additional_schema, path=f"{path}.additionalProperties.*", choices=choices)
    items = schema.get("items")
    if isinstance(items, Mapping):
        _collect_type_choices(items, path=f"{path}.items[*]", choices=choices)
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list):
            for index, variant in enumerate(variants):
                _collect_type_choices(variant, path=f"{path}.{keyword}[{index}]", choices=choices)
    return choices


def _coverage_path_matches_plan(path: str, plan: Mapping[str, int]) -> bool:
    for match in re.finditer(r"\.(?:anyOf|oneOf)\[\d+\]", path):
        branch_index = int(match.group(0).split("[")[1][:-1])
        if plan.get(path[: match.start()]) != branch_index:
            return False
    return True


def _required_structural_type_choices(
    path: str,
    type_options: Mapping[str, tuple[str, ...]],
) -> dict[str, str]:
    choices: dict[str, str] = {}
    for ancestor_path, ancestor_options in type_options.items():
        if not path.startswith(f"{ancestor_path}."):
            continue
        relative_path = path.removeprefix(ancestor_path)
        wanted_type = "array" if relative_path.startswith(".items[") else "object"
        structural_type = next(
            (item for item in ancestor_options if item == wanted_type),
            next((item for item in ancestor_options if item in {"array", "object"}), None),
        )
        if structural_type is not None:
            choices[ancestor_path] = structural_type
    return choices


def _type_choice_witness_groups(
    choices: Collection[tuple[str, Mapping[str, str]]],
) -> list[list[tuple[str, Mapping[str, str]]]]:
    """Partition type alternatives into bounded, path-independent witnesses."""
    groups: list[list[tuple[str, Mapping[str, str]]]] = []
    for choice in sorted(choices, key=lambda value: (value[0].count("."), value[0], tuple(sorted(value[1].items())))):
        path = choice[0]
        for group in groups:
            if all(
                path != other_path
                and not path.startswith(f"{other_path}.")
                and not other_path.startswith(f"{path}.")
                and all(other_choices.get(key) in {None, value} for key, value in choice[1].items())
                and all(choice[1].get(key) in {None, value} for key, value in other_choices.items())
                for other_path, other_choices in group
            ):
                group.append(choice)
                break
        else:
            groups.append([choice])
    return groups


def generate_coverage_witnesses(
    corpus: _CoverageCorpus,
    *,
    seed: int,
    max_witnesses: int = 128,
) -> list[bytes]:
    """Generate bounded, route-shaped witnesses for schema-node coverage."""
    plans: list[dict[str, int]] = [{}] + _coverage_branch_plans(corpus.schema)
    unique_plans: list[dict[str, int]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for plan in plans:
        key = tuple(sorted(plan.items()))
        if key in seen:
            continue
        seen.add(key)
        unique_plans.append(plan)
    type_choices = _collect_type_choices(corpus.schema)
    type_options: dict[str, tuple[str, ...]] = {}
    for type_path, type_name in type_choices:
        type_options[type_path] = (*type_options.get(type_path, ()), type_name)
    choices_by_plan: dict[int, list[tuple[str, Mapping[str, str]]]] = {}
    for type_path, type_name in type_choices:
        assignment: dict[str, str] = {type_path: type_name}
        assignment.update(_required_structural_type_choices(type_path, type_options))
        for plan_index, plan in enumerate(unique_plans):
            if _coverage_path_matches_plan(type_path, plan):
                choices_by_plan.setdefault(plan_index, []).append((type_path, assignment))
                break

    choice_groups = {
        plan_index: _type_choice_witness_groups(choices) for plan_index, choices in choices_by_plan.items()
    }
    witness_count = len(unique_plans) + sum(len(groups) for groups in choice_groups.values())
    if witness_count > max_witnesses:
        raise ValueError(f"coverage witness plan has {witness_count} witnesses, exceeding bound {max_witnesses}")

    original_schema = corpus.schema
    original_profile = corpus.workload_profile
    original_choices = corpus._coverage_branch_choices
    original_type_choices = corpus._coverage_type_choices
    original_null_paths = corpus._coverage_null_paths
    original_mode = corpus._coverage_witness_mode
    raw_items: list[bytes] = []
    try:
        corpus.schema = copy.deepcopy(original_schema)
        _force_coverage_frequencies(corpus.schema)
        corpus.workload_profile = None
        corpus._coverage_witness_mode = True
        corpus._coverage_type_choices = {}
        corpus._coverage_null_paths = set()
        plan_type_choices = {
            index: {
                key: value
                for branch_path in plan
                for key, value in _required_structural_type_choices(branch_path, type_options).items()
            }
            for index, plan in enumerate(unique_plans)
        }
        for index, plan in enumerate(unique_plans):
            corpus._coverage_branch_choices = plan
            corpus._coverage_type_choices = plan_type_choices[index]
            batch = corpus.generate_batch(count=1, messages_per_session=range(4, 5), seed=seed + index)
            raw_items.extend(batch.raw_items)
        for plan_index, groups in choice_groups.items():
            for group in groups:
                group_type_choices = {path: type_name for _, choices in group for path, type_name in choices.items()}
                corpus._coverage_type_choices = {**plan_type_choices[plan_index], **group_type_choices}
                corpus._coverage_branch_choices = unique_plans[plan_index]
                batch = corpus.generate_batch(count=1, messages_per_session=range(4, 5), seed=seed + len(raw_items))
                raw_items.extend(batch.raw_items)
        corpus._coverage_branch_choices = {}
        corpus._coverage_type_choices = {}
        corpus._coverage_null_paths = set()
        for index in range(len(raw_items), max_witnesses):
            batch = corpus.generate_batch(count=1, messages_per_session=range(4, 5), seed=seed + index)
            raw_items.extend(batch.raw_items)
    finally:
        corpus.schema = original_schema
        corpus.workload_profile = original_profile
        corpus._coverage_branch_choices = original_choices
        corpus._coverage_type_choices = original_type_choices
        corpus._coverage_null_paths = original_null_paths
        corpus._coverage_witness_mode = original_mode
    return raw_items


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
        if not isinstance(schema.get(keyword), list):
            continue
        matches = _matching_variants(schema, payload)
        key = _coverage_key(keyword, path)
        if keyword in handlers and matches:
            exercised_keywords.add(key)
        for index, variant in matches:
            _collect_payload_coverage(
                variant,
                payload,
                path=f"{path}.{keyword}[{index}]",
                handlers=handlers,
                schema_keywords=schema_keywords,
                exercised_keywords=exercised_keywords,
            )

    if isinstance(payload, dict):
        properties = schema.get("properties")
        property_names = set(properties) if isinstance(properties, Mapping) else set()
        if isinstance(properties, Mapping):
            key = _coverage_key("properties", path)
            exercised_keywords.add(key)

        if "required" in schema:
            key = _coverage_key("required", path)
            required = schema.get("required")
            required_names = (
                {item for item in required if isinstance(item, str)} if isinstance(required, list) else set()
            )
            if required_names <= payload.keys():
                exercised_keywords.add(key)

        if "additionalProperties" in schema:
            extra_names = payload.keys() - property_names
            additional_schema = schema.get("additionalProperties")
            key = _coverage_key("additionalProperties", path)
            if additional_schema is False:
                if not extra_names:
                    exercised_keywords.add(key)
            elif additional_schema is True or extra_names:
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
                        path=f"{path}.additionalProperties.*",
                        handlers=handlers,
                        schema_keywords=schema_keywords,
                        exercised_keywords=exercised_keywords,
                    )

    if isinstance(payload, list) and isinstance(schema.get("items"), Mapping):
        key = _coverage_key("items", path)
        if payload:
            exercised_keywords.add(key)
        for item in payload:
            _collect_payload_coverage(
                schema["items"],
                item,
                path=f"{path}.items[*]",
                handlers=handlers,
                schema_keywords=schema_keywords,
                exercised_keywords=exercised_keywords,
            )


def construct_coverage(
    schema: SchemaRecord,
    payloads: Sequence[JSONValue],
    *,
    handler_names: Collection[str] | None = None,
    nonrepresentable_keywords: Collection[str] = (),
    nonrepresentable_reason: str = "explicitly classified as nonrepresentable by the selected route",
    nonrepresentable_reasons: Mapping[str, str] | None = None,
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
    _collect_schema_obligations(schema, path="$", obligations=schema_keywords)
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

    nonrepresentable = tuple(sorted(set(nonrepresentable_keywords) & schema_keywords))
    nonrepresentable_set = set(nonrepresentable)
    reasons = tuple(
        (keyword, (nonrepresentable_reasons or {}).get(keyword, nonrepresentable_reason))
        for keyword in nonrepresentable
    )
    return ConstructCoverage(
        schema_keywords=tuple(sorted(schema_keywords)),
        exercised_keywords=tuple(sorted(exercised)),
        missing_keywords=tuple(sorted(schema_keywords - exercised - nonrepresentable_set)),
        nonrepresentable_keywords=nonrepresentable,
        nonrepresentable_reasons=reasons,
    )


def _payload_string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(text for child in value.values() for text in _payload_string_values(child))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(text for child in value for text in _payload_string_values(child))
    return ()


def _normalise_evidence_text(value: str) -> str:
    return " ".join(value.split())


def _parser_evidence_nodes(provider: str, payload: JSONValue) -> tuple[Mapping[str, JSONValue], ...]:
    """Return the route-owned conversational nodes from one wire artifact."""
    nodes: list[Mapping[str, JSONValue]] = []
    if isinstance(payload, dict):
        session = payload.get("session")
        turns = session.get("turns") if isinstance(session, dict) else None
        if isinstance(turns, list):
            nodes.extend(
                turn
                for turn in turns
                if isinstance(turn, dict)
                and isinstance(turn.get("provider_turn_id"), str)
                and isinstance(turn.get("role"), str)
                and isinstance(turn.get("text"), str)
            )
        raw_provider_payload = payload.get("raw_provider_payload")
        if isinstance(raw_provider_payload, (dict, list)):
            nodes.extend(_parser_evidence_nodes(provider, raw_provider_payload))
    if provider == "chatgpt" and isinstance(payload, dict):
        mapping = payload.get("mapping")
        if isinstance(mapping, dict):
            for node in mapping.values():
                if not isinstance(node, dict):
                    continue
                message = node.get("message")
                if (
                    isinstance(message, dict)
                    and isinstance(message.get("author"), dict)
                    and isinstance(message.get("content"), dict)
                ):
                    nodes.append(node)
    if provider == "claude-ai" and isinstance(payload, dict):
        messages = payload.get("chat_messages")
        if isinstance(messages, list):
            nodes.extend(
                message
                for message in messages
                if isinstance(message, dict)
                and isinstance(message.get("uuid"), str)
                and isinstance(message.get("sender"), str)
                and isinstance(message.get("text"), str)
            )
    if provider == "gemini" and isinstance(payload, dict):
        prompt = payload.get("chunkedPrompt")
        chunks = prompt.get("chunks") if isinstance(prompt, dict) else None
        if isinstance(chunks, list):
            nodes.extend(
                chunk
                for chunk in chunks
                if isinstance(chunk, dict) and isinstance(chunk.get("role"), str) and isinstance(chunk.get("text"), str)
            )
    if provider in {"claude-code", "codex"} and isinstance(payload, list):
        for record in payload:
            if not isinstance(record, dict):
                continue
            if provider == "claude-code":
                if record.get("type") in {"user", "assistant", "system"} and isinstance(record.get("message"), dict):
                    nodes.append(record)
                continue
            record_type = record.get("type")
            if record_type == "message" and isinstance(record.get("id"), str):
                nodes.append(record)
                continue
            nested = record.get("payload")
            if (
                record_type == "response_item"
                and isinstance(nested, dict)
                and nested.get("type") == "message"
                and isinstance(nested.get("id"), str)
            ) or (
                record_type == "event_msg"
                and isinstance(nested, dict)
                and nested.get("type") in {"user_message", "agent_message"}
            ):
                nodes.append(nested)
    return tuple(nodes)


def _parser_artifact_evidence(
    sessions: Sequence[ParsedSession],
    provider: str,
    payload: JSONValue,
    fallback_id: str,
) -> tuple[str, ...]:
    """Return materialized-message evidence found in this exact artifact.

    Parser counts are deliberately insufficient here. A misrouted parser can
    return a valid session from another artifact, which would otherwise let
    one witness borrow another witness's success. The evidence digest is
    emitted only when meaningful parsed message/block text is present in the
    artifact being validated.
    """

    from polylogue.sources.dispatch import message_carries_authored_content

    evidence_nodes = _parser_evidence_nodes(provider, payload)
    evidence: list[str] = []
    seen: set[str] = set()
    for session in sessions:
        session_identity = session.provider_session_id
        session_is_bound = session_identity == fallback_id or session_identity in _payload_string_values(payload)
        if not session_is_bound:
            continue
        for message in session.messages:
            if not message_carries_authored_content(message):
                continue
            message_identity = message.provider_message_id
            candidates = [message.text, *(block.text for block in message.blocks)]
            for candidate in candidates:
                if not isinstance(candidate, str) or not candidate.strip():
                    continue
                normalized = _normalise_evidence_text(candidate)
                if not normalized:
                    continue
                for node in evidence_nodes:
                    node_texts = tuple(_normalise_evidence_text(text) for text in _payload_string_values(node))
                    identity_bound = not message_identity or message_identity in node_texts
                    content_bound = normalized in node_texts
                    structured_content_bound = (
                        bool(message.blocks) and bool(message_identity) and identity_bound and bool(node_texts)
                    )
                    if not identity_bound or not (content_bound or structured_content_bound):
                        continue
                    node_bytes = json.dumps(node, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    digest = hashlib.sha256(node_bytes).hexdigest()
                    token = f"{session_identity}:message:{message_identity or 'unidentified'}:sha256:{digest}"
                    if token not in seen:
                        seen.add(token)
                        evidence.append(token)
                    break
                if evidence and evidence[-1].startswith(
                    f"{session_identity}:message:{message_identity or 'unidentified'}:"
                ):
                    break
    return tuple(evidence)


def build_wire_support_receipt(*, registry: object | None = None, seed: int = 20260805) -> WireSupportReceipt:
    """Validate executable routes through the selected schema and parser.

    The provider set comes from the package registry.  The parser call is the
    public ``parse_payload`` entry point used by production dispatch. Every
    artifact, including the baseline, must pass the production
    positive-conversational-evidence filter and carry artifact-local
    materialized content evidence.
    """

    if registry is None:
        from polylogue.schemas.runtime_registry import SchemaRegistry

        registry = SchemaRegistry()
    from polylogue.schemas.synthetic.core import SyntheticCorpus
    from polylogue.schemas.synthetic.selection import select_synthetic_schema
    from polylogue.schemas.validator import SchemaValidator, ValidationResult
    from polylogue.sources.dispatch import parse_payload, require_positive_conversational_evidence

    catalog_providers = tuple(sorted(registry.list_providers()))  # type: ignore[attr-defined]
    entries: list[WireSupportEntry] = []
    missing_routes: list[str] = []
    for provider in catalog_providers:
        route = PROVIDER_WIRE_ROUTES.get(provider)
        catalog = registry.load_package_catalog(provider)  # type: ignore[attr-defined]
        selections = tuple(
            (package, element)
            for package in (catalog.packages if catalog is not None else ())
            for element in package.elements
        )
        if not selections:
            package = registry.get_package(provider, version="default")  # type: ignore[attr-defined]
            selections = ((package, None),)

        for package, element in selections:
            package_version = package.version if package is not None else None
            element_kind = (
                element.element_kind
                if element is not None
                else package.default_element_kind
                if package is not None
                else None
            )
            if element is not None and not element.supported:
                entries.append(
                    WireSupportEntry(
                        provider=provider,
                        status="unsupported",
                        reason="catalog element is marked unsupported",
                        package_version=package_version,
                        element_kind=element_kind,
                        schema_valid=None,
                        parsed_session_count=0,
                        parsed_message_count=0,
                        construct_coverage=None,
                    )
                )
                continue
            if route is None:
                if provider not in missing_routes:
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

            if package is None or route.wire_format is None or element_kind is None:
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
            selection = None
            parser_witnesses: list[WireParserWitness] = []
            parser_errors: list[str] = []
            try:
                selection = select_synthetic_schema(
                    provider,
                    version=package.version,
                    element_kind=element_kind,
                    registry_factory=cast(Any, lambda: registry),
                )
                if selection.package_version != package.version or selection.element_kind != element_kind:
                    raise ValueError(
                        "synthetic selection identity diverged from receipt package: "
                        f"{selection.package_version}/{selection.element_kind} != {package.version}/{element_kind}"
                    )
                if selection.wire_format != route.wire_format:
                    raise ValueError("synthetic selection wire format diverged from declared route")

                corpus = SyntheticCorpus.from_selection(selection)
                raw_items = [
                    corpus.generate_batch(count=1, messages_per_session=range(4, 5), seed=seed).raw_items[0],
                    *generate_coverage_witnesses(corpus, seed=seed + 1),
                ]
                validator = SchemaValidator(selection.schema, strict=False)
                validation_results: list[ValidationResult] = []
                schema_resolution = None
                if selection.element_kind is not None:
                    from polylogue.schemas.packages import SchemaResolution

                    schema_resolution = SchemaResolution(
                        provider=selection.provider,
                        package_version=selection.package_version,
                        element_kind=selection.element_kind,
                        exact_structure_id=None,
                        bundle_scope=None,
                        reason="package_catalog",
                    )
                for index, raw in enumerate(raw_items):
                    if route.wire_format.encoding == "jsonl":
                        payload: JSONValue = [
                            json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
                        ]
                        payload_items = payload if isinstance(payload, list) else []
                    else:
                        payload = json.loads(raw)
                        payload_items = [payload] if isinstance(payload, (dict, list)) else []
                    artifact_results = [validator.validate(item) for item in payload_items]
                    validation_results.extend(artifact_results)
                    artifact_validation_error = (
                        "; ".join(error for result in artifact_results for error in result.errors) or None
                    )
                    artifact_coverage = construct_coverage(selection.schema, payload_items)
                    parse_error: str | None = None
                    artifact_sessions = []
                    try:
                        parser_payload = payload
                        if provider == "chatgpt" and isinstance(payload, dict):
                            # Validate and account for the complete envelope, but
                            # keep the optional native subpayload from selecting a
                            # second schema-shaped tree during parser dispatch.
                            parser_payload = dict(payload)
                            parser_payload.pop("raw_provider_payload", None)
                        parsed_sessions_for_artifact = parse_payload(
                            provider,
                            parser_payload,
                            f"synthetic-wire-receipt:{provider}:{package.version}:{element_kind}:{index}",
                            schema_resolution=schema_resolution,
                        )
                        artifact_sessions = require_positive_conversational_evidence(
                            parsed_sessions_for_artifact,
                            provider=provider,
                            source_path=f"synthetic-wire-receipt:{provider}:{package.version}:{element_kind}:{index}",
                        )
                    except Exception as exc:  # Keep the witness failure in the receipt.
                        parse_error = f"{type(exc).__name__}: {exc}"
                    artifact_evidence = _parser_artifact_evidence(
                        artifact_sessions,
                        provider,
                        payload,
                        f"synthetic-wire-receipt:{provider}:{package.version}:{element_kind}:{index}",
                    )
                    parsed_sessions.extend(artifact_sessions)
                    artifact_kind: Literal["baseline", "coverage"] = "baseline" if index == 0 else "coverage"
                    parser_witnesses.append(
                        WireParserWitness(
                            index=-1 if index == 0 else index - 1,
                            exercised_keywords=artifact_coverage.exercised_keywords,
                            parsed_session_count=len(artifact_sessions),
                            parsed_message_count=sum(len(session.messages) for session in artifact_sessions),
                            validation_error=parse_error or artifact_validation_error,
                            artifact_kind=artifact_kind,
                            artifact_evidence=artifact_evidence,
                        )
                    )
                    if (
                        artifact_sessions
                        and artifact_evidence
                        and artifact_validation_error is None
                        and parse_error is None
                    ):
                        payloads.extend(payload_items)
                    if parse_error is not None:
                        label = "baseline" if index == 0 else f"coverage witness {index - 1}"
                        parser_errors.append(f"{label} parser: {parse_error}")
                schema_valid = bool(validation_results) and all(result.is_valid for result in validation_results)
                if not schema_valid:
                    parser_errors.append(
                        "; ".join(error for result in validation_results for error in result.errors)
                        or "selected package schema rejected generated payload"
                    )
            except Exception as exc:  # Receipt records route failures instead of hiding them.
                validation_error = f"{type(exc).__name__}: {exc}"

            if parser_errors:
                validation_error = "; ".join(parser_errors)

            if selection is not None:
                witnessed = construct_coverage(selection.schema, payloads)
                nonrepresentable_reasons = _route_nonrepresentable_reasons(
                    provider,
                    witnessed.missing_keywords,
                    package_version=package.version,
                )
                coverage = construct_coverage(
                    selection.schema,
                    payloads,
                    nonrepresentable_keywords=nonrepresentable_reasons,
                    nonrepresentable_reasons=nonrepresentable_reasons,
                )
            else:
                coverage = None
            entries.append(
                WireSupportEntry(
                    provider=provider,
                    status=route.status,
                    reason=None,
                    package_version=selection.package_version if selection is not None else package_version,
                    element_kind=selection.element_kind if selection is not None else element_kind,
                    schema_valid=schema_valid,
                    parsed_session_count=len(parsed_sessions),
                    parsed_message_count=sum(len(session.messages) for session in parsed_sessions),
                    construct_coverage=coverage,
                    validation_error=validation_error,
                    parser_witnesses=tuple(parser_witnesses),
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
    "WireParserWitness",
    "WireRoute",
    "WireSupportEntry",
    "WireSupportReceipt",
    "UnsupportedSyntheticWireRouteError",
    "build_wire_support_receipt",
    "construct_coverage",
    "generate_coverage_witnesses",
]
