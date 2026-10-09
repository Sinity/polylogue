"""Typed parser contracts shared across provider parsers."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    PrivateAttr,
    SkipValidation,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import (
    BlockType,
    MaterialOrigin,
    PolylogueStrEnum,
    Provider,
    SessionKind,
    TitleSource,
    ToolOutcome,
    ToolResultUnknownReason,
    WebConstructType,
)
from polylogue.core.json import detach_borrowed_json
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate, MemberAddressingMode
from polylogue.core.security import sanitize_path as _sanitize_path_helper
from polylogue.core.timestamps import parse_timestamp
from polylogue.core.types import AttachmentDirection, AttachmentUploadOrigin
from polylogue.sources.staged_raw_payload import StagedRawPayload


def _require_string_mapping_keys(value: object, *, field: str) -> object:
    """Refuse keys that Pydantic's ``str`` mapping schema would coerce.

    In particular, ``{b"a": 1, "a": 2}`` silently becomes ``{"a": 2}``
    during validation.  These fields declare JSON object keys, so accepting
    that input would discard parser evidence before hashing or storage sees it.
    Nested values typed as ``object`` retain their original mapping keys and
    are handled by the semantic hash projection.
    """
    if value is None:
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} requires an object mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} requires string mapping keys")
    return detach_borrowed_json(value)


class AdmissionUnit(PolylogueStrEnum):
    """Input-unit levels covered by the parser admission contract."""

    OUTER_RECORD = "outer_record"
    MESSAGE = "message"
    PART = "part"
    BLOCK = "block"


class AdmissionDisposition(PolylogueStrEnum):
    """The only terminal dispositions an input unit may receive."""

    MATERIALIZED = "materialized"
    TYPED_UNKNOWN = "typed_unknown"
    TYPED_REFUSAL = "typed_refusal"


class AdmissionUnknownReason(PolylogueStrEnum):
    UNRECOGNIZED_TYPE = "unrecognized_type"
    UNSUPPORTED_SHAPE = "unsupported_shape"
    MISSING_PAYLOAD = "missing_payload"
    DRIFTED_SIBLING = "drifted_sibling"
    EMPTY_CONTENT = "empty_content"


class AdmissionRefusalReason(PolylogueStrEnum):
    MALFORMED = "malformed"
    INVALID_ROLE = "invalid_role"
    UNSUPPORTED_PROVIDER = "unsupported_provider"
    CONSERVATION_MISMATCH = "conservation_mismatch"
    EMPTY_SESSION = "empty_session"


class AdmissionOutcome(BaseModel):
    """One terminal, typed outcome for one parser input unit."""

    unit: AdmissionUnit
    ordinal: int
    key: str
    disposition: AdmissionDisposition
    reason: AdmissionUnknownReason | AdmissionRefusalReason | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> AdmissionOutcome:
        if self.disposition is AdmissionDisposition.MATERIALIZED and self.reason is not None:
            raise ValueError("materialized admission outcomes cannot carry a reason")
        if self.disposition is not AdmissionDisposition.MATERIALIZED and self.reason is None:
            raise ValueError("unknown/refusal admission outcomes require a typed reason")
        return self


class AdmissionOutcomeCollection(Protocol):
    """Replayable outcomes with a count and a bounded iteration interface."""

    accounting_id: str

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[AdmissionOutcome]: ...

    def iter_unit(self, unit: str) -> Iterator[dict[str, object]]: ...

    def count_by_unit(self) -> dict[str, int]: ...

    def overlaps(self, unit: str, start: int, end: int) -> bool: ...


def _ordinal_in_ranges(ranges: Sequence[tuple[int, int]], ordinal: int) -> bool:
    """Membership test over sorted, disjoint half-open ordinal ranges."""
    index = bisect_right(ranges, (ordinal, math.inf))
    if index == 0:
        return False
    start, end = ranges[index - 1]
    return start <= ordinal < end


class ParseAccounting(BaseModel):
    """Closed admission ledger attached to a parsed session.

    ``expected`` is the denominator observed by the parser before lowering;
    every denominator member has exactly one terminal result. The writer
    validates this algebra before it mutates index state.

    A materialized outcome carries no reason and no evidence beyond its own
    ordinal, so the overwhelmingly common disposition is held as compact
    half-open ordinal ranges in ``materialized_ordinals`` rather than as one
    model instance per input record (polylogue-ro922: an untrusted session of
    200k outer records otherwise costs hundreds of megabytes of resident
    Pydantic objects for a proof that is one interval). ``outcomes`` holds the
    dispositions that do carry evidence -- typed unknowns and typed refusals.
    A caller that builds the accounting directly may still put materialized
    outcomes in ``outcomes``; both representations count toward the same
    conserved denominator and must not name the same ordinal twice.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    expected: dict[AdmissionUnit, int] = Field(default_factory=dict)
    outcomes: SkipValidation[list[AdmissionOutcome] | AdmissionOutcomeCollection] = Field(default_factory=list)
    materialized_ordinals: dict[AdmissionUnit, list[tuple[int, int]]] = Field(default_factory=dict)

    def stable_binding_digest(self) -> str:
        """Hash the complete accounting witness without expanding materialized ranges.

        The digest deliberately excludes the preparation database path and
        scratch ``accounting_id``. Spilled exceptional outcomes are consumed
        one row at a time; compact materialized ranges remain compact.
        """
        digest = hashlib.sha256(b"polylogue.parse-accounting-binding.v1\0")

        def add(value: object) -> None:
            encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

        expected = sorted((unit.value, int(count)) for unit, count in self.expected.items())
        add({"expected": expected})
        for unit in sorted(self.materialized_ordinals, key=lambda item: item.value):
            values = self.materialized_ordinals[unit]
            add({"materialized_unit": unit.value, "range_count": len(values)})
            for start, end in values:
                add([int(start), int(end)])
        outcomes = self.outcomes
        if isinstance(outcomes, list):
            # Outcome order is not semantic; replay addresses them by unit and
            # ordinal. Sorting the existing resident list makes inline and
            # spilled representations share one identity.
            for outcome in sorted(outcomes, key=lambda item: (item.unit.value, item.ordinal)):
                add(outcome.model_dump(mode="json"))
        else:
            # Spilled storage is ordered by unit and ordinal and does not expose
            # its random scratch identity as part of the witness.
            units = {item.value for item in self.expected} | {item.value for item in self.materialized_ordinals}
            units.update(outcomes.count_by_unit())
            for unit_name in sorted(units):
                for outcome_record in outcomes.iter_unit(unit_name):
                    add(outcome_record)
        return digest.hexdigest()

    def to_prepared_payload(self) -> dict[str, object]:
        if not isinstance(self.outcomes, list):
            outcomes: object = {
                "$polylogue_spilled_outcomes": self.outcomes.accounting_id,
                "count": len(self.outcomes),
            }
        else:
            outcomes = [item.model_dump(mode="json") for item in self.outcomes]
        return {
            "expected": {unit.value: count for unit, count in self.expected.items()},
            "outcomes": outcomes,
            "materialized_ordinals": {
                unit.value: [[start, end] for start, end in ranges]
                for unit, ranges in self.materialized_ordinals.items()
            },
        }

    @classmethod
    def from_prepared_payload(cls, value: object, path: Path) -> ParseAccounting:
        from polylogue.sources.parse_accounting_spool import SpilledParseAccountingOutcomes

        if not isinstance(value, Mapping):
            raise ValueError("prepared parse accounting must be an object")
        raw_outcomes = value.get("outcomes", [])
        if isinstance(raw_outcomes, Mapping) and isinstance(raw_outcomes.get("$polylogue_spilled_outcomes"), str):
            accounting_id = str(raw_outcomes["$polylogue_spilled_outcomes"])
            raw_outcomes = SpilledParseAccountingOutcomes.from_prepared_reference(
                path, accounting_id, int(raw_outcomes.get("count", -1))
            )
        return cls(
            expected=value.get("expected", {}),
            outcomes=raw_outcomes,
            materialized_ordinals=value.get("materialized_ordinals", {}),
        )

    @field_validator("outcomes", mode="before")
    @classmethod
    def validate_outcomes(cls, value: object) -> object:
        from polylogue.sources.parse_accounting_spool import SpilledParseAccountingOutcomes

        if isinstance(value, SpilledParseAccountingOutcomes):
            return value
        if value is None:
            return []
        if isinstance(value, (str, bytes, Mapping)):
            raise ValueError("parse accounting outcomes require an iterable of outcome records")
        if not isinstance(value, Iterable):
            raise ValueError("parse accounting outcomes require an iterable of outcome records")
        return [item if isinstance(item, AdmissionOutcome) else AdmissionOutcome.model_validate(item) for item in value]

    def iter_outcomes(self) -> Iterator[AdmissionOutcome]:
        """Yield every terminal outcome, materialized ranges expanded, in ordinal order per unit."""
        outcomes = self.outcomes
        if not isinstance(outcomes, list):
            units = list(dict.fromkeys((*self.expected, *self.materialized_ordinals)))
            for unit in units:
                spilled_exceptional = (AdmissionOutcome.model_validate(item) for item in outcomes.iter_unit(unit.value))
                spilled_ranges = _materialized_outcomes(unit, self.materialized_ordinals.get(unit, ()))
                yield from heapq.merge(spilled_exceptional, spilled_ranges, key=lambda item: item.ordinal)
            return
        by_unit: dict[AdmissionUnit, list[AdmissionOutcome]] = {}
        for outcome in outcomes:
            by_unit.setdefault(outcome.unit, []).append(outcome)
        units = list(dict.fromkeys((*self.materialized_ordinals, *by_unit)))
        for unit in units:
            exceptional = sorted(by_unit.get(unit, ()), key=lambda item: item.ordinal)
            ranges = _materialized_outcomes(unit, self.materialized_ordinals.get(unit, ()))
            yield from heapq.merge(iter(exceptional), ranges, key=lambda item: item.ordinal)

    def assert_conserved(self) -> None:
        expected = {unit: int(count) for unit, count in self.expected.items()}
        if any(count < 0 for count in expected.values()):
            raise ValueError("admission denominators cannot be negative")
        if not isinstance(self.outcomes, list):
            spilled_actual = self.outcomes.count_by_unit()
            for unit, raw_ranges in self.materialized_ordinals.items():
                ordered = sorted((int(start), int(end)) for start, end in raw_ranges)
                previous_end = 0
                for start, end in ordered:
                    if end <= start or start < 0 or end > expected.get(unit, 0) or start < previous_end:
                        raise ValueError(f"invalid materialized admission range for {unit.value}: [{start}, {end})")
                    if self.outcomes.overlaps(unit.value, start, end):
                        raise ValueError(f"duplicate admission outcome for {unit.value}[{start}]")
                    previous_end = end
                    spilled_actual[unit.value] = spilled_actual.get(unit.value, 0) + end - start
            normalized_expected = {unit.value: count for unit, count in expected.items() if count}
            if spilled_actual != normalized_expected:
                raise ValueError(
                    f"admission denominator mismatch: expected={normalized_expected}, actual={spilled_actual}"
                )
            return
        actual: dict[AdmissionUnit, int] = {}
        # Ranges are validated without enumerating their members: sorted,
        # non-empty, and pairwise disjoint proves the same no-duplicate
        # property the explicit set below proves for evidence-bearing
        # outcomes, at a cost that does not grow with the record count.
        ranges: dict[AdmissionUnit, list[tuple[int, int]]] = {}
        for unit, raw_ranges in self.materialized_ordinals.items():
            ordered = sorted((int(start), int(end)) for start, end in raw_ranges)
            previous_end = 0
            for start, end in ordered:
                if end <= start or start < 0 or end > expected.get(unit, 0):
                    raise ValueError(f"invalid materialized admission range for {unit.value}: [{start}, {end})")
                if start < previous_end:
                    raise ValueError(f"duplicate admission outcome for {unit.value}[{start}]")
                previous_end = end
                actual[unit] = actual.get(unit, 0) + (end - start)
            if ordered:
                ranges[unit] = ordered
        seen: set[tuple[AdmissionUnit, int]] = set()
        for outcome in self.outcomes:
            if not 0 <= outcome.ordinal < expected.get(outcome.unit, 0):
                raise ValueError(f"admission ordinal outside denominator: {outcome.unit.value}[{outcome.ordinal}]")
            identity = (outcome.unit, outcome.ordinal)
            if identity in seen or _ordinal_in_ranges(ranges.get(outcome.unit, ()), outcome.ordinal):
                raise ValueError(f"duplicate admission outcome for {outcome.unit.value}[{outcome.ordinal}]")
            seen.add(identity)
            actual[outcome.unit] = actual.get(outcome.unit, 0) + 1
        if actual != {unit: count for unit, count in expected.items() if count}:
            raise ValueError(f"admission denominator mismatch: expected={expected}, actual={actual}")


def _materialized_outcomes(unit: AdmissionUnit, ranges: Sequence[tuple[int, int]]) -> Iterator[AdmissionOutcome]:
    for start, end in ranges:
        for ordinal in range(start, end):
            yield AdmissionOutcome(
                unit=unit,
                ordinal=ordinal,
                key=str(ordinal),
                disposition=AdmissionDisposition.MATERIALIZED,
            )


class ParsedWebConstruct(BaseModel):
    """Typed construct projected from rich web UI/export payloads.

    Raw provider JSON remains in source evidence. This model carries the
    normalized fields that are useful for archive reads, search, and later
    provider-specific projections without reintroducing provider_meta bags.
    """

    construct_type: WebConstructType
    provider_key: str | None = None
    title: str | None = None
    url: str | None = None
    text: str | None = None
    source_id: str | None = None
    group_id: str | None = None
    group_title: str | None = None
    query: str | None = None
    asset_pointer: str | None = None
    mime_type: str | None = None
    status: str | None = None
    task_id: str | None = None
    task_type: str | None = None
    rank: int | None = None
    start_index: int | None = None
    end_index: int | None = None

    @field_validator("construct_type", mode="before")
    @classmethod
    def coerce_construct_type(cls, v: object) -> WebConstructType:
        if isinstance(v, WebConstructType):
            return v
        return WebConstructType(str(v).strip().lower())


class ParsedFileEdit(BaseModel):
    """File-edit tool-call evidence (Claude Code Edit/Write toolUseResult).

    polylogue-2qx.4 / polylogue-cgfy: structuredPatch (105,123 occurrences),
    originalFile (92,313), oldString/newString/replaceAll/filePath are
    entirely discarded today. Attach this to the TOOL_RESULT block that
    reports the edit outcome (where the wire carries these fields);
    materialization resolves the paired tool_use block id via ``tool_id``
    and writes one ``file_edits`` row keyed by that id.
    """

    file_path: str | None = None
    # Raw structured-patch hunks as the provider emits them (list of
    # {oldStart, oldLines, newStart, newLines, lines} dicts) -- stored
    # verbatim as JSON, not decomposed into columns.
    structured_patch: list[Mapping[str, object]] | None = None
    original_file: str | None = None
    old_string: str | None = None
    new_string: str | None = None
    replace_all: bool | None = None
    user_modified: bool | None = None

    @field_validator("structured_patch", mode="before")
    @classmethod
    def validate_structured_patch_keys(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_require_string_mapping_keys(patch, field="structured_patch") for patch in value]
        return value


class ParsedContentBlock(BaseModel):
    """A single structured content block within a parsed message.

    Block types:
    - text: regular text content
    - thinking: extended reasoning traces
    - tool_use: tool invocation (tool_name, tool_id, tool_input required)
    - tool_result: tool response (tool_id, text required)
    - image: image reference (media_type, metadata for asset pointer)
    - code: code block, language-detected (text required)
    - document: document reference
    """

    type: BlockType
    # Immutable semantic identity captured before cross-block outcome
    # association. This lowering carrier is not part of Source content.
    source_content_identity: str | None = Field(default=None, exclude=True, repr=False, pattern=r"^[0-9a-f]{64}$")
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: Mapping[str, object] | None = None
    media_type: str | None = None
    metadata: dict[str, object] | None = None
    # polylogue-vf9x: provider-issued cryptographic attestation for a THINKING
    # block (Claude's extended-thinking `signature`; Gemini's
    # `thoughtSignatures` are the same construct under a different name).
    # Since roughly 2026-06 Anthropic ships thinking blocks with an EMPTY
    # `thinking` body plus this signature only -- text is genuinely absent
    # from the wire, not merely unparsed. Recorded so the fact that the model
    # reasoned survives even when no text does; deliberately excluded from
    # `_block_content_hash`/lineage prefix signatures (write.py) because the
    # provider re-signs on every replay, so including it would break
    # citation-anchor and fork-prefix matching for otherwise-identical
    # replayed content.
    signature: str | None = None
    # Canonical tool-result outcome, resolved from source structure by the
    # archive writer. It is never inferred from output text.
    is_error: bool | None = None
    exit_code: int | None = None
    tool_outcome: ToolOutcome | None = None
    # Why `is_error` is None on a tool_result block -- see
    # ``core.enums.ToolResultUnknownReason``. Required whenever a tool_result
    # carries no structural outcome; construction refuses without it.
    outcome_unknown_reason: str | None = None
    # polylogue-2qx.4 / polylogue-cgfy: file-edit evidence, attached to the
    # TOOL_RESULT block carrying the provider's edit outcome fields.
    file_edit: ParsedFileEdit | None = None
    web_constructs: list[ParsedWebConstruct] = Field(default_factory=list)

    @field_validator("tool_input", "metadata", mode="before")
    @classmethod
    def validate_mapping_keys(cls, value: object, info: ValidationInfo) -> object:
        return _require_string_mapping_keys(value, field=info.field_name or "content block mapping")

    @model_validator(mode="after")
    def validate_tool_result_outcome(self) -> ParsedContentBlock:
        """Keep known outcomes and unknown reasons mutually exclusive."""
        if self.type is not BlockType.TOOL_RESULT:
            return self
        if self.is_error is None and self.exit_code is not None:
            self.is_error = self.exit_code != 0
        if self.outcome_unknown_reason is not None:
            try:
                self.outcome_unknown_reason = ToolResultUnknownReason(self.outcome_unknown_reason).value
            except ValueError as exc:
                raise ValueError("tool-result unknown reason is outside the normalized vocabulary") from exc
        if self.tool_outcome is ToolOutcome.UNKNOWN and self.outcome_unknown_reason is None:
            raise ValueError("unknown tool-result outcomes must carry one ToolResultUnknownReason")
        if (
            self.tool_outcome in (ToolOutcome.OK, ToolOutcome.ERROR, ToolOutcome.NO_RESULT)
            and self.outcome_unknown_reason is not None
        ):
            raise ValueError("known tool-result outcomes cannot carry an unknown reason")
        if self.tool_outcome is ToolOutcome.UNKNOWN and self.is_error is not None:
            raise ValueError("unknown tool-result outcomes cannot carry is_error")
        if self.is_error is None and self.outcome_unknown_reason is None:
            # Fail closed at the producer boundary: a reason assigned here
            # would be one no parser derived from the record, which is exactly
            # what the unknown-reason contract exists to exclude. The parser
            # reads its own construct and states why.
            raise ValueError(
                "tool_result without a structural outcome must carry one "
                "ToolResultUnknownReason derived from its own construct"
            )
        if self.is_error is not None and self.outcome_unknown_reason is not None:
            raise ValueError("known tool-result outcomes cannot carry an unknown reason")
        return self

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: object) -> BlockType:
        return BlockType.from_string(str(v))


#: Validation context flag for data parsed from JSON text by a parser other
#: than pydantic's (the prepared sink reads surrogate escapes with the
#: stdlib): fields rendered differently in JSON mode parse as JSON mode would.
SINK_JSON_CONTEXT_KEY = "polylogue.json_sourced"
SINK_JSON_CONTEXT: dict[str, object] = {SINK_JSON_CONTEXT_KEY: True}


class ParsedPasteEvidence(BaseModel):
    position: int = 0
    start_offset: int | None = None
    end_offset: int | None = None
    boundary_state: str = "hash_only"
    source_event_id: str | None = None
    source_marker: str | None = None
    content_hash: bytes | None = None
    observed_at_ms: int | None = None

    @field_validator("content_hash", mode="before")
    @classmethod
    def _parse_content_hash(cls, value: object, info: ValidationInfo) -> object:
        # JSON mode receives the hex representation emitted below; Python
        # callers continue to supply the raw digest bytes.
        json_sourced = info.mode == "json" or bool(info.context and info.context.get(SINK_JSON_CONTEXT_KEY))
        if json_sourced and isinstance(value, str):
            return bytes.fromhex(value)
        return value

    @field_serializer("content_hash", when_used="json")
    def _serialize_content_hash(self, value: bytes | None) -> str | None:
        """Render the digest as hex in JSON mode.

        ``content_hash`` is a raw SHA-256 digest, and pydantic's JSON
        serializer decodes ``bytes`` as UTF-8 -- which a digest is not. The
        semantic hash payload (``pipeline/ids.py``'s ``_hash_field_value``)
        dumps every paste span in JSON mode, so a message carrying a
        content-bearing paste raised ``UnicodeDecodeError`` at the write
        boundary instead of hashing (polylogue-ximhz: reached once the
        retained ``history.jsonl`` evidence made that span reachable on the
        canonical ingest route). Hex is the same digest, stably encoded.
        """
        return value.hex() if value is not None else None


# 2100-01-01T00:00:00Z. A declared absolute ceiling, not a window around the
# current clock: parser validation must stay deterministic and clock-free, and
# every realistic corruption of a millisecond timestamp (a microsecond or
# nanosecond value read as milliseconds, a sentinel like 2**53) lands tens of
# thousands of years past it. ``occurred_at_ms`` becomes ``sessions.updated_at_ms``
# when an export carries no session-level timestamp, so one far-future record
# would otherwise pin that session at the top of freshness ordering forever and
# defeat staleness-driven reconvergence. Refuse the record rather than clamp it:
# a clamp is a silent rewrite of provider evidence.
IMPLAUSIBLE_OCCURRED_AT_MS_CEILING = 4102444800000


def _require_plausible_occurred_at_ms(value: int | None) -> int | None:
    if value is not None and value > IMPLAUSIBLE_OCCURRED_AT_MS_CEILING:
        raise ValueError(
            f"occurred_at_ms {value} is past the declared plausibility ceiling "
            f"{IMPLAUSIBLE_OCCURRED_AT_MS_CEILING} (2100-01-01T00:00:00Z)"
        )
    return value


class ParsedMessage(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider_message_id: str
    role: Role
    text: str | None = None
    timestamp: str | None = None
    occurred_at_ms: int | None = None
    blocks: list[ParsedContentBlock] = Field(default_factory=list)
    message_type: MessageType = MessageType.MESSAGE
    material_origin: MaterialOrigin = MaterialOrigin.UNKNOWN
    parent_message_provider_id: str | None = None
    # Parser-local linkage for a parent that has no provider-issued id. The
    # writer resolves this coordinate against this parsed batch, then stores
    # only the resulting archive message id. It is never provider identity or
    # persisted parser data.
    parent_message_position: int | None = Field(default=None, exclude=True, repr=False)
    # Private parser-to-writer evidence. This never becomes provider identity
    # or a public message id.
    owner_coordinate: MessageOwnerCoordinate | None = Field(default=None, exclude=True, repr=False)
    position: int | None = None
    branch_index: int = 0
    variant_index: int | None = None
    is_active_path: bool | None = None
    is_active_leaf: bool | None = None
    # Private lowering evidence: ``is_active_leaf`` above is the storage
    # default (the last message) because the producer marked no single leaf.
    # It is never provider evidence, so a later lowering pass does not read
    # it as a producer leaf and infer an active path from it. It qualifies a
    # marked leaf only; on any other message it means nothing.
    active_leaf_fallback: bool = Field(default=False, exclude=True, repr=False)
    # Token usage flows through from provider raw records to MaterializedMessage.
    # Parsers populate when the raw record carries usage info; otherwise None.
    # Materialization writes these into the messages table, where they drive
    # cost estimation downstream. Were previously dropped on the parser floor,
    # leaving 2.5M rows with input_tokens=output_tokens=0 and dead cost
    # rollups across the entire archive.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    model_name: str | None = None
    model_effort: str | None = None
    duration_ms: int | None = None
    sender_name: str | None = None
    recipient: str | None = None
    delivery_status: str | None = None
    end_turn: bool | None = None
    user_context_text: str | None = None
    paste_spans: list[ParsedPasteEvidence] = Field(default_factory=list)
    # polylogue-2qx.4 / polylogue-cuxz.8: the provider's own terminal-state
    # signal for this turn (Claude ``message.stop_reason``, 608,608
    # occurrences on the wire). None when the provider did not report one or
    # this is not an assistant turn -- never a guess.
    stop_reason: str | None = None
    # Claude Code's explicit producer outcome for a generation cut short
    # before completion. The durable session event carries this even on
    # records that have no message body.
    is_aborted_mid_stream: bool = False

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> Role:
        if isinstance(v, Role):
            return v
        return Role.normalize(str(v) if v is not None else "unknown")

    @field_validator("message_type", mode="before")
    @classmethod
    def coerce_message_type(cls, v: object) -> MessageType:
        return MessageType.normalize(v)

    @field_validator("material_origin", mode="before")
    @classmethod
    def coerce_material_origin(cls, v: object) -> MaterialOrigin:
        return MaterialOrigin.normalize(v)

    @field_validator("occurred_at_ms", "parent_message_position", "position", "variant_index", "duration_ms")
    @classmethod
    def non_negative_optional_int(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("parser contract integer fields cannot be negative")
        return value

    @field_validator("occurred_at_ms")
    @classmethod
    def plausible_occurred_at_ms(cls, value: int | None) -> int | None:
        return _require_plausible_occurred_at_ms(value)

    @model_validator(mode="after")
    def derive_occurred_at_ms(self) -> ParsedMessage:
        if self.occurred_at_ms is None and self.timestamp:
            parsed = parse_timestamp(self.timestamp)
            if parsed is not None:
                self.occurred_at_ms = _require_plausible_occurred_at_ms(int(parsed.timestamp() * 1000))
        # Authoredness and message type must classify the same TEXT projection,
        # never a context-looking marker contributed only by a THINKING block.
        classification_text = self.text
        if self.blocks:
            classification_text = (
                "\n".join(block.text for block in self.blocks if block.type is BlockType.TEXT and block.text) or None
            )
        if self.message_type is MessageType.MESSAGE:
            from polylogue.archive.message.artifacts import classify_message_type

            self.message_type = classify_message_type(
                role=self.role,
                message_type=self.message_type,
                text=classification_text,
                block_types=tuple(block.type for block in self.blocks),
            )
        if self.material_origin is MaterialOrigin.UNKNOWN:
            from polylogue.archive.message.artifacts import classify_material_origin

            self.material_origin = classify_material_origin(
                role=self.role,
                message_type=self.message_type,
                text=classification_text,
                block_types=tuple(block.type for block in self.blocks),
            )
        return self


class ParsedAttachment(BaseModel):
    """Parsed attachment shape with first-class native identifiers (#1252).

    Native identifiers used for lookups — `provider_attachment_id`,
    `provider_file_id`, `provider_drive_id` — and the origin classification
    `upload_origin` are typed top-level fields. Downstream storage promotes
    them into stored columns so attachment lookups never JSON-extract on the
    hot path. See `polylogue/storage/sqlite/archive_tiers/index.py:attachments`.

    `upload_origin` is a closed vocabulary ({"drive","paste","url","oauth"}
    or None); the attachment-library UI (#1199) groups by `(source_name,
    upload_origin)` without scanning JSON.

    `attachment_kind` classifies non-downloadable attachment shapes
    (`"inline_file"`, `"youtube_video"`); the Drive download path skips
    acquisition for those kinds.
    """

    provider_attachment_id: str
    message_provider_id: str | None = None
    # Transport-only linkage for id-less source messages. This is parser-local
    # bookkeeping, never a provider identity or stored attachment column.
    message_position: int | None = Field(default=None, exclude=True, repr=False)
    message_variant_index: int | None = Field(default=None, exclude=True, repr=False)
    # Full private owner evidence. ``message_position`` and
    # ``message_variant_index`` remain as parser compatibility fields, while
    # hash/write/repair paths consume this typed coordinate.
    owner_coordinate: MessageOwnerCoordinate | None = Field(default=None, exclude=True, repr=False)
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_file_id: str | None = None
    provider_drive_id: str | None = None
    upload_origin: AttachmentUploadOrigin | None = None
    direction: AttachmentDirection | None = None
    producer_ref: str | None = None
    attachment_kind: str | None = None
    source_url: str | None = None
    caption: str | None = None
    # Transport-only (#2468): raw bytes already present in the source export (e.g.
    # Gemini inline base64). When set, ingestion writes them to the content-
    # addressed blob store and records the true SHA-256 + 'acquired' status instead
    # of fabricating a hash. Excluded from serialization/repr; not a stored field.
    inline_bytes: bytes | None = Field(default=None, exclude=True, repr=False)
    # Transport-only (polylogue-8ac0): a (blob_hash_hex, size_bytes) pair for
    # bytes ALREADY streamed into the content-addressed blob store during
    # sidecar discovery (e.g. ChatGPT ``.dat`` asset acquisition -- see
    # ``sources/assembly_chatgpt.py``). Distinct from `inline_bytes`: those
    # bytes still need hashing/writing at ingest time; this records a write
    # that already happened, so ingestion must record it without re-hashing.
    # Excluded from serialization/repr; not a stored field.
    precomputed_blob: tuple[str, int] | None = Field(default=None, exclude=True, repr=False)
    prepared_carrier_key: tuple[str, int, int] | None = Field(default=None, exclude=True, repr=False)
    # Shallow value copies (including chunk-coordinate rebasing) retain the
    # acquisition identity, without publishing process-local object identity.
    _acquisition_origin: object | None = PrivateAttr(default=None)

    @property
    def acquisition_key(self) -> object:
        if self.prepared_carrier_key is not None:
            return self.prepared_carrier_key
        return id(self._acquisition_origin if self._acquisition_origin is not None else self)

    @field_validator("path")
    @classmethod
    def sanitize_path(cls, v: str | None) -> str | None:
        """Sanitize path to prevent traversal attacks and other security issues."""
        return _sanitize_path_helper(v)

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str | None) -> str | None:
        """Sanitize filename to prevent control chars and invalid names."""
        if v is None:
            return v

        v = v.replace("\x00", "")
        v = "".join(c for c in v if ord(c) >= 32 and ord(c) != 127)

        if v and v.strip(".") == "":
            v = "file"

        return v if v else None


class ParsedSessionRef(BaseModel):
    """Tracker-agnostic external reference observed in a session.

    polylogue-2qx.4 / polylogue-cgfy: Claude Code's pr-link (20,702
    occurrences on the wire) generalized so a future issue-tracker reference
    lands in the same relation. ``kind`` is ``core.enums.SessionRefKind``.
    """

    kind: str
    url: str
    repo: str | None = None
    number: int | None = None


class ParsedSessionEvent(BaseModel):
    """Non-message semantic artifact in the session timeline."""

    event_type: str  # "compaction", "turn_context", etc.
    timestamp: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    source_message_provider_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_message_provider_id", "source_message_id"),
    )
    owner_coordinate: MessageOwnerCoordinate | None = Field(default=None, exclude=True, repr=False)
    boundary_start_position: int | None = None
    boundary_end_position: int | None = None
    boundary_message_position: int | None = Field(default=None, exclude=True)

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload_keys(cls, value: object) -> object:
        return _require_string_mapping_keys(value, field="session event payload")


class ParsedDispatchObservation(BaseModel):
    """Provider evidence for one parent-side dispatch."""

    provider_tool_id: str
    child_provider_id: str | None = None
    child_identity_namespace: str = "provider-session"
    observation_kind: Literal["parent_dispatch"] = "parent_dispatch"
    # Display metadata the dispatching side recorded about the child
    # (Claude Code ``toolUseResult.agentType`` / ``description``). Never an
    # identity input to the resolver.
    agent_type: str | None = None
    description: str | None = None
    first_seen: str | None = None
    last_seen: str | None = None
    resolution_reason: (
        Literal[
            "target-not-yet-observed",
            "source-unavailable",
            "target-refused-non-session",
            "identity-contradiction",
            "cycle-quarantine",
            "unknown",
        ]
        | None
    ) = None


def upgrade_chat_export_user_authorship(provider: Provider, message: ParsedMessage) -> ParsedMessage:
    """Apply the session-level user-channel guarantee to one parsed message."""
    from polylogue.core.sources import provider_to_source

    if provider_to_source(provider).runtime_root is None and not any(
        block.type is BlockType.DOCUMENT for block in message.blocks
    ):
        from .base_support import human_authored_override

        message.material_origin = human_authored_override(message.role, message.message_type, message.material_origin)
    return message


class ParsedSession(BaseModel):
    source_name: Provider
    provider_session_id: str
    title: str | None = None
    session_kind: SessionKind = SessionKind.STANDARD
    created_at: str | None = None
    updated_at: str | None = None
    # Internal authority carrier. These fields are excluded from serialized
    # payloads/content hashes but survive model_copy/pipeline hand-offs, so a
    # normalization pass cannot later mistake its own derived value for
    # producer evidence.
    created_at_provenance: str = Field(default="unknown", exclude=True, repr=False)
    updated_at_provenance: str = Field(default="unknown", exclude=True, repr=False)
    # Parse-side identity carrier.  The ingest worker binds this after all
    # parser and timestamp normalization is complete so downstream writers can
    # validate and publish the source-bound digest without hashing the tree a
    # second time.  It is excluded from payloads and semantic identity.
    content_hash: str | None = Field(default=None, exclude=True, repr=False)
    # Enrichment carrier: the digest of the retained evidence this session was
    # enriched from (``session_enrichment_evidence_key``), stamped where the
    # enrichment ran. The writer binds it only when the archive still holds
    # that evidence, so a session enriched before its index or thread state
    # arrived is re-derived rather than certified. Not session content.
    enrichment_evidence_key: str | None = Field(default=None, exclude=True, repr=False)
    messages: list[ParsedMessage]
    # Parser-only admission proof. It is excluded from serialized payloads and
    # content hashes, but the storage writer validates it before lowering.
    unit_accounting: ParseAccounting | None = Field(default=None, exclude=True, repr=False)
    active_leaf_message_provider_id: str | None = None
    attachments: list[ParsedAttachment] = Field(default_factory=list)

    @field_serializer("attachments")
    def serialize_attachments(
        self, value: list[ParsedAttachment], info: FieldSerializationInfo
    ) -> list[dict[str, object]]:
        return [attachment.model_dump(mode=info.mode) for attachment in value]

    session_events: list[ParsedSessionEvent] = Field(default_factory=list)
    parent_session_provider_id: str | None = None
    # The parent-session message a provider record names as this session's
    # divergence point (Claude Code ``fork-context-ref.parentLastUuid``). A
    # provider-native id in the PARENT's namespace, so it is only meaningful
    # alongside ``parent_session_provider_id``; the archive writer binds it to
    # ``session_links.branch_point_message_id`` once that parent message
    # exists. Distinct from the branch point the writer derives by aligning a
    # physically replayed prefix: this one is asserted, and is the only
    # evidence available when the child does not replay the parent at all.
    branch_point_provider_message_id: str | None = None
    # Exact provider-native names that can refer to this emitted session.
    # These are parser evidence, never prefix/suffix guesses.
    provider_session_aliases: list[str] = Field(default_factory=list, exclude=True, repr=False)
    branch_type: BranchType | None = None
    title_source: TitleSource | None = None
    # Specific provenance beyond title_source's coarse strategy label: which
    # exact evidence row won, and a 0..1 confidence signal for it
    # (polylogue-ih67). Optional -- most parsers leave both None and only
    # title_source is set; assemblies that resolve title from a specific
    # dated row (Codex thread name / history / state db / message) set both.
    title_ref: str | None = None
    instructions_text: str | None = None
    reported_duration_ms: int | None = None
    reported_cost_usd: float | None = None
    models_used: list[str] = Field(default_factory=list)
    # Universal session-context semantics graduated out of provider metadata.
    working_directories: list[str] = Field(default_factory=list)
    git_branch: str | None = None
    git_repository_url: str | None = None
    # Provider workspace/project grouping (ChatGPT "project": the g-p-<id> token,
    # surfaced in the backend payload as gizmo_id/conversation_template_id). Lets
    # web sessions be grouped/enumerated by project and upgrades the canonical URL
    # to the project-scoped form. None for sessions with no project.
    provider_project_ref: str | None = None
    # Claude Code's team/campaign scope, stamped on every record in the file.
    team_name: str | None = None
    # Specific commit the agent session was anchored to (codex records this
    # per-session in their meta.git.commit_hash). Lets downstream attribution
    # pin a session to an exact commit instead of the looser "session_date
    # +/- N hours" window. Empty string treated as None.
    git_commit_hash: str | None = None
    # Parser-level ingest flags that the storage layer persists as auto-tags
    # (tag_source='auto', method='parser') during write_parsed_session_to_archive.
    # Parsers set these to communicate structural quality issues without requiring
    # new storage columns — the existing session_tags table (index.db) absorbs them.
    # Example: ["degraded:brain-metadata-fragment"] for Antigravity brain-artifact
    # fallback sessions that fragment one work session into N single-message sessions.
    ingest_flags: list[str] = Field(default_factory=list)
    # polylogue-2qx.4 / polylogue-cgfy: subagent/agent display name behind an
    # opaque native id (Claude Code Task-tool slug, e.g.
    # "greedy-squishing-hamming") -- distinct from ``title``, which is the
    # session's own resolved content title.
    display_name: str | None = None
    # polylogue-o4j2: non-blank chunkedPrompt.pendingInputs entries (unsent
    # AI Studio textbox drafts), each {"text": ..., "role": ..., optionally
    # "token_count": ...}. Deliberately NOT a session_event: a draft is
    # mutable current UI state -- edited in place, gone entirely once
    # submitted -- and session_events participate in
    # session_revision_projection's append-only comparison axes
    # (polylogue-aggz Invariant 1). Stored verbatim as a JSON column --
    # see sessions.pending_drafts_json.
    pending_drafts: list[dict[str, object]] = Field(default_factory=list)
    # polylogue-2qx.4 / polylogue-cgfy: tracker-agnostic external references
    # (pr-link today, issue refs generalize to the same relation).
    session_refs: list[ParsedSessionRef] = Field(default_factory=list)

    @field_validator("pending_drafts", mode="before")
    @classmethod
    def validate_pending_draft_keys(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return [_require_string_mapping_keys(draft, field="pending_drafts") for draft in value]
        return value

    @field_validator("source_name", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider:
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v) if v is not None else "unknown")

    @field_validator("session_kind", mode="before")
    @classmethod
    def coerce_session_kind(cls, v: object) -> SessionKind:
        return SessionKind.normalize(v)

    @field_validator("reported_cost_usd")
    @classmethod
    def non_negative_optional_float(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value < 0:
            raise ValueError("reported_cost_usd cannot be negative")
        if not math.isfinite(value):
            # ``"Infinity"``/``"1e309"``/``"NaN"`` are valid JSON strings that
            # ``float()`` accepts, and a provider field carrying one reaches
            # here unchanged. A non-finite reported cost is not a measurement:
            # ``inf`` and ``nan`` are absorbing, so admitting one poisons every
            # total it is summed into and no reader can tell it from a real
            # figure. Refusing the session is the observable outcome; silently
            # substituting 0.0 would report an unmeasured cost as a real one.
            raise ValueError("reported_cost_usd must be finite")
        return value

    @model_validator(mode="after")
    def _upgrade_chat_export_user_authorship(self) -> ParsedSession:
        """Chat exports have a clean human user channel; agent runtimes do not.

        `classify_material_origin` defaults role=user MESSAGE rows to UNKNOWN
        because agent-runtime user channels (Claude Code, Codex) carry tool
        results, hook injections, and pasted corpora — text shape is not proof a
        human typed it. But for turn-based chat exports (ChatGPT, Claude.ai,
        Gemini/AI-Studio, Grok — sources with no on-disk runtime_root, delivered
        as export ZIPs/Drive) the user channel IS a structural human-input
        guarantee. Without this the authored-user metrics read ~0 for chat
        exports after re-ingest. Provider context only exists at the session
        level, so the positive upgrade is applied here, not in the per-message
        classifier.
        """
        for message in self.messages:
            upgrade_chat_export_user_authorship(self.source_name, message)
        return self


class RawSessionData(BaseModel):
    """One raw representation with captured acquisition metadata.

    Preparation passes a private sealed file to the creator; publication
    replaces it with the blob hash. Already retained blobs carry no byte copy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_bytes: bytes = b""
    staged_payload: StagedRawPayload | None = Field(default=None, exclude=True, repr=False)
    source_path: str
    # Frozen by acquisition; publication never resolves a mutable source alias.
    canonical_source_path: str | None = None
    captured_profile_key: str | None = None
    captured_profile_source_path: str | None = Field(default=None, exclude=True)
    captured_file_observation: tuple[int, int, int, int, int] | None = Field(default=None, exclude=True)
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = Field(default=None, exclude=True)
    source_index: int | None = None
    # The address kind this payload was acquired under. ``source_index`` is a
    # position inside a container member and cannot express "the member
    # document itself"; ``None`` means the acquiring route is not a container
    # member at all.
    addressing_mode: MemberAddressingMode | None = None
    # Structural value identity for container members.  This is the replay
    # authority; source_index remains only a coordinate hint.
    content_identity: str | None = None
    file_mtime: str | None = None
    provider_hint: Provider | None = None
    blob_hash: str | None = None
    blob_size: int | None = None
    blob_publication_receipt_id: str | None = Field(default=None, exclude=True)
    # Captured during acquisition; never rediscovered from source_path.
    sidecar_snapshot: dict[str, object] | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def exclusive_raw_representation(self) -> RawSessionData:
        if sum((bool(self.raw_bytes), self.staged_payload is not None, self.blob_hash is not None)) != 1:
            raise ValueError("raw payload bytes, staged file and published blob are mutually exclusive")
        return self

    @field_validator("provider_hint", mode="before")
    @classmethod
    def coerce_provider_hint(cls, v: object) -> Provider | None:
        if v is None:
            return None
        return Provider.from_string(str(v))
