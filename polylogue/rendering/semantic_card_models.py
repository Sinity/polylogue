"""Pure data contracts for provider-neutral semantic transcripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from polylogue.core.json import JSONDocument, is_json_value

CARD_SCHEMA_VERSION: Literal["semantic-card.v1"] = "semantic-card.v1"
TRANSCRIPT_SCHEMA_VERSION: Literal["semantic-transcript.v1"] = "semantic-transcript.v1"


class SemanticCardKind(str, Enum):
    """Closed card vocabulary shared by CLI and web payloads."""

    SHELL = "shell"
    FILE_READ = "file_read"
    FILE_EDIT = "file_edit"
    SEARCH = "search"
    WEB = "web"
    TASK = "task"
    MCP = "mcp"
    LINEAGE = "lineage"
    ATTACHMENT = "attachment"
    FALLBACK = "fallback"


class CardOutcomeState(str, Enum):
    """Outcome established from source structure, not prose."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class PreviewStrategy(str, Enum):
    """How a bounded evidence preview was selected."""

    FULL = "full"
    HEAD_TAIL = "head_tail"
    CHARACTER_BOUNDED = "character_bounded"


class SemanticNoticeKind(str, Enum):
    """Typed compact notices that retain otherwise-empty evidence."""

    EMPTY_THINKING = "empty_thinking"


class LineageAvailability(str, Enum):
    """How much lineage authority the supplied descriptor carries."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class LineageAuthority(str, Enum):
    """Substrate that established the lineage descriptor."""

    ARCHIVE_ENVELOPE = "archive_envelope"
    SESSION_ROW = "session_row"
    TOPOLOGY = "topology"


@dataclass(frozen=True, slots=True)
class SemanticCardSource:
    """Exact source coordinates and message metadata retained by a card."""

    session_id: str
    provider_family: str = "unknown"
    origin: str | None = None
    message_id: str | None = None
    block_id: str | None = None
    block_index: int | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    attachment_id: str | None = None
    material_origin: str | None = None
    occurred_at: str | None = None
    duration_ms: int | None = None
    parent_message_id: str | None = None
    variant_index: int | None = None
    is_active_path: bool | None = None
    is_active_leaf: bool | None = None
    inherited_prefix: bool | None = None
    result_message_id: str | None = None
    result_block_id: str | None = None
    result_block_index: int | None = None
    result_duration_ms: int | None = None
    result_material_origin: str | None = None
    result_inherited_prefix: bool | None = None

    def to_document(self) -> JSONDocument:
        return _without_none(
            {
                "session_id": self.session_id,
                "provider_family": self.provider_family,
                "origin": self.origin,
                "message_id": self.message_id,
                "block_id": self.block_id,
                "block_index": self.block_index,
                "tool_name": self.tool_name,
                "tool_id": self.tool_id,
                "attachment_id": self.attachment_id,
                "material_origin": self.material_origin,
                "occurred_at": self.occurred_at,
                "duration_ms": self.duration_ms,
                "parent_message_id": self.parent_message_id,
                "variant_index": self.variant_index,
                "is_active_path": self.is_active_path,
                "is_active_leaf": self.is_active_leaf,
                "inherited_prefix": self.inherited_prefix,
                "result_message_id": self.result_message_id,
                "result_block_id": self.result_block_id,
                "result_block_index": self.result_block_index,
                "result_duration_ms": self.result_duration_ms,
                "result_material_origin": self.result_material_origin,
                "result_inherited_prefix": self.result_inherited_prefix,
            }
        )


@dataclass(frozen=True, slots=True)
class SemanticCardOutcome:
    """Source-structural outcome; both fields absent means unknown."""

    state: CardOutcomeState
    is_error: bool | None = None
    exit_code: int | None = None

    def to_document(self) -> JSONDocument:
        return _without_none(
            {
                "state": self.state.value,
                "is_error": self.is_error,
                "exit_code": self.exit_code,
            }
        )


@dataclass(frozen=True, slots=True)
class SemanticCardField:
    label: str
    value: str

    def to_document(self) -> JSONDocument:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True, slots=True)
class SemanticCardPreview:
    """Bounded preview that discloses omissions and decoding replacement."""

    kind: str
    text: str
    line_count: int
    omitted_lines: int = 0
    omitted_characters: int = 0
    truncated: bool = False
    strategy: PreviewStrategy = PreviewStrategy.FULL
    encoding_replacements: int = 0

    def to_document(self) -> JSONDocument:
        return {
            "kind": self.kind,
            "text": self.text,
            "line_count": self.line_count,
            "omitted_lines": self.omitted_lines,
            "omitted_characters": self.omitted_characters,
            "truncated": self.truncated,
            "strategy": self.strategy.value,
            "encoding_replacements": self.encoding_replacements,
        }


@dataclass(frozen=True, slots=True)
class SemanticCardRawEvidence:
    """Raw input/result retained when a card cannot safely specialize."""

    tool_input: JSONDocument | None = None
    tool_input_raw: str | None = None
    result_preview: SemanticCardPreview | None = None

    def to_document(self) -> JSONDocument:
        result: JSONDocument = {}
        if self.tool_input is not None:
            result["tool_input"] = self.tool_input
        if self.tool_input_raw is not None:
            result["tool_input_raw"] = self.tool_input_raw
        if self.result_preview is not None:
            result["result_preview"] = self.result_preview.to_document()
        return result


@dataclass(frozen=True, slots=True)
class SemanticCard:
    """Provider-neutral card contract serialized as ``semantic-card.v1``."""

    kind: SemanticCardKind
    title: str
    source: SemanticCardSource
    summary: str | None = None
    outcome: SemanticCardOutcome | None = None
    fields: tuple[SemanticCardField, ...] = ()
    previews: tuple[SemanticCardPreview, ...] = ()
    raw_evidence: SemanticCardRawEvidence | None = None
    caveats: tuple[str, ...] = ()
    schema_version: Literal["semantic-card.v1"] = CARD_SCHEMA_VERSION

    def to_document(self) -> JSONDocument:
        result: JSONDocument = {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "title": self.title,
            "source": self.source.to_document(),
            "fields": [item.to_document() for item in self.fields],
            "previews": [item.to_document() for item in self.previews],
            "caveats": list(self.caveats),
        }
        if self.summary is not None:
            result["summary"] = self.summary
        if self.outcome is not None:
            result["outcome"] = self.outcome.to_document()
        if self.raw_evidence is not None:
            result["raw_evidence"] = self.raw_evidence.to_document()
        return result


@dataclass(frozen=True, slots=True)
class LineageDescriptor:
    """Bounded lineage input supplied by a caller that already read authority."""

    session_id: str
    provider_family: str = "unknown"
    origin: str | None = None
    root_session_id: str | None = None
    parent_session_id: str | None = None
    parent_native_id: str | None = None
    relation: str = "unknown"
    resolved: bool | None = None
    cycle_detected: bool = False
    lineage_complete: bool | None = None
    lineage_truncation_reason: str | None = None
    inherited_prefix: bool | None = None
    branch_point_message_id: str | None = None
    active_leaf_message_id: str | None = None
    authority: LineageAuthority = LineageAuthority.SESSION_ROW
    availability: LineageAvailability = LineageAvailability.PARTIAL


@dataclass(frozen=True, slots=True)
class TranscriptProse:
    """Typed prose/code/thinking entry with exact authoredness and placement."""

    message_id: str
    role: str
    message_type: str
    text: str
    provider_family: str = "unknown"
    origin: str | None = None
    material_origin: str = "unknown"
    block_id: str | None = None
    block_index: int | None = None
    block_type: str | None = None
    language: str | None = None
    occurred_at: str | None = None
    duration_ms: int | None = None
    parent_message_id: str | None = None
    variant_index: int | None = None
    is_active_path: bool | None = None
    is_active_leaf: bool | None = None
    inherited_prefix: bool | None = None

    def to_document(self) -> JSONDocument:
        return _without_none(
            {
                "message_id": self.message_id,
                "role": self.role,
                "message_type": self.message_type,
                "provider_family": self.provider_family,
                "origin": self.origin,
                "material_origin": self.material_origin,
                "text": self.text,
                "block_id": self.block_id,
                "block_index": self.block_index,
                "block_type": self.block_type,
                "language": self.language,
                "occurred_at": self.occurred_at,
                "duration_ms": self.duration_ms,
                "parent_message_id": self.parent_message_id,
                "variant_index": self.variant_index,
                "is_active_path": self.is_active_path,
                "is_active_leaf": self.is_active_leaf,
                "inherited_prefix": self.inherited_prefix,
            }
        )


@dataclass(frozen=True, slots=True)
class TranscriptNoticeSource:
    """One exact typed-block coordinate represented by a compact notice."""

    message_id: str
    block_index: int
    block_type: str
    block_id: str | None = None
    role: str = "unknown"
    message_type: str = "message"
    provider_family: str = "unknown"
    origin: str | None = None
    material_origin: str = "unknown"
    occurred_at: str | None = None
    duration_ms: int | None = None
    parent_message_id: str | None = None
    variant_index: int | None = None
    is_active_path: bool | None = None
    is_active_leaf: bool | None = None
    inherited_prefix: bool | None = None

    def to_document(self) -> JSONDocument:
        return _without_none(
            {
                "message_id": self.message_id,
                "block_index": self.block_index,
                "block_type": self.block_type,
                "block_id": self.block_id,
                "role": self.role,
                "message_type": self.message_type,
                "provider_family": self.provider_family,
                "origin": self.origin,
                "material_origin": self.material_origin,
                "occurred_at": self.occurred_at,
                "duration_ms": self.duration_ms,
                "parent_message_id": self.parent_message_id,
                "variant_index": self.variant_index,
                "is_active_path": self.is_active_path,
                "is_active_leaf": self.is_active_leaf,
                "inherited_prefix": self.inherited_prefix,
            }
        )


@dataclass(frozen=True, slots=True)
class TranscriptNotice:
    """Compact operator presentation that retains every source coordinate."""

    kind: SemanticNoticeKind
    sources: tuple[TranscriptNoticeSource, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("semantic transcript notice requires at least one source")

    @property
    def count(self) -> int:
        return len(self.sources)

    def to_document(self) -> JSONDocument:
        return {
            "kind": self.kind.value,
            "count": self.count,
            "sources": [source.to_document() for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class SemanticTranscriptEntry:
    """Exactly one ordered prose, card, or typed notice entry."""

    prose: TranscriptProse | None = None
    card: SemanticCard | None = None
    notice: TranscriptNotice | None = None

    def __post_init__(self) -> None:
        if sum(item is not None for item in (self.prose, self.card, self.notice)) != 1:
            raise ValueError("semantic transcript entry requires exactly one of prose, card, or notice")

    @property
    def primary_message_id(self) -> str | None:
        if self.prose is not None:
            return self.prose.message_id
        if self.card is not None:
            return self.card.source.message_id or self.card.source.result_message_id
        assert self.notice is not None
        return self.notice.sources[0].message_id

    def to_document(self) -> JSONDocument:
        if self.card is not None:
            return {"entry_type": "card", "card": self.card.to_document()}
        if self.prose is not None:
            return {"entry_type": "prose", "prose": self.prose.to_document()}
        assert self.notice is not None
        return {"entry_type": "notice", "notice": self.notice.to_document()}


@dataclass(frozen=True, slots=True)
class SemanticTranscript:
    session_id: str
    entries: tuple[SemanticTranscriptEntry, ...] = field(default_factory=tuple)
    schema_version: Literal["semantic-transcript.v1"] = TRANSCRIPT_SCHEMA_VERSION

    @property
    def cards(self) -> tuple[SemanticCard, ...]:
        return tuple(entry.card for entry in self.entries if entry.card is not None)

    @property
    def notices(self) -> tuple[TranscriptNotice, ...]:
        return tuple(entry.notice for entry in self.entries if entry.notice is not None)

    def to_document(self) -> JSONDocument:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "entries": [entry.to_document() for entry in self.entries],
        }


def card_json_documents(cards: tuple[SemanticCard, ...] | list[SemanticCard]) -> list[JSONDocument]:
    return [card.to_document() for card in cards]


def _without_none(values: dict[str, object]) -> JSONDocument:
    result: JSONDocument = {}
    for key, value in values.items():
        if value is not None and is_json_value(value):
            result[key] = value
    return result


__all__ = [
    "CARD_SCHEMA_VERSION",
    "card_json_documents",
    "CardOutcomeState",
    "LineageAuthority",
    "LineageAvailability",
    "LineageDescriptor",
    "PreviewStrategy",
    "SemanticCard",
    "SemanticCardField",
    "SemanticCardKind",
    "SemanticCardOutcome",
    "SemanticCardPreview",
    "SemanticCardRawEvidence",
    "SemanticCardSource",
    "SemanticNoticeKind",
    "SemanticTranscript",
    "SemanticTranscriptEntry",
    "TRANSCRIPT_SCHEMA_VERSION",
    "TranscriptNotice",
    "TranscriptNoticeSource",
    "TranscriptProse",
]
