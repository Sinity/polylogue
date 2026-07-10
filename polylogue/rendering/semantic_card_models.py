"""Pure data contracts for provider-neutral semantic transcript cards."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from polylogue.core.json import JSONDocument, is_json_value

CARD_SCHEMA_VERSION: Literal["semantic-card.v1"] = "semantic-card.v1"


class SemanticCardKind(str, Enum):
    """Closed card vocabulary shared by CLI and web payloads."""

    SHELL = "shell"
    FILE_EDIT = "file_edit"
    TASK = "task"
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


@dataclass(frozen=True, slots=True)
class SemanticCardSource:
    """Exact source coordinates retained by a semantic card."""

    session_id: str
    provider_family: str = "unknown"
    message_id: str | None = None
    block_id: str | None = None
    block_index: int | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    result_message_id: str | None = None
    result_block_id: str | None = None
    result_block_index: int | None = None

    def to_document(self) -> JSONDocument:
        return _without_none(
            {
                "session_id": self.session_id,
                "provider_family": self.provider_family,
                "message_id": self.message_id,
                "block_id": self.block_id,
                "block_index": self.block_index,
                "tool_name": self.tool_name,
                "tool_id": self.tool_id,
                "result_message_id": self.result_message_id,
                "result_block_id": self.result_block_id,
                "result_block_index": self.result_block_index,
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
    """Pure lineage input supplied by a caller that already read topology."""

    session_id: str
    root_session_id: str
    parent_session_id: str | None = None
    parent_native_id: str | None = None
    relation: str = "unknown"
    resolved: bool = True
    cycle_detected: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptProse:
    """Prose entry retained alongside semantic cards."""

    message_id: str
    role: str
    message_type: str
    text: str
    block_id: str | None = None
    block_type: str | None = None

    def to_document(self) -> JSONDocument:
        return _without_none(
            {
                "message_id": self.message_id,
                "role": self.role,
                "message_type": self.message_type,
                "text": self.text,
                "block_id": self.block_id,
                "block_type": self.block_type,
            }
        )


@dataclass(frozen=True, slots=True)
class SemanticTranscriptEntry:
    """Exactly one ordered prose or card entry."""

    prose: TranscriptProse | None = None
    card: SemanticCard | None = None

    def __post_init__(self) -> None:
        if (self.prose is None) == (self.card is None):
            raise ValueError("semantic transcript entry requires exactly one of prose or card")

    def to_document(self) -> JSONDocument:
        if self.card is not None:
            return {"entry_type": "card", "card": self.card.to_document()}
        assert self.prose is not None
        return {"entry_type": "prose", "prose": self.prose.to_document()}


@dataclass(frozen=True, slots=True)
class SemanticTranscript:
    session_id: str
    entries: tuple[SemanticTranscriptEntry, ...] = field(default_factory=tuple)

    @property
    def cards(self) -> tuple[SemanticCard, ...]:
        return tuple(entry.card for entry in self.entries if entry.card is not None)

    def to_document(self) -> JSONDocument:
        return {
            "schema_version": "semantic-transcript.v1",
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
    "LineageDescriptor",
    "PreviewStrategy",
    "SemanticCard",
    "SemanticCardField",
    "SemanticCardKind",
    "SemanticCardOutcome",
    "SemanticCardPreview",
    "SemanticCardRawEvidence",
    "SemanticCardSource",
    "SemanticTranscript",
    "SemanticTranscriptEntry",
    "TranscriptProse",
]
