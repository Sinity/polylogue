"""Plain input structs the encoder accepts.

Deliberately decoupled from the ORM-flavored ``polylogue.archive.session`` /
``polylogue.archive.message`` pydantic models: those lack lineage
(``session_links``) and usage (``session_model_usage``) fields entirely (they
live in separate repository reads), and coupling the wire encoder to the live
domain model's exact shape would make every unrelated domain-model field
addition a protocol-versioning question. A caller (302r.2's producer, or a
test fixture) assembles a ``SessionMaterial`` from whatever domain reads it
has; encoding needs nothing else, in particular no archive DB handle.

Id formulas mirror the real ``index.db`` generated columns exactly
(``sessions.session_id``, ``messages.message_id``, ``blocks.block_id``,
``session_events.event_id``, ``attachment_refs.ref_id``) so record_id values
in the wire protocol are byte-identical to what a live archive would compute
for the same content -- not required by the acceptance criteria, but it means
a decoded record's id is directly cross-checkable against a running archive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.core.enums import BlockType, LinkType, MaterialOrigin, MessageType, Origin, Role, SessionKind
from polylogue.core.json import JSONValue


@dataclass(frozen=True, slots=True)
class BlockInput:
    position: int
    block_type: BlockType
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: dict[str, JSONValue] | None = None
    tool_result_is_error: bool | None = None
    tool_result_exit_code: int | None = None
    semantic_type: str | None = None
    media_type: str | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class AttachmentInput:
    position: int
    attachment_id: str
    display_name: str | None = None
    media_type: str | None = None
    byte_count: int = 0
    blob_sha256: str | None = None
    acquisition_status: str = "unfetched"
    upload_origin: str | None = None
    source_url: str | None = None
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class MessageInput:
    native_id: str | None
    position: int
    role: Role
    text: str | None = None
    variant_index: int = 0
    message_type: MessageType = MessageType.MESSAGE
    material_origin: MaterialOrigin = MaterialOrigin.UNKNOWN
    occurred_at_ms: int | None = None
    model_name: str | None = None
    parent_native_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ms: int | None = None
    blocks: tuple[BlockInput, ...] = ()
    attachments: tuple[AttachmentInput, ...] = ()


@dataclass(frozen=True, slots=True)
class LineageInput:
    dst_origin: Origin
    dst_native_id: str
    link_type: LinkType
    branch_point_message_native_id: str | None = None
    inheritance: str | None = None
    status: str | None = None
    confidence: float = 1.0
    observed_at_ms: int | None = None


@dataclass(frozen=True, slots=True)
class UsageInput:
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    cost_credits: float | None = None
    cost_provenance: str | None = None


@dataclass(frozen=True, slots=True)
class SessionEventInput:
    position: int
    event_type: str
    summary: str
    payload: dict[str, JSONValue] = field(default_factory=dict)
    source_message_native_id: str | None = None
    occurred_at_ms: int | None = None


@dataclass(frozen=True, slots=True)
class FidelityGapInput:
    scope: str
    record_id: str
    gap_kind: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SessionMaterial:
    origin: Origin
    native_id: str
    title: str | None = None
    session_kind: SessionKind = SessionKind.STANDARD
    created_at_ms: int | None = None
    updated_at_ms: int | None = None
    git_branch: str | None = None
    git_repository_url: str | None = None
    provider_project_ref: str | None = None
    working_directories: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    messages: tuple[MessageInput, ...] = ()
    lineage: tuple[LineageInput, ...] = ()
    usage: tuple[UsageInput, ...] = ()
    session_events: tuple[SessionEventInput, ...] = ()
    fidelity_gaps: tuple[FidelityGapInput, ...] = ()

    @property
    def session_id(self) -> str:
        return f"{self.origin.value}:{self.native_id}"


__all__ = [
    "AttachmentInput",
    "BlockInput",
    "FidelityGapInput",
    "LineageInput",
    "MessageInput",
    "SessionEventInput",
    "SessionMaterial",
    "UsageInput",
]
