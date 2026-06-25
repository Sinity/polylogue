"""Typed parser contracts shared across provider parsers."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider, SessionKind, TitleSource, WebConstructType
from polylogue.core.security import sanitize_path as _sanitize_path_helper
from polylogue.core.timestamps import parse_timestamp


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
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: Mapping[str, object] | None = None
    media_type: str | None = None
    metadata: dict[str, object] | None = None
    web_constructs: list[ParsedWebConstruct] = Field(default_factory=list)

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: object) -> BlockType:
        return BlockType.from_string(str(v))


class ParsedPasteEvidence(BaseModel):
    position: int = 0
    start_offset: int | None = None
    end_offset: int | None = None
    boundary_state: str = "hash_only"
    source_event_id: str | None = None
    source_marker: str | None = None
    content_hash: bytes | None = None
    observed_at_ms: int | None = None


class ParsedMessage(BaseModel):
    provider_message_id: str
    role: Role
    text: str | None = None
    timestamp: str | None = None
    occurred_at_ms: int | None = None
    blocks: list[ParsedContentBlock] = Field(default_factory=list)
    message_type: MessageType = MessageType.MESSAGE
    material_origin: MaterialOrigin = MaterialOrigin.UNKNOWN
    parent_message_provider_id: str | None = None
    position: int | None = None
    branch_index: int = 0
    variant_index: int | None = None
    is_active_path: bool | None = None
    is_active_leaf: bool | None = None
    # Token usage flows through from provider raw records to MaterializedMessage.
    # Parsers populate when the raw record carries usage info; otherwise None.
    # Materialization writes these into the messages table, where they drive
    # cost estimation downstream. Were previously dropped on the parser floor,
    # leaving 2.5M rows with input_tokens=output_tokens=0 and dead cost
    # rollups across the entire archive.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model_name: str | None = None
    model_effort: str | None = None
    duration_ms: int | None = None
    sender_name: str | None = None
    recipient: str | None = None
    delivery_status: str | None = None
    end_turn: bool | None = None
    user_context_text: str | None = None
    paste_spans: list[ParsedPasteEvidence] = Field(default_factory=list)

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

    @field_validator("occurred_at_ms", "position", "variant_index", "duration_ms")
    @classmethod
    def non_negative_optional_int(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("parser contract integer fields cannot be negative")
        return value

    @model_validator(mode="after")
    def derive_occurred_at_ms(self) -> ParsedMessage:
        if self.occurred_at_ms is None and self.timestamp:
            parsed = parse_timestamp(self.timestamp)
            if parsed is not None:
                self.occurred_at_ms = int(parsed.timestamp() * 1000)
        if self.material_origin is MaterialOrigin.UNKNOWN:
            from polylogue.archive.message.artifacts import classify_material_origin

            self.material_origin = classify_material_origin(
                role=self.role,
                message_type=self.message_type,
                text=self.text,
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
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_file_id: str | None = None
    provider_drive_id: str | None = None
    upload_origin: str | None = None
    attachment_kind: str | None = None
    source_url: str | None = None
    caption: str | None = None

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


class ParsedSessionEvent(BaseModel):
    """Non-message semantic artifact in the session timeline."""

    event_type: str  # "compaction", "turn_context", etc.
    timestamp: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    source_message_provider_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_message_provider_id", "source_message_id"),
    )


class ParsedSession(BaseModel):
    source_name: Provider
    provider_session_id: str
    title: str | None = None
    session_kind: SessionKind = SessionKind.STANDARD
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[ParsedMessage]
    active_leaf_message_provider_id: str | None = None
    attachments: list[ParsedAttachment] = Field(default_factory=list)
    session_events: list[ParsedSessionEvent] = Field(default_factory=list)
    parent_session_provider_id: str | None = None
    branch_type: BranchType | None = None
    title_source: TitleSource | None = None
    instructions_text: str | None = None
    reported_duration_ms: int | None = None
    reported_cost_usd: float | None = None
    models_used: list[str] = Field(default_factory=list)
    # Universal session-context semantics graduated out of provider metadata.
    working_directories: list[str] = Field(default_factory=list)
    git_branch: str | None = None
    git_repository_url: str | None = None
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
        if value is not None and value < 0:
            raise ValueError("reported_cost_usd cannot be negative")
        return value


class RawSessionData(BaseModel):
    """Container for raw session bytes with metadata.

    When ``blob_hash`` is set, the content has been written to the blob
    store and ``raw_bytes`` may be empty (only a detection prefix was
    needed). Consumers should load from the blob store using ``blob_hash``.
    """

    raw_bytes: bytes = b""
    source_path: str
    source_index: int | None = None
    file_mtime: str | None = None
    provider_hint: Provider | None = None
    blob_hash: str | None = None
    blob_size: int | None = None

    @field_validator("provider_hint", mode="before")
    @classmethod
    def coerce_provider_hint(cls, v: object) -> Provider | None:
        if v is None:
            return None
        return Provider.from_string(str(v))
