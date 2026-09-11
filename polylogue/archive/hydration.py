"""Canonical hydration from compact archive rows to domain models.

One owner per archive row family. ``ArchiveBlockRow``/``ArchiveMessageRow``/
``ArchiveAttachmentRow``/``ArchiveSessionSummary``/``ArchiveSessionEnvelope``
reach ``Message``/``Attachment``/``SessionSummary``/``Session`` through the
declarations below and nowhere else; public contracts then apply their own
masks (``polylogue.surfaces.payloads``) to the resulting domain object.

Before this module the same mapping was restated four times
(``api/archive.py``, ``archive/query/archive_execution.py``,
``cli/read_views/standard.py`` and the direct MCP/HTTP row adapters) and the
copies had diverged: the API message mapper dropped ``stop_reason`` and
``tool_result_outcome_unknown_reason``, the query summary mapper dropped
``display_name``, the CLI summary mapper dropped parent identity, branch type
and title provenance. A declaration makes the divergence impossible to
reintroduce silently: every row field carries a disposition, and
``unmapped_row_fields`` (exercised by the focused tests) turns a new
undeclared field into a failure.

Placement: the row dataclasses live in ``polylogue.storage`` and the domain
models live here in ``polylogue.archive``. The conversion is owned on the
domain side because every public surface (``api``/``cli``/``mcp``/``daemon``)
must be able to import it, and the layering gate ratchets new
surface-to-``polylogue/storage`` imports.

Ownership boundaries. ``polylogue.storage.hydrators`` keeps the *record*
family (``MessageRecord``/``SessionRecord``, driven by ``TableColumnSpec``
under polylogue-a7xr.24); the compact archive read models here are a
different row family with their own names, so they carry their own
declaration rather than a second ``TableColumnSpec`` catalog. Attachment
physical readability stays with polylogue-hb9o6: this module consumes the
typed ``ArchiveAttachmentRow.availability`` result and never re-derives it
from ``blob_hash``/``acquisition_status``/``generation_id``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal, cast

from polylogue.core.enums import Origin, SessionKind, TitleSource
from polylogue.core.json import loads
from polylogue.core.timestamps import parse_archive_datetime
from polylogue.core.types import SessionId

if TYPE_CHECKING:
    # Re-exported for type-only consumers that must not import storage
    # internals directly (``polylogue/surfaces`` ratchets those imports).
    from polylogue.archive.attachment.models import Attachment
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary
    from polylogue.storage.sqlite.archive_tiers.archive_query_reads import ArchiveMessageQueryRow
    from polylogue.storage.sqlite.archive_tiers.write import (
        ArchiveAttachmentRow,
        ArchiveBlockRow,
        ArchiveSessionEnvelope,
    )
    from polylogue.storage.sqlite.archive_tiers.write import (
        ArchiveMessageRow as ArchiveMessageRow,  # explicit re-export for surfaces
    )


# ---------------------------------------------------------------------------
# Field dispositions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldDisposition:
    """What happens to one archive-row field on the way to the domain.

    ``exposed`` fields are transferred (optionally through ``transform``) to
    ``domain_name``. ``excluded`` and ``delegated`` fields are deliberately not
    transferred and say why: ``excluded`` means the value is not domain
    semantics at all (a storage coordinate, or an identity component already
    folded into a computed id), ``delegated`` means the semantics are real but
    are owned by another read model that names itself in ``reason``.
    """

    kind: Literal["exposed", "excluded", "delegated"]
    domain_name: str | None = None
    transform: Callable[[Any], Any] | None = None
    reason: str | None = None


def exposed(domain_name: str, transform: Callable[[Any], Any] | None = None) -> FieldDisposition:
    return FieldDisposition(kind="exposed", domain_name=domain_name, transform=transform)


def excluded(reason: str) -> FieldDisposition:
    return FieldDisposition(kind="excluded", reason=reason)


def delegated(reason: str) -> FieldDisposition:
    return FieldDisposition(kind="delegated", reason=reason)


Dispositions = Mapping[str, FieldDisposition]


# ---------------------------------------------------------------------------
# Shared typed transforms
# ---------------------------------------------------------------------------


def _json_object(value: object) -> dict[str, object] | None:
    """Decode a stored JSON object column back into a mapping."""
    if not value:
        return None
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    try:
        parsed = loads(value if isinstance(value, str) else str(value))
    except (ValueError, TypeError):
        return None
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_session_id(value: object) -> SessionId | None:
    return SessionId(str(value)) if value else None


def _optional_title_source(value: object) -> TitleSource | None:
    return TitleSource(str(value)) if value is not None else None


def _optional_branch_type(value: object) -> Any:
    from polylogue.archive.session.branch_type import BranchType

    return BranchType(str(value)) if value else None


def _strings(value: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(value or ())


def _datetime_from_ms(value: object) -> Any:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(float(cast("int", value)) / 1000.0, UTC) if value is not None else None


# ---------------------------------------------------------------------------
# Display-text policy
# ---------------------------------------------------------------------------


def archive_display_text(blocks: Iterable[ArchiveBlockRow]) -> str:
    """The one flattened ``message.text`` policy for archive-row reads.

    Named as a policy input rather than inlined so that a surface wanting
    different display text supplies a different policy instead of forking the
    semantic block/message mapping (which is what produced the divergence this
    module removes). Every current route uses this one.
    """
    from polylogue.storage.sqlite.archive_tiers.write import archive_message_display_text

    return archive_message_display_text(blocks)


def archive_provider_title(title: str | None, title_source: str | None) -> str | None:
    """Return the title a *full session* read may present as its own title.

    ``TitleSource.PATH`` is a legacy structural fallback, not provider title
    evidence, so the detail route suppresses it and lets the display-label
    projection speak instead. Summaries deliberately keep the stored title as
    stored (``ArchiveSessionSummary.title`` is the row value and
    ``display_label`` carries the read-time projection beside it); that
    difference is a declared policy, tested independently, and it does not
    fork any other field.
    """
    return title if title_source in {TitleSource.ORIGIN.value, TitleSource.HEURISTIC.value} else None


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

ARCHIVE_BLOCK_DISPOSITIONS: Dispositions = {
    "block_id": exposed("id"),
    "message_id": excluded("block identity prefix; the domain block is already nested under its message"),
    "block_type": exposed("type"),
    "text": exposed("text"),
    "tool_name": exposed("tool_name"),
    "tool_id": exposed("tool_id"),
    "semantic_type": exposed("semantic_type"),
    "tool_input": exposed("tool_input", _json_object),
    "metadata": exposed("metadata", _json_object),
    "language": exposed("language"),
    "tool_result_is_error": exposed("tool_result_is_error"),
    "tool_result_exit_code": exposed("tool_result_exit_code"),
    "tool_outcome": exposed("tool_outcome", _optional_str),
    "tool_result_outcome_unknown_reason": exposed("tool_result_outcome_unknown_reason"),
}

ARCHIVE_MESSAGE_DISPOSITIONS: Dispositions = {
    "message_id": exposed("id"),
    "native_id": excluded("provider-native id is a computed component of messages.message_id"),
    "role": exposed("role"),
    "position": exposed("position"),
    "variant_index": exposed("branch_index"),
    "is_active_path": exposed("is_active_path"),
    "is_active_leaf": exposed("is_active_leaf"),
    "blocks": delegated("hydrated per block by archive_block_to_domain"),
    "identity_source": exposed("identity_source"),
    "message_type": exposed("message_type"),
    "material_origin": exposed("material_origin"),
    "word_count": delegated(
        "the stored per-message counter; Message.word_count is derived from text by "
        "MessageRuntimeMixin, and the surfaces that report the stored counter read the row directly"
    ),
    "has_tool_use": exposed("has_tool_use"),
    "has_thinking": exposed("has_thinking"),
    "has_paste": exposed("has_paste"),
    "paste_boundary_state": exposed("paste_boundary_state"),
    "occurred_at": exposed("timestamp", parse_archive_datetime),
    "duration_ms": exposed("duration_ms"),
    "parent_message_id": exposed("parent_id"),
    "attachments": delegated("hydrated per attachment by archive_attachment_to_domain"),
    "source_session_id": delegated(
        "lineage composition provenance; the session-detail route exposes it as "
        "source_session_id/inherited_prefix beside the message, not as a Message field"
    ),
    "stop_reason": exposed("stop_reason"),
}

ARCHIVE_ATTACHMENT_DISPOSITIONS: Dispositions = {
    "attachment_id": exposed("id"),
    "message_id": excluded("owning message coordinate; the domain attachment hangs off that message"),
    "display_name": exposed("name"),
    "media_type": exposed("mime_type"),
    "byte_count": exposed("size_bytes"),
    "upload_origin": exposed("upload_origin"),
    "direction": exposed("direction"),
    "producer_ref": exposed("producer_ref"),
    "source_url": exposed("source_url"),
    "caption": exposed("caption"),
    "blob_hash": delegated("physical-readability evidence folded into availability (polylogue-hb9o6)"),
    "acquisition_status": delegated("physical-readability evidence folded into availability (polylogue-hb9o6)"),
    "generation_id": delegated("physical-readability evidence folded into availability (polylogue-hb9o6)"),
    "availability": exposed("availability"),
}

_SUMMARY_AGGREGATE_COUNTERS: tuple[str, ...] = (
    "word_count",
    "reported_duration_ms",
    "tool_use_count",
    "thinking_count",
    "paste_count",
    "user_message_count",
    "authored_user_message_count",
    "assistant_message_count",
    "system_message_count",
    "tool_message_count",
    "user_word_count",
    "authored_user_word_count",
    "assistant_word_count",
)

ARCHIVE_SUMMARY_DISPOSITIONS: Dispositions = {
    "session_id": exposed("id", lambda value: SessionId(str(value))),
    "native_id": excluded("provider-native id is a computed component of sessions.session_id"),
    "origin": exposed("origin", Origin.from_string),
    "title": exposed("title"),
    "created_at": exposed("created_at", parse_archive_datetime),
    "updated_at": exposed("updated_at", parse_archive_datetime),
    "message_count": exposed("message_count"),
    "tags": exposed("tags_m2m", _strings),
    "parent_id": exposed("parent_id", _optional_session_id),
    "branch_type": exposed("branch_type", _optional_branch_type),
    "session_kind": exposed("session_kind", SessionKind.normalize),
    # Aggregate counters: the domain summary carries session identity and
    # provenance; totals are the session-stats / query-row projections'
    # (``surfaces.query_rows.session_row``, ``analysis`` insights).
    **{
        name: delegated("aggregate counter owned by the session-stats / query-row projection")
        for name in _SUMMARY_AGGREGATE_COUNTERS
    },
    "working_directories": exposed("working_directories", _strings),
    "title_source": exposed("title_source", _optional_title_source),
    "title_ref": exposed("title_ref"),
    "git_branch": exposed("git_branch"),
    "git_repository_url": exposed("git_repository_url"),
    "provider_project_ref": exposed("provider_project_ref"),
    "display_name": exposed("display_name"),
    "display_label": exposed("display_label"),
    "terminal_state": exposed("terminal_state"),
    "total_cost_usd": exposed("total_cost_usd"),
    "cost_provenance": exposed("cost_provenance"),
}

ARCHIVE_ENVELOPE_DISPOSITIONS: Dispositions = {
    "session_id": exposed("id", lambda value: SessionId(str(value))),
    "native_id": excluded("provider-native id is a computed component of sessions.session_id"),
    "origin": exposed("origin", Origin.from_string),
    "title": delegated("suppressed for legacy path-sourced titles by archive_provider_title"),
    "active_leaf_message_id": delegated("resume target resolved by the continue/resume route, not a Session field"),
    "messages": delegated("hydrated per message by archive_message_to_domain"),
    "session_kind": exposed("session_kind", SessionKind.normalize),
    "parent_session_id": exposed("parent_id", _optional_session_id),
    "root_session_id": delegated("topology roots are served by the session-links/topology read model"),
    "branch_type": exposed("branch_type", _optional_branch_type),
    "title_source": exposed("title_source", _optional_title_source),
    "title_ref": exposed("title_ref"),
    "display_name": exposed("display_name"),
    "instructions_text": delegated("system-instruction text is served by the instructions read model"),
    "created_at": delegated(
        "stored timestamp, with the message envelope as fallback (see archive_envelope_to_session)"
    ),
    "updated_at": delegated(
        "stored timestamp, with the message envelope as fallback (see archive_envelope_to_session)"
    ),
    "working_directories": exposed("working_directories", _strings),
    "git_branch": exposed("git_branch"),
    "git_repository_url": exposed("git_repository_url"),
    "provider_project_ref": exposed("provider_project_ref"),
    "reported_cost_usd": exposed("reported_cost_usd"),
    "orphan_attachments": delegated("hydrated into Session.attachments by archive_attachment_to_domain"),
    **{
        name: delegated("composition fact reported by the lineage descriptor, not as session metadata")
        for name in (
            "lineage_complete",
            "lineage_truncation_reason",
            "lineage_inheritance",
            "lineage_branch_point_message_id",
        )
    },
    "total_message_count": delegated("page cardinality belongs to the paginated message-row envelope"),
}

# The bounded message-page projection is deliberately narrower than the composed
# read: it answers "one page of this session's messages" with one bounded SQL
# statement and never materializes the composed transcript. Its message-level
# gaps are declared here rather than silently defaulted, and the surfaces that
# serve it name the richer operation (see ``MESSAGE_QUERY_ROW_RICHER_OPERATION``).
ARCHIVE_MESSAGE_QUERY_ROW_DISPOSITIONS: Dispositions = {
    "message_id": exposed("id"),
    "session_id": excluded("owning session coordinate; the page envelope already names its session"),
    "origin": exposed("origin", Origin.from_string),
    "title": delegated("session title belongs to the session summary, not a message"),
    "repo": delegated("session repository context belongs to the session summary, not a message"),
    "role": exposed("role"),
    "message_type": exposed("message_type"),
    "material_origin": exposed("material_origin"),
    "occurred_at_ms": exposed("timestamp", _datetime_from_ms),
    "position": exposed("position"),
    "word_count": delegated("Message.word_count is derived from text by MessageRuntimeMixin"),
    "text": delegated(
        "the row's pre-flattened search text; the display-text policy joins the block rows and "
        "this column is only the fallback when a message has no block text at all"
    ),
    "blocks": delegated("hydrated per block by archive_block_to_domain"),
}

# Message fields the bounded page projection does not select. They keep their
# domain defaults, which is only honest because the surface says so.
MESSAGE_QUERY_ROW_UNPROJECTED: tuple[str, ...] = (
    "identity_source",
    "parent_id",
    "branch_index",
    "is_active_path",
    "is_active_leaf",
    "has_tool_use",
    "has_thinking",
    "has_paste",
    "paste_boundary_state",
    "duration_ms",
    "stop_reason",
    "attachments",
)

MESSAGE_QUERY_ROW_RICHER_OPERATION = (
    "Branch/paste/tool flags, attachments and stop_reason are not selected by "
    "the bounded message-page projection; read the composed session "
    "(get_session / read_session) for those fields."
)

ROW_DISPOSITIONS: Mapping[str, Dispositions] = {
    "ArchiveBlockRow": ARCHIVE_BLOCK_DISPOSITIONS,
    "ArchiveMessageRow": ARCHIVE_MESSAGE_DISPOSITIONS,
    "ArchiveAttachmentRow": ARCHIVE_ATTACHMENT_DISPOSITIONS,
    "ArchiveSessionSummary": ARCHIVE_SUMMARY_DISPOSITIONS,
    "ArchiveSessionEnvelope": ARCHIVE_ENVELOPE_DISPOSITIONS,
    "ArchiveMessageQueryRow": ARCHIVE_MESSAGE_QUERY_ROW_DISPOSITIONS,
}


def unmapped_row_fields(row_type: type) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(undeclared_fields, stale_declarations)`` for one row dataclass.

    A row field with no disposition is undeclared; a disposition naming a field
    the dataclass no longer has is stale. Both are failures in the focused
    tests, so adding a semantic archive field without deciding whether it is
    exposed, excluded or delegated cannot pass verification.
    """
    declared = ROW_DISPOSITIONS[row_type.__name__]
    actual = {field.name for field in fields(row_type)}
    return (
        tuple(sorted(actual - set(declared))),
        tuple(sorted(set(declared) - actual)),
    )


def exposed_domain_names(dispositions: Dispositions) -> frozenset[str]:
    """Domain field names this declaration transfers."""
    return frozenset(
        disposition.domain_name
        for disposition in dispositions.values()
        if disposition.kind == "exposed" and disposition.domain_name is not None
    )


def _transfer(row: object, dispositions: Dispositions) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field_name, disposition in dispositions.items():
        if disposition.kind != "exposed" or disposition.domain_name is None:
            continue
        value = getattr(row, field_name)
        if disposition.transform is not None:
            value = disposition.transform(value)
        values[disposition.domain_name] = value
    return values


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------


def archive_block_to_domain(block: ArchiveBlockRow) -> dict[str, object]:
    """Hydrate one compact block row into a domain content block.

    ``None`` values are dropped so an absent field stays absent rather than
    becoming an explicit null in every public block payload.
    """
    return {key: value for key, value in _transfer(block, ARCHIVE_BLOCK_DISPOSITIONS).items() if value is not None}


def archive_attachment_to_domain(attachment: ArchiveAttachmentRow) -> Attachment:
    """Hydrate a compact attachment row into the domain attachment."""
    from polylogue.archive.attachment.models import Attachment

    return Attachment(path=None, **_transfer(attachment, ARCHIVE_ATTACHMENT_DISPOSITIONS))


def archive_message_to_domain(
    message: ArchiveMessageRow,
    *,
    origin: Origin | None = None,
    display_text: Callable[[Iterable[ArchiveBlockRow]], str] = archive_display_text,
) -> Message:
    """Hydrate a compact message row (plus its blocks/attachments) into ``Message``.

    ``origin`` is session-level, not a message column, so the caller supplies
    it. ``None`` means "this route does not carry origin per message" (the
    render envelope does not project it), never a guessed origin.
    """
    from polylogue.archive.message.models import Message

    return Message(
        **_transfer(message, ARCHIVE_MESSAGE_DISPOSITIONS),
        origin=origin,
        text=display_text(message.blocks) or None,
        blocks=[archive_block_to_domain(block) for block in message.blocks],
        attachments=[archive_attachment_to_domain(attachment) for attachment in message.attachments],
    )


def archive_message_query_row_to_domain(
    row: ArchiveMessageQueryRow,
    *,
    display_text: Callable[[Iterable[ArchiveBlockRow]], str] = archive_display_text,
) -> Message:
    """Hydrate one bounded message-page row into ``Message``.

    Block semantics (including ``tool_outcome`` and
    ``tool_result_outcome_unknown_reason``) come from the same block hydrator
    the composed read uses, and the text is the same block-join policy, so a
    page read and a composed read agree on every field this projection
    selects. Only the message-level fields listed in
    ``MESSAGE_QUERY_ROW_UNPROJECTED`` are absent.
    """
    from polylogue.archive.message.models import Message

    return Message(
        **_transfer(row, ARCHIVE_MESSAGE_QUERY_ROW_DISPOSITIONS),
        text=display_text(row.blocks) or row.text or None,
        blocks=[archive_block_to_domain(block) for block in row.blocks],
    )


def archive_summary_to_domain(summary: ArchiveSessionSummary) -> SessionSummary:
    """Hydrate an archive summary projection into ``SessionSummary``.

    Every persisted semantic field the summary row carries reaches the domain
    model here; nothing loads full messages to recover summary metadata.
    """
    from polylogue.archive.session.domain_models import SessionSummary

    return SessionSummary(**_transfer(summary, ARCHIVE_SUMMARY_DISPOSITIONS))


def archive_envelope_to_session(
    session: ArchiveSessionEnvelope,
    *,
    display_label: str | None = None,
    display_text: Callable[[Iterable[ArchiveBlockRow]], str] = archive_display_text,
) -> Session:
    """Hydrate a full session envelope into ``Session``."""
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.session.domain_models import Session

    origin = Origin.from_string(session.origin)
    messages = [
        archive_message_to_domain(message, origin=origin, display_text=display_text) for message in session.messages
    ]
    timestamps = [message.timestamp for message in messages if message.timestamp is not None]
    # Prefer the stored session timestamps (sessions.created_at_ms/updated_at_ms)
    # so a full read and a summary read report the same timeline; fall back to
    # the message-timestamp envelope only when the session row carries none.
    stored_created = parse_archive_datetime(session.created_at)
    stored_updated = parse_archive_datetime(session.updated_at)
    values = _transfer(session, ARCHIVE_ENVELOPE_DISPOSITIONS)
    values["origin"] = origin
    return Session(
        **values,
        title=archive_provider_title(session.title, session.title_source),
        display_label=display_label,
        messages=MessageCollection(messages=messages),
        created_at=stored_created or (min(timestamps) if timestamps else None),
        updated_at=stored_updated or (max(timestamps) if timestamps else None),
        attachments=[archive_attachment_to_domain(attachment) for attachment in session.orphan_attachments],
    )


__all__ = [
    "ARCHIVE_ATTACHMENT_DISPOSITIONS",
    "ARCHIVE_BLOCK_DISPOSITIONS",
    "ARCHIVE_ENVELOPE_DISPOSITIONS",
    "ARCHIVE_MESSAGE_DISPOSITIONS",
    "ARCHIVE_MESSAGE_QUERY_ROW_DISPOSITIONS",
    "MESSAGE_QUERY_ROW_RICHER_OPERATION",
    "MESSAGE_QUERY_ROW_UNPROJECTED",
    "ARCHIVE_SUMMARY_DISPOSITIONS",
    "ROW_DISPOSITIONS",
    "FieldDisposition",
    "archive_attachment_to_domain",
    "archive_block_to_domain",
    "archive_display_text",
    "archive_envelope_to_session",
    "archive_message_query_row_to_domain",
    "archive_message_to_domain",
    "archive_provider_title",
    "archive_summary_to_domain",
    "delegated",
    "excluded",
    "exposed",
    "exposed_domain_names",
    "unmapped_row_fields",
]
