"""Adapt a live archive ``Session`` read into a ``SessionMaterial`` input.

This is the first real Polylogue-side producer glue for polylogue-303r.2: it
turns an already-hydrated :class:`~polylogue.archive.models.Session` (for
example from ``SessionRepository.get_session_tree``) into the
:class:`~polylogue.material_protocol.v1.SessionMaterial` the v1 encoder
accepts, using the SAME id/vocabulary formulas the live archive uses
(``native_id_from_session_id``, the real ``Origin``/``Role``/``BlockType``/
``MaterialOrigin``/``MessageType`` enums) rather than inventing a parallel
vocabulary.

Known, explicitly-declared scope gap (v1 of this adapter): lineage
(``session_links``), usage (``session_model_usage``), and session events are
separate repository reads the caller has not necessarily loaded alongside the
session tree, so this adapter does not populate them and instead records a
:class:`~polylogue.material_protocol.v1.FidelityGapInput` naming the omission
-- honest under-coverage using the protocol's own fidelity-gap vocabulary,
not silent data loss. Wiring those additional repository reads through is
follow-up scope (tracked on polylogue-303r.2's own notes), not required to
prove the durable-obligation/transport contract this package's tests target.
"""

from __future__ import annotations

from datetime import datetime

from polylogue.archive.models import Session
from polylogue.core.enums import BlockType
from polylogue.core.json import JSONValue
from polylogue.core.web_urls import native_id_from_session_id
from polylogue.material_protocol.v1 import BlockInput, FidelityGapInput, MessageInput, SessionMaterial


def _timestamp_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp() * 1000)


def _json_safe(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _block_input(position: int, block: dict[str, object]) -> BlockInput | None:
    raw_type = block.get("type")
    if raw_type is None:
        return None
    try:
        block_type = BlockType.from_string(str(raw_type))
    except ValueError:
        return None
    tool_input = block.get("tool_input")
    text = block.get("text")
    tool_name = block.get("tool_name")
    tool_id = block.get("tool_id")
    is_error = block.get("tool_result_is_error")
    exit_code = block.get("tool_result_exit_code")
    semantic_type = block.get("semantic_type")
    media_type = block.get("media_type")
    return BlockInput(
        position=position,
        block_type=block_type,
        text=text if isinstance(text, str) else None,
        tool_name=tool_name if isinstance(tool_name, str) else None,
        tool_id=tool_id if isinstance(tool_id, str) else None,
        tool_input=dict(tool_input) if isinstance(tool_input, dict) else None,
        tool_result_is_error=is_error if isinstance(is_error, bool) else None,
        tool_result_exit_code=exit_code if isinstance(exit_code, int) else None,
        semantic_type=semantic_type if isinstance(semantic_type, str) else None,
        media_type=media_type if isinstance(media_type, str) else None,
    )


def session_material_from_session(session: Session) -> SessionMaterial:
    """Build a ``SessionMaterial`` from a hydrated archive ``Session``.

    Message/block ``native_id`` is deliberately left ``None``: the encoder's
    own documented fallback (``position || '.' || variant_index``) is a real
    production identity path, not a placeholder -- many provider payloads
    have no native per-message id at all. See the module docstring for the
    lineage/usage/session-event scope gap this v1 adapter declares.

    Raises:
        ValueError: if ``session.id`` is not a well-formed ``origin:native_id``
            string (every archive-read session id is, by construction of the
            generated ``sessions.session_id`` column).
    """
    native_id = native_id_from_session_id(session.id)
    if native_id is None:
        raise ValueError(f"session.id {session.id!r} is not a well-formed 'origin:native_id' session id")
    messages = tuple(
        MessageInput(
            native_id=None,
            position=index,
            role=message.role,
            text=message.text,
            message_type=message.message_type,
            material_origin=message.material_origin,
            occurred_at_ms=_timestamp_ms(message.timestamp),
            model_name=message.model_name,
            input_tokens=message.input_tokens,
            output_tokens=message.output_tokens,
            cache_read_tokens=message.cache_read_tokens,
            cache_write_tokens=message.cache_write_tokens,
            duration_ms=message.duration_ms or None,
            blocks=tuple(
                block_input
                for block_position, raw_block in enumerate(message.blocks)
                if (block_input := _block_input(block_position, raw_block)) is not None
            ),
        )
        for index, message in enumerate(session.messages)
    )
    fidelity_gaps = (
        FidelityGapInput(
            scope="session",
            record_id=session.id,
            gap_kind="omitted_relation",
            detail=(
                "polylogue.sinex.material_adapter v1 does not populate lineage, usage, "
                "or session_events -- see module docstring"
            ),
        ),
    )
    return SessionMaterial(
        origin=session.origin,
        native_id=native_id,
        title=session.title,
        session_kind=session.session_kind,
        created_at_ms=_timestamp_ms(session.created_at),
        updated_at_ms=_timestamp_ms(session.updated_at),
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
        provider_project_ref=session.provider_project_ref,
        working_directories=session.working_directories,
        metadata={key: _json_safe(value) for key, value in session.metadata.items()},
        tags=session.tags_m2m,
        messages=messages,
        lineage=(),
        usage=(),
        session_events=(),
        fidelity_gaps=fidelity_gaps,
    )


__all__ = ["session_material_from_session"]
