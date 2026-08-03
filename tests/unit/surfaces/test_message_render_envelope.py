"""``MessageRenderEnvelope`` contract for ``SessionMessagePayload`` (#1487).

Pins the unified envelope every reader path emits — session detail,
paginated message windows, MCP ``get_messages``, future query-set reads. The
contract enumerates which fields must be present (with their defaults)
and asserts that a roundtrip through ``Message`` populates every typed
slot from the canonical Message model.

If a new field is added to the canonical ``Message`` model, this test
must learn about it (the field list is exhaustively checked). That
prevents the divergence the issue worried about — detail emitting one
field set, paginated emitting another.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType
from polylogue.core.types import ContentHash, MessageId, SessionId
from polylogue.storage.hydrators import message_from_record
from polylogue.storage.runtime.archive.records import BlockRecord, MessageRecord
from polylogue.surfaces.payloads import (
    _MESSAGE_MASK,
    MessageRenderEnvelope,
    ReaderActionAvailabilityPayload,
    TargetRefPayload,
    message_render_envelope_from_archive_row,
    message_render_envelope_from_domain,
)


def _build_message(**overrides: object) -> Message:
    defaults: dict[str, object] = {
        "id": "m1",
        "role": Role.USER,
        "text": "hello",
        "timestamp": datetime(2026, 5, 27, 10, 0, tzinfo=UTC),
        "message_type": MessageType.MESSAGE,
        "parent_id": None,
        "branch_index": 0,
        "has_paste": False,
        "has_tool_use": False,
        "has_thinking": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "model_name": None,
    }
    defaults.update(overrides)
    return Message(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Field-set contract — exhaustive
# ---------------------------------------------------------------------------


def test_envelope_field_set_is_exhaustive() -> None:
    """The derived envelope contains the mask plus computed affordances.

    The contract is exact equality — extra fields would silently widen
    the surface; missing fields would silently drop one. Either case
    requires updating the domain mask deliberately.
    """
    expected = {surface_name for _, surface_name in _MESSAGE_MASK} | {
        "target_ref",
        "anchor",
        "actions",
        "attachment_refs",
        "raw_id",
        "source_path",
    }
    assert set(MessageRenderEnvelope.model_fields) == expected, (
        "MessageRenderEnvelope drifted from the domain mask and affordance set. "
        "Update the mask only when the canonical Message model grows."
    )


def test_domain_roundtrip_populates_every_masked_message_field() -> None:
    message = _build_message(
        role=Role.ASSISTANT,
        text="answer",
        duration_ms=1234,
        stop_reason="max_tokens",
        has_paste=True,
    )
    payload = message_render_envelope_from_domain(message, session_id="c1")
    dumped = message.model_dump(mode="python")

    for domain_name, surface_name in _MESSAGE_MASK:
        expected = dumped[domain_name]
        if domain_name in {"role", "message_type", "material_origin"}:
            expected = expected.value
        elif domain_name == "text":
            expected = expected or ""
        elif domain_name == "has_paste":
            expected = bool(expected)
        assert getattr(payload, surface_name) == expected, surface_name

    assert payload.duration_ms == 1234
    assert payload.stop_reason == "max_tokens"


def test_envelope_minimal_construction_uses_default_envelope_fields() -> None:
    """The minimum required kwargs are id/role/text; everything else has a default."""
    payload = MessageRenderEnvelope(id="m1", role="user", text="hi")

    assert payload.target_ref is None
    assert payload.anchor is None
    assert payload.timestamp is None
    assert payload.parent_id is None
    assert payload.branch_index == 0
    assert payload.position == 0
    assert payload.is_active_path is None
    assert payload.is_active_leaf is False
    assert payload.has_paste_evidence is False
    assert payload.has_tool_use is False
    assert payload.has_thinking is False
    assert payload.input_tokens == 0
    assert payload.output_tokens == 0
    assert payload.cache_read_tokens == 0
    assert payload.cache_write_tokens == 0
    assert payload.model_name is None
    assert payload.attachment_refs == ()
    assert payload.raw_id is None
    assert payload.source_path is None
    assert payload.message_type == "message"


# ---------------------------------------------------------------------------
# Domain projection: every typed slot populated from the canonical Message model
# ---------------------------------------------------------------------------


def test_message_envelope_propagates_branch_lineage_state() -> None:
    msg = _build_message(branch_index=3, parent_id="m-parent")
    payload = message_render_envelope_from_domain(msg, session_id="c1")
    assert payload.branch_index == 3
    assert payload.parent_id == "m-parent"


def test_message_envelope_propagates_position_and_active_path_state() -> None:
    """polylogue-ksgg: position/is_active_path/is_active_leaf must survive
    the read surface so branch/thread structure can be reconstructed
    without direct SQL."""
    msg = _build_message(position=7, is_active_path=True, is_active_leaf=True)
    payload = message_render_envelope_from_domain(msg, session_id="c1")
    assert payload.position == 7
    assert payload.is_active_path is True
    assert payload.is_active_leaf is True


def test_message_envelope_preserves_unknown_active_path() -> None:
    """``is_active_path=None`` (unknown) must not collapse to False."""
    msg = _build_message(is_active_path=None)
    payload = message_render_envelope_from_domain(msg, session_id="c1")
    assert payload.is_active_path is None


def test_from_archive_row_propagates_position_and_active_path_state() -> None:
    # Note: MessageRecord's sibling-order field is named ``branch_index``
    # while the archive-row projection reads the real ArchiveMessageRow's
    # ``variant_index`` attribute -- a pre-existing naming mismatch that
    # means MessageRecord-as-stand-in only exercises the fields whose
    # attribute names agree (position/is_active_path/is_active_leaf).
    row = MessageRecord(
        message_id=MessageId("m1"),
        session_id=SessionId("c1"),
        provider_message_id="native-m1",
        role=Role.USER,
        content_hash=ContentHash("0" * 64),
        is_active_path=True,
        position=5,
        is_active_leaf=True,
    )
    payload = message_render_envelope_from_archive_row(row, session_id="c1")
    assert payload.position == 5
    assert payload.is_active_path is True
    assert payload.is_active_leaf is True


def test_message_envelope_propagates_content_flags() -> None:
    msg = _build_message(
        has_paste=True,
        paste_boundary_state="projected",
        has_tool_use=True,
        has_thinking=True,
    )
    payload = message_render_envelope_from_domain(msg, session_id="c1")
    assert payload.has_paste_evidence is True
    assert payload.paste_boundary_state == "projected"
    assert payload.has_tool_use is True
    assert payload.has_thinking is True


def test_from_archive_row_propagates_paste_boundary_state() -> None:
    """Archive row payloads must not collapse paste evidence to has_paste."""
    row = MessageRecord(
        message_id=MessageId("m1"),
        session_id=SessionId("c1"),
        provider_message_id="native-m1",
        role=Role.USER,
        content_hash=ContentHash("0" * 64),
        blocks=[
            BlockRecord(
                block_id="b1",
                message_id=MessageId("m1"),
                session_id=SessionId("c1"),
                block_index=0,
                type=BlockType.TEXT,
                text="See [Pasted text #1]",
            )
        ],
        has_paste=1,
        paste_boundary_state="projected",
    )

    payload = message_render_envelope_from_archive_row(row, session_id="c1")

    assert payload.has_paste_evidence is True
    assert payload.paste_boundary_state == "projected"


def test_hydrated_message_envelope_preserves_paste_boundary_state() -> None:
    """Storage hydration must not collapse paste evidence to has_paste."""
    record = MessageRecord(
        message_id=MessageId("m1"),
        session_id=SessionId("c1"),
        provider_message_id="native-m1",
        role=Role.USER,
        content_hash=ContentHash("0" * 64),
        blocks=[
            BlockRecord(
                block_id="b1",
                message_id=MessageId("m1"),
                session_id=SessionId("c1"),
                block_index=0,
                type=BlockType.TEXT,
                text="Large pasted body",
            )
        ],
        has_paste=1,
        paste_boundary_state="whole_message_fallback",
    )

    message = message_from_record(record, [])
    payload = message_render_envelope_from_domain(message, session_id="c1")

    assert payload.has_paste_evidence is True
    assert payload.paste_boundary_state == "whole_message_fallback"


def test_message_envelope_propagates_usage_and_model() -> None:
    msg = _build_message(
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=5,
        cache_write_tokens=2,
        model_name="claude-sonnet-4-6",
    )
    payload = message_render_envelope_from_domain(msg, session_id="c1")
    assert payload.input_tokens == 10
    assert payload.output_tokens == 20
    assert payload.cache_read_tokens == 5
    assert payload.cache_write_tokens == 2
    assert payload.model_name == "claude-sonnet-4-6"


def test_message_envelope_carries_explicit_raw_and_source_refs() -> None:
    """``raw_id``/``source_path`` are caller-supplied because they live on
    the session, not the message."""
    msg = _build_message()
    payload = message_render_envelope_from_domain(
        msg,
        session_id="c1",
        raw_id="raw-sha256-abc",
        source_path="/home/user/.claude/projects/p/c1.jsonl",
    )
    assert payload.raw_id == "raw-sha256-abc"
    assert payload.source_path == "/home/user/.claude/projects/p/c1.jsonl"


def test_message_envelope_carries_target_ref_when_session_id_supplied() -> None:
    msg = _build_message()
    payload = message_render_envelope_from_domain(msg, session_id="c1")
    assert payload.target_ref == TargetRefPayload.message(session_id="c1", message_id="m1")
    assert payload.anchor == "message-m1"


def test_message_envelope_omits_target_ref_when_no_session_id() -> None:
    """Without ``session_id`` the message can't be deep-linked."""
    msg = _build_message()
    payload = message_render_envelope_from_domain(msg)
    assert payload.target_ref is None
    # Anchor stays present because it only needs the message id.
    assert payload.anchor == "message-m1"


# ---------------------------------------------------------------------------
# Minimal construction contract
# ---------------------------------------------------------------------------


def test_minimal_message_payload_constructs() -> None:
    payload = MessageRenderEnvelope(
        id="m-c1",
        role="user",
        text="Hello reader",
        target_ref=TargetRefPayload.message(session_id="c1", message_id="m-c1"),
        anchor="message-m-c1",
        actions={"annotate": ReaderActionAvailabilityPayload(enabled=True)},
    )
    dump = payload.model_dump(mode="json", exclude_none=True)
    assert dump["target_ref"]["identity_key"] == "message:c1:m-c1"
    assert dump["actions"]["annotate"]["enabled"] is True


# ---------------------------------------------------------------------------
# Serialization stability — additive fields don't bloat the common payload
# ---------------------------------------------------------------------------


def test_minimal_payload_serializes_compactly_with_exclude_none() -> None:
    """The default envelope serializes without the typed envelope additions
    crowding the JSON. Defaults that are False/0/() must still appear so
    the contract is observable; only the optional ``None`` defaults are
    omitted under ``exclude_none``."""
    payload = MessageRenderEnvelope(id="m1", role="user", text="hi")
    blob = json.loads(payload.to_json(exclude_none=True))

    # Required: typed envelope fields are observable (the test would
    # fail-fast if a new None-default field crept in).
    assert blob["branch_index"] == 0
    assert blob["position"] == 0
    assert blob["is_active_leaf"] is False
    assert blob["has_paste_evidence"] is False
    assert blob["input_tokens"] == 0
    assert blob["attachment_refs"] == []

    # None defaults are correctly omitted.
    assert "target_ref" not in blob
    assert "parent_id" not in blob
    assert "is_active_path" not in blob
    assert "raw_id" not in blob
    assert "source_path" not in blob
    assert "model_name" not in blob


# ---------------------------------------------------------------------------
# Negative: envelope rejects free-form extras
# ---------------------------------------------------------------------------


def test_envelope_rejects_unknown_fields() -> None:
    """``extra="forbid"`` on SurfacePayloadModel keeps the contract closed."""
    with pytest.raises(ValidationError):
        MessageRenderEnvelope(
            id="m1",
            role="user",
            text="hi",
            mystery_field="surprise",
        )
