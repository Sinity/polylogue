"""``MessageRenderEnvelope`` contract for ``SessionMessagePayload`` (#1487).

Pins the unified envelope every reader path emits — session detail,
paginated message windows, MCP ``get_messages``, future query-set reads. The
contract enumerates which fields must be present (with their defaults)
and asserts that a roundtrip through ``Message`` populates every typed
slot from the canonical Message model.

If a new field is added to ``SessionMessagePayload``, this test
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
from polylogue.surfaces.payloads import (
    ReaderActionAvailabilityPayload,
    SessionMessagePayload,
    TargetRefPayload,
)

# The canonical envelope field set. Adding a new field to
# ``SessionMessagePayload`` requires extending this list — that's
# the point: the contract should know about every emitted field so
# downstream surfaces never silently drop one.
_ENVELOPE_FIELDS: tuple[str, ...] = (
    "id",
    "role",
    "text",
    "target_ref",
    "anchor",
    "actions",
    "timestamp",
    "message_type",
    "material_origin",
    "content_blocks",
    "parent_id",
    "branch_index",
    "has_paste",
    "paste_boundary_state",
    "has_tool_use",
    "has_thinking",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "model_name",
    "attachment_refs",
    "raw_id",
    "source_path",
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
    """``SessionMessagePayload`` field set matches the canonical envelope.

    The contract is exact equality — extra fields would silently widen
    the surface; missing fields would silently drop one. Either case
    requires updating ``_ENVELOPE_FIELDS`` deliberately.
    """
    assert set(SessionMessagePayload.model_fields) == set(_ENVELOPE_FIELDS), (
        "SessionMessagePayload field set drifted from the canonical envelope. "
        "Update tests/unit/surfaces/test_message_render_envelope.py:_ENVELOPE_FIELDS "
        "to match, and verify the detail and paginated message endpoints both emit "
        "the new field."
    )


def test_envelope_minimal_construction_uses_default_envelope_fields() -> None:
    """The minimum required kwargs are id/role/text; everything else has a default."""
    payload = SessionMessagePayload(id="m1", role="user", text="hi")

    assert payload.target_ref is None
    assert payload.anchor is None
    assert payload.timestamp is None
    assert payload.parent_id is None
    assert payload.branch_index == 0
    assert payload.has_paste is False
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
# from_message: every typed slot populated from the canonical Message model
# ---------------------------------------------------------------------------


def test_from_message_propagates_branch_lineage_state() -> None:
    msg = _build_message(branch_index=3, parent_id="m-parent")
    payload = SessionMessagePayload.from_message(msg, session_id="c1")
    assert payload.branch_index == 3
    assert payload.parent_id == "m-parent"


def test_from_message_propagates_content_flags() -> None:
    msg = _build_message(has_paste=True, has_tool_use=True, has_thinking=True)
    payload = SessionMessagePayload.from_message(msg, session_id="c1")
    assert payload.has_paste is True
    assert payload.has_tool_use is True
    assert payload.has_thinking is True


def test_from_message_propagates_usage_and_model() -> None:
    msg = _build_message(
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=5,
        cache_write_tokens=2,
        model_name="claude-sonnet-4-6",
    )
    payload = SessionMessagePayload.from_message(msg, session_id="c1")
    assert payload.input_tokens == 10
    assert payload.output_tokens == 20
    assert payload.cache_read_tokens == 5
    assert payload.cache_write_tokens == 2
    assert payload.model_name == "claude-sonnet-4-6"


def test_from_message_carries_explicit_raw_and_source_refs() -> None:
    """``raw_id``/``source_path`` are caller-supplied because they live on
    the session, not the message."""
    msg = _build_message()
    payload = SessionMessagePayload.from_message(
        msg,
        session_id="c1",
        raw_id="raw-sha256-abc",
        source_path="/home/user/.claude/projects/p/c1.jsonl",
    )
    assert payload.raw_id == "raw-sha256-abc"
    assert payload.source_path == "/home/user/.claude/projects/p/c1.jsonl"


def test_from_message_carries_target_ref_when_session_id_supplied() -> None:
    msg = _build_message()
    payload = SessionMessagePayload.from_message(msg, session_id="c1")
    assert payload.target_ref == TargetRefPayload.message(session_id="c1", message_id="m1")
    assert payload.anchor == "message-m1"


def test_from_message_omits_target_ref_when_no_session_id() -> None:
    """Without ``session_id`` the message can't be deep-linked."""
    msg = _build_message()
    payload = SessionMessagePayload.from_message(msg)
    assert payload.target_ref is None
    # Anchor stays present because it only needs the message id.
    assert payload.anchor == "message-m1"


# ---------------------------------------------------------------------------
# Minimal construction contract
# ---------------------------------------------------------------------------


def test_minimal_message_payload_constructs() -> None:
    payload = SessionMessagePayload(
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
    payload = SessionMessagePayload(id="m1", role="user", text="hi")
    blob = json.loads(payload.to_json(exclude_none=True))

    # Required: typed envelope fields are observable (the test would
    # fail-fast if a new None-default field crept in).
    assert blob["branch_index"] == 0
    assert blob["has_paste"] is False
    assert blob["input_tokens"] == 0
    assert blob["attachment_refs"] == []

    # None defaults are correctly omitted.
    assert "target_ref" not in blob
    assert "parent_id" not in blob
    assert "raw_id" not in blob
    assert "source_path" not in blob
    assert "model_name" not in blob


# ---------------------------------------------------------------------------
# Negative: envelope rejects free-form extras
# ---------------------------------------------------------------------------


def test_envelope_rejects_unknown_fields() -> None:
    """``extra="forbid"`` on SurfacePayloadModel keeps the contract closed."""
    with pytest.raises(ValidationError):
        SessionMessagePayload(  # type: ignore[call-arg]
            id="m1",
            role="user",
            text="hi",
            mystery_field="surprise",
        )
