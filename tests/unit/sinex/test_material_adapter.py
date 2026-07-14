"""session_material_from_session against the REAL v1 encoder.

Anti-vacuity: this test feeds the adapter's output into
``polylogue.material_protocol.v1.encode_session_revision`` (the real,
already-shipped production encoder from polylogue-303r.1) and decodes the
result back. Breaking the adapter's field mapping (wrong role/text/native_id/
origin) makes the round-trip assertions fail, not just a mock's call count.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.material_protocol.v1 import decode_session_revision, encode_session_revision
from polylogue.sinex.material_adapter import session_material_from_session


def _real_session() -> Session:
    messages = [
        Message(
            id="claude-code-session:s1:0",
            role=Role.USER,
            text="hello there",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        Message(
            id="claude-code-session:s1:1",
            role=Role.ASSISTANT,
            text="hi!",
            timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            blocks=[
                {"type": "text", "text": "hi!"},
                {"type": "tool_use", "tool_name": "Bash", "tool_id": "tool-1", "tool_input": {"command": "ls"}},
                {"type": "not-a-real-block-type"},  # must be dropped, not raise
            ],
        ),
    ]
    return Session(
        id=SessionId("claude-code-session:s1"),
        origin=Origin.CLAUDE_CODE_SESSION,
        title="Adapter fixture session",
        messages=MessageCollection(messages=messages),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        metadata={"nested": {"a": 1}, "flag": True},
        tags_m2m=("dogfood",),
    )


def test_adapter_output_round_trips_through_the_real_v1_encoder() -> None:
    material = session_material_from_session(_real_session())

    assert material.origin is Origin.CLAUDE_CODE_SESSION
    assert material.native_id == "s1"
    assert len(material.messages) == 2
    assert material.messages[0].text == "hello there"
    assert material.messages[1].blocks[0].block_type.value == "text"
    assert material.messages[1].blocks[1].tool_name == "Bash"
    # The invalid block type was silently dropped, not raised or mis-mapped.
    assert len(material.messages[1].blocks) == 2
    assert len(material.fidelity_gaps) == 1
    assert material.fidelity_gaps[0].gap_kind == "omitted_relation"

    encoded = encode_session_revision(material, revision_created_at="2026-01-01T00:00:02+00:00")
    assert encoded.manifest.session_id == "claude-code-session:s1"
    assert encoded.manifest.origin == "claude-code-session"
    assert encoded.manifest.native_id == "s1"

    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    assert decoded.session["session_id"] == "claude-code-session:s1"
    assert [m.text for m in decoded.messages] == ["hello there", "hi!"]


def test_adapter_rejects_a_malformed_session_id() -> None:
    session = Session(
        id=SessionId("not-well-formed"),
        origin=Origin.CLAUDE_CODE_SESSION,
        messages=MessageCollection(messages=[]),
    )
    with pytest.raises(ValueError, match="well-formed"):
        session_material_from_session(session)
