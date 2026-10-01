"""Content identity never merges distinct admitted content.

Three preimage merges, each through the real hash functions and the archive
writer (polylogue-vp5qk, polylogue-sf7ii, polylogue-aki9t):

- absence and emptiness must stay disjoint from every literal string,
  including the markers the hash once substituted for them;
- mapping keys that are canonically equivalent Unicode stay distinct slots,
  so no field drops and key insertion order cannot move the hash;
- operational strings (tool arguments, paths, names) hash exactly, while
  declared prose keeps its NFC equivalence.
"""

from __future__ import annotations

import itertools
import sqlite3
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline.ids import (
    idless_session_identity,
    message_content_identity,
    session_content_hash,
    session_revision_projection,
)
from polylogue.sources.parsers.base_models import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.sqlite.archive_tiers.write import ArchiveWriteOutcome
from tests.infra.prepared_session import write_prepared_session

_NULL_LITERAL = "__POLYLOGUE_NULL__"
_EMPTY_LITERAL = "__POLYLOGUE_EMPTY__"
_NFC = "café"
_NFD = "café"
#: Three distinct keys that all NFC-normalize to "Å".
_ANGSTROMS = ("Å", "Å", "Å")


def _session(
    *,
    text: str | None = "unchanged",
    tool_input: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    title: str | None = "title",
    events: list[ParsedSessionEvent] | None = None,
    attachments: list[ParsedAttachment] | None = None,
    provider_message_id: str = "m1",
    timestamp: str | None = "2026-01-01T00:00:00Z",
    instructions_text: str | None = None,
    git_branch: str | None = None,
    working_directories: list[str] | None = None,
) -> ParsedSession:
    blocks = []
    if tool_input is not None or metadata is not None:
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="read_file",
                tool_id="call-1",
                tool_input=tool_input,
                metadata=metadata,
            )
        )
    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="identity-contract",
        title=title,
        messages=[
            ParsedMessage(
                provider_message_id=provider_message_id,
                role=Role.ASSISTANT,
                text=text,
                timestamp=timestamp,
                blocks=blocks,
            )
        ],
        session_events=events or [],
        attachments=attachments or [],
        instructions_text=instructions_text,
        git_branch=git_branch,
        working_directories=working_directories or [],
    )


def _event(payload: dict[str, object]) -> list[ParsedSessionEvent]:
    return [ParsedSessionEvent(event_type="turn_context", timestamp="2026-01-01T00:00:01Z", payload=payload)]


def _assert_distinct(left: ParsedSession, right: ParsedSession, *, axis: str = "message") -> None:
    """Distinct sessions differ in hash and on the revision axis that carries the change."""
    assert session_content_hash(left) != session_content_hash(right)
    left_projection = session_revision_projection(left)
    right_projection = session_revision_projection(right)
    assert left_projection.session_hash != right_projection.session_hash
    if axis == "message":
        assert left_projection.message_contents != right_projection.message_contents
    elif axis == "event":
        assert left_projection.event_contents != right_projection.event_contents


def _assert_same(left: ParsedSession, right: ParsedSession) -> None:
    assert session_content_hash(left) == session_content_hash(right)
    assert message_content_identity(left.messages[0]) == message_content_identity(right.messages[0])
    left_projection = session_revision_projection(left)
    right_projection = session_revision_projection(right)
    assert left_projection.message_contents == right_projection.message_contents
    assert left_projection.event_contents == right_projection.event_contents


# -- absence is not a literal (polylogue-vp5qk) -------------------------------

_ABSENT_VALUES = pytest.mark.parametrize(("absent", "literal"), [(None, _NULL_LITERAL), ("", _EMPTY_LITERAL)])


@_ABSENT_VALUES
def test_absent_message_text_is_not_a_literal(absent: str | None, literal: str) -> None:
    left, right = _session(text=absent), _session(text=literal)
    _assert_distinct(left, right)
    assert message_content_identity(left.messages[0]) != message_content_identity(right.messages[0])


@_ABSENT_VALUES
def test_absent_title_is_not_a_literal(absent: str | None, literal: str) -> None:
    _assert_distinct(_session(title=absent), _session(title=literal), axis="session")


@_ABSENT_VALUES
@pytest.mark.parametrize(
    "shape",
    [
        lambda value: {"value": value},
        lambda value: {"outer": {"inner": value}},
        lambda value: {"items": ["a", value, "b"]},
    ],
    ids=["top-level", "nested-object", "list-member"],
)
def test_absent_tool_argument_is_not_a_literal(
    absent: str | None, literal: str, shape: Callable[[str | None], dict[str, object]]
) -> None:
    left = _session(tool_input=shape(absent))
    right = _session(tool_input=shape(literal))
    _assert_distinct(left, right)
    assert message_content_identity(left.messages[0]) != message_content_identity(right.messages[0])


@_ABSENT_VALUES
def test_absent_block_metadata_value_is_not_a_literal(absent: str | None, literal: str) -> None:
    _assert_distinct(_session(metadata={"ref": absent}), _session(metadata={"ref": literal}))


@_ABSENT_VALUES
def test_absent_event_payload_value_is_not_a_literal(absent: str | None, literal: str) -> None:
    _assert_distinct(_session(events=_event({"cwd": absent})), _session(events=_event({"cwd": literal})), axis="event")


@_ABSENT_VALUES
def test_idless_message_anchor_does_not_equate_absence_with_a_literal(absent: str | None, literal: str) -> None:
    """The revision match id of a timestamp-less id-less message hashes its text."""
    left = _session(text=absent, provider_message_id="", timestamp=None)
    right = _session(text=literal, provider_message_id="", timestamp=None)
    _assert_distinct(left, right)


@_ABSENT_VALUES
def test_idless_session_identity_does_not_equate_absence_with_a_literal(absent: str | None, literal: str) -> None:
    def identity(text: str | None) -> str:
        return idless_session_identity(first_message_provider_id=None, first_message_text=text, created_at=None)

    assert identity(absent) != identity(literal)


def test_null_and_empty_stay_distinct_from_each_other() -> None:
    _assert_distinct(_session(text=None), _session(text=""))
    _assert_distinct(_session(tool_input={"value": None}), _session(tool_input={"value": ""}))
    _assert_distinct(_session(tool_input={"value": None}), _session(tool_input={"value": "null"}))


def _attachment(mime_type: str | None, size: int) -> ParsedAttachment:
    return ParsedAttachment(
        provider_attachment_id="a", message_provider_id="m1", name="same.txt", mime_type=mime_type, size_bytes=size
    )


def test_attachment_order_is_not_content_when_sort_fields_tie() -> None:
    """Attachments that share owner, id and name hash alike in either input order.

    Anti-vacuity: drop the whole-payload tiebreak from ``_attachment_sort_key``
    and the tie falls back to input order, so the two hashes differ.
    """
    forward = _session(attachments=[_attachment("text/plain", 1), _attachment("text/plain", 2)])
    backward = _session(attachments=[_attachment("text/plain", 2), _attachment("text/plain", 1)])
    assert session_content_hash(forward) == session_content_hash(backward)
    assert session_revision_projection(forward).session_hash == session_revision_projection(backward).session_hash
    changed = _session(attachments=[_attachment("text/plain", 1), _attachment("text/plain", 3)])
    assert session_content_hash(forward) != session_content_hash(changed)


def test_absent_attachment_field_is_not_an_empty_one() -> None:
    _assert_distinct(
        _session(attachments=[_attachment(None, 1)]), _session(attachments=[_attachment("", 1)]), axis="session"
    )


# -- colliding keys keep every field (polylogue-sf7ii) -------------------------


def _nest(payload: dict[str, object], depth: int) -> dict[str, object]:
    for _ in range(depth):
        payload = {"level": payload}
    return payload


@pytest.mark.parametrize("keys", [_ANGSTROMS[:2], _ANGSTROMS], ids=["two-keys", "three-keys"])
@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("carrier", ["tool_input", "metadata", "event"])
def test_equivalent_keys_are_order_independent_and_lossless(keys: tuple[str, ...], depth: int, carrier: str) -> None:
    def build(order: tuple[str, ...], values: dict[str, str]) -> ParsedSession:
        payload = _nest({key: values[key] for key in order}, depth)
        if carrier == "event":
            return _session(events=_event(payload))
        if carrier == "metadata":
            return _session(metadata=payload)
        return _session(tool_input=payload)

    axis = "event" if carrier == "event" else "message"

    values = {key: f"value-{index}" for index, key in enumerate(keys)}
    orders = list(itertools.permutations(keys))
    baseline = build(orders[0], values)
    for order in orders[1:]:
        _assert_same(baseline, build(order, values))
    # Dropping any one spelling, or letting its value win the NFC slot, is a
    # different payload.
    for dropped in keys:
        survivors = tuple(key for key in keys if key != dropped)
        _assert_distinct(baseline, build(survivors, values), axis=axis)
    collapsed = dict.fromkeys(keys[:1], values[keys[-1]])
    _assert_distinct(baseline, build(keys[:1], collapsed), axis=axis)


def test_ascii_key_order_stays_harmless() -> None:
    _assert_same(_session(tool_input={"a": 1, "b": 2}), _session(tool_input={"b": 2, "a": 1}))


# -- operational strings hash exactly (polylogue-aki9t) -----------------------


def test_equivalent_paths_name_two_files_and_two_tool_calls(tmp_path: Path) -> None:
    for name, content in ((f"{_NFC}.txt", "LEFT"), (f"{_NFD}.txt", "RIGHT")):
        (tmp_path / name).write_text(content)
    if len(list(tmp_path.iterdir())) != 2:
        pytest.skip("this filesystem normalizes file names, so the two paths name one file")
    assert (tmp_path / f"{_NFC}.txt").read_text() != (tmp_path / f"{_NFD}.txt").read_text()

    left = _session(tool_input={"path": str(tmp_path / f"{_NFC}.txt")})
    right = _session(tool_input={"path": str(tmp_path / f"{_NFD}.txt")})
    _assert_distinct(left, right)
    assert message_content_identity(left.messages[0]) != message_content_identity(right.messages[0])


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (_session(tool_input={"command": f"cat {_NFC}"}), _session(tool_input={"command": f"cat {_NFD}"})),
        (_session(tool_input={_NFC: "x"}), _session(tool_input={_NFD: "x"})),
        (_session(metadata={"file": _NFC}), _session(metadata={"file": _NFD})),
        (_session(events=_event({"cwd": _NFC})), _session(events=_event({"cwd": _NFD}))),
        (_session(git_branch=_NFC), _session(git_branch=_NFD)),
        (_session(working_directories=[_NFC]), _session(working_directories=[_NFD])),
        (
            _session(attachments=[ParsedAttachment(provider_attachment_id="a", message_provider_id="m1", name=_NFC)]),
            _session(attachments=[ParsedAttachment(provider_attachment_id="a", message_provider_id="m1", name=_NFD)]),
        ),
    ],
    ids=["command", "argument-key", "metadata", "event-payload", "git-branch", "working-directory", "attachment-name"],
)
def test_operational_strings_hash_exactly(left: ParsedSession, right: ParsedSession) -> None:
    assert session_content_hash(left) != session_content_hash(right)
    assert session_revision_projection(left).session_hash != session_revision_projection(right).session_hash


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (_session(text=_NFC), _session(text=_NFD)),
        (_session(title=_NFC), _session(title=_NFD)),
        (_session(instructions_text=_NFC), _session(instructions_text=_NFD)),
    ],
    ids=["message-text", "title", "instructions"],
)
def test_declared_prose_keeps_its_nfc_equivalence(left: ParsedSession, right: ParsedSession) -> None:
    _assert_same(left, right)


def test_prose_nfc_equivalence_reaches_idless_anchors() -> None:
    _assert_same(
        _session(text=_NFC, provider_message_id="", timestamp=None),
        _session(text=_NFD, provider_message_id="", timestamp=None),
    )
    assert unicodedata.normalize("NFC", _NFD) == _NFC


# -- the archive writer stores what the identity distinguishes ----------------


def _write(db_path: Path, session: ParsedSession) -> dict[str, object]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    outcomes: list[ArchiveWriteOutcome] = []
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        session_id = write_prepared_session(
            conn, session, content_hash=session_content_hash(session), write_outcome=outcomes
        )
        conn.commit()
        stored = conn.execute("SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        message = conn.execute(
            "SELECT content_address FROM messages WHERE session_id = ? ORDER BY position", (session_id,)
        ).fetchone()
        tool_inputs = [
            row[0]
            for row in conn.execute(
                "SELECT tool_input FROM blocks WHERE session_id = ? AND tool_input IS NOT NULL", (session_id,)
            )
        ]
    finally:
        conn.close()
    assert outcomes and outcomes[0].wrote
    return {"hash": bytes(stored[0]), "address": bytes(message[0]), "tool_inputs": tool_inputs}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_session(tool_input={"path": f"{_NFC}.txt"}), _session(tool_input={"path": f"{_NFD}.txt"})),
        (_session(tool_input={"value": None}), _session(tool_input={"value": _NULL_LITERAL})),
        (_session(text=None), _session(text=_NULL_LITERAL)),
        (_session(text=None), _session(text="")),
        (
            _session(tool_input={_ANGSTROMS[0]: "left", _ANGSTROMS[1]: "right"}),
            _session(tool_input={_ANGSTROMS[0]: "right"}),
        ),
    ],
    ids=["equivalent-path", "null-argument", "null-text", "null-vs-empty-text", "colliding-key"],
)
def test_rewrite_of_distinct_content_moves_the_stored_identity(
    test_db: Path, first: ParsedSession, second: ParsedSession
) -> None:
    """Both acquisitions reach storage, and the stored identities tell them apart.

    ``sessions.content_hash`` gates idempotent re-ingest and
    ``messages.content_address`` witnesses a branch point; either one equal
    across distinct content lets the second acquisition pass as the first.
    """
    before = _write(test_db, first)
    after = _write(test_db, second)
    assert before["hash"] == bytes.fromhex(session_content_hash(first))
    assert after["hash"] == bytes.fromhex(session_content_hash(second))
    assert before["hash"] != after["hash"]
    assert before["address"] != after["address"]
    if second.messages[0].blocks:
        assert after["tool_inputs"] != before["tool_inputs"]


def test_prose_rewrite_keeps_the_stored_identity(test_db: Path) -> None:
    before = _write(test_db, _session(text=_NFC))
    after = _write(test_db, _session(text=_NFD))
    assert before["hash"] == after["hash"]
    assert before["address"] == after["address"]
