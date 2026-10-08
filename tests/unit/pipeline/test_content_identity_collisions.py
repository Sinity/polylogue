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
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
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
    ParsedFileEdit,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.write import ArchiveWriteOutcome
from tests.infra.index_writer import close_fixture_index_connection, write_fixture_index_session

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
    file_edit: ParsedFileEdit | None = None,
    pending_drafts: list[dict[str, object]] | None = None,
) -> ParsedSession:
    blocks = []
    if tool_input is not None or metadata is not None or file_edit is not None:
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="read_file",
                tool_id="call-1",
                tool_input=tool_input,
                metadata=metadata,
                file_edit=file_edit,
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
        pending_drafts=pending_drafts or [],
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


@pytest.mark.parametrize("carrier", ["tool_input", "metadata", "event", "file_edit", "pending_drafts"])
def test_nested_nonstring_mapping_keys_keep_typed_associations(carrier: str) -> None:
    first: dict[object, object] = {1: "integer", "1": "string"}
    second: dict[object, object] = {"1": "string", 1: "integer"}

    def build(mapping: dict[object, object]) -> ParsedSession:
        if carrier == "event":
            return _session(events=_event({"nested": mapping}))
        if carrier == "file_edit":
            return _session(file_edit=ParsedFileEdit(structured_patch=[{"nested": mapping}]))
        if carrier == "pending_drafts":
            return _session(pending_drafts=[{"nested": mapping}])
        if carrier == "metadata":
            return _session(metadata={"nested": mapping})
        return _session(tool_input={"nested": mapping})

    baseline = build(first)
    _assert_same(baseline, build(second))
    # The legacy QUERY lowering stringifies 1 to "1" and used to drop one
    # value.  The one typed sidecar now keeps both associations in all carriers.
    axis = "event" if carrier == "event" else ("session" if carrier == "pending_drafts" else "message")
    _assert_distinct(baseline, build({1: "integer"}), axis=axis)


@pytest.mark.parametrize(
    "key,other_key,value,other",
    [
        (b"path", "b'path'", "bytes-key", "string-key"),
        (date(2026, 1, 2), "2026-01-02", "date-key", "string-key"),
        ("cafe\u0301", "café", "NFD-key", "NFC-key"),
    ],
    ids=["bytes-key", "date-key", "exact-nfd-key"],
)
def test_nested_mapping_keys_preserve_runtime_type_and_exact_spelling(
    key: object, other_key: str, value: str, other: str
) -> None:
    payload: dict[str, object] = {"nested": {key: value, other_key: other}}
    forward = _session(tool_input=payload)
    reverse = _session(tool_input={"nested": {other_key: other, key: value}})
    _assert_same(forward, reverse)
    _assert_distinct(forward, _session(tool_input={"nested": {key: value}}))


@pytest.mark.parametrize("field", ["tool_input", "metadata", "event_payload", "structured_patch", "pending_drafts"])
def test_declared_string_key_maps_reject_before_pydantic_can_drop_bytes_key(field: str) -> None:
    collision = {b"a": "bytes", "a": "string"}
    with pytest.raises(ValueError, match="string mapping keys"):
        if field in {"tool_input", "metadata"}:
            ParsedContentBlock.model_validate({"type": BlockType.TOOL_USE, "tool_name": "read_file", field: collision})
        elif field == "event_payload":
            ParsedSessionEvent.model_validate({"event_type": "turn_context", "payload": collision})
        elif field == "structured_patch":
            ParsedFileEdit.model_validate({"structured_patch": [collision]})
        else:
            ParsedSession.model_validate(
                {
                    "source_name": Provider.CHATGPT,
                    "provider_session_id": "identity-contract",
                    "pending_drafts": [collision],
                }
            )


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


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            ParsedMessage(
                provider_message_id="",
                role=Role.USER,
                text="ordinary",
                timestamp="2026-01-02T03:04:05Z",
            ),
            "94acc691e6494c71bb4a3181b22f6c33",
        ),
        (
            ParsedMessage(
                provider_message_id="",
                role=Role.ASSISTANT,
                text=None,
                timestamp=None,
                user_context_text="",
            ),
            "db98315063e94bc832168f0c6031c3f0",
        ),
        (
            ParsedMessage(
                provider_message_id="",
                role=Role.ASSISTANT,
                text="A message",
                timestamp="2026-01-02T03:04:05Z",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="read_file",
                        tool_id="call-7",
                        tool_input={"path": "/tmp/example", "flags": ["x", None]},
                        metadata={"note": "ordinary"},
                    )
                ],
            ),
            "51bcefa015c2fceef3a4a458fddb1038",
        ),
    ],
    ids=["plain", "null-empty", "nested-operational-fields"],
)
def test_ordinary_idless_message_ids_keep_pre_a44_vectors(message: ParsedMessage, expected: str) -> None:
    """Ordinary durable anchors keep their shipped digest preimages."""
    assert message_content_identity(message) == expected


def test_marker_shaped_mapping_key_keeps_its_noncolliding_pre_a44_id() -> None:
    message = ParsedMessage(
        provider_message_id="",
        role=Role.ASSISTANT,
        text="anchor",
        timestamp=None,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_input={"__POLYLOGUE_NULL__": "value"},
                tool_id="tool",
                tool_name="read_file",
            )
        ],
    )
    # Computed by the shipped pre-a44 encoder, not the candidate helper.
    assert message_content_identity(message) == "622c8a6196d308c682ca401518fa82b7"


def test_recursive_marker_literals_and_empty_lists_remain_disjoint() -> None:
    def id_for(*, payload: dict[str, object], block: bool = True) -> str:
        message = ParsedMessage(
            provider_message_id="",
            role=Role.ASSISTANT,
            text="anchor",
            timestamp=None,
            blocks=(
                [
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="call",
                        tool_input=payload,
                    )
                ]
                if block
                else []
            ),
        )
        return message_content_identity(message)

    assert id_for(payload={"outer": {"value": None}}) != id_for(payload={"outer": {"value": _NULL_LITERAL}})
    assert id_for(payload={"outer": {"value": ""}}) != id_for(payload={"outer": {"value": _EMPTY_LITERAL}})
    assert id_for(payload={"outer": {"__POLYLOGUE_NULL__": "value"}}) != id_for(
        payload={"outer": {"__POLYLOGUE_NULL__": "different"}}
    )
    assert id_for(payload={"items": []}) != id_for(payload={"items": _EMPTY_LITERAL})
    assert id_for(payload={}, block=False) != id_for(payload={"items": []})


def test_lossy_identity_lowerings_are_injective_and_declared_equivalences_remain() -> None:
    def id_for(value: object) -> str:
        message = ParsedMessage(
            provider_message_id="",
            role=Role.ASSISTANT,
            text="anchor",
            timestamp=None,
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
                    tool_name="call",
                    tool_input={"value": value},
                )
            ],
        )
        return message_content_identity(message)

    marker = _NULL_LITERAL
    first_set = {
        frozenset({("a", None), ("b", "x")}),
        frozenset({("a", "y"), ("b", marker)}),
    }
    second_set = {
        frozenset({("a", marker), ("b", "x")}),
        frozenset({("a", "y"), ("b", None)}),
    }
    assert id_for(first_set) != id_for(second_set)

    # Bytes and temporal values have a declared lowering, but are not equal
    # to arbitrary user strings that happen to use that serialized spelling.
    assert id_for(b"a") != id_for(b"61")
    assert id_for(datetime(2026, 1, 2, 3, 4, 5)) != id_for("2026-01-02T03:04:05")
    assert id_for(Decimal("0.123456789012345678901")) != id_for({"$decimal": "0.123456789012345678901"})
    assert session_content_hash(_session(tool_input={"value": b"a"})) != session_content_hash(
        _session(tool_input={"value": "61"})
    )
    assert session_content_hash(_session(tool_input={"value": datetime(2026, 1, 2)})) != session_content_hash(
        _session(tool_input={"value": "2026-01-02T00:00:00"})
    )

    class WireToken(str, Enum):
        VALUE = "wire-value"

    # The established parser-boundary equivalences remain deliberate.
    assert id_for(WireToken.VALUE) == id_for("wire-value")
    assert id_for(("a", "b")) == id_for(["a", "b"])
    assert id_for({"a", "b"}) == id_for(["a", "b"])
    assert id_for(Decimal("0.5")) == id_for(0.5)


# -- the archive writer stores what the identity distinguishes ----------------


def _write(db_path: Path, session: ParsedSession) -> dict[str, object]:
    conn = connect_measured(db_path)
    conn.row_factory = sqlite3.Row
    outcomes: list[ArchiveWriteOutcome] = []
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        session_id = write_fixture_index_session(
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
        close_fixture_index_connection(conn)
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
