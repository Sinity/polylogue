"""Export members address content, not positions.

Every test here fails if reacquisition goes back to reading
``source_index`` as an array address: each one rewrites a member so the
recorded position now holds a different, equally valid conversation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.content_identity import structural_content_identity, structurally_equal
from polylogue.core.enums import Provider
from polylogue.core.json import dumps_bytes
from polylogue.core.raw_coordinates import MemberAddressingMode
from polylogue.operations.zip_acquisition_replay import (
    MemberCandidate,
    resolve_member_candidate,
    zip_reacquisition_payload,
)
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    iter_zip_entry_raw_data,
    replay_zip_entry_acquisition_payloads,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.source_write import record_raw_container_coordinate

_META = {"metadata": "bundle sibling"}


def _session(native_id: str) -> dict[str, object]:
    return {"id": native_id, "mapping": {"node": {"message": {"author": {"role": "user"}}}}}


def _write_member(path: Path, records: object, *, member: str = "conversations.json") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, json.dumps(records, separators=(",", ":")))


def _row(
    source_path: str,
    *,
    payload: bytes,
    source_index: int,
    addressing_mode: str = "",
) -> dict[str, object]:
    return {
        "coordinate_format": "",
        "entry_ordinal": None,
        "split_index": None,
        "raw_id": "legacy-raw-id",
        "source_path": source_path,
        "source_index": source_index,
        "addressing_mode": addressing_mode,
        "blob_hash": hashlib.sha256(payload).hexdigest(),
        "capture_mode": "chatgpt",
    }


def test_reordered_member_returns_the_recorded_conversation(tmp_path: Path) -> None:
    """A reordered export resolves the recorded element, not the recorded slot.

    Anti-vacuity: element 0 after the reorder is ``second``, a valid
    conversation. A positional read returns it and the assertion fails.
    """
    zip_path = tmp_path / "export.zip"
    _write_member(zip_path, [_META, _session("first"), _session("second")])
    recorded_path = f"{zip_path}:conversations.json"
    expected = dumps_bytes(_session("first"))

    _write_member(zip_path, [_META, _session("second"), _session("first")])

    payload, error = zip_reacquisition_payload(
        _row(recorded_path, payload=expected, source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert payload == expected


def test_inserted_element_shifts_the_hint_without_losing_the_conversation(tmp_path: Path) -> None:
    """An element inserted ahead of the recorded one still resolves it.

    Anti-vacuity: after the insertion the recorded slot holds ``inserted``;
    returning it would satisfy a positional reader and fail here.
    """
    zip_path = tmp_path / "export.zip"
    recorded_path = f"{zip_path}:conversations.json"
    expected = dumps_bytes(_session("kept"))
    _write_member(zip_path, [_META, _session("inserted"), _session("kept"), _session("tail")])

    payload, error = zip_reacquisition_payload(
        _row(recorded_path, payload=expected, source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert payload == expected


def test_duplicate_equal_elements_never_resolve_to_an_unrelated_conversation(tmp_path: Path) -> None:
    """Repeated equal elements are duplicate observations of one item.

    Anti-vacuity: the recorded slot now holds ``unrelated``. Returning the
    hinted slot, or refusing because two elements are equal, both fail.
    """
    zip_path = tmp_path / "export.zip"
    recorded_path = f"{zip_path}:conversations.json"
    expected = dumps_bytes(_session("duplicated"))
    _write_member(
        zip_path,
        [_META, _session("unrelated"), _session("duplicated"), _session("duplicated")],
    )

    payload, error = zip_reacquisition_payload(
        _row(recorded_path, payload=expected, source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert payload == expected

    candidates = tuple(MemberCandidate(MemberAddressingMode.ELEMENT_OF_CONTAINER, index, expected) for index in (1, 2))
    resolution = resolve_member_candidate(
        candidates,
        expected_digest=hashlib.sha256(expected).hexdigest(),
        hint_mode=None,
        hint_index=0,
    )
    assert resolution.outcome == "duplicate_observations"


def test_hinted_element_is_refused_when_its_content_is_not_the_recorded_one(tmp_path: Path) -> None:
    """A member that no longer holds the recorded value proves nothing.

    Anti-vacuity: element 0 exists and parses; accepting it because the hint
    points at it is the defect, and would make ``error`` None.
    """
    zip_path = tmp_path / "export.zip"
    recorded_path = f"{zip_path}:conversations.json"
    removed = dumps_bytes(_session("removed"))
    _write_member(zip_path, [_META, _session("survivor"), _session("other")])

    payload, error = zip_reacquisition_payload(
        _row(recorded_path, payload=removed, source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert payload is None
    assert error == "content_identity:unmatched"


@pytest.mark.parametrize("records", [_session("only"), [_META, _session("only")]])
def test_whole_member_document_is_acquired_as_a_whole_member(tmp_path: Path, records: object) -> None:
    """A member holding one session is the document, not element 0.

    Anti-vacuity: stamping ``source_index`` 0 on it -- the collapse this
    replaces -- makes the mode assertion fail. The array case is the shape
    that matters: it looks addressable by position and is not.
    """
    zip_path = tmp_path / "single.zip"
    _write_member(zip_path, records)
    blob_store = BlobStore(tmp_path / "blob")

    with zipfile.ZipFile(zip_path) as archive:
        entry = archive.infolist()[0]
        context = ZipEntryReadContext(
            source=Source(name="chatgpt", path=tmp_path),
            zip_path=zip_path,
            entry=entry,
            file_mtime=None,
            provider_hint=Provider.CHATGPT,
            blob_store=blob_store,
        )
        records = list(iter_zip_entry_raw_data(archive, context))
        replayed = list(replay_zip_entry_acquisition_payloads(archive, context))

    assert [record.addressing_mode for record in records] == [MemberAddressingMode.WHOLE_MEMBER]
    assert [record.source_index for record in records] == [None]
    assert [item.addressing_mode for item in replayed] == [MemberAddressingMode.WHOLE_MEMBER]


def test_split_member_elements_are_acquired_as_elements(tmp_path: Path) -> None:
    """A member holding several sessions yields indexed elements.

    Anti-vacuity: marking split payloads whole-member would erase the
    distinction this bead exists to draw, and fail the mode assertion.
    """
    zip_path = tmp_path / "split.zip"
    _write_member(zip_path, [_META, _session("a"), _session("b")])
    blob_store = BlobStore(tmp_path / "blob")

    with zipfile.ZipFile(zip_path) as archive:
        entry = archive.infolist()[0]
        context = ZipEntryReadContext(
            source=Source(name="chatgpt", path=tmp_path),
            zip_path=zip_path,
            entry=entry,
            file_mtime=None,
            provider_hint=Provider.CHATGPT,
            blob_store=blob_store,
        )
        records = list(iter_zip_entry_raw_data(archive, context))

    assert [record.addressing_mode for record in records] == [
        MemberAddressingMode.ELEMENT_OF_CONTAINER,
        MemberAddressingMode.ELEMENT_OF_CONTAINER,
    ]
    assert [record.source_index for record in records] == [0, 1]


def test_whole_member_hint_resolves_the_member_document(tmp_path: Path) -> None:
    """A row recorded as whole-member reads the member, not an element.

    Anti-vacuity: element resolution would look for index 0 in a member that
    yields no elements and report the blob unrecoverable.
    """
    zip_path = tmp_path / "single.zip"
    document = _session("only")
    _write_member(zip_path, document)
    recorded_path = f"{zip_path}:conversations.json"
    member_bytes = json.dumps(document, separators=(",", ":")).encode()

    payload, error = zip_reacquisition_payload(
        _row(
            recorded_path,
            payload=member_bytes,
            source_index=0,
            addressing_mode=MemberAddressingMode.WHOLE_MEMBER.value,
        ),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert payload == member_bytes


def test_corrupt_member_reports_a_typed_failure(tmp_path: Path) -> None:
    """An unreadable member leaves the reference unproven.

    Anti-vacuity: returning any payload, or raising out of the replay, both
    fail -- verification must fail closed with a reason.
    """
    zip_path = tmp_path / "corrupt.zip"
    _write_member(zip_path, [_META, _session("a"), _session("b")])
    raw = bytearray(zip_path.read_bytes())
    body = raw.index(b"PK\x03\x04") + 40
    raw[body] ^= 0xFF
    zip_path.write_bytes(bytes(raw))
    recorded_path = f"{zip_path}:conversations.json"

    payload, error = zip_reacquisition_payload(
        _row(recorded_path, payload=dumps_bytes(_session("a")), source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert payload is None
    assert error is not None
    assert error.startswith("error:") or error == "content_identity:unmatched"


def test_relocated_archive_resolves_against_the_path_in_force(tmp_path: Path) -> None:
    """A moved container resolves at the path the caller supplies.

    Anti-vacuity: resolving the recorded path would report source_missing.
    """
    original = tmp_path / "old" / "export.zip"
    original.parent.mkdir()
    relocated = tmp_path / "new" / "export.zip"
    relocated.parent.mkdir()
    _write_member(relocated, [_META, _session("kept"), _session("sibling")])
    expected = dumps_bytes(_session("kept"))
    recorded_path = f"{original}:conversations.json"
    resolved_path = f"{relocated}:conversations.json"

    payload, error = zip_reacquisition_payload(
        _row(recorded_path, payload=expected, source_index=0),
        source_path=resolved_path,
        zip_payload_cache={},
    )

    assert error is None
    assert payload == expected


def test_equal_content_without_a_recorded_digest_is_one_logical_item() -> None:
    """Repeated equal elements resolve; genuinely different content does not.

    Anti-vacuity: returning a payload for the mixed member, or refusing the
    equal one, both fail. Byte-identical candidates make this vacuous, so the
    equal pair differs only in serialization.
    """
    equal = (
        MemberCandidate(MemberAddressingMode.ELEMENT_OF_CONTAINER, 0, b'{"a": 1, "b": 2}'),
        MemberCandidate(MemberAddressingMode.ELEMENT_OF_CONTAINER, 1, b'{"b":2.0,"a":1.0}'),
    )
    resolution = resolve_member_candidate(equal, expected_digest=None, hint_mode=None, hint_index=0)
    assert resolution.outcome == "duplicate_observations"
    assert resolution.error is None

    mixed = (
        equal[0],
        MemberCandidate(MemberAddressingMode.ELEMENT_OF_CONTAINER, 1, b'{"a": 1, "b": 3}'),
    )
    refused = resolve_member_candidate(mixed, expected_digest=None, hint_mode=None, hint_index=0)
    assert refused.payload_bytes is None
    assert refused.error == "content_identity:unavailable"


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (1, 1.0, True),
        ({"n": 1}, {"n": 1.0}, True),
        ([1, 2], [1.0, 2.0], True),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        (1, "1", False),
        (True, 1, False),
        (False, 0, False),
        (None, 0, False),
        (1.5, 1, False),
        ([1, 2], [2, 1], False),
    ],
)
def test_structural_identity_follows_the_provider_value_contract(left: object, right: object, equal: bool) -> None:
    """Integral numbers are one number; JSON's own type distinctions survive.

    Anti-vacuity: a digest over serialized bytes fails the ``1``/``1.0`` and
    key-order rows; a digest that coerces types fails the ``True``/``1`` and
    ``1``/``"1"`` rows.
    """
    assert structurally_equal(left, right) is equal
    assert (structural_content_identity(left) == structural_content_identity(right)) is equal


def test_recorded_addressing_mode_survives_a_round_trip(tmp_path: Path) -> None:
    """The source tier stores the reading, not just the coordinate.

    Anti-vacuity: dropping the column, or defaulting it, makes the stored
    value differ from the acquired one.
    """
    db_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SOURCE_DDL)
        conn.execute(
            """
            INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-1', 'chatgpt-export', 'export.zip:conversations.json', 0, zeroblob(32), 1, 1)
            """
        )
        record_raw_container_coordinate(
            conn,
            "raw-1",
            coordinate_format="zip-v2",
            entry_ordinal=0,
            split_index=0,
            addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
            manage_transaction=False,
        )
        stored = conn.execute("SELECT addressing_mode FROM raw_container_coordinates WHERE raw_id = 'raw-1'").fetchone()
        assert stored[0] == MemberAddressingMode.WHOLE_MEMBER.value

        with pytest.raises(ValueError, match="addressing_mode"):
            record_raw_container_coordinate(
                conn,
                "raw-1",
                coordinate_format="zip-v2",
                entry_ordinal=0,
                split_index=0,
                addressing_mode="element",
                manage_transaction=False,
            )
