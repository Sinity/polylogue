"""Export members address content, not positions.

Every test here fails if reacquisition goes back to reading
``source_index`` as an array address: each one rewrites a member so the
recorded position now holds a different, equally valid conversation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import sqlite3
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import pytest

from polylogue.archive.zip_admission import MAX_COMPRESSION_RATIO
from polylogue.config import Source
from polylogue.core.content_identity import structural_content_identity, structurally_equal
from polylogue.core.enums import Provider
from polylogue.core.json import dumps_bytes
from polylogue.core.raw_coordinates import MemberAddressingMode, zip_member_container, zip_member_coordinate
from polylogue.operations.zip_acquisition_replay import (
    MemberCandidate,
    resolve_member_candidate,
    zip_reacquired_unit,
)
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    iter_zip_entry_raw_data,
    replay_zip_entry_acquisition_revisions,
    stream_preserved_zip_entry_raw_data,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.source_write import record_raw_container_coordinate

_META = {"metadata": "bundle sibling"}


def _session(native_id: str) -> dict[str, object]:
    return {"id": native_id, "mapping": {"node": {"message": {"author": {"role": "user"}}}}}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _candidate(mode: MemberAddressingMode, index: int | None, payload: bytes) -> MemberCandidate:
    return MemberCandidate(mode, index, structural_content_identity(json.loads(payload)), _sha(payload), len(payload))


def _write_member(path: Path, records: object, *, member: str = "conversations.json") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, json.dumps(records, separators=(",", ":")))


def _row(
    source_path: str,
    *,
    payload: bytes,
    source_index: int | None,
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

    unit, error = zip_reacquired_unit(
        _row(recorded_path, payload=expected, source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert unit is not None and unit.byte_identity == _sha(expected)


def test_inserted_element_shifts_the_hint_without_losing_the_conversation(tmp_path: Path) -> None:
    """An element inserted ahead of the recorded one still resolves it.

    Anti-vacuity: after the insertion the recorded slot holds ``inserted``;
    returning it would satisfy a positional reader and fail here.
    """
    zip_path = tmp_path / "export.zip"
    recorded_path = f"{zip_path}:conversations.json"
    expected = dumps_bytes(_session("kept"))
    _write_member(zip_path, [_META, _session("inserted"), _session("kept"), _session("tail")])

    unit, error = zip_reacquired_unit(
        _row(recorded_path, payload=expected, source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert unit is not None and unit.byte_identity == _sha(expected)


def test_reacquisition_refuses_a_member_acquisition_admission_rejects(tmp_path: Path) -> None:
    """Replay applies acquisition's ZIP admission before decompressing a member.

    The same member bytes replay from an archive that stores them plainly.
    Stored at a compression ratio above the admission limit, acquisition
    refuses the member, so replay refuses it too and caches nothing.
    Anti-vacuity: without the admission check the high-ratio archive yields
    the recorded payload.
    """
    padded = {**_session("padded"), "pad": " " * 1_000_000}
    member_bytes = json.dumps([_META, padded, _session("other")], separators=(",", ":")).encode()
    expected = dumps_bytes(padded)
    stored_zip = tmp_path / "stored.zip"
    high_ratio_zip = tmp_path / "high-ratio.zip"
    with zipfile.ZipFile(stored_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("conversations.json", member_bytes)
    with zipfile.ZipFile(high_ratio_zip, "w", compression=zipfile.ZIP_BZIP2) as archive:
        archive.writestr("conversations.json", member_bytes)
    with zipfile.ZipFile(high_ratio_zip) as archive:
        entry = archive.infolist()[0]
    assert entry.file_size / entry.compress_size > MAX_COMPRESSION_RATIO

    stored_path = f"{stored_zip}:conversations.json"
    unit, error = zip_reacquired_unit(
        _row(stored_path, payload=expected, source_index=0),
        source_path=stored_path,
        zip_payload_cache={},
    )
    assert error is None
    assert unit is not None and unit.byte_identity == _sha(expected)

    high_ratio_path = f"{high_ratio_zip}:conversations.json"
    cache: dict[str, tuple[MemberCandidate, ...]] = {}
    assert zip_reacquired_unit(
        _row(high_ratio_path, payload=expected, source_index=0),
        source_path=high_ratio_path,
        zip_payload_cache=cache,
    ) == (None, "container_member_rejected")
    assert cache == {}


def test_reacquisition_accepts_structural_identity_after_reserialization(tmp_path: Path) -> None:
    """A provider's harmless JSON rewrite does not lose a member."""
    zip_path = tmp_path / "rewritten.zip"
    recorded_path = f"{zip_path}:conversations.json"
    expected_value = {**_session("kept"), "ordinal": 1}
    _write_member(zip_path, [_META, {**_session("kept"), "ordinal": 1.0}, _session("other")])

    unit, error = zip_reacquired_unit(
        {
            **_row(recorded_path, payload=b"old", source_index=0),
            "content_identity": structural_content_identity(expected_value),
            "addressing_mode": MemberAddressingMode.ELEMENT_OF_CONTAINER.value,
        },
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert unit is not None
    assert unit.content_identity == structural_content_identity(expected_value)


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

    unit, error = zip_reacquired_unit(
        _row(recorded_path, payload=expected, source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert unit is not None and unit.byte_identity == _sha(expected)

    candidates = tuple(_candidate(MemberAddressingMode.ELEMENT_OF_CONTAINER, index, expected) for index in (1, 2))
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

    unit, error = zip_reacquired_unit(
        _row(recorded_path, payload=removed, source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert unit is None
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
        replayed = list(replay_zip_entry_acquisition_revisions(archive, context))

    assert [record.addressing_mode for record in records] == [MemberAddressingMode.WHOLE_MEMBER]
    assert [record.source_index for record in records] == [None]
    assert [item.addressing_mode for item in replayed] == [MemberAddressingMode.WHOLE_MEMBER]


def test_preserved_whole_member_drops_element_index_hint(tmp_path: Path) -> None:
    """A transport coordinate must not become a positional member address."""
    zip_path = tmp_path / "preserved.zip"
    _write_member(zip_path, {"metadata": "document"})
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
        record = stream_preserved_zip_entry_raw_data(
            archive,
            context,
            provider_hint=Provider.CHATGPT,
            source_index=17,
        )

    assert record.addressing_mode is MemberAddressingMode.WHOLE_MEMBER
    assert record.source_index is None


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


def test_split_elements_persist_structural_identity_for_replay(tmp_path: Path) -> None:
    """Production split rows use value identity, not their serialization bytes.

    Anti-vacuity: the member is rewritten with different JSON separators and
    key order after acquisition. A byte-hash producer leaves the durable
    identity unable to match the replay candidate even though the value is
    unchanged.
    """
    zip_path = tmp_path / "structural-split.zip"
    first = {**_session("one"), "ordinal": 1}
    second = {**_session("two"), "ordinal": 2}
    _write_member(zip_path, [first, second])
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

    assert [record.content_identity for record in records] == [
        structural_content_identity(first),
        structural_content_identity(second),
    ]


def test_whole_member_hint_resolves_the_member_document(tmp_path: Path) -> None:
    """A legacy NULL index resolves the preserved whole member at index zero.

    Anti-vacuity: returning ``None`` for the legacy split index would refuse
    the valid member before reopening the ZIP.
    """
    zip_path = tmp_path / "single.zip"
    document = _session("only")
    _write_member(zip_path, document)
    recorded_path = f"{zip_path}:conversations.json"
    member_bytes = json.dumps(document, separators=(",", ":")).encode()

    unit, error = zip_reacquired_unit(
        _row(
            recorded_path,
            payload=member_bytes,
            source_index=None,
            addressing_mode="",
        ),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert error is None
    assert unit is not None and unit.byte_identity == _sha(member_bytes)


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

    unit, error = zip_reacquired_unit(
        _row(recorded_path, payload=dumps_bytes(_session("a")), source_index=0),
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert unit is None
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

    unit, error = zip_reacquired_unit(
        _row(recorded_path, payload=expected, source_index=0),
        source_path=resolved_path,
        zip_payload_cache={},
    )

    assert error is None
    assert unit is not None and unit.byte_identity == _sha(expected)


def test_equal_content_without_a_recorded_digest_is_one_logical_item() -> None:
    """Repeated equal elements resolve; genuinely different content does not.

    Anti-vacuity: returning a payload for the mixed member, or refusing the
    equal one, both fail. Byte-identical candidates make this vacuous, so the
    equal pair differs only in serialization.
    """
    equal = (
        _candidate(MemberAddressingMode.ELEMENT_OF_CONTAINER, 0, b'{"a": 1, "b": 2}'),
        _candidate(MemberAddressingMode.ELEMENT_OF_CONTAINER, 1, b'{"b":2.0,"a":1.0}'),
    )
    resolution = resolve_member_candidate(equal, expected_digest=None, hint_mode=None, hint_index=0)
    assert resolution.outcome == "duplicate_observations"
    assert resolution.error is None

    mixed = (
        equal[0],
        _candidate(MemberAddressingMode.ELEMENT_OF_CONTAINER, 1, b'{"a": 1, "b": 3}'),
    )
    refused = resolve_member_candidate(mixed, expected_digest=None, hint_mode=None, hint_index=0)
    assert refused.candidate is None
    assert refused.error == "content_identity:unavailable"


def test_replay_uses_structural_digest_when_serialization_changes() -> None:
    """The durable identity is value-structural, not the retained byte hash.

    Anti-vacuity: the candidate has different key order and integral numeric
    spelling, so a raw ``blob_hash`` comparison cannot resolve it.
    """
    expected = {"id": "kept", "ordinal": 1}
    candidate = _candidate(
        MemberAddressingMode.ELEMENT_OF_CONTAINER,
        0,
        b'{"ordinal":1.0,"id":"kept"}',
    )

    resolution = resolve_member_candidate(
        (candidate,),
        expected_digest=structural_content_identity(expected),
        hint_mode=MemberAddressingMode.ELEMENT_OF_CONTAINER,
        hint_index=0,
    )

    assert resolution.error is None
    assert resolution.outcome == "hint_verified"
    assert resolution.candidate == candidate


def test_structural_identity_does_not_fall_back_to_a_colliding_byte_hash(tmp_path: Path) -> None:
    """A declared value identity is authoritative, even if bytes collide.

    Anti-vacuity: the mutation sets ``content_identity`` to the candidate's
    byte hash.  Treating the two digests as interchangeable would accept a
    payload whose provider value is not the recorded value.
    """
    zip_path = tmp_path / "collision.zip"
    recorded_path = f"{zip_path}:conversations.json"
    payload = dumps_bytes(_session("kept"))
    _write_member(zip_path, [_META, _session("kept")])

    unit, error = zip_reacquired_unit(
        {
            **_row(recorded_path, payload=payload, source_index=0),
            "content_identity": hashlib.sha256(payload).hexdigest(),
            "addressing_mode": MemberAddressingMode.ELEMENT_OF_CONTAINER.value,
        },
        source_path=recorded_path,
        zip_payload_cache={},
    )

    assert unit is None
    assert error == "content_identity:unmatched"


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
        ({"path": "é/file"}, {"path": "é/file"}, False),
        ({"é": 1}, {"é": 1}, False),
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
        assert conn.execute(
            "SELECT addressing_mode, content_identity FROM raw_sessions WHERE raw_id = 'raw-1'"
        ).fetchone() == (MemberAddressingMode.WHOLE_MEMBER.value, None)

        identity = structural_content_identity(_session("one"))
        record_raw_container_coordinate(
            conn,
            "raw-1",
            coordinate_format="zip-v2",
            entry_ordinal=0,
            split_index=0,
            addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
            content_identity=identity,
            manage_transaction=False,
        )
        assert (
            conn.execute("SELECT content_identity FROM raw_container_coordinates WHERE raw_id = 'raw-1'").fetchone()[0]
            == identity
        )
        assert conn.execute(
            "SELECT addressing_mode, content_identity FROM raw_sessions WHERE raw_id = 'raw-1'"
        ).fetchone() == (MemberAddressingMode.WHOLE_MEMBER.value, identity)

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


class _ReadRecorder:
    """Delegates to a member handle and records every requested read size."""

    def __init__(self, handle: IO[bytes], sizes: list[int]) -> None:
        self._handle = handle
        self._sizes = sizes

    def read(self, size: int = -1) -> bytes:
        self._sizes.append(size)
        return self._handle.read(size)

    def __getattr__(self, name: str) -> object:
        return getattr(self._handle, name)


def test_replay_proves_preserved_members_without_holding_their_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proving a preserved member streams it, and the pass caches digests only.

    Anti-vacuity: read a whole member with ``handle.read()`` and a recorded
    read size is unbounded; keep a unit's bytes on its cached candidate and
    the no-bytes assertion fails.
    """
    from polylogue.sources import source_acquisition_components as components
    from polylogue.sources.acquisition_boundary import open_bound_member as open_member

    asset = b"".join(hashlib.sha256(str(index).encode()).digest() for index in range(96_000))
    document = _session("only")
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("file-abc.png", asset)
        archive.writestr("conversations.json", json.dumps(document, separators=(",", ":")))

    sizes: list[int] = []

    @contextlib.contextmanager
    def recording_open(zf: zipfile.ZipFile, entry: zipfile.ZipInfo, location: Provider | None) -> Iterator[object]:
        with open_member(zf, entry, location) as handle:
            yield _ReadRecorder(handle, sizes)

    monkeypatch.setattr(components, "open_bound_member", recording_open)
    cache: dict[str, tuple[MemberCandidate, ...]] = {}
    asset_path = f"{zip_path}:file-abc.png"
    asset_unit, asset_error = zip_reacquired_unit(
        _row(asset_path, payload=asset, source_index=None),
        source_path=asset_path,
        zip_payload_cache=cache,
    )
    document_path = f"{zip_path}:conversations.json"
    document_unit, document_error = zip_reacquired_unit(
        {
            **_row(document_path, payload=b"unused", source_index=None),
            "content_identity": structural_content_identity(document),
            "addressing_mode": MemberAddressingMode.WHOLE_MEMBER.value,
        },
        source_path=document_path,
        zip_payload_cache=cache,
    )

    assert asset_error is None and document_error is None
    assert asset_unit is not None and asset_unit.byte_identity == _sha(asset)
    assert asset_unit.size_bytes == len(asset)
    assert document_unit is not None and document_unit.addressing_mode is MemberAddressingMode.WHOLE_MEMBER
    assert sizes and all(0 <= size <= 1024 * 1024 for size in sizes)
    cached = [candidate for candidates in cache.values() for candidate in candidates]
    assert len(cached) == 2
    assert not any(
        isinstance(getattr(candidate, field.name), (bytes, bytearray))
        for candidate in cached
        for field in dataclasses.fields(candidate)
    )


def test_replay_refuses_to_guess_a_provider_from_the_public_origin(tmp_path: Path) -> None:
    """An unrecorded capture mode leaves the member unproven.

    The public ``aistudio-drive`` origin maps onto more than one provider, so
    it cannot stand in for the recorded capture mode.

    Anti-vacuity: restore the ``provider_from_origin`` fallback and replay
    proceeds under a guessed provider instead of returning this refusal.
    """
    zip_path = tmp_path / "export.zip"
    _write_member(zip_path, [_META, _session("only")])
    recorded_path = f"{zip_path}:conversations.json"
    row = {
        **_row(recorded_path, payload=dumps_bytes(_session("only")), source_index=0),
        "capture_mode": "",
        "origin": "aistudio-drive",
    }

    unit, error = zip_reacquired_unit(row, source_path=recorded_path, zip_payload_cache={})

    assert unit is None
    assert error == "replay_provider_unrecorded"


def test_zip_member_coordinate_splits_after_a_colon_in_the_container_path(tmp_path: Path) -> None:
    """Anti-vacuity: splitting at the first colon names ``<tmp>/odd`` as the
    container, which is not a file, so both lookups return ``None``.
    """
    container = tmp_path / "odd:name.zip"
    with zipfile.ZipFile(container, "w") as archive:
        archive.writestr("conversations.json", "[]")
    coordinate = f"{container}:conversations.json"
    assert zip_member_coordinate(coordinate) == (container, "conversations.json")
    assert zip_member_container(coordinate) == container
    # A loose file whose literal name holds a colon is never a member coordinate.
    loose = tmp_path / "plain:file.json"
    loose.write_text("{}")
    assert zip_member_coordinate(str(loose)) is None


def test_split_zip_member_text_separates_after_the_zip_suffix_when_the_container_is_gone() -> None:
    """Anti-vacuity: a first-colon split names ``C`` as the container of a
    Windows-style coordinate, and relocation and heartbeat labels lose the ZIP.
    """
    from polylogue.core.raw_coordinates import split_zip_member_text

    assert split_zip_member_text(r"C:\imports\chat.zip:conversations.json") == (
        r"C:\imports\chat.zip",
        "conversations.json",
    )
    assert split_zip_member_text("/gone/odd:name.ZIP:a:b.json") == ("/gone/odd:name.ZIP", "a:b.json")
    assert split_zip_member_text("/gone/plain:file.json") is None


def test_a_colon_path_names_a_container_only_when_its_prefix_is_a_real_zip(tmp_path: Path) -> None:
    """A missing colon path whose prefix exists but is no ZIP is not a member (polylogue-zrb9y).

    Every reader that splits ``<container>:<member>`` goes through
    ``core/raw_coordinates``: blob integrity's availability, archive debt's
    source presence, and ZIP reacquisition.

    Anti-vacuity: the former first-colon split read the existing ``odd``
    directory (and the plain ``notes`` file) as the container, so a deleted
    loose file reported its source as available and reacquisition opened a
    non-ZIP as the container.
    """
    from polylogue.operations import archive_debt
    from polylogue.storage import blob_integrity

    (tmp_path / "odd").mkdir()
    (tmp_path / "notes").write_text("plain file, not a ZIP")
    deleted_loose_files = (str(tmp_path / "odd:name.json"), str(tmp_path / "notes:conversations.json"))
    for recorded in deleted_loose_files:
        assert blob_integrity._source_path_availability(recorded)[0] is False
        assert archive_debt._source_artifact_exists(recorded) is False
        unit, reason = zip_reacquired_unit(
            _row(recorded, payload=b"{}", source_index=0), source_path=recorded, zip_payload_cache={}
        )
        assert (unit, reason) == (None, "container_coordinate_missing")

    fake_zip = tmp_path / "fake.zip"
    fake_zip.write_text("not a ZIP archive")
    member_path = f"{fake_zip}:conversations.json"
    unit, reason = zip_reacquired_unit(
        _row(member_path, payload=b"{}", source_index=0), source_path=member_path, zip_payload_cache={}
    )
    assert (unit, reason) == (None, "source_missing")
    assert archive_debt._source_artifact_exists(member_path) is False

    real_zip = tmp_path / "real.zip"
    _write_member(real_zip, [_session("one")])
    assert archive_debt._source_artifact_exists(f"{real_zip}:conversations.json") is True
    assert blob_integrity._source_path_availability(f"{real_zip}:conversations.json")[0] is True


def test_zip_member_proof_refuses_a_changed_operational_string(tmp_path: Path) -> None:
    zip_path = tmp_path / "export.zip"
    source_path = f"{zip_path}:conversations.json"
    original = {**_session("neutral"), "tool_path": "é/file"}
    changed = {**original, "tool_path": "é/file"}
    expected = dumps_bytes(original)
    row = {
        **_row(source_path, payload=expected, source_index=0),
        "coordinate_format": "zip-v2",
        "entry_ordinal": 0,
        "split_index": 0,
        "addressing_mode": MemberAddressingMode.WHOLE_MEMBER.value,
        "content_identity": structural_content_identity(original),
    }
    _write_member(zip_path, changed)
    unit, error = zip_reacquired_unit(row, source_path=source_path, zip_payload_cache={})
    assert unit is None
    assert error == "content_identity:unmatched"
    _write_member(zip_path, dict(reversed(list(original.items()))))
    unit, error = zip_reacquired_unit(row, source_path=source_path, zip_payload_cache={})
    assert error is None
    assert unit is not None
    assert unit.content_identity == structural_content_identity(original)
