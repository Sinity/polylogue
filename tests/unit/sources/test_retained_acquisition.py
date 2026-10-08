"""Retained acquisition reads the admitted physical blob, not its old path."""

from __future__ import annotations

import zipfile
import zlib
from pathlib import Path

import pytest

from polylogue.archive.zip_admission import MAX_REPORTED_MEMBER_DETAIL_CHARS, MAX_REPORTED_MEMBER_DETAILS
from polylogue.core.enums import Provider
from polylogue.core.raw_coordinates import MemberAddressingMode
from polylogue.sources import retained_acquisition
from polylogue.sources.acquisition_boundary import open_bound_container
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.retained_acquisition import SourceInputRecord, iter_retained_source_records
from polylogue.sources.source_acquisition_components import SourceReadContext
from polylogue.sources.source_staging import bind_source_input
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

_MEMBER = "projects/synthetic/session.jsonl"


def _raw_data(record: SourceInputRecord) -> RawSessionData:
    assert record.data is not None
    return record.data


def _retain_container(store: BlobStore, path: Path) -> tuple[str, int, CapturedSourceInputIdentity]:
    with bind_source_input(path) as binding, open_bound_container(store, binding) as capture:
        identity = capture.captured_identity
        blob_hash, size, _receipt = capture.retain()
        return blob_hash, size, identity


def test_declared_source_name_preserves_label_and_exact_provider_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size = store.write_from_bytes(b"synthetic retained input")
    observed: list[tuple[str, Provider]] = []

    def capture(context: SourceReadContext) -> RawSessionData:
        observed.append((context.source.name, context.provider_hint))
        return RawSessionData(raw_bytes=b"synthetic retained input", source_path=str(tmp_path / "capture.jsonl"))

    monkeypatch.setattr(retained_acquisition, "read_plain_source_file", capture)
    for name in ("codex", "my_custom_source"):
        list(
            iter_retained_source_records(
                enumeration_fingerprint="b" * 64,
                source_path=str(tmp_path / "capture.jsonl"),
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_store=store,
                source_name=name,
            )
        )

    assert observed == [("codex", Provider.CODEX), ("my_custom_source", Provider.UNKNOWN)]


def _write_duplicate_zip(path: Path) -> tuple[bytes, bytes]:
    first = b'{"retained":"first"}\n'
    second = b'{"retained":"second"}\n'
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for payload in (first, second):
            info = zipfile.ZipInfo(_MEMBER)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return first, second


def test_retained_zip_uses_exact_blob_after_original_path_is_deleted(tmp_path: Path) -> None:
    original = tmp_path / "synthetic-export.zip"
    first, second = _write_duplicate_zip(original)
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, captured_identity = _retain_container(store, original)
    original.unlink()

    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
            captured_identity=captured_identity,
        )
    )

    assert [record.coordinate for record in records] == [
        '["zip-v2",0,0,"whole_member"]',
        '["zip-v2",1,0,"whole_member"]',
    ]
    assert [record.entry_ordinal for record in records] == [0, 1]
    assert [record.split_index for record in records] == [0, 0]
    assert [_raw_data(record).addressing_mode for record in records] == [
        MemberAddressingMode.WHOLE_MEMBER,
        MemberAddressingMode.WHOLE_MEMBER,
    ]
    assert [store.read_all(_raw_data(record).blob_hash or "") for record in records] == [first, second]
    assert records[0].raw_id != records[1].raw_id
    assert all(_raw_data(record).source_path == f"{original}:{_MEMBER}" for record in records)


@pytest.mark.parametrize("cut", [1, 8])
def test_interrupted_retained_zip_raises_instead_of_claiming_complete(tmp_path: Path, cut: int) -> None:
    original = tmp_path / "synthetic-export.zip"
    _write_duplicate_zip(original)
    interrupted = original.read_bytes()[:-cut]
    store = BlobStore(tmp_path / "blob")
    original.write_bytes(interrupted)
    blob_hash, blob_size, captured_identity = _retain_container(store, original)

    with pytest.raises(zipfile.BadZipFile):
        list(
            iter_retained_source_records(
                enumeration_fingerprint="b" * 64,
                source_path=str(original),
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_store=store,
                captured_identity=captured_identity,
            )
        )


def test_corrupt_retained_zip_never_yields_a_completed_record_set(tmp_path: Path) -> None:
    original = tmp_path / "synthetic-export.zip"
    _write_duplicate_zip(original)
    corrupt = bytearray(original.read_bytes())
    # First local header: fixed 30 bytes, then the filename and extra field.
    # Corrupt compressed payload rather than the optional archive comment.
    with zipfile.ZipFile(original) as archive:
        first = archive.infolist()[0]
        payload_offset = first.header_offset + 30 + len(first.filename.encode()) + len(first.extra)
    corrupt[payload_offset] ^= 0xFF
    store = BlobStore(tmp_path / "blob")
    original.write_bytes(bytes(corrupt))
    blob_hash, blob_size, captured_identity = _retain_container(store, original)

    with pytest.raises((zipfile.BadZipFile, ValueError, zlib.error)):
        list(
            iter_retained_source_records(
                enumeration_fingerprint="b" * 64,
                source_path=str(original),
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_store=store,
                captured_identity=captured_identity,
            )
        )


def test_retained_plain_input_does_not_reopen_deleted_acquisition_path(tmp_path: Path) -> None:
    original = tmp_path / "capture.json"
    payload = b'{"synthetic":"retained input"}'
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size = store.write_from_bytes(payload)
    assert not original.exists()
    (record,) = iter_retained_source_records(
        enumeration_fingerprint="b" * 64,
        source_path=str(original),
        blob_hash=blob_hash,
        blob_size=blob_size,
        blob_store=store,
    )
    assert record.coordinate == '["physical-file-v1",0]'
    assert _raw_data(record).source_path == str(original)
    assert _raw_data(record).blob_hash == blob_hash
    assert store.read_all(blob_hash) == payload


def _write_zip_with_refused_identity_member(path: Path) -> None:
    """One number exceeds a synthetic physical value limit."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("first.jsonl", b'{"retained":"first"}\n')
        archive.writestr("limited.jsonl", b'{"n": 1.' + b"2" * 200 + b"}\n")
        archive.writestr("third.jsonl", b'{"retained":"third"}\n')


def test_one_physically_refused_member_does_not_abort_its_whole_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("polylogue.core.content_identity.physical_value_limit", lambda: 64)
    original = tmp_path / "synthetic-export.zip"
    _write_zip_with_refused_identity_member(original)
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, captured_identity = _retain_container(store, original)

    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
            captured_identity=captured_identity,
        )
    )

    # The admitted members keep their central-directory ordinals, and the
    # generator exhausts normally so its caller can close the source item.
    assert [record.entry_ordinal for record in records] == [0, 2]
    assert [store.read_all(_raw_data(record).blob_hash or "") for record in records] == [
        b'{"retained":"first"}\n',
        b'{"retained":"third"}\n',
    ]


def test_retained_zip_exposes_non_admitted_members_for_durable_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production caller can persist each skipped ordinal, not just a log count."""
    monkeypatch.setattr("polylogue.core.content_identity.physical_value_limit", lambda: 64)
    original = tmp_path / "synthetic-export.zip"
    _write_zip_with_refused_identity_member(original)
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, captured_identity = _retain_container(store, original)
    published: list[tuple[int, str, str, str]] = []

    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
            captured_identity=captured_identity,
            on_member_disposition=lambda ordinal, name, disposition, diagnostic: published.append(
                (ordinal, name, disposition, diagnostic)
            ),
        )
    )

    assert [record.entry_ordinal for record in records] == [0, 2, 1]
    assert len(published) == 1
    assert published[0][:3] == (1, "limited.jsonl", "refused")
    assert "content_identity_refused" in published[0][3]


_DECLARED_ASSET = "file-abc123XYZ.png"
_PNG_BYTES = bytes.fromhex("89504e470d0a1a0a") + b"synthetic-asset-bytes"


def _write_export_zip_with_declared_artifact(path: Path) -> None:
    """A provider-agnostic export ZIP: one JSON member and one declared asset."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("conversations.json", b'[{"title":"synthetic","mapping":{}}]')
        archive.writestr(_DECLARED_ASSET, _PNG_BYTES)


def _capture_retained_events(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, object]]]:
    captured: list[tuple[str, dict[str, object]]] = []

    def record(event: str, /, **fields: object) -> None:
        captured.append((event, dict(fields)))

    monkeypatch.setattr("polylogue.sources.retained_acquisition.emit", record)
    return captured


def test_retained_zip_keeps_declared_artifact_members_under_an_unknown_provider(
    tmp_path: Path,
) -> None:
    """A retained export ZIP retains its declared assets, not only JSON.

    Anti-vacuity: the retained blob has no provider-bearing path, so this route
    resolves ``Provider.UNKNOWN``; dropping either the provider sniff or the
    ``is_declared_artifact_path`` fallback restores the old inferred rule
    ``artifact_rule_for_path(Provider.UNKNOWN, name)``, which is always ``None``,
    and ``file-abc123XYZ.png`` disappears from the retained set while the
    generator still exhausts normally.
    """
    original = tmp_path / "synthetic-export.zip"
    _write_export_zip_with_declared_artifact(original)
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, captured_identity = _retain_container(store, original)

    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
            captured_identity=captured_identity,
        )
    )

    assert [_raw_data(record).source_path.rsplit(":", 1)[-1] for record in records] == [
        "conversations.json",
        _DECLARED_ASSET,
    ]
    # The asset's exact bytes are retained, not a re-encoded interpretation.
    assert store.read_all(_raw_data(records[1]).blob_hash or "") == _PNG_BYTES


def test_retained_zip_counts_an_unselected_member_instead_of_dropping_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member no declaration owns is reported, never silently skipped.

    Anti-vacuity: the caller records normal exhaustion as proven-complete
    enumeration. Reverting the ``on_unselected`` branch in
    ``ZipAdmission.filter_entries`` to a bare ``continue`` leaves ``chat.html``
    out of the retained records *and* out of every event, so this assertion on
    ``sources.retained_zip.members_unselected`` goes red.
    """
    original = tmp_path / "synthetic-export.zip"
    with zipfile.ZipFile(original, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("conversations.json", b'[{"title":"synthetic","mapping":{}}]')
        archive.writestr("chat.html", b"<html>not a declared artifact</html>")
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, captured_identity = _retain_container(store, original)
    captured = _capture_retained_events(monkeypatch)

    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
            captured_identity=captured_identity,
        )
    )

    assert [_raw_data(record).source_path.rsplit(":", 1)[-1] for record in records] == ["conversations.json"]
    unselected = [fields for event, fields in captured if event == "sources.retained_zip.members_unselected"]
    assert len(unselected) == 1
    assert unselected[0]["skipped"] == 1
    assert "chat.html" in str(unselected[0]["error_detail"])


def test_retained_zip_counts_an_inadmissible_member_as_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member refused by admission reaches the refusal event with a count.

    A real physical identity refusal is counted separately from valid input
    size or compression, which cannot authorize member loss.
    """
    monkeypatch.setattr("polylogue.core.content_identity.physical_value_limit", lambda: 64)
    original = tmp_path / "synthetic-export.zip"
    _write_zip_with_refused_identity_member(original)
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, captured_identity = _retain_container(store, original)
    captured = _capture_retained_events(monkeypatch)

    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
            captured_identity=captured_identity,
        )
    )

    assert [record.entry_ordinal for record in records] == [0, 2]
    refused = [fields for event, fields in captured if event == "sources.retained_zip.members_refused"]
    assert len(refused) == 1
    assert refused[0]["skipped"] == 1
    assert "content_identity_refused" in str(refused[0]["error_detail"])


def test_retained_zip_bounds_unselected_detail_while_counting_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many skipped members cost a bounded report, not one string each.

    Anti-vacuity: ZIP member names and counts are attacker-controlled. Reverting
    the ``BoundedMemberReport`` accumulation in ``iter_retained_source_records``
    to a per-member list plus ``"; ".join(...)`` makes ``error_detail`` grow with
    the archive -- it would carry every one of the 400 names and far exceed the
    bound asserted here -- so both the length assertion and the explicit
    "withheld" accounting go red while ``skipped`` alone stays green.
    """
    member_count = 400
    long_name = "A" * 500
    original = tmp_path / "hostile-export.zip"
    with zipfile.ZipFile(original, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("conversations.json", b'[{"title":"synthetic","mapping":{}}]')
        for index in range(member_count):
            archive.writestr(f"{long_name}-{index}.html", b"")
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, captured_identity = _retain_container(store, original)
    captured = _capture_retained_events(monkeypatch)

    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
            captured_identity=captured_identity,
        )
    )

    assert [_raw_data(record).source_path.rsplit(":", 1)[-1] for record in records] == ["conversations.json"]
    unselected = [fields for event, fields in captured if event == "sources.retained_zip.members_unselected"]
    assert len(unselected) == 1
    # The count stays exact: that denominator is what the event exists for.
    assert unselected[0]["skipped"] == member_count
    detail = str(unselected[0]["error_detail"])
    # The bound is observable, not a silent shortening: the detail says how many
    # members it named and how many it withheld.
    assert f"{member_count - MAX_REPORTED_MEMBER_DETAILS} withheld" in detail
    assert f"{MAX_REPORTED_MEMBER_DETAILS} of {member_count} members named" in detail
    # Bounded sample count x bounded per-name length, plus the accounting clause.
    assert len(detail) < MAX_REPORTED_MEMBER_DETAILS * (MAX_REPORTED_MEMBER_DETAIL_CHARS + 120) + 200
    # A single oversized name is truncated rather than retained whole.
    assert long_name not in detail


def test_historical_zip_without_captured_namespace_is_an_explicit_gap(tmp_path: Path) -> None:
    from polylogue.core.raw_failure_evidence import RetainedZipMembershipUnprovedError

    original = tmp_path / "historical.zip"
    _write_duplicate_zip(original)
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, _identity = _retain_container(store, original)
    original.unlink()
    with pytest.raises(RetainedZipMembershipUnprovedError):
        list(
            iter_retained_source_records(
                enumeration_fingerprint="b" * 64,
                source_path=str(original),
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_store=store,
            )
        )


def test_zip_database_basename_does_not_reclassify_its_container_as_native_state(tmp_path: Path) -> None:
    original = tmp_path / "state.db"
    _write_duplicate_zip(original)
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, identity = _retain_container(store, original)
    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
            captured_identity=identity,
        )
    )
    assert [store.read_all(_raw_data(record).blob_hash or "") for record in records] == [
        b'{"retained":"first"}\n',
        b'{"retained":"second"}\n',
    ]


def test_highly_compressible_valid_member_is_retained_without_ratio_refusal(tmp_path: Path) -> None:
    original = tmp_path / "compressible.zip"
    payload = b'{"text":"' + b"a" * (4 * 1024 * 1024) + b'"}\n'
    with zipfile.ZipFile(original, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.jsonl", payload)
    store = BlobStore(tmp_path / "blob")
    blob_hash, size, identity = _retain_container(store, original)
    records = list(
        iter_retained_source_records(
            enumeration_fingerprint="b" * 64,
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=size,
            blob_store=store,
            captured_identity=identity,
        )
    )
    assert len(records) == 1
    assert records[0].data is not None
    assert store.read_all(records[0].data.blob_hash or "") == payload


def test_empty_retained_zip_does_not_allocate_a_disposition_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "empty.zip"
    with zipfile.ZipFile(original, "w"):
        pass
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size, identity = _retain_container(store, original)

    def forbidden_spool() -> None:
        pytest.fail("a disposition-free container allocated private member storage")

    monkeypatch.setattr(retained_acquisition, "PickleSpool", forbidden_spool)
    assert (
        list(
            iter_retained_source_records(
                enumeration_fingerprint="b" * 64,
                source_path=str(original),
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_store=store,
                captured_identity=identity,
                on_member_disposition=lambda *_fields: None,
            )
        )
        == []
    )
