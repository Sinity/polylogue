"""Retained acquisition reads the admitted physical blob, not its old path."""

from __future__ import annotations

import zipfile
import zlib
from pathlib import Path

import pytest

from polylogue.core.raw_coordinates import MemberAddressingMode
from polylogue.sources.retained_acquisition import iter_retained_source_records
from polylogue.storage.blob_store import BlobStore

_MEMBER = "projects/synthetic/session.jsonl"


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
    blob_hash, blob_size = store.write_from_path(original)
    original.unlink()

    records = list(
        iter_retained_source_records(
            source_path=str(original),
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_store=store,
        )
    )

    assert [record.coordinate for record in records] == [
        '["zip-v2",0,0,"whole_member"]',
        '["zip-v2",1,0,"whole_member"]',
    ]
    assert [record.entry_ordinal for record in records] == [0, 1]
    assert [record.split_index for record in records] == [0, 0]
    assert [record.data.addressing_mode for record in records] == [
        MemberAddressingMode.WHOLE_MEMBER,
        MemberAddressingMode.WHOLE_MEMBER,
    ]
    assert [store.read_all(record.data.blob_hash or "") for record in records] == [first, second]
    assert records[0].raw_id != records[1].raw_id
    assert all(record.data.source_path == f"{original}:{_MEMBER}" for record in records)


@pytest.mark.parametrize("cut", [1, 8])
def test_interrupted_retained_zip_raises_instead_of_claiming_complete(tmp_path: Path, cut: int) -> None:
    original = tmp_path / "synthetic-export.zip"
    _write_duplicate_zip(original)
    interrupted = original.read_bytes()[:-cut]
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size = store.write_from_bytes(interrupted)

    with pytest.raises(zipfile.BadZipFile):
        list(
            iter_retained_source_records(
                source_path=str(original),
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_store=store,
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
    blob_hash, blob_size = store.write_from_bytes(bytes(corrupt))

    with pytest.raises((zipfile.BadZipFile, ValueError, zlib.error)):
        list(
            iter_retained_source_records(
                source_path=str(original),
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_store=store,
            )
        )


def test_retained_plain_input_does_not_reopen_deleted_acquisition_path(tmp_path: Path) -> None:
    original = tmp_path / "capture.json"
    payload = b'{"synthetic":"retained input"}'
    store = BlobStore(tmp_path / "blob")
    blob_hash, blob_size = store.write_from_bytes(payload)
    assert not original.exists()
    (record,) = iter_retained_source_records(
        source_path=str(original), blob_hash=blob_hash, blob_size=blob_size, blob_store=store
    )
    assert record.coordinate == '["physical-file-v1",0]'
    assert record.data.source_path == str(original)
    assert record.data.blob_hash == blob_hash
    assert store.read_all(blob_hash) == payload
