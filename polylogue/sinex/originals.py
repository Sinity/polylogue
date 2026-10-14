"""Resolve acquired original bytes exclusively from retained Source witnesses."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from polylogue.sinex.models import PublicationAcquisition, PublicationOriginal
from polylogue.storage.blob_store import BlobStore


class PublicationOriginalUnavailableError(ValueError):
    """Accepted material has no intact acquired Source byte witness."""


def originals_for_raw_ids(
    conn: sqlite3.Connection, raw_ids: Iterable[str], *, blob_store: BlobStore
) -> tuple[PublicationOriginal, ...]:
    """Capture exact raw/attachment CAS coordinates, never mutable source paths.

    A raw acquisition can be an append byte window or logical SQLite export.
    Its retained blob is the original acquisition under that coordinate, not
    a claim to the provider's whole physical file.
    """
    originals: list[PublicationOriginal] = []
    for raw_id in dict.fromkeys(raw_ids):
        row = conn.execute(
            "SELECT blob_hash,blob_size,revision_kind,append_start_offset,append_end_offset,addressing_mode "
            "FROM raw_sessions WHERE raw_id=?",
            (raw_id,),
        ).fetchone()
        if row is None:
            raise PublicationOriginalUnavailableError("publication_raw_witness_missing")
        kind = "byte_window" if row[2] == "append" else "full_snapshot" if row[2] == "full" else "unknown"
        tables = None
        with blob_store.open(bytes(row[0]).hex()) as source:
            from polylogue.sources.sqlite_export import (
                EXPORT_PROBE_BYTES,
                _parse_header,
                looks_like_logical_export_bytes,
            )

            prefix = source.read(EXPORT_PROBE_BYTES)
            if looks_like_logical_export_bytes(prefix):
                kind = "logical_snapshot"
                tables = _parse_header(prefix + source.readline()).tables
        acquisition = PublicationAcquisition(
            kind=kind,
            byte_start=row[3] if kind == "byte_window" else None,
            byte_end=row[4] if kind == "byte_window" else None,
            addressing_mode=row[5],
            tables=tables,
        )
        if kind == "byte_window":
            assert acquisition.byte_start is not None and acquisition.byte_end is not None
            if acquisition.byte_end - acquisition.byte_start != int(row[1]):
                raise PublicationOriginalUnavailableError("publication_original_window_size_mismatch")
        originals.append(
            PublicationOriginal(
                "provider_native",
                f"polylogue.raw:{raw_id}",
                bytes(row[0]).hex(),
                int(row[1]),
                "application/octet-stream",
                acquisition,
            )
        )
        for ref in conn.execute(
            "SELECT blob_hash,size_bytes,source_path FROM blob_refs WHERE ref_type='attachment' AND ref_id=? "
            "ORDER BY hex(blob_hash),source_path",
            (raw_id,),
        ):
            # Preserve the exact acquisition coordinate without interpreting it
            # as a URI or rereading that provider location.
            source_ref = f"polylogue.raw:{raw_id}:attachment:{bytes(ref[0]).hex()}:{str(ref[2] or '')}"
            originals.append(
                PublicationOriginal(
                    "attachment", source_ref, bytes(ref[0]).hex(), int(ref[1]), "application/octet-stream"
                )
            )
    if not originals:
        raise PublicationOriginalUnavailableError("publication_provider_original_missing")
    for original in originals:
        if (
            conn.execute(
                "SELECT 1 FROM excised_content WHERE removed_hash=? AND hash_kind='blob_hash'",
                (bytes.fromhex(original.blob_hash),),
            ).fetchone()
            is not None
        ):
            raise PublicationOriginalUnavailableError("publication_original_excised")
    return tuple(originals)
