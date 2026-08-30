"""Read-only replay of source ZIP acquisition units for evidence checks."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path

from polylogue.config import Source
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import provider_from_origin
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    replay_zip_entry_acquisition_payloads,
)
from polylogue.storage.blob_store import BlobStore


def zip_reacquisition_payload(
    row: Mapping[str, object],
    *,
    source_path: str,
    zip_payload_cache: dict[str, dict[int, bytes]],
) -> tuple[bytes | None, str | None]:
    """Replay one ZIP member through the production acquisition splitter."""
    coordinate = _zip_coordinate(row)
    if coordinate is None:
        return None, "container_coordinate_missing"
    entry_ordinal, split_index = coordinate
    zip_path_text, _separator, member = source_path.partition(":")
    zip_path = Path(zip_path_text)
    if not zip_path.exists():
        return None, "source_missing"
    cache_key = source_path
    payloads = zip_payload_cache.get(cache_key)
    if payloads is None:
        try:
            with zipfile.ZipFile(zip_path) as archive:
                central_directory = archive.infolist()
                if entry_ordinal >= len(central_directory):
                    return None, "container_coordinate_mismatch"
                entry = central_directory[entry_ordinal]
                if entry.filename != member:
                    return None, "container_coordinate_mismatch"
                provider = Provider.from_string(str(row.get("capture_mode") or ""))
                if provider is Provider.UNKNOWN:
                    provider = provider_from_origin(Origin.from_string(str(row.get("origin") or "")))
                context = ZipEntryReadContext(
                    source=Source(name=provider.value, path=zip_path.parent),
                    zip_path=zip_path,
                    entry=entry,
                    file_mtime=None,
                    provider_hint=provider,
                    blob_store=BlobStore(zip_path.parent / "blob"),
                )
                payloads = {}
                for acquired in replay_zip_entry_acquisition_payloads(archive, context):
                    acquired_index = acquired.source_index if acquired.source_index is not None else 0
                    payloads[acquired_index] = acquired.payload_bytes
        except Exception as exc:
            # Source replay is evidence, not a prerequisite for constructing
            # the backup. Any unreadable or unparseable container therefore
            # leaves this reference unproven and lets verification fail closed.
            return None, f"error:{exc}"
        zip_payload_cache[cache_key] = payloads
    return payloads.get(split_index), None if split_index in payloads else "source_index:unmatched"


def _zip_coordinate(row: Mapping[str, object]) -> tuple[int, int] | None:
    if row.get("coordinate_format") == "zip-v2":
        entry_ordinal = row.get("entry_ordinal")
        split_index = row.get("split_index")
        if isinstance(entry_ordinal, (int, str)) and isinstance(split_index, (int, str)):
            return int(entry_ordinal), int(split_index)
    source_path = row.get("source_path")
    source_index = row.get("source_index")
    raw_id = str(row.get("raw_id") or row.get("ref_id") or "")
    blob_hash = str(row.get("blob_hash") or "")
    if (
        not isinstance(source_path, str)
        or not source_path
        or ":" not in source_path
        or not isinstance(source_index, (int, str))
    ):
        return None
    from polylogue.core.raw_coordinates import zip_member_identity_coordinate

    return zip_member_identity_coordinate(
        raw_id=raw_id,
        source_path=source_path,
        source_index=int(source_index),
        blob_hash=blob_hash,
    )


__all__ = ["zip_reacquisition_payload"]
