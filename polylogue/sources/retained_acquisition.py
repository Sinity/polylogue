"""Enumerate accepted physical inputs without reopening acquisition paths.

The caller owns compute admission and blob publication. Normal exhaustion is
the only enumeration-complete signal; a rejected member raises even when an
earlier member has already yielded retained raw evidence.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.raw_coordinates import MemberAddressingMode, zip_member_raw_id, zip_member_source_index
from polylogue.sources.decoder_zip import ZipEntryValidator
from polylogue.sources.live.admission import ArtifactIdentity
from polylogue.sources.origin_specs import database_member_for_filename
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.source_acquisition_components import (
    SourceReadContext,
    ZipEntryReadContext,
    iter_zip_entry_raw_data,
    read_plain_source_file,
)
from polylogue.storage.blob_store import BlobStore


@dataclass(frozen=True, slots=True)
class RetainedRawRecord:
    coordinate: str
    data: RawSessionData
    raw_id: str | None = None
    entry_ordinal: int | None = None
    split_index: int | None = None


def iter_retained_source_records(
    *,
    source_path: str,
    blob_hash: str,
    blob_size: int,
    blob_store: BlobStore,
) -> Iterator[RetainedRawRecord]:
    """Use canonical bounded decoders over the exact retained physical blob.

    ``source_path`` supplies identity and declared filename semantics only.
    ZIP entries are opened by their admitted ZipInfo, preserving duplicate
    names as separate central-directory coordinates.
    """
    logical_path = Path(source_path)
    source = Source(name="machine-ingest", path=logical_path)
    binding = database_member_for_filename(logical_path.name)
    provider = binding.provider if binding is not None else Provider.UNKNOWN
    if logical_path.suffix.lower() != ".zip":
        data = read_plain_source_file(
            SourceReadContext(
                source=source,
                path=logical_path,
                file_mtime=None,
                provider_hint=provider,
                blob_store=blob_store,
                retained_blob=ArtifactIdentity(blob_hash, blob_size),
            )
        )
        yield RetainedRawRecord('["physical-file-v1",0]', data)
        return

    rejected: list[str] = []
    with blob_store.open(blob_hash) as physical, zipfile.ZipFile(physical) as archive:
        entries = archive.infolist()
        ordinals = {id(entry): ordinal for ordinal, entry in enumerate(entries)}
        validator = ZipEntryValidator(provider, cursor_state=None, zip_path=logical_path)
        for entry in validator.filter_entries(entries, on_rejected=lambda _entry, reason: rejected.append(reason)):
            ordinal = ordinals[id(entry)]
            context = ZipEntryReadContext(source, logical_path, entry, None, provider, blob_store)
            for data in iter_zip_entry_raw_data(archive, context):
                split = data.source_index or 0
                mode = data.addressing_mode
                if mode not in {MemberAddressingMode.WHOLE_MEMBER, MemberAddressingMode.ELEMENT_OF_CONTAINER}:
                    raise ValueError("retained ZIP record has no exact addressing mode")
                if data.blob_hash is None:
                    raise ValueError("retained ZIP decoder did not retain its raw bytes")
                yield RetainedRawRecord(
                    json.dumps(["zip-v2", ordinal, split, mode.value], separators=(",", ":")),
                    data.model_copy(
                        update={"source_index": zip_member_source_index(entry_ordinal=ordinal, split_index=split)}
                    ),
                    zip_member_raw_id(
                        source_path=data.source_path,
                        entry_ordinal=ordinal,
                        split_index=split,
                        blob_hash=data.blob_hash,
                    ),
                    ordinal,
                    split,
                )
    if rejected:
        raise ValueError(f"retained ZIP enumeration rejected {len(rejected)} member(s): {rejected[0]}")
