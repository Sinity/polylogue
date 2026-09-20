"""Enumerate accepted physical inputs without reopening acquisition paths.

The caller owns compute admission and blob publication. Normal exhaustion is
the only enumeration-complete signal. A member refused by ZIP admission was
never ingested by ordinary acquisition either, so it is skipped and logged
rather than raised: one pathological member must not cost every other member
of its input.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.zip_admission import ZIP_JSON_SUFFIXES, BoundedMemberReport, ZipAdmission
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.raw_coordinates import MemberAddressingMode, zip_member_raw_id, zip_member_source_index
from polylogue.logging import WARNING, emit
from polylogue.sources.decoder_zip import (
    ZipEntryValidator,
    declared_artifact_provider,
    is_declared_artifact_path,
    provider_detection_path,
)
from polylogue.sources.live.admission import ArtifactIdentity
from polylogue.sources.origin_specs import database_member_for_filename
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.source_acquisition_components import (
    SourceReadContext,
    ZipEntryReadContext,
    iter_zip_entry_raw_data,
    read_plain_source_file,
    sniff_zip_provider,
)
from polylogue.storage.blob_store import BlobStore


@dataclass(frozen=True, slots=True)
class RetainedRawRecord:
    coordinate: str
    data: RawSessionData
    raw_id: str | None = None
    entry_ordinal: int | None = None
    split_index: int | None = None
    member_disposition: str | None = None
    member_name: str | None = None
    diagnostic: str | None = None
    member_count: int | None = None


def iter_retained_source_records(
    *,
    source_path: str,
    blob_hash: str,
    blob_size: int,
    blob_store: BlobStore,
    on_member_disposition: Callable[[int, str, str, str], None] | None = None,
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
        # A plain source has no central directory. Its source-43 record is
        # still enumerated and digested, but it must not be mistaken for a
        # ZIP member ordinal by retained-member completeness checks.
        yield RetainedRawRecord('["physical-file-v1",0]', data)
        return

    # Both channels are driven by attacker-controlled central-directory
    # metadata, so they count exactly under a bounded detail sample rather than
    # accumulating one string per member (and then joining them all).
    rejected = BoundedMemberReport()
    unselected = BoundedMemberReport()
    dispositions: list[tuple[int, str, str, str]] = []
    with blob_store.open(blob_hash) as physical, zipfile.ZipFile(physical) as archive:
        entries = archive.infolist()
        ordinals = {id(entry): ordinal for ordinal, entry in enumerate(entries)}
        # A retained physical blob carries no provider-bearing directory: the
        # database-member binding above only resolves for a declared database
        # export, so an account export ZIP always arrives as UNKNOWN. Recover
        # the real provider the way the inbox route does, then let its own
        # artifact declarations decide which members are material.
        if provider is Provider.UNKNOWN:
            detection_entries = [
                info
                for info in ZipAdmission(zip_path=logical_path).filter_entries(
                    entries,
                    allowed_suffixes=ZIP_JSON_SUFFIXES,
                )
                if provider_detection_path(info.filename)
            ]
            provider = sniff_zip_provider(archive, detection_entries) or Provider.UNKNOWN
        # A genuinely mixed ZIP keeps UNKNOWN, under which no single provider's
        # path rule can fire. Admitting any declared artifact path is what stops
        # every ChatGPT export asset, Antigravity protobuf, brain Markdown and
        # tool-result sidecar from being dropped while enumeration still reports
        # itself complete (polylogue-ojxpn).
        allowed_path = is_declared_artifact_path if provider is Provider.UNKNOWN else None

        def record_rejected(entry: zipfile.ZipInfo, reason: str) -> None:
            rejected.record(reason)
            dispositions.append((ordinals[id(entry)], entry.filename, "refused", reason))

        def record_unselected(entry: zipfile.ZipInfo, reason: str) -> None:
            unselected.record(f"{entry.filename}: {reason}")
            dispositions.append((ordinals[id(entry)], entry.filename, "unselected", reason))

        validator = ZipEntryValidator(provider, cursor_state=None, zip_path=logical_path)
        for entry in validator.filter_entries(
            entries,
            allowed_path=allowed_path,
            on_rejected=record_rejected,
            on_unselected=record_unselected,
        ):
            ordinal = ordinals[id(entry)]
            # Under a residual UNKNOWN the container hint cannot name the family
            # that owns this member, so its ``raw-only`` declaration would not
            # fire and arbitrary binary bytes would take the JSON split route.
            entry_provider = provider
            if provider is Provider.UNKNOWN:
                entry_provider = declared_artifact_provider(entry.filename) or provider
            context = ZipEntryReadContext(source, logical_path, entry, None, entry_provider, blob_store)
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
                    member_count=len(entries),
                )
    for ordinal, member_name, disposition, diagnostic in dispositions:
        if on_member_disposition is None:
            continue
        record = RetainedRawRecord(
            coordinate=json.dumps(["zip-member-v1", ordinal], separators=(",", ":")),
            data=None,  # type: ignore[arg-type]  # disposition records carry no raw payload
            entry_ordinal=ordinal,
            member_disposition=disposition,
            member_name=member_name,
            diagnostic=diagnostic,
            member_count=len(entries),
        )
        if on_member_disposition is not None:
            on_member_disposition(ordinal, member_name, disposition, diagnostic)
        yield record
    if rejected:
        emit(
            "sources.retained_zip.members_refused",
            level=WARNING,
            outcome="degraded",
            reason="member_refused",
            path=logical_path,
            skipped=rejected.count,
            error_detail=rejected.detail(),
        )
    if unselected:
        # Non-selection is ordinary, but this route's caller records normal
        # exhaustion as *proven-complete* enumeration. A member dropped without
        # a denominator is indistinguishable from an input that never held it,
        # so name it here rather than letting it vanish.
        emit(
            "sources.retained_zip.members_unselected",
            level=WARNING,
            outcome="degraded",
            reason="member_unselected",
            path=logical_path,
            skipped=unselected.count,
            error_detail=unselected.detail(),
        )
