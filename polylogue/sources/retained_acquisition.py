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
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

from polylogue.archive.zip_admission import BoundedMemberReport, open_zip_entry
from polylogue.config import Source
from polylogue.core.content_identity import ContentIdentityRefusal
from polylogue.core.enums import Provider
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.core.raw_coordinates import (
    MemberAddressingMode,
    captured_zip_member_raw_id,
    zip_member_record_coordinate,
    zip_member_source_index,
)
from polylogue.core.raw_failure_evidence import RetainedZipMembershipUnprovedError
from polylogue.logging import WARNING, emit
from polylogue.sources.acquisition_boundary import refuse_declared_foreign
from polylogue.sources.decoder_zip import ZipEntryValidator
from polylogue.sources.dispatch import ForeignOriginContentError, bound_location_provider
from polylogue.sources.origin_specs import database_member_for_filename
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.source_acquisition_components import (
    ArtifactIdentity,
    ObservationCallback,
    SourceReadContext,
    StatusCallback,
    ZipEntryReadContext,
    iter_zip_entry_raw_data,
    read_plain_source_file,
    zip_acquisition_fingerprint,
    zip_member_admission,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

from .pickle_spool import PickleSpool

if TYPE_CHECKING:
    from .acquisition_boundary import BoundContainerCapture


@dataclass(frozen=True, slots=True)
class SourceInputRecord:
    coordinate: str
    data: RawSessionData | None
    raw_id: str | None = None
    entry_ordinal: int | None = None
    split_index: int | None = None
    member_disposition: str | None = None
    member_name: str | None = None
    diagnostic: str | None = None
    member_refusal_code: str | None = None
    member_count: int | None = None
    input_blob_hash: str | None = None
    input_blob_size: int | None = None
    input_publication_receipt_id: str | None = None
    captured_input_identity: CapturedSourceInputIdentity | None = None
    input_source_name: str | None = None
    enumeration_fingerprint: str | None = None
    enumeration_complete: bool = False
    file_observation: tuple[int, int, int, int, int] | None = None


@contextmanager
def _container_input(
    blob_store: BlobStore,
    blob_hash: str,
    input_capture: BoundContainerCapture | None,
) -> Iterator[IO[bytes]]:
    if input_capture is not None:
        if input_capture.blob_hash != blob_hash:
            raise ValueError("retained decoder differs from its accepted input")
        input_capture.stream.seek(0)
        yield input_capture.stream
    else:
        with blob_store.open(blob_hash) as stream:
            yield stream


def iter_retained_source_records(
    *,
    enumeration_fingerprint: str,
    source_path: str,
    blob_hash: str,
    blob_size: int,
    blob_store: BlobStore,
    source_name: str | None = None,
    captured_identity: CapturedSourceInputIdentity | None = None,
    on_member_disposition: Callable[[int, str, str, str], None] | None = None,
    input_capture: BoundContainerCapture | None = None,
    file_mtime: str | None = None,
    observation_callback: ObservationCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> Generator[SourceInputRecord, None, int]:
    """Use canonical bounded decoders over the exact retained physical blob.

    ``source_path`` supplies identity and declared filename semantics only.
    ZIP entries are opened by their admitted ZipInfo, preserving duplicate
    names as separate central-directory coordinates.
    """
    logical_path = Path(source_path)
    source = Source(name=source_name or "machine-ingest", path=logical_path)
    binding = database_member_for_filename(logical_path.name)
    # Source aliases (``codex-state``, ``aistudio``) resolve to their origin.
    declared_provider = Provider.from_string(canonical_acquisition_provider(source.name, source_name=source.name))
    # The declared source location binds (not a sniffed dominant provider), so
    # an operator-imported archive stays unbound and classifies its members.
    location_binding = bound_location_provider(declared_provider)
    is_zip = input_capture is not None or logical_path.suffix.lower() == ".zip"
    if not is_zip:
        with blob_store.open(blob_hash) as probe:
            try:
                with zipfile.ZipFile(probe):
                    is_zip = True
            except zipfile.BadZipFile:
                pass
    # Native database filename scope applies to database bytes. A physical
    # ZIP with that basename keeps its declared source location and its own
    # per-member provider evidence.
    provider = declared_provider if is_zip else (binding.provider if binding is not None else declared_provider)
    if not is_zip:
        # A declared database member of another origin is not this
        # location's material; the decode below reads through the boundary.
        refuse_declared_foreign(logical_path.name, declared_provider)
        data = read_plain_source_file(
            SourceReadContext(
                source=source,
                path=logical_path,
                file_mtime=None,
                provider_hint=provider,
                blob_store=blob_store,
                retained_blob=ArtifactIdentity(blob_hash, blob_size),
                captured_input_identity=captured_identity,
            )
        )
        # A plain source has no central directory. Its source-43 record is
        # still enumerated and digested, but it must not be mistaken for a
        # ZIP member ordinal by retained-member completeness checks.
        yield SourceInputRecord('["physical-file-v1",0]', data)
        return 1

    if captured_identity is None:
        raise RetainedZipMembershipUnprovedError("retained ZIP input lacks its original captured namespace receipt")

    # Both channels are driven by attacker-controlled central-directory
    # metadata, so they count exactly under a bounded detail sample rather than
    # accumulating one string per member (and then joining them all).
    rejected = BoundedMemberReport()
    unselected = BoundedMemberReport()
    dispositions: PickleSpool[tuple[int, str, str, str, str | None]] | None = None
    try:
        with _container_input(blob_store, blob_hash, input_capture) as physical, zipfile.ZipFile(physical) as archive:
            entries = archive.infolist()
            # A retained physical blob carries no provider-bearing directory: the
            # database-member binding above only resolves for a declared database
            # export, so an account export ZIP always arrives as UNKNOWN. Recover
            # the real provider the way the inbox route does, then let its own
            # artifact declarations decide which members are material. A genuinely
            # mixed ZIP keeps UNKNOWN and admits any declared artifact path, so no
            # export asset, Antigravity protobuf, brain Markdown or tool-result
            # sidecar is dropped while enumeration reports itself complete
            # (polylogue-ojxpn).
            with zip_member_admission(
                archive, logical_path, entries, provider, container_blob_hash=blob_hash
            ) as admission:

                def record_rejected(entry: zipfile.ZipInfo, reason: str, code: str) -> None:
                    nonlocal dispositions
                    if dispositions is None:
                        dispositions = PickleSpool()
                    rejected.record(reason)
                    dispositions.append((ordinal, entry.filename, "refused", reason, code))

                def record_unselected(entry: zipfile.ZipInfo, reason: str) -> None:
                    nonlocal dispositions
                    if dispositions is None:
                        dispositions = PickleSpool()
                    unselected.record(f"{entry.filename}: {reason}")
                    dispositions.append((ordinal, entry.filename, "unselected", reason, None))

                validator = ZipEntryValidator(admission.provider_hint, cursor_state=None, zip_path=logical_path)
                for ordinal, entry in enumerate(entries):
                    selected = tuple(
                        validator.filter_entries(
                            (entry,),
                            allowed_path=admission.allowed_path,
                            on_unselected=record_unselected,
                        )
                    )
                    if not selected:
                        continue
                    if entry.file_size == 0:
                        # Verify the selected empty member rather than trusting
                        # central metadata as proof that its read completed.
                        with open_zip_entry(archive, entry) as empty:
                            if empty.read(1):
                                raise zipfile.BadZipFile("empty member differs from its central-directory size")
                        record_unselected(entry, "member is empty")
                        continue
                    # Under a residual UNKNOWN the container hint cannot name the family
                    # that owns this member, so its ``raw-only`` declaration would not
                    # fire and arbitrary binary bytes would take the JSON split route.
                    context = ZipEntryReadContext(
                        source,
                        logical_path,
                        entry,
                        file_mtime,
                        admission.entry_provider_hint(entry, entry_ordinal=ordinal),
                        blob_store,
                        observation_callback=observation_callback,
                        status_callback=status_callback,
                        bound_provider=location_binding,
                        captured_input_identity=captured_identity,
                        container_blob_hash=blob_hash,
                        decoder_fingerprint=enumeration_fingerprint,
                        entry_ordinal=ordinal,
                    )
                    try:
                        # A member's splits leave only once the whole member validated;
                        # a foreign member raises before any is yielded.
                        produced = False
                        for data in iter_zip_entry_raw_data(archive, context):
                            split = (
                                data.captured_zip_coordinate.split_index
                                if data.captured_zip_coordinate is not None
                                else data.source_index or 0
                            )
                            if data.captured_zip_coordinate is None:
                                raise RetainedZipMembershipUnprovedError("ZIP decoder lost its accepted member receipt")
                            mode = data.addressing_mode
                            if mode not in {
                                MemberAddressingMode.WHOLE_MEMBER,
                                MemberAddressingMode.ELEMENT_OF_CONTAINER,
                            }:
                                raise ValueError("retained ZIP record has no exact addressing mode")
                            if data.blob_hash is None:
                                raise ValueError("retained ZIP decoder did not retain its raw bytes")
                            produced = True
                            yield SourceInputRecord(
                                zip_member_record_coordinate(
                                    entry_ordinal=ordinal, split_index=split, addressing_mode=mode
                                ),
                                data.model_copy(
                                    update={
                                        "source_index": zip_member_source_index(
                                            entry_ordinal=ordinal, split_index=split
                                        )
                                    }
                                ),
                                captured_zip_member_raw_id(data.captured_zip_coordinate, data.blob_hash),
                                ordinal,
                                split,
                                member_count=len(entries),
                            )
                        if not produced:
                            record_unselected(entry, "decoded member contains no retained records")
                    except ForeignOriginContentError as exc:
                        # The declared source binds; a foreign member is a typed
                        # refusal in the member denominator, never a retained raw.
                        record_rejected(entry, f"{exc.code}: {exc}", exc.code)
                    except ContentIdentityRefusal as exc:
                        # The member cannot be stored: a recorded refusal, not an
                        # aborted acquisition of the whole ZIP.
                        record_rejected(entry, f"content_identity_refused: {exc}", "content_identity_refused")
        for ordinal, member_name, disposition, diagnostic, refusal_code in () if dispositions is None else dispositions:
            if on_member_disposition is None:
                continue
            record = SourceInputRecord(
                coordinate=json.dumps(["zip-member-v1", ordinal], separators=(",", ":")),
                data=None,  # disposition records carry no raw payload
                entry_ordinal=ordinal,
                member_disposition=disposition,
                member_name=member_name,
                diagnostic=diagnostic,
                member_refusal_code=refusal_code,
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
        return len(entries)
    finally:
        if dispositions is not None:
            dispositions.close()


def iter_captured_zip_input(
    *,
    capture: BoundContainerCapture,
    source_path: str,
    source_name: str,
    blob_store: BlobStore,
    file_mtime: str | None = None,
    observation_callback: ObservationCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> Generator[SourceInputRecord, None, None]:
    """Transport one exact input and its normal-exhaustion denominator.

    The initial record hands off the already accepted private publication.
    The final record is emitted only after the canonical decoder reaches
    normal exhaustion, including member CRC checks and explicit dispositions.
    Neither a container hash nor a yielded prefix proves completion.
    """
    provider = Provider.from_string(canonical_acquisition_provider(None, source_name=source_name))
    fingerprint = zip_acquisition_fingerprint(provider)
    blob_hash, blob_size, receipt_id = capture.retain()
    yield SourceInputRecord(
        coordinate='["physical-input-v1",0]',
        data=None,
        input_blob_hash=blob_hash,
        input_blob_size=blob_size,
        input_publication_receipt_id=receipt_id,
        captured_input_identity=capture.captured_identity,
        input_source_name=source_name,
        enumeration_fingerprint=fingerprint,
        file_observation=capture.file_observation,
    )
    records = iter_retained_source_records(
        enumeration_fingerprint=fingerprint,
        source_path=source_path,
        blob_hash=blob_hash,
        blob_size=blob_size,
        blob_store=blob_store,
        source_name=source_name,
        captured_identity=capture.captured_identity,
        on_member_disposition=lambda *_: None,
        input_capture=capture,
        file_mtime=file_mtime,
        observation_callback=observation_callback,
        status_callback=status_callback,
    )
    try:
        member_count = yield from records
    finally:
        records.close()
    yield SourceInputRecord(
        coordinate='["physical-input-v1",0]',
        data=None,
        member_count=member_count,
        enumeration_complete=True,
    )
