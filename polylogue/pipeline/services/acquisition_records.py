"""Acquisition record models and raw-record preparation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typing_extensions import TypedDict

from polylogue.core.enums import Provider
from polylogue.core.provider_identity import canonical_acquisition_provider, captured_hermes_profile_key
from polylogue.core.raw_coordinates import captured_zip_member_raw_id
from polylogue.core.raw_failure_evidence import MissingProfileIdentityError
from polylogue.core.sources import origin_from_provider
from polylogue.security.excision_policy import ExcisionPolicySnapshot
from polylogue.sources.parsers.base import RawSessionData
from polylogue.sources.sqlite_snapshot import hermes_profile_raw_id
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.raw_admission import (
    PendingPreParseRawAdmissionRequest,
    acquisition_timestamp_ms,
)
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id


class ScanCounts(TypedDict):
    scanned: int
    errors: int


class ScanResult:
    """Result of scanning raw payloads from sources without persisting them."""

    def __init__(self) -> None:
        self.counts: ScanCounts = {
            "scanned": 0,
            "errors": 0,
        }
        self.cursors: dict[str, CursorStatePayload] = {}


def make_raw_record(
    raw_data: RawSessionData,
    source_name: str,
    *,
    blob_root: Path | None = None,
    blob_store: BlobStore | None = None,
    acquired_at: str | None = None,
) -> RawSessionRecord:
    """Prepare a raw session record from acquisition data.

    Blob identity is content-addressed, while raw identity includes the
    acquisition coordinate. Hermes retains its profile-scoped identity because
    its provider-native session IDs are profile-local.
    """
    blob_hash: str | None = None
    if raw_data.staged_payload is not None:
        from polylogue.core.compute_cancel import check_compute_cancelled
        from polylogue.paths import blob_store_root
        from polylogue.storage.blob_publication import ArchiveBlobPublisher, publication_receipt_id

        blob_store = blob_store or BlobStore(blob_root or blob_store_root())
        staged = raw_data.staged_payload
        owner_root = blob_store.root.resolve()
        publisher_id = blob_store.publisher_id if isinstance(blob_store, ArchiveBlobPublisher) else None
        if staged.adopted is None:
            prepared = None
            try:
                staged.seal.verify(staged.path, full=True)
                prepared = blob_store.prepare_from_path(staged.path, heartbeat=check_compute_cancelled)
                staged.seal.verify(staged.path, full=False)
                if prepared.hash_hex != staged.seal.sha256 or prepared.size_bytes != staged.seal.size:
                    raise ValueError("staged raw copy disagrees with its sealed capture")
                if isinstance(blob_store, ArchiveBlobPublisher):
                    blob_hash, blob_size = blob_store.queue_prepared(prepared)
                else:
                    blob_hash, blob_size = blob_store.publish_prepared(prepared)
                prepared = None
                staged.adopted = (
                    owner_root,
                    publisher_id,
                    blob_hash,
                    blob_size,
                    publication_receipt_id(blob_store, blob_hash),
                )
            finally:
                if prepared is not None:
                    blob_store.discard_prepared(prepared)
                staged.discard()
        adopted_root, adopted_publisher, blob_hash, blob_size, receipt = staged.adopted
        if (adopted_root, adopted_publisher) != (owner_root, publisher_id):
            raise ValueError("staged raw capture belongs to another creator publication owner")
        raw_data.staged_payload = None
        raw_data.blob_hash, raw_data.blob_size = blob_hash, blob_size
        raw_data.blob_publication_receipt_id = receipt
    if raw_data.blob_hash is not None:
        blob_hash = raw_data.blob_hash
        blob_size = raw_data.blob_size or 0
    elif raw_data.raw_bytes:
        # Explicit in-memory captures use the same creator publication owner.
        from polylogue.paths import blob_store_root

        resolved_blob_root = blob_root or blob_store_root()
        blob_store = blob_store or BlobStore(resolved_blob_root)
        blob_hash, blob_size = blob_store.write_from_bytes(raw_data.raw_bytes)
        from polylogue.storage.blob_publication import publication_receipt_id

        raw_data.blob_publication_receipt_id = publication_receipt_id(blob_store, blob_hash)
    else:
        raise ValueError("RawSessionData has neither blob_hash nor raw_bytes")

    acquired_at = acquired_at if acquired_at is not None else datetime.now(timezone.utc).isoformat()
    source_capture_mode = Provider.from_string(canonical_acquisition_provider(None, source_name=source_name))
    source_name = canonical_acquisition_provider(
        str(raw_data.provider_hint) if raw_data.provider_hint is not None else None,
        source_name=source_name,
    )
    capture_mode = source_capture_mode
    if capture_mode is Provider.UNKNOWN:
        capture_mode = Provider.from_string(source_name)
    if (
        source_name == "hermes"
        and raw_data.captured_zip_coordinate is None
        and (raw_data.captured_profile_source_path is None or raw_data.captured_profile_key is None)
    ):
        raise MissingProfileIdentityError("Hermes acquisition is missing its captured profile identity")
    if raw_data.captured_zip_coordinate is not None:
        if source_name == "hermes":
            namespace = raw_data.captured_zip_coordinate.profile_namespace
            if namespace is None:
                if raw_data.captured_profile_key is not None or raw_data.captured_profile_source_path is not None:
                    raise ValueError("ZIP member has profile evidence without an accepted namespace")
                # Exact member bytes and coordinate remain retained. The
                # retained parser records the distinct typed profile gap.
            elif captured_hermes_profile_key(Path(namespace)) != raw_data.captured_profile_key:
                raise ValueError("Hermes ZIP acquisition has mismatched profile evidence")
        raw_id = captured_zip_member_raw_id(raw_data.captured_zip_coordinate, blob_hash)
    elif source_name == "hermes":
        assert raw_data.captured_profile_source_path is not None
        assert raw_data.captured_profile_key is not None
        raw_id = hermes_profile_raw_id(
            raw_data.source_path,
            raw_data.source_index or 0,
            blob_hash,
            identity_path=Path(raw_data.captured_profile_source_path),
            profile_identity=raw_data.captured_profile_key,
        )
    else:
        raw_id = deterministic_raw_session_id(
            # Raw identity names the acquisition coordinate.  Parser hints
            # refine the retained origin later, but must never create a new
            # observation for the same captured bytes.
            origin_from_provider(capture_mode),
            raw_data.source_path,
            raw_data.source_index or 0,
            bytes.fromhex(blob_hash),
        )

    return RawSessionRecord(
        raw_id=raw_id,
        blob_hash=blob_hash,
        blob_publication_receipt_id=raw_data.blob_publication_receipt_id,
        capture_mode=capture_mode,
        source_name=source_name,
        source_path=raw_data.source_path,
        canonical_source_path=raw_data.canonical_source_path,
        captured_profile_key=raw_data.captured_profile_key,
        captured_zip_coordinate=raw_data.captured_zip_coordinate,
        captured_file_observation=raw_data.captured_file_observation,
        source_index=(
            raw_data.captured_zip_coordinate.source_index
            if raw_data.captured_zip_coordinate is not None
            else raw_data.source_index
        ),
        addressing_mode=raw_data.addressing_mode,
        content_identity=raw_data.content_identity,
        blob_size=blob_size,
        acquired_at=acquired_at,
        file_mtime=raw_data.file_mtime,
        sidecar_snapshot=raw_data.sidecar_snapshot,
    )


def pending_pre_parse_raw_admission_request(
    record: RawSessionRecord,
    *,
    policy_snapshot: ExcisionPolicySnapshot | None = None,
) -> PendingPreParseRawAdmissionRequest:
    """Map an acquisition record into the canonical pending-admission input."""
    source_name = record.source_name or Provider.UNKNOWN.value
    origin = origin_from_provider(Provider.from_string(source_name))
    blob_hash_hex = record.blob_hash
    if blob_hash_hex is None:
        raise ValueError("acquisition record must carry a content-addressed blob hash")
    try:
        blob_hash = bytes.fromhex(blob_hash_hex)
    except ValueError as exc:
        raise ValueError("acquisition record blob_hash must be hexadecimal") from exc
    if record.file_mtime is None:
        file_mtime_ms = None
    else:
        try:
            file_mtime_ms = acquisition_timestamp_ms(record.file_mtime)
        except ValueError:
            # File mtime is optional observation metadata; malformed mtime is
            # unknown, never substituted for the required acquisition clock.
            file_mtime_ms = None
    return PendingPreParseRawAdmissionRequest(
        origin=origin,
        capture_mode=record.capture_mode,
        source_path=record.source_path,
        canonical_source_path=record.frozen_canonical_source_path(),
        captured_profile_key=record.captured_profile_key,
        captured_zip_coordinate=record.captured_zip_coordinate,
        source_item=record.source_item,
        source_index=record.source_index or 0,
        blob_hash=blob_hash,
        blob_size=record.blob_size,
        acquired_at_ms=acquisition_timestamp_ms(record.acquired_at),
        file_mtime_ms=file_mtime_ms,
        raw_id=record.raw_id,
        addressing_mode=record.addressing_mode,
        content_identity=record.content_identity,
        blob_publication_receipt_id=record.blob_publication_receipt_id,
        policy_snapshot=policy_snapshot,
    )


__all__ = ["ScanCounts", "ScanResult", "make_raw_record", "pending_pre_parse_raw_admission_request"]
