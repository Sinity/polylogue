"""Bind durable block selectors to retained Source under their operation pin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from polylogue.core.identity_law import block_id as make_block_id
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

if TYPE_CHECKING:
    from polylogue.operations.operation_context import PinnedOperationRead
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


class SourceTargetUnavailableError(ValueError):
    """A selector has no retained Source supplier in the pinned revision."""


class SourceTargetChangedError(ValueError):
    """The supplying Source revision moved before the durable reference write."""


@dataclass(frozen=True, slots=True)
class _BlockSupplier:
    raw_id: str
    marker: tuple[object, ...]
    raw: tuple[object, ...]
    membership: tuple[object, ...]
    parser_sidecars: str


def _checkpoint(snapshot: PinnedOperationRead) -> None:
    if snapshot.checkpoint is not None:
        snapshot.checkpoint()


def _supplier_rows(archive: ArchiveStore, session_id: str):
    # Enumerate Source memberships rather than sessions.raw_id: a union can
    # retain a block supplied by an earlier, separately accepted acquisition.
    return archive.source_connection.execute(
        "SELECT a.sequence,a.identity,a.raw_id,a.payload_sha256 "
        "FROM raw_session_memberships m JOIN accepted_marker_inputs a ON a.raw_id=m.raw_id "
        "WHERE m.logical_source_key=? ORDER BY a.sequence DESC",
        (session_id,),
    )


def bind_source_block(snapshot: PinnedOperationRead, *, session_id: str, block_id: str) -> None:
    """Prove the exact selected stable ID from an accepted retained Raw parse.

    The verified Source carrier establishes admission and session membership;
    the retained parser and canonical lowering establish the selected block.
    Neither an Index digest nor a Source file epoch alone grants authority.
    Cached supplier coordinates live only as long as this operation pin.
    """
    key = (session_id, block_id)
    if key in snapshot.source_block_reads:
        return
    from polylogue.sources.dispatch import is_jsonl_source_path
    from polylogue.sources.revision_backfill import (
        _retained_parser_sidecar_digest,
        prepare_retained_jsonl_artifact,
        prepare_retained_non_json_artifact,
    )
    from polylogue.storage.accepted_marker_inputs import (
        AcceptedMarkerInput,
        AcceptedMarkerInputReference,
        verified_marker_payload_from_blob,
    )
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC
    from polylogue.storage.sqlite.archive_tiers.write import prepared_session_rows_from_shard

    archive = snapshot.archive
    source = archive.source_connection
    columns = tuple(column.name for column in BLOCKS_SPEC.insert_columns)
    for marker_row in _supplier_rows(archive, session_id):
        _checkpoint(snapshot)
        marker = tuple(marker_row)
        sequence, identity, raw_id, payload_digest = marker
        accepted = AcceptedMarkerInput(
            "",
            int(sequence),
            AcceptedMarkerInputReference(str(raw_id), str(identity), str(payload_digest)),
        )
        payload = verified_marker_payload_from_blob(source, accepted)
        try:
            if session_id not in payload.iter_items("sessions.item.session_id"):
                continue
        finally:
            payload.close()
        raw_row = source.execute("SELECT * FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
        membership_row = source.execute(
            "SELECT * FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?",
            (raw_id, session_id),
        ).fetchone()
        if raw_row is None or membership_row is None:
            continue
        provider, blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(str(raw_id))
        if not BlobStore(archive.archive_root / "blob").verify(blob_hash):
            raise SourceTargetUnavailableError("retained block supplier bytes are unavailable")
        sidecars = _retained_parser_sidecar_digest(source, provider=provider, source_path=source_path)
        with TemporaryDirectory(prefix="polylogue-source-target-") as directory:
            prepare = (
                prepare_retained_jsonl_artifact
                if is_jsonl_source_path(source_path) or Path(source_path).suffix.lower() == ".json"
                else prepare_retained_non_json_artifact
            )
            artifact = prepare(
                archive,
                str(raw_id),
                directory=Path(directory),
                prepare_blob_publications=False,
            )
            try:
                _checkpoint(snapshot)
                if artifact.error is not None or artifact.shard_path is None:
                    continue
                artifact.verify_files(full=True)
                try:
                    rows = prepared_session_rows_from_shard(artifact.shard_path, session_id)
                except KeyError:
                    continue
                for values in rows.block_rows:
                    _checkpoint(snapshot)
                    row = dict(zip(columns, values, strict=True))
                    source_id = make_block_id(
                        str(row["message_id"]),
                        content_identity=str(row["content_identity"]),
                        content_occurrence=int(row["content_occurrence"]),
                    )
                    if source_id == block_id:
                        snapshot.source_block_reads[key] = _BlockSupplier(
                            str(raw_id),
                            marker,
                            tuple(raw_row),
                            tuple(membership_row),
                            sidecars,
                        )
                        return
            finally:
                artifact.discard()
    raise SourceTargetUnavailableError("block target has no accepted retained Source supplier")


def revalidate_source_block(
    snapshot: PinnedOperationRead,
    archive: ArchiveStore,
    *,
    session_id: str,
    block_id: str,
) -> None:
    """Check the original supplier against current currency immediately before apply."""
    _checkpoint(snapshot)
    if ArchiveIdentity.resolve_location(ArchiveLocation.resolve(archive.archive_root)) != snapshot.identity:
        raise SourceTargetChangedError("archive changed after the block selector was pinned")
    supplier = snapshot.source_block_reads.get((session_id, block_id))
    if not isinstance(supplier, _BlockSupplier):
        raise SourceTargetUnavailableError("block target was not bound to pinned Source")
    source = archive.source_connection
    marker = source.execute(
        "SELECT sequence,identity,raw_id,payload_sha256 FROM accepted_marker_inputs WHERE sequence=?",
        (supplier.marker[0],),
    ).fetchone()
    raw = source.execute("SELECT * FROM raw_sessions WHERE raw_id=?", (supplier.raw_id,)).fetchone()
    membership = source.execute(
        "SELECT * FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?",
        (supplier.raw_id, session_id),
    ).fetchone()
    if (
        marker is None
        or tuple(marker) != supplier.marker
        or raw is None
        or tuple(raw) != supplier.raw
        or membership is None
        or tuple(membership) != supplier.membership
    ):
        raise SourceTargetChangedError("retained block supplier changed before durable apply")
    from polylogue.sources.revision_backfill import _retained_parser_sidecar_digest
    from polylogue.storage.blob_store import BlobStore

    provider, blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(supplier.raw_id)
    if (
        not BlobStore(archive.archive_root / "blob").verify(blob_hash)
        or _retained_parser_sidecar_digest(source, provider=provider, source_path=source_path)
        != supplier.parser_sidecars
    ):
        raise SourceTargetChangedError("retained block supplier bytes or parser evidence changed before durable apply")
