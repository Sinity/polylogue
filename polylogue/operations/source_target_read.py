"""Bind durable block selectors to retained Source under their operation pin."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, BinaryIO

from polylogue.core.identity_law import block_id as make_block_id
from polylogue.sources.revision_backfill import ConnectionRetainedEnrichmentRead
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.write import ConnectionSessionSourceRead

if TYPE_CHECKING:
    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.core.enums import Provider
    from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
    from polylogue.operations.operation_context import PinnedOperationRead
    from polylogue.sources.sidecar_evidence import SidecarResolver
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


class SourceTargetUnavailableError(ValueError):
    """A selector has no retained Source supplier in the pinned revision."""


class SourceTargetChangedError(ValueError):
    """The supplying Source revision moved before the durable reference write."""


class _PinnedRetainedRead(ConnectionSessionSourceRead, ConnectionRetainedEnrichmentRead):
    """Borrow the pin's existing Raw, sidecar and enrichment readers."""

    def __init__(self, archive: ArchiveStore) -> None:
        ConnectionSessionSourceRead.__init__(self, archive.source_connection)
        ConnectionRetainedEnrichmentRead.__init__(
            self,
            archive._conn,
            archive.source_connection,
            archive.archive_root / "blob",
        )
        self._archive = archive

    @property
    def archive_root(self) -> Path:
        return self._archive.archive_root

    def raw_revision_descriptor(self, raw_id: str) -> tuple[Provider, str, str, RawRevisionKind, int]:
        return self._archive.raw_revision_descriptor(raw_id)

    def raw_revision_blob_path(self, raw_id: str) -> Path | None:
        _, blob_hash, _, _, _ = self.raw_revision_descriptor(raw_id)
        return self._archive.blob_path_for_hash(blob_hash)

    def raw_profile_identity(self, raw_id: str) -> str | None:
        return self._archive.raw_profile_identity(raw_id)

    def raw_captured_zip_coordinate(self, raw_id: str) -> CapturedZipMemberCoordinate | None:
        return self._archive.raw_captured_zip_coordinate(raw_id)

    def raw_revision_file_mtime(self, raw_id: str) -> str | None:
        return self._archive.raw_revision_file_mtime(raw_id)

    def raw_native_id(self, raw_id: str) -> str | None:
        return self._archive.raw_native_id(raw_id)

    def raw_append_logical_key(self, raw_id: str) -> str | None:
        from polylogue.storage.sqlite.archive_tiers.source_write import read_raw_append_logical_key

        return read_raw_append_logical_key(self._connection, raw_id)

    def open_raw_revision_material(
        self,
        raw_id: str,
    ) -> AbstractContextManager[tuple[Provider, BinaryIO, str, RawRevisionKind]]:
        return self._archive.open_raw_revision_material(raw_id)

    def raw_revision_material(self, raw_id: str) -> tuple[Provider, bytes, str, RawRevisionKind]:
        return self._archive.raw_revision_material(raw_id)

    def open_raw_container_material(self, raw_id: str) -> AbstractContextManager[BinaryIO | None]:
        return self._archive.open_raw_container_material(raw_id)

    def retained_children_rows(self, low: str, high: str):
        from polylogue.sources.live.sidecar_resolution import _RETAINED_CHILDREN_SQL

        return self._statement(_RETAINED_CHILDREN_SQL, (low, high))

    def retained_sibling_rows(self, root_path: str, low: str, high: str):
        from polylogue.sources.live.sidecar_resolution import _RETAINED_SIBLINGS_SQL

        return self._statement(_RETAINED_SIBLINGS_SQL, (root_path, low, high))

    def retained_sidecar_resolver(self) -> SidecarResolver:
        from polylogue.sources.live.sidecar_resolution import RetainedSidecarResolver

        return RetainedSidecarResolver(self.archive_root, blob_root=self.blob_store.root, source_read=self)


@dataclass(frozen=True, slots=True)
class _BlockSupplier:
    raw_id: str
    marker: tuple[object, ...] | None
    raw: tuple[object, ...]
    membership: tuple[object, ...] | None
    parser_sidecars: str


def _checkpoint(snapshot: PinnedOperationRead) -> None:
    if snapshot.checkpoint is not None:
        snapshot.checkpoint()


def _supplier_rows(archive: ArchiveStore, session_id: str):
    # Enumerate Source memberships rather than sessions.raw_id: a union can
    # retain a block supplied by an earlier, separately accepted acquisition.
    return archive.source_connection.execute(
        "SELECT a.sequence,a.identity,r.raw_id,a.payload_sha256 "
        "FROM raw_sessions r LEFT JOIN accepted_marker_inputs a ON a.raw_id=r.raw_id "
        "WHERE r.logical_source_key=? OR EXISTS (SELECT 1 FROM raw_session_memberships m "
        "WHERE m.raw_id=r.raw_id AND m.logical_source_key=?) "
        "ORDER BY r.acquisition_generation DESC,r.raw_id,a.sequence DESC",
        (session_id, session_id),
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
    retained = _PinnedRetainedRead(archive)
    source = archive.source_connection
    columns = tuple(column.name for column in BLOCKS_SPEC.insert_columns)
    for marker_row in _supplier_rows(archive, session_id):
        _checkpoint(snapshot)
        sequence, identity, raw_id, payload_digest = marker_row
        marker = tuple(marker_row) if sequence is not None else None
        if marker is not None:
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
        if raw_row is None:
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
                retained,
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
                            tuple(membership_row) if membership_row is not None else None,
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
    marker = (
        source.execute(
            "SELECT sequence,identity,raw_id,payload_sha256 FROM accepted_marker_inputs WHERE sequence=?",
            (supplier.marker[0],),
        ).fetchone()
        if supplier.marker is not None
        else None
    )
    raw = source.execute("SELECT * FROM raw_sessions WHERE raw_id=?", (supplier.raw_id,)).fetchone()
    membership = source.execute(
        "SELECT * FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?",
        (supplier.raw_id, session_id),
    ).fetchone()
    if (
        (tuple(marker) if marker is not None else None) != supplier.marker
        or raw is None
        or tuple(raw) != supplier.raw
        or (tuple(membership) if membership is not None else None) != supplier.membership
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
