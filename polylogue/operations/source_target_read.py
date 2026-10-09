"""Bind durable block selectors to retained Source under their operation pin."""

from __future__ import annotations

from contextlib import AbstractContextManager, closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, BinaryIO

from polylogue.core.identity_law import block_id as make_block_id
from polylogue.sources.revision_backfill import ConnectionRetainedEnrichmentRead
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.write import ConnectionSessionSourceRead

if TYPE_CHECKING:
    import sqlite3

    from polylogue.archive.revision_authority import RawRevisionKind
    from polylogue.core.enums import Provider
    from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
    from polylogue.operations.operation_context import PinnedOperationRead
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.sources.revision_backfill import RetainedSessionRead
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

    def retained_children_rows(self, low: str, high: str) -> AbstractContextManager[sqlite3.Cursor]:
        from polylogue.sources.live.sidecar_resolution import _RETAINED_CHILDREN_SQL

        return self._statement(_RETAINED_CHILDREN_SQL, (low, high))

    def retained_sibling_rows(self, root_path: str, low: str, high: str) -> AbstractContextManager[sqlite3.Cursor]:
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
    provider_session_id: str
    dependencies: str


def _checkpoint(snapshot: PinnedOperationRead) -> None:
    if snapshot.checkpoint is not None:
        snapshot.checkpoint()


def _supplier_rows(archive: ArchiveStore, session_id: str) -> AbstractContextManager[sqlite3.Cursor]:
    # Enumerate Source memberships rather than sessions.raw_id: a union can
    # retain a block supplied by an earlier, separately accepted acquisition.
    from polylogue.storage.io_phase_metrics import connection_cursor

    return connection_cursor(
        archive.source_connection,
        "SELECT a.sequence,a.identity,r.raw_id,a.payload_sha256 "
        "FROM raw_sessions r LEFT JOIN accepted_marker_inputs a ON a.raw_id=r.raw_id "
        "WHERE r.logical_source_key=? OR EXISTS (SELECT 1 FROM raw_session_memberships m "
        "WHERE m.raw_id=r.raw_id AND m.logical_source_key=?) "
        "ORDER BY r.acquisition_generation DESC,r.raw_id,a.sequence DESC",
        (session_id, session_id),
    )


def _prepare_source_target_artifact(retained: RetainedSessionRead, raw_id: str, *, directory: Path) -> PreparedJsonl:
    """Use original bounded preparation with no Source publication or provider request."""
    from polylogue.core.enums import Provider
    from polylogue.sources.revision_backfill import (
        prepare_retained_jsonl_artifact,
        prepare_retained_non_json_artifact,
    )
    from polylogue.sources.sqlite_export import looks_like_logical_source_path

    provider, _blob_hash, source_path, _kind, _size = retained.raw_revision_descriptor(raw_id)
    if provider is Provider.ANTIGRAVITY and Path(source_path).suffix.lower() == ".pb":
        # That old parser requires a live language-server response. A
        # separately captured JSON or SQLite supplier can establish Source
        # membership; a current provider response cannot do so for this Raw.
        raise SourceTargetUnavailableError("retained trajectory has no immutable session interpretation")
    blob_path = retained.raw_revision_blob_path(raw_id)
    if blob_path is not None and looks_like_logical_source_path(blob_path):
        if provider not in {Provider.HERMES, Provider.ANTIGRAVITY}:
            raise SourceTargetUnavailableError("retained state material has no session block supplier")
        return prepare_retained_non_json_artifact(
            retained, raw_id, directory=directory, prepare_blob_publications=False
        )
    # JSON session grammars do not require a particular filename. Keep the
    # captured path for identity and classification while using the existing
    # streamed document/record parser and its operation-owned SQLite sink.
    return prepare_retained_jsonl_artifact(
        retained,
        raw_id,
        directory=directory,
        allow_generic_object_alias=True,
        prepare_blob_publications=False,
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
    from polylogue.sources.revision_backfill import (
        enrichment_dependency_digest,
    )
    from polylogue.storage.accepted_marker_inputs import (
        AcceptedMarkerInput,
        AcceptedMarkerInputReference,
        verified_marker_payload_from_blob,
    )
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC
    from polylogue.storage.sqlite.archive_tiers.write import (
        PreparedSessionWriteRefusedError,
        prepared_session_rows_from_shard,
    )

    archive = snapshot.archive
    retained = _PinnedRetainedRead(archive)
    source = archive.source_connection
    columns = tuple(column.name for column in BLOCKS_SPEC.insert_columns)
    with _supplier_rows(archive, session_id) as suppliers:
        for marker_row in suppliers:
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
            with TemporaryDirectory(prefix="polylogue-source-target-") as directory:
                try:
                    artifact = _prepare_source_target_artifact(retained, str(raw_id), directory=Path(directory))
                except SourceTargetUnavailableError:
                    continue
                try:
                    _checkpoint(snapshot)
                    if artifact.error is not None or artifact.shard_path is None:
                        continue
                    artifact.verify_files(full=True)
                    try:
                        rows = prepared_session_rows_from_shard(artifact.shard_path, session_id)
                    except PreparedSessionWriteRefusedError:
                        continue
                    for values in rows.block_rows:
                        _checkpoint(snapshot)
                        row = dict(zip(columns, values, strict=True))
                        occurrence = row["content_occurrence"]
                        if type(occurrence) is not int:
                            raise SourceTargetUnavailableError("retained block identity has an invalid occurrence")
                        source_id = make_block_id(
                            str(row["message_id"]),
                            content_identity=str(row["content_identity"]),
                            content_occurrence=occurrence,
                        )
                        if source_id == block_id:
                            from polylogue.core.identity_law import session_id as archive_session_id
                            from polylogue.core.sources import origin_from_provider

                            with closing(artifact.iter_sessions()) as sessions:
                                provider_session_id = next(
                                    value.provider_session_id
                                    for value in sessions
                                    if archive_session_id(
                                        origin_from_provider(value.source_name).value, value.provider_session_id
                                    )
                                    == session_id
                                )
                            dependencies = enrichment_dependency_digest(
                                provider=provider,
                                source_path=source_path,
                                captured_zip_coordinate=archive.raw_captured_zip_coordinate(str(raw_id)),
                                provider_session_ids=(provider_session_id,),
                                index_conn=archive._conn,
                                source_conn=source,
                                blob_root=archive.archive_root / "blob",
                                parser_sidecars=True,
                            )
                            snapshot.source_block_reads[key] = _BlockSupplier(
                                str(raw_id),
                                marker,
                                tuple(raw_row),
                                tuple(membership_row) if membership_row is not None else None,
                                provider_session_id,
                                dependencies,
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
    from polylogue.sources.revision_backfill import enrichment_dependency_digest
    from polylogue.storage.blob_store import BlobStore

    provider, blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(supplier.raw_id)
    if (
        not BlobStore(archive.archive_root / "blob").verify(blob_hash)
        or enrichment_dependency_digest(
            provider=provider,
            source_path=source_path,
            captured_zip_coordinate=archive.raw_captured_zip_coordinate(supplier.raw_id),
            provider_session_ids=(supplier.provider_session_id,),
            index_conn=archive._conn,
            source_conn=source,
            blob_root=archive.archive_root / "blob",
            parser_sidecars=True,
        )
        != supplier.dependencies
    ):
        raise SourceTargetChangedError("retained block supplier bytes or parser evidence changed before durable apply")
