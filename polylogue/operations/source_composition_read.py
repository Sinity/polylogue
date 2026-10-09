"""Prove composed block membership by replaying retained Source into scratch."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from polylogue.operations.source_target_read import (
    SourceTargetChangedError,
    SourceTargetUnavailableError,
    _checkpoint,
    _PinnedRetainedRead,
    _prepare_source_target_artifact,
    _supplier_rows,
)
from polylogue.storage.io_phase_metrics import connect_measured, connection_cursor

if TYPE_CHECKING:
    from polylogue.operations.operation_context import PinnedOperationRead
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _json_blob(value: object) -> dict[str, str]:
    if not isinstance(value, bytes):
        raise TypeError("Source witness contains an unsupported SQL value")
    return {"sql_blob": value.hex()}


def _scope_fingerprint(
    snapshot: PinnedOperationRead, archive: ArchiveStore, scope: str, pending: sqlite3.Connection | None = None
) -> str:
    digest = hashlib.sha256()
    source = archive.source_connection
    with _supplier_rows(archive, scope) as rows:
        for marker in rows:
            _checkpoint(snapshot)
            raw_id = str(marker[2])
            with connection_cursor(
                source, "SELECT acquisition_generation,acquired_at_ms,* FROM raw_sessions WHERE raw_id=?", (raw_id,)
            ) as cursor:
                raw = cursor.fetchone()
            with connection_cursor(
                source, "SELECT * FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?", (raw_id, scope)
            ) as cursor:
                membership = cursor.fetchone()
            if raw is None:
                continue
            encoded = json.dumps(
                (tuple(marker), tuple(raw), tuple(membership) if membership is not None else None),
                ensure_ascii=True,
                separators=(",", ":"),
                default=_json_blob,
            ).encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            if pending is not None:
                with connection_cursor(
                    pending,
                    "INSERT OR IGNORE INTO pending_raws VALUES (?,?,?)",
                    (raw_id, raw[0], raw[1]),
                ):
                    pass
                if marker[0] is not None:
                    from polylogue.storage.accepted_marker_inputs import (
                        AcceptedMarkerInput,
                        AcceptedMarkerInputReference,
                        verified_marker_payload_from_blob,
                    )

                    accepted = AcceptedMarkerInput(
                        "", int(marker[0]), AcceptedMarkerInputReference(raw_id, str(marker[1]), str(marker[3]))
                    )
                    payload = verified_marker_payload_from_blob(source, accepted)
                    payload.close()
    return digest.hexdigest()


def _candidate_scopes(snapshot: PinnedOperationRead, session_id: str) -> tuple[str, ...]:
    # These are candidate coordinates only. Their relationships are recomputed
    # below from retained Source, never accepted as membership evidence.
    from polylogue.storage.sqlite.archive_tiers.write import _prefix_sharing_edge_sync

    scopes: list[str] = []
    seen: set[str] = set()
    current = session_id
    while current not in seen:
        _checkpoint(snapshot)
        seen.add(current)
        scopes.append(current)
        edge = _prefix_sharing_edge_sync(snapshot.archive._conn, current)
        if edge is None:
            break
        current = edge[0]
    return tuple(scopes)


@dataclass(frozen=True, slots=True)
class SourceCompositionRead:
    witnesses: sqlite3.Connection

    def revalidate(self, snapshot: PinnedOperationRead, archive: ArchiveStore) -> None:
        from polylogue.sources.revision_backfill import enrichment_dependency_digest
        from polylogue.storage.blob_store import BlobStore

        with connection_cursor(self.witnesses, "SELECT scope_id,fingerprint FROM scopes") as scopes:
            for scope, fingerprint in scopes:
                _checkpoint(snapshot)
                if _scope_fingerprint(snapshot, archive, str(scope)) != fingerprint:
                    raise SourceTargetChangedError("composing Source suppliers changed before durable apply")
        with connection_cursor(
            self.witnesses, "SELECT raw_id,provider_session_json,dependencies FROM suppliers"
        ) as rows:
            for raw_id, provider_session_json, dependencies in rows:
                _checkpoint(snapshot)
                provider, blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(str(raw_id))
                if not BlobStore(archive.archive_root / "blob").verify(blob_hash):
                    raise SourceTargetChangedError("composing Source supplier bytes changed before durable apply")
                provider_session_id = json.loads(str(provider_session_json))
                if (
                    enrichment_dependency_digest(
                        provider=provider,
                        source_path=source_path,
                        captured_zip_coordinate=archive.raw_captured_zip_coordinate(str(raw_id)),
                        provider_session_ids=(provider_session_id,),
                        index_conn=archive._conn,
                        source_conn=archive.source_connection,
                        blob_root=archive.archive_root / "blob",
                        parser_sidecars=True,
                    )
                    != dependencies
                ):
                    raise SourceTargetChangedError("composing Source parser evidence changed before durable apply")


def bind_source_composed_block(snapshot: PinnedOperationRead, *, session_id: str, block_id: str) -> None:
    from polylogue.sources.revision_backfill import enrichment_dependency_digest
    from polylogue.storage.blob_store import BlobStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_runtime_tier_probe
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.write import (
        locate_composed_message,
        prepare_session_write,
        write_parsed_session_to_archive,
    )
    from polylogue.storage.sqlite.reference_seal import IndexMutationDestination

    scopes = _candidate_scopes(snapshot, session_id)
    retained = _PinnedRetainedRead(snapshot.archive)
    with ExitStack() as resources:
        directory = Path(resources.enter_context(TemporaryDirectory(prefix="polylogue-source-composition-")))
        witnesses = connect_measured(str(directory / "source-witnesses.sqlite"))
        resources.callback(witnesses.close)
        witnesses.executescript(
            "CREATE TABLE scopes(scope_id TEXT PRIMARY KEY,fingerprint TEXT NOT NULL);"
            "CREATE TABLE pending_raws(raw_id TEXT PRIMARY KEY,generation INTEGER,acquired_at INTEGER NOT NULL);"
            "CREATE TABLE suppliers(raw_id TEXT,provider_session_json TEXT,dependencies TEXT NOT NULL,"
            "PRIMARY KEY(raw_id,provider_session_json));"
        )
        for scope_id in scopes:
            fingerprint = _scope_fingerprint(snapshot, snapshot.archive, scope_id, witnesses)
            with connection_cursor(witnesses, "INSERT INTO scopes VALUES (?,?)", (scope_id, fingerprint)):
                pass
        witnesses.commit()
        scratch_path = directory / "composition.sqlite"
        scratch = connect_measured(str(scratch_path))
        resources.callback(scratch.close)
        scratch.row_factory = sqlite3.Row
        initialize_runtime_tier_probe(scratch, ArchiveTier.INDEX, probe_path=scratch_path)
        destination = IndexMutationDestination.standalone(scratch_path)
        with connection_cursor(
            witnesses, "SELECT raw_id FROM pending_raws ORDER BY COALESCE(generation,0),acquired_at,raw_id"
        ) as raw_ids:
            for ordinal, (raw_id,) in enumerate(raw_ids):
                _checkpoint(snapshot)
                provider, blob_hash, source_path, _kind, _size = snapshot.archive.raw_revision_descriptor(str(raw_id))
                if not BlobStore(snapshot.archive.archive_root / "blob").verify(blob_hash):
                    raise SourceTargetUnavailableError("composing retained Source bytes are unavailable")
                prepared_directory = directory / f"prepared-{ordinal}"
                prepared_directory.mkdir()
                try:
                    artifact = _prepare_source_target_artifact(retained, str(raw_id), directory=prepared_directory)
                except SourceTargetUnavailableError:
                    continue
                try:
                    _checkpoint(snapshot)
                    if artifact.error is not None:
                        continue
                    artifact.verify_files(full=True)
                    with closing(artifact.iter_sessions()) as sessions:
                        for session in sessions:
                            _checkpoint(snapshot)
                            prepared = prepare_session_write(
                                scratch, session, merge_append=False, source_read=retained, raw_id=str(raw_id)
                            )
                            try:
                                if prepared.session_id not in scopes:
                                    continue
                                with destination.mutation_scope(scratch) as mutation_scope:
                                    write_parsed_session_to_archive(
                                        scratch,
                                        session,
                                        raw_id=str(raw_id),
                                        prepared_write=prepared,
                                        source_read=retained,
                                        mutation_scope=mutation_scope,
                                        manage_transaction=False,
                                    )
                                    mutation_scope.commit()
                            finally:
                                prepared.close()
                            dependencies = enrichment_dependency_digest(
                                provider=provider,
                                source_path=source_path,
                                captured_zip_coordinate=snapshot.archive.raw_captured_zip_coordinate(str(raw_id)),
                                provider_session_ids=(session.provider_session_id,),
                                index_conn=snapshot.archive._conn,
                                source_conn=snapshot.archive.source_connection,
                                blob_root=snapshot.archive.archive_root / "blob",
                                parser_sidecars=True,
                            )
                            with connection_cursor(
                                witnesses,
                                "INSERT OR REPLACE INTO suppliers VALUES (?,?,?)",
                                (raw_id, json.dumps(session.provider_session_id, ensure_ascii=True), dependencies),
                            ):
                                pass
                finally:
                    artifact.discard()
        with connection_cursor(scratch, "SELECT message_id FROM blocks WHERE block_id=?", (block_id,)) as blocks:
            selected = blocks.fetchone()
        if selected is None or locate_composed_message(scratch, session_id, str(selected[0])) is None:
            raise SourceTargetUnavailableError("retained Source does not contain this block in the composed scope")
        witnesses.commit()
        _checkpoint(snapshot)
        # The witness roster stays on disk until this exact operation pin
        # closes. The scratch transcript itself is no longer needed.
        scratch.close()
        snapshot.source_target_resources.enter_context(resources.pop_all())
        snapshot.source_block_reads[(session_id, block_id)] = SourceCompositionRead(witnesses)
