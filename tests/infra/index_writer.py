"""Declare transaction ownership for synthetic Index fixture producers."""

from __future__ import annotations

import sqlite3
import sys
from builtins import BaseExceptionGroup
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from polylogue.pipeline.services.ingest_batch._core import _prepare_ingest_payloads
from polylogue.pipeline.services.ingest_batch._core import _write_session as _lower_ingest_session
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedSessionRows,
    PreparedSessionShardRows,
    prepare_session_write,
    prepared_session_rows_from_shard,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.archive_tiers.write_shard import ShardIdentitySequence
from polylogue.storage.sqlite.reference_seal import (
    IndexMutationDestination,
    IndexMutationScope,
    PreparedIndexMutation,
    current_index_mutation_scope,
    index_path_for_connection,
)


@contextmanager
def fixture_index_mutation_scope(
    conn: sqlite3.Connection, *, archive_root: Path | None = None, standalone_memory: bool = False
) -> Iterator[IndexMutationScope]:
    """Declare the destination for one actual fixture producer commit window."""
    current = current_index_mutation_scope()
    if current is not None:
        current.require_new_work(conn)
        yield current
        return
    if standalone_memory:
        if archive_root is not None:
            raise ValueError("a standalone memory fixture cannot declare an archive root")
        with IndexMutationDestination.standalone_memory(conn).mutation_scope(conn) as scope:
            yield scope
        return
    path = index_path_for_connection(conn)
    root = archive_root if archive_root is not None else path.parent
    if ".index-generations" in path.parts or (path.parent / "generation.json").exists():
        raise ValueError("a generation fixture requires its actual ArchiveStore-owned scope")
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root

    with write_lease("test.fixture.archive", archive_root=root):
        # This named fixture declares its existing synthetic database as the
        # archive's active Index before full bootstrap. Durable tiers still
        # receive the production format/identity construction, and all later
        # writes must match this same resolved destination.
        conventional_index = root / "index.db"
        if path != conventional_index.resolve():
            if conventional_index.exists() or conventional_index.is_symlink():
                raise ValueError("fixture producer does not own this archive's active Index")
            conventional_index.symlink_to(path)
        bootstrap_archive_root(root)
        with PreparedIndexMutation(path, archive_root=root) as seal, seal.mutation_scope(conn) as scope:
            yield scope


def write_fixture_ingest_payload(conn: sqlite3.Connection, payload: Any, **kwargs: Any) -> Any:
    """Use canonical preparation, publication and receipt retirement for a fixture.

    This starts at an admitted ParsedSession, and does not claim provider-byte
    fidelity. Preparation finishes before this owner acquires writer custody.
    """
    import tempfile

    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from polylogue.storage.blob_publication import ArchiveBlobPublisher, consume_blob_publication_receipt
    from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root

    if current_index_mutation_scope() is not None:
        raise ValueError("fixture preparation must precede its Index mutation scope")
    path = index_path_for_connection(conn)
    root = path.parent
    with write_lease("test.fixture.bootstrap", archive_root=root):
        bootstrap_archive_root(root)
    publisher = kwargs.get("blob_publisher") or ArchiveBlobPublisher(root / "source.db", root / "blob")
    if not isinstance(publisher, ArchiveBlobPublisher):
        raise TypeError("fixture publication requires the actual ArchiveBlobPublisher")
    kwargs["blob_publisher"] = publisher
    kwargs["manage_transaction"] = False
    artifact = None
    try:
        with PreparedIndexMutation(path, archive_root=root) as seal:
            source = kwargs.get("source_conn")
            if source is not None:
                source_path = next((row[2] for row in source.execute("PRAGMA database_list") if row[1] == "main"), None)
                if not source_path:
                    raise ValueError("fixture Source connection must own an archive file")
                seal.require_source_target(Path(source_path))
            directory = Path(
                tempfile.mkdtemp(prefix="fixture-ingest-", dir=publisher._prepared_staging_directory(None))
            )
            artifact = PreparedJsonl.from_sessions(
                (payload.parsed_session,),
                blob_hash=payload.content_hash,
                artifact_directory=directory,
                publication_publisher=publisher,
            )
            payload.prepared_artifact = artifact
            payload.prepared_session_ordinal = 0
            payload.parsed_session = artifact.session_by_id(payload.session_id)
            index = seal.observer("index")
            index.row_factory = sqlite3.Row
            if kwargs.get("source_conn") is None:
                kwargs["source_conn"] = seal.observer("source")
            _prepare_ingest_payloads(index, kwargs["source_conn"], (payload,))
            seal.validate_observers_current()
            with write_lease("test.fixture.ingest", archive_root=root):
                artifact.publish_blobs(reference_seal=seal)
                with seal.mutation_scope(conn):
                    result = _lower_ingest_session(conn, payload, **kwargs)
                with (
                    closing(
                        open_isolated_write_connection(
                            root / "source.db", purpose="fixture blob receipt", archive_root=root
                        )
                    ) as source,
                    source,
                ):
                    source.execute("BEGIN IMMEDIATE")
                    for _ordinal, _attachment, claim in artifact.iter_attachment_claims():
                        consume_blob_publication_receipt(
                            source, claim.receipt.publication_id, bytes.fromhex(claim.receipt.blob_hash)
                        )
                    for _ordinal, _tool, claim, _present in artifact.iter_sidecar_claims():
                        consume_blob_publication_receipt(
                            source, claim.receipt.publication_id, bytes.fromhex(claim.receipt.blob_hash)
                        )
            return result
    finally:
        primary = sys.exception()
        failures: list[BaseException] = []
        if payload.prepared_write is not None:
            try:
                payload.prepared_write.close()
            except BaseException as failure:
                failures.append(failure)
            else:
                payload.prepared_write = None
        # A failed carrier close retains the artifact for creator-thread
        # settlement. Never remove its files underneath a live native owner.
        if artifact is not None and not failures:
            try:
                artifact.discard()
            except BaseException as failure:
                failures.append(failure)
            else:
                payload.prepared_artifact = None
                payload.prepared_session_ordinal = None
        if failures:
            raise BaseExceptionGroup(
                "fixture ingest cleanup remains unsettled", ([primary] if primary else []) + failures
            )


def write_fixture_index_session(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    archive_root: Path | None = None,
    standalone_memory: bool = False,
    **kwargs: Any,
) -> str:
    """Seed an explicitly declared archive fixture through its real writer.

    Named fixtures own a full bootstrap and a durable-reference proof. Genuine
    in-memory indexes declare that separate destination explicitly. An existing
    batch or ArchiveStore scope is borrowed only for its exact connection.
    """
    owned_prepared = None
    if "prepared_write" not in kwargs:
        rows = kwargs.pop("prepared_rows", None)
        if isinstance(rows, PreparedSessionShardRows):
            identities = rows.entry.content_identities
            if not isinstance(identities, ShardIdentitySequence):
                raise TypeError("fixture shard must carry its sealed identity sequence")
            rows = prepared_session_rows_from_shard(identities.path, rows.session_id)
        if rows is not None and not isinstance(rows, PreparedSessionRows):
            raise TypeError("fixture rows must be the canonical prepared carrier")
        owned_prepared = prepare_session_write(
            conn,
            session,
            merge_append=bool(kwargs.get("merge_append", False)),
            fallback_timestamp=kwargs.get("fallback_timestamp"),
            source_conn=kwargs.get("source_conn"),
            raw_id=kwargs.get("raw_id"),
            force_replace=bool(kwargs.get("force_replace", False)),
            prepared_rows=rows,
            signature_cache=kwargs.get("signature_cache"),
        )
        kwargs["prepared_write"] = owned_prepared
        if kwargs.get("content_hash") is None:
            kwargs["content_hash"] = owned_prepared.input_content_hash.hex()
        if kwargs.get("pending_input_content_hash") is None:
            kwargs["pending_input_content_hash"] = owned_prepared.input_content_hash.hex()
    try:
        scope = kwargs.pop("mutation_scope", None) or current_index_mutation_scope()
        if scope is not None:
            scope.require_connection(conn)
            kwargs["manage_transaction"] = False
            result = write_parsed_session_to_archive(conn, session, mutation_scope=scope, **kwargs)
        else:
            if not kwargs.pop("manage_transaction", True):
                raise ValueError("a fixture batch requires its explicitly owned Index transaction scope")
            with fixture_index_mutation_scope(
                conn, archive_root=archive_root, standalone_memory=standalone_memory
            ) as scope:
                result = write_parsed_session_to_archive(
                    conn, session, mutation_scope=scope, manage_transaction=False, **kwargs
                )
    except BaseException as primary:
        if owned_prepared is not None:
            try:
                owned_prepared.close()
            except BaseException as cleanup:
                primary.add_note(f"prepared fixture cleanup failed: {cleanup!r}")
        raise
    else:
        if owned_prepared is not None:
            owned_prepared.close()
        return result
