"""Declare transaction ownership for synthetic Index fixture producers."""

from __future__ import annotations

import sqlite3
import sys
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from polylogue.core.stage_admission import stage_write_admission
from polylogue.pipeline.services.ingest_batch._core import _prepare_ingest_payloads
from polylogue.pipeline.services.ingest_batch._core import _write_session as _lower_ingest_session
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.blob_store import blob_store_for_connection
from polylogue.storage.index_generation import ActiveWriterLease
from polylogue.storage.sqlite.archive_tiers.write import (
    ConnectionSessionSourceRead,
    PreparedSessionRows,
    PreparedSessionSourceRead,
    PreparedSessionWrite,
    prepare_session_write,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.connection_profile import native_sql_owner_for_connection
from polylogue.storage.sqlite.reference_seal import (
    IndexMutationDestination,
    IndexMutationScope,
    PreparedIndexMutation,
    current_index_mutation_scope,
    index_path_for_connection,
)


@contextmanager
def _fixture_writer_admission(conn: sqlite3.Connection, actor: str, root: Path) -> Iterator[None]:
    from polylogue.storage.sqlite.write_lease import write_lease

    owner = native_sql_owner_for_connection(conn)
    admission = (
        write_lease(actor, archive_root=root)
        if owner is None or owner.custody is None
        else write_lease(actor, archive_root=root, _custody=owner.custody, _sql_owner=owner)
    )
    with admission:
        yield


def close_fixture_index_connection(conn: sqlite3.Connection) -> None:
    owner = native_sql_owner_for_connection(conn)
    if owner is None:
        conn.close()
    else:
        owner.close()


@contextmanager
def fixture_index_connection(index_path: Path) -> Iterator[sqlite3.Connection]:
    """A lease-free, measured Index connection on a bootstrapped archive.

    Canonical session preparation refuses a caller that already holds writer
    custody, so fixture writers take their own lease after preparing; a test
    handing them a connection must not hold one.
    """
    from polylogue.storage.io_phase_metrics import connect_measured
    from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop

    run_off_event_loop(lambda: bootstrap_archive_root(index_path.parent))
    conn = connect_measured(index_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        close_fixture_index_connection(conn)


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
    from tests.infra.archive_templates import bootstrap_archive_root

    with _fixture_writer_admission(conn, "test.fixture.archive", root):
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
    from tests.infra.archive_templates import bootstrap_archive_root

    if current_index_mutation_scope() is not None:
        raise ValueError("fixture preparation must precede its Index mutation scope")
    path = index_path_for_connection(conn)
    root = path.parent
    with _fixture_writer_admission(conn, "test.fixture.bootstrap", root):
        bootstrap_archive_root(root)
    publisher = kwargs.get("blob_publisher") or ArchiveBlobPublisher(root / "source.db", root / "blob")
    if not isinstance(publisher, ArchiveBlobPublisher):
        raise TypeError("fixture publication requires the actual ArchiveBlobPublisher")
    kwargs["blob_publisher"] = publisher
    kwargs["manage_transaction"] = False
    artifact = None
    # The prepared view below is retired with its artifact; the caller's
    # payload keeps the session it supplied so it can be written again.
    supplied_session = payload.parsed_session
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

            # Blob reservations are prepared lease-free against the seal and
            # each page is flushed through stage admission, as the retained
            # owner does under the daemon writer.
            def admitted(actor: str, work: Callable[[], Any]) -> Any:
                with _fixture_writer_admission(conn, actor, root):
                    return work()

            with stage_write_admission(admitted):
                artifact.publish_blobs(reference_seal=seal)
            with _fixture_writer_admission(conn, "test.fixture.ingest", root):
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
                payload.parsed_session = supplied_session
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
    """Prepare on original fixture readers, then publish under that same seal.

    A borrowed Index transaction supplies its already prepared carrier. Named
    fixtures otherwise open their full parent before preparing; a supplied
    Source handle declares its target, never a second preparation authority.
    """
    scope = kwargs.pop("mutation_scope", None) or current_index_mutation_scope()
    source_target = kwargs.pop("source_conn", None)
    input_demand = kwargs.pop("input_demand", None)
    scope_factory = kwargs.pop("index_scope_factory", None)

    def require_source_target(seal: PreparedIndexMutation) -> None:
        if source_target is None:
            return
        with closing(source_target.execute("PRAGMA database_list")) as rows:
            path = next((row[2] for row in rows if row[1] == "main"), None)
        if not path:
            raise ValueError("fixture Source input must name its actual archive file")
        seal.require_source_target(Path(path))

    def prepare(index: sqlite3.Connection, source_read: Any, before_input: Any = None) -> Any:
        rows = kwargs.pop("prepared_rows", None)
        if rows is not None and not isinstance(rows, PreparedSessionRows):
            raise TypeError("fixture rows must be the canonical prepared carrier")
        prepared = prepare_session_write(
            index,
            session,
            merge_append=bool(kwargs.get("merge_append", False)),
            fallback_timestamp=kwargs.get("fallback_timestamp"),
            source_read=source_read,
            raw_id=kwargs.get("raw_id"),
            force_replace=bool(kwargs.get("force_replace", False)),
            prepared_rows=rows,
            signature_cache=kwargs.get("signature_cache"),
            before_input=before_input,
        )
        kwargs["prepared_write"] = prepared
        return prepared

    def publish(current: IndexMutationScope, source_read: Any) -> str:
        current.require_connection(conn)
        prepared = kwargs.get("prepared_write")
        if not isinstance(prepared, PreparedSessionWrite):
            raise TypeError("fixture publication requires its original prepared session write")
        if kwargs.get("content_hash") is None:
            kwargs["content_hash"] = prepared.input_content_hash.hex()
        if kwargs.get("pending_input_content_hash") is None:
            kwargs["pending_input_content_hash"] = prepared.input_content_hash.hex()
        kwargs.pop("manage_transaction", None)
        return write_parsed_session_to_archive(
            conn,
            session,
            mutation_scope=current,
            manage_transaction=False,
            source_read=source_read,
            **kwargs,
        )

    if scope is not None:
        scope.require_new_work(conn)
        if "prepared_write" not in kwargs:
            raise ValueError("fixture preparation must precede its Index transaction scope")
        if scope.seal is None:
            if source_target is not None or kwargs.get("raw_id") is not None:
                raise ValueError("standalone Index fixtures have no durable Source capability")
            return publish(scope, None)
        require_source_target(scope.seal)
        return publish(scope, ConnectionSessionSourceRead(scope.seal.observer("source")))
    if "prepared_write" in kwargs:
        raise ValueError("a supplied prepared fixture requires its original Index transaction scope")
    if not kwargs.pop("manage_transaction", True):
        raise ValueError("a fixture batch requires its explicitly owned Index transaction scope")
    owned_prepared = None
    primary: BaseException | None = None
    try:
        if standalone_memory:
            if source_target is not None or kwargs.get("raw_id") is not None:
                raise ValueError("standalone Index fixtures have no durable Source capability")
            owned_prepared = prepare(conn, None)
            with fixture_index_mutation_scope(conn, standalone_memory=True) as current:
                return publish(current, None)
        path = index_path_for_connection(conn)
        root = archive_root if archive_root is not None else path.parent
        if ".index-generations" in path.parts or (path.parent / "generation.json").exists():
            raise ValueError("a generation fixture requires its actual ArchiveStore-owned scope")
        from tests.infra.archive_templates import bootstrap_archive_root

        with _fixture_writer_admission(conn, "test.fixture.bootstrap", root):
            conventional_index = root / "index.db"
            if path != conventional_index.resolve():
                if conventional_index.exists() or conventional_index.is_symlink():
                    raise ValueError("fixture producer does not own this archive's active Index")
                conventional_index.symlink_to(path)
            bootstrap_archive_root(root)
        with PreparedIndexMutation(path, archive_root=root, input_demand=input_demand) as seal:
            require_source_target(seal)
            index = seal.observer("index")
            index.row_factory = sqlite3.Row
            with seal.original_read_snapshot(), seal.source_producer():
                source_read = PreparedSessionSourceRead(
                    seal,
                    blob_store=blob_store_for_connection(seal.observer("source")),
                )
                owned_prepared = prepare(index, source_read, seal.before_index_input)
            if scope_factory is not None:
                with scope_factory(seal) as current:
                    return publish(current, ConnectionSessionSourceRead(seal.observer("source")))
            with (
                _fixture_writer_admission(conn, "test.fixture.index.publish", root),
                seal.mutation_scope(conn) as current,
            ):
                return publish(current, ConnectionSessionSourceRead(seal.observer("source")))
    except BaseException as failure:
        primary = failure
        raise
    finally:
        # The original seal settles its native children before the prepared
        # carrier removes scratch files. Cleanup never hides the primary fault.
        if owned_prepared is not None:
            try:
                owned_prepared.close()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("fixture preparation and cleanup failed", [primary, cleanup]) from None
                raise


@contextmanager
def prepared_fixture_index_batch(
    conn: sqlite3.Connection, sessions: Sequence[ParsedSession], *, archive_root: Path
) -> Iterator[tuple[PreparedIndexMutation, tuple[PreparedSessionWrite, ...]]]:
    """Prepare all neutral session inputs before one original Index publication."""
    if current_index_mutation_scope() is not None:
        raise ValueError("fixture batch preparation must precede its Index transaction scope")
    path = index_path_for_connection(conn)
    exclusion = ActiveWriterLease(archive_root)
    exclusion.acquire()
    seal: PreparedIndexMutation | None = None
    pending: list[PreparedSessionWrite] = []

    def close_prepared() -> None:
        failures: list[BaseException] = []
        for prepared in tuple(pending):
            try:
                prepared.close()
            except BaseException as failure:
                failures.append(failure)
            else:
                pending[:] = [entry for entry in pending if entry is not prepared]
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("fixture batch physical payload cleanup failed", failures)

    try:
        with PreparedIndexMutation(path, archive_root=archive_root) as seal:
            # The original seal retires native children before this callback;
            # failed payload close keeps both the carrier and actual exclusion.
            seal.retain_publication_lifetime(exclusion, close_prepared)
            with seal.original_read_snapshot(), seal.source_producer():
                source_read = PreparedSessionSourceRead(
                    seal, blob_store=blob_store_for_connection(seal.observer("source"))
                )
                for session in sessions:
                    pending.append(
                        prepare_session_write(
                            seal.observer("index"),
                            session,
                            merge_append=False,
                            source_read=source_read,
                            before_input=seal.before_index_input,
                        )
                    )
            yield seal, tuple(pending)
    finally:
        if seal is None or not seal.publication_lifetime_bound:
            exclusion.close()


def write_fixture_prepared_session(
    conn: sqlite3.Connection,
    session: ParsedSession,
    *,
    inspect: Callable[[PreparedSessionWrite], None] | None = None,
    **kwargs: Any,
) -> str:
    """Prepare one write on an original seal, let the law inspect it, then publish.

    For laws about the prepared carrier itself (its union, its resolved root):
    preparation reads the seal's original Source snapshot and publication runs
    under that seal's admitted Index scope, as ``write_fixture_index_session``
    does internally. ``kwargs`` are the write's options (``raw_id``,
    ``merge_append``, ``force_replace``, ``content_hash``).
    """
    path = index_path_for_connection(conn)
    root = path.parent
    with PreparedIndexMutation(path, archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            prepared = prepare_session_write(
                seal.observer("index"),
                session,
                merge_append=bool(kwargs.get("merge_append", False)),
                source_read=PreparedSessionSourceRead(
                    seal, blob_store=blob_store_for_connection(seal.observer("source"))
                ),
                raw_id=kwargs.get("raw_id"),
                force_replace=bool(kwargs.get("force_replace", False)),
                before_input=seal.before_index_input,
            )
        try:
            if inspect is not None:
                inspect(prepared)
            with _fixture_writer_admission(conn, "test.fixture.prepared.publish", root), seal.mutation_scope(conn):
                return write_fixture_index_session(conn, session, prepared_write=prepared, **kwargs)
        finally:
            prepared.close()
