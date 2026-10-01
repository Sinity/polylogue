"""Declare transaction ownership for synthetic Index fixture producers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from polylogue.pipeline.services.ingest_batch._core import _write_session as _lower_ingest_session
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
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
    """Run the actual ingest lowering with its explicit fixture transaction."""
    kwargs["manage_transaction"] = False
    with fixture_index_mutation_scope(conn) as scope:
        scope.require_new_work(conn)
        return _lower_ingest_session(conn, payload, **kwargs)


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
    scope = kwargs.pop("mutation_scope", None) or current_index_mutation_scope()
    if scope is not None:
        scope.require_connection(conn)
        kwargs["manage_transaction"] = False
        return write_parsed_session_to_archive(conn, session, mutation_scope=scope, **kwargs)
    if not kwargs.pop("manage_transaction", True):
        raise ValueError("a fixture batch requires its explicitly owned Index transaction scope")
    with fixture_index_mutation_scope(conn, archive_root=archive_root, standalone_memory=standalone_memory) as scope:
        return write_parsed_session_to_archive(conn, session, mutation_scope=scope, manage_transaction=False, **kwargs)
