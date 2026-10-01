"""Publication preserves the original lookup, not only its former row."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind, Provider
from polylogue.core.refs import EvidenceRef
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.reference_seal import ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import write_index_session
from tests.infra.reference_sessions import reference_session


@pytest.mark.parametrize("lookup", ["shared", "codex:shared", "codex-session:shared"])
def test_insert_preserves_original_session_alias_resolution(tmp_path: Path, lookup: str) -> None:
    with write_lease("test.reference-alias", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            original = write_index_session(archive, reference_session("shared"))
            archive.save_annotation("original-lookup", "session", lookup, "Retain the original lookup")
            archive.commit()
            if lookup == "shared":
                with pytest.raises(ReferenceSealError):
                    write_index_session(archive, reference_session("shared", provider=Provider.CHATGPT))
                assert archive.resolve_session_id(lookup) == original
                assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            else:
                write_index_session(archive, reference_session("shared", provider=Provider.CHATGPT))
                assert archive.resolve_session_id(lookup) == original
            assert archive.get_annotation("original-lookup") is not None


@pytest.mark.parametrize("unrelated_anchors", [0, 96])
def test_batch_reuses_one_durable_reference_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unrelated_anchors: int
) -> None:
    from polylogue.storage.sqlite import reference_seal

    census_calls = 0
    assertion_queries: list[str] = []
    original_open = reference_seal.PreparedIndexMutation._open_observer
    original = reference_seal._references_from_user

    def observed(conn: sqlite3.Connection) -> Iterator[str]:
        nonlocal census_calls
        census_calls += 1
        yield from original(conn)

    def open_observer(seal: reference_seal.PreparedIndexMutation, name: str, path: Path) -> sqlite3.Connection:
        connection = original_open(seal, name, path)
        if name == "user":
            connection.set_trace_callback(
                lambda sql: assertion_queries.append(sql) if sql.startswith("SELECT scope_ref,") else None
            )
        return connection

    with write_lease("test.reference-batch", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            existing = write_index_session(archive, reference_session("unrelated"))
            for number in range(unrelated_anchors):
                archive.save_annotation(str(number), "session", existing, "Retained unrelated note")
            archive.commit()
            monkeypatch.setattr(reference_seal, "_references_from_user", observed)
            monkeypatch.setattr(reference_seal.PreparedIndexMutation, "_open_observer", open_observer)
            with archive.index_mutation_scope():
                for number in range(8):
                    write_index_session(archive, reference_session(f"batch-{number}"))
            assert census_calls == 1
            assert len(assertion_queries) == 1
            assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 9


@pytest.mark.parametrize("block_position", [None, 0])
def test_parent_replacement_preserves_evidence_scoped_to_a_composed_child(
    tmp_path: Path, block_position: int | None
) -> None:
    with write_lease("test.reference-descendant", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            parent = write_index_session(archive, reference_session("parent", messages=(("prefix", "prefix"),)))
            child = write_index_session(
                archive, reference_session("child", parent="parent", messages=(("prefix", "prefix"), ("tail", "tail")))
            )
            message_id = archive._conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ?", (parent,)
            ).fetchone()[0]
            evidence = EvidenceRef(child, message_id, block_position).format()
            with closing(open_connection(tmp_path / "user.db", archive_root=tmp_path)) as user, user:
                upsert_assertion(
                    user,
                    assertion_id="inherited-evidence",
                    target_ref=f"session:{child}",
                    kind=AssertionKind.ANNOTATION,
                    key="inherited-evidence",
                    body_text="Retain the inherited locator",
                    author_kind="user",
                    evidence_refs=(evidence,),
                )
            with pytest.raises(ReferenceSealError):
                write_index_session(archive, reference_session("parent", messages=(("replacement", "replacement"),)))
            assert (
                archive._conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (parent,)).fetchone()[0]
                == message_id
            )


def test_cancelled_active_mutation_rolls_back_and_original_owner_closes(tmp_path: Path) -> None:
    import asyncio
    import threading

    from polylogue.core.compute_cancel import compute_cancel
    from tests.infra.archive_custody_probe import archive_custody_available

    cancelled = threading.Event()
    token = compute_cancel.set(cancelled)
    archive = None
    try:
        with write_lease("test.cancelled-reference-scope", archive_root=tmp_path):
            bootstrap_archive_root(tmp_path)
            archive = ArchiveStore.open_existing(tmp_path, read_only=False)
            try:
                with pytest.raises(asyncio.CancelledError):
                    with archive.index_mutation_scope():
                        write_index_session(archive, reference_session("cancelled-pending"))
                        assert archive._conn.in_transaction
                        cancelled.set()
                        archive._conn.set_progress_handler(lambda: int(cancelled.is_set()), 1)
                        # Admission refuses another mutation. The scope must
                        # still roll back and close on this original owner.
                        write_index_session(archive, reference_session("must-not-start"))
                assert not archive._conn.in_transaction
                assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
                with pytest.raises(asyncio.CancelledError):
                    archive.delete_sessions(("cancelled-pending",))
            finally:
                archive.close()
                archive = None
        assert archive_custody_available(tmp_path)
    finally:
        compute_cancel.reset(token)
        if archive is not None:
            archive.close()


def test_active_seal_refuses_an_archive_shadow_index(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.reference-shadow", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        shadow = tmp_path / "shadow.db"
        with closing(sqlite3.connect(shadow)) as connection, connection:
            connection.execute("CREATE TABLE shadow(value INTEGER)")
        with pytest.raises(ReferenceSealError):
            PreparedIndexMutation(shadow, archive_root=tmp_path)
