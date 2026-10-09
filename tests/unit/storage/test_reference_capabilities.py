"""Unavailable capability never substitutes for stale or unsettled authority."""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.refs import EvidenceRef
from polylogue.storage.io_phase_metrics import _MeasuredConnection, connection_cursor
from polylogue.storage.sqlite import connection_profile as profiles
from polylogue.storage.sqlite.reference_seal import (
    PreparedIndexMutation,
    ReferenceSealError,
    ReferenceSealStaleError,
    _resolve_target,
)
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement


@pytest.fixture
def capability_root(tmp_path: Path) -> Iterator[Path]:
    with write_lease("test.reference-capability", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        yield tmp_path


def _change_original_source(root: Path) -> None:
    with closing(profiles.open_source_tier_write_connection(root / "source.db", archive_root=root)) as connection:
        with connection_cursor(
            connection,
            "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
            "VALUES('foreign-input','unknown-export','synthetic/foreign',?,1,1)",
            (b"f" * 32,),
        ):
            pass
        connection.commit()


def test_source_only_capability_query_opens_no_additional_tier(
    capability_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with PreparedIndexMutation.source_only(archive_root=capability_root) as seal:
        identities = {tier: id(connection) for tier, connection in seal._observers.items()}
        assert set(identities) == {"source"}

        def unexpected(*_args: object, **_kwargs: object) -> sqlite3.Connection:
            raise AssertionError("capability query cannot open another observer")

        monkeypatch.setattr(PreparedIndexMutation, "_open_observer", unexpected)
        monkeypatch.setattr(profiles, "connect_measured", unexpected)
        for tier in ("index", "user", "audit"):
            assert not seal.has_tier_capability(tier)
        assert seal.has_tier_capability("source")
        with seal.original_read_snapshot():
            assert not seal.has_tier_capability("index")
            assert seal.has_tier_capability("source")
        assert {tier: id(connection) for tier, connection in seal._observers.items()} == identities


def test_capability_query_refuses_unknown_tier(capability_root: Path) -> None:
    with PreparedIndexMutation.source_only(archive_root=capability_root) as seal:
        with pytest.raises(ReferenceSealError):
            seal.has_tier_capability("indxe")


def test_stale_original_source_cannot_report_index_unavailable(capability_root: Path) -> None:
    with PreparedIndexMutation.source_only(archive_root=capability_root) as seal:
        _change_original_source(capability_root)
        with pytest.raises(ReferenceSealStaleError):
            seal.has_tier_capability("index")


def test_original_window_currency_owns_capability_query_completion(capability_root: Path) -> None:
    with PreparedIndexMutation.source_only(archive_root=capability_root) as seal:
        with pytest.raises(ReferenceSealStaleError):
            with seal.original_read_snapshot():
                assert not seal.has_tier_capability("index")
                _change_original_source(capability_root)
                assert not seal.has_tier_capability("index")


def test_closed_original_witness_cannot_report_unavailable_tier(capability_root: Path) -> None:
    seal = PreparedIndexMutation.source_only(archive_root=capability_root)
    seal.close()
    with pytest.raises(ReferenceSealError):
        seal.has_tier_capability("index")


def test_unsettled_original_child_retains_files_and_refuses_capability_query(
    capability_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_connect = sqlite3.connect

    def controlled(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if kwargs.get("factory", sqlite3.Connection) in (sqlite3.Connection, _MeasuredConnection):
            kwargs["factory"] = SettlementConnection
        return cast(sqlite3.Connection, original_connect(database, *args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", controlled)
    seal = PreparedIndexMutation.source_only(archive_root=capability_root)
    child = arm_settlement(seal._scratch)
    assert seal._scratch_directory is not None
    witness = Path(seal._scratch_directory.name) / "refs.db"
    try:
        with pytest.raises(profiles.NativeConnectionSettlementError):
            seal.close()
        assert witness.exists() and seal._cleanup_requested and not seal._closed
        with pytest.raises(ReferenceSealError):
            seal.has_tier_capability("index")
        child.allow_cleanup.set()
        seal.close()
        assert not witness.exists()
        with pytest.raises(ReferenceSealError):
            seal.has_tier_capability("index")
    finally:
        child.allow_cleanup.set()
        seal.close()


def test_foreign_creator_cannot_query_original_capability(capability_root: Path) -> None:
    with PreparedIndexMutation.source_only(archive_root=capability_root) as seal:
        failures: list[BaseException] = []

        def query() -> None:
            try:
                seal.has_tier_capability("index")
            except BaseException as failure:
                failures.append(failure)

        thread = threading.Thread(target=query)
        thread.start()
        thread.join()
    assert len(failures) == 1 and isinstance(failures[0], ReferenceSealError)


def test_durable_reference_resolution_uses_stable_block_identity(capability_root: Path) -> None:
    from tests.infra.storage_records import SessionBuilder

    builder = SessionBuilder(capability_root / "index.db", "stable-ref")
    builder.provider("codex").add_message(
        "message-1", role="user", text="hello", blocks=[{"type": "text", "text": "hello"}]
    ).save()
    session_id = builder.native_session_id()

    with closing(sqlite3.connect(capability_root / "index.db")) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT message_id, block_id, content_identity, content_occurrence "
            "FROM messages JOIN blocks USING(message_id) WHERE blocks.session_id=?",
            (session_id,),
        ).fetchone()
        assert row is not None
        positional = EvidenceRef(session_id=session_id, message_id=str(row["message_id"]), block_index=0)
        stable = EvidenceRef(
            session_id=session_id,
            message_id=str(row["message_id"]),
            block_id=str(row["block_id"]),
        )

        assert _resolve_target(conn, positional) is None
        resolved = _resolve_target(conn, stable)

    assert resolved is not None
    assert resolved.kind == "block-id"
    assert resolved.object_id == str(row["block_id"])
    assert resolved.qualifier == f"{row['content_identity']}:{row['content_occurrence']}"
