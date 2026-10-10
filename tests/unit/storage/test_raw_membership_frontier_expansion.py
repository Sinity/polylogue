"""Each invocation expands fresh Source edges without revisiting its frontier."""

import sqlite3
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.revision_governance import (
    RawMembershipSelectionFamily,
    _load_raw_session_input,
    expand_raw_membership_selection_sync,
)
from polylogue.storage.sqlite.archive_tiers.write import ConnectionSessionSourceRead, PreparedSessionSourceRead
from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


@pytest.fixture
def membership_root(tmp_path: Path) -> Iterator[Path]:
    with write_lease("test.membership-frontier", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        yield tmp_path


def _seed(root: Path, records: Sequence[tuple[str, str, str | None]], memberships: Sequence[tuple[str, str]]) -> None:
    with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as conn:
        conn.executemany(
            "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms,logical_source_key) "
            "VALUES(?,'unknown-export',?,?,1,1,?)",
            ((raw, path, b"x" * 32, key) for raw, path, key in records),
        )
        conn.executemany(
            "INSERT INTO raw_session_memberships(raw_id,logical_source_key,provider_session_id,source_revision, "
            "normalized_content_hash,message_count) VALUES(?,?,'neutral','neutral',?,0)",
            ((raw, key, b"n" * 32) for raw, key in memberships),
        )
        conn.commit()


def test_membership_expansion_queries_each_distinct_frontier_once(
    membership_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A short genuine chain suffices to expose repeated whole-component queries.
    records = [(f"r{i}", f"p{i}", f"k{i}") for i in range(8)]
    _seed(membership_root, records, [(f"r{i + 1}", f"k{i}") for i in range(7)])
    counts: Counter[tuple[str, str]] = Counter()
    original = ConnectionSessionSourceRead.raw_selection_values

    def observed(
        self: ConnectionSessionSourceRead, family: RawMembershipSelectionFamily, operands: Sequence[str]
    ) -> set[str]:
        counts.update((family, operand) for operand in operands)
        return original(self, family, operands)

    monkeypatch.setattr(ConnectionSessionSourceRead, "raw_selection_values", observed)
    with closing(sqlite3.connect(membership_root / "source.db")) as conn:
        conn.execute("BEGIN")
        assert expand_raw_membership_selection_sync(conn, ["r0", "r0"]) == (
            tuple(f"r{i}" for i in range(8)),
            tuple(f"k{i}" for i in range(8)),
        )
    assert set(counts) == {
        (family, f"{prefix}{i}")
        for family, prefix in (("paths", "r"), ("keys", "r"), ("path_raws", "p"), ("key_raws", "k"))
        for i in range(8)
    }
    assert set(counts.values()) == {1}


def test_membership_expansion_preserves_cycles_multigraph_and_absent_hints(membership_root: Path) -> None:
    _seed(
        membership_root,
        [("a", "shared", "a-key"), ("b", "shared", "b-key"), ("c", "other", None), ("isolated", "alone", "z-key")],
        [("a", "cycle"), ("b", "cycle"), ("c", "b-key"), ("c", "a-key")],
    )
    with closing(sqlite3.connect(membership_root / "source.db")) as conn:
        conn.execute("BEGIN")
        assert expand_raw_membership_selection_sync(conn, ["c", "a", "absent", "c"]) == (
            ("a", "absent", "b", "c"),
            ("a-key", "b-key", "cycle"),
        )
        assert expand_raw_membership_selection_sync(conn, []) == ((), ())
        assert expand_raw_membership_selection_sync(conn, None) == (
            ("a", "b", "c", "isolated"),
            ("a-key", "b-key", "cycle", "z-key"),
        )


def test_prepared_membership_expansion_restarts_after_staged_census_change(membership_root: Path) -> None:
    _seed(membership_root, [("a", "a-path", "a-key"), ("b", "b-path", "b-key")], [])
    with PreparedIndexMutation.source_only(archive_root=membership_root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            reader = PreparedSessionSourceRead(seal, blob_store=BlobStore(membership_root / "blob"))
            assert reader.expand_raw_membership_selection(("a",)) == (("a",), ("a-key",))
            _load_raw_session_input(seal, "b")
            raw = seal.retain_literal_scalar("b")
            key = seal.retain_literal_scalar("a-key")
            raw_sql, raw_parameters = seal.source_literal_expression(raw)
            key_sql, key_parameters = seal.source_literal_expression(key)
            with seal.source_statement(
                "INSERT INTO raw_session_memberships(rowid,raw_id,logical_source_key,provider_session_id,"
                "source_revision,normalized_content_hash,message_count) "
                f"VALUES(?,{raw_sql},{key_sql},'neutral','neutral',zeroblob(32),0)",
                (None, *raw_parameters, *key_parameters),
                table="raw_session_memberships",
                writable_targets=(("raw_session_memberships", (raw, key)),),
                allocation_parameter=0,
                prepared_cells={"raw_id": raw, "logical_source_key": key},
            ):
                pass
            assert reader.expand_raw_membership_selection(("a",)) == (("a", "b"), ("a-key", "b-key"))
            with seal.source_statement(
                f"DELETE FROM raw_session_memberships WHERE raw_id={raw_sql} AND logical_source_key={key_sql}",
                (*raw_parameters, *key_parameters),
                table="raw_session_memberships",
                writable_targets=(("raw_session_memberships", (raw, key)),),
            ):
                pass
            assert reader.expand_raw_membership_selection(("a",)) == (("a",), ("a-key",))
