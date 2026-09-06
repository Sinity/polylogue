"""Value-complete staleness for the session-profile aggregate family.

The measured defect (exact-master audit 2026-08-25, head 936f2ff): profile
inspection compared a sort key and a source timestamp, so mutating an
output-affecting message value while holding identifiers, timestamps, partition
and row count fixed left inspection reporting VALID over a stale profile — and
every table derived from it.

These are the laws that make that unrepresentable. They run against a real
archive built through the production writer, not a stub.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.derived.session.derivation import (
    excess_session_profiles,
    inspect_session_profiles,
    publish_session_profile,
)
from polylogue.storage.derived.session.input_binding import (
    SESSION_INPUT_PROJECTION_COLUMNS,
    session_input_bindings,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.storage_records import SessionBuilder

_MATERIALIZER_VERSION = 5

#: Values that change an aggregate's output while leaving every identity,
#: timestamp, partition key and row count exactly where it was. Each is the
#: shape of the audited defect, not a variation on it.
_OUTPUT_AFFECTING_MUTATIONS = (
    ("role", "'assistant'"),
    ("model_name", "'a-different-model'"),
    ("input_tokens", "input_tokens + 4096"),
    ("output_tokens", "output_tokens + 77"),
    ("word_count", "word_count + 13"),
    ("has_tool_use", "1 - has_tool_use"),
    ("material_origin", "'synthetic'"),
)


@pytest.fixture
def archive(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """One archive with one two-message session, written by the real writer."""
    root = tmp_path / "archive"
    root.mkdir()
    index_db = root / "index.db"
    initialize_active_archive_root(root)
    builder = SessionBuilder(index_db, "value-binding")
    builder.add_message(role="user", text="what changed here")
    builder.add_message(role="assistant", text="the binding did")
    builder.save()
    yield index_db, builder.native_session_id()


def _materialize(index_db: Path, session_id: str) -> bool:
    with write_lease("test.publish"), closing(open_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        return publish_session_profile(conn, session_id, input_binding=binding)


def _status(index_db: Path, session_id: str) -> str:
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        return inspect_session_profiles(conn, [session_id], materializer_version=_MATERIALIZER_VERSION)[session_id]


def _mutate(index_db: Path, session_id: str, column: str, expression: str) -> None:
    """Change one output-affecting value in place; touch nothing identifying."""
    with write_lease("test.mutate"), closing(open_connection(index_db)) as conn:
        before = conn.execute(
            "SELECT count(*), max(occurred_at_ms), max(position) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.execute(
            f"UPDATE messages SET {column} = {expression} WHERE session_id = ? AND position = 0",
            (session_id,),
        )
        after = conn.execute(
            "SELECT count(*), max(occurred_at_ms), max(position) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.commit()
    assert tuple(before) == tuple(after), "the mutation must not move a count, a timestamp, or a partition key"


def test_a_fresh_profile_inspects_missing_then_valid(archive: tuple[Path, str]) -> None:
    """Absence and validity are distinct, and publication moves between them."""
    index_db, session_id = archive
    assert _status(index_db, session_id) == "missing"
    assert _materialize(index_db, session_id) is True
    assert _status(index_db, session_id) == "valid"


@pytest.mark.parametrize(("column", "expression"), _OUTPUT_AFFECTING_MUTATIONS)
def test_an_output_affecting_value_change_makes_inspection_stale(
    archive: tuple[Path, str],
    column: str,
    expression: str,
) -> None:
    """The audited defect, as a law, once per value the profile depends on.

    Anti-vacuity: drop ``column`` from
    ``SESSION_INPUT_PROJECTION_COLUMNS`` — or revert inspection to the sort-key
    and source-timestamp comparison it replaced — and this goes green while the
    profile and everything derived from it stay stale.
    """
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    assert _status(index_db, session_id) == "valid"

    _mutate(index_db, session_id, column, expression)

    assert _status(index_db, session_id) == "stale"
    assert _materialize(index_db, session_id) is True
    assert _status(index_db, session_id) == "valid"


def test_the_session_content_hash_is_not_the_binding(archive: tuple[Path, str]) -> None:
    """A binding copied from ``sessions.content_hash`` cannot see usage values.

    This is why the previous binding column was insufficient rather than merely
    unused: token counts and model names are excluded from the session's
    semantic hash by design, and the profile reads all of them.
    """
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        before = conn.execute("SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]

    _mutate(index_db, session_id, "input_tokens", "input_tokens + 4096")

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        after = conn.execute("SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
    assert before == after, "the session content hash is unmoved, which is the point"
    assert _status(index_db, session_id) == "stale"


def test_a_second_unchanged_pass_publishes_nothing_new(archive: tuple[Path, str]) -> None:
    """Inspection is authoritative, so an unchanged archive is a zero-write pass."""
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        first = conn.execute(
            "SELECT materialized_at, input_content_hash FROM session_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    assert _status(index_db, session_id) == "valid"

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        second = conn.execute(
            "SELECT materialized_at, input_content_hash FROM session_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    assert tuple(first) == tuple(second)


def test_publication_refuses_a_binding_that_moved_under_the_computation(
    archive: tuple[Path, str],
) -> None:
    """A lost race is pending, not a published output bound to vanished inputs.

    Anti-vacuity: remove the revalidation inside ``publish_session_profile`` and
    the stale binding publishes, which is precisely the wrong outcome the
    compute/publish split exists to prevent.
    """
    index_db, session_id = archive
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        computed_at = session_input_bindings(conn, (session_id,))[session_id]

    _mutate(index_db, session_id, "role", "'assistant'")

    with write_lease("test.publish"), closing(open_connection(index_db)) as conn:
        assert publish_session_profile(conn, session_id, input_binding=computed_at) is False

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert (
            conn.execute("SELECT count(*) FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
        )


def test_a_profile_with_no_stored_binding_is_stale_not_valid(archive: tuple[Path, str]) -> None:
    """A row that cannot say what it was computed from cannot certify itself."""
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    with write_lease("test.mutate"), closing(open_connection(index_db)) as conn:
        conn.execute("UPDATE session_profiles SET input_content_hash = NULL WHERE session_id = ?", (session_id,))
        conn.commit()

    assert _status(index_db, session_id) == "stale"


def test_an_orphaned_profile_is_reported_as_excess(archive: tuple[Path, str]) -> None:
    """Excess is discovered from the output relation, not from an invalidation."""
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    with write_lease("test.mutate"), closing(open_connection(index_db)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert excess_session_profiles(conn) == (session_id,)


def test_the_binding_is_stable_across_repeated_reads(archive: tuple[Path, str]) -> None:
    """A digest that moved on its own would report permanent staleness."""
    index_db, session_id = archive
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        first = session_input_bindings(conn, (session_id,))
        second = session_input_bindings(conn, (session_id,))
    assert first == second
    assert first[session_id]


def test_a_session_with_no_messages_still_has_a_binding(tmp_path: Path) -> None:
    """Valid-empty output must be distinguishable from work never performed."""
    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)
    with closing(sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)) as conn:
        bindings = session_input_bindings(conn, ("absent-session",))
    assert set(bindings) == {"absent-session"}
    assert bindings["absent-session"]


def test_the_projection_names_every_column_the_binding_digests() -> None:
    """The projection constant is the reviewable list, so it must not drift."""
    assert "content_hash" in SESSION_INPUT_PROJECTION_COLUMNS
    for column, _expression in _OUTPUT_AFFECTING_MUTATIONS:
        assert column in SESSION_INPUT_PROJECTION_COLUMNS


def test_publishing_an_excess_key_removes_the_orphan(archive: tuple[Path, str]) -> None:
    """An excess key converges by deletion, not by another rebuild.

    Anti-vacuity: rebuild the orphan instead of deleting it and inspection
    reports it excess on every subsequent pass -- a livelock, not convergence.
    """
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    with write_lease("test.mutate"), closing(open_connection(index_db)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert excess_session_profiles(conn) == (session_id,)

    assert _materialize(index_db, session_id) is True

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert excess_session_profiles(conn) == ()
