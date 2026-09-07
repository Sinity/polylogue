"""Convergence laws for the session partition and the families derived from it.

The session partition — profile, latency profile, work events, phases — is the
only aggregate family with an output relation of its own. Threads, tag rollups
and provider/day rollups are query-time projections of that relation
(``CREATE VIEW threads``, ``CREATE VIEW session_tag_rollups`` in
``archive_tiers/index.py``), so they have nothing separate to inspect: they are
current exactly when the partitions feeding them are.

Each test names the mutation that makes it red. The family tests would all pass
against a stale-by-timestamp inspection if the mutation moved a timestamp, so
every mutation here holds identifiers, timestamps, partition keys and row counts
fixed and changes only a value the output depends on.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.derived.session.derivation import (
    archive_session_partition_statuses,
    inspect_session_profiles,
    publish_session_profile,
)
from polylogue.storage.derived.session.input_binding import session_input_bindings
from polylogue.storage.derived.session.status import session_insight_status_sync
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.storage_records import SessionBuilder

_MATERIALIZER_VERSION = SESSION_INSIGHT_MATERIALIZER_VERSION

#: Relations one partition replacement writes. Convergence equality is stated
#: over their whole contents, minus the columns that record when a row was
#: built rather than what it says.
_PARTITION_RELATIONS = ("session_profiles", "session_latency_profiles", "session_work_events", "session_phases")

#: Wall-clock and build-order columns. They differ between two runs that
#: produced identical semantics, so comparing them would make every convergence
#: law fail for a reason the law is not about.
_NON_SEMANTIC_COLUMNS = frozenset({"materialized_at", "input_high_water_mark", "input_high_water_mark_source"})


@pytest.fixture
def archive_root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)
    return root


def _index_db(root: Path) -> Path:
    return root / "index.db"


def _write_connection(index_db: Path) -> sqlite3.Connection:
    conn = open_connection(index_db)
    conn.row_factory = sqlite3.Row
    return conn


def _read_connection(index_db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)


#: Status counts defer sessions whose source may still be being written. A
#: session seeded at the wall clock would fall inside that window and never
#: reach the freshness accounting these tests are about, so every seeded
#: session is dated well outside it.
_SEEDED_DAY = "2026-01-01"


def _seed(index_db: Path, name: str, *, messages: Sequence[tuple[str, str]], parent: str | None = None) -> str:
    """Write one session through the real writer. ``parent`` is another seed's name."""
    builder = SessionBuilder(index_db, name)
    builder = builder.created_at(f"{_SEEDED_DAY}T00:00:00+00:00").updated_at(f"{_SEEDED_DAY}T01:00:00+00:00")
    if parent is not None:
        # The writer resolves lineage by the parent's provider id, which is the
        # builder's native id for that name.
        builder = builder.parent_session(f"ext-{parent}").branch_type("fork")
    for index, (role, text) in enumerate(messages):
        builder.add_message(role=role, text=text, timestamp=f"{_SEEDED_DAY}T00:{index:02d}:00+00:00")
    builder.save()
    return builder.native_session_id()


def _converge(index_db: Path, session_ids: Sequence[str]) -> None:
    """Publish every non-valid partition, the way the kernel drives a domain."""
    with write_lease("test.converge"), closing(_write_connection(index_db)) as conn:
        for session_id in session_ids:
            binding = session_input_bindings(conn, (session_id,)).get(session_id, "")
            publish_session_profile(conn, session_id, input_binding=binding)


def _pending(index_db: Path) -> list[str]:
    with closing(_read_connection(index_db)) as conn:
        statuses = archive_session_partition_statuses(conn, materializer_version=_MATERIALIZER_VERSION)
    return sorted(session_id for session_id, status in statuses.items() if status != "valid")


def _converge_to_fixpoint(index_db: Path) -> None:
    for _ in range(4):
        pending = _pending(index_db)
        if not pending:
            return
        _converge(index_db, pending)
    raise AssertionError(f"convergence did not reach a fixpoint: {_pending(index_db)}")


def _semantic_relations(index_db: Path) -> dict[str, list[tuple[object, ...]]]:
    """Every partition relation's contents, keyed by relation, order-normalized."""
    relations: dict[str, list[tuple[object, ...]]] = {}
    with closing(_read_connection(index_db)) as conn:
        for relation in _PARTITION_RELATIONS:
            columns = [
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({relation})")
                if str(row[1]) not in _NON_SEMANTIC_COLUMNS
            ]
            projected = ", ".join(columns)
            relations[relation] = sorted(
                tuple(row) for row in conn.execute(f"SELECT {projected} FROM {relation}").fetchall()
            )
    return relations


def _status(index_db: Path):
    with closing(_read_connection(index_db)) as conn:
        return session_insight_status_sync(conn)


def _mutate_role(index_db: Path, session_id: str) -> None:
    """Change one message's role, holding every identity and count fixed."""
    with write_lease("test.mutate"), closing(_write_connection(index_db)) as conn:
        before = conn.execute(
            "SELECT count(*), max(occurred_at_ms), max(position) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.execute(
            "UPDATE messages SET role = 'assistant' WHERE session_id = ? AND position = 0",
            (session_id,),
        )
        after = conn.execute(
            "SELECT count(*), max(occurred_at_ms), max(position) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.commit()
    assert tuple(before) == tuple(after), "the mutation must not move a count, a timestamp, or a partition key"


def _tag(index_db: Path, session_id: str, tag: str) -> None:
    with write_lease("test.tag"), closing(_write_connection(index_db)) as conn:
        conn.execute(
            "UPDATE session_profiles SET tags_json = ? WHERE session_id = ?",
            (f'["{tag}"]', session_id),
        )
        conn.commit()


# ── The partition is the family ──────────────────────────────────────────────


def test_a_half_replaced_partition_is_stale(archive_root: Path) -> None:
    """Removing one sibling row must not leave the partition certified.

    Red when inspection reads the profile row alone: the profile still carries
    a matching binding and the current materializer version, and only its own
    declared work-event count contradicts what is stored.
    """
    index_db = _index_db(archive_root)
    session_id = _seed(index_db, "half", messages=[("user", "run the thing"), ("assistant", "ran it")])
    _converge_to_fixpoint(index_db)

    with write_lease("test.delete"), closing(_write_connection(index_db)) as conn:
        deleted = conn.execute("DELETE FROM session_work_events WHERE session_id = ?", (session_id,)).rowcount
        conn.commit()
    if deleted == 0:
        pytest.skip("this session produced no work events, so there is no sibling row to remove")

    with closing(_read_connection(index_db)) as conn:
        assert inspect_session_profiles(conn, [session_id], materializer_version=_MATERIALIZER_VERSION) == {
            session_id: "stale"
        }


def test_a_session_with_no_messages_converges_to_a_valid_empty_partition(archive_root: Path) -> None:
    """Valid-empty is a converged state, not work that was never done.

    Red if an empty projection were treated as "nothing to bind to": the
    partition would inspect non-valid forever and the pass would never settle.
    """
    index_db = _index_db(archive_root)
    session_id = _seed(index_db, "empty", messages=[])
    _converge_to_fixpoint(index_db)

    assert _pending(index_db) == []
    with closing(_read_connection(index_db)) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM session_work_events WHERE session_id = ?", (session_id,)).fetchone()[0]
            == 0
        )


def test_the_pending_set_is_reconstructed_rather_than_remembered(archive_root: Path) -> None:
    """The pending set is derived, so nothing that stores it can be authority.

    Red if a queue, cursor or dirty-key table decided the work: a fresh
    connection that has never seen one would then report a different set.
    """
    index_db = _index_db(archive_root)
    first = _seed(index_db, "kept", messages=[("user", "hello")])
    second = _seed(index_db, "dropped", messages=[("user", "hi"), ("assistant", "hey")])
    _converge(index_db, [first])

    assert _pending(index_db) == [second]
    assert _pending(index_db) == [second]


def test_freshness_counts_are_recomputed_on_every_call(archive_root: Path) -> None:
    """No status call may carry a freshness answer forward.

    Red if any count were memoized or read from a stored freshness marker: the
    second call would still report the converged archive after the mutation,
    and the third would still report the mutation after the repair.
    """
    index_db = _index_db(archive_root)
    session_id = _seed(index_db, "recounted", messages=[("user", "one"), ("assistant", "two")])
    _converge_to_fixpoint(index_db)
    assert _status(index_db).stale_profile_row_count == 0

    _mutate_role(index_db, session_id)
    assert _status(index_db).stale_profile_row_count == 1

    _converge_to_fixpoint(index_db)
    assert _status(index_db).stale_profile_row_count == 0


def test_an_orphaned_partition_is_reported_then_removed(archive_root: Path) -> None:
    """Excess output is pending work, and publishing it converges rather than looping."""
    index_db = _index_db(archive_root)
    session_id = _seed(index_db, "orphan", messages=[("user", "here")])
    _converge_to_fixpoint(index_db)

    with write_lease("test.orphan"), closing(_write_connection(index_db)) as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

    _converge(index_db, [session_id])
    with closing(_read_connection(index_db)) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
        )


# ── The families projected from it ───────────────────────────────────────────


def test_a_value_change_makes_the_thread_and_tag_families_stale(archive_root: Path) -> None:
    """The red law, carried through to the projections.

    Red on the shape this replaced: the thread check compared the same sort key
    and materializer version the profile check did, and the tag-rollup check
    compared ``session_tag_rollups.materialized_at`` against the profiles'
    — a column the view emits as the literal ``'query-time'`` on every row, so
    the count was structurally always zero.
    """
    index_db = _index_db(archive_root)
    session_id = _seed(index_db, "projected", messages=[("user", "before"), ("assistant", "after")])
    _converge_to_fixpoint(index_db)
    _tag(index_db, session_id, "release")

    converged = _status(index_db)
    assert converged.stale_thread_count == 0
    assert converged.stale_tag_rollup_count == 0
    assert converged.tag_rollup_count > 0, "the tag rollup projection must have a row to go stale"

    _mutate_role(index_db, session_id)

    mutated = _status(index_db)
    assert mutated.stale_profile_row_count == 1
    assert mutated.stale_thread_count == 1
    assert mutated.stale_tag_rollup_count == 1
    assert mutated.stale_latency_profile_row_count == 1


def test_a_converged_archive_reports_no_family_stale(archive_root: Path) -> None:
    """Anti-vacuity for the counts above: they are not always non-zero."""
    index_db = _index_db(archive_root)
    _seed(index_db, "root", messages=[("user", "start")])
    _seed(index_db, "child", messages=[("user", "start"), ("assistant", "branch")], parent="root")
    _converge_to_fixpoint(index_db)

    status = _status(index_db)
    assert status.stale_profile_row_count == 0
    assert status.stale_thread_count == 0
    assert status.stale_tag_rollup_count == 0
    assert status.stale_latency_profile_row_count == 0


# ── Convergence equality ─────────────────────────────────────────────────────


def _clean_rebuild(tmp_path: Path, sessions: Sequence[tuple[str, Sequence[tuple[str, str]], str | None]]) -> Path:
    """A second archive built from scratch to the same intended end state."""
    root = tmp_path / "control"
    root.mkdir()
    initialize_active_archive_root(root)
    index_db = _index_db(root)
    for name, messages, parent in sessions:
        _seed(index_db, name, messages=messages, parent=parent)
    _converge_to_fixpoint(index_db)
    return index_db


def test_incremental_append_converges_to_the_clean_rebuild(archive_root: Path, tmp_path: Path) -> None:
    """Appending a message and reconverging equals building the final state once.

    Red if publication left any part of the partition behind: the incremental
    relations would still carry the two-message projection.
    """
    index_db = _index_db(archive_root)
    _seed(index_db, "append", messages=[("user", "one"), ("assistant", "two")])
    _converge_to_fixpoint(index_db)
    _seed(index_db, "append", messages=[("user", "one"), ("assistant", "two"), ("user", "three")])
    _converge_to_fixpoint(index_db)

    control = _clean_rebuild(tmp_path, [("append", [("user", "one"), ("assistant", "two"), ("user", "three")], None)])
    assert _semantic_relations(index_db) == _semantic_relations(control)


def test_replacement_converges_to_the_clean_rebuild(archive_root: Path, tmp_path: Path) -> None:
    """A full replace that shortens the session leaves no residue behind."""
    index_db = _index_db(archive_root)
    _seed(index_db, "replace", messages=[("user", "a"), ("assistant", "b"), ("user", "c")])
    _converge_to_fixpoint(index_db)
    _seed(index_db, "replace", messages=[("user", "a")])
    _converge_to_fixpoint(index_db)

    control = _clean_rebuild(tmp_path, [("replace", [("user", "a")], None)])
    assert _semantic_relations(index_db) == _semantic_relations(control)


def test_deletion_removes_the_partition_rather_than_stranding_it(archive_root: Path, tmp_path: Path) -> None:
    """A deleted session's partition is excess, and convergence removes it."""
    index_db = _index_db(archive_root)
    kept = _seed(index_db, "kept", messages=[("user", "stay")])
    removed = _seed(index_db, "removed", messages=[("user", "go")])
    _converge_to_fixpoint(index_db)

    with write_lease("test.delete-session"), closing(_write_connection(index_db)) as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (removed,))
        conn.commit()
    _converge(index_db, [removed])

    control = _clean_rebuild(tmp_path, [("kept", [("user", "stay")], None)])
    assert _semantic_relations(index_db) == _semantic_relations(control)
    assert _pending(index_db) == []
    with closing(_read_connection(index_db)) as conn:
        profiled = [str(row[0]) for row in conn.execute("SELECT session_id FROM session_profiles").fetchall()]
    assert profiled == [kept]


def test_batch_boundaries_do_not_change_the_converged_relation(archive_root: Path, tmp_path: Path) -> None:
    """The property: how the work is cut into batches cannot change the result.

    Red if publication carried anything across keys — a shared accumulator, a
    batch-scoped timestamp folded into a semantic column — because the one-shot
    and per-session cuts would then differ.
    """
    index_db = _index_db(archive_root)
    first = _seed(index_db, "s1", messages=[("user", "alpha")])
    second = _seed(index_db, "s2", messages=[("user", "beta"), ("assistant", "gamma")])
    third = _seed(index_db, "s3", messages=[("user", "delta")])
    _converge(index_db, [first, second, third])
    one_batch = _semantic_relations(index_db)

    control_root = tmp_path / "cut"
    control_root.mkdir()
    initialize_active_archive_root(control_root)
    control = _index_db(control_root)
    ids = [
        _seed(control, "s1", messages=[("user", "alpha")]),
        _seed(control, "s2", messages=[("user", "beta"), ("assistant", "gamma")]),
        _seed(control, "s3", messages=[("user", "delta")]),
    ]
    for session_id in reversed(ids):
        _converge(control, [session_id])

    assert one_batch == _semantic_relations(control)


def test_reparenting_converges_to_the_clean_rebuild(archive_root: Path, tmp_path: Path) -> None:
    """A session that gains a parent is a partition whose inputs moved.

    ``parent_session_id`` is in the session-row projection, so reparenting
    changes the binding; red if the binding covered messages alone.
    """
    index_db = _index_db(archive_root)
    _seed(index_db, "parent", messages=[("user", "root")])
    _seed(index_db, "orphaned", messages=[("user", "root"), ("assistant", "tail")])
    _converge_to_fixpoint(index_db)

    _seed(index_db, "orphaned", messages=[("user", "root"), ("assistant", "tail")], parent="parent")
    assert _pending(index_db), "reparenting must make the child's partition pending"
    _converge_to_fixpoint(index_db)

    control = _clean_rebuild(
        tmp_path,
        [
            ("parent", [("user", "root")], None),
            ("orphaned", [("user", "root"), ("assistant", "tail")], "parent"),
        ],
    )
    assert _semantic_relations(index_db) == _semantic_relations(control)


def test_a_second_pass_over_unchanged_inputs_publishes_nothing(archive_root: Path) -> None:
    """Inspection is authoritative rather than advisory."""
    index_db = _index_db(archive_root)
    _seed(index_db, "settled", messages=[("user", "one"), ("assistant", "two")])
    _converge_to_fixpoint(index_db)
    settled = _semantic_relations(index_db)

    assert _pending(index_db) == []
    _converge_to_fixpoint(index_db)
    assert _semantic_relations(index_db) == settled
