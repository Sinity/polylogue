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

import itertools
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from polylogue.storage.derived.session.derivation import (
    archive_session_partition_statuses,
    inspect_session_profiles,
    publish_session_profile,
)
from polylogue.storage.derived.session.input_binding import session_input_bindings
from polylogue.storage.derived.session.rebuild import rebuild_session_insights_sync
from polylogue.storage.derived.session.runtime import SessionInsightStatusSnapshot
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


def _status(index_db: Path) -> SessionInsightStatusSnapshot:
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
    a matching binding and the current materializer version, and nothing about
    the profile row itself records that a sibling relation lost its rows.

    The latency profile is the sibling every replacement writes exactly once,
    so removing it is the one half-replacement that does not depend on what the
    session's content happened to infer.
    """
    index_db = _index_db(archive_root)
    session_id = _seed(index_db, "half", messages=[("user", "run the thing"), ("assistant", "ran it")])
    _converge_to_fixpoint(index_db)

    with write_lease("test.delete"), closing(_write_connection(index_db)) as conn:
        removed = conn.execute("DELETE FROM session_latency_profiles WHERE session_id = ?", (session_id,)).rowcount
        conn.commit()
    assert removed == 1, "every replaced partition writes exactly one latency profile"

    with closing(_read_connection(index_db)) as conn:
        assert inspect_session_profiles(conn, [session_id], materializer_version=_MATERIALIZER_VERSION) == {
            session_id: "stale"
        }


def test_a_partition_missing_its_inferred_rows_is_stale(archive_root: Path) -> None:
    """The same law for the count-bearing siblings, when the content has them."""
    index_db = _index_db(archive_root)
    session_id = _seed(index_db, "counted", messages=[("user", "run the thing"), ("assistant", "ran it")])
    _converge_to_fixpoint(index_db)

    with closing(_read_connection(index_db)) as conn:
        declared = conn.execute(
            "SELECT work_event_count, phase_count FROM session_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    relation = "session_work_events" if declared[0] else "session_phases" if declared[1] else None
    if relation is None:
        pytest.skip("this session inferred no work events or phases to remove")

    with write_lease("test.delete-inferred"), closing(_write_connection(index_db)) as conn:
        conn.execute(f"DELETE FROM {relation} WHERE session_id = ?", (session_id,))
        conn.commit()

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
    # Every rollup partition the session contributes to goes stale with it, and
    # this session contributes two: its explicit tag and its ``origin:`` auto
    # tag. Stating the count as the converged total keeps the law about "all of
    # them" rather than about how many tags the fixture happens to carry.
    assert mutated.stale_tag_rollup_count == converged.tag_rollup_count
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
    """A second archive built from scratch to the same intended end state.

    The name/messages/parent shape the lineage tests are written in, over the
    one control builder in :func:`_rebuilt_archive`; a second copy of "seed
    these sessions and converge" could drift from the mutation path it is the
    control for.
    """
    return _rebuilt_archive(
        tmp_path,
        "control",
        [_SessionSpec(name, tuple(messages), parent=parent) for name, messages, parent in sessions],
    )


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


# ── The red law, over every output-affecting value the binding must carry ─────


def _mutate_model_name(index_db: Path, session_id: str) -> None:
    """Change one message's model, holding every identity and count fixed."""
    with write_lease("test.model"), closing(_write_connection(index_db)) as conn:
        before = conn.execute(
            """
            SELECT message_id, occurred_at_ms, position
            FROM messages
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        changed = conn.execute(
            "UPDATE messages SET model_name = 'model-after' WHERE session_id = ? AND model_name IS NOT NULL",
            (session_id,),
        ).rowcount
        after = conn.execute(
            """
            SELECT message_id, occurred_at_ms, position
            FROM messages
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        conn.commit()
    assert changed > 0, "the mutation must change a row to be a mutation at all"
    assert [tuple(row) for row in before] == [tuple(row) for row in after], (
        "the mutation must not move an id, timestamp, count, or partition key"
    )


def _mutate_output_tokens(index_db: Path, session_id: str) -> None:
    """Change one message's token measurement, holding identity and counts fixed."""
    with write_lease("test.tokens"), closing(_write_connection(index_db)) as conn:
        before = conn.execute(
            "SELECT count(*), max(occurred_at_ms), max(position), sum(word_count) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        changed = conn.execute(
            "UPDATE messages SET output_tokens = COALESCE(output_tokens, 0) + 4096 WHERE session_id = ? AND position = 1",
            (session_id,),
        ).rowcount
        after = conn.execute(
            "SELECT count(*), max(occurred_at_ms), max(position), sum(word_count) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.commit()
    assert changed == 1
    assert tuple(before) == tuple(after), "the mutation must not move a count, a timestamp, or a partition key"


def _priced_session(index_db: Path, name: str) -> str:
    """A session whose assistant turn carries a model and a usage measurement.

    ``model_name`` and the token columns are exactly the values ``pipeline/ids.py``
    excludes from the message content hash ("owned by usage/cost derivation"), so
    a binding that leaned on that hash could not see either of them move.
    """
    builder = SessionBuilder(index_db, name)
    builder = builder.created_at(f"{_SEEDED_DAY}T00:00:00+00:00").updated_at(f"{_SEEDED_DAY}T01:00:00+00:00")
    builder.add_message(role="user", text="ask", timestamp=f"{_SEEDED_DAY}T00:00:00+00:00")
    builder.add_message(
        role="assistant",
        text="answer",
        timestamp=f"{_SEEDED_DAY}T00:01:00+00:00",
        model_name="model-before",
        input_tokens=120,
        output_tokens=48,
    )
    builder.save()
    return builder.native_session_id()


def test_a_model_name_change_makes_the_partition_stale(archive_root: Path) -> None:
    """The measured defect's sibling: a model name moves no identity at all.

    Red against a binding over the message content hash alone — ``model_name``
    is a hashed ParsedMessage field, but the stored ``messages.model_name`` can
    be corrected by a usage/cost derivation without a re-parse, and the profile
    reads it. Red against any sort-key or updated-at predicate outright.
    """
    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "modelled")
    _converge_to_fixpoint(index_db)
    assert _pending(index_db) == []

    _mutate_model_name(index_db, session_id)

    assert _pending(index_db) == [session_id]
    _converge_to_fixpoint(index_db)
    with closing(_read_connection(index_db)) as conn:
        stored = conn.execute(
            "SELECT primary_model_name FROM session_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()
    assert stored[0] == "model-after", "convergence must republish the value that moved"


def test_rebuild_reconciles_model_usage_after_a_fixed_id_model_correction(archive_root: Path) -> None:
    """A renamed message removes only its unsupported usage row on rebuild.

    Anti-vacuity: remove ``_reconcile_session_model_usage_rows`` from the
    rebuild refresh and ``model-before`` survives beside ``model-after``. The
    provider-event row proves reconciliation does not infer that messages are
    the sole source of a model's usage.
    """
    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "model-usage-reconciliation")
    with write_lease("test.provider-usage"), closing(_write_connection(index_db)) as conn:
        conn.execute(
            """
            INSERT INTO session_provider_usage_events (
                session_id, position, provider_event_type, model_name,
                last_input_tokens, last_output_tokens
            ) VALUES (?, 99, 'token_count', 'provider-event-model', 17, 9)
            """,
            (session_id,),
        )
        # The rebuild must re-apportion a session-level reported amount after
        # the message correction; checking only token rows would miss stale
        # provider_cost_usd shares left on the old model row.
        conn.execute(
            "UPDATE sessions SET reported_cost_usd = 12.0 WHERE session_id = ?",
            (session_id,),
        )
        rebuild_session_insights_sync(conn, session_ids=[session_id])

    _mutate_model_name(index_db, session_id)
    assert _pending(index_db) == [session_id]

    with write_lease("test.rebuild"), closing(_write_connection(index_db)) as conn:
        rebuild_session_insights_sync(conn, session_ids=[session_id])
        usage_rows = conn.execute(
            """
            SELECT model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
            FROM session_model_usage
            WHERE session_id = ?
            ORDER BY model_name
            """,
            (session_id,),
        ).fetchall()
        usage_costs = conn.execute(
            """
            SELECT model_name, provider_cost_usd
            FROM session_model_usage
            WHERE session_id = ?
            ORDER BY model_name
            """,
            (session_id,),
        ).fetchall()

    assert [tuple(row) for row in usage_rows] == [
        ("model-after", 120, 48, 0, 0),
        ("provider-event-model", 17, 9, 0, 0),
    ]
    assert sum(float(row[1]) for row in usage_costs) == pytest.approx(12.0)
    assert all(row[1] is not None for row in usage_costs)
    assert _pending(index_db) == []


def test_rebuild_rederives_usage_when_a_model_keeps_some_messages(archive_root: Path) -> None:
    """A fixed-id model correction may reduce, rather than remove, a rollup.

    Anti-vacuity: the former monotonic message upsert leaves ``model-before``
    at 120 tokens because one message still carries that model, even though its
    remaining source evidence totals only 48.  A rebuild must derive both model
    rows from the current messages instead of retaining the larger stale value.
    """
    index_db = _index_db(archive_root)
    builder = SessionBuilder(index_db, "partial-model-usage-reconciliation")
    builder = builder.created_at(f"{_SEEDED_DAY}T00:00:00+00:00").updated_at(f"{_SEEDED_DAY}T01:00:00+00:00")
    builder.add_message(role="user", text="ask", timestamp=f"{_SEEDED_DAY}T00:00:00+00:00")
    builder.add_message(
        role="assistant",
        text="first answer",
        timestamp=f"{_SEEDED_DAY}T00:01:00+00:00",
        model_name="model-before",
        input_tokens=48,
        output_tokens=12,
    )
    builder.add_message(
        role="assistant",
        text="corrected answer",
        timestamp=f"{_SEEDED_DAY}T00:02:00+00:00",
        model_name="model-before",
        input_tokens=72,
        output_tokens=24,
    )
    builder.save()
    session_id = builder.native_session_id()
    _converge_to_fixpoint(index_db)

    with write_lease("test.partial-model-correction"), closing(_write_connection(index_db)) as conn:
        changed = conn.execute(
            "UPDATE messages SET model_name = 'model-after' WHERE session_id = ? AND position = 2",
            (session_id,),
        ).rowcount
        conn.commit()
    assert changed == 1
    assert _pending(index_db) == [session_id]

    with write_lease("test.partial-model-rebuild"), closing(_write_connection(index_db)) as conn:
        rebuild_session_insights_sync(conn, session_ids=[session_id])
        usage_rows = conn.execute(
            """
            SELECT model_name, input_tokens, output_tokens
            FROM session_model_usage
            WHERE session_id = ?
            ORDER BY model_name
            """,
            (session_id,),
        ).fetchall()

    assert [tuple(row) for row in usage_rows] == [
        ("model-after", 72, 24),
        ("model-before", 48, 12),
    ]
    assert _pending(index_db) == []


def test_a_token_count_change_makes_the_partition_stale(archive_root: Path) -> None:
    """Usage measurements are excluded from the content hash and read by the profile.

    ``pipeline/ids.py`` excludes ``output_tokens`` from ``ParsedMessage``'s
    semantic hash by design, so a binding that used ``messages.content_hash``
    as its whole message projection would report VALID here. The explicit token
    columns in ``SESSION_INPUT_PROJECTION_COLUMNS`` are what makes it STALE.
    """
    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "metered")
    _converge_to_fixpoint(index_db)
    assert _pending(index_db) == []

    _mutate_output_tokens(index_db, session_id)

    assert _pending(index_db) == [session_id]


def test_provider_usage_correction_makes_the_partition_stale(archive_root: Path) -> None:
    """Provider usage is a profile input even though messages stay unchanged.

    Anti-vacuity: remove the ``provider_usage_events`` projection from
    ``session_input_bindings`` and this fixed-id correction leaves the stored
    profile binding equal, so inspection incorrectly reports VALID. The
    provider event's larger output total also proves publication refreshes the
    profile from the canonical usage rollup rather than merely changing a
    digest.
    """
    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "provider-usage-correction")
    with write_lease("test.provider-usage-seed"), closing(_write_connection(index_db)) as conn:
        conn.execute(
            """
            INSERT INTO session_provider_usage_events (
                session_id, position, provider_event_type, model_name, last_output_tokens
            ) VALUES (?, 99, 'token_count', 'provider-corrected-model', 1)
            """,
            (session_id,),
        )
        conn.commit()
    _converge_to_fixpoint(index_db)

    with write_lease("test.provider-usage-correction"), closing(_write_connection(index_db)) as conn:
        before = conn.execute(
            "SELECT primary_model_name FROM session_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()
        message_before = conn.execute(
            "SELECT COUNT(*), MAX(occurred_at_ms), MAX(position) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        changed = conn.execute(
            """
            UPDATE session_provider_usage_events
            SET last_output_tokens = 10_000
            WHERE session_id = ? AND position = 99
            """,
            (session_id,),
        ).rowcount
        message_after = conn.execute(
            "SELECT COUNT(*), MAX(occurred_at_ms), MAX(position) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.commit()

    assert before[0] == "model-before"
    assert changed == 1
    assert tuple(message_after) == tuple(message_before), "the correction must not move a message identity or timestamp"
    assert _pending(index_db) == [session_id]

    _converge_to_fixpoint(index_db)
    with closing(_read_connection(index_db)) as conn:
        after = conn.execute(
            "SELECT primary_model_name FROM session_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()
    assert after[0] == "provider-corrected-model"


def test_expected_row_counts_cannot_certify_a_stale_partition(archive_root: Path) -> None:
    """Work-event and phase readiness may not rest on the profile's own declaration.

    ``expected_work_event_inference_count`` sums ``session_profiles.work_event_count``
    and compares it to the stored work-event rows: both sides come from the
    partition being judged, so a profile that is wrong about its inputs is
    wrong on both and the comparison agrees with itself. Red before the
    readiness gates took the value-complete inspection as their authority —
    the two equalities below still hold after the mutation, and the readiness
    flags must still be False.
    """
    from polylogue.storage.derived.derived_status import _session_insight_metrics

    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "accounted")
    _converge_to_fixpoint(index_db)

    converged = _session_insight_metrics(_status(index_db))
    assert converged["work_event_rows_ready"] is True
    assert converged["phase_rows_ready"] is True

    _mutate_role(index_db, session_id)

    status = _status(index_db)
    assert status.work_event_inference_count == status.expected_work_event_inference_count
    assert status.phase_count == status.expected_phase_count
    assert status.stale_profile_row_count == 1

    mutated = _session_insight_metrics(status)
    assert mutated["work_event_rows_ready"] is False
    assert mutated["phase_rows_ready"] is False
    assert mutated["profile_rows_ready"] is False


def test_stale_work_event_and_phase_counts_are_reported_rather_than_defaulted(archive_root: Path) -> None:
    """Anti-vacuity for the two counts the readiness gates compare to zero.

    Both were snapshot fields no status descriptor emitted, so they read zero
    on every archive and their ``== 0`` conjuncts could not fail. Red if either
    stops being derived from the inspection's non-valid key set.
    """
    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "counted")
    _converge_to_fixpoint(index_db)

    converged = _status(index_db)
    assert converged.stale_work_event_inference_count == 0
    assert converged.stale_phase_inference_count == 0
    assert converged.work_event_inference_count > 0, "the archive must have work-event rows to report stale"
    assert converged.phase_count > 0, "the archive must have phase rows to report stale"

    _mutate_role(index_db, session_id)

    mutated = _status(index_db)
    assert mutated.stale_work_event_inference_count == converged.work_event_inference_count
    assert mutated.stale_phase_inference_count == converged.phase_count


# ── Faults: a frame that moved, and a crash on either side of publication ─────


def test_a_publication_whose_inputs_moved_is_refused_rather_than_stamped(archive_root: Path) -> None:
    """Publication revalidates the binding it was computed against.

    This is the revalidation edge. Delete it — stamp the binding the caller
    brought — and the profile certifies itself against inputs it never read:
    the partition would report VALID while its rows describe the pre-mutation
    session. Red exactly then.
    """
    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "raced")
    _converge_to_fixpoint(index_db)

    with closing(_read_connection(index_db)) as conn:
        stale_frame = session_input_bindings(conn, (session_id,))[session_id]

    _mutate_role(index_db, session_id)

    with write_lease("test.race"), closing(_write_connection(index_db)) as conn:
        accepted = publish_session_profile(conn, session_id, input_binding=stale_frame)

    assert accepted is False, "a frame whose inputs moved may not be published"
    assert _pending(index_db) == [session_id], "the key stays pending for the next pass"


def test_a_crash_between_the_rows_and_the_binding_leaves_the_key_pending(archive_root: Path) -> None:
    """A partition that cannot say what it was built from is never valid.

    The crash-after-rows, before-stamp window: the rows are current, the
    binding column is not yet written. Inspection must call that STALE rather
    than trust the rows, and the next pass must converge it without help.
    """
    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "half-stamped")
    _converge_to_fixpoint(index_db)
    settled = _semantic_relations(index_db)

    with write_lease("test.crash"), closing(_write_connection(index_db)) as conn:
        conn.execute("UPDATE session_profiles SET input_content_hash = NULL WHERE session_id = ?", (session_id,))
        conn.commit()

    assert _pending(index_db) == [session_id]
    _converge_to_fixpoint(index_db)
    assert _semantic_relations(index_db) == settled


def test_a_crash_before_publication_leaves_the_output_untouched(archive_root: Path) -> None:
    """Computing a replacement writes nothing; only publication does.

    Red if ``compute`` ever acquired a side effect on the output relations: the
    relations after a computed-but-unpublished key would differ from the ones
    before it.
    """
    from polylogue.storage.derived.session.derivation import SessionProfileDerivation

    index_db = _index_db(archive_root)
    session_id = _priced_session(index_db, "computed-only")
    _converge_to_fixpoint(index_db)
    settled = _semantic_relations(index_db)

    adapter = SessionProfileDerivation(
        lambda: _read_connection(index_db),
        lambda: _write_connection(index_db),
        materializer_version=_MATERIALIZER_VERSION,
        session_scope=lambda frame: (session_id,),
    )
    replacement = adapter.compute(object(), session_id)

    assert replacement.key == session_id
    assert _semantic_relations(index_db) == settled
    assert _pending(index_db) == []


# ── The differential: the projection entries are load-bearing ─────────────────


def test_the_partition_input_columns_are_projected_or_declared_excluded(archive_root: Path) -> None:
    """Every column of the two input relations is classified, one way or the other.

    Completeness is the property the binding exists for, and an unclassified
    column is the one way to lose it silently: added to ``sessions`` or
    ``messages``, read by the profile, absent from both the projection and the
    exclusion table, and the binding reports VALID after the output moved.

    Red the moment a column is added to either relation without a decision.
    """
    from polylogue.storage.derived.session.input_binding import (
        SESSION_INPUT_EXCLUDED_COLUMNS,
        SESSION_INPUT_PROJECTION_COLUMNS,
        SESSION_ROW_EXCLUDED_COLUMNS,
        SESSION_ROW_PROJECTION_COLUMNS,
    )

    index_db = _index_db(archive_root)
    _seed(index_db, "classified", messages=[("user", "one")])

    cases = (
        ("sessions", SESSION_ROW_PROJECTION_COLUMNS, SESSION_ROW_EXCLUDED_COLUMNS),
        ("messages", SESSION_INPUT_PROJECTION_COLUMNS, SESSION_INPUT_EXCLUDED_COLUMNS),
    )
    with closing(_read_connection(index_db)) as conn:
        for relation, projected, excluded in cases:
            # ``table_xinfo`` rather than ``table_info``: generated columns are
            # hidden from the latter, and ``sort_key_ms`` is both generated and
            # projected.
            columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_xinfo({relation})")}
            assert not (set(projected) - columns), f"{relation}: projected column that does not exist"
            unclassified = sorted(columns - set(projected) - set(excluded))
            assert not unclassified, f"{relation}: columns neither projected nor declared excluded: {unclassified}"
            assert not (set(excluded) & set(projected)), f"{relation}: a column cannot be both"
            assert all(reason.strip() for reason in excluded.values()), f"{relation}: an exclusion needs a reason"


def test_dropping_a_projection_column_stops_the_binding_from_seeing_its_defect(
    archive_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The differential over the dependency projection itself.

    Each declared column is there because some output-affecting mutation is
    invisible without it. Removing ``role`` — the column the measured defect
    was found on — and the role mutation stops moving the binding, which is
    precisely the pre-fix behaviour. This is the test that would go red if a
    future edit dropped a column from the projection believing it inert.
    """
    from polylogue.storage.derived.session import input_binding

    index_db = _index_db(archive_root)
    session_id = _seed(index_db, "differential", messages=[("user", "before"), ("assistant", "after")])
    _converge_to_fixpoint(index_db)

    with closing(_read_connection(index_db)) as conn:
        full_before = session_input_bindings(conn, (session_id,))[session_id]
    _mutate_role(index_db, session_id)
    with closing(_read_connection(index_db)) as conn:
        full_after = session_input_bindings(conn, (session_id,))[session_id]
    assert full_before != full_after, "the projection as declared must see the role change"

    monkeypatch.setattr(
        input_binding,
        "SESSION_INPUT_PROJECTION_COLUMNS",
        tuple(column for column in input_binding.SESSION_INPUT_PROJECTION_COLUMNS if column != "role"),
    )
    with closing(_read_connection(index_db)) as conn:
        narrowed_after = session_input_bindings(conn, (session_id,))[session_id]
    with write_lease("test.restore"), closing(_write_connection(index_db)) as conn:
        conn.execute("UPDATE messages SET role = 'user' WHERE session_id = ? AND position = 0", (session_id,))
        conn.commit()
    with closing(_read_connection(index_db)) as conn:
        narrowed_before = session_input_bindings(conn, (session_id,))[session_id]

    assert narrowed_before == narrowed_after, "without the column the binding cannot see the mutation"


# ── Convergence equality over the remaining mutation kinds ────────────────────


@dataclass(frozen=True, slots=True)
class _SessionSpec:
    """One intended session, as both the mutation path and the control build it."""

    name: str
    messages: tuple[tuple[str, str], ...]
    parent: str | None = None
    day: str = _SEEDED_DAY
    repository_url: str | None = None


def _seed_spec(index_db: Path, spec: _SessionSpec) -> str:
    """Write one spec through the real writer, replacing any session of that name."""
    builder = SessionBuilder(index_db, spec.name)
    builder = builder.created_at(f"{spec.day}T00:00:00+00:00").updated_at(f"{spec.day}T01:00:00+00:00")
    if spec.repository_url is not None:
        builder = builder.git_repository_url(spec.repository_url)
    if spec.parent is not None:
        builder = builder.parent_session(f"ext-{spec.parent}").branch_type("fork")
    for index, (role, text) in enumerate(spec.messages):
        builder.add_message(role=role, text=text, timestamp=f"{spec.day}T00:{index:02d}:00+00:00")
    builder.save()
    return builder.native_session_id()


def _rebuilt_archive(tmp_path: Path, name: str, specs: Sequence[_SessionSpec]) -> Path:
    """A second archive built from scratch to the same intended end state."""
    root = tmp_path / name
    root.mkdir()
    initialize_active_archive_root(root)
    control = _index_db(root)
    for spec in specs:
        _seed_spec(control, spec)
    _converge_to_fixpoint(control)
    return control


def _tag_rollup_keys(index_db: Path) -> list[tuple[str, str, str]]:
    with closing(_read_connection(index_db)) as conn:
        rows = conn.execute("SELECT tag, bucket_day, source_name FROM session_tag_rollups ORDER BY 1, 2, 3").fetchall()
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]


def test_a_tag_change_converges_to_the_clean_rebuild(archive_root: Path, tmp_path: Path) -> None:
    """A repository that appears adds an auto tag, and with it a rollup partition.

    The tag arm of the red law, driven from the authoritative input rather than
    from the output: ``git_repository_url`` is in the session-row projection, an
    auto tag is derived from it, and the tag rollup is a query-time projection
    of the profile that carries it. Red if the binding covered messages alone,
    or if the rollup's currency were read off its own ``materialized_at`` — a
    column the view emits as the literal ``'query-time'`` on every row.
    """
    index_db = _index_db(archive_root)
    plain = _SessionSpec("tagged", (("user", "start"), ("assistant", "reply")))
    tagged = replace(plain, repository_url="https://example.invalid/demo.git")

    _seed_spec(index_db, plain)
    _converge_to_fixpoint(index_db)
    before = _tag_rollup_keys(index_db)
    assert not any(tag.startswith("repo:") for tag, _, _ in before)

    _seed_spec(index_db, tagged)
    assert _pending(index_db), "a tag-moving input change must make the partition pending"
    _converge_to_fixpoint(index_db)

    after = _tag_rollup_keys(index_db)
    assert [key for key in after if key not in before] == [("repo:demo", _SEEDED_DAY, "unknown-export")]

    control = _rebuilt_archive(tmp_path, "tag-control", [tagged])
    assert _semantic_relations(index_db) == _semantic_relations(control)
    assert after == _tag_rollup_keys(control)


def test_a_provider_day_move_retires_the_old_partition(archive_root: Path, tmp_path: Path) -> None:
    """A session that moves to another day leaves no row on the day it left.

    The provider/day rollup's partition key is (source, canonical day), and the
    day is derived from values in the session-row and message projections. Red
    if convergence only added the new bucket: the old one would survive as a
    rollup partition no session contributes to, which no per-session freshness
    check can see.
    """
    index_db = _index_db(archive_root)
    first_day = _SessionSpec("moved", (("user", "one"), ("assistant", "two")), day="2026-01-01")
    second_day = replace(first_day, day="2026-01-02")

    _seed_spec(index_db, first_day)
    _converge_to_fixpoint(index_db)
    assert {day for _, day, _ in _tag_rollup_keys(index_db)} == {"2026-01-01"}

    _seed_spec(index_db, second_day)
    assert _pending(index_db), "a day move must make the partition pending"
    _converge_to_fixpoint(index_db)

    assert {day for _, day, _ in _tag_rollup_keys(index_db)} == {"2026-01-02"}
    control = _rebuilt_archive(tmp_path, "day-control", [second_day])
    assert _semantic_relations(index_db) == _semantic_relations(control)


def test_a_zero_output_partition_matches_the_clean_rebuild(archive_root: Path, tmp_path: Path) -> None:
    """Valid-empty is a converged state the control archive reaches too.

    Red if a message-less session were treated as work never performed: the
    mutated archive would still hold a pending key while the control held none,
    and the two relations would differ by the empty partition's own rows.
    """
    index_db = _index_db(archive_root)
    empty = _SessionSpec("silent", ())
    kept = _SessionSpec("spoken", (("user", "hello"),))

    _seed_spec(index_db, empty)
    _seed_spec(index_db, kept)
    _converge_to_fixpoint(index_db)

    assert _pending(index_db) == []
    control = _rebuilt_archive(tmp_path, "empty-control", [kept, empty])
    assert _semantic_relations(index_db) == _semantic_relations(control)


def test_event_order_permutations_converge_to_the_same_relation(archive_root: Path, tmp_path: Path) -> None:
    """The property: independent writes may arrive in any order.

    Each permutation writes the same three sessions and converges after every
    write, so discovery, inspection and publication all see a different archive
    each time. Red if any step carried state across keys — a batch-scoped
    timestamp folded into a semantic column, an accumulator reused between
    sessions, a partition retired by arrival order rather than by membership.
    """
    specs = (
        _SessionSpec("alpha", (("user", "a1"),)),
        _SessionSpec("beta", (("user", "b1"), ("assistant", "b2"))),
        _SessionSpec("gamma", (("user", "g1"),), repository_url="https://example.invalid/gamma.git"),
    )

    index_db = _index_db(archive_root)
    for spec in specs:
        _seed_spec(index_db, spec)
        _converge_to_fixpoint(index_db)
    reference = _semantic_relations(index_db)
    reference_rollups = _tag_rollup_keys(index_db)

    for index, order in enumerate(itertools.permutations(specs)):
        root = tmp_path / f"order-{index}"
        root.mkdir()
        initialize_active_archive_root(root)
        permuted = _index_db(root)
        for spec in order:
            _seed_spec(permuted, spec)
            _converge_to_fixpoint(permuted)
        assert _semantic_relations(permuted) == reference, f"order {[spec.name for spec in order]} diverged"
        assert _tag_rollup_keys(permuted) == reference_rollups


def test_deleting_every_scheduling_hint_reconstructs_the_same_pending_set(archive_root: Path) -> None:
    """The domain-side half of the kernel's deleted-hint law.

    ``derived_refresh_guard``, ``fts_freshness_state`` and
    ``delegation_refresh_scope`` are the index tier's disposable refresh hints.
    Emptying all of them must not change which session partitions are pending,
    because required membership is the ``sessions`` relation itself and validity
    is re-derived from the inputs. Red if any of them became authority.
    """
    index_db = _index_db(archive_root)
    converged = _seed(index_db, "settled", messages=[("user", "one")])
    mutated = _seed(index_db, "moved", messages=[("user", "one"), ("assistant", "two")])
    _converge_to_fixpoint(index_db)
    _mutate_role(index_db, mutated)
    before = _pending(index_db)
    assert before == [mutated], "the mutated partition is the pending one before the hints are dropped"

    with write_lease("test.hints"), closing(_write_connection(index_db)) as conn:
        for relation in (
            "derived_refresh_guard",
            "fts_freshness_state",
            "delegation_refresh_scope",
        ):
            conn.execute(f"DELETE FROM {relation}")
        conn.commit()

    assert _pending(index_db) == before
    _converge_to_fixpoint(index_db)
    assert _pending(index_db) == []
    with closing(_read_connection(index_db)) as conn:
        profiled = sorted(str(row[0]) for row in conn.execute("SELECT session_id FROM session_profiles"))
    assert profiled == sorted((converged, mutated))
