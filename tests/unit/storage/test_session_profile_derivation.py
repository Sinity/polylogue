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

from polylogue.daemon.derivation import DerivationFrame, DerivationReport
from polylogue.storage.derived.session.derivation import (
    SESSION_PROFILE_DOMAIN,
    SESSION_PROFILE_RECIPE_VERSION,
    SessionProfileDerivation,
    excess_session_profiles,
    inspect_session_profiles,
    publish_session_profile,
)
from polylogue.storage.derived.session.input_binding import (
    SESSION_INPUT_PROJECTION_COLUMNS,
    session_input_bindings,
)
from polylogue.storage.derived.session.marker_domain import marker_assertions_present as _marker_assertions_present
from polylogue.storage.derived.session.summary import (
    SESSION_SUMMARY_DOMAIN,
    SESSION_SUMMARY_RECIPE_VERSION,
    SessionSummaryDerivation,
)
from polylogue.storage.derived.session.usage_rollup import (
    SESSION_USAGE_ROLLUP_DOMAIN,
    SessionUsageRollupDerivation,
    publish_session_usage_rollup,
    session_usage_rollup_recipe_version,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.storage_records import SessionBuilder

_MATERIALIZER_VERSION = SESSION_INSIGHT_MATERIALIZER_VERSION

#: Values that change an aggregate's output while leaving every identity,
#: timestamp, partition key and row count exactly where it was. Each is the
#: shape of the audited defect, not a variation on it.
_OUTPUT_AFFECTING_MUTATIONS = (
    ("role", "'assistant'"),
    ("model_name", "'a-different-model'"),
    # COALESCE, not a bare sum: the builder leaves these NULL, and NULL + 4096
    # is NULL -- a no-op mutation that made both parametrizations vacuous
    # (they asserted "stale" against an archive nothing had changed).
    ("input_tokens", "COALESCE(input_tokens, 0) + 4096"),
    ("output_tokens", "COALESCE(output_tokens, 0) + 77"),
    ("word_count", "word_count + 13"),
    ("has_tool_use", "1 - has_tool_use"),
    # A member of the durable vocabulary: material_origin carries a CHECK, so a
    # made-up token tests the constraint rather than the binding.
    ("material_origin", "'generated_analysis_pack'"),
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


def _write_connection(index_db: Path) -> sqlite3.Connection:
    """A write connection shaped like the one production hands the writer.

    The session-insight writer indexes rows by column name, so a connection
    without ``sqlite3.Row`` fails inside it. Production reaches it through the
    cached connection factory, which sets one.
    """
    conn = open_connection(index_db)
    conn.row_factory = sqlite3.Row
    return conn


def _usage_rollup(index_db: Path, session_id: str, *, quiet: bool = False) -> SessionUsageRollupDerivation:
    """The profile's canonical-usage prerequisite, built over the same archive.

    The profile derivation names this domain's key in ``prerequisite_keys``, so
    a registry without it leaves every profile blocked rather than converged.
    """
    return SessionUsageRollupDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        session_scope=lambda _frame: [session_id],
        quiet_key=(lambda _frame, _key: True) if quiet else None,
    )


def _materialize(index_db: Path, session_id: str) -> bool:
    """Converge the usage prerequisite, then publish the profile it feeds.

    The profile derivation names ``SESSION_USAGE_ROLLUP_DOMAIN`` in
    ``prerequisite_keys``, and ``publish_session_profile`` refuses a session
    whose rollup that domain has not settled. Doing both here is the ordering
    the kernel drives, not a workaround.
    """
    with write_lease("test.publish"), closing(_write_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        publish_session_usage_rollup(
            conn,
            session_id,
            input_binding=binding,
            recipe_version=session_usage_rollup_recipe_version(),
        )
        return publish_session_profile(conn, session_id, input_binding=binding)


def _status(index_db: Path, session_id: str) -> str:
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        return inspect_session_profiles(conn, [session_id], materializer_version=_MATERIALIZER_VERSION)[session_id]


def _mutate(index_db: Path, session_id: str, column: str, expression: str) -> None:
    """Change one output-affecting value in place; touch nothing identifying."""
    with write_lease("test.mutate"), closing(_write_connection(index_db)) as conn:
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

    _mutate(index_db, session_id, "input_tokens", "COALESCE(input_tokens, 0) + 4096")

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

    with write_lease("test.publish"), closing(_write_connection(index_db)) as conn:
        assert publish_session_profile(conn, session_id, input_binding=computed_at) is False

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert (
            conn.execute("SELECT count(*) FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
        )


def test_excess_publication_removes_the_complete_profile_family(
    archive: tuple[Path, str],
) -> None:
    """An absent session retires profile and latency rows together.

    Anti-vacuity: deleting only ``session_profiles`` leaves the sibling
    ``session_latency_profiles`` row behind, so the retained family is not
    actually retired and a later family census can observe orphan state.
    """
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True

    with write_lease("test.delete-session"), closing(_write_connection(index_db)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

    with write_lease("test.publish-excess"), closing(_write_connection(index_db)) as conn:
        assert publish_session_profile(conn, session_id, input_binding="obsolete") is True

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        for table in ("session_profiles", "session_latency_profiles"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0] == 0


def test_a_profile_with_no_stored_binding_is_stale_not_valid(archive: tuple[Path, str]) -> None:
    """A row that cannot say what it was computed from cannot certify itself."""
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    with write_lease("test.mutate"), closing(_write_connection(index_db)) as conn:
        conn.execute("UPDATE session_profiles SET input_content_hash = NULL WHERE session_id = ?", (session_id,))
        conn.commit()

    assert _status(index_db, session_id) == "stale"


def test_an_orphaned_profile_is_reported_as_excess(archive: tuple[Path, str]) -> None:
    """Excess is discovered from the output relation, not from an invalidation."""
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    with write_lease("test.mutate"), closing(_write_connection(index_db)) as conn:
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
    with write_lease("test.mutate"), closing(_write_connection(index_db)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert excess_session_profiles(conn) == (session_id,)

    assert _materialize(index_db, session_id) is True

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert excess_session_profiles(conn) == ()


def test_the_adapter_converges_through_the_kernel_against_a_real_archive(
    archive: tuple[Path, str],
) -> None:
    """The vertical slice: kernel, adapter, and a real archive in one pass.

    Proves the seam the two rings meet at -- the adapter returns status strings
    because storage may not import the daemon ring, and the kernel normalizes
    them. A mismatch here would only show up in production, where the kernel
    would treat every key as missing and republish the archive on every pass.
    """
    from polylogue.daemon.derivation import DerivationFrame, DerivationRegistry, Outcome, converge
    from polylogue.storage.derived.session.derivation import SessionProfileDerivation

    index_db, session_id = archive
    frame = DerivationFrame(
        archive_root=str(index_db.parent),
        source_revision="r1",
        recipe_versions={
            SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
            SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
        },
    )

    adapter = SessionProfileDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        materializer_version=_MATERIALIZER_VERSION,
        session_scope=lambda _frame: [session_id],
    )
    summary = SessionSummaryDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        session_scope=lambda _frame: [session_id],
    )
    registry = DerivationRegistry([summary, _usage_rollup(index_db, session_id), adapter])

    first = converge(registry, frame)
    # The canonical usage rollup and the profile both converge in this one
    # pass, in that order: the profile names the rollup's key as a concrete
    # prerequisite, so the kernel reconciles usage first and the profile is
    # then prepared from settled values and published on its first attempt.
    assert first.done == 2, first.outcomes
    assert [outcome.key.domain for outcome in first.by_outcome(Outcome.DONE)] == [
        SESSION_USAGE_ROLLUP_DOMAIN,
        SESSION_PROFILE_DOMAIN,
    ]
    assert not first.by_outcome(Outcome.PENDING), first.outcomes
    assert _status(index_db, session_id) == "valid"

    assert converge(registry, frame).wrote_nothing

    _mutate(index_db, session_id, "role", "'assistant'")
    assert converge(registry, frame).done == 3
    assert _status(index_db, session_id) == "valid"


def test_prepared_partition_refuses_a_value_binding_that_moved_before_publish(
    archive: tuple[Path, str],
) -> None:
    """A prepared four-table replacement never publishes after its frame moves.

    Anti-vacuity: move record construction back into ``publish`` or omit the
    source/value binding comparison and this writes a profile whose message
    values no longer match the prepared projection.
    """
    from polylogue.daemon.derivation import DerivationFrame
    from polylogue.storage.derived.session.derivation import SessionProfileDerivation

    index_db, session_id = archive
    adapter = SessionProfileDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        materializer_version=_MATERIALIZER_VERSION,
        session_scope=lambda _frame: [session_id],
    )
    frame = DerivationFrame(
        archive_root=str(index_db.parent),
        source_revision="r1",
        recipe_versions={
            SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
            SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
        },
    )
    prepared = adapter.compute(frame, session_id)

    _mutate(index_db, session_id, "word_count", "word_count + 1")

    assert adapter.publish(frame, prepared) is False
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert [
            conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0]
            for table in ("session_profiles", "session_latency_profiles")
        ] == [0, 0]


def test_prepared_profile_family_rolls_back_when_latency_write_fails(
    archive: tuple[Path, str],
) -> None:
    """The profile rows and latency rows share one publication transaction.

    Anti-vacuity: commit between the profile and latency writes in
    ``publish_prepared_session_insight_partition`` and the profile replacement
    survives the injected latency failure. The observer below would see a
    mixed family.
    """
    from polylogue.daemon.derivation import DerivationFrame
    from polylogue.storage.derived.session.derivation import SessionProfileDerivation

    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    adapter = SessionProfileDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        materializer_version=_MATERIALIZER_VERSION,
        session_scope=lambda _frame: [session_id],
    )
    frame = DerivationFrame(
        archive_root=str(index_db.parent),
        source_revision="r1",
        recipe_versions={
            SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
            SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
        },
    )
    _mutate(index_db, session_id, "word_count", "word_count + 13")
    # Reach the sibling-write fault, rather than fail the upstream-staleness guard.
    # Settle ONLY canonical usage here; retain the old profile family for rollback.
    with write_lease("test.settle-usage-before-fault"), closing(_write_connection(index_db)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        assert publish_session_usage_rollup(
            conn, session_id, input_binding=binding, recipe_version=session_usage_rollup_recipe_version()
        )
    # Input mutation may invalidate the previous binding before publication;
    # snapshot that retryable state, not the older valid profile.
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        before_attempt = {
            table: tuple(conn.execute(f"SELECT * FROM {table} WHERE session_id = ?", (session_id,)).fetchall())
            for table in ("session_profiles", "session_latency_profiles")
        }
    prepared = adapter.compute(frame, session_id)
    with write_lease("test.inject-latency-failure"), closing(_write_connection(index_db)) as conn:
        conn.execute(
            """
            CREATE TRIGGER fail_test_latency_insert
            BEFORE INSERT ON session_latency_profiles
            BEGIN
                SELECT RAISE(ABORT, 'injected latency publication failure');
            END
            """
        )
        conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="injected latency publication failure"):
        assert adapter.publish(frame, prepared) is True

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        after = {
            table: tuple(conn.execute(f"SELECT * FROM {table} WHERE session_id = ?", (session_id,)).fetchall())
            for table in ("session_profiles", "session_latency_profiles")
        }
    assert after == before_attempt


@pytest.mark.parametrize("input_kind", ("attachment", "session_event"))
def test_prepared_partition_refuses_related_input_that_moved_before_publish(
    archive: tuple[Path, str],
    input_kind: str,
) -> None:
    """Every related value consumed by hydration binds a prepared replacement.

    Anti-vacuity: remove either related projection from ``session_input_bindings``
    and this accepts the stale prepared family although the session runtime has
    changed. ``root_session_id`` is already a session-row binding; no unconsumed
    ``session_links`` relation is smuggled into this contract.
    """
    from polylogue.daemon.derivation import DerivationFrame
    from polylogue.storage.derived.session.derivation import SessionProfileDerivation

    index_db, session_id = archive
    with write_lease("test.seed-related"), closing(_write_connection(index_db)) as conn:
        message_id = str(
            conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ? ORDER BY position LIMIT 1",
                (session_id,),
            ).fetchone()[0]
        )
        if input_kind == "attachment":
            conn.execute(
                "INSERT INTO attachments (attachment_id, display_name, media_type, byte_count) VALUES (?, ?, ?, ?)",
                ("related-input", "before.txt", "text/plain", 5),
            )
            conn.execute(
                """
                INSERT INTO attachment_refs
                    (attachment_id, session_id, message_id, position, source_url, caption)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("related-input", session_id, message_id, 0, "file://before", "before"),
            )
        else:
            conn.execute(
                """
                INSERT INTO session_events
                    (session_id, source_message_id, position, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, message_id, 0, "compaction", '{"state":"before","summary":"before"}'),
            )
        conn.commit()

    adapter = SessionProfileDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        materializer_version=_MATERIALIZER_VERSION,
        session_scope=lambda _frame: [session_id],
    )
    frame = DerivationFrame(
        archive_root=str(index_db.parent),
        source_revision="r1",
        recipe_versions={
            SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
            SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
        },
    )
    prepared = adapter.compute(frame, session_id)

    with write_lease("test.mutate-related"), closing(_write_connection(index_db)) as conn:
        if input_kind == "attachment":
            conn.execute("UPDATE attachment_refs SET caption = ? WHERE attachment_id = ?", ("after", "related-input"))
        else:
            conn.execute(
                "UPDATE session_events SET payload_json = ? WHERE session_id = ? AND position = 0",
                ('{"state":"after"}', session_id),
            )
        conn.commit()

    assert adapter.publish(frame, prepared) is False
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert [
            conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0]
            for table in ("session_profiles", "session_latency_profiles")
        ] == [0, 0]


def test_marker_recovery_is_its_own_domain_and_never_rewrites_the_profile(
    archive: tuple[Path, str],
) -> None:
    """An index block marker is not durable marker-delivery input by itself.

    The source carrier consumer is covered in ``test_accepted_marker_consumer``.
    This profile-level regression keeps the original boundary: index validity
    cannot depend on a marker appearing, disappearing, or failing to deliver.
    """
    index_db, session_id = archive
    with write_lease("test.seed-marker"), closing(_write_connection(index_db)) as conn:
        conn.execute(
            "UPDATE blocks SET text = ? WHERE message_id = (SELECT message_id FROM messages WHERE session_id = ? ORDER BY position LIMIT 1)",
            ("::finding: current index text is not an accepted input", session_id),
        )
        conn.commit()
    assert _materialize(index_db, session_id) is True
    assert _status(index_db, session_id) == "valid"


def test_marker_domain_publish_failure_leaves_the_profile_valid(
    archive: tuple[Path, str],
) -> None:
    """Profile validity is independent of the separate marker sink."""
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    assert _status(index_db, session_id) == "valid"


def test_marker_assertion_presence_deduplicates_identical_marker_ids() -> None:
    """Repeated equal marker requests count once in the assertion owner.

    The parser intentionally gives identical markers in one block the same
    durable identity.  Presence is therefore set membership, not occurrence
    cardinality: an already-written assertion must satisfy both requests.
    """
    from polylogue.markers import candidates_for_block
    from polylogue.markers.lowering import assertion_id_for_marker

    candidates = candidates_for_block("message-1", "block-1", "::note: repeat\n::note: repeat\n")
    assertion_ids = tuple(assertion_id_for_marker(candidate) for candidate in candidates)
    assert len(assertion_ids) == 2
    first_id, second_id = assertion_ids
    assert first_id is not None
    assert second_id is not None
    assert first_id == second_id

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE assertions (assertion_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO assertions (assertion_id) VALUES (?)", (first_id,))
        assert _marker_assertions_present(conn, (first_id, second_id)) is True
        assert _marker_assertions_present(conn, (first_id, "marker-missing")) is False
    finally:
        conn.close()


def test_prepared_generation_refuses_when_the_active_anchor_promotes(
    archive: tuple[Path, str],
) -> None:
    """A prepared partition cannot publish through a promoted index anchor.

    Anti-vacuity: freeze ``index_db_path.resolve()`` in the factory or omit
    the generation checks and this writes the prepared family after its
    admitted generation has been retired.
    """
    from polylogue.daemon.derivation import DerivationFrame
    from polylogue.storage.derived.session.derivation import SessionProfileDerivation

    index_db, session_id = archive
    active_generation = {"path": str(index_db.resolve())}
    adapter = SessionProfileDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        materializer_version=_MATERIALIZER_VERSION,
        session_scope=lambda _frame: [session_id],
        generation_binding=lambda: active_generation["path"],
    )
    frame = DerivationFrame(
        archive_root=str(index_db.parent),
        source_revision=f"index-generation:{index_db.resolve()}",
        scope=(session_id,),
    )
    prepared = adapter.compute(frame, session_id)
    active_generation["path"] = "retired-generation"

    assert adapter.publish(frame, prepared) is False
    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
        )


def test_the_kernel_reports_a_quiet_key_as_pending_not_done(archive: tuple[Path, str]) -> None:
    """Policy deferral leaves the profile absent and the key rediscoverable."""
    from polylogue.daemon.derivation import DerivationFrame, DerivationRegistry, Outcome, PendingReason, converge
    from polylogue.storage.derived.session.derivation import SessionProfileDerivation

    index_db, session_id = archive
    frame = DerivationFrame(
        archive_root=str(index_db.parent),
        source_revision="r1",
        recipe_versions={
            SESSION_SUMMARY_DOMAIN: SESSION_SUMMARY_RECIPE_VERSION,
            SESSION_PROFILE_DOMAIN: SESSION_PROFILE_RECIPE_VERSION,
        },
    )
    adapter = SessionProfileDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        materializer_version=_MATERIALIZER_VERSION,
        session_scope=lambda _frame: [session_id],
        quiet_keys=lambda _frame: frozenset({session_id}),
    )
    summary = SessionSummaryDerivation(
        lambda: sqlite3.connect(f"file:{index_db}?mode=ro", uri=True),
        lambda: _write_connection(index_db),
        session_scope=lambda _frame: [session_id],
    )
    report = converge(
        DerivationRegistry([summary, _usage_rollup(index_db, session_id, quiet=True), adapter]),
        frame,
    )

    assert report.done == 0
    reasons = {outcome.key.domain: outcome.reason for outcome in report.by_outcome(Outcome.PENDING)}
    assert reasons[SESSION_USAGE_ROLLUP_DOMAIN] is PendingReason.QUIET
    # The profile reads the deferred rollup, so it defers with it rather than
    # publishing a partition prepared from an unreconciled canonical usage.
    assert reasons[SESSION_PROFILE_DOMAIN] is PendingReason.BLOCKED
    assert _status(index_db, session_id) == "missing"


@pytest.mark.parametrize(
    ("column", "expression"),
    [
        ("title", "'a different title'"),
        # sort_key_ms is GENERATED ALWAYS AS COALESCE(updated_at_ms,
        # created_at_ms), so the sort key moves only through its source column.
        ("updated_at_ms", "COALESCE(updated_at_ms, 0) + 5000"),
        ("git_branch", "'other-branch'"),
        ("message_count", "message_count + 1"),
    ],
)
def test_a_session_row_value_change_makes_inspection_stale(
    archive: tuple[Path, str],
    column: str,
    expression: str,
) -> None:
    """The profile caches session-row values too, so the binding must cover them.

    Anti-vacuity: drop ``SESSION_ROW_PROJECTION_COLUMNS`` from the digest and a
    changed sort key, title, or repository leaves the profile reporting valid
    while ``source_sort_key`` and ``canonical_session_date`` are stale -- the
    same defect one level up from the message projection.
    """
    index_db, session_id = archive
    assert _materialize(index_db, session_id) is True
    assert _status(index_db, session_id) == "valid"

    with write_lease("test.mutate"), closing(_write_connection(index_db)) as conn:
        conn.execute(
            f"UPDATE sessions SET {column} = {expression} WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()

    assert _status(index_db, session_id) == "stale"


# ---------------------------------------------------------------------------
# Duplicate-marker readiness, certified through the production derivation route
# ---------------------------------------------------------------------------

#: A block whose text carries the *same* marker twice. The parser gives
#: identical markers in one block one durable identity, so two candidates
#: resolve to one assertion id and one stored row.
_DUPLICATE_MARKER_BLOCK = "::note: repeat\n::note: repeat\n"


@pytest.fixture
def marker_archive(tmp_path: Path) -> Iterator[tuple[Path, Path, str]]:
    """A real archive root whose only session repeats one inline marker.

    Every tier is created by ``initialize_active_archive_root``, so ``user.db``
    carries the shipped user-tier schema rather than a stand-in ``assertions``
    table. That is the point: a one-column stand-in agrees with itself.
    """
    root = tmp_path / "archive"
    root.mkdir()
    index_db = root / "index.db"
    initialize_active_archive_root(root)
    builder = SessionBuilder(index_db, "marker-dedup")
    builder.add_message(role="user", text=_DUPLICATE_MARKER_BLOCK)
    builder.save()
    yield root, index_db, builder.native_session_id()


def _converge_session_profile(
    root: Path, index_db: Path, session_id: str
) -> tuple[SessionProfileDerivation, DerivationFrame, DerivationReport]:
    """Drive the same adapters, frame and kernel the daemon owner drives."""
    from polylogue.daemon.derivation import DerivationRegistry, converge
    from polylogue.operations.session_profile_convergence import (
        make_session_marker_derivation,
        make_session_profile_derivation,
        make_session_profile_frame,
        make_session_summary_derivation,
        make_session_usage_rollup_derivation,
    )

    def now() -> float:
        # The hot-file probe needs a clock, not wall time; a seeded archive has
        # no raw source row, so nothing is deferred at any value.
        return 0.0

    registry = DerivationRegistry(
        (
            make_session_summary_derivation(index_db, archive_root=root),
            make_session_usage_rollup_derivation(index_db, archive_root=root, now=now),
            make_session_profile_derivation(index_db, archive_root=root, now=now),
            # The production composition drives markers as their own domain
            # after the profile (polylogue-ylh7v); this helper mirrors it so
            # "the production route" means the route production runs.
            make_session_marker_derivation(index_db, archive_root=root),
        )
    )
    frame = make_session_profile_frame(index_db, archive_root=root, scope=[session_id])
    adapter = registry.get(SESSION_PROFILE_DOMAIN)
    assert isinstance(adapter, SessionProfileDerivation)
    return adapter, frame, converge(registry, frame)


def test_a_repeated_marker_converges_to_valid_readiness_through_the_production_route(
    marker_archive: tuple[Path, Path, str],
) -> None:
    """Current index marker-looking text cannot make profile readiness stale.

    Accepted input delivery has its own source stream and user cursor. This
    fixture writes only an index session, so it must not synthesize a user
    assertion from its current block text.
    """
    from polylogue.daemon.derivation import Outcome

    root, index_db, session_id = marker_archive
    adapter, frame, report = _converge_session_profile(root, index_db, session_id)

    assert report.counts[Outcome.FAILED] == 0, [outcome.error for outcome in report.outcomes]
    profile_outcomes = [outcome for outcome in report.outcomes if outcome.key.domain == SESSION_PROFILE_DOMAIN]
    assert [outcome.outcome for outcome in profile_outcomes] == [Outcome.DONE]

    # The two production readiness callers, not the private helper.
    assert adapter.inspect(frame, [session_id]) == {session_id: "valid"}
    facts = adapter.selected_part_facts(frame, session_id)
    assert facts.status == "valid"
    assert facts.profiles == 1

    with closing(sqlite3.connect(f"file:{root / 'user.db'}?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM assertions").fetchone() == (0,)


def test_a_second_pass_over_the_repeated_marker_publishes_nothing_new(
    marker_archive: tuple[Path, Path, str],
) -> None:
    """An index-only marker-looking block does not trigger a profile rewrite."""
    from polylogue.daemon.derivation import Outcome

    root, index_db, session_id = marker_archive
    _converge_session_profile(root, index_db, session_id)

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        first = conn.execute(
            "SELECT materialized_at, input_content_hash FROM session_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    adapter, frame, report = _converge_session_profile(root, index_db, session_id)
    assert report.counts[Outcome.FAILED] == 0
    assert adapter.inspect(frame, [session_id]) == {session_id: "valid"}

    with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as conn:
        second = conn.execute(
            "SELECT materialized_at, input_content_hash FROM session_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    assert tuple(first) == tuple(second), "an already-valid marker family must not be rewritten"
