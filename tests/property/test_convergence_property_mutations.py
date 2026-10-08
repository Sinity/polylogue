"""Mutation red twins for the production dependencies of convergence."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import tests.infra.convergence_harness as convergence_harness
from polylogue.storage.derived.session.derivation import SessionProfileDerivation
from polylogue.storage.fts.derivation import FtsDerivationAdapter
from polylogue.storage.sqlite.archive_tiers import write as archive_write
from polylogue.storage.sqlite.connection_profile import open_connection
from tests.infra.convergence_harness import (
    build_converged_archive,
    converge_convergence_archive,
    ingest_composed_sources,
    initialize_active_archive,
    rich_convergence_sources,
)
from tests.infra.convergence_laws import (
    ConvergenceLaw,
    assert_projection_matches_oracle,
    authoritative_sessions,
    generated_convergence_workload,
    read_semantic_projection,
    semantic_oracle,
)
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.live_ingest import prepared_live_convergence_owner


def test_convergence_property_fts_publication_mutation_red_twin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bypassed common FTS publication cannot report a converged archive."""
    composed = rich_convergence_sources()
    initialize_active_archive(tmp_path / "mutated")
    monkeypatch.setattr(FtsDerivationAdapter, "publish", lambda *_args, **_kwargs: False)

    archive = ingest_composed_sources(
        tmp_path / "mutated",
        composed,
        session_indexes=tuple(range(len(composed.sessions))),
        converge_after_each=False,
    )
    with pytest.raises(AssertionError, match="common FTS derivation left pending work"):
        converge_convergence_archive(archive)


def test_convergence_property_insight_repair_mutation_red_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bypassed production insight rebuild cannot report a converged archive."""
    composed = rich_convergence_sources()
    initialize_active_archive(tmp_path / "mutated")
    monkeypatch.setattr(SessionProfileDerivation, "publish", lambda *_args, **_kwargs: False)

    archive = ingest_composed_sources(
        tmp_path / "mutated",
        composed,
        session_indexes=tuple(range(len(composed.sessions))),
        converge_after_each=False,
    )
    with pytest.raises(AssertionError, match="typed session-profile convergence left pending work"):
        converge_convergence_archive(archive)


def test_retained_writer_refuses_session_when_raw_acquisition_is_bypassed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retained writer refuses a session when its source producer returns no acquired Raw."""
    composed = rich_convergence_sources()
    mutated_root = tmp_path / "mutated"
    initialize_active_archive(mutated_root)

    monkeypatch.setattr(convergence_harness, "write_source_raw_session", lambda *_args, **_kwargs: "bypassed-raw")
    with pytest.raises(KeyError, match="unknown raw revision bypassed-raw"):
        ingest_composed_sources(
            mutated_root,
            composed,
            session_indexes=tuple(range(len(composed.sessions))),
            converge_after_each=False,
        )


def test_convergence_harness_binds_raw_receipt_before_equal_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal raw/attachment bytes cannot strand the earlier publication receipt."""
    composed = rich_convergence_sources()
    session = composed.sessions[0]
    shared_payload = f"fixture attachment bytes {session.id}".encode()
    monkeypatch.setattr(convergence_harness, "_raw_payload", lambda _session: shared_payload)
    initialize_active_archive(tmp_path)

    ingest_composed_sources(
        tmp_path,
        composed,
        session_indexes=(0,),
        converge_after_each=False,
    )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)


def _profiles_disagreeing_with_canonical_titles(root: Path) -> list[str]:
    """Session profiles whose title is not the canonical session title."""
    with closing(sqlite3.connect(root / "index.db")) as conn:
        rows = conn.execute(
            """
            SELECT profile.session_id
            FROM session_profiles AS profile
            JOIN sessions AS session ON session.session_id = profile.session_id
            WHERE profile.title IS NOT session.title
            ORDER BY profile.session_id
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def test_convergence_property_materialized_content_mutation_red_twin(tmp_path: Path) -> None:
    """A corrupted materialized profile is rejected by the canonical-row oracle.

    The profile title is derived from the canonical ``sessions`` row, so the
    oracle compares the two tiers rather than confirming the corruption.
    Anti-vacuity: the converged archive must agree before the mutation, and
    the mutated session is the one the oracle reports; an oracle that stopped
    reading ``session_profiles`` would report nothing after it.
    """
    composed = rich_convergence_sources()
    mutated = build_converged_archive(tmp_path / "mutated", composed)
    assert _profiles_disagreeing_with_canonical_titles(mutated.root) == []

    with closing(sqlite3.connect(mutated.root / "index.db")) as conn, conn:
        target = conn.execute("SELECT session_id FROM session_profiles ORDER BY session_id LIMIT 1").fetchone()
        assert target is not None
        cursor = conn.execute(
            """
            UPDATE session_profiles
            SET title = COALESCE(title, '') || ' [materialized-content-mutation]'
            WHERE session_id = ?
            """,
            (target[0],),
        )
        if cursor.rowcount != 1:
            raise AssertionError("materialized-content mutation did not change one profile row")

    assert _profiles_disagreeing_with_canonical_titles(mutated.root) == [str(target[0])]


@pytest.mark.parametrize("mutated", [False, True], ids=["green", "mutant"])
def test_order_sensitive_overwrite_has_permutation_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutated: bool
) -> None:
    """The stale-write gate keeps a newer revision independent of ingest order."""
    from tests.infra.source_composer import compose_append_revision_chain

    composed = compose_append_revision_chain(revision_count=2, messages_per_revision=1)
    root = tmp_path / "mutated"
    initialize_active_archive(root)

    def write_revision(index: int) -> None:
        parsed = convergence_harness._parsed_session(composed.sessions[index], corpus_index=index).model_copy(
            update={"attachments": []}
        )
        with closing(open_connection(root / "index.db")) as conn:
            conn.row_factory = sqlite3.Row
            write_fixture_index_session(conn, parsed)

    write_revision(1)
    if mutated:
        monkeypatch.setattr(archive_write, "should_skip_stale_replace", lambda **_kwargs: False)
    write_revision(0)
    observed = read_semantic_projection(root, probe_terms=("revision",))
    expected = semantic_oracle(authoritative_sessions(composed), probe_terms=("revision",))
    if mutated:
        with pytest.raises(AssertionError, match=ConvergenceLaw.PERMUTATION.value):
            assert_projection_matches_oracle(observed, expected, law=ConvergenceLaw.PERMUTATION)
    else:
        assert_projection_matches_oracle(
            observed,
            expected,
            law=ConvergenceLaw.PERMUTATION,
        )


@pytest.mark.parametrize("mutated", [False, True], ids=["green", "mutant"])
def test_omitted_fts_required_member_has_pending_work_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutated: bool
) -> None:
    """The convergence guard rejects a common FTS pass that omits required work."""
    workload = generated_convergence_workload()
    initialize_active_archive(tmp_path / "mutated")
    archive = ingest_composed_sources(
        tmp_path / "mutated",
        workload.sources,
        session_indexes=tuple(range(len(workload.sources.sessions))),
        converge_after_each=False,
    )
    required_page = FtsDerivationAdapter.required_page

    def omit_tail(
        self: FtsDerivationAdapter, frame: object, *, cursor: str | None, limit: int
    ) -> tuple[tuple[str, ...], str | None]:
        keys, next_cursor = required_page(self, frame, cursor=cursor, limit=limit)
        if cursor is None and getattr(frame, "scope", None) and len(keys) > 1:
            return keys[:1], None
        return keys, next_cursor

    if mutated:
        monkeypatch.setattr(FtsDerivationAdapter, "required_page", omit_tail)
        with pytest.raises(AssertionError, match="common FTS derivation left pending work"):
            converge_convergence_archive(archive)
    else:
        converge_convergence_archive(archive)


@pytest.mark.parametrize("mutated", [False, True], ids=["green", "mutant"])
def test_late_parent_prefix_resolution_has_append_prefix_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutated: bool
) -> None:
    """Late lineage resolution removes an already-written replayed prefix."""
    from polylogue.storage.sqlite.archive_tiers import write as write_module
    from tests.infra.source_composer import compose_fork_prefix_tail_lineage

    composed = compose_fork_prefix_tail_lineage()
    if mutated:
        # The mutant skips resolution and so reports no session it changed.
        monkeypatch.setattr(write_module, "_resolve_session_graph", lambda *_args, **_kwargs: set())
    archive = build_converged_archive(tmp_path / "archive", composed, session_order=(1, 0))
    observed = read_semantic_projection(archive.root, probe_terms=("shared",))
    expected = semantic_oracle(authoritative_sessions(composed), probe_terms=("shared",))
    if mutated:
        with pytest.raises(AssertionError, match=ConvergenceLaw.APPEND_PREFIX.value):
            assert_projection_matches_oracle(observed, expected, law=ConvergenceLaw.APPEND_PREFIX)
    else:
        assert_projection_matches_oracle(
            observed,
            expected,
            law=ConvergenceLaw.APPEND_PREFIX,
        )


def test_unchanged_reingest_does_not_reach_the_production_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settled Raw owner leaves the production writer untouched on repeat."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    payload = (
        b'{"type":"session_meta","payload":{"id":"unchanged-reingest"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m1",'
        b'"role":"user","content":[{"type":"input_text","text":"keep this prompt"}]}}\n'
    )

    def acquire() -> str:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="unchanged-reingest.jsonl",
                canonical_source_path="unchanged-reingest.jsonl",
                acquired_at_ms=1,
            )

    writer_calls: list[str] = []
    import polylogue.storage.sqlite.archive_tiers.revision_governance as revision_governance

    async def settle_then_repeat() -> None:
        raw_id = await run_archive_fixture_write(archive_root, acquire)
        async with prepared_live_convergence_owner(archive_root) as owner:
            first = await owner.converge_raw_id(raw_id)
            assert first.done == 1 and first.failed == first.pending == 0, first.outcomes

            from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

            write = write_parsed_session_to_archive

            def observe_writer(*args: object, **kwargs: object) -> str:
                writer_calls.append("write")
                return write(*args, **kwargs)  # type: ignore[arg-type]

            monkeypatch.setattr(revision_governance, "write_parsed_session_to_archive", observe_writer)
            repeated = await owner.converge_raw_id(raw_id)
            assert repeated.failed == repeated.pending == 0, repeated.outcomes

    asyncio.run(settle_then_repeat())
    assert not writer_calls
