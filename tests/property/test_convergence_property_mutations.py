"""Mutation red twins for the production dependencies of convergence."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
import tests.infra.convergence_harness as convergence_harness
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.storage.fts import fts_lifecycle
from polylogue.storage.insights.session import rebuild as insight_rebuild
from polylogue.storage.sqlite.archive_tiers import write as archive_write
from polylogue.storage.sqlite.connection_profile import open_connection
from tests.infra.convergence_harness import (
    assert_archives_equivalent,
    build_converged_archive,
    converge_convergence_archive,
    ingest_convergence_pathology,
    initialize_active_archive,
    rich_convergence_pathology,
)
from tests.infra.convergence_laws import (
    ConvergenceLaw,
    assert_projection_matches_oracle,
    authoritative_sessions,
    generated_convergence_workload,
    read_semantic_projection,
    semantic_oracle,
)


def test_convergence_property_fts_repair_mutation_red_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bypassed production FTS repair cannot report a converged archive."""
    pathology = rich_convergence_pathology()
    initialize_active_archive(tmp_path / "mutated")
    monkeypatch.setattr(fts_lifecycle, "repair_message_fts_index_sync", lambda *_args, **_kwargs: None)

    archive = ingest_convergence_pathology(
        tmp_path / "mutated",
        pathology,
        session_indexes=tuple(range(len(pathology.sessions))),
        converge_after_each=False,
    )
    with pytest.raises(AssertionError, match="production convergence left pending work"):
        converge_convergence_archive(archive)


def test_convergence_property_insight_repair_mutation_red_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bypassed production insight rebuild cannot report a converged archive."""
    pathology = rich_convergence_pathology()
    initialize_active_archive(tmp_path / "mutated")
    monkeypatch.setattr(insight_rebuild, "rebuild_session_insights_sync", lambda *_args, **_kwargs: None)

    archive = ingest_convergence_pathology(
        tmp_path / "mutated",
        pathology,
        session_indexes=tuple(range(len(pathology.sessions))),
        converge_after_each=False,
    )
    with pytest.raises(AssertionError, match="production convergence left pending work"):
        converge_convergence_archive(archive)


def test_convergence_property_raw_replay_mutation_red_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The comparator catches a replay path that bypasses durable raw acquisition."""
    pathology = rich_convergence_pathology()
    canonical = build_converged_archive(tmp_path / "canonical", pathology)
    mutated_root = tmp_path / "mutated"
    initialize_active_archive(mutated_root)

    monkeypatch.setattr(convergence_harness, "write_source_raw_session", lambda *_args, **_kwargs: "bypassed-raw")
    mutated = ingest_convergence_pathology(
        mutated_root,
        pathology,
        session_indexes=tuple(range(len(pathology.sessions))),
        converge_after_each=False,
    )
    converge_convergence_archive(mutated)

    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(canonical, mutated)


def test_convergence_harness_binds_raw_receipt_before_equal_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal raw/attachment bytes cannot strand the earlier publication receipt."""
    pathology = rich_convergence_pathology()
    session = pathology.sessions[0]
    shared_payload = f"fixture attachment bytes {session.id}".encode()
    monkeypatch.setattr(convergence_harness, "_raw_payload", lambda _session: shared_payload)
    initialize_active_archive(tmp_path)

    ingest_convergence_pathology(
        tmp_path,
        pathology,
        session_indexes=(0,),
        converge_after_each=False,
    )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)


def test_convergence_property_materialized_content_mutation_red_twin(tmp_path: Path) -> None:
    """Changing a materialized insight row cannot pass the semantic comparator."""
    pathology = rich_convergence_pathology()
    canonical = build_converged_archive(tmp_path / "canonical", pathology)
    mutated = build_converged_archive(tmp_path / "mutated", pathology)

    with sqlite3.connect(mutated.root / "index.db") as conn:
        cursor = conn.execute(
            """
            UPDATE session_work_events
            SET summary = summary || ' [materialized-content-mutation]'
            WHERE event_id = (SELECT event_id FROM session_work_events ORDER BY event_id LIMIT 1)
            """
        )
        if cursor.rowcount != 1:
            raise AssertionError("materialized-content mutation did not change one work-event row")
        conn.commit()

    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(canonical, mutated)


@pytest.mark.parametrize("mutated", [False, True], ids=["green", "mutant"])
def test_order_sensitive_overwrite_has_permutation_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutated: bool
) -> None:
    """The stale-write gate keeps a newer revision independent of ingest order."""
    from polylogue.pipeline.services.ingest_batch import _core as ingest_core
    from tests.infra.pathology_composer import compose_append_revision_chain

    pathology = compose_append_revision_chain(revision_count=2, messages_per_revision=1)
    root = tmp_path / "mutated"
    initialize_active_archive(root)

    def write_revision(index: int) -> None:
        parsed = convergence_harness._parsed_session(pathology.sessions[index], corpus_index=index).model_copy(
            update={"attachments": []}
        )
        payload = SessionWritePayload(
            session_id=str(make_session_id(Provider.CODEX, parsed.provider_session_id)),
            content_hash=str(session_content_hash(parsed)),
            parsed_session=parsed,
            message_count=len(parsed.messages),
            attachment_count=len(parsed.attachments),
            raw_id=None,
        )
        with closing(open_connection(root / "index.db")) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                ingest_core._write_session(conn, payload)

    write_revision(1)
    if mutated:
        monkeypatch.setattr(ingest_core, "should_skip_stale_replace", lambda **_kwargs: False)
    write_revision(0)
    observed = read_semantic_projection(root, probe_terms=("revision",))
    expected = semantic_oracle(authoritative_sessions(pathology), probe_terms=("revision",))
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
def test_omitted_fts_batch_member_has_pending_work_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutated: bool
) -> None:
    """The convergence guard rejects a targeted FTS repair that omits work."""
    workload = generated_convergence_workload()
    initialize_active_archive(tmp_path / "mutated")
    archive = ingest_convergence_pathology(
        tmp_path / "mutated",
        workload.pathology,
        session_indexes=tuple(range(len(workload.pathology.sessions))),
        converge_after_each=False,
    )
    repair = fts_lifecycle.repair_message_fts_index_sync

    def omit_tail(conn: sqlite3.Connection, session_ids: object, **kwargs: object) -> None:
        repair(conn, tuple(session_ids)[:1], **kwargs)  # type: ignore[arg-type]

    if mutated:
        monkeypatch.setattr(fts_lifecycle, "repair_message_fts_index_sync", omit_tail)
        with pytest.raises(AssertionError, match="production convergence left pending work"):
            converge_convergence_archive(archive)
    else:
        converge_convergence_archive(archive)


@pytest.mark.parametrize("mutated", [False, True], ids=["green", "mutant"])
def test_late_parent_prefix_resolution_has_append_prefix_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mutated: bool
) -> None:
    """Late lineage resolution removes an already-written replayed prefix."""
    from polylogue.storage.sqlite.archive_tiers import write as write_module
    from tests.infra.pathology_composer import compose_fork_prefix_tail_lineage

    pathology = compose_fork_prefix_tail_lineage()
    if mutated:
        monkeypatch.setattr(write_module, "_resolve_session_graph", lambda *_args, **_kwargs: None)
    archive = build_converged_archive(tmp_path / "archive", pathology, session_order=(1, 0))
    observed = read_semantic_projection(archive.root, probe_terms=("shared",))
    expected = semantic_oracle(authoritative_sessions(pathology), probe_terms=("shared",))
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
    """The content-hash gate bypasses the production writer for unchanged input."""
    workload = generated_convergence_workload()
    archive = build_converged_archive(tmp_path / "archive", workload.pathology)
    writer_calls: list[str] = []
    write = archive_write.write_parsed_session_to_archive

    def observe_writer(*args: object, **kwargs: object) -> str:
        writer_calls.append("write")
        return write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ingest_batch_core, "write_parsed_session_to_archive", observe_writer)
    ingest_convergence_pathology(
        archive.root,
        workload.pathology,
        session_indexes=tuple(range(len(workload.pathology.sessions))),
        converge_after_each=False,
    )
    assert not writer_calls
