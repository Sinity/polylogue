"""The inferred manifest's supported specs must reach real archive convergence."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from tests.infra.inferred_corpus import (
    assert_inferred_corpus_convergence_handoff_complete,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
    read_inferred_corpus_manifest,
    representative_inferred_corpus_registry,
    write_inferred_corpus_manifest,
)


def test_actual_catalog_manifest_remains_fail_closed() -> None:
    manifest = compile_inferred_corpus_manifest(registry=SchemaRegistry(storage_root=SCHEMA_DIR))
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert_inferred_corpus_convergence_handoff_complete(manifest, handoff)
    assert handoff.specs == manifest.supported_specs == ()
    assert handoff.selections == ()
    assert manifest.unsupported_records


def test_persisted_representative_manifest_reaches_real_ingest_and_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = representative_inferred_corpus_registry(SchemaRegistry(storage_root=SCHEMA_DIR))
    manifest = compile_inferred_corpus_manifest(registry=registry)
    manifest_path = tmp_path / "manifest.json"
    write_inferred_corpus_manifest(manifest, manifest_path)
    persisted = read_inferred_corpus_manifest(manifest_path)
    handoff = build_inferred_corpus_convergence_handoff(manifest_path)

    assert persisted.supported_specs
    assert len(handoff.specs) == len(handoff.selections) == 1
    spec = handoff.specs[0]
    selection = handoff.selections[0]
    source_root = tmp_path / "synthetic-source"
    written = SyntheticCorpus.write_selection_artifacts(
        selection,
        spec,
        source_root / spec.provider,
        prefix="inferred",
    )
    assert written.batch.report.generated_count == spec.count == 1
    assert written.files and all(path.stat().st_size > 0 for path in written.files)

    archive_root = tmp_path / "archive"
    source_path = written.files[0].relative_to(source_root)
    monkeypatch.chdir(source_root)
    ingest_result = asyncio.run(parse_sources_archive(archive_root, [Source(name=spec.provider, path=source_path)]))
    assert ingest_result.counts["sessions"] > 0
    assert ingest_result.counts["messages"] > 0

    with sqlite3.connect(archive_root / "index.db") as conn:
        session_ids = tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))
    converger = DaemonConverger(
        (make_fts_stage(archive_root / "index.db"), make_insights_stage(archive_root / "index.db"))
    )
    states, _timings = converger.converge_sessions(session_ids)
    assert states and all(state.converged and state.last_error is None for state in states.values())

    with sqlite3.connect(archive_root / "index.db") as conn:
        session_count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        message_count = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        profile_count = int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0])
        profile_message_count = int(
            conn.execute("SELECT COALESCE(SUM(message_count), 0) FROM session_profiles").fetchone()[0]
        )
        materialized_profiles = int(
            conn.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE materialized_at != '' AND message_count > 0"
            ).fetchone()[0]
        )
        fts_source_count = int(
            conn.execute("SELECT COUNT(*) FROM blocks WHERE NULLIF(search_text, '') IS NOT NULL").fetchone()[0]
        )
        fts_index_count = int(conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0])

    assert session_count > 0 and message_count > 0
    assert profile_count == session_count
    assert profile_message_count == message_count
    assert materialized_profiles == session_count
    assert fts_source_count == fts_index_count
