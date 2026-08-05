"""The inferred manifest's supported specs must reach real archive convergence."""

from __future__ import annotations

import asyncio
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.outcomes import OutcomeStatus
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
from polylogue.maintenance.archive_verification import verify_archive
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from tests.infra.inferred_corpus import (
    assert_inferred_corpus_convergence_handoff_complete,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
    read_inferred_corpus_manifest,
    write_inferred_corpus_manifest,
)


def test_actual_catalog_manifest_remains_fail_closed() -> None:
    manifest = compile_inferred_corpus_manifest(registry=SchemaRegistry(storage_root=SCHEMA_DIR))
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert_inferred_corpus_convergence_handoff_complete(manifest, handoff)
    assert handoff.specs == manifest.supported_specs
    assert any(spec.provider == "codex" for spec in handoff.specs)
    assert handoff.selections
    assert any(selection.provider == "codex" for selection in handoff.selections)
    assert manifest.unsupported_records


def _assert_fts_match(conn: sqlite3.Connection, token: str) -> None:
    rows = conn.execute(
        """
        SELECT b.block_id
        FROM messages_fts
        JOIN blocks AS b ON b.rowid = messages_fts.rowid
        WHERE messages_fts MATCH ?
        """,
        (token,),
    ).fetchall()
    assert rows, f"FTS MATCH returned no blocks for generated token {token!r}"


def test_persisted_catalog_manifest_reaches_real_ingest_and_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    manifest = compile_inferred_corpus_manifest(registry=registry)
    manifest_path = tmp_path / "manifest.json"
    write_inferred_corpus_manifest(manifest, manifest_path)
    persisted = read_inferred_corpus_manifest(manifest_path)
    handoff = build_inferred_corpus_convergence_handoff(manifest_path)

    assert persisted.supported_specs
    assert len(handoff.specs) == len(handoff.selections) >= 1
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
        searchable_texts = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT search_text FROM blocks WHERE NULLIF(search_text, '') IS NOT NULL ORDER BY rowid"
            ).fetchall()
        )

    assert session_count > 0 and message_count > 0
    assert profile_count == session_count
    assert profile_message_count == message_count
    assert materialized_profiles == session_count
    assert fts_source_count == fts_index_count
    assert searchable_texts and all(text.strip() for text in searchable_texts)

    generated_text = written.batch.raw_items[0].decode("utf-8")
    generated_tokens = tuple(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9_]{4,}", generated_text.lower())))
    with sqlite3.connect(archive_root / "index.db") as conn:
        search_token = next(
            (
                token
                for token in generated_tokens
                if conn.execute("SELECT 1 FROM messages_fts WHERE messages_fts MATCH ? LIMIT 1", (token,)).fetchone()
                is not None
            ),
            None,
        )
        assert search_token is not None, "generated Codex content produced no searchable FTS token"
        _assert_fts_match(conn, search_token)

        conn.execute(
            "UPDATE blocks SET text = '', tool_name = '', tool_input = NULL WHERE session_id IN ({})".format(
                ",".join("?" for _ in session_ids)
            ),
            session_ids,
        )
        cleared_count = int(
            conn.execute("SELECT COUNT(*) FROM blocks WHERE NULLIF(search_text, '') IS NOT NULL").fetchone()[0]
        )
        assert cleared_count == 0
        with pytest.raises(AssertionError, match="FTS MATCH returned no blocks"):
            _assert_fts_match(conn, search_token)


def test_every_supported_inferred_element_reaches_convergence_and_red_twin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every persisted schema element through the production route.

    The manifest is the authority for both supported and explicitly
    unsupported elements.  This test must never silently fall back to the
    default provider schema or drop an element because its wire route is not
    synthesizable.  The final mutation is a ground-truth red twin for the
    archive verification registry, proving that a green convergence run does
    not merely verify the generator's own bookkeeping.
    """

    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    manifest = compile_inferred_corpus_manifest(registry=registry)
    manifest_path = tmp_path / "manifest.json"
    write_inferred_corpus_manifest(manifest, manifest_path)
    persisted = read_inferred_corpus_manifest(manifest_path)
    handoff = build_inferred_corpus_convergence_handoff(manifest_path)
    assert handoff.specs
    assert len(handoff.specs) == len(handoff.selections)
    assert persisted.supported_specs == manifest.supported_specs
    assert handoff.specs == persisted.supported_specs
    assert all(
        all(construct.state == "supported" for construct in entry.key.construct_support)
        for entry in manifest.entries
        if entry.spec is not None
    )
    assert all(entry.unsupported is not None for entry in manifest.entries if entry.spec is None)

    source_root = tmp_path / "inferred-source"
    sources: list[Source] = []
    expected_session_count = 0
    expected_source_paths: set[str] = set()
    for index, (spec, selection) in enumerate(zip(handoff.specs, handoff.selections, strict=True)):
        output_dir = source_root / f"{index:03d}-{selection.provider}-{selection.element_kind or 'root'}"
        written = SyntheticCorpus.write_selection_artifacts(
            selection,
            spec,
            output_dir,
            prefix=f"inferred-{index:03d}",
        )
        assert written.batch.report.generated_count == spec.count
        assert written.files and all(path.stat().st_size > 0 for path in written.files)
        expected_session_count += written.batch.report.generated_count
        source_paths = tuple(path.relative_to(source_root) for path in written.files)
        expected_source_paths.update(str(path) for path in source_paths)
        sources.extend(Source(name=selection.provider, path=path) for path in source_paths)

    archive_root = tmp_path / "archive"
    monkeypatch.chdir(source_root)
    ingest_result = asyncio.run(parse_sources_archive(archive_root, sources))
    assert ingest_result.parse_failures == 0
    assert ingest_result.counts["sessions"] == expected_session_count
    assert ingest_result.counts["sessions"] > 0
    assert ingest_result.counts["messages"] > 0

    with sqlite3.connect(archive_root / "source.db") as conn:
        admitted_source_paths = {str(row[0]) for row in conn.execute("SELECT DISTINCT source_path FROM raw_sessions")}
    assert expected_source_paths <= admitted_source_paths

    with sqlite3.connect(archive_root / "index.db") as conn:
        session_ids = tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))
    states, _timings = DaemonConverger(
        (make_fts_stage(archive_root / "index.db"), make_insights_stage(archive_root / "index.db"))
    ).converge_sessions(session_ids)
    assert states and all(state.converged and state.last_error is None for state in states.values())
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("ANALYZE")

    green = verify_archive(archive_root)
    green_summary = [(check.name, check.status.value, check.summary) for check in green.checks]
    assert green.blocking is False, green_summary
    assert all(check.status is OutcomeStatus.OK for check in green.checks), green_summary

    broken_root = tmp_path / "broken"
    shutil.copytree(archive_root, broken_root)
    with sqlite3.connect(broken_root / "index.db") as conn:
        conn.execute("UPDATE sessions SET message_count = message_count + 1 WHERE session_id = ?", (session_ids[0],))
        conn.commit()
    red = verify_archive(broken_root, checks=("message-count-projection",))
    check = next(item for item in red.checks if item.name == "message-count-projection")
    assert check.status is OutcomeStatus.ERROR
