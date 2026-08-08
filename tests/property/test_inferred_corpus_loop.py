"""The inferred manifest's supported specs must reach real archive convergence."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.outcomes import OutcomeStatus
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
from polylogue.daemon.fts_startup import record_fts_freshness_snapshot_sync
from polylogue.maintenance.archive_verification import verify_archive
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusSpec
from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.models import SyntheticSchemaSelection
from polylogue.schemas.synthetic.wire_formats import build_wire_support_receipt
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_canonical_snapshot import archive_snapshot, assert_archives_equivalent
from tests.infra.convergence_harness import rebuild_retained_raw_index, set_debt_retry_at
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


def _run_retry_in_fresh_process(index_db: Path) -> int:
    """Exercise the production debt drain after a real interpreter restart."""
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(index_db.parent)
    script = (
        "from pathlib import Path\n"
        "from polylogue.daemon.cli import _drain_convergence_debt_once\n"
        f"print('RETRIED=' + str(_drain_convergence_debt_once(Path({str(index_db)!r}))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"fresh-process convergence retry failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RETRIED="):
            return int(line.removeprefix("RETRIED="))
    raise AssertionError(f"fresh-process convergence retry emitted no result marker: {completed.stdout!r}")


def _inferred_selection() -> tuple[CorpusSpec, SyntheticSchemaSelection]:
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    manifest = compile_inferred_corpus_manifest(
        registry=registry,
        wire_support_receipt=build_wire_support_receipt(registry=registry),
    )
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    for spec, selection in zip(handoff.specs, handoff.selections, strict=True):
        if spec.provider == "codex":
            return spec, selection
    raise AssertionError("the persisted inferred corpus has no Codex selection")


def _ingest_and_converge_sources(
    archive_root: Path,
    sources: Sequence[Source],
) -> tuple[str, ...]:
    initialize_active_archive_root(archive_root)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for source_index, source in enumerate(sources):
            if source.path is None:
                raise AssertionError(f"source path required for inferred fixture: {source.name}")
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.from_string(source.name),
                    payload=source.path.read_bytes(),
                    source_path=str(source.path),
                    source_index=source_index,
                    acquired_at_ms=source_index + 1,
                )
            )
    backfill = backfill_historical_revision_evidence(archive_root, selected_raw_ids=raw_ids, ingest_workers=1)
    assert backfill.scanned == backfill.classified_full > 0
    assert backfill.quarantined == 0
    assert backfill.adoption_deferred == 0
    with sqlite3.connect(archive_root / "index.db") as conn:
        session_ids = tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))
    states, _timings = DaemonConverger(
        (make_fts_stage(archive_root / "index.db"), make_insights_stage(archive_root / "index.db"))
    ).converge_sessions(session_ids)
    assert states and all(state.converged and state.last_error is None for state in states.values())
    with sqlite3.connect(archive_root / "index.db") as conn:
        record_fts_freshness_snapshot_sync(conn)
    return session_ids


def _converge_existing_archive(archive_root: Path) -> None:
    """Run post-reindex convergence over the promoted generation."""
    with sqlite3.connect(archive_root / "index.db") as conn:
        session_ids = tuple(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))
    states, _timings = DaemonConverger(
        (make_fts_stage(archive_root / "index.db"), make_insights_stage(archive_root / "index.db"))
    ).converge_sessions(session_ids)
    assert states and all(state.converged and state.last_error is None for state in states.values())
    with sqlite3.connect(archive_root / "index.db") as conn:
        record_fts_freshness_snapshot_sync(conn)


def _lineage_material() -> tuple[bytes, bytes, str, str]:
    spec, selection = _inferred_selection()
    parent_spec = replace(
        spec,
        count=1,
        messages_min=3,
        messages_max=3,
        seed=101,
        session_native_ids=("inferred-lineage-parent",),
        style="demo-attachments",
    )
    child_spec = replace(
        spec,
        count=1,
        messages_min=3,
        messages_max=3,
        seed=202,
        session_native_ids=("inferred-lineage-child",),
        style="demo-attachments",
    )
    parent_raw = SyntheticCorpus.generate_batch_for_selection(selection, parent_spec).artifacts[0].raw_bytes
    child_raw = SyntheticCorpus.generate_batch_for_selection(selection, child_spec).artifacts[0].raw_bytes
    parent_records = [json.loads(line) for line in parent_raw.decode().splitlines() if line]
    child_records = [json.loads(line) for line in child_raw.decode().splitlines() if line]
    child_meta = next(record for record in child_records if record.get("type") == "session_meta")
    child_meta.setdefault("payload", {})["forked_from_id"] = "inferred-lineage-parent"
    child_records = [
        child_meta,
        *(record for record in parent_records if record.get("type") != "session_meta"),
        *(record for record in child_records if record.get("type") != "session_meta"),
    ]
    return (
        ("\n".join(json.dumps(record, sort_keys=True) for record in parent_records) + "\n").encode(),
        ("\n".join(json.dumps(record, sort_keys=True) for record in child_records) + "\n").encode(),
        "codex-session:inferred-lineage-parent",
        "codex-session:inferred-lineage-child",
    )


def _build_lineage_archive(
    archive_root: Path,
    parent_raw: bytes,
    child_raw: bytes,
) -> tuple[str, ...]:
    source_root = archive_root.parent / "inferred-lineage-material"
    source_root.mkdir(parents=True, exist_ok=True)
    parent_path = source_root / "parent.jsonl"
    child_path = source_root / "child.jsonl"
    parent_path.write_bytes(parent_raw)
    child_path.write_bytes(child_raw)
    return _ingest_and_converge_sources(
        archive_root,
        (
            Source(name="codex", path=source_root / "parent.jsonl"),
            Source(name="codex", path=source_root / "child.jsonl"),
        ),
    )


def test_persisted_catalog_manifest_reaches_real_ingest_and_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SchemaRegistry(storage_root=SCHEMA_DIR)
    manifest = compile_inferred_corpus_manifest(
        registry=registry,
        wire_support_receipt=build_wire_support_receipt(registry=registry),
    )
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
    manifest = compile_inferred_corpus_manifest(
        registry=registry,
        wire_support_receipt=build_wire_support_receipt(registry=registry),
    )
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


@pytest.mark.frozen_clock_modules("polylogue.storage.sqlite.archive_tiers.revision_governance")
def test_inferred_selection_retained_raw_reindex_matches_canonical_snapshot(
    tmp_path: Path,
    frozen_clock: object,
) -> None:
    spec, selection = _inferred_selection()
    source_root = tmp_path / "retained-source"
    written = SyntheticCorpus.write_selection_artifacts(selection, spec, source_root, prefix="retained")
    archive_root = tmp_path / "archive"
    session_ids = _ingest_and_converge_sources(
        archive_root,
        (Source(name=spec.provider, path=written.files[0]),),
    )
    before = archive_snapshot(archive_root, session_ids=session_ids)

    receipt = rebuild_retained_raw_index(archive_root)
    _converge_existing_archive(archive_root)

    assert receipt.raw_session_count == receipt.selected_raw_count > 0
    assert archive_snapshot(archive_root, session_ids=session_ids) == before


def test_inferred_selection_debt_recovers_in_a_fresh_process(tmp_path: Path) -> None:
    spec, selection = _inferred_selection()
    source_root = tmp_path / "recovery-source"
    written = SyntheticCorpus.write_selection_artifacts(selection, spec, source_root, prefix="recovery")
    archive_root = tmp_path / "archive"
    session_ids = _ingest_and_converge_sources(
        archive_root,
        (Source(name=spec.provider, path=written.files[0]),),
    )
    baseline = archive_snapshot(archive_root, session_ids=session_ids)
    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute(
            "DELETE FROM session_profiles WHERE session_id IN ({})".format(",".join("?" for _ in session_ids)),
            session_ids,
        )
        conn.commit()
    cursor = CursorStore(archive_root / "index.db")
    for session_id in session_ids:
        cursor.record_convergence_debt(
            stage="insights",
            subject_type="session_id",
            subject_id=session_id,
            error="inferred-corpus convergence interruption",
        )
        set_debt_retry_at(
            archive_root / "ops.db",
            stage="insights",
            subject_type="session_id",
            subject_id=session_id,
            retry_at="1970-01-01T00:00:00+00:00",
        )

    assert _run_retry_in_fresh_process(archive_root / "index.db") == len(session_ids)
    assert CursorStore(archive_root / "index.db").list_convergence_debt(limit=100) == []
    assert archive_snapshot(archive_root, session_ids=session_ids) == baseline


@pytest.mark.frozen_clock_modules("polylogue.storage.sqlite.archive_tiers.revision_governance")
def test_inferred_lineage_reindex_preserves_composition_and_detects_tail_mutation(
    tmp_path: Path,
    frozen_clock: object,
) -> None:
    parent_raw, child_raw, parent_id, child_id = _lineage_material()
    canonical_root = tmp_path / "canonical"
    canonical_ids = _build_lineage_archive(canonical_root, parent_raw, child_raw)

    assert set(canonical_ids) == {parent_id, child_id}
    with sqlite3.connect(canonical_root / "index.db") as conn:
        parent = conn.execute(
            "SELECT root_session_id, message_count FROM sessions WHERE session_id = ?",
            (parent_id,),
        ).fetchone()
        child = conn.execute(
            "SELECT parent_session_id, root_session_id, message_count FROM sessions WHERE session_id = ?",
            (child_id,),
        ).fetchone()
        link = conn.execute(
            "SELECT resolved_dst_session_id, branch_point_message_id, inheritance, status "
            "FROM session_links WHERE src_session_id = ?",
            (child_id,),
        ).fetchone()
    assert parent is not None and child is not None and link is not None
    assert parent[0] == parent_id
    assert child[0] == parent_id and child[1] == parent_id and int(child[2]) == 3
    assert link[0] == parent_id and link[1] is not None and link[2] == "prefix-sharing" and link[3] is None
    with ArchiveStore.open_existing(canonical_root) as archive:
        composed = archive.read_session(child_id)
    assert len(composed.messages) > int(parent[1])
    assert any(message.message_id.startswith(parent_id + ":") for message in composed.messages)

    rebuilt_root = tmp_path / "rebuilt"
    _build_lineage_archive(rebuilt_root, parent_raw, child_raw)
    rebuild_retained_raw_index(rebuilt_root)
    _converge_existing_archive(rebuilt_root)
    assert archive_snapshot(rebuilt_root) == archive_snapshot(canonical_root)

    mutated_records = [json.loads(line) for line in child_raw.decode().splitlines() if line]
    mutated_message = mutated_records[-1].get("payload", mutated_records[-1])
    assert isinstance(mutated_message, dict)
    mutated_message["content"] = [{"type": "output_text", "text": "mutation-sensitive inferred lineage tail"}]
    mutated_raw = ("\n".join(json.dumps(record, sort_keys=True) for record in mutated_records) + "\n").encode()
    mutated_root = tmp_path / "mutated"
    _build_lineage_archive(mutated_root, parent_raw, mutated_raw)
    with pytest.raises(AssertionError, match="canonical archive snapshots differ"):
        assert_archives_equivalent(canonical_root, mutated_root)
    with sqlite3.connect(mutated_root / "index.db") as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM blocks WHERE session_id = ? AND text = ?",
                (child_id, "mutation-sensitive inferred lineage tail"),
            ).fetchone()
            is not None
        )
        assert conn.execute(
            "SELECT resolved_dst_session_id, inheritance, status FROM session_links WHERE src_session_id = ?",
            (child_id,),
        ).fetchone() == (parent_id, "prefix-sharing", None)
