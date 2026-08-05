"""The inferred manifest's supported specs must reach real archive convergence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.schemas.registry import SCHEMA_DIR, SchemaRegistry
from tests.infra.convergence_harness import (
    assert_corpus_materialization,
    build_converged_archive,
    inferred_convergence_corpus,
)
from tests.infra.inferred_corpus import (
    assert_inferred_corpus_convergence_handoff_complete,
    build_inferred_corpus_convergence_handoff,
    compile_inferred_corpus_manifest,
)


def test_actual_catalog_manifest_remains_fail_closed() -> None:
    manifest = compile_inferred_corpus_manifest(registry=SchemaRegistry(storage_root=SCHEMA_DIR))
    handoff = build_inferred_corpus_convergence_handoff(manifest)
    assert_inferred_corpus_convergence_handoff_complete(manifest, handoff)
    assert handoff.specs == manifest.supported_specs == ()
    assert handoff.selections == ()
    assert manifest.unsupported_records


def test_every_supported_persisted_selection_reaches_real_ingest_and_convergence(tmp_path: Path) -> None:
    corpus = inferred_convergence_corpus()
    assert corpus.manifest is not None
    assert corpus.manifest.supported_specs
    archive = build_converged_archive(tmp_path / "all-supported", corpus)
    assert_corpus_materialization(archive)
    with sqlite3.connect(archive.root / "index.db") as conn:
        session_count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        message_count = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    assert session_count == len(corpus.members)
    assert message_count >= session_count
