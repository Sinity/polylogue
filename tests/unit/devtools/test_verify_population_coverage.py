"""Red twins for ``devtools gate population-coverage``.

Anti-vacuity: removing one origin declaration, one artifact rule, or one
matrix witness turns the respective construct ``uncovered``; an unknown
artifact kind is typed unsupported evidence; the gate creates nothing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from devtools.verify_population_coverage import (
    COVERED,
    UNCOVERED,
    UNSUPPORTED_DECLARED,
    CoverageConstruct,
    declaration_constructs,
    evaluate_population_coverage,
    inventory_constructs,
    main,
)
from polylogue.core.enums import Origin
from polylogue.sources.origin_specs import ORIGIN_SPECS
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.origin_capability_matrix import load_manifest

FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures"


def _by_key(constructs: tuple[CoverageConstruct, ...], family: str) -> dict[str, CoverageConstruct]:
    return {construct.key: construct for construct in constructs if construct.family == family}


def _seed_inventory(root: Path) -> Path:
    initialize_active_archive_root(root)
    conn = sqlite3.connect(root / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms,
                                     detected_provider)
            VALUES ('raw-codex', 'codex-session', 'c1', '/src/c1.jsonl', ?, 10, 100, 'codex'),
                   ('raw-claude', 'claude-code-session', 'a1', '/src/agent-1.meta.json', ?, 10, 100, 'claude-code')
            """,
            (b"a" * 32, b"b" * 32),
        )
        conn.execute(
            """
            INSERT INTO raw_artifacts(artifact_id, raw_id, origin, source_path, artifact_kind, support_status,
                                      classification_reason, parse_as_session, first_observed_at_ms,
                                      last_observed_at_ms)
            VALUES ('art-codex', 'raw-codex', 'codex-session', '/src/c1.jsonl', 'session_record_stream',
                    'supported_parseable', 'test', 1, 100, 100),
                   ('art-claude', 'raw-claude', 'claude-code-session', '/src/agent-1.meta.json',
                    'agent_sidecar_meta', 'recognized_unparsed', 'test', 0, 100, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()
    return root / "source.db"


def test_every_declared_origin_is_witnessed_or_declared_unsupported() -> None:
    constructs = declaration_constructs()
    assert constructs
    assert not [c for c in constructs if c.status == UNCOVERED]
    statuses = {c.key: c.status for c in constructs}
    assert statuses[Origin.CODEX_SESSION.value] == COVERED
    assert statuses[Origin.BEADS_ISSUE.value] == UNSUPPORTED_DECLARED


def test_seeded_inventory_is_fully_covered(tmp_path: Path) -> None:
    constructs = inventory_constructs(_seed_inventory(tmp_path))
    assert not [c for c in constructs if c.status == UNCOVERED], constructs
    origins = _by_key(constructs, "origin")
    assert origins["codex-session"].count == 1
    routes = _by_key(constructs, "detector-route")
    assert routes["codex-session/codex"].status == COVERED
    kinds = _by_key(constructs, "artifact-kind")
    assert kinds["claude-code-session/agent_sidecar_meta/recognized_unparsed"].status == COVERED
    assert kinds["claude-code-session/agent_sidecar_meta/recognized_unparsed"].witness == "attempt_meta"
    assert kinds["codex-session/session_record_stream/supported_parseable"].status == COVERED


def test_removed_origin_declaration_turns_its_inventory_uncovered(tmp_path: Path) -> None:
    """Anti-vacuity: the inventory half must consult the declarations, not the enum."""
    specs = tuple(spec for spec in ORIGIN_SPECS if spec.origin is not Origin.CODEX_SESSION)
    constructs = inventory_constructs(_seed_inventory(tmp_path), specs=specs)
    origins = _by_key(constructs, "origin")
    assert origins["codex-session"].status == UNCOVERED
    assert origins["codex-session"].route == "no OriginSpec"
    routes = _by_key(constructs, "detector-route")
    assert routes["codex-session/codex"].status == UNCOVERED


def test_removed_artifact_rule_turns_its_kind_uncovered(tmp_path: Path) -> None:
    specs = tuple(
        replace(spec, artifact_rules=()) if spec.origin is Origin.CLAUDE_CODE_SESSION else spec for spec in ORIGIN_SPECS
    )
    constructs = inventory_constructs(_seed_inventory(tmp_path), specs=specs)
    kinds = _by_key(constructs, "artifact-kind")
    assert kinds["claude-code-session/agent_sidecar_meta/recognized_unparsed"].status == UNCOVERED


def test_removed_matrix_witness_turns_declaration_uncovered() -> None:
    manifest = load_manifest()
    stripped = replace(
        manifest,
        entries=tuple(
            replace(entry, witnesses=()) if entry.origin is Origin.CODEX_SESSION else entry
            for entry in manifest.entries
        ),
    )
    constructs = declaration_constructs(manifest=stripped)
    statuses = {c.key: c for c in constructs}
    assert statuses[Origin.CODEX_SESSION.value].status == UNCOVERED
    assert statuses[Origin.CODEX_SESSION.value].witness == "no matrix witness"


def test_unknown_artifact_kind_is_typed_unsupported_evidence(tmp_path: Path) -> None:
    source_db = _seed_inventory(tmp_path)
    conn = sqlite3.connect(source_db)
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-odd', 'aistudio-drive', NULL, '/src/odd.json', ?, 10, 100)
            """,
            (b"c" * 32,),
        )
        conn.execute(
            """
            INSERT INTO raw_artifacts(artifact_id, raw_id, origin, source_path, artifact_kind, support_status,
                                      classification_reason, parse_as_session, first_observed_at_ms,
                                      last_observed_at_ms)
            VALUES ('art-odd', 'raw-odd', 'aistudio-drive', '/src/odd.json', 'unknown', 'unknown', 'test', 0, 100, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()
    report = evaluate_population_coverage(tmp_path)
    assert not report.ok
    assert [c.key for c in report.uncovered] == ["aistudio-drive/unknown/unknown"]
    assert report.uncovered[0].route == "no artifact declaration"


def test_gate_reports_static_only_without_an_archive_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = sorted(path for path in FIXTURE_ROOT.rglob("*") if path.is_file())
    assert main(["--archive-root", str(tmp_path / "absent"), "--json"]) == 0
    payload = capsys.readouterr().out
    assert '"inventory_evaluated": false' in payload
    assert main(["--archive-root", str(tmp_path / "absent")]) == 0
    assert "not evaluated" in capsys.readouterr().out
    after = sorted(path for path in FIXTURE_ROOT.rglob("*") if path.is_file())
    assert before == after


def test_gate_exit_code_follows_inventory_coverage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_inventory(tmp_path)
    assert main(["--archive-root", str(tmp_path)]) == 0
    assert "PASS" in capsys.readouterr().out
    conn = sqlite3.connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms,
                                     detected_provider)
            VALUES ('raw-mis', 'codex-session', 'x', '/src/x.jsonl', ?, 10, 100, 'chatgpt')
            """,
            (b"d" * 32,),
        )
        conn.commit()
    finally:
        conn.close()
    assert main(["--archive-root", str(tmp_path)]) == 1
    assert "UNCOVERED detector-route codex-session/chatgpt" in capsys.readouterr().out
