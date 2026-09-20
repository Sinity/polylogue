"""Red twins for ``devtools gate population-coverage``.

Anti-vacuity: removing one origin declaration, one artifact rule, or one
matrix witness turns the respective construct ``uncovered``; an unknown
artifact kind is typed unsupported evidence; the gate creates nothing.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from devtools.parser_census import Census, build_census, source_denominator, write_census
from devtools.verify_population_coverage import (
    COVERED,
    UNCOVERED,
    UNSUPPORTED_DECLARED,
    CoverageConstruct,
    PopulationCoverageError,
    census_constructs,
    declaration_constructs,
    evaluate_population_coverage,
    inventory_constructs,
    main,
    resolve_census,
)
from polylogue.core.enums import Origin
from polylogue.sources.origin_specs import ORIGIN_SPECS
from polylogue.storage.blob_store import BlobStore
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


def test_stale_unknown_artifact_is_covered_only_by_fresh_shape_evidence(tmp_path: Path) -> None:
    """A bounded old observation is covered by its current positive shape only."""
    source_db = _seed_inventory(tmp_path)
    payload = (
        b'{"runSettings":{"model":"models/synthetic"},'
        b'"systemInstruction":{},"chunkedPrompt":{"chunks":['
        b'{"role":"user","text":"synthetic"}]}}'
    )
    blob_hash, blob_size = BlobStore(tmp_path / "blob").write_from_bytes(payload)
    conn = sqlite3.connect(source_db)
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-stale-drive', 'aistudio-drive', NULL, '/drive-cache/gemini/synthetic.json', ?, ?, 100)
            """,
            (bytes.fromhex(blob_hash), blob_size),
        )
        conn.execute(
            """
            INSERT INTO raw_artifacts(artifact_id, raw_id, origin, source_path, artifact_kind, support_status,
                                      classification_reason, parse_as_session, first_observed_at_ms,
                                      last_observed_at_ms)
            VALUES ('art-stale-drive', 'raw-stale-drive', 'aistudio-drive', '/drive-cache/gemini/synthetic.json',
                    'unknown', 'decode_failed', 'decode failure: JSONDecodeError', 0, 100, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    kinds = _by_key(inventory_constructs(source_db), "artifact-kind")
    stale = kinds["aistudio-drive/unknown/decode_failed"]
    assert stale.status == COVERED
    assert stale.count == 1
    assert "fresh shape=session_document/supported_parseable" in stale.route
    assert stale.witness == "fresh classification: session-bearing document"


def test_stale_missing_drive_export_uses_only_its_narrow_path_declaration(tmp_path: Path) -> None:
    """Missing retained bytes do not become a provider-wide unknown allow-list."""
    source_db = _seed_inventory(tmp_path)
    conn = sqlite3.connect(source_db)
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-missing-drive', 'aistudio-drive', NULL,
                    '/drive-cache/gemini/Synthetic_Conversation-0123456789abcdef0123456789abcdef.json', ?, 100, 100)
            """,
            (b"f" * 32,),
        )
        conn.execute(
            """
            INSERT INTO raw_artifacts(artifact_id, raw_id, origin, source_path, artifact_kind, support_status,
                                      classification_reason, parse_as_session, first_observed_at_ms,
                                      last_observed_at_ms)
            VALUES ('art-missing-drive', 'raw-missing-drive', 'aistudio-drive',
                    '/drive-cache/gemini/Synthetic_Conversation-0123456789abcdef0123456789abcdef.json',
                    'unknown', 'decode_failed', 'decode failure: JSONDecodeError', 0, 100, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    kinds = _by_key(inventory_constructs(source_db), "artifact-kind")
    stale = kinds["aistudio-drive/unknown/decode_failed"]
    assert stale.status == COVERED
    assert "declared path shape=session_document" in stale.route
    assert "drive_session_export" in stale.witness

    stripped = tuple(
        replace(
            spec,
            artifact_rules=tuple(rule for rule in spec.artifact_rules if rule.coverage_role != "drive_session_export"),
        )
        if spec.origin is Origin.AISTUDIO_DRIVE
        else spec
        for spec in ORIGIN_SPECS
    )
    without = _by_key(inventory_constructs(source_db, specs=stripped), "artifact-kind")
    assert without["aistudio-drive/unknown/decode_failed"].status == UNCOVERED


def test_stale_missing_unknown_drive_payload_is_not_covered_by_session_path(tmp_path: Path) -> None:
    """A filename outside the declared Drive export shape remains uncovered."""
    source_db = _seed_inventory(tmp_path)
    conn = sqlite3.connect(source_db)
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-missing-odd-drive', 'aistudio-drive', NULL, '/drive-cache/gemini/not-an-export.json', ?, 100, 100)
            """,
            (b"g" * 32,),
        )
        conn.execute(
            """
            INSERT INTO raw_artifacts(artifact_id, raw_id, origin, source_path, artifact_kind, support_status,
                                      classification_reason, parse_as_session, first_observed_at_ms,
                                      last_observed_at_ms)
            VALUES ('art-missing-odd-drive', 'raw-missing-odd-drive', 'aistudio-drive',
                    '/drive-cache/gemini/not-an-export.json', 'unknown', 'decode_failed',
                    'decode failure: JSONDecodeError', 0, 100, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    stale = _by_key(inventory_constructs(source_db), "artifact-kind")["aistudio-drive/unknown/decode_failed"]
    assert stale.status == UNCOVERED


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


def test_drive_applet_log_is_covered_by_its_declared_artifact_rule(tmp_path: Path) -> None:
    """polylogue-1wjiw: the AI Studio applet access log has a declared route."""
    source_db = _seed_inventory(tmp_path)
    conn = sqlite3.connect(source_db)
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-applet', 'aistudio-drive', NULL, '/drive-cache/gemini/applet_access_history.json', ?, 10, 100)
            """,
            (b"e" * 32,),
        )
        conn.execute(
            """
            INSERT INTO raw_artifacts(artifact_id, raw_id, origin, source_path, artifact_kind, support_status,
                                      classification_reason, parse_as_session, first_observed_at_ms,
                                      last_observed_at_ms)
            VALUES ('art-applet', 'raw-applet', 'aistudio-drive', '/drive-cache/gemini/applet_access_history.json',
                    'metadata_document', 'recognized_unparsed', 'test', 0, 100, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    kinds = _by_key(inventory_constructs(source_db), "artifact-kind")
    covered = kinds["aistudio-drive/metadata_document/recognized_unparsed"]
    assert covered.status == COVERED
    assert covered.witness == "applet_access_log"

    stripped = tuple(
        replace(spec, artifact_rules=()) if spec.origin is Origin.AISTUDIO_DRIVE else spec for spec in ORIGIN_SPECS
    )
    without = _by_key(inventory_constructs(source_db, specs=stripped), "artifact-kind")
    assert without["aistudio-drive/metadata_document/recognized_unparsed"].status == UNCOVERED


# ---------------------------------------------------------------------------
# Census-backed population (polylogue-olw5e AC3)
# ---------------------------------------------------------------------------


def _census_of(corpus_root: Path) -> Census:
    """A census over a two-origin synthetic corpus, through the real builder."""
    claude = corpus_root / "claude-code"
    chatgpt = corpus_root / "chatgpt"
    claude.mkdir(parents=True)
    chatgpt.mkdir(parents=True)
    for source in sorted((FIXTURE_ROOT / "claude-code").glob("*.jsonl")):
        shutil.copy(source, claude / source.name)
    shutil.copy(
        FIXTURE_ROOT / "chatgpt" / "native-conversation-v1.json",
        chatgpt / "native-conversation-v1.json",
    )
    members, denominator = source_denominator([("claude-code", claude), ("chatgpt", chatgpt)])
    return build_census(members, denominator, workers=1)


def test_coverage_reads_the_recorded_denominator_from_a_census(tmp_path: Path) -> None:
    """The real source population is evidence the census already recorded.

    Anti-vacuity: the origin constructs below carry the census member counts,
    so a reader that returned no constructs -- or one that ignored the census
    and reported only the declarations -- leaves ``origins`` empty and the
    count assertion red.
    """
    census = _census_of(tmp_path / "corpus")

    constructs = census_constructs(census)

    origins = _by_key(constructs, "origin")
    assert "claude-code-session" in origins
    assert origins["claude-code-session"].status == COVERED
    recorded = sum(1 for member in census.members if member.origin == "claude-code-session")
    assert recorded > 0
    assert origins["claude-code-session"].count == recorded
    assert sum(construct.count for construct in origins.values()) == len(census.members)
    assert all(construct.status != UNCOVERED for construct in constructs), [
        construct.to_dict() for construct in constructs if construct.status == UNCOVERED
    ]


def test_a_census_origin_without_a_declaration_is_uncovered(tmp_path: Path) -> None:
    """A construct the census observed and nothing declares fails the gate.

    Anti-vacuity: if the reader silently dropped members whose origin has no
    ``OriginSpec``, this stays green while an undeclared population exists.
    """
    census = _census_of(tmp_path / "corpus")
    undeclared = replace(census.members[0], origin="invented-origin")
    census = replace(census, members=(undeclared, *census.members[1:]))

    report = evaluate_population_coverage(None, census=census)

    assert report.census_evaluated
    assert not report.ok
    # Both the origin token and the (origin, artifact kind) pair it carries
    # lose their declaration, and both are reported.
    assert [construct.key for construct in report.uncovered] == [
        "invented-origin",
        "invented-origin/session_document/",
    ]


def test_an_absent_or_bounded_census_is_refused_not_reported_as_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unavailable measurement must never read as an empty uncovered set.

    Anti-vacuity: returning an empty construct list instead of raising makes
    the gate exit 0 here, which is exactly the failure this guards -- a
    missing census reported as a fully covered population.
    """
    empty = tmp_path / "census"
    empty.mkdir()
    with pytest.raises(PopulationCoverageError, match="no census in"):
        resolve_census(empty)
    with pytest.raises(PopulationCoverageError, match="no census directory"):
        resolve_census(tmp_path / "absent")

    census = _census_of(tmp_path / "corpus")
    write_census(replace(census, partial=True), empty / "census-20260920T000000Z.json")
    with pytest.raises(PopulationCoverageError, match="bounded"):
        resolve_census(empty)

    assert main(["--census-dir", str(empty), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["refusal"]["type"] == "PopulationCoverageError"


def test_the_gate_reads_the_newest_census_in_the_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--census-dir`` selects the latest document, and says which one.

    Anti-vacuity: a reader pinned to the first (or any fixed) document reports
    the stale census's path, and the printed path assertion goes red.
    """
    directory = tmp_path / "census"
    directory.mkdir()
    census = _census_of(tmp_path / "corpus")
    stale = replace(census, members=(replace(census.members[0], origin="invented-origin"),))
    write_census(stale, directory / "census-20260101T000000Z.json")
    newest = directory / "census-20260920T000000Z.json"
    write_census(census, newest)

    assert main(["--census-dir", str(directory)]) == 0
    printed = capsys.readouterr().out
    assert f"Census: read from {newest}" in printed
    assert "PASS" in printed
