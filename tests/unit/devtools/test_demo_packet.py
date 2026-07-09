from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_demo_packet_registry
from devtools.demo_packet import (
    PACKET_FILENAMES,
    PROVENANCE_STANZA_FIELDS,
    REPORT_SECTION_ORDER,
    validate_packet,
)


def _write_conforming_packet(packet_dir: Path) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    (packet_dir / "PROMPT.md").write_text("# prompt\n", encoding="utf-8")
    stanza_lines = "\n".join(f"{field}: value-{field}" for field in PROVENANCE_STANZA_FIELDS)
    (packet_dir / "finding.yaml").write_text(f"{stanza_lines}\nclaim: a claim\n", encoding="utf-8")
    report_sections = "\n".join(f"## {section}\ntext\n" for section in REPORT_SECTION_ORDER)
    (packet_dir / "report.md").write_text(report_sections, encoding="utf-8")
    (packet_dir / "evidence.ndjson").write_text('{"ref": "x"}\n', encoding="utf-8")
    (packet_dir / "queries.ndjson").write_text('{"text": "find"}\n', encoding="utf-8")
    (packet_dir / "checks.json").write_text(
        json.dumps({"pass": True, "unsupported_claims": [], "coverage_notes": ""}),
        encoding="utf-8",
    )
    (packet_dir / "run.log").write_text("$ polylogue find\n", encoding="utf-8")


def test_validate_packet_accepts_a_conforming_packet(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)

    result = validate_packet(packet_dir)

    assert result.ok is True
    assert result.missing_files == ()
    assert result.missing_stanza_fields == ()
    assert result.malformed_sections == ()
    assert result.errors == ()


def test_validate_packet_reports_missing_files(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)
    (packet_dir / "run.log").unlink()

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert "run.log" in result.missing_files


def test_validate_packet_reports_missing_stanza_fields(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)
    (packet_dir / "finding.yaml").write_text("archive_cursor: only-one-field\n", encoding="utf-8")

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert set(result.missing_stanza_fields) == set(PROVENANCE_STANZA_FIELDS) - {"archive_cursor"}


def test_validate_packet_reports_missing_report_sections(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)
    (packet_dir / "report.md").write_text("## claim\nonly one section\n", encoding="utf-8")

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert "reproduce" in result.malformed_sections
    assert "claim" not in result.malformed_sections


def test_validate_packet_reports_invalid_checks_json(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)
    (packet_dir / "checks.json").write_text("{not valid json", encoding="utf-8")

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert any("checks.json" in error for error in result.errors)


def test_validate_packet_reports_invalid_ndjson(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)
    (packet_dir / "evidence.ndjson").write_text("not json at all\n", encoding="utf-8")

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert any("evidence.ndjson" in error for error in result.errors)


def test_all_packet_filenames_are_checked(tmp_path: Path) -> None:
    # Sanity: every required filename actually participates in the missing-file check.
    packet_dir = tmp_path / "empty"
    packet_dir.mkdir()
    result = validate_packet(packet_dir)
    assert set(result.missing_files) == set(PACKET_FILENAMES)


def _write_registry(registry_path: Path, entries: list[dict[str, object]]) -> None:
    registry_path.write_text(json.dumps(entries), encoding="utf-8")


def test_registry_lint_passes_for_conforming_registry(tmp_path: Path) -> None:
    packet_dir = tmp_path / "demos" / "stub"
    _write_conforming_packet(packet_dir)
    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        [
            {
                "slug": "stub",
                "prompt_path": str(packet_dir / "PROMPT.md"),
                # lint_demo_registry resolves packet_dir relative to repo_root
                # (Path.cwd() in the real CLI) -- use an absolute path here so
                # this test doesn't depend on the pytest process's cwd.
                "packet_dir": str(packet_dir),
                "mode": "private",
                "required_primitives": [],
            }
        ],
    )

    exit_code = verify_demo_packet_registry.main(["--registry", str(registry_path), "--json"])

    assert exit_code == 0


def test_registry_lint_catches_a_missing_packet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        [
            {
                "slug": "ghost-demo",
                "prompt_path": "nope/PROMPT.md",
                "packet_dir": "nope/packet",
                "mode": "private",
                "required_primitives": [],
            }
        ],
    )

    exit_code = verify_demo_packet_registry.main(["--registry", str(registry_path), "--json"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["entries"][0]["packet_missing"] is True


def test_registry_lint_reports_missing_registry_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_path = tmp_path / "does-not-exist.json"

    exit_code = verify_demo_packet_registry.main(["--registry", str(missing_path), "--json"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
