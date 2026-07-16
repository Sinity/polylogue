## Turn 92 (role=assistant) — complete committed Python test file printed inline: tests/unit/devtools/test_demo_packet.py (contract tests for Demo Packet v2 and the committed packet registry)

bash -lc cd /mnt/data/demo-v2-work && cat > tests/unit/devtools/test_demo_packet.py <<'PY'
"""Contract tests for Demo Packet v2 and the committed packet registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_demo_packet_registry
from devtools.demo_packet import (
    PACKET_FILENAMES,
    PROVENANCE_STANZA_FIELDS,
    REPORT_SECTION_ORDER,
    lint_demo_registry,
    validate_packet,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "demo-packet-v2"


def _valid_packet_payload() -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / "valid-minimal.json").read_text(encoding="utf-8"))


def _write_conforming_packet(packet_dir: Path, *, payload: dict[str, object] | None = None) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    resolved_payload = payload or _valid_packet_payload()
    receipt_ref = str(resolved_payload["receipts"][0]["ref"])  # type: ignore[index]
    (packet_dir / "PROMPT.md").write_text("# prompt\n", encoding="utf-8")
    (packet_dir / "packet.json").write_text(
        json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stanza_lines = "\n".join(f"{field}: value-{field}" for field in PROVENANCE_STANZA_FIELDS)
    (packet_dir / "finding.yaml").write_text(f"{stanza_lines}\nclaim: a claim\n", encoding="utf-8")
    report_sections = "\n".join(f"## {section}\ntext\n" for section in REPORT_SECTION_ORDER)
    (packet_dir / "report.md").write_text(report_sections, encoding="utf-8")
    (packet_dir / "evidence.ndjson").write_text(
        json.dumps({"ref": receipt_ref, "evidence": "fixture receipt"}) + "\n",
        encoding="utf-8",
    )
    (packet_dir / "queries.ndjson").write_text('{"text": "find"}\n', encoding="utf-8")
    (packet_dir / "checks.json").write_text(
        json.dumps({"pass": True, "unsupported_claims": [], "coverage_notes": "complete fixture"}),
        encoding="utf-8",
    )
    (packet_dir / "NON-CLAIMS.md").write_text(
        "# Non-claims\n\nThis fixture does not establish archive behavior.\n",
        encoding="utf-8",
    )
    (packet_dir / "run.log").write_text("$ polylogue find\n", encoding="utf-8")
    (packet_dir / "receipt.json").write_text(
        json.dumps({"ref": receipt_ref, "status": "resolved"}) + "\n",
        encoding="utf-8",
    )


def test_validate_packet_accepts_a_conforming_packet(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)

    result = validate_packet(packet_dir)

    assert result.ok is True
    assert result.missing_files == ()
    assert result.missing_stanza_fields == ()
    assert result.malformed_sections == ()
    assert result.schema_errors == ()
    assert result.receipt_errors == ()
    assert result.errors == ()


@pytest.mark.parametrize(
    ("fixture_name", "pointer"),
    [
        ("invalid-missing-falsifier.json", "packet.json/: 'falsifier' is a required property"),
        ("invalid-missing-controls.json", "packet.json/: 'controls' is a required property"),
    ],
)
def test_validate_packet_rejects_missing_epistemic_contract_fields(
    tmp_path: Path,
    fixture_name: str,
    pointer: str,
) -> None:
    packet_dir = tmp_path / "packet"
    payload = json.loads((FIXTURE_ROOT / fixture_name).read_text(encoding="utf-8"))
    _write_conforming_packet(packet_dir, payload=payload)

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert pointer in result.schema_errors


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


def test_validate_packet_rejects_a_missing_receipt_artifact(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)
    (packet_dir / "receipt.json").unlink()

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert any("artifact missing" in error for error in result.receipt_errors)


def test_validate_packet_rejects_receipt_path_escape(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    payload = _valid_packet_payload()
    payload["receipts"][0]["artifact_path"] = "../outside.json"  # type: ignore[index]
    _write_conforming_packet(packet_dir, payload=payload)

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert any("escapes packet directory" in error for error in result.receipt_errors)


def test_validate_packet_rejects_unresolved_receipt_text(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    _write_conforming_packet(packet_dir)
    (packet_dir / "receipt.json").write_text('{"ref": "artifact:other"}\n', encoding="utf-8")

    result = validate_packet(packet_dir)

    assert result.ok is False
    assert any("is not present" in error for error in result.receipt_errors)


def test_all_packet_filenames_are_checked(tmp_path: Path) -> None:
    packet_dir = tmp_path / "empty"
    packet_dir.mkdir()
    result = validate_packet(packet_dir)
    assert set(result.missing_files) == set(PACKET_FILENAMES)


def _write_registry(registry_path: Path, entries: list[dict[str, object]]) -> None:
    registry_path.write_text(json.dumps(entries), encoding="utf-8")


def _registry_entry(packet_dir: Path, *, slug: str = "stub") -> dict[str, object]:
    return {
        "slug": slug,
        "prompt_path": str(packet_dir / "PROMPT.md"),
        "packet_dir": str(packet_dir),
        "mode": "fixture",
        "required_primitives": [],
    }


def test_registry_lint_passes_for_conforming_registry(tmp_path: Path) -> None:
    packet_dir = tmp_path / "demos" / "stub"
    _write_conforming_packet(packet_dir)
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, [_registry_entry(packet_dir)])

    result = lint_demo_registry(registry_path, repo_root=tmp_path)

    assert result.ok is True


def test_registry_lint_rejects_unregistered_v2_packet(tmp_path: Path) -> None:
    registered = tmp_path / ".agent" / "demos" / "registered"
    unregistered = tmp_path / ".agent" / "demos" / "unregistered"
    _write_conforming_packet(registered)
    _write_conforming_packet(unregistered)
    registry_path = tmp_path / ".agent" / "demos" / "registry.json"
    _write_registry(registry_path, [_registry_entry(registered)])

    result = lint_demo_registry(registry_path, repo_root=tmp_path)

    assert result.ok is False
    assert result.unregistered_packet_dirs == (".agent/demos/unregistered",)


def test_registry_lint_rejects_duplicate_slugs(tmp_path: Path) -> None:
    packet_a = tmp_path / "a"
    packet_b = tmp_path / "b"
    _write_conforming_packet(packet_a)
    _write_conforming_packet(packet_b)
    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        [_registry_entry(packet_a, slug="same"), _registry_entry(packet_b, slug="same")],
    )

    result = lint_demo_registry(registry_path, repo_root=tmp_path)

    assert result.ok is False
    assert "duplicate registry slug: same" in result.registry_errors


def test_registry_cli_catches_a_missing_packet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
    assert payload["registry_errors"] == ["ghost-demo: prompt missing at nope/PROMPT.md"]


def test_registry_lint_reports_missing_registry_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_path = tmp_path / "does-not-exist.json"

    exit_code = verify_demo_packet_registry.main(["--registry", str(missing_path), "--json"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
PY
python -m py_compile tests/unit/devtools/test_demo_packet.py


---

