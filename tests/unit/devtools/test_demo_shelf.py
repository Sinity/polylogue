from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import demo_shelf


def test_demo_shelf_writes_manifest_and_summary_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "demos"
    (root / "01-demo").mkdir(parents=True)
    (root / "01-demo" / "README.md").write_text("# Demo\n\nReadable.\n")
    (root / "01-demo" / "payload.bin").write_bytes(b"\x00\x01")
    (root / "01-demo" / "summary.json").write_text(
        json.dumps(
            {
                "artifact": "01-demo",
                "claim": "Readable artifact exists.",
                "non_claim": "Binary payload is not included in generated indexes.",
                "proofs": ["README.md included"],
                "caveats": ["Synthetic fixture"],
                "index_schema_version": 18,
            }
        )
    )

    exit_code = demo_shelf.main(["--root", str(root)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "demo shelf refreshed" in output
    manifest = json.loads((root / "MANIFEST.readable.json").read_text())
    assert manifest["contract"] == "current-curated-demo-set"
    assert "append-only" in manifest["curation_policy"]
    assert "read-package" in manifest["packaging"]
    assert manifest["file_count"] == 3
    assert manifest["readable_count"] == 2
    paths = {item["path"]: item for item in manifest["files"]}
    assert paths["01-demo/README.md"]["readable"] is True
    assert paths["01-demo/summary.json"]["readable"] is True
    assert paths["01-demo/payload.bin"]["readable"] is False
    assert "MANIFEST.readable.json" not in paths
    assert "CONCATENATED_READABLE.md" not in paths
    assert "SUMMARY_INDEX.json" not in paths
    assert not (root / "CONCATENATED_READABLE.md").exists()
    summary_index = json.loads((root / "SUMMARY_INDEX.json").read_text())
    assert summary_index["summary_count"] == 1
    assert summary_index["coverage"]["without_claim"] == []
    assert summary_index["coverage"]["without_non_claim"] == []
    assert summary_index["records"][0]["proof_fields"] == ["proofs"]
    assert summary_index["records"][0]["caveat_fields"] == ["caveats"]


def test_demo_shelf_check_reports_drift(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n")

    assert demo_shelf.main(["--root", str(root)]) == 0
    (root / "README.md").write_text("# Demo\n\nchanged\n")

    assert demo_shelf.main(["--root", str(root), "--check"]) == 1


def test_demo_shelf_check_json_reports_current_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "demos"
    root.mkdir()
    (root / "README.md").write_text("# Demo\n")
    assert demo_shelf.main(["--root", str(root)]) == 0
    capsys.readouterr()

    exit_code = demo_shelf.main(["--root", str(root), "--check", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "check"
    assert payload["file_count"] == 1
    assert payload["readable_count"] == 1
    assert payload["summary_count"] == 0


def test_demo_shelf_summary_index_reports_coverage_gaps(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    (root / "01-demo").mkdir(parents=True)
    (root / "01-demo" / "summary.json").write_text(json.dumps({"artifact": "01-demo"}))

    assert demo_shelf.main(["--root", str(root)]) == 0

    summary_index = json.loads((root / "SUMMARY_INDEX.json").read_text())
    assert summary_index["summary_count"] == 1
    assert summary_index["coverage"]["without_claim"] == ["01-demo/summary.json"]
    assert summary_index["coverage"]["without_non_claim"] == ["01-demo/summary.json"]
    assert summary_index["coverage"]["without_proof_fields"] == ["01-demo/summary.json"]
    assert summary_index["coverage"]["without_caveat_fields"] == ["01-demo/summary.json"]


def test_demo_shelf_require_summary_coverage_fails_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "demos"
    (root / "01-demo").mkdir(parents=True)
    (root / "01-demo" / "summary.json").write_text(json.dumps({"artifact": "01-demo"}))
    assert demo_shelf.main(["--root", str(root)]) == 0
    capsys.readouterr()

    exit_code = demo_shelf.main(
        [
            "--root",
            str(root),
            "--check",
            "--json",
            "--require-summary-coverage",
            "claim,non_claim,proof_fields,caveat_fields",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["summary_coverage_failures"] == {
        "caveat_fields": ["01-demo/summary.json"],
        "claim": ["01-demo/summary.json"],
        "non_claim": ["01-demo/summary.json"],
        "proof_fields": ["01-demo/summary.json"],
    }


def test_demo_shelf_require_summary_coverage_passes_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "demos"
    (root / "01-demo").mkdir(parents=True)
    (root / "01-demo" / "summary.json").write_text(
        json.dumps(
            {
                "artifact": "01-demo",
                "claim": "Claim.",
                "non_claim": "Non-claim.",
                "proofs": ["proof"],
                "caveats": ["caveat"],
            }
        )
    )
    assert demo_shelf.main(["--root", str(root)]) == 0
    capsys.readouterr()

    exit_code = demo_shelf.main(
        [
            "--root",
            str(root),
            "--check",
            "--json",
            "--require-summary-coverage",
            "claim,non_claim,proof_fields,caveat_fields",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["summary_coverage_failures"] == {}
