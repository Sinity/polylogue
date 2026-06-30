from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from devtools.read_package import build_read_package_plan, load_read_package_spec, main


def test_load_read_package_spec_validates_and_builds_plan(tmp_path: Path) -> None:
    spec_path = tmp_path / "package.json"
    spec_path.write_text(
        json.dumps(
            {
                "version": 1,
                "prune": ["transcript.md"],
                "artifacts": [
                    {"name": "dialogue", "view": "dialogue", "format": "markdown", "path": "dialogue.md"},
                    {
                        "name": "spec",
                        "view": "temporal,chronicle",
                        "format": "json",
                        "path": "spec.json",
                        "spec": True,
                    },
                ],
            }
        )
    )

    spec = load_read_package_spec(spec_path)
    plan = build_read_package_plan(spec, session_id="019f", out_dir=tmp_path / "out", polylogue_bin="polylogue")

    assert spec.prune == ("transcript.md",)
    assert [item.artifact.name for item in plan] == ["dialogue", "spec"]
    assert plan[1].artifact.spec is True
    assert "--spec" in plan[1].argv
    assert plan[0].argv == (
        "polylogue",
        "--id",
        "019f",
        "read",
        "--view",
        "dialogue",
        "--format",
        "markdown",
        "--to",
        "file",
        "--out",
        str(tmp_path / "out" / "dialogue.md"),
    )


def test_read_package_dry_run_emits_summary_without_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "package.json"
    spec_path.write_text(
        json.dumps(
            {
                "version": 1,
                "prune": ["stale.md"],
                "artifacts": [
                    {"name": "dialogue", "view": "dialogue", "format": "json", "path": "dialogue.json"},
                ],
            }
        )
    )

    result = main(
        [
            "--spec",
            str(spec_path),
            "--session-id",
            "019f",
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["prune"] == ["stale.md"]
    assert payload["pruned"] == []
    assert payload["artifacts"][0]["view"] == "dialogue"
    assert not (tmp_path / "out").exists()


def test_read_package_json_keeps_child_output_off_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "package.json"
    spec_path.write_text(
        json.dumps(
            {
                "version": 1,
                "artifacts": [
                    {"name": "dialogue", "view": "dialogue", "format": "json", "path": "dialogue.json"},
                ],
            }
        )
    )

    def fake_run(argv: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[tuple[str, ...]]:
        stdout = kwargs.get("stdout")
        if stdout is not None:
            stdout.write("child progress line\n")
        out_index = argv.index("--out") + 1
        Path(argv[out_index]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = main(
        [
            "--spec",
            str(spec_path),
            "--session-id",
            "019f",
            "--out-dir",
            str(tmp_path / "out"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "child progress line" not in captured.out
    assert "child progress line" in captured.err
    payload = json.loads(captured.out)
    assert payload["artifacts"][0]["bytes"] == 2


def test_read_package_json_reports_pruned_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "package.json"
    spec_path.write_text(
        json.dumps(
            {
                "version": 1,
                "prune": ["stale.md", "missing.md"],
                "artifacts": [
                    {"name": "dialogue", "view": "dialogue", "format": "json", "path": "dialogue.json"},
                ],
            }
        )
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.md").write_text("old", encoding="utf-8")

    def fake_run(argv: tuple[str, ...], **_: Any) -> subprocess.CompletedProcess[tuple[str, ...]]:
        out_index = argv.index("--out") + 1
        Path(argv[out_index]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = main(
        [
            "--spec",
            str(spec_path),
            "--session-id",
            "019f",
            "--out-dir",
            str(out_dir),
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prune"] == ["stale.md", "missing.md"]
    assert payload["pruned"] == [str(out_dir / "stale.md")]
    assert not (out_dir / "stale.md").exists()
