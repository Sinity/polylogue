from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from devtools import cli_surface_audit


def _archive(root: Path) -> Path:
    root.mkdir()
    conn = sqlite3.connect(root / "index.db")
    try:
        conn.executescript(
            """
            PRAGMA user_version = 18;
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY);
            CREATE TABLE messages (message_id TEXT PRIMARY KEY);
            INSERT INTO sessions VALUES ('s1'), ('s2');
            INSERT INTO messages VALUES ('m1'), ('m2'), ('m3');
            """
        )
        conn.commit()
    finally:
        conn.close()
    return root


def test_cli_surface_audit_prunes_stale_unbounded_output_by_default(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    archive_root = _archive(tmp_path / "archive")
    out_dir = tmp_path / "audit"
    stale = out_dir / "outputs" / "read_dialogue_json.stdout"
    stale.parent.mkdir(parents=True)
    stale.write_text("large stale payload", encoding="utf-8")

    def fake_run(argv: tuple[str, ...], **_: Any) -> subprocess.CompletedProcess[str]:
        stdout = json.dumps(
            {
                "argv": list(argv),
                "large": ["x" * 2000 for _ in range(5)] if "status" in argv else [],
                "small": {"ok": True},
            }
        )
        return subprocess.CompletedProcess(args=" ".join(argv), returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = cli_surface_audit.run_audit(
        out_dir=out_dir,
        archive_root=archive_root,
        include_unbounded_dialogue=False,
        timeout=1,
    )

    names = {record["name"] for record in payload["commands"]}
    assert "read_dialogue_bounded_json" in names
    assert "read_dialogue_unbounded_json" not in names
    assert not stale.exists()
    matrix = json.loads((out_dir / "command-matrix.json").read_text(encoding="utf-8"))
    assert matrix["archive"] == {"messages": 3, "schema_version": 18, "sessions": 2}
    status_record = next(record for record in matrix["commands"] if record["name"] == "status_plain")
    assert status_record["json_top_level_bytes"]["large"] > status_record["json_top_level_bytes"]["small"]
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "current demo shelf should prefer bounded dialogue" in readme
    assert "## Large JSON Payloads" in readme
    assert "`status_plain`" in readme


def test_cli_surface_audit_can_include_unbounded_diagnostic(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    archive_root = _archive(tmp_path / "archive")

    def fake_run(argv: tuple[str, ...], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=" ".join(argv), returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = cli_surface_audit.run_audit(
        out_dir=tmp_path / "audit",
        archive_root=archive_root,
        include_unbounded_dialogue=True,
        timeout=1,
    )

    names = {record["name"] for record in payload["commands"]}
    assert "read_dialogue_unbounded_json" in names
