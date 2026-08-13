from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from devtools import evidence_dashboard


def test_static_gates_read_shared_verify_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history = tmp_path / "xdg-state" / "polylogue" / "devtools" / "verify-history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-12T00:00:00+00:00",
                "checkout_root": str(tmp_path.resolve()),
                "git_head": "current-head",
                "steps": [{"name": "ruff check", "duration_s": 1.0, "exit": 0}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "timestamp": "2026-08-12T00:01:00+00:00",
                "checkout_root": str((tmp_path / "other-worktree").resolve()),
                "git_head": "other-head",
                "steps": [{"name": "mypy", "duration_s": 1.0, "exit": 0}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(evidence_dashboard, "VERIFY_HISTORY_PATH", history)
    monkeypatch.setattr(evidence_dashboard, "git_head", lambda _root: "current-head")

    gates = evidence_dashboard._static_gates(tmp_path, now=datetime(2026, 8, 12, tzinfo=timezone.utc))

    ruff = next(gate for gate in gates["gates"] if gate["name"] == "ruff check")
    assert gates["history_path"] == str(history)
    assert ruff["status"] == "ok"
    mypy = next(gate for gate in gates["gates"] if gate["name"] == "mypy")
    assert mypy["available"] is False
