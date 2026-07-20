from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from devtools import bead_batch_show


def test_dep_str_prefers_depends_on_id() -> None:
    assert bead_batch_show._dep_str({"depends_on_id": "polylogue-b", "type": "blocks"}) == "polylogue-b(blocks)"


def test_dep_str_falls_back_to_alternate_keys() -> None:
    assert bead_batch_show._dep_str({"to_id": "polylogue-c", "dep_type": "related"}) == "polylogue-c(related)"
    assert bead_batch_show._dep_str({}) == "?(?)"


def test_show_one_prints_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    record = {
        "id": "polylogue-kapb",
        "status": "open",
        "priority": 2,
        "issue_type": "task",
        "title": "Integrate .agent tooling",
        "description": "A" * 400,
        "dependencies": [{"depends_on_id": "polylogue-x", "type": "blocks"}],
        "notes": "B" * 400,
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(stdout=json.dumps([record]), stderr=""),
    )

    bead_batch_show._show_one("polylogue-kapb")

    out = capsys.readouterr().out
    assert "polylogue-kapb [open] P2 task" in out
    assert "polylogue-x(blocks)" in out
    assert len(out.splitlines()[1].split("DESC: ")[1]) == 280


def test_show_one_reports_missing_bead(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(stdout="", stderr="not found"),
    )

    bead_batch_show._show_one("polylogue-missing")

    out = capsys.readouterr().out
    assert "polylogue-missing MISSING/ERROR" in out


def test_main_shows_each_requested_bead(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[str] = []
    monkeypatch.setattr(bead_batch_show, "_show_one", lambda bead_id: seen.append(bead_id))

    rc = bead_batch_show.main(["polylogue-a", "polylogue-b"])

    assert rc == 0
    assert seen == ["polylogue-a", "polylogue-b"]
