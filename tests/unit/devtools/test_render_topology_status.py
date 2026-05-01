"""Tests for topology-status dashboard rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import render_topology_status


def test_dashboard_counts_targets_and_conflicts_from_targets() -> None:
    rows: list[dict[str, object]] = [
        {"owner": "lib-query", "path": "polylogue/lib/old_a.py", "target": "polylogue/lib/new/a.py"},
        {"owner": "lib-query", "path": "polylogue/lib/stable.py", "target": "polylogue/lib/stable.py"},
        {"owner": "storage-repository", "path": "polylogue/storage/old_b.py", "target": "polylogue/lib/new/a.py"},
    ]

    body = render_topology_status.render_dashboard(rows, {"polylogue/lib/new/a.py", "polylogue/lib/stable.py"})

    assert "| lib-query | query-domain primitives | 2 | 2 | 0 | 1 |" in body
    assert "| storage-repository | repository persistence adapters | 1 | 1 | 0 | 1 |" in body


def test_check_compares_full_dashboard_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "topology-status.md"
    rows: list[dict[str, object]] = [
        {"owner": "lib-query", "path": "polylogue/lib/old.py", "target": "polylogue/lib/new.py"}
    ]
    expected_body = render_topology_status.render_dashboard(rows, {"polylogue/lib/new.py"})
    expected = render_topology_status.dashboard_contents(expected_body)
    output.write_text(expected.replace("# Topology status", "# Edited title"), encoding="utf-8")

    monkeypatch.setattr(render_topology_status, "parse_topology_yaml", lambda _text: rows)
    monkeypatch.setattr(render_topology_status, "realized_paths", lambda: {"polylogue/lib/new.py"})

    projection = tmp_path / "topology-target.yaml"
    projection.write_text("files: []\n", encoding="utf-8")

    assert render_topology_status.main(["--check", "--yaml", str(projection), "--output", str(output)]) == 1
