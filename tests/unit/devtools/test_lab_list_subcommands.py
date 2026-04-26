"""Tests for the `lab-scenario list` and `lab-corpus list` subcommands.

Closes the discoverability gap surfaced by #446.
"""

from __future__ import annotations

import json

import pytest

from devtools import artifact_graph, lab_corpus, lab_scenario


def test_lab_scenario_list_human(capsys: pytest.CaptureFixture[str]) -> None:
    rc = lab_scenario.main(["list"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "archive-smoke" in captured.out


def test_lab_scenario_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = lab_scenario.main(["list", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert isinstance(payload, dict)
    assert "scenarios" in payload
    assert any(entry["name"] == "archive-smoke" for entry in payload["scenarios"])


def test_lab_corpus_list_human(capsys: pytest.CaptureFixture[str]) -> None:
    rc = lab_corpus.main(["list"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "default" in captured.out


def test_lab_corpus_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = lab_corpus.main(["list", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert "corpus_sources" in payload
    names = [entry["name"] for entry in payload["corpus_sources"]]
    assert "default" in names


def test_artifact_graph_strict_fails_on_uncovered(capsys: pytest.CaptureFixture[str]) -> None:
    """Without explicit --strict, the command exits 0; with --strict, it fails on uncovered targets."""
    rc_default = artifact_graph.main([])
    capsys.readouterr()
    assert rc_default == 0

    rc_strict = artifact_graph.main(["--strict"])
    capsys.readouterr()
    # The realized tree currently has uncovered maintenance targets, so --strict
    # MUST fail. If/when those are covered, this test flips and the assertion
    # should be inverted in the same PR that covers them.
    assert rc_strict == 1
