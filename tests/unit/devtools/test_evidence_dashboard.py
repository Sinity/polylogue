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
                "worktree_fingerprint": "current-fingerprint",
                "final_worktree_fingerprint": "current-fingerprint",
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
    monkeypatch.setattr(evidence_dashboard, "git_dirty", lambda _root: False)
    monkeypatch.setattr(evidence_dashboard, "_worktree_fingerprint", lambda _root: "current-fingerprint")

    gates = evidence_dashboard._static_gates(tmp_path, now=datetime(2026, 8, 12, tzinfo=timezone.utc))

    ruff = next(gate for gate in gates["gates"] if gate["name"] == "ruff check")
    assert gates["history_path"] == str(history)
    assert ruff["status"] == "ok"
    mypy = next(gate for gate in gates["gates"] if gate["name"] == "mypy")
    assert mypy["available"] is False


def test_static_gates_accept_exactly_bound_last_verify_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result_path = tmp_path / evidence_dashboard.LAST_VERIFY_RESULT_REL
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "result": {
                    "timestamp": "2026-08-12T00:00:00+00:00",
                    "checkout_root": str(tmp_path.resolve()),
                    "git_head": "current-head",
                    "worktree_fingerprint": "current-fingerprint",
                    "final_worktree_fingerprint": "current-fingerprint",
                    "steps": [{"name": "ruff check", "duration_s": 1.0, "exit": 0}],
                }
            }
        )
    )
    monkeypatch.setattr(evidence_dashboard, "VERIFY_HISTORY_PATH", tmp_path / "history.jsonl")
    monkeypatch.setattr(evidence_dashboard, "git_head", lambda _root: "current-head")
    monkeypatch.setattr(evidence_dashboard, "git_dirty", lambda _root: False)
    monkeypatch.setattr(evidence_dashboard, "_worktree_fingerprint", lambda _root: "current-fingerprint")

    gates = evidence_dashboard._static_gates(tmp_path, now=datetime(2026, 8, 12, tzinfo=timezone.utc))

    assert gates["available"] is True
    assert next(gate for gate in gates["gates"] if gate["name"] == "ruff check")["status"] == "ok"


def test_static_gates_withhold_evidence_for_dirty_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps(
            {
                "checkout_root": str(tmp_path.resolve()),
                "git_head": "current-head",
                "worktree_fingerprint": "current-fingerprint",
                "steps": [{"name": "ruff check", "exit": 0}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(evidence_dashboard, "VERIFY_HISTORY_PATH", history)
    monkeypatch.setattr(evidence_dashboard, "git_head", lambda _root: "current-head")
    monkeypatch.setattr(evidence_dashboard, "git_dirty", lambda _root: True)

    gates = evidence_dashboard._static_gates(tmp_path, now=datetime(2026, 8, 12, tzinfo=timezone.utc))

    assert gates["available"] is False
    assert all(gate["reason"] == "checkout has uncommitted changes" for gate in gates["gates"])


@pytest.mark.parametrize(
    ("checkout_head", "fingerprint"),
    [(None, "unavailable"), ("current-head", "unavailable")],
)
def test_static_gates_withhold_evidence_when_git_identity_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkout_head: str | None,
    fingerprint: str,
) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps(
            {
                "checkout_root": str(tmp_path.resolve()),
                "git_head": checkout_head,
                "worktree_fingerprint": fingerprint,
                "steps": [{"name": "ruff check", "exit": 0}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(evidence_dashboard, "VERIFY_HISTORY_PATH", history)
    monkeypatch.setattr(evidence_dashboard, "git_head", lambda _root: checkout_head)
    monkeypatch.setattr(evidence_dashboard, "git_dirty", lambda _root: False)
    monkeypatch.setattr(evidence_dashboard, "_worktree_fingerprint", lambda _root: fingerprint)

    gates = evidence_dashboard._static_gates(tmp_path, now=datetime(2026, 8, 12, tzinfo=timezone.utc))

    assert gates["available"] is False
    assert all(gate["reason"] == "checkout Git identity is unavailable" for gate in gates["gates"])


def test_static_gates_reject_wrong_checkout_fingerprint_and_legacy_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    history = tmp_path / "history.jsonl"
    legacy_result = tmp_path / evidence_dashboard.LAST_VERIFY_RESULT_REL
    legacy_result.parent.mkdir(parents=True)
    legacy_result.write_text(json.dumps({"result": {"steps": [{"name": "ruff check", "exit": 0}]}}))
    history.write_text(
        "\n".join(
            json.dumps(entry)
            for entry in (
                {
                    "checkout_root": str(tmp_path / "other"),
                    "git_head": "current-head",
                    "worktree_fingerprint": "current-fingerprint",
                    "steps": [{"name": "ruff check", "exit": 0}],
                },
                {
                    "checkout_root": str(tmp_path.resolve()),
                    "git_head": "current-head",
                    "worktree_fingerprint": "other-fingerprint",
                    "steps": [{"name": "mypy", "exit": 0}],
                },
                {
                    "checkout_root": str(tmp_path.resolve()),
                    "git_head": "current-head",
                    "steps": [{"name": "render all", "exit": 0}],
                },
            )
        )
        + "\n"
    )
    monkeypatch.setattr(evidence_dashboard, "VERIFY_HISTORY_PATH", history)
    monkeypatch.setattr(evidence_dashboard, "git_head", lambda _root: "current-head")
    monkeypatch.setattr(evidence_dashboard, "git_dirty", lambda _root: False)
    monkeypatch.setattr(evidence_dashboard, "_worktree_fingerprint", lambda _root: "current-fingerprint")

    gates = evidence_dashboard._static_gates(tmp_path, now=datetime(2026, 8, 12, tzinfo=timezone.utc))

    assert gates["available"] is False
    assert all(gate["available"] is False for gate in gates["gates"])


def test_static_gates_reject_a_run_whose_checkout_changed_mid_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps(
            {
                "checkout_root": str(tmp_path.resolve()),
                "git_head": "current-head",
                "worktree_fingerprint": "current-fingerprint",
                "final_worktree_fingerprint": "changed-during-run",
                "steps": [{"name": "ruff check", "exit": 0}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(evidence_dashboard, "VERIFY_HISTORY_PATH", history)
    monkeypatch.setattr(evidence_dashboard, "git_head", lambda _root: "current-head")
    monkeypatch.setattr(evidence_dashboard, "git_dirty", lambda _root: False)
    monkeypatch.setattr(evidence_dashboard, "_worktree_fingerprint", lambda _root: "current-fingerprint")

    gates = evidence_dashboard._static_gates(tmp_path, now=datetime(2026, 8, 12, tzinfo=timezone.utc))

    assert gates["available"] is False
    assert all(gate["available"] is False for gate in gates["gates"])
