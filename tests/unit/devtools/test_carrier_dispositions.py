from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from devtools import bd_reimport_guard as guard
from devtools import carrier_dispositions, merge_boundary, pr_scope


def _carrier() -> str:
    payload: dict[str, Any] = {
        "version": 2,
        "scope_kind": "bead",
        "assigned_beads": ["polylogue-a"],
        "mutated_beads": [],
        "dispositions": [
            {
                "bead_id": "polylogue-a",
                "disposition": "satisfied",
                "evidence": [{"kind": "test", "ref": "tests/unit/devtools/test_carrier_dispositions.py"}],
                "successors": [],
            }
        ],
    }
    payload["scope_digest"] = pr_scope.carrier_digest(payload)
    return pr_scope.render_carrier(payload)


def _scope() -> pr_scope.ScopeVerdict:
    return pr_scope.ScopeVerdict(
        ok=True,
        scope_digest="scope",
        beads_digest="beads",
        assigned_beads=["polylogue-a"],
        carrier_version=2,
        scope_kind=pr_scope.ScopeKind.BEAD,
        dispositions=(
            pr_scope.ValidatedDisposition(
                "polylogue-a",
                pr_scope.ScopeDisposition.SATISFIED,
                (pr_scope.EvidenceRef(pr_scope.EvidenceKind.TEST, "tests/unit/devtools/test_carrier_dispositions.py"),),
                (),
            ),
        ),
    )


def test_workspace_command_is_registered_and_forwards_to_production_main() -> None:
    from devtools import click_dispatch

    assert click_dispatch._dispatch(["workspace", "carrier-dispositions", "--inner-help"]) == 0


def test_merged_scope_refuses_a_pr_without_its_exact_head_ledger_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    info = {
        "state": "MERGED",
        "headRefOid": "h" * 40,
        "baseRefOid": "b" * 40,
        "mergeCommit": {"oid": "m" * 40},
        "body": _carrier(),
        "isDraft": False,
    }
    monkeypatch.setattr(carrier_dispositions, "_gh_json", lambda _args: info)
    monkeypatch.setattr(pr_scope, "_ensure_local_commit", lambda _revision: None)
    monkeypatch.setattr(
        carrier_dispositions, "_git_show", lambda *_args, **_kwargs: '{"id":"polylogue-a","status":"open"}\n'
    )
    monkeypatch.setattr(
        pr_scope, "_bead_records_at", lambda _revision: {"polylogue-a": {"id": "polylogue-a", "status": "open"}}
    )
    monkeypatch.setattr(pr_scope, "changed_bead_ids", lambda **_kwargs: [])
    monkeypatch.setattr(merge_boundary, "_read_ledger", lambda: {"merges": []})

    with pytest.raises(carrier_dispositions.DispositionError, match="exact-head carrier attestation"):
        carrier_dispositions._scope_from_merged_pr(42, cwd=tmp_path)


def test_apply_uses_one_guarded_batch_and_writes_only_typed_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base.jsonl"
    output = tmp_path / ".beads" / "issues.jsonl"
    before = {"polylogue-a": {"id": "polylogue-a", "status": "open"}}
    after = {"polylogue-a": {"id": "polylogue-a", "status": "closed", "close_reason": "marker"}}
    base.write_text(json.dumps(before["polylogue-a"]) + "\n")
    monkeypatch.setattr(
        carrier_dispositions,
        "_scope_from_merged_pr",
        lambda _pr, **_kwargs: ({}, "h" * 40, "m" * 40, _scope(), "a" * 64),
    )
    monkeypatch.setattr(guard, "_export_live_state", lambda **_kwargs: before)
    monkeypatch.setattr(
        carrier_dispositions, "_validate_live_plan", lambda *_args, **_kwargs: ["close polylogue-a marker"]
    )
    calls: list[list[str]] = []

    def batch(**kwargs: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        calls.append(kwargs["prepare"](before))
        return before, after

    monkeypatch.setattr(guard, "run_guarded_batch", batch)

    assert carrier_dispositions.cmd_apply(42, base_export=base, output=output, dry_run=False) == 0
    assert calls == [["close polylogue-a marker"]]
    assert guard.parse_and_validate_jsonl(output.read_text()) == after
    receipt = next((output.parent / "disposition-receipts").glob("*.json"))
    assert json.loads(receipt.read_text())["changed_beads"] == ["polylogue-a"]


def test_apply_refuses_to_sweep_an_unrelated_live_bead_into_follow_on_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base.jsonl"
    output = tmp_path / ".beads" / "issues.jsonl"
    before = {
        "polylogue-a": {"id": "polylogue-a", "status": "open"},
        "polylogue-unrelated": {"id": "polylogue-unrelated", "status": "open"},
    }
    after = {
        "polylogue-a": {"id": "polylogue-a", "status": "closed"},
        "polylogue-unrelated": {"id": "polylogue-unrelated", "status": "deferred"},
    }
    base.write_text("".join(json.dumps(row) + "\n" for row in before.values()))
    monkeypatch.setattr(
        carrier_dispositions,
        "_scope_from_merged_pr",
        lambda _pr, **_kwargs: ({}, "h" * 40, "m" * 40, _scope(), "a" * 64),
    )
    monkeypatch.setattr(guard, "_export_live_state", lambda **_kwargs: before)
    monkeypatch.setattr(
        carrier_dispositions, "_validate_live_plan", lambda *_args, **_kwargs: ["close polylogue-a marker"]
    )
    monkeypatch.setattr(guard, "run_guarded_batch", lambda *_args, **_kwargs: (before, after))

    assert carrier_dispositions.cmd_apply(42, base_export=base, output=output, dry_run=False) == 1
    assert not output.exists()


def test_live_plan_rejects_closed_successor_and_closed_registry_waiver(monkeypatch: pytest.MonkeyPatch) -> None:
    residual = pr_scope.ScopeVerdict(
        ok=True,
        assigned_beads=["polylogue-a"],
        scope_kind=pr_scope.ScopeKind.BEAD,
        dispositions=(
            pr_scope.ValidatedDisposition("polylogue-a", pr_scope.ScopeDisposition.PARTIAL, (), ("polylogue-next",)),
        ),
    )
    live = {
        "polylogue-a": {"id": "polylogue-a", "status": "open"},
        "polylogue-next": {"id": "polylogue-next", "status": "closed"},
    }
    monkeypatch.setattr(
        "polylogue.maintenance.archive_verification.validate_archive_verification_registry", lambda **_kwargs: None
    )

    with pytest.raises(carrier_dispositions.DispositionError, match="missing or closed"):
        carrier_dispositions._validate_live_plan(residual, live, marker="marker")


def test_guarded_batch_proves_rollback_when_the_real_batch_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshots = iter(
        [
            {"polylogue-a": {"id": "polylogue-a", "status": "open"}},
            {"polylogue-a": {"id": "polylogue-a", "status": "open"}},
        ]
    )
    monkeypatch.setattr(guard, "_validated_real_bd_path", lambda _value: Path("/real/bd"))
    monkeypatch.setattr(guard, "_export_live_state", lambda **_kwargs: next(snapshots))
    monkeypatch.setattr(guard, "_acquire_invocation_lock", lambda _cwd: _NullLock())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: MagicMock(returncode=1, stdout="", stderr="second operation rejected"),
    )

    with pytest.raises(guard.BatchExecutionError, match="rolled back"):
        guard.run_guarded_batch(
            expected_ids={"polylogue-a"},
            prepare=lambda _before: ["close polylogue-a reason"],
            cwd=tmp_path,
        )


class _NullLock:
    def __enter__(self) -> _NullLock:
        return self

    def __exit__(self, *_args: object) -> None:
        return None
