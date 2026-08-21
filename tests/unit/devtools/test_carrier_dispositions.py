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
    monkeypatch.setattr(
        merge_boundary,
        "_read_ledger",
        lambda: {
            "merges": [{"pr": 42, "head_sha": "h" * 40, "base_sha": "b" * 40, "scope_attestation_digest": "a" * 64}]
        },
    )

    with pytest.raises(carrier_dispositions.DispositionError, match="exact-head carrier attestation"):
        carrier_dispositions._scope_from_merged_pr(42, cwd=tmp_path)


def test_recovery_derives_and_records_an_exact_head_attestation_for_an_unledgered_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    info = {
        "state": "MERGED",
        "headRefOid": "h" * 40,
        "mergeCommit": {"oid": "m" * 40},
        "body": _carrier(),
        "isDraft": False,
        "title": "fix: recover carrier dispositions",
    }
    monkeypatch.setattr(carrier_dispositions, "_gh_json", lambda _args: info)
    monkeypatch.setattr(pr_scope, "_ensure_local_commit", lambda _revision: None)
    monkeypatch.setattr(merge_boundary, "_read_ledger", lambda: {"merges": []})
    monkeypatch.setattr(carrier_dispositions, "_squash_merge_base", lambda *_args, **_kwargs: "b" * 40)
    monkeypatch.setattr(carrier_dispositions, "_validate_merged_scope", lambda *_args, **_kwargs: (_scope(), "a" * 64))
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(
        merge_boundary,
        "record_post_merge_carrier_recovery",
        lambda *args, **kwargs: recorded.append({"args": args, **kwargs}),
    )

    assert carrier_dispositions._scope_from_merged_pr(42, cwd=tmp_path, recover_unledgered_merge=True)[-1] == "a" * 64
    assert recorded == [
        {
            "args": (42, "h" * 40, "fix: recover carrier dispositions"),
            "base_sha": "b" * 40,
            "merge_sha": "m" * 40,
            "scope_attestation_digest": "a" * 64,
        }
    ]


def test_recovery_dry_run_does_not_publish_a_ledger_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = {
        "state": "MERGED",
        "headRefOid": "h" * 40,
        "mergeCommit": {"oid": "m" * 40},
        "body": _carrier(),
        "isDraft": False,
        "title": "fix: recover carrier dispositions",
    }
    monkeypatch.setattr(carrier_dispositions, "_gh_json", lambda _args: info)
    monkeypatch.setattr(pr_scope, "_ensure_local_commit", lambda _revision: None)
    monkeypatch.setattr(merge_boundary, "_read_ledger", lambda: {"merges": []})
    monkeypatch.setattr(carrier_dispositions, "_squash_merge_base", lambda *_args, **_kwargs: "b" * 40)
    monkeypatch.setattr(carrier_dispositions, "_validate_merged_scope", lambda *_args, **_kwargs: (_scope(), "a" * 64))
    monkeypatch.setattr(
        merge_boundary,
        "record_post_merge_carrier_recovery",
        lambda *_args, **_kwargs: pytest.fail("dry-run recovery must not write a ledger entry"),
    )

    assert (
        carrier_dispositions._scope_from_merged_pr(
            42, cwd=tmp_path, recover_unledgered_merge=True, persist_recovery=False
        )[-1]
        == "a" * 64
    )


def test_recover_subcommand_forwards_the_explicit_recovery_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def apply(*args: object, **kwargs: object) -> int:
        calls.append({"args": args, **kwargs})
        return 0

    monkeypatch.setattr(carrier_dispositions, "cmd_apply", apply)

    assert (
        carrier_dispositions.main(
            ["recover", "42", "--base-export", str(tmp_path / "base.jsonl"), "--output", str(tmp_path / "output.jsonl")]
        )
        == 0
    )
    assert calls[0]["args"] == (42,)
    assert calls[0]["recover_unledgered_merge"] is True


def test_merged_scope_uses_the_recorded_merge_time_base_for_its_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    merge_base = "c" * 40
    info = {
        "state": "MERGED",
        "headRefOid": "h" * 40,
        "baseRefOid": "n" * 40,
        "mergeCommit": {"oid": "m" * 40},
        "body": _carrier(),
        "isDraft": False,
    }
    expected = pr_scope.attestation_payload(_scope(), head_sha="h" * 40, base_sha=merge_base)["attestation_digest"]
    monkeypatch.setattr(carrier_dispositions, "_gh_json", lambda _args: info)
    monkeypatch.setattr(pr_scope, "_ensure_local_commit", lambda _revision: None)
    monkeypatch.setattr(
        carrier_dispositions, "_git_show", lambda *_args, **_kwargs: '{"id":"polylogue-a","status":"open"}\n'
    )
    observed_bases: list[str] = []

    def validate(*_args: object, **kwargs: object) -> pr_scope.ScopeVerdict:
        observed_bases.append(str(kwargs["base_sha"]))
        return _scope()

    monkeypatch.setattr(pr_scope, "validate_pr_body", validate)
    monkeypatch.setattr(
        merge_boundary,
        "_read_ledger",
        lambda: {
            "merges": [{"pr": 42, "head_sha": "h" * 40, "base_sha": merge_base, "scope_attestation_digest": expected}]
        },
    )

    assert carrier_dispositions._scope_from_merged_pr(42, cwd=tmp_path)[-1] == expected
    assert observed_bases == [merge_base]


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
        lambda _pr, **_kwargs: ({}, "h" * 40, "m" * 40, "b" * 40, _scope(), "a" * 64),
    )
    monkeypatch.setattr(guard, "_validated_real_bd_path", lambda _value: Path("/real/bd"))
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
    receipt = json.loads(next((output.parent / "disposition-receipts").glob("*.json")).read_text())
    assert receipt["version"] == 2
    assert receipt["base_sha"] == "b" * 40
    assert receipt["changed_beads"] == ["polylogue-a"]
    assert receipt["changed_records"] == after


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
        lambda _pr, **_kwargs: ({}, "h" * 40, "m" * 40, "b" * 40, _scope(), "a" * 64),
    )
    monkeypatch.setattr(guard, "_validated_real_bd_path", lambda _value: Path("/real/bd"))
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
    live: dict[str, dict[str, Any]] = {
        "polylogue-a": {"id": "polylogue-a", "status": "open"},
        "polylogue-next": {"id": "polylogue-next", "status": "closed"},
    }
    monkeypatch.setattr(
        "polylogue.maintenance.archive_verification.validate_archive_verification_registry", lambda **_kwargs: None
    )

    with pytest.raises(carrier_dispositions.DispositionError, match="missing or closed"):
        carrier_dispositions._validate_live_plan(residual, live, marker="marker")


def test_live_plan_rejects_a_successor_closed_by_the_same_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    scope = pr_scope.ScopeVerdict(
        ok=True,
        assigned_beads=["polylogue-a", "polylogue-next"],
        scope_kind=pr_scope.ScopeKind.BEAD,
        dispositions=(
            pr_scope.ValidatedDisposition("polylogue-a", pr_scope.ScopeDisposition.PARTIAL, (), ("polylogue-next",)),
            pr_scope.ValidatedDisposition("polylogue-next", pr_scope.ScopeDisposition.SATISFIED, (), ()),
        ),
    )
    live: dict[str, dict[str, Any]] = {
        "polylogue-a": {
            "id": "polylogue-a",
            "status": "open",
            "dependencies": [{"type": "relates-to", "depends_on_id": "polylogue-next"}],
        },
        "polylogue-next": {"id": "polylogue-next", "status": "open"},
    }
    monkeypatch.setattr(
        "polylogue.maintenance.archive_verification.validate_archive_verification_registry", lambda **_kwargs: None
    )

    with pytest.raises(carrier_dispositions.DispositionError, match="projected Beads"):
        carrier_dispositions._validate_live_plan(scope, live, marker="marker")


def test_dry_run_uses_the_real_bd_binary_and_rejects_unrelated_export_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base.jsonl"
    output = tmp_path / ".beads" / "issues.jsonl"
    base.write_text('{"id":"polylogue-a","status":"open"}\n{"id":"polylogue-other","status":"open"}\n')
    live = {
        "polylogue-a": {"id": "polylogue-a", "status": "open"},
        "polylogue-other": {"id": "polylogue-other", "status": "deferred"},
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        carrier_dispositions,
        "_scope_from_merged_pr",
        lambda _pr, **_kwargs: ({}, "h" * 40, "m" * 40, "b" * 40, _scope(), "a" * 64),
    )
    monkeypatch.setattr(guard, "_validated_real_bd_path", lambda _value: Path("/real/bd"))

    def export(**kwargs: object) -> dict[str, dict[str, Any]]:
        captured.update(kwargs)
        return live

    monkeypatch.setattr(guard, "_export_live_state", export)

    assert carrier_dispositions.cmd_apply(42, base_export=base, output=output, dry_run=True) == 1
    assert captured["bd_command"] == "/real/bd"
    assert isinstance(captured["env"], dict) and captured["env"]["BD_IMPORT_AUTO"] == "false"


def test_retry_preserves_the_first_immutable_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "base.jsonl"
    output = tmp_path / ".beads" / "issues.jsonl"
    before = {"polylogue-a": {"id": "polylogue-a", "status": "open"}}
    after = {"polylogue-a": {"id": "polylogue-a", "status": "closed", "close_reason": "marker"}}
    base.write_text(json.dumps(before["polylogue-a"]) + "\n")
    monkeypatch.setattr(
        carrier_dispositions,
        "_scope_from_merged_pr",
        lambda _pr, **_kwargs: ({}, "h" * 40, "m" * 40, "b" * 40, _scope(), "a" * 64),
    )
    monkeypatch.setattr(guard, "_validated_real_bd_path", lambda _value: Path("/real/bd"))
    monkeypatch.setattr(guard, "_export_live_state", lambda **_kwargs: before)
    monkeypatch.setattr(
        carrier_dispositions, "_validate_live_plan", lambda *_args, **_kwargs: ["close polylogue-a marker"]
    )
    monkeypatch.setattr(guard, "run_guarded_batch", lambda **_kwargs: (before, after))

    assert carrier_dispositions.cmd_apply(42, base_export=base, output=output, dry_run=False) == 0
    receipt_path = next((output.parent / "disposition-receipts").glob("*.json"))
    first = json.loads(receipt_path.read_text())
    base.write_text(json.dumps(after["polylogue-a"]) + "\n")
    monkeypatch.setattr(guard, "_export_live_state", lambda **_kwargs: after)
    monkeypatch.setattr(guard, "run_guarded_batch", lambda **_kwargs: (after, after))

    assert carrier_dispositions.cmd_apply(42, base_export=base, output=output, dry_run=False) == 0
    assert json.loads(receipt_path.read_text()) == first

    after["polylogue-a"]["title"] = "edited after receipt"

    assert carrier_dispositions.cmd_apply(42, base_export=base, output=output, dry_run=False) == 1
    assert json.loads(receipt_path.read_text()) == first


def test_existing_legacy_receipt_remains_valid_for_a_receipt_v2_execution(tmp_path: Path) -> None:
    identity = carrier_dispositions._receipt_identity(
        execution_id="e" * 64,
        pr=42,
        head_sha="h" * 40,
        merge_sha="m" * 40,
        base_sha="b" * 40,
        attestation="a" * 64,
        marker="carrier-disposition:v1:pr=42:merge=" + "m" * 40 + ":execution=" + "e" * 64,
        scope=_scope(),
    )
    legacy = {key: value for key, value in identity.items() if key != "base_sha"} | {"version": 1, "changed_beads": []}
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(legacy))

    assert carrier_dispositions._existing_receipt(receipt_path, identity) == legacy


def test_interruption_after_export_keeps_the_original_mutation_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An abrupt post-export interruption cannot erase the guarded batch's evidence."""
    monkeypatch.chdir(tmp_path)
    output = tmp_path / ".beads" / "issues.jsonl"
    before = {"polylogue-a": {"id": "polylogue-a", "status": "open"}}
    after = {"polylogue-a": {"id": "polylogue-a", "status": "closed", "close_reason": "marker"}}
    output.parent.mkdir()
    output.write_text(json.dumps(before["polylogue-a"]) + "\n")
    monkeypatch.setattr(
        carrier_dispositions,
        "_scope_from_merged_pr",
        lambda _pr, **_kwargs: ({}, "h" * 40, "m" * 40, "b" * 40, _scope(), "a" * 64),
    )
    monkeypatch.setattr(guard, "_validated_real_bd_path", lambda _value: Path("/real/bd"))
    monkeypatch.setattr(guard, "_export_live_state", lambda **_kwargs: after)
    monkeypatch.setattr(
        carrier_dispositions, "_validate_live_plan", lambda *_args, **_kwargs: ["close polylogue-a marker"]
    )
    monkeypatch.setattr(guard, "run_guarded_batch", lambda **_kwargs: (before, after))
    write_export = guard.atomic_write_jsonl

    def write_then_interrupt(path: Path, rows: dict[str, dict[str, Any]]) -> None:
        write_export(path, rows)
        raise KeyboardInterrupt

    monkeypatch.setattr(guard, "atomic_write_jsonl", write_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        carrier_dispositions.cmd_apply(42, base_export=output, output=output, dry_run=False)

    receipt_path = next((output.parent / "disposition-receipts").glob("*.json"))
    first = json.loads(receipt_path.read_text())
    assert first["changed_beads"] == ["polylogue-a"]
    monkeypatch.setattr(guard, "atomic_write_jsonl", write_export)
    monkeypatch.setattr(guard, "run_guarded_batch", lambda **_kwargs: (after, after))

    assert carrier_dispositions.cmd_apply(42, base_export=output, output=output, dry_run=False) == 0
    assert json.loads(receipt_path.read_text()) == first


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


def test_guarded_batch_timeout_inspects_postimage_before_failing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch timeout may follow a committed transaction, so its postimage is load-bearing."""
    before = {"polylogue-a": {"id": "polylogue-a", "status": "open"}}
    after = {"polylogue-a": {"id": "polylogue-a", "status": "closed"}}
    snapshots = iter([before, after])
    monkeypatch.setattr(guard, "_validated_real_bd_path", lambda _value: Path("/real/bd"))
    monkeypatch.setattr(guard, "_export_live_state", lambda **_kwargs: next(snapshots))
    monkeypatch.setattr(guard, "_acquire_invocation_lock", lambda _cwd: _NullLock())

    def timeout(*_args: object, **_kwargs: object) -> MagicMock:
        raise subprocess.TimeoutExpired(["/real/bd", "batch", "--json"], 60)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(guard.BatchExecutionError, match="timed out after changing selected row"):
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
