"""Apply a merged PR's typed carrier dispositions as an explicit Beads batch.

``workspace merge`` deliberately stops at GitHub's squash boundary.  This
command is the separate, operator-visible continuation: it verifies the
merged PR still matches the merge ledger's exact-head carrier attestation,
performs one guarded ``bd batch``, and writes only the resulting tracker
snapshot plus an immutable receipt to a follow-on PR worktree.  It never
pushes, creates a PR, or runs as an implicit merge side effect.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from devtools import bd_reimport_guard as bd_guard
from devtools import merge_boundary, pr_scope


class DispositionError(RuntimeError):
    """A post-merge carrier plan cannot be safely applied or published."""


def _gh_json(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise DispositionError(result.stderr.strip()[:300] or f"gh {' '.join(args)} failed")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise DispositionError("GitHub returned a malformed pull-request payload")
    return payload


def _git_show(revision: str, path: str, *, cwd: Path) -> str:
    result = subprocess.run(["git", "show", f"{revision}:{path}"], capture_output=True, text=True, timeout=60, cwd=cwd)
    if result.returncode != 0:
        raise DispositionError(f"cannot load {path} at PR head {revision[:8]}")
    return result.stdout


def _squash_merge_base(merge_sha: str, *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", merge_sha],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise DispositionError(f"cannot resolve merged squash commit {merge_sha[:8]} locally")
    fields = result.stdout.strip().split()
    if len(fields) != 2 or fields[0] != merge_sha:
        raise DispositionError(f"merged PR squash {merge_sha[:8]} must have exactly one parent for recovery")
    return fields[1]


def _validate_merged_scope(
    info: dict[str, Any], *, head_sha: str, base_sha: str, cwd: Path
) -> tuple[pr_scope.ScopeVerdict, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as handle:
        beads_path = Path(handle.name)
        handle.write(_git_show(head_sha, ".beads/issues.jsonl", cwd=cwd))
    try:
        scope = pr_scope.validate_pr_body(
            str(info.get("body") or ""),
            head_sha=head_sha,
            is_draft=bool(info.get("isDraft")),
            beads_path=beads_path,
            base_sha=base_sha,
        )
    finally:
        beads_path.unlink(missing_ok=True)
    if not scope.ok:
        raise DispositionError("merged PR carrier is invalid: " + "; ".join(scope.reasons))
    attestation = str(pr_scope.attestation_payload(scope, head_sha=head_sha, base_sha=base_sha)["attestation_digest"])
    return scope, attestation


def _scope_from_merged_pr(
    pr: int, *, cwd: Path, recover_unledgered_merge: bool = False, persist_recovery: bool = True
) -> tuple[dict[str, Any], str, str, str, pr_scope.ScopeVerdict, str]:
    info = _gh_json(
        [
            "pr",
            "view",
            str(pr),
            "--json",
            "state,mergeCommit,headRefOid,baseRefOid,body,isDraft,author,files,title",
        ]
    )
    if info.get("state") != "MERGED":
        raise DispositionError(f"PR #{pr} is {info.get('state')!r}, not MERGED")
    head_sha = info.get("headRefOid")
    merge_commit = info.get("mergeCommit")
    merge_sha = merge_commit.get("oid") if isinstance(merge_commit, dict) else None
    if not isinstance(head_sha, str) or not head_sha or not isinstance(merge_sha, str) or not merge_sha:
        raise DispositionError(f"merged PR #{pr} lacks an observable head or squash merge SHA")
    try:
        pr_scope._ensure_local_commit(head_sha)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise DispositionError(f"cannot resolve merged PR #{pr}'s exact head locally: {exc}") from exc
    ledger = merge_boundary._read_ledger()
    matching_entries = [
        entry for entry in ledger["merges"] if entry.get("pr") == pr and entry.get("head_sha") == head_sha
    ]
    if len(matching_entries) != 1:
        if not recover_unledgered_merge or matching_entries:
            raise DispositionError(f"PR #{pr} has no unique merged ledger entry for its exact head")
        merge_base_sha = _squash_merge_base(merge_sha, cwd=cwd)
        scope, attestation = _validate_merged_scope(info, head_sha=head_sha, base_sha=merge_base_sha, cwd=cwd)
        title = info.get("title")
        if not isinstance(title, str) or not title:
            raise DispositionError(f"merged PR #{pr} lacks a title for recovery evidence")
        if persist_recovery:
            merge_boundary.record_post_merge_carrier_recovery(
                pr,
                head_sha,
                title,
                base_sha=merge_base_sha,
                merge_sha=merge_sha,
                scope_attestation_digest=attestation,
            )
        return info, head_sha, merge_sha, merge_base_sha, scope, attestation
    entry = matching_entries[0]
    merge_base_sha = entry.get("base_sha")
    if not isinstance(merge_base_sha, str) or not merge_base_sha:
        raise DispositionError(f"PR #{pr} ledger entry lacks its merge-time base SHA")
    scope, attestation = _validate_merged_scope(info, head_sha=head_sha, base_sha=merge_base_sha, cwd=cwd)
    if entry.get("scope_attestation_digest") != attestation:
        raise DispositionError(
            f"PR #{pr} has no merged ledger entry bound to its current exact-head carrier attestation"
        )
    return info, head_sha, merge_sha, merge_base_sha, scope, attestation


def _execution_id(*, pr: int, head_sha: str, merge_sha: str, attestation: str) -> str:
    return pr_scope._execution_id(pr=pr, head_sha=head_sha, merge_sha=merge_sha, attestation=attestation)


def _marker(*, execution_id: str, pr: int, merge_sha: str) -> str:
    return pr_scope._execution_marker(pr=pr, merge_sha=merge_sha, execution_id=execution_id)


def _validate_live_plan(scope: pr_scope.ScopeVerdict, live: dict[str, dict[str, Any]], *, marker: str) -> list[str]:
    statuses = {bead_id: str(row.get("status", "")) for bead_id, row in live.items()}
    from polylogue.maintenance.archive_verification import validate_archive_verification_registry

    def validate_registry(current_statuses: dict[str, str]) -> None:
        try:
            validate_archive_verification_registry(waiver_bead_statuses=current_statuses)
        except ValueError as exc:
            raise DispositionError(f"live archive-verification registry is invalid: {exc}") from exc

    validate_registry(statuses)
    projected_statuses = dict(statuses)
    for item in scope.dispositions:
        if item.disposition in {pr_scope.ScopeDisposition.SATISFIED, pr_scope.ScopeDisposition.SUPERSEDED}:
            projected_statuses[item.bead_id] = "closed"
        elif item.disposition is pr_scope.ScopeDisposition.DEFERRED:
            projected_statuses[item.bead_id] = "deferred"
    operations: list[str] = []
    for item in scope.dispositions:
        target = live.get(item.bead_id)
        if target is None:
            raise DispositionError(f"carrier target {item.bead_id} is absent from live Beads")
        status = target.get("status")
        terminal = item.disposition in {pr_scope.ScopeDisposition.SATISFIED, pr_scope.ScopeDisposition.SUPERSEDED}
        if terminal:
            if status == "closed":
                if target.get("close_reason") != marker:
                    raise DispositionError(f"carrier target {item.bead_id} was closed independently")
            elif isinstance(status, str) and status:
                operations.append(f"close {item.bead_id} {marker}")
            else:
                raise DispositionError(f"carrier target {item.bead_id} has malformed live status")
        elif status == "closed":
            raise DispositionError(f"nonterminal carrier target {item.bead_id} is already closed")
        elif item.disposition is pr_scope.ScopeDisposition.DEFERRED and status != "deferred":
            operations.append(f"update {item.bead_id} status=deferred")
        for successor in item.successors:
            successor_row = live.get(successor)
            if successor_row is None or projected_statuses.get(successor) == "closed":
                raise DispositionError(f"carrier successor {successor} is missing or closed in projected Beads")
            if not pr_scope._successor_is_linked(item.bead_id, successor, live):
                raise DispositionError(f"carrier successor {successor} is no longer durably linked to {item.bead_id}")
    validate_registry(projected_statuses)
    return operations


def _changed_ids(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> set[str]:
    return {bead_id for bead_id in set(before) | set(after) if before.get(bead_id) != after.get(bead_id)}


def _receipt_path(output: Path, execution_id: str) -> Path:
    return output.parent / "disposition-receipts" / f"{execution_id}.json"


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DispositionError(f"existing disposition receipt is unreadable: {path}") from exc
        if existing != receipt:
            raise DispositionError(f"refusing to overwrite immutable disposition receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _assert_scoped_export(
    base: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]], selected: set[str]
) -> set[str]:
    changed = _changed_ids(base, candidate)
    unexpected = sorted(changed - selected)
    if unexpected:
        raise DispositionError("refusing scoped export with unrelated live Bead changes: " + ", ".join(unexpected))
    return changed


def _receipt_identity(
    *,
    execution_id: str,
    pr: int,
    head_sha: str,
    merge_sha: str,
    base_sha: str,
    attestation: str,
    marker: str,
    scope: pr_scope.ScopeVerdict,
) -> dict[str, Any]:
    return {
        "version": 2,
        "execution_id": execution_id,
        "pr": pr,
        "head_sha": head_sha,
        "merge_sha": merge_sha,
        "base_sha": base_sha,
        "scope_attestation_digest": attestation,
        "marker": marker,
        "dispositions": [
            {
                "bead_id": item.bead_id,
                "disposition": item.disposition.value,
                "evidence": [asdict(ref) for ref in item.evidence],
                "successors": list(item.successors),
            }
            for item in scope.dispositions
        ],
    }


def _existing_receipt(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DispositionError(f"existing disposition receipt is unreadable: {path}") from exc
    if not isinstance(existing, dict):
        raise DispositionError(f"existing disposition receipt conflicts with this execution: {path}")
    matches_current = all(existing.get(key) == value for key, value in identity.items())
    legacy_identity = {key: value for key, value in identity.items() if key != "base_sha"} | {"version": 1}
    matches_legacy = existing.get("version") == 1 and all(
        existing.get(key) == value for key, value in legacy_identity.items()
    )
    if not matches_current and not matches_legacy:
        raise DispositionError(f"existing disposition receipt conflicts with this execution: {path}")
    if not isinstance(existing.get("changed_beads"), list):
        raise DispositionError(f"existing disposition receipt is incomplete: {path}")
    return existing


def _assert_existing_receipt_matches_result(
    existing: dict[str, Any],
    *,
    changed: set[str],
    after: dict[str, dict[str, Any]],
) -> None:
    """Refuse a retry whose retained receipt no longer names its exported result."""
    recorded_ids = existing.get("changed_beads")
    if not isinstance(recorded_ids, list) or not all(isinstance(bead_id, str) for bead_id in recorded_ids):
        raise DispositionError("existing disposition receipt is incomplete")
    if changed and sorted(recorded_ids) != sorted(changed):
        raise DispositionError("existing disposition receipt does not match the current batch result")
    if existing.get("version") != 2:
        return
    recorded_results = existing.get("changed_records")
    if not isinstance(recorded_results, dict) or set(recorded_results) != set(recorded_ids):
        raise DispositionError("existing disposition receipt is incomplete")
    if any(after.get(bead_id) != record for bead_id, record in recorded_results.items()):
        raise DispositionError("existing disposition receipt does not match the current batch result")


def apply_live_dispositions(
    pr: int,
    *,
    scope: pr_scope.ScopeVerdict,
    head_sha: str,
    merge_sha: str,
    attestation: str,
    cwd: Path,
) -> Path | None:
    """Execute a validated carrier immediately after a successful merge.

    This is deliberately a post-merge operation: failure is reported to the
    caller with recovery commands and never attempts to roll back GitHub.
    ``issues.jsonl`` is exported only after the guarded Beads transaction.
    """
    if scope.scope_kind is pr_scope.ScopeKind.SELF_CONTAINED:
        print(f"PR #{pr}: self-contained carrier; no Beads disposition is required")
        return None
    if scope.scope_kind is not pr_scope.ScopeKind.BEAD or not scope.dispositions:
        raise DispositionError(f"PR #{pr} has no executable Bead disposition scope")
    real_bd = str(bd_guard._validated_real_bd_path(os.environ.get("POLYLOGUE_BD_REAL", "bd")))
    env = os.environ.copy()
    env["BD_IMPORT_AUTO"] = "false"
    live = bd_guard._export_live_state(bd_command=real_bd, env=env, cwd=cwd)
    marker = _marker(
        execution_id=_execution_id(pr=pr, head_sha=head_sha, merge_sha=merge_sha, attestation=attestation),
        pr=pr,
        merge_sha=merge_sha,
    )
    _validate_live_plan(scope, live, marker=marker)
    selected = set(scope.assigned_beads)
    before, after = bd_guard.run_guarded_batch(
        expected_ids=selected,
        prepare=lambda current: _validate_live_plan(scope, current, marker=marker),
        cwd=cwd,
    )
    changed = _changed_ids(before, after)
    if not changed.issubset(selected):
        raise DispositionError("guarded merge-boundary batch changed rows outside its typed carrier scope")
    output = cwd / ".beads" / "issues.jsonl"
    bd_guard.atomic_write_jsonl(output, after)
    execution_id = _execution_id(pr=pr, head_sha=head_sha, merge_sha=merge_sha, attestation=attestation)
    _write_receipt(
        _receipt_path(output, execution_id),
        {
            **_receipt_identity(
                execution_id=execution_id,
                pr=pr,
                head_sha=head_sha,
                merge_sha=merge_sha,
                base_sha="merge-boundary-live",
                attestation=attestation,
                marker=marker,
                scope=scope,
            ),
            "changed_beads": sorted(changed),
            "changed_records": {bead_id: after[bead_id] for bead_id in sorted(changed)},
        },
    )
    print(
        f"applied carrier dispositions for PR #{pr}; exported {output} (receipt: {_receipt_path(output, execution_id)})"
    )
    return output


def cmd_apply(
    pr: int, *, base_export: Path, output: Path, dry_run: bool, recover_unledgered_merge: bool = False
) -> int:
    root = Path.cwd().resolve()
    try:
        _info, head_sha, merge_sha, base_sha, scope, attestation = _scope_from_merged_pr(
            pr,
            cwd=root,
            recover_unledgered_merge=recover_unledgered_merge,
            persist_recovery=not dry_run,
        )
        if scope.scope_kind is pr_scope.ScopeKind.SELF_CONTAINED:
            print(f"PR #{pr}: self-contained carrier; no Beads disposition is required")
            return 0
        if scope.scope_kind is not pr_scope.ScopeKind.BEAD or not scope.dispositions:
            raise DispositionError(f"PR #{pr} has no executable Bead disposition scope")
        base = bd_guard.parse_and_validate_jsonl(base_export.read_text(encoding="utf-8"))
        execution_id = _execution_id(pr=pr, head_sha=head_sha, merge_sha=merge_sha, attestation=attestation)
        marker = _marker(execution_id=execution_id, pr=pr, merge_sha=merge_sha)
        selected = set(scope.assigned_beads)
        identity = _receipt_identity(
            execution_id=execution_id,
            pr=pr,
            head_sha=head_sha,
            merge_sha=merge_sha,
            base_sha=base_sha,
            attestation=attestation,
            marker=marker,
            scope=scope,
        )
        receipt_path = _receipt_path(output, execution_id)
        existing_receipt = _existing_receipt(receipt_path, identity)
        real_bd = str(bd_guard._validated_real_bd_path(os.environ.get("POLYLOGUE_BD_REAL", "bd")))
        safe_env = os.environ.copy()
        safe_env["BD_IMPORT_AUTO"] = "false"
        live = bd_guard._export_live_state(bd_command=real_bd, env=safe_env, cwd=root)
        _assert_scoped_export(base, live, selected)
        operations = _validate_live_plan(scope, live, marker=marker)
        if dry_run:
            print(json.dumps({"pr": pr, "execution_id": execution_id, "operations": operations}, indent=2))
            return 0
        before, after = bd_guard.run_guarded_batch(
            expected_ids=selected,
            prepare=lambda current: _validate_live_plan(scope, current, marker=marker),
            cwd=root,
        )
        changed = _assert_scoped_export(base, after, selected)
        changed_after_batch = _changed_ids(before, after)
        if not changed_after_batch.issubset(selected):
            raise DispositionError("guarded batch changed rows outside its typed carrier scope")
        if existing_receipt is not None:
            _assert_existing_receipt_matches_result(existing_receipt, changed=changed, after=after)
        else:
            _write_receipt(
                receipt_path,
                {
                    **identity,
                    "changed_beads": sorted(changed),
                    "changed_records": {bead_id: after[bead_id] for bead_id in sorted(changed)},
                },
            )
        bd_guard.atomic_write_jsonl(output, after)
    except (
        DispositionError,
        bd_guard.BatchExecutionError,
        bd_guard.InvalidJsonlError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"REFUSING carrier dispositions for PR #{pr}: {exc}", file=sys.stderr)
        return 1
    print(
        f"prepared scoped Beads export for PR #{pr}: {output} (receipt: {receipt_path}); "
        "commit these two paths on a normal follow-on PR"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    recover_unledgered_merge = raw_argv[:1] == ["recover"]
    if recover_unledgered_merge:
        raw_argv.pop(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pr",
        type=int,
        help="Merged PR whose carrier should be applied (use `recover` only for an unledgered external merge)",
    )
    parser.add_argument("--base-export", required=True, type=Path, help="Current follow-on branch .beads/issues.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Target .beads/issues.jsonl for the follow-on PR")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the one batch without mutating Beads"
    )
    args = parser.parse_args(raw_argv)
    return cmd_apply(
        args.pr,
        base_export=args.base_export,
        output=args.output,
        dry_run=args.dry_run,
        recover_unledgered_merge=recover_unledgered_merge,
    )
