"""Monotonic, receipted reconciliation for Beads JSONL snapshots.

Linked worktrees share one live Dolt database but retain the JSONL snapshot
from their branch point. Beads' automatic JSONL import path can blind-upsert,
so the repository's ``scripts/bd`` entry point compares the snapshot with live
state before every normal invocation and delegates with automatic imports
disabled. The 2026-07-15 planning audit showed why: a staged issues.jsonl can
be syntactically valid yet contain stale rows.

This module owns the explicit reconciliation flows that remain legitimate:

- ``merge_rows`` compares every row by revision (`updated_at`, the only
  per-row freshness signal bd's export exposes) and classifies each id as
  new / updated / equal / skipped_downgrade / conflicted (incomparable) /
  recovered_downgrade. Ordinary synchronization can never silently apply a
  downgrade; incomparable rows (missing revision on either side) are never
  guessed at -- they are reported as conflicts, not merged.
- ``SyncReceipt`` is the union outcome: every row's classification plus the
  actor/reason/source-fingerprint that produced it, written to
  ``.cache/bd-sync-receipts/`` so downstream policy consumers (the
  portfolio/frontier gate, polylogue-8jg9.1) can require a receipt and
  reject corrupt/incomplete/downgrading synchronization instead of trusting
  bare command exit status.
- ``parse_and_validate_jsonl``/``atomic_write_jsonl`` give any writer (this
  guard's own repair step, or an explicit `reconcile`/`export` invocation)
  snapshot-consistent, conflict-marker-refusing, duplicate-id-refusing,
  temp+fsync+rename writes -- so a valid whole-file replacement built from a
  stale source cannot silently win over newer per-row state, and a
  marker-bearing file can never be staged as if it were clean.
- ``cmd_reconcile`` is the explicit, operator-invokable entry point for
  explicit-recovery flows such as `git reset --hard` or a hand-merged
  conflict file. Normal ``scripts/bd`` invocations apply only new/updated
  rows before delegating to Beads. Recovering a genuine downgrade (accepting
  an older row on purpose, e.g. because live state itself was corrupted)
  requires ``--allow-downgrade`` plus a recorded ``--actor``/``--reason``,
  and every downgraded row is still named in the receipt.

This module deliberately depends on nothing beyond the Python stdlib so an
operator can use it from a repository checkout without the devshell/venv.

Usage:
  bd_reimport_guard.py invoke <real-bd-path> <bd-args...>
  bd_reimport_guard.py reconcile <candidate-jsonl-path> [--source NAME]
      [--allow-downgrade --actor ACTOR --reason REASON] [--dry-run]
  bd_reimport_guard.py export <target-jsonl-path>
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal

# --- revision-comparable row classification ---------------------------------

Outcome = Literal[
    "new",
    "updated",
    "equal",
    "skipped_downgrade",
    "conflicted",
    "recovered_downgrade",
]

CONFLICT_MARKERS: tuple[str, ...] = ("<<<<<<<", "=======", ">>>>>>>")


class InvalidJsonlError(ValueError):
    """Raised when a candidate JSONL payload fails structural validation."""


@dataclass(frozen=True, slots=True)
class RowOutcome:
    issue_id: str
    outcome: Outcome
    current_revision: str | None
    candidate_revision: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.issue_id,
            "outcome": self.outcome,
            "current_revision": self.current_revision,
            "candidate_revision": self.candidate_revision,
        }


@dataclass(frozen=True, slots=True)
class SyncReceipt:
    created_at: str
    source: str
    actor: str | None
    reason: str | None
    recovery: bool
    outcomes: list[RowOutcome] = field(default_factory=list)

    @property
    def downgraded_ids(self) -> list[str]:
        return [o.issue_id for o in self.outcomes if o.outcome in ("skipped_downgrade", "recovered_downgrade")]

    @property
    def conflicted_ids(self) -> list[str]:
        return [o.issue_id for o in self.outcomes if o.outcome == "conflicted"]

    @property
    def changed_ids(self) -> list[str]:
        return [o.issue_id for o in self.outcomes if o.outcome in ("new", "updated", "recovered_downgrade")]

    @property
    def is_clean(self) -> bool:
        """True when synchronization applied with no conflicts and no unauthorized downgrades."""
        if self.conflicted_ids:
            return False
        return not any(o.outcome == "skipped_downgrade" for o in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "source": self.source,
            "actor": self.actor,
            "reason": self.reason,
            "recovery": self.recovery,
            "is_clean": self.is_clean,
            "summary": {
                "total": len(self.outcomes),
                "new": sum(1 for o in self.outcomes if o.outcome == "new"),
                "updated": sum(1 for o in self.outcomes if o.outcome == "updated"),
                "equal": sum(1 for o in self.outcomes if o.outcome == "equal"),
                "skipped_downgrade": sum(1 for o in self.outcomes if o.outcome == "skipped_downgrade"),
                "recovered_downgrade": sum(1 for o in self.outcomes if o.outcome == "recovered_downgrade"),
                "conflicted": sum(1 for o in self.outcomes if o.outcome == "conflicted"),
            },
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def _revision_of(row: dict[str, Any]) -> str | None:
    """The only per-row freshness signal bd's export exposes.

    A missing/empty revision is deliberately treated as "unknown", never as
    the oldest-possible or newest-possible value -- guessing either way
    would let an incomparable row silently win or lose a merge.
    """
    value = row.get("updated_at")
    if isinstance(value, str) and value:
        return value
    return None


def merge_rows(
    current: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    allow_downgrade: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[RowOutcome]]:
    """Merge `candidate` rows into `current` by per-row revision.

    Returns (merged, outcomes). `merged` always contains every id from
    `current` plus every new id from `candidate`; a candidate row only
    replaces a current row when its revision is strictly newer, or when
    `allow_downgrade=True` and it is older (recorded as an explicit
    recovery, never silent). Rows present only in `current` are left
    untouched and produce no outcome entry (nothing to reconcile).
    """
    merged = dict(current)
    outcomes: list[RowOutcome] = []
    for issue_id, cand_row in candidate.items():
        cur_row = current.get(issue_id)
        cand_rev = _revision_of(cand_row)
        if cur_row is None:
            merged[issue_id] = cand_row
            outcomes.append(RowOutcome(issue_id, "new", None, cand_rev))
            continue

        cur_rev = _revision_of(cur_row)
        if cur_rev is None or cand_rev is None:
            # Incomparable -- never guess a winner. Current state is kept.
            outcomes.append(RowOutcome(issue_id, "conflicted", cur_rev, cand_rev))
            continue

        if cand_rev > cur_rev:
            merged[issue_id] = cand_row
            outcomes.append(RowOutcome(issue_id, "updated", cur_rev, cand_rev))
        elif cand_rev == cur_rev:
            outcomes.append(RowOutcome(issue_id, "equal", cur_rev, cand_rev))
        else:
            if allow_downgrade:
                merged[issue_id] = cand_row
                outcomes.append(RowOutcome(issue_id, "recovered_downgrade", cur_rev, cand_rev))
            else:
                outcomes.append(RowOutcome(issue_id, "skipped_downgrade", cur_rev, cand_rev))

    return merged, outcomes


def build_receipt(
    outcomes: list[RowOutcome],
    *,
    source: str,
    actor: str | None = None,
    reason: str | None = None,
    recovery: bool = False,
) -> SyncReceipt:
    return SyncReceipt(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source=source,
        actor=actor,
        reason=reason,
        recovery=recovery,
        outcomes=outcomes,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _invocation_directory(args: list[str]) -> Path:
    """Resolve bd's optional ``-C``/``--directory`` target for guard I/O."""
    for index, arg in enumerate(args):
        if arg in ("-C", "--directory") and index + 1 < len(args):
            return Path(args[index + 1]).expanduser().resolve()
        for prefix in ("-C=", "--directory="):
            if arg.startswith(prefix):
                return Path(arg[len(prefix) :]).expanduser().resolve()
    return Path.cwd().resolve()


def _shared_lock_path(start: Path) -> Path:
    """Return one lock path shared by all linked worktrees of this repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            common_dir = Path(result.stdout.strip())
            if not common_dir.is_absolute():
                common_dir = start / common_dir
            return common_dir.resolve() / "polylogue-bd-guard.lock"
    except (OSError, subprocess.SubprocessError):
        pass

    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
    token = hashlib.sha256(str(start).encode()).hexdigest()[:20]
    return runtime_dir / f"polylogue-bd-guard-{token}.lock"


def _acquire_invocation_lock(start: Path) -> IO[str]:
    """Serialize guarded bd calls across linked worktrees and retain the lock.

    The file descriptor is made inheritable by ``cmd_invoke`` before it
    replaces this process with the real bd binary. This covers the check,
    filtered import, and delegated command as one writer boundary.
    """
    path = _shared_lock_path(start)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _receipts_dir() -> Path:
    return _repo_root() / ".cache" / "bd-sync-receipts"


def write_receipt(receipt: SyncReceipt) -> Path:
    directory = _receipts_dir()
    directory.mkdir(parents=True, exist_ok=True)
    token = receipt.created_at.replace(":", "").replace("-", "")
    path = directory / f"{token}-{receipt.source}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


# --- validated, atomic JSONL I/O ---------------------------------------------


def _has_conflict_markers(text: str) -> bool:
    return any(line.startswith(marker) for line in text.splitlines() for marker in CONFLICT_MARKERS)


def parse_and_validate_jsonl(text: str) -> dict[str, dict[str, Any]]:
    """Parse a JSONL payload into {id: row}, refusing corrupt/ambiguous input.

    Raises InvalidJsonlError when the text carries literal merge-conflict
    markers, contains a line that isn't valid JSON, contains a row without
    an `id`, or contains a duplicate id -- all cases where "the file parses"
    is not proof the content is a coherent, mergeable snapshot.
    """
    if _has_conflict_markers(text):
        raise InvalidJsonlError("payload contains unresolved merge-conflict markers")

    rows: dict[str, dict[str, Any]] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InvalidJsonlError(f"line {lineno}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict) or not row.get("id"):
            raise InvalidJsonlError(f"line {lineno}: row missing a non-empty 'id'")
        issue_id = row["id"]
        if issue_id in rows:
            raise InvalidJsonlError(f"duplicate id {issue_id!r} (line {lineno})")
        rows[issue_id] = row
    return rows


def atomic_write_jsonl(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    """Write `rows` to `path` via temp-write + fsync + atomic rename.

    Refuses to overwrite a target that currently contains unresolved
    conflict markers -- that state must go through an explicit recovery
    path, never a routine overwrite.
    """
    if path.exists():
        existing = path.read_text()
        if _has_conflict_markers(existing):
            raise InvalidJsonlError(f"refusing to overwrite {path}: existing content has unresolved conflict markers")

    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows.values())
    # Validate our own output round-trips cleanly before it ever touches disk.
    parse_and_validate_jsonl(text)

    fd_dir = path.parent
    fd_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=fd_dir, suffix=".tmp", delete=False) as tf:
        tmp_path = Path(tf.name)
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
    tmp_path.replace(path)


# --- live bd state ------------------------------------------------------------


def _export_live_state(
    *,
    bd_command: str = "bd",
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return {id: full_row_dict} for every issue currently live.

    Uses default `bd export` filtering (excludes infra beads/memories) --
    consistently applied to both before/after snapshots, so the exclusion
    itself introduces no asymmetry.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        path = tf.name
    try:
        process_result = subprocess.run(
            [bd_command, "export", "-o", path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=cwd,
        )
        if process_result.returncode != 0:
            raise RuntimeError(f"bd export failed ({process_result.returncode}): {process_result.stderr.strip()}")
        result: dict[str, dict[str, Any]] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                issue_id = row.get("id")
                if issue_id:
                    result[issue_id] = row
        return result
    finally:
        Path(path).unlink(missing_ok=True)


def _bd_import_rows(
    rows: dict[str, dict[str, Any]],
    *,
    bd_command: str = "bd",
    env: dict[str, str] | None = None,
    reexport: bool = True,
    cwd: Path | None = None,
) -> None:
    if not rows:
        return
    # The installed bd importer performs its own conditional upsert. Never
    # pass --allow-stale here: an explicit downgrade remains an audited
    # reconcile operation, not part of ordinary wrapper synchronization.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
        import_path = tf.name
        for row in rows.values():
            tf.write(json.dumps(row) + "\n")
    try:
        result = subprocess.run(
            [bd_command, "import", import_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=cwd,
        )
        if result.returncode != 0:
            raise RuntimeError(f"bd import failed ({result.returncode}): {result.stderr.strip()}")
        if reexport:
            # Re-export so the working tree's issues.jsonl matches restored live
            # state (avoids the restore looking like a phantom git diff).
            subprocess.run(
                [bd_command, "export"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=cwd,
            )
    finally:
        Path(import_path).unlink(missing_ok=True)


def _find_candidate_jsonl(start: Path | None = None) -> Path | None:
    """Find the checked-out workspace snapshot without invoking bd."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".beads" / "issues.jsonl"
        if candidate.is_file():
            return candidate
    return None


def _preflight_invocation(
    bd_command: str,
    candidate_path: Path | None = None,
    *,
    cwd: Path | None = None,
) -> None:
    """Import only candidate rows newer than live state before ``bd`` runs.

    The delegated command receives ``BD_IMPORT_AUTO=false`` so Beads cannot
    perform its blind automatic import after this comparison. Failures are
    reported as warnings and leave the live database untouched; the requested
    bd command still gets to run against the live database.
    """
    candidate_path = candidate_path or _find_candidate_jsonl()
    if candidate_path is None:
        return

    safe_env = os.environ.copy()
    safe_env["BD_IMPORT_AUTO"] = "false"
    try:
        live = _export_live_state(bd_command=bd_command, env=safe_env, cwd=cwd)
        candidate = parse_and_validate_jsonl(candidate_path.read_text())
        merged, outcomes = merge_rows(live, candidate)
    except (OSError, InvalidJsonlError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"bd guard: skipped automatic JSONL reconciliation: {exc}", file=sys.stderr)
        return

    to_apply = {o.issue_id: merged[o.issue_id] for o in outcomes if o.outcome in ("new", "updated")}
    skipped = [o.issue_id for o in outcomes if o.outcome == "skipped_downgrade"]
    if skipped:
        print(
            f"bd guard: skipped {len(skipped)} stale JSONL row(s) before bd invocation",
            file=sys.stderr,
        )
    if not to_apply:
        return

    try:
        _bd_import_rows(to_apply, bd_command=bd_command, env=safe_env, reexport=False, cwd=cwd)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"bd guard: skipped automatic JSONL reconciliation: {exc}", file=sys.stderr)


def cmd_invoke(args: list[str]) -> int:
    """Run the installed bd binary behind the monotonic preflight boundary."""
    if len(args) < 1:
        print("usage: bd_reimport_guard.py invoke <real-bd-path> <bd-args...>", file=sys.stderr)
        return 1

    bd_command, *bd_args = args
    invocation_dir = _invocation_directory(bd_args)
    candidate_path = _find_candidate_jsonl(invocation_dir)
    lock = _acquire_invocation_lock(invocation_dir)
    _preflight_invocation(bd_command, candidate_path, cwd=invocation_dir)
    safe_env = os.environ.copy()
    safe_env["BD_IMPORT_AUTO"] = "false"
    os.set_inheritable(lock.fileno(), True)
    os.execvpe(bd_command, [bd_command, *bd_args], safe_env)
    return 1


# --- commands ------------------------------------------------------------------


def cmd_reconcile(args: list[str]) -> int:
    """Explicit, operator-invokable monotonic sync of a candidate JSONL file.

    Covers flows the git hooks never see -- most importantly `git reset
    --hard`, which rewrites .beads/issues.jsonl in the working tree without
    firing any hook, leaving it live-hazardous until the next `bd`
    invocation reimports it (bd reimports from the file on every call).
    Run this immediately after such a reset, pointing at the reverted file,
    to fold any genuinely newer live rows back in before anything else
    touches `bd`.
    """
    if not args:
        print("usage: bd_reimport_guard.py reconcile <candidate-jsonl-path> [options]", file=sys.stderr)
        return 1

    candidate_path = Path(args[0])
    rest = args[1:]
    source = "reconcile"
    allow_downgrade = False
    actor: str | None = None
    reason: str | None = None
    dry_run = False

    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg == "--source" and i + 1 < len(rest):
            source = rest[i + 1]
            i += 2
        elif arg == "--allow-downgrade":
            allow_downgrade = True
            i += 1
        elif arg == "--actor" and i + 1 < len(rest):
            actor = rest[i + 1]
            i += 2
        elif arg == "--reason" and i + 1 < len(rest):
            reason = rest[i + 1]
            i += 2
        elif arg == "--dry-run":
            dry_run = True
            i += 1
        else:
            print(f"reconcile: unknown argument {arg!r}", file=sys.stderr)
            return 1

    if allow_downgrade and (not actor or not reason):
        print("reconcile: --allow-downgrade requires both --actor and --reason", file=sys.stderr)
        return 1

    if not candidate_path.exists():
        print(f"reconcile: candidate file not found: {candidate_path}", file=sys.stderr)
        return 1

    try:
        candidate = parse_and_validate_jsonl(candidate_path.read_text())
    except InvalidJsonlError as exc:
        print(f"reconcile: candidate file failed validation: {exc}", file=sys.stderr)
        receipt = build_receipt([], source=source, actor=actor, reason=reason or str(exc), recovery=allow_downgrade)
        write_receipt(receipt)
        return 2

    current = _export_live_state()
    merged, outcomes = merge_rows(current, candidate, allow_downgrade=allow_downgrade)
    receipt = build_receipt(outcomes, source=source, actor=actor, reason=reason, recovery=allow_downgrade)
    receipt_path = write_receipt(receipt)

    apply_outcomes = ("new", "updated", "recovered_downgrade")
    to_apply = {o.issue_id: merged[o.issue_id] for o in outcomes if o.outcome in apply_outcomes}

    print(
        f"reconcile: {len(outcomes)} candidate row(s) -- {receipt.to_dict()['summary']} (receipt: {receipt_path})",
        file=sys.stderr,
    )
    if receipt.conflicted_ids:
        print(
            f"reconcile: {len(receipt.conflicted_ids)} incomparable row(s) left untouched: {receipt.conflicted_ids}",
            file=sys.stderr,
        )
    if any(o.outcome == "skipped_downgrade" for o in outcomes):
        skipped = [o.issue_id for o in outcomes if o.outcome == "skipped_downgrade"]
        print(
            f"reconcile: {len(skipped)} downgrade(s) refused (rerun with --allow-downgrade to recover): {skipped}",
            file=sys.stderr,
        )

    if dry_run:
        print("reconcile: --dry-run set, no changes applied", file=sys.stderr)
        return 0 if receipt.is_clean or allow_downgrade else 1

    _bd_import_rows(to_apply)

    return 0 if receipt.is_clean or allow_downgrade else 1


def cmd_export(args: list[str]) -> int:
    """Atomic, validated export of current live state to a target JSONL path."""
    if not args:
        print("usage: bd_reimport_guard.py export <target-jsonl-path>", file=sys.stderr)
        return 1
    target = Path(args[0])
    state = _export_live_state()
    try:
        atomic_write_jsonl(target, state)
    except InvalidJsonlError as exc:
        print(f"export: refused: {exc}", file=sys.stderr)
        return 1
    print(f"export: wrote {len(state)} row(s) to {target}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) < 1:
        print(
            "usage: bd_reimport_guard.py <invoke|reconcile|export> ...",
            file=sys.stderr,
        )
        return 1
    command = args[0]
    if command == "invoke":
        return cmd_invoke(args[1:])
    if command == "reconcile":
        return cmd_reconcile(args[1:])
    if command == "export":
        return cmd_export(args[1:])
    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
