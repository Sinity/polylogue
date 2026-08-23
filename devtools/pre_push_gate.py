"""Reuse or run exact-head quick verification for a Git pre-push update."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root
from devtools.checkout_guard import CheckoutImportMismatchError, assert_polylogue_matches_checkout
from devtools.verify_runs import worktree_fingerprint


@dataclass(frozen=True, slots=True)
class PushUpdate:
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str


_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _validate_sha(value: str, *, field: str) -> str:
    if not _SHA_RE.fullmatch(value):
        raise ValueError(f"malformed pre-push {field}: expected a 40-character hexadecimal SHA")
    return value


def parse_updates(text: str) -> list[PushUpdate]:
    updates: list[PushUpdate] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"malformed pre-push update on line {number}: expected four fields")
        local_ref, local_sha, remote_ref, remote_sha = parts
        updates.append(
            PushUpdate(
                local_ref,
                _validate_sha(local_sha, field="local SHA"),
                remote_ref,
                _validate_sha(remote_sha, field="remote SHA"),
            )
        )
    return updates


def _is_zero_sha(value: str) -> bool:
    return bool(value) and set(value) == {"0"}


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _worktree_is_clean(cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", "."],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and not result.stderr and not result.stdout


def _current_provenance(cwd: Path) -> tuple[dict[str, object], str] | None:
    """Return the live provenance required to trust a cached quick receipt."""
    if not _worktree_is_clean(cwd):
        return None
    try:
        environment = assert_polylogue_matches_checkout(cwd, context="pre-push").as_dict()
    except (CheckoutImportMismatchError, ImportError, OSError, ValueError):
        return None
    fingerprint = worktree_fingerprint(cwd)
    if fingerprint == "unavailable":
        return None
    return environment, fingerprint


def _has_compatible_quick_receipt(
    *,
    cwd: Path,
    head: str | None,
    provenance: tuple[dict[str, object], str] | None,
) -> bool:
    """Accept only a successful receipt bound to this exact checkout state."""
    if head is None or provenance is None:
        return False
    environment, fingerprint = provenance
    receipt_path = cwd / ".cache" / "verify" / "current-run.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(receipt, Mapping):
        return False
    return (
        receipt.get("tier") == "quick"
        and receipt.get("status") == "success"
        and type(receipt.get("exit_code")) is int
        and receipt.get("exit_code") == 0
        and receipt.get("git_head") == head
        and receipt.get("final_git_head") == head
        and receipt.get("checkout_root") == str(cwd.resolve())
        and receipt.get("worktree_fingerprint") == fingerprint
        and receipt.get("final_worktree_fingerprint") == fingerprint
        and receipt.get("environment_fingerprint") == environment
    )


def _updates_match_head(updates: list[PushUpdate], head: str | None) -> bool:
    return head is not None and all(_is_zero_sha(update.local_sha) or update.local_sha == head for update in updates)


def run_gate(updates: list[PushUpdate], *, cwd: Path) -> str:
    del updates
    print("pre-push: running quick verification baseline", file=sys.stderr)
    _run([sys.executable, "-m", "devtools", "verify", "--quick"], cwd=cwd)
    return "quick"


def main(argv: list[str] | None = None) -> int:
    command_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("updates", type=Path, help="file containing Git pre-push update lines")
    args = parser.parse_args(command_argv)
    try:
        updates = parse_updates(args.updates.read_text(encoding="utf-8"))
        run_gate(updates, cwd=repo_root())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"pre-push: verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
