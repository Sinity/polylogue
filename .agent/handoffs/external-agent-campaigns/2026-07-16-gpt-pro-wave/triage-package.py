#!/usr/bin/env python3
"""Triage an incoming GPT Pro result package (the receiving half of the wave).

Given a downloaded result ZIP, this script produces the evidence a human (or
coordinating agent) needs before spending any review effort:

1. package identity: SHA-256, byte size, member listing;
2. contract validation: required members (``HANDOFF.md``, ``PATCH.diff``,
   ``TESTS.md``, ``EVIDENCE.md``) present, no copied input archives, no
   suspiciously huge members;
3. patch sanity: non-empty unified diff, no placeholder markers;
4. apply check: ``git apply --check`` in a throwaway detached worktree at the
   snapshot commit the package claims (``--snapshot``), so a stale or
   corrupted patch is caught before anyone reads it.

It deliberately does NOT execute commands from TESTS.md — running a package's
own test commands happens later, in a reviewed lane worktree, after the patch
content has been read. Triage is evidence gathering, not integration.

Emits an attempt-record JSON to stdout; ``--record`` appends it to the
workload's ``results/index.json`` and stores nothing else. Store the raw ZIP
under ``results/<job>/<attempt>/raw/`` per the results README.

Usage::

    triage-package.py path/to/testdiet-01-...-r01.zip \
        --workload testdiet --job testdiet-01 --snapshot <commit> [--record]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

WAVE_ROOT = Path(__file__).resolve().parent
REQUIRED_MEMBERS = {"HANDOFF.md", "PATCH.diff", "TESTS.md", "EVIDENCE.md"}
PLACEHOLDER_PATTERNS = [
    re.compile(p)
    for p in (
        r"<placeholder",
        r"TODO: fill",
        r"rest of (the )?file (is )?unchanged",
        r"\.\.\. existing code \.\.\.",
        r"# unchanged below",
    )
]
HUGE_MEMBER_BYTES = 20 * 1024 * 1024


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "-C", str(WAVE_ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _member_basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def apply_check(repo: Path, snapshot: str, patch_text: str) -> tuple[bool, str]:
    if (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{snapshot}^{{commit}}"],
            capture_output=True,
        ).returncode
        != 0
    ):
        return False, f"snapshot commit {snapshot} not found in repository"
    with tempfile.TemporaryDirectory(prefix="triage-wt-", dir="/realm/tmp") as tmp:
        wt = Path(tmp) / "wt"
        add = subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(wt), snapshot],
            capture_output=True,
            text=True,
        )
        if add.returncode != 0:
            return False, f"worktree add failed: {add.stderr.strip()[:300]}"
        try:
            patch_file = Path(tmp) / "package.diff"
            patch_file.write_text(patch_text)
            res = subprocess.run(
                ["git", "-C", str(wt), "apply", "--check", "--verbose", str(patch_file)],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                return True, "patch applies cleanly"
            return False, (res.stderr.strip() or res.stdout.strip())[:500]
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)],
                capture_output=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--attempt", required=True, help="immutable attempt id (aNN)")
    parser.add_argument("--package-revision", required=True, help="immutable package revision (rNN)")
    parser.add_argument("--prompt-sha256", required=True, help="64-character hash of the dispatched prompt")
    parser.add_argument("--snapshot", help="commit the package claims to target (from its HANDOFF.md)")
    parser.add_argument(
        "--write", action="store_true", help="preserve immutable raw custody and write the result receipt"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"a[0-9]{2}", args.attempt):
        raise SystemExit("error: --attempt must be aNN")
    if not re.fullmatch(r"r[0-9]{2}", args.package_revision):
        raise SystemExit("error: --package-revision must be rNN")
    if not re.fullmatch(r"[0-9a-f]{64}", args.prompt_sha256):
        raise SystemExit("error: --prompt-sha256 must be a lowercase SHA-256")

    zp: Path = args.zip_path
    if not zp.exists():
        raise SystemExit(f"error: {zp} does not exist")

    data = zp.read_bytes()
    record: dict = {
        "job": args.job,
        "workload": args.workload,
        "zip": zp.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot": args.snapshot,
        "problems": [],
        "state": "triaged",
    }

    try:
        zf = zipfile.ZipFile(zp)
    except zipfile.BadZipFile:
        record["problems"].append("not a valid ZIP")
        record["state"] = "rejected"
        zf = None

    if zf is not None:
        names = zf.namelist()
        basenames = {_member_basename(n) for n in names if not n.endswith("/")}
        record["members"] = sorted(names)

        for required in sorted(REQUIRED_MEMBERS - basenames):
            record["problems"].append(f"missing required member {required}")
        for n in names:
            info = zf.getinfo(n)
            if n.lower().endswith(".zip"):
                record["problems"].append(f"nested archive in package: {n} (copied inputs are forbidden)")
            if info.file_size > HUGE_MEMBER_BYTES:
                record["problems"].append(f"oversized member {n} ({info.file_size} bytes)")

        patch_names = [n for n in names if _member_basename(n) == "PATCH.diff"]
        if patch_names:
            patch_text = zf.read(patch_names[0]).decode("utf-8", errors="replace")
            if not patch_text.strip():
                record["problems"].append("PATCH.diff is empty")
            elif "+++ " not in patch_text:
                record["problems"].append("PATCH.diff has no unified-diff hunks")
            hits = sorted({pat.pattern for pat in PLACEHOLDER_PATTERNS if pat.search(patch_text)})
            if hits:
                record["problems"].append(f"placeholder markers in PATCH.diff: {hits}")
            record["patch_lines"] = patch_text.count("\n")
            if args.snapshot and not record["problems"]:
                ok, detail = apply_check(_repo_root(), args.snapshot, patch_text)
                record["patch_applies"] = ok
                record["apply_detail"] = detail
                if not ok:
                    record["problems"].append(f"apply-check failed: {detail[:200]}")
            elif not args.snapshot:
                record["patch_applies"] = None
                record["problems"].append("no --snapshot given: apply-check skipped (get the commit from HANDOFF.md)")

    if any(p.startswith(("missing required member", "not a valid", "PATCH.diff is empty")) for p in record["problems"]):
        record["state"] = "rejected"

    print(json.dumps(record, indent=2))

    if args.write:
        receipt_path = WAVE_ROOT / args.workload / "results" / args.job / args.attempt / "result.json"
        raw_dir = receipt_path.parent / "raw"
        if receipt_path.exists():
            raise SystemExit(f"error: immutable receipt already exists: {receipt_path}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        custody = raw_dir / zp.name
        adopted = zp.resolve() == custody.resolve()
        if custody.exists() and not adopted:
            raise SystemExit(f"error: immutable raw artifact already exists: {custody}")
        if not adopted:
            shutil.copy2(zp, custody)
        receipt = {
            "schema_version": 1,
            "campaign_id": _load_campaign_id(args.workload),
            "workload_id": args.workload,
            "job_id": args.job,
            "attempt_id": args.attempt,
            "package_revision": args.package_revision,
            "state": record["state"],
            "provider": None,
            "provider_conversation_id": None,
            "provider_turn_id": None,
            "polylogue_session_ref": None,
            "prompt_sha256": args.prompt_sha256,
            "snapshot_identity": args.snapshot,
            "supersedes_attempt_id": None,
            "artifacts": [
                {
                    "filename": zp.name,
                    "sha256": record["sha256"],
                    "size_bytes": record["bytes"],
                    "custody_path": str(custody.relative_to(WAVE_ROOT)),
                    "polylogue_attachment_ref": None,
                }
            ],
            "triage": "; ".join(record["problems"]) if record["problems"] else record.get("apply_detail", "validated"),
            "beads": [],
            "worktree": None,
            "agent_handle": None,
            "commits": [],
            "pull_request": None,
            "verification_receipts": [],
            "notes": "created by triage-package.py; later integration evidence belongs in a new immutable receipt revision",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        reconcile = subprocess.run(
            [sys.executable, str(WAVE_ROOT.parent / "reconcile_results.py"), str(WAVE_ROOT), "--write"],
            capture_output=True,
            text=True,
        )
        if reconcile.returncode:
            raise SystemExit(f"error: receipt was written but index materialization failed: {reconcile.stderr.strip()}")
        print(f"wrote immutable receipt {receipt_path} and rebuilt indexes", file=sys.stderr)
    return 0 if record["state"] == "triaged" and not record["problems"] else 2


def _load_campaign_id(workload: str) -> str:
    campaign = json.loads((WAVE_ROOT / workload / "campaign.json").read_text())
    campaign_id = campaign.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise SystemExit(f"error: {workload}/campaign.json has no campaign_id")
    return campaign_id


if __name__ == "__main__":
    sys.exit(main())
