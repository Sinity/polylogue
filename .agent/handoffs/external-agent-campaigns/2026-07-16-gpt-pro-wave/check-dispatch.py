#!/usr/bin/env python3
"""Mechanical dispatch gate for the 2026-07-16 GPT Pro wave.

Replaces prose-only launch gating. For each job in each workload it answers:
may this job be dispatched right now, and if not, exactly which gate is unmet?

Gates checked
-------------
1. **foundation** gate: every Test Diet job requires
   ``foundation-receipt.json`` at the wave root::

       {"merged_commit": "<sha>", "verified_at": "<iso8601>", "notes": "..."}

   The receipt is written manually (or by the test-harness agent) once the
   local Test Diet foundation is merged to master. The commit must exist in
   the repository.
2. **job dependencies**: ``depends_on: ["testdiet-02", ...]`` require an
   immutable per-attempt ``result.json`` for that job in state
   ``verified | published``. The `index.json` ledger is a projection and is
   never dispatch authority.
3. **context manifest freshness** (testdiet only): every file listed in
   ``testdiet/context/MANIFEST.sha256`` must hash to its recorded value.
   The manifest's own SHA-256 is printed so it can be compared against the
   pinned baseline in the foundation mission; drift there is a report, not
   a failure (post-foundation corpus changes are expected to update it).
4. **owning-beads validity** (informational): ids in ``owning-beads.json``
   must exist in ``.beads/issues.jsonl``; closed or in-progress owners are
   flagged as coordination notes at dispatch time.

Exit status: 0 when every selected job is dispatchable, 2 when any selected
job is blocked, 3 on structural errors (missing files, bad JSON).

Usage::

    check-dispatch.py                       # all workloads, all jobs
    check-dispatch.py --workload testdiet   # one workload
    check-dispatch.py --job testdiet-04     # one job (implies its workload)
    check-dispatch.py --json                # machine-readable report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

WAVE_ROOT = Path(__file__).resolve().parent
WORKLOADS = ("testdiet", "beads", "analysis", "deep-research")
SATISFYING_STATES = {"verified", "published"}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"error: missing {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: bad JSON in {path}: {exc}") from exc


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "-C", str(WAVE_ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _commit_exists(repo: Path, sha: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
        ).returncode
        == 0
    )


def check_foundation(repo: Path) -> tuple[bool, str]:
    receipt_path = WAVE_ROOT / "foundation-receipt.json"
    if not receipt_path.exists():
        return False, (
            "foundation receipt missing: write foundation-receipt.json "
            '{"merged_commit": ..., "verified_at": ...} at the wave root '
            "after the test-harness foundation merges"
        )
    receipt = _load_json(receipt_path)
    sha = receipt.get("merged_commit", "")
    if not sha:
        return False, "foundation-receipt.json has no merged_commit"
    if not _commit_exists(repo, sha):
        return False, f"foundation merged_commit {sha} not found in repository"
    return True, f"foundation merged @ {sha[:12]} ({receipt.get('verified_at', 'unverified')})"


def check_manifest() -> tuple[bool, list[str]]:
    manifest = WAVE_ROOT / "testdiet" / "context" / "MANIFEST.sha256"
    if not manifest.exists():
        return False, ["testdiet/context/MANIFEST.sha256 missing"]
    problems: list[str] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            digest, rel = line.split(maxsplit=1)
        except ValueError:
            problems.append(f"unparseable manifest line: {line!r}")
            continue
        target = manifest.parent / rel
        if not target.exists():
            problems.append(f"missing context file: {rel}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            problems.append(f"context drift: {rel} (manifest {digest[:12]}, actual {actual[:12]})")
    return not problems, problems


def manifest_self_hash() -> str:
    manifest = WAVE_ROOT / "testdiet" / "context" / "MANIFEST.sha256"
    if not manifest.exists():
        return "(missing)"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def load_attempt_receipts(workload: str) -> dict[str, list[dict]]:
    by_job: dict[str, list[dict]] = {}
    results = WAVE_ROOT / workload / "results"
    for receipt_path in results.glob("*/a[0-9][0-9]/result.json"):
        receipt = _load_json(receipt_path)
        job_id = receipt.get("job_id")
        if not job_id:
            raise SystemExit(f"error: receipt has no job_id: {receipt_path}")
        by_job.setdefault(job_id, []).append(receipt)
    return by_job


def bead_states(repo: Path) -> dict[str, str]:
    jsonl = repo / ".beads" / "issues.jsonl"
    states: dict[str, str] = {}
    if not jsonl.exists():
        return states
    for line in jsonl.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        states[d.get("id", "")] = d.get("status", "?")
    return states


def owning_bead_notes(workload: str, job_id: str, states: dict[str, str]) -> list[str]:
    path = WAVE_ROOT / workload / "owning-beads.json"
    if not path.exists():
        return []
    data = _load_json(path)
    entry = data.get("jobs", {}).get(job_id)
    if not entry:
        return []
    notes = []
    for bead in entry.get("owning_beads", []):
        state = states.get(bead)
        if state is None:
            notes.append(f"owning bead {bead} NOT FOUND in issues.jsonl — fix owning-beads.json")
        elif state == "closed":
            notes.append(f"owning bead {bead} is closed — decisions final, read its terminal notes")
        elif state == "in_progress":
            notes.append(f"owning bead {bead} is IN PROGRESS locally — coordination hotspot")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--job")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    repo = _repo_root()
    states = bead_states(repo)
    foundation_ok, foundation_msg = check_foundation(repo)
    manifest_ok, manifest_problems = check_manifest()

    report: dict = {
        "foundation": {"ok": foundation_ok, "detail": foundation_msg},
        "context_manifest": {
            "ok": manifest_ok,
            "problems": manifest_problems,
            "manifest_sha256": manifest_self_hash(),
        },
        "jobs": [],
    }

    workloads = [args.workload] if args.workload else list(WORKLOADS)
    any_blocked = False
    for workload in workloads:
        campaign = _load_json(WAVE_ROOT / workload / "campaign.json")
        attempts = load_attempt_receipts(workload)
        for job in campaign.get("jobs", []):
            job_id = job["id"]
            if args.job and job_id != args.job:
                continue
            unmet: list[str] = []
            if workload == "testdiet" and not foundation_ok:
                unmet.append(foundation_msg)
            for dep in job.get("depends_on", []):
                if dep == "foundation":
                    continue
                if not any(a.get("state") in SATISFYING_STATES for a in attempts.get(dep, [])):
                    unmet.append(f"dependency {dep} has no result receipt with state in {sorted(SATISFYING_STATES)}")
            if workload == "testdiet" and not manifest_ok:
                unmet.extend(manifest_problems[:3])
            prior = len(attempts.get(job_id, []))
            notes = owning_bead_notes(workload, job_id, states)
            blocked = bool(unmet)
            any_blocked = any_blocked or blocked
            report["jobs"].append(
                {
                    "workload": workload,
                    "job": job_id,
                    "title": job.get("title", ""),
                    "dispatchable": not blocked,
                    "unmet": unmet,
                    "prior_attempts": prior,
                    "coordination_notes": notes,
                }
            )

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"foundation: {'OK ' if foundation_ok else 'BLOCKED '}- {foundation_msg}")
        mstate = "OK" if manifest_ok else "DRIFT"
        print(f"context manifest: {mstate} (sha256 {report['context_manifest']['manifest_sha256'][:16]}…)")
        for p in manifest_problems:
            print(f"  ! {p}")
        for j in report["jobs"]:
            flag = "READY  " if j["dispatchable"] else "BLOCKED"
            print(
                f"{flag} {j['workload']}/{j['job']}  {j['title']}"
                + (f"  [attempts: {j['prior_attempts']}]" if j["prior_attempts"] else "")
            )
            for u in j["unmet"]:
                print(f"        - {u}")
            for n in j["coordination_notes"]:
                print(f"        ~ {n}")
    return 2 if any_blocked else 0


if __name__ == "__main__":
    sys.exit(main())
