"""Aggregate verification evidence into a structured status report.

This command reads cached verification data and produces a status dashboard
without running tests. It aggregates:

- Verify history (success/failure runs)
- Contract evidence artifacts
- Suppression registry state
- Witness lifecycle
- Benchmark campaign runs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from devtools import repo_root as _get_root
from devtools.verify_suppressions import discover_source_suppressions
from polylogue.proof.suppressions import load_suppressions

ROOT = _get_root()


def _load_verify_history(history_file: Path) -> list[dict[str, Any]]:
    """Load verify history from JSONL file."""
    if not history_file.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with open(history_file) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return entries


def _load_contract_evidence(evidence_dir: Path) -> list[dict[str, Any]]:
    """Load contract evidence artifacts from .json files."""
    if not evidence_dir.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    try:
        for json_file in sorted(evidence_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    artifacts.append(data)
            except (OSError, json.JSONDecodeError):
                pass
    except OSError:
        pass
    return artifacts


def _load_witnesses(witnesses_dir: Path) -> list[dict[str, Any]]:
    """Load witness metadata from .witness.json files."""
    if not witnesses_dir.exists():
        return []
    witnesses: list[dict[str, Any]] = []
    try:
        for json_file in sorted(witnesses_dir.glob("*.witness.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    witnesses.append(data)
            except (OSError, json.JSONDecodeError):
                pass
    except OSError:
        pass
    return witnesses


def _load_benchmark_campaigns(campaigns_dir: Path) -> list[tuple[str, datetime]]:
    """Load benchmark campaign run names and dates."""
    if not campaigns_dir.exists():
        return []
    runs: list[tuple[str, datetime]] = []
    try:
        for json_file in sorted(campaigns_dir.glob("**/*.json"), reverse=True):
            try:
                mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
                runs.append((json_file.stem, mtime))
            except OSError:
                pass
    except OSError:
        pass
    return runs


def _get_last_n_days(entries: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    """Filter entries to last N days."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    result = []
    for entry in entries:
        try:
            ts_str = entry.get("timestamp", "")
            if ts_str:
                # Parse ISO 8601 timestamp
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts >= cutoff:
                    result.append(entry)
        except (ValueError, AttributeError):
            pass
    return result


def _format_timestamp(ts_str: str | None) -> str:
    """Format timestamp for display."""
    if not ts_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return ts_str


def _is_stale_witness(witness: dict[str, Any], days: int = 30) -> bool:
    """Check if witness is stale (not exercised in N days)."""
    lifecycle = witness.get("lifecycle", {})
    last_exercised = lifecycle.get("last_exercised_at")
    if not last_exercised:
        return False
    try:
        dt = datetime.fromisoformat(last_exercised.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_days = (now - dt).days
        return age_days > days
    except (ValueError, AttributeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Output JSON format")
    args = p.parse_args(argv)

    cache_dir = ROOT / ".cache"
    verify_history_file = cache_dir / "verify-history.jsonl"
    evidence_dir = cache_dir / "verification" / "evidence"
    witnesses_dir = ROOT / "tests" / "witnesses"
    campaigns_dir = ROOT / ".local" / "benchmark-campaigns"

    # Load all data
    verify_history = _load_verify_history(verify_history_file)
    contract_evidence = _load_contract_evidence(evidence_dir)
    witnesses = _load_witnesses(witnesses_dir)
    benchmark_runs = _load_benchmark_campaigns(campaigns_dir)
    registry_path = ROOT / "docs" / "plans" / "suppressions.yaml"
    registry = load_suppressions(registry=registry_path) if registry_path.exists() else []
    suppressions = discover_source_suppressions(ROOT, suppressions=registry)

    # Process verify history
    last_5_runs = verify_history[-5:] if verify_history else []
    last_30d_runs = _get_last_n_days(verify_history, 30)
    pass_count = sum(1 for r in last_30d_runs if r.get("exit_code") == 0)
    fail_count = sum(1 for r in last_30d_runs if r.get("exit_code") != 0)

    # Process contract evidence
    evidence_by_prefix: dict[str, int] = defaultdict(int)
    stale_artifacts = []
    for artifact in contract_evidence:
        contract = artifact.get("contract", "unknown")
        prefix = contract.split(".")[0] if contract else "unknown"
        evidence_by_prefix[prefix] += 1
        if artifact.get("dirty") or not artifact.get("git_sha"):
            stale_artifacts.append(contract)

    # Process suppressions
    suppression_by_kind = Counter(s.kind for s in suppressions)
    unregistered_suppressions = [s for s in suppressions if not s.registered]

    # Process witnesses
    stale_witnesses = [w for w in witnesses if _is_stale_witness(w)]
    witness_names = [w.get("witness_id", "unknown") for w in stale_witnesses]

    # Process benchmark campaigns
    latest_campaigns: dict[str, datetime] = {}
    for name, mtime in benchmark_runs:
        if name not in latest_campaigns or mtime > latest_campaigns[name]:
            latest_campaigns[name] = mtime

    # Report timestamp
    now = datetime.now(timezone.utc)

    if args.json:
        output = {
            "timestamp": now.isoformat(),
            "verify_history": {
                "last_5_runs": [
                    {
                        "timestamp": r.get("timestamp"),
                        "tier": r.get("tier"),
                        "exit_code": r.get("exit_code"),
                        "duration_s": r.get("total_duration_s"),
                    }
                    for r in last_5_runs
                ],
                "last_30_days": {
                    "total": len(last_30d_runs),
                    "passed": pass_count,
                    "failed": fail_count,
                },
            },
            "contract_evidence": {
                "total_artifacts": len(contract_evidence),
                "stale_artifacts": len(stale_artifacts),
                "by_prefix": dict(evidence_by_prefix),
            },
            "suppressions": {
                "total_discovered": len(suppressions),
                "by_kind": dict(suppression_by_kind),
                "unregistered": len(unregistered_suppressions),
            },
            "witnesses": {
                "total": len(witnesses),
                "stale": len(stale_witnesses),
                "stale_names": witness_names,
            },
            "benchmark_campaigns": {
                "total_campaigns": len(latest_campaigns),
                "campaigns": [
                    {"name": name, "latest_run": mtime.isoformat()} for name, mtime in sorted(latest_campaigns.items())
                ],
            },
            "blocking": False,
        }
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        # Human-readable output
        print(f"Evidence Report ({now.isoformat()})")
        print()

        # Verify history section
        print("VERIFY HISTORY (last 5 runs)")
        if last_5_runs:
            for run in reversed(last_5_runs[-5:]):
                ts = _format_timestamp(run.get("timestamp"))
                tier = run.get("tier", "unknown")[:5].ljust(5)
                exit_code = run.get("exit_code", -1)
                status = "pass" if exit_code == 0 else "fail"
                duration = run.get("total_duration_s", 0)
                print(f"  {ts} {tier} {status}  {duration:.2f}s")
        else:
            print("  (no history)")
        print()

        # Contract evidence section
        print(f"CONTRACT EVIDENCE ({len(contract_evidence)} artifacts, {len(stale_artifacts)} stale)")
        for prefix in sorted(evidence_by_prefix.keys()):
            count = evidence_by_prefix[prefix]
            print(f"  {prefix}: {count} artifacts")
        print()

        # Suppressions section
        print("SUPPRESSIONS")
        total_supp = len(suppressions)
        if total_supp > 0:
            kind_str = " ".join(f"{kind}={count}" for kind, count in sorted(suppression_by_kind.items()))
            print(f"  {total_supp} discovered: {kind_str}")
            print(f"  {len(unregistered_suppressions)} unregistered")
        else:
            print("  (none discovered)")
        print()

        # Witnesses section
        print(f"WITNESSES ({len(witnesses)} total, {len(stale_witnesses)} stale)")
        if stale_witnesses:
            for name in sorted(witness_names):
                print(f"  {name}")
        else:
            print("  All fresh")
        print()

        # Benchmark campaigns section
        print("BENCHMARK CAMPAIGNS")
        if latest_campaigns:
            for name in sorted(latest_campaigns.keys()):
                mtime = latest_campaigns[name]
                print(f"  {name}: {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("  No local campaign runs found (run devtools benchmark-campaign run <name>)")
        print()

        print("blocking=False")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
