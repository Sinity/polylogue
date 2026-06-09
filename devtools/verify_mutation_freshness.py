"""Verify mutation-campaign coverage discipline (#1304).

Reads ``docs/plans/campaign-coverage.yaml`` and checks, for every
``status: active`` mutation campaign, whether a recent run artifact
exists under the campaign's artifact glob (default
``.local/mutation-campaigns/<name>/*.json``).

Reports three classes of finding:

* ``missing``  — campaign has no run artifact at all.
* ``stale``    — newest artifact is older than ``freshness_days``
  (defaults to 60 when the entry omits it).
* ``unknown``  — campaign artifact references a name not present in
  the manifest. Surfaced so artifact directories don't silently fork
  away from the registry.

Default behavior is **soft**: the command always exits 0 and reports
findings as warnings, so ``devtools verify`` can include it without
gating on local mutation-run cadence. Pass ``--strict`` to fail when
any campaign is missing or stale — intended for nightly jobs and
``devtools verify --lab``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from devtools import repo_root as _get_root

ROOT = _get_root()
MANIFEST = ROOT / "docs" / "plans" / "campaign-coverage.yaml"
DEFAULT_FRESHNESS_DAYS = 60
DEFAULT_ARTIFACT_GLOB = ".local/mutation-campaigns/{name}/*.json"
# Conservative kill-rate floor (#1733 AC2/AC3). Mutation kill rates for
# well-tested modules sit well above this; 0.5 flags a genuinely under-killed
# module without false-alarming on a healthy campaign. Ratchet up per-entry in
# the manifest as real run data accrues. Only enforced under --enforce-kill-rate
# and only against fresh campaigns (those that actually have a recent artifact).
DEFAULT_MIN_KILL_RATE = 0.5


@dataclass(frozen=True)
class CampaignFreshness:
    name: str
    status: str
    freshness_days: int
    artifact_glob: str
    artifact_count: int
    newest_artifact: str | None
    newest_created_at: str | None
    newest_age_days: float | None
    kill_rate: float | None
    min_kill_rate: float | None
    counts: dict[str, int]
    state: str  # "fresh" | "stale" | "missing" | "inactive"


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return default


def _coerce_float(value: object, default: float | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _entry_glob(entry: dict[str, object]) -> str:
    glob = entry.get("artifact_glob")
    if not isinstance(glob, str) or not glob.strip():
        glob = DEFAULT_ARTIFACT_GLOB.format(name=entry["name"])
    return glob


def _resolve_artifacts(repo_root: Path, glob: str) -> list[Path]:
    return sorted(repo_root.glob(glob))


def _load_summary(path: Path) -> tuple[str | None, dict[str, int]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, {}
    created_at = payload.get("created_at")
    counts_raw = payload.get("counts", {})
    counts = {str(k): int(v) for k, v in counts_raw.items() if isinstance(v, int)}
    return (created_at if isinstance(created_at, str) else None), counts


def _kill_rate(counts: dict[str, int]) -> float | None:
    total = sum(v for k, v in counts.items() if k in {"killed", "survived", "timeout", "suspicious"})
    if total <= 0:
        return None
    return counts.get("killed", 0) / total


def _age_days(created_at: str | None, now: datetime) -> float | None:
    if not created_at:
        return None
    try:
        recorded = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=UTC)
    return (now - recorded).total_seconds() / 86400.0


def assess_campaign(
    entry: dict[str, object],
    *,
    repo_root: Path,
    now: datetime,
    default_freshness_days: int,
    default_min_kill_rate: float | None = None,
) -> CampaignFreshness:
    name = str(entry["name"])
    status = str(entry.get("status", "active"))
    freshness_days = _coerce_int(entry.get("freshness_days"), default_freshness_days)
    min_kill_rate = _coerce_float(entry.get("min_kill_rate"), default_min_kill_rate)
    glob = _entry_glob(entry)
    artifacts = _resolve_artifacts(repo_root, glob)
    if status != "active":
        return CampaignFreshness(
            name=name,
            status=status,
            freshness_days=freshness_days,
            artifact_glob=glob,
            artifact_count=len(artifacts),
            newest_artifact=None,
            newest_created_at=None,
            newest_age_days=None,
            kill_rate=None,
            min_kill_rate=min_kill_rate,
            counts={},
            state="inactive",
        )
    if not artifacts:
        return CampaignFreshness(
            name=name,
            status=status,
            freshness_days=freshness_days,
            artifact_glob=glob,
            artifact_count=0,
            newest_artifact=None,
            newest_created_at=None,
            newest_age_days=None,
            kill_rate=None,
            min_kill_rate=min_kill_rate,
            counts={},
            state="missing",
        )
    # Use most recent by created_at if available, else mtime.
    by_recency = sorted(
        artifacts,
        key=lambda p: (_load_summary(p)[0] or "", p.stat().st_mtime),
        reverse=True,
    )
    newest = by_recency[0]
    created_at, counts = _load_summary(newest)
    age = _age_days(created_at, now)
    if age is None:
        # Fall back to mtime if artifact created_at missing/unparseable.
        age = (now.timestamp() - newest.stat().st_mtime) / 86400.0
    state = "stale" if age > freshness_days else "fresh"
    return CampaignFreshness(
        name=name,
        status=status,
        freshness_days=freshness_days,
        artifact_glob=glob,
        artifact_count=len(artifacts),
        newest_artifact=newest.relative_to(repo_root).as_posix(),
        newest_created_at=created_at,
        newest_age_days=age,
        kill_rate=_kill_rate(counts),
        min_kill_rate=min_kill_rate,
        counts=counts,
        state=state,
    )


def _orphan_artifact_names(repo_root: Path, registered: Iterable[str]) -> list[str]:
    """Names appearing under .local/mutation-campaigns/ but not in the manifest."""
    base = repo_root / ".local" / "mutation-campaigns"
    if not base.is_dir():
        return []
    registered_set = set(registered)
    orphans: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name in registered_set:
            continue
        # Only count as orphan if it actually contains run artifacts.
        if any(child.glob("*.json")):
            orphans.append(name)
    return orphans


def load_manifest(path: Path) -> dict[str, object]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping at root")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=MANIFEST)
    parser.add_argument(
        "--default-freshness-days",
        type=int,
        default=DEFAULT_FRESHNESS_DAYS,
        help=f"Freshness budget for entries without freshness_days (default {DEFAULT_FRESHNESS_DAYS}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any campaign is missing or stale. Default is soft (always exit 0).",
    )
    parser.add_argument(
        "--enforce-kill-rate",
        action="store_true",
        help=(
            "Exit non-zero when a fresh campaign's kill rate is below its "
            "min_kill_rate threshold (per-entry, else --default-min-kill-rate)."
        ),
    )
    parser.add_argument(
        "--default-min-kill-rate",
        type=float,
        default=None,
        help=(
            "Kill-rate floor for entries without min_kill_rate. Defaults to the "
            f"manifest's top-level default_min_kill_rate, else {DEFAULT_MIN_KILL_RATE}."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report instead of human output.")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.yaml)
    entries = manifest.get("mutation_campaigns")
    if not isinstance(entries, list):
        print(f"{args.yaml}: missing or invalid mutation_campaigns list", file=sys.stderr)
        return 2

    default_min_kill_rate = (
        args.default_min_kill_rate
        if args.default_min_kill_rate is not None
        else _coerce_float(manifest.get("default_min_kill_rate"), DEFAULT_MIN_KILL_RATE)
    )

    now = datetime.now(UTC)
    assessments = [
        assess_campaign(
            entry,
            repo_root=ROOT,
            now=now,
            default_freshness_days=args.default_freshness_days,
            default_min_kill_rate=default_min_kill_rate,
        )
        for entry in entries
        if isinstance(entry, dict) and "name" in entry
    ]

    registered_names = [a.name for a in assessments]
    orphan_names = _orphan_artifact_names(ROOT, registered_names)

    missing = [a for a in assessments if a.state == "missing"]
    stale = [a for a in assessments if a.state == "stale"]
    fresh = [a for a in assessments if a.state == "fresh"]
    inactive = [a for a in assessments if a.state == "inactive"]
    below_threshold = [
        a for a in fresh if a.kill_rate is not None and a.min_kill_rate is not None and a.kill_rate < a.min_kill_rate
    ]

    blocking = (args.strict and (bool(missing) or bool(stale))) or (args.enforce_kill_rate and bool(below_threshold))

    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "strict": bool(args.strict),
                "default_freshness_days": args.default_freshness_days,
                "enforce_kill_rate": bool(args.enforce_kill_rate),
                "default_min_kill_rate": default_min_kill_rate,
                "counts": {
                    "registered": len(assessments),
                    "fresh": len(fresh),
                    "stale": len(stale),
                    "missing": len(missing),
                    "inactive": len(inactive),
                    "below_kill_threshold": len(below_threshold),
                    "orphan_artifact_names": len(orphan_names),
                },
                "campaigns": [a.__dict__ for a in assessments],
                "below_kill_threshold": [a.name for a in below_threshold],
                "orphan_artifact_names": orphan_names,
            },
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
    else:
        prefix = "[BLOCK]" if args.strict else "[warn]"
        print(f"registered active mutation campaigns: {len(assessments) - len(inactive)}")
        print(f"  fresh:   {len(fresh)}")
        print(f"  stale:   {len(stale)} (older than freshness_days)")
        print(f"  missing: {len(missing)} (no run artifact)")
        if inactive:
            print(f"  inactive (skipped): {len(inactive)}")
        for a in missing:
            print(f"{prefix} missing artifact: {a.name} (glob={a.artifact_glob})")
        for a in stale:
            assert a.newest_age_days is not None
            print(
                f"{prefix} stale: {a.name} "
                f"newest={a.newest_artifact} "
                f"age={a.newest_age_days:.1f}d (budget {a.freshness_days}d)"
            )
        kill_prefix = "[BLOCK]" if args.enforce_kill_rate else "[warn]"
        for a in below_threshold:
            assert a.kill_rate is not None and a.min_kill_rate is not None
            print(
                f"{kill_prefix} kill rate below threshold: {a.name} "
                f"kill={a.kill_rate * 100:.1f}% (floor {a.min_kill_rate * 100:.1f}%)"
            )
        if orphan_names:
            print(f"[warn] orphan artifact directories (not in manifest): {len(orphan_names)}")
            for name in orphan_names[:25]:
                print(f"    {name}")
        if fresh:
            print(f"fresh campaigns: {len(fresh)}")
            for a in fresh[:5]:
                kr = "n/a" if a.kill_rate is None else f"{a.kill_rate * 100:.1f}%"
                age = "n/a" if a.newest_age_days is None else f"{a.newest_age_days:.1f}d"
                print(f"    {a.name}: kill={kr} age={age}")
            if len(fresh) > 5:
                print(f"    ... and {len(fresh) - 5} more")
        print()
        print(f"blocking={blocking} (strict={args.strict})")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
