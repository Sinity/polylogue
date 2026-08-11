"""Verify mutation-campaign result freshness and kill rate.

Reads the executable mutation-campaign catalog and checks, for every
authored campaign, whether a recent run artifact
exists under the campaign's artifact glob (default
``.local/mutation-campaigns/<name>/*.json``).

Reports three classes of finding:

* ``missing``  — campaign has no run artifact at all.
* ``stale``    — newest artifact is older than the selected freshness budget.
* ``unknown``  — campaign artifact references a name not present in
  the executable catalog. Surfaced so artifact directories don't silently fork
  away from the registry.

Default behavior is **soft**: the command exits 0 and reports missing or stale
artifacts as warnings. Pass ``--strict`` when an operator intentionally requires
a complete recent campaign set. Rotating CI uses ``--enforce-kill-rate`` so it
gates the campaigns actually run in that job without pretending the absent
campaigns ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from devtools import repo_root as _get_root
from devtools.mutation_catalog import build_mutation_entries

ROOT = _get_root()
DEFAULT_FRESHNESS_DAYS = 60
DEFAULT_ARTIFACT_GLOB = ".local/mutation-campaigns/{name}/*.json"
# Conservative kill-rate floor (#1733 AC2/AC3). Mutation kill rates for
# well-tested modules sit well above this; 0.5 flags a genuinely under-killed
# module without false-alarming on a healthy campaign. Only enforced under --enforce-kill-rate
# and only against fresh campaigns (those that actually have a recent artifact).
DEFAULT_MIN_KILL_RATE = 0.5


@dataclass(frozen=True)
class CampaignFreshness:
    name: str
    freshness_days: int
    artifact_glob: str
    artifact_count: int
    newest_artifact: str | None
    newest_created_at: str | None
    newest_age_days: float | None
    kill_rate: float | None
    min_kill_rate: float | None
    counts: dict[str, int]
    state: str  # "fresh" | "stale" | "missing"


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
    name: str,
    *,
    repo_root: Path,
    now: datetime,
    freshness_days: int,
    min_kill_rate: float | None = None,
) -> CampaignFreshness:
    glob = DEFAULT_ARTIFACT_GLOB.format(name=name)
    artifacts = _resolve_artifacts(repo_root, glob)
    if not artifacts:
        return CampaignFreshness(
            name=name,
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
    """Names appearing under .local/mutation-campaigns/ but not in the catalog."""
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


def catalog_entries() -> list[str]:
    """Project executable campaigns, without a second declarative registry."""
    return [entry.name for entry in build_mutation_entries()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--default-freshness-days",
        type=int,
        default=DEFAULT_FRESHNESS_DAYS,
        help=f"Freshness budget for campaign artifacts (default {DEFAULT_FRESHNESS_DAYS}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any campaign is missing or stale. Default is soft (always exit 0).",
    )
    parser.add_argument(
        "--enforce-kill-rate",
        action="store_true",
        help="Exit non-zero when a fresh campaign's kill rate is below --default-min-kill-rate.",
    )
    parser.add_argument(
        "--default-min-kill-rate",
        type=float,
        default=None,
        help=(f"Kill-rate floor for every fresh campaign (default {DEFAULT_MIN_KILL_RATE})."),
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report instead of human output.")
    args = parser.parse_args(argv)

    entries = catalog_entries()
    default_min_kill_rate = (
        args.default_min_kill_rate if args.default_min_kill_rate is not None else DEFAULT_MIN_KILL_RATE
    )

    now = datetime.now(UTC)
    assessments = [
        assess_campaign(
            name,
            repo_root=ROOT,
            now=now,
            freshness_days=args.default_freshness_days,
            min_kill_rate=default_min_kill_rate,
        )
        for name in entries
    ]

    registered_names = [a.name for a in assessments]
    orphan_names = _orphan_artifact_names(ROOT, registered_names)

    missing = [a for a in assessments if a.state == "missing"]
    stale = [a for a in assessments if a.state == "stale"]
    fresh = [a for a in assessments if a.state == "fresh"]
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
        print(f"registered active mutation campaigns: {len(assessments)}")
        print(f"  fresh:   {len(fresh)}")
        print(f"  stale:   {len(stale)} (older than freshness_days)")
        print(f"  missing: {len(missing)} (no run artifact)")
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
            print(f"[warn] orphan artifact directories (not in catalog): {len(orphan_names)}")
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
