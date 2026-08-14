"""Scoped mutmut campaign runner with isolated workspace state.

This runner exists because mutmut's default on-disk cache is global to the
current worktree. Once the test inventory changes, focused reruns can fall back
to broad stats collection and effectively stop being focused. Campaigns here
solve that by:

1. copying the current worktree into an isolated temporary workspace,
2. narrowing `paths_to_mutate` and test selection in that workspace only,
3. running mutmut there,
4. exporting a durable summary artifact.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from devtools import repo_root as _get_root

from .mutation_scenario_catalog import MUTATION_CAMPAIGNS, MutationCampaign
from .verify_mutation_freshness import (
    assess_campaign as _freshness_assess,
)

ROOT = _get_root()
CAMPAIGN_ARTIFACT_DIR = Path(".local/mutation-campaigns")


def default_artifact_path(campaign_name: str, created_at: datetime) -> Path:
    """Return the default JSON artifact path for a campaign run.

    Layout: ``.local/mutation-campaigns/<campaign>/<timestamp>.json``. It is
    consumed by ``verify-mutation-freshness`` and ``mutmut-campaign status``.
    """
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    base = CAMPAIGN_ARTIFACT_DIR / campaign_name / stamp
    return base.with_suffix(".json")


STATUS_IGNORE_PREFIXES = (f"{CAMPAIGN_ARTIFACT_DIR.as_posix()}/",)
DEFAULT_IGNORE_PATTERNS = shutil.ignore_patterns(
    ".git",
    ".direnv",
    ".venv",
    ".cache",
    ".local",
    "mutants",
    "__pycache__",
    ".claude",
)
CAMPAIGNS = MUTATION_CAMPAIGNS


@dataclass(frozen=True)
class CampaignResult:
    campaign: str
    description: str
    commit: str
    worktree_dirty: bool
    status_summary: list[str]
    created_at: str
    workspace: str
    command: list[str]
    paths_to_mutate: list[str]
    tests: list[str]
    counts: dict[str, int]
    dominant_survivors: list[tuple[str, int]]
    dominant_timeouts: list[tuple[str, int]]
    dominant_not_checked: list[tuple[str, int]]
    survivor_keys: list[str]
    timeout_keys: list[str]
    not_checked_keys: list[str]
    runtime_seconds: float
    exit_code: int
    notes: list[str]
    origin: str = "authored"
    path_targets: list[str] = field(default_factory=list)
    artifact_targets: list[str] = field(default_factory=list)
    operation_targets: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def _toml_list(items: tuple[str, ...] | list[str]) -> str:
    rendered = ", ".join(json.dumps(item) for item in items)
    return f"[{rendered}]"


def patch_mutmut_section(
    pyproject_text: str,
    *,
    paths_to_mutate: tuple[str, ...],
    tests: tuple[str, ...],
) -> str:
    """Patch the [tool.mutmut] section in a pyproject.toml text blob."""
    lines = pyproject_text.splitlines()
    result: list[str] = []
    in_mutmut = False
    skip_assignment = False
    keys_to_replace = {
        "paths_to_mutate",
        "pytest_add_cli_args_test_selection",
        "tests_dir",
    }
    inserted = False

    for line in lines:
        stripped = line.strip()
        if stripped == "[tool.mutmut]":
            in_mutmut = True
            inserted = False
            result.append(line)
            continue

        if in_mutmut and stripped.startswith("[") and stripped != "[tool.mutmut]":
            if not inserted:
                result.extend(
                    [
                        f"paths_to_mutate = {_toml_list(paths_to_mutate)}",
                        f"pytest_add_cli_args_test_selection = {_toml_list(tests)}",
                        "tests_dir = []",
                    ]
                )
                inserted = True
            in_mutmut = False

        if in_mutmut:
            if skip_assignment:
                if "]" in line:
                    skip_assignment = False
                continue
            for key in keys_to_replace:
                if stripped.startswith(f"{key} ="):
                    skip_assignment = "]" not in line
                    break
            else:
                result.append(line)
            continue

        result.append(line)

    if in_mutmut and not inserted:
        result.extend(
            [
                f"paths_to_mutate = {_toml_list(paths_to_mutate)}",
                f"pytest_add_cli_args_test_selection = {_toml_list(tests)}",
                "tests_dir = []",
            ]
        )

    return "\n".join(result) + "\n"


def copy_workspace(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=DEFAULT_IGNORE_PATTERNS,
        dirs_exist_ok=True,
        symlinks=True,
        ignore_dangling_symlinks=True,
    )


def git_commit_sha(cwd: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        text=True,
    ).strip()


def git_status_summary(cwd: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=cwd,
        text=True,
    )
    return [
        line.rstrip() for line in output.splitlines() if line.strip() and not _status_line_is_ignored(line.rstrip())
    ]


def _status_line_path(status_line: str) -> str:
    payload = status_line[3:] if len(status_line) > 3 else status_line
    if " -> " in payload:
        payload = payload.split(" -> ", 1)[1]
    return payload.strip()


def _status_line_is_ignored(status_line: str) -> bool:
    path = _status_line_path(status_line)
    return any(path.startswith(prefix) for prefix in STATUS_IGNORE_PREFIXES)


def _status_from_exit_code(code: int) -> str:
    return {
        0: "survived",
        1: "killed",
        -24: "timeout",
        5: "suspicious",
        7: "skipped",
    }.get(code, "not_checked")


def _function_from_key(mutant_key: str) -> str:
    parts = mutant_key.split("ǁ")
    if len(parts) >= 3:
        return parts[2].split("__mutmut_", 1)[0]
    return mutant_key


def _iter_meta_paths(mutants_dir: Path) -> list[Path]:
    return sorted(mutants_dir.rglob("*.meta"))


def summarize_mutmut_results(
    mutants_dir: Path, prefixes: tuple[str, ...]
) -> tuple[dict[str, int], Counter[str], Counter[str], Counter[str], list[str], list[str], list[str]]:
    """Summarize mutmut results for matching mutant prefixes."""
    counts: Counter[str] = Counter()
    survivors: Counter[str] = Counter()
    timeouts: Counter[str] = Counter()
    not_checked: Counter[str] = Counter()
    survivor_keys: list[str] = []
    timeout_keys: list[str] = []
    not_checked_keys: list[str] = []

    for meta_path in _iter_meta_paths(mutants_dir):
        try:
            payload = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        for mutant_key, code in payload.get("exit_code_by_key", {}).items():
            if not prefixes or any(fnmatch.fnmatch(mutant_key, prefix) for prefix in prefixes):
                status = "not_checked" if code is None else _status_from_exit_code(int(code))
                counts[status] += 1
                function_name = _function_from_key(mutant_key)
                if status == "survived":
                    survivors[function_name] += 1
                    survivor_keys.append(mutant_key)
                elif status == "timeout":
                    timeouts[function_name] += 1
                    timeout_keys.append(mutant_key)
                elif status == "not_checked":
                    not_checked[function_name] += 1
                    not_checked_keys.append(mutant_key)

    for expected in ("killed", "survived", "timeout", "not_checked", "suspicious", "skipped"):
        counts.setdefault(expected, 0)

    return (
        dict(counts),
        survivors,
        timeouts,
        not_checked,
        survivor_keys,
        timeout_keys,
        not_checked_keys,
    )


def write_json_artifact(result: CampaignResult, *, json_out: Path | None) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(asdict(result), indent=2) + "\n")


def run_campaign(
    campaign: MutationCampaign,
    *,
    repo_root: Path,
    json_out: Path | None,
    keep_workspace: bool,
) -> CampaignResult:
    commit = git_commit_sha(repo_root)
    status_summary = git_status_summary(repo_root)
    created_at = datetime.now(UTC).isoformat()
    prefixes = tuple(f"{path.replace('/', '.').removesuffix('.py')}*" for path in campaign.paths_to_mutate)

    tmp_prefix = f"mutmut-{campaign.name}-"
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if keep_workspace:
        workspace = Path(tempfile.mkdtemp(prefix=tmp_prefix))
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix=tmp_prefix)
        workspace = Path(temp_dir_obj.name)

    try:
        workspace_repo = workspace / "repo"
        copy_workspace(repo_root, workspace_repo)
        pyproject_path = workspace_repo / "pyproject.toml"
        pyproject_path.write_text(
            patch_mutmut_section(
                pyproject_path.read_text(),
                paths_to_mutate=campaign.paths_to_mutate,
                tests=campaign.tests,
            )
        )

        command = ["mutmut", "run"]
        started = datetime.now(UTC)
        completed = subprocess.run(
            command,
            cwd=workspace_repo,
            text=True,
            capture_output=False,
        )
        runtime = (datetime.now(UTC) - started).total_seconds()

        (
            counts,
            survivors,
            timeouts,
            not_checked,
            survivor_keys,
            timeout_keys,
            not_checked_keys,
        ) = summarize_mutmut_results(
            workspace_repo / "mutants",
            prefixes,
        )
        result = CampaignResult(
            campaign=campaign.name,
            description=campaign.description,
            commit=commit,
            worktree_dirty=bool(status_summary),
            status_summary=status_summary,
            created_at=created_at,
            workspace=str(workspace_repo),
            command=command,
            paths_to_mutate=list(campaign.paths_to_mutate),
            tests=list(campaign.tests),
            counts=counts,
            dominant_survivors=survivors.most_common(10),
            dominant_timeouts=timeouts.most_common(10),
            dominant_not_checked=not_checked.most_common(10),
            survivor_keys=survivor_keys,
            timeout_keys=timeout_keys,
            not_checked_keys=not_checked_keys,
            runtime_seconds=runtime,
            exit_code=completed.returncode,
            notes=list(campaign.notes),
            origin="authored.mutation-campaign" if campaign.origin == "authored" else campaign.origin,
            path_targets=list(campaign.path_targets),
            artifact_targets=list(campaign.artifact_targets),
            operation_targets=list(campaign.operation_targets),
            tags=list(campaign.tags or ("mutation",)),
        )
        write_json_artifact(result, json_out=json_out)
        return result
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available campaigns")
    list_parser.set_defaults(command_fn=cmd_list)

    run_parser = subparsers.add_parser("run", help="Run one isolated mutation campaign")
    run_parser.add_argument("campaign", choices=sorted(CAMPAIGNS))
    run_parser.add_argument("--json-out", type=Path)
    run_parser.add_argument("--keep-workspace", action="store_true")
    run_parser.set_defaults(command_fn=cmd_run)

    status_parser = subparsers.add_parser(
        "status",
        help="Show per-campaign last-run, kill rate, and freshness from the executable catalog.",
    )
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument(
        "--default-freshness-days",
        type=int,
        default=60,
        help="Freshness budget for campaign artifacts (default 60).",
    )
    status_parser.set_defaults(command_fn=cmd_status)

    return parser.parse_args(argv)


def cmd_list(_args: argparse.Namespace) -> int:
    for campaign in CAMPAIGNS.values():
        print(f"{campaign.name}: {campaign.description}")
        print(f"  paths: {', '.join(campaign.paths_to_mutate)}")
        print(f"  tests: {', '.join(campaign.tests)}")
        if campaign.notes:
            for note in campaign.notes:
                print(f"  note: {note}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    campaign = CAMPAIGNS[args.campaign]
    if args.json_out is None:
        # Default per-campaign artifact layout so freshness lint and status
        # readouts always have something to discover.
        json_out = ROOT / default_artifact_path(campaign.name, datetime.now(UTC))
    else:
        json_out = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
    result = run_campaign(
        campaign,
        repo_root=ROOT,
        json_out=json_out,
        keep_workspace=args.keep_workspace,
    )
    print(json.dumps(asdict(result), indent=2))
    return result.exit_code


def cmd_status(args: argparse.Namespace) -> int:
    now = datetime.now(UTC)
    assessments = [
        _freshness_assess(
            campaign.name,
            repo_root=ROOT,
            now=now,
            freshness_days=args.default_freshness_days,
        )
        for campaign in CAMPAIGNS.values()
    ]
    if args.json:
        json.dump(
            {"campaigns": [a.__dict__ for a in assessments]},
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
        return 0
    print(f"{'campaign':<24} {'state':<8} {'age':>10} {'budget':>8} {'kill':>8} artifact")
    for a in assessments:
        age = "n/a" if a.newest_age_days is None else f"{a.newest_age_days:.1f}d"
        budget = f"{a.freshness_days}d"
        kill = "n/a" if a.kill_rate is None else f"{a.kill_rate * 100:.1f}%"
        artifact = a.newest_artifact or "-"
        print(f"{a.name:<24} {a.state:<8} {age:>10} {budget:>8} {kill:>8} {artifact}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.command_fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
