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
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_ARTIFACT_DIR = Path("docs/mutation-campaigns")
STATUS_IGNORE_PREFIXES = (
    f"{CAMPAIGN_ARTIFACT_DIR.as_posix()}/",
)
DEFAULT_IGNORE_PATTERNS = shutil.ignore_patterns(
    ".git",
    ".direnv",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".benchmarks",
    "mutants",
    "__pycache__",
    ".claude",
    "qa_archive",
    "qa_outputs",
    "qa_2026-03-10",
    "demos/output",
)


@dataclass(frozen=True)
class Campaign:
    name: str
    description: str
    paths_to_mutate: tuple[str, ...]
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()


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


CAMPAIGNS: dict[str, Campaign] = {
    "filters": Campaign(
        name="filters",
        description="ConversationFilter semantics and summary/picker contracts",
        paths_to_mutate=("polylogue/lib/filters.py",),
        tests=(
            "tests/unit/core/test_filters.py",
            "tests/unit/core/test_filters_adv.py",
            "tests/unit/core/test_filters_props.py",
        ),
        notes=(
            "Targets the historical largest no-test blind spot.",
            "Timeout tail is expected in filter pipeline helpers.",
        ),
    ),
    "models": Campaign(
        name="models",
        description="Message/Conversation semantic helpers and pairing logic",
        paths_to_mutate=("polylogue/lib/models.py",),
        tests=(
            "tests/unit/core/test_models.py",
            "tests/unit/core/test_message_laws.py",
        ),
    ),
    "json": Campaign(
        name="json",
        description="JSON serialization and parser laws",
        paths_to_mutate=("polylogue/lib/json.py",),
        tests=(
            "tests/unit/core/test_json.py",
            "tests/unit/core/test_json_laws.py",
        ),
    ),
    "hybrid": Campaign(
        name="hybrid",
        description="Hybrid search fusion and ranked-conversation resolution",
        paths_to_mutate=("polylogue/storage/search_providers/hybrid.py",),
        tests=(
            "tests/unit/storage/test_hybrid.py",
            "tests/unit/storage/test_hybrid_laws.py",
        ),
    ),
    "fts5": Campaign(
        name="fts5",
        description="FTS5 query escaping and conversation search semantics",
        paths_to_mutate=("polylogue/storage/search_providers/fts5.py",),
        tests=(
            "tests/unit/storage/test_fts5.py",
            "tests/unit/storage/test_fts5_laws.py",
        ),
    ),
    "schema-core": Campaign(
        name="schema-core",
        description="Schema generation, privacy, verification, and safety contracts",
        paths_to_mutate=(
            "polylogue/schemas/schema_inference.py",
            "polylogue/schemas/validator.py",
            "polylogue/schemas/verification.py",
        ),
        tests=(
            "tests/unit/core/test_schema.py",
            "tests/unit/core/test_schema_laws.py",
            "tests/unit/core/test_schema_privacy.py",
            "tests/unit/core/test_schema_verification.py",
            "tests/unit/storage/test_schema_safety.py",
        ),
        notes=("Larger campaign; use when law and privacy work are stable.",),
    ),
    "schema-inference": Campaign(
        name="schema-inference",
        description="Schema inference and privacy heuristics",
        paths_to_mutate=("polylogue/schemas/schema_inference.py",),
        tests=(
            "tests/unit/core/test_schema.py",
            "tests/unit/core/test_schema_laws.py",
            "tests/unit/core/test_schema_privacy.py",
        ),
    ),
    "schema-validation": Campaign(
        name="schema-validation",
        description="Schema validator and verification contracts",
        paths_to_mutate=(
            "polylogue/schemas/validator.py",
            "polylogue/schemas/verification.py",
        ),
        tests=(
            "tests/unit/core/test_schema.py",
            "tests/unit/core/test_schema_laws.py",
            "tests/unit/core/test_schema_verification.py",
            "tests/unit/storage/test_schema_safety.py",
        ),
    ),
    "pipeline-services": Campaign(
        name="pipeline-services",
        description="Acquire/validate/parse planning and stage contracts",
        paths_to_mutate=("polylogue/pipeline/services",),
        tests=(
            "tests/unit/pipeline/test_services.py",
            "tests/unit/pipeline/test_service_laws.py",
        ),
        notes=("Likely to need more helper-level laws to reduce timeout noise.",),
    ),
    "cli-query": Campaign(
        name="cli-query",
        description="Query command planning, action routing, and summary output contracts",
        paths_to_mutate=(
            "polylogue/cli/query.py",
            "polylogue/cli/query_plan.py",
            "polylogue/cli/query_actions.py",
            "polylogue/cli/query_output.py",
        ),
        tests=(
            "tests/unit/cli/test_query.py",
            "tests/unit/cli/test_query_exec.py",
            "tests/unit/cli/test_query_exec_laws.py",
            "tests/unit/cli/test_query_fmt.py",
            "tests/unit/cli/test_query_plan.py",
        ),
    ),
    "cli-run": Campaign(
        name="cli-run",
        description="Run command execution, display, and watch contracts",
        paths_to_mutate=("polylogue/cli/commands/run.py",),
        tests=(
            "tests/unit/cli/test_run.py",
            "tests/unit/cli/test_run_int.py",
            "tests/unit/cli/test_run_laws.py",
        ),
    ),
    "site-builder": Campaign(
        name="site-builder",
        description="Static-site builder and CLI archive contracts",
        paths_to_mutate=(
            "polylogue/site/builder.py",
            "polylogue/cli/commands/site.py",
        ),
        tests=(
            "tests/integration/test_site.py",
            "tests/integration/test_site_laws.py",
        ),
    ),
    "drive-client": Campaign(
        name="drive-client",
        description="Drive auth, transport, JSON payload parsing, and ingest attachment contracts",
        paths_to_mutate=("polylogue/sources/drive_client.py",),
        tests=(
            "tests/unit/sources/test_drive_client_laws.py",
            "tests/unit/sources/test_drive_utils.py",
            "tests/unit/sources/test_drive_ops.py",
        ),
        notes=("Targets the historical Drive not_checked cluster with focused tests.",),
    ),
    "repository": Campaign(
        name="repository",
        description="Repository query, projection, and CRUD contracts",
        paths_to_mutate=("polylogue/storage/repository.py",),
        tests=(
            "tests/unit/storage/test_repository_laws.py",
        ),
        notes=("Large surface; use to gauge storage law readiness before repository-law work.",),
    ),
    "source-detection": Campaign(
        name="source-detection",
        description="Source detection, sniffing, and parser dispatch",
        paths_to_mutate=("polylogue/sources/source.py",),
        tests=(
            "tests/unit/sources/test_source_laws.py",
            "tests/unit/sources/test_parsers.py",
            "tests/unit/sources/test_parsers_base.py",
            "tests/unit/sources/test_parser_misc.py",
            "tests/unit/sources/test_parser_edge.py",
            "tests/unit/sources/test_parsers_chatgpt.py",
            "tests/unit/sources/test_parsers_props.py",
            "tests/unit/sources/test_parsers_drive.py",
        ),
    ),
    "providers-semantics": Campaign(
        name="providers-semantics",
        description="Provider semantic extraction, harmonization, and viewport contracts",
        paths_to_mutate=(
            "polylogue/sources/providers",
            "polylogue/schemas/unified.py",
        ),
        tests=(
            "tests/unit/sources/test_parse_laws.py",
            "tests/unit/sources/test_harmonization_contracts.py",
            "tests/unit/sources/test_extraction.py",
            "tests/unit/sources/test_unified_semantic_laws.py",
            "tests/unit/sources/test_null_guard_properties.py",
            "tests/unit/sources/test_models.py",
        ),
        notes=("Directly relevant to the next provider-law wave.",),
    ),
    "sources-parse": Campaign(
        name="sources-parse",
        description="Provider detection, parsing, harmonization, and parser laws",
        paths_to_mutate=(
            "polylogue/sources",
            "polylogue/schemas/unified.py",
        ),
        tests=(
            "tests/unit/sources/test_parse_laws.py",
            "tests/unit/sources/test_parsers_props.py",
            "tests/unit/sources/test_harmonization_contracts.py",
            "tests/unit/sources/test_source_laws.py",
            "tests/unit/sources/test_extraction.py",
            "tests/unit/sources/test_unified_semantic_laws.py",
            "tests/unit/sources/test_parsers.py",
            "tests/unit/sources/test_parsers_base.py",
            "tests/unit/sources/test_parser_misc.py",
            "tests/unit/sources/test_parser_edge.py",
            "tests/unit/sources/test_parsers_chatgpt.py",
            "tests/unit/sources/test_parsers_drive.py",
            "tests/unit/sources/test_drive_client_laws.py",
            "tests/unit/sources/test_drive_ops.py",
            "tests/unit/sources/test_drive_utils.py",
            "tests/unit/sources/test_null_guard_properties.py",
            "tests/unit/sources/test_models.py",
            "tests/unit/sources/test_token_store.py",
        ),
        notes=("Broadest campaign here; best run after law-wave work lands.",),
    ),
}


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
    shutil.copytree(src, dst, ignore=DEFAULT_IGNORE_PATTERNS, dirs_exist_ok=True)


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
        line.rstrip()
        for line in output.splitlines()
        if line.strip() and not _status_line_is_ignored(line.rstrip())
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


def format_markdown(result: CampaignResult) -> str:
    def render_table(rows: list[tuple[str, int]]) -> str:
        if not rows:
            return "| Function | Count |\n| --- | ---: |\n| _none_ | 0 |"
        body = "\n".join(f"| `{name}` | {count} |" for name, count in rows)
        return f"| Function | Count |\n| --- | ---: |\n{body}"

    lines = [
        f"# Mutmut Campaign: `{result.campaign}`",
        "",
        f"- Recorded on `{result.created_at}`",
        f"- Commit: `{result.commit}`",
        f"- Worktree dirty: `{'yes' if result.worktree_dirty else 'no'}`",
        f"- Description: {result.description}",
        f"- Workspace: `{result.workspace}`",
        f"- Command: `{' '.join(result.command)}`",
        "",
        "## Scope",
        "",
        f"- Mutated paths: {', '.join(f'`{path}`' for path in result.paths_to_mutate)}",
        f"- Selected tests: {', '.join(f'`{path}`' for path in result.tests)}",
        "",
        "## Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        f"| Killed | {result.counts.get('killed', 0)} |",
        f"| Survived | {result.counts.get('survived', 0)} |",
        f"| Timeout | {result.counts.get('timeout', 0)} |",
        f"| Not checked | {result.counts.get('not_checked', 0)} |",
        f"| Suspicious | {result.counts.get('suspicious', 0)} |",
        f"| Skipped | {result.counts.get('skipped', 0)} |",
        "",
        f"- Runtime: `{result.runtime_seconds:.2f}s`",
        f"- Exit code: `{result.exit_code}`",
        "",
        "## Dominant Survivors",
        "",
        render_table(result.dominant_survivors),
        "",
        "## Dominant Timeouts",
        "",
        render_table(result.dominant_timeouts),
        "",
        "## Dominant Not-Checked Clusters",
        "",
        render_table(result.dominant_not_checked),
    ]
    if result.survivor_keys:
        lines.extend(
            [
                "",
                "## Survivor Keys",
                "",
            ]
        )
        lines.extend(f"- `{key}`" for key in result.survivor_keys[:25])
        if len(result.survivor_keys) > 25:
            lines.append(f"- ... {len(result.survivor_keys) - 25} more")
    if result.timeout_keys:
        lines.extend(
            [
                "",
                "## Timeout Keys",
                "",
            ]
        )
        lines.extend(f"- `{key}`" for key in result.timeout_keys[:25])
        if len(result.timeout_keys) > 25:
            lines.append(f"- ... {len(result.timeout_keys) - 25} more")
    if result.not_checked_keys:
        lines.extend(
            [
                "",
                "## Not-Checked Keys",
                "",
            ]
        )
        lines.extend(f"- `{key}`" for key in result.not_checked_keys[:25])
        if len(result.not_checked_keys) > 25:
            lines.append(f"- ... {len(result.not_checked_keys) - 25} more")
    if result.status_summary:
        lines.extend(["", "## Source Worktree Status", ""])
        lines.extend(f"- `{line}`" for line in result.status_summary[:50])
        if len(result.status_summary) > 50:
            lines.append(f"- ... {len(result.status_summary) - 50} more")
    if result.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in result.notes)
    lines.append("")
    return "\n".join(lines)


def load_results(campaign_dir: Path) -> list[CampaignResult]:
    results: list[CampaignResult] = []
    for path in sorted(campaign_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        result = CampaignResult(
            campaign=payload["campaign"],
            description=payload["description"],
            commit=payload["commit"],
            worktree_dirty=bool(payload.get("worktree_dirty", False)),
            status_summary=list(payload.get("status_summary", [])),
            created_at=payload["created_at"],
            workspace=payload["workspace"],
            command=list(payload["command"]),
            paths_to_mutate=list(payload["paths_to_mutate"]),
            tests=list(payload["tests"]),
            counts=dict(payload["counts"]),
            dominant_survivors=[tuple(item) for item in payload["dominant_survivors"]],
            dominant_timeouts=[tuple(item) for item in payload["dominant_timeouts"]],
            dominant_not_checked=[tuple(item) for item in payload["dominant_not_checked"]],
            survivor_keys=list(payload.get("survivor_keys", [])),
            timeout_keys=list(payload.get("timeout_keys", [])),
            not_checked_keys=list(payload.get("not_checked_keys", [])),
            runtime_seconds=float(payload["runtime_seconds"]),
            exit_code=int(payload["exit_code"]),
            notes=list(payload.get("notes", [])),
        )
        results.append(result)
    return results


def latest_results_by_campaign(results: list[CampaignResult]) -> list[CampaignResult]:
    latest: dict[str, CampaignResult] = {}
    for result in results:
        existing = latest.get(result.campaign)
        if existing is None or result.created_at > existing.created_at:
            latest[result.campaign] = result
    return sorted(latest.values(), key=lambda result: result.campaign)


def format_index(results: list[CampaignResult]) -> str:
    latest = latest_results_by_campaign(results)
    lines = [
        "# Mutation Campaign Index",
        "",
        "Latest recorded artifact per campaign.",
        "",
        "| Campaign | Recorded | Commit | Killed | Survived | Timeout | Not checked | Dirty | Runtime |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for result in latest:
        lines.append(
            "| "
            f"`{result.campaign}` | "
            f"`{result.created_at}` | "
            f"`{result.commit[:12]}` | "
            f"{result.counts.get('killed', 0)} | "
            f"{result.counts.get('survived', 0)} | "
            f"{result.counts.get('timeout', 0)} | "
            f"{result.counts.get('not_checked', 0)} | "
            f"{'yes' if result.worktree_dirty else 'no'} | "
            f"{result.runtime_seconds:.2f}s |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Artifacts live in this directory as per-campaign JSON and Markdown files.",
            "- `Dirty` reflects non-artifact worktree changes in the source repository at campaign start.",
            "- Use `nix develop -c python -m devtools.mutmut_campaign list` to inspect available campaign scopes.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def write_artifacts(result: CampaignResult, *, json_out: Path | None, markdown_out: Path | None) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(asdict(result), indent=2) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(format_markdown(result))


def run_campaign(
    campaign: Campaign,
    *,
    repo_root: Path,
    json_out: Path | None,
    markdown_out: Path | None,
    keep_workspace: bool,
) -> CampaignResult:
    commit = git_commit_sha(repo_root)
    status_summary = git_status_summary(repo_root)
    created_at = datetime.now(UTC).isoformat()
    prefixes = tuple(
        f"{path.replace('/', '.').removesuffix('.py')}*"
        for path in campaign.paths_to_mutate
    )

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
        )
        write_artifacts(result, json_out=json_out, markdown_out=markdown_out)
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
    run_parser.add_argument("--markdown-out", type=Path)
    run_parser.add_argument("--keep-workspace", action="store_true")
    run_parser.set_defaults(command_fn=cmd_run)

    index_parser = subparsers.add_parser("index", help="Build an index over recorded campaign artifacts")
    index_parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=ROOT / CAMPAIGN_ARTIFACT_DIR,
    )
    index_parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / CAMPAIGN_ARTIFACT_DIR / "README.md",
    )
    index_parser.set_defaults(command_fn=cmd_index)

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
    json_out = None if args.json_out is None else (args.json_out if args.json_out.is_absolute() else ROOT / args.json_out)
    markdown_out = (
        None
        if args.markdown_out is None
        else (args.markdown_out if args.markdown_out.is_absolute() else ROOT / args.markdown_out)
    )
    result = run_campaign(
        campaign,
        repo_root=ROOT,
        json_out=json_out,
        markdown_out=markdown_out,
        keep_workspace=args.keep_workspace,
    )
    print(format_markdown(result))
    return result.exit_code


def cmd_index(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir if args.campaign_dir.is_absolute() else ROOT / args.campaign_dir
    out = args.out if args.out.is_absolute() else ROOT / args.out
    results = load_results(campaign_dir)
    rendered = format_index(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    print(rendered)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.command_fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
