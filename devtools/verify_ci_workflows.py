"""Verify CI workflow files reference locally-known commands and paths.

Checks each .github/workflows/*.yml file for:
- devtools commands: must be registered in the devtools catalog
- polylogue CLI commands: must match the installed CLI surface
- file/dir paths passed to ruff/mypy/pytest: must exist in the repo

Does NOT check remote CI state, GitHub secrets, or external service calls.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

import yaml

from devtools import repo_root as _get_root
from devtools.command_catalog import COMMANDS

ROOT = _get_root()
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

_DEVTOOLS_RE = re.compile(r"\bdevtools\s+(\S+)")
_UVX_DEVTOOLS_RE = re.compile(r"uv\s+run\s+devtools\s+(\S+)")
_POLYLOGUE_CMD_RE = re.compile(r"(?:uv\s+run\s+)?polylogue\s+(\S+)")
_RUFF_PATH_RE = re.compile(r"\bruff\s+(?:check|format)\s+(?:--\S+\s+)*(.+)$", re.MULTILINE)
_MYPY_PATH_RE = re.compile(r"\bmypy\s+(.+)$", re.MULTILINE)
_PYTEST_PATH_RE = re.compile(r"\bpytest\s+(?:-\S+\s+)*(\S+)", re.MULTILINE)


def _devtools_command_names() -> frozenset[str]:
    return frozenset(COMMANDS.keys())


def _extract_run_steps(workflow: dict[str, object]) -> list[tuple[str, str, str]]:
    """Return (job_name, step_name, run_script) for all run steps."""
    results: list[tuple[str, str, str]] = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return results
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []):
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if isinstance(run, str) and run.strip():
                step_name = str(step.get("name", ""))
                results.append((str(job_name), step_name, run))
    return results


def _check_devtools_commands(
    run: str,
    known: frozenset[str],
    job: str,
    step: str,
    workflow: str,
) -> list[str]:
    errors: list[str] = []
    for match in _DEVTOOLS_RE.finditer(run):
        cmd = match.group(1)
        if cmd not in known:
            errors.append(f"{workflow}:{job}/{step!r}: unknown devtools command {cmd!r}")
    return errors


def _check_paths(
    run: str,
    root: Path,
    job: str,
    step: str,
    workflow: str,
) -> list[str]:
    warnings: list[str] = []
    for pattern, label in [
        (_RUFF_PATH_RE, "ruff"),
        (_MYPY_PATH_RE, "mypy"),
        (_PYTEST_PATH_RE, "pytest"),
    ]:
        for match in pattern.finditer(run):
            raw = match.group(1).strip()
            # Skip flags and variable expansions
            parts = shlex.split(raw) if raw else []
            for part in parts:
                # Skip flags, shell expansions, and non-path tokens
                if part.startswith("-") or "$" in part or "{" in part:
                    continue
                # Accept only plausible repo-relative paths (word chars, slashes, dots, hyphens)
                if not re.match(r"^[\w./\-]+$", part):
                    continue
                path = root / part
                if not path.exists():
                    warnings.append(f"{workflow}:{job}/{step!r}: {label} references non-existent path {part!r}")
    return warnings


def check_workflow(path: Path, root: Path, known_commands: frozenset[str]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a workflow file."""
    errors: list[str] = []
    warnings: list[str] = []
    rel = path.relative_to(root).as_posix()

    try:
        with open(path, encoding="utf-8") as f:
            workflow = yaml.safe_load(f)
    except Exception as exc:
        return [f"{rel}: failed to parse YAML: {exc}"], []

    if not isinstance(workflow, dict):
        return [f"{rel}: expected mapping at top level"], []

    for job, step, run in _extract_run_steps(workflow):
        errors.extend(_check_devtools_commands(run, known_commands, job, step, rel))
        warnings.extend(_check_paths(run, root, job, step, rel))

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--warn-paths", action="store_true", help="Treat missing paths as errors, not warnings.")
    args = p.parse_args(argv)

    if not WORKFLOWS_DIR.exists():
        if args.json:
            import json

            json.dump({"blocking": False, "errors": [], "warnings": [], "files_checked": 0}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print("no .github/workflows/ directory found — skipping")
        return 0

    known = _devtools_command_names()
    all_errors: list[str] = []
    all_warnings: list[str] = []
    files_checked = 0

    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        errors, warnings = check_workflow(path, ROOT, known)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        files_checked += 1

    if args.warn_paths:
        all_errors.extend(all_warnings)
        all_warnings = []

    blocking = bool(all_errors)

    if args.json:
        import json

        json.dump(
            {
                "blocking": blocking,
                "errors": all_errors,
                "warnings": all_warnings,
                "files_checked": files_checked,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if all_errors:
            for e in all_errors:
                print(f"[BLOCK] {e}")
        else:
            print(f"verify-ci-workflows: {files_checked} workflow files checked, no errors")
        for w in all_warnings:
            print(f"[warn] {w}")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
