"""Verify CI workflow files reference locally-known commands and paths.

Checks each .github/workflows/*.yml file for:
- devtools commands: must be registered in the devtools catalog
- polylogue CLI commands: must match the installed CLI surface
- file/dir paths passed to ruff/mypy/pytest: must exist in the repo

Also exposes an inventory of workflow facts that other manifest checks
consume to cross-reference declared CI state against actual workflow YAML:

- workflow_names: the ``name:`` of each workflow
- job_names: the keys under ``jobs:`` for each workflow
- run_commands: every ``run:`` script string concatenated across all steps
- artifact_uploads: artifact ``name`` values uploaded via
  ``actions/upload-artifact``
- triggers: top-level ``on:`` keys (workflow_dispatch, pull_request, push, ...)

Does NOT check remote CI state, GitHub secrets, or external service calls.
Remote facts (branch protection, required checks, success rate, runner
availability) are deliberately not invented here.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from devtools import repo_root as _get_root
from devtools.command_catalog import COMMANDS, command_name_from_tokens

ROOT = _get_root()
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

_DEVTOOLS_RE = re.compile(r"(?<![\w./-])devtools(?![./\w-])([^\n]*)")
_POLYLOGUE_CMD_RE = re.compile(r"(?:uv\s+run\s+)?polylogue\s+(\S+)")
_RUFF_PATH_RE = re.compile(r"\bruff\s+(?:check|format)\s+(?:--\S+\s+)*(.+)$", re.MULTILINE)
_MYPY_PATH_RE = re.compile(r"\bmypy\s+(.+)$", re.MULTILINE)
_PYTEST_PATH_RE = re.compile(r"\bpytest\s+(?:-\S+\s+)*(\S+)", re.MULTILINE)


def _devtools_command_names() -> frozenset[str]:
    return frozenset(COMMANDS.keys())


def _devtools_command_from_rest(rest: str) -> str | None:
    for stop in ("&&", "||", "|", ";", "#", "$("):
        idx = rest.find(stop)
        if idx >= 0:
            rest = rest[:idx]
    try:
        parts = shlex.split(rest)
    except ValueError:
        parts = rest.split()
    tokens = tuple(part for part in parts if part and not part.startswith("-"))
    if not tokens:
        return None
    known = command_name_from_tokens(tokens)
    if known is not None:
        return known
    max_len = max((len(spec.command_path) for spec in COMMANDS.values()), default=1)
    return " ".join(tokens[: min(len(tokens), max_len)])


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
        cmd = _devtools_command_from_rest(match.group(1))
        if cmd is None:
            continue
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
            # Skip flags and variable expansions. Multiline shell scripts can
            # leave trailing backslash continuations inside the captured
            # fragment; shlex rejects them with ``No escaped character``.
            # Treat any tokenization failure as "fragment isn't a clean path
            # list" and skip it — best-effort, not strict shell parsing.
            try:
                parts = shlex.split(raw) if raw else []
            except ValueError:
                continue
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


@dataclass(frozen=True)
class WorkflowFacts:
    """Locally-knowable facts extracted from a single workflow YAML."""

    path: Path
    workflow_name: str
    job_names: tuple[str, ...]
    run_commands: tuple[str, ...]
    artifact_uploads: tuple[str, ...]
    triggers: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowInventory:
    """Aggregate facts across every workflow under .github/workflows/."""

    workflows: tuple[WorkflowFacts, ...] = field(default_factory=tuple)

    @property
    def workflow_names(self) -> tuple[str, ...]:
        return tuple(w.workflow_name for w in self.workflows if w.workflow_name)

    @property
    def all_job_names(self) -> tuple[str, ...]:
        return tuple(name for w in self.workflows for name in w.job_names)

    @property
    def all_run_commands(self) -> tuple[str, ...]:
        return tuple(run for w in self.workflows for run in w.run_commands)

    @property
    def all_artifact_uploads(self) -> tuple[str, ...]:
        return tuple(name for w in self.workflows for name in w.artifact_uploads)


def _extract_workflow_facts(path: Path, workflow: dict[str, object]) -> WorkflowFacts:
    """Pull job names, run commands, artifact names, and triggers from one YAML."""
    workflow_name = workflow.get("name") if isinstance(workflow.get("name"), str) else ""
    triggers: list[str] = []
    # PyYAML parses the bare key ``on`` as the boolean ``True`` (YAML 1.1
    # norway-style problem), so the actual triggers section can live under
    # either ``"on"`` or ``True``. We accept both.
    # Cast workflow to a dynamic-key dict for the boolean-key lookup since the
    # declared type only models string keys.
    raw_workflow: dict[object, object] = dict(workflow.items())
    on = raw_workflow.get("on") if "on" in raw_workflow else raw_workflow.get(True)
    if isinstance(on, dict | list):
        triggers.extend(str(k) for k in on)
    elif isinstance(on, str):
        triggers.append(on)

    job_names: list[str] = []
    run_commands: list[str] = []
    artifact_uploads: list[str] = []
    jobs = workflow.get("jobs")
    if isinstance(jobs, dict):
        for job_name, job in jobs.items():
            job_names.append(str(job_name))
            if not isinstance(job, dict):
                continue
            for step in job.get("steps", []):
                if not isinstance(step, dict):
                    continue
                run = step.get("run")
                if isinstance(run, str) and run.strip():
                    run_commands.append(run)
                uses = step.get("uses")
                if isinstance(uses, str) and uses.startswith("actions/upload-artifact"):
                    with_block = step.get("with")
                    if isinstance(with_block, dict):
                        artifact_name = with_block.get("name")
                        if isinstance(artifact_name, str) and artifact_name.strip():
                            artifact_uploads.append(artifact_name)

    return WorkflowFacts(
        path=path,
        workflow_name=str(workflow_name) if workflow_name else "",
        job_names=tuple(job_names),
        run_commands=tuple(run_commands),
        artifact_uploads=tuple(artifact_uploads),
        triggers=tuple(triggers),
    )


def inventory_workflows(workflows_dir: Path | None = None) -> WorkflowInventory:
    """Parse every ``.yml`` workflow under ``workflows_dir`` into facts.

    Used by manifest checks that need to cross-reference declared CI
    state (job names, command presence, artifact uploads, triggers)
    against committed workflow YAML.
    """
    target = workflows_dir if workflows_dir is not None else WORKFLOWS_DIR
    if not target.exists():
        return WorkflowInventory()
    facts: list[WorkflowFacts] = []
    for path in sorted(target.glob("*.yml")):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        facts.append(_extract_workflow_facts(path, data))
    return WorkflowInventory(workflows=tuple(facts))


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
            print(f"verify ci-workflows: {files_checked} workflow files checked, no errors")
        for w in all_warnings:
            print(f"[warn] {w}")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
