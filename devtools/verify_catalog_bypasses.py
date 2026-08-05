"""Reject unregistered direct devtools command execution in control surfaces.

The CommandSpec catalog is the public execution registry. Workflow ``run:``
blocks, repository hooks, and devtools process-launch calls must invoke it as
``devtools ...``. This checker catches literal direct forms such as
``python -m devtools.some_module`` and ``python devtools/some_module.py``.

Scope is deliberately execution-only:

* workflow ``run:`` blocks and hook shell scripts are scanned as shell text;
* ``devtools/**/*.py`` is parsed with ``ast`` and only command-runner call
  arguments are inspected.

Generated provenance headers, argparse ``prog`` values, comments, and
docstrings are outside the scope because they do not launch a process. A real
hook adapter can remain only through a structured ``sanctioned-bypass`` entry
in ``CATALOG_BYPASS_SITES`` with a reason. Dynamic command construction is
outside this literal-vector check and must use catalog helpers at review.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from devtools import repo_root as _get_root
from devtools.command_catalog import CATALOG_BYPASS_SITES

ROOT = _get_root()
_COMMAND_RUNNERS = {"run", "check_call", "check_output", "Popen", "system", "_run", "run_command", "_run_command"}
_PYTHON = r"python(?:3(?:\.\d+)?)?"
_MODULE_INVOCATION = re.compile(rf"(?<![\w./-])(?:uv\s+run\s+)?{_PYTHON}\s+-m\s+(devtools\.[A-Za-z_][A-Za-z0-9_]*)")
_SCRIPT_INVOCATION = re.compile(
    rf"(?<![\w./-])(?:uv\s+run\s+)?{_PYTHON}\s+(?:\./)?(devtools/[A-Za-z_][A-Za-z0-9_]*\.py)"
)


@dataclass(frozen=True, slots=True)
class DirectDevtoolsInvocation:
    path: str
    lineno: int
    invocation: str


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _shell_invocations(source: str, *, path: str) -> list[DirectDevtoolsInvocation]:
    findings: list[DirectDevtoolsInvocation] = []
    for pattern, prefix in ((_MODULE_INVOCATION, "python -m"), (_SCRIPT_INVOCATION, "python")):
        for match in pattern.finditer(source):
            findings.append(
                DirectDevtoolsInvocation(
                    path=path,
                    lineno=source.count("\n", 0, match.start()) + 1,
                    invocation=f"{prefix} {match.group(1)}",
                )
            )
    return findings


def _is_command_runner(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in _COMMAND_RUNNERS
    return isinstance(func, ast.Attribute) and func.attr in _COMMAND_RUNNERS


def _is_python_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return re.fullmatch(_PYTHON, node.value) is not None
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "executable"
    )


def _literal_command_parts(node: ast.expr) -> list[ast.expr] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node]
    if isinstance(node, ast.List | ast.Tuple):
        return list(node.elts)
    return None


def _literal_string(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _invocations_from_command_expression(node: ast.expr, *, path: str, lineno: int) -> list[DirectDevtoolsInvocation]:
    parts = _literal_command_parts(node)
    if parts is None:
        return []
    if len(parts) == 1:
        shell = _literal_string(parts[0])
        return _shell_invocations(shell, path=path) if shell is not None else []

    findings: list[DirectDevtoolsInvocation] = []
    for index, part in enumerate(parts):
        if not _is_python_expression(part) or index + 1 >= len(parts):
            continue
        next_part = _literal_string(parts[index + 1])
        if next_part == "-m" and index + 2 < len(parts):
            module = _literal_string(parts[index + 2])
            if module is not None and re.fullmatch(r"devtools\.[A-Za-z_][A-Za-z0-9_]*", module):
                findings.append(DirectDevtoolsInvocation(path=path, lineno=lineno, invocation=f"python -m {module}"))
        elif next_part is not None and re.fullmatch(r"(?:\./)?devtools/[A-Za-z_][A-Za-z0-9_]*\.py", next_part):
            findings.append(
                DirectDevtoolsInvocation(path=path, lineno=lineno, invocation=f"python {next_part.removeprefix('./')}")
            )
    return findings


def _python_invocations(source: str, *, path: str) -> list[DirectDevtoolsInvocation]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return []
    findings: list[DirectDevtoolsInvocation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_command_runner(node) or not node.args:
            continue
        findings.extend(_invocations_from_command_expression(node.args[0], path=path, lineno=node.lineno))
    return findings


def _workflow_run_blocks(path: Path) -> Iterable[str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return ()

    def _walk(value: object) -> Iterable[str]:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "run" and isinstance(nested, str):
                    yield nested
                yield from _walk(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from _walk(nested)

    return tuple(_walk(data))


def control_surface_paths(root: Path = ROOT) -> tuple[Path, ...]:
    paths: list[Path] = []
    workflows = root / ".github" / "workflows"
    if workflows.exists():
        paths.extend(sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml"))))
    for hook_dir in (root / ".githooks", root / ".beads-hooks"):
        if hook_dir.exists():
            paths.extend(sorted(path for path in hook_dir.iterdir() if path.is_file()))
    devtools = root / "devtools"
    if devtools.exists():
        paths.extend(sorted(devtools.rglob("*.py")))
    return tuple(paths)


def scan_control_surfaces(root: Path = ROOT, *, paths: Iterable[Path] | None = None) -> list[DirectDevtoolsInvocation]:
    findings: list[DirectDevtoolsInvocation] = []
    for candidate in paths if paths is not None else control_surface_paths(root):
        path = candidate if candidate.is_absolute() else root / candidate
        if not path.is_file():
            continue
        relative = _relative_path(path, root)
        source = path.read_text(encoding="utf-8")
        if path.suffix in {".yml", ".yaml"}:
            for run in _workflow_run_blocks(path):
                findings.extend(_shell_invocations(run, path=relative))
        elif path.suffix == ".py":
            findings.extend(_python_invocations(source, path=relative))
        else:
            findings.extend(_shell_invocations(source, path=relative))
    return findings


def _sanctioned_keys() -> set[tuple[str, str]]:
    return {(site.path, site.marker) for site in CATALOG_BYPASS_SITES if site.disposition == "sanctioned-bypass"}


def collect_violations(root: Path = ROOT, *, paths: Iterable[Path] | None = None) -> list[DirectDevtoolsInvocation]:
    sanctioned = _sanctioned_keys()
    return [item for item in scan_control_surfaces(root, paths=paths) if (item.path, item.invocation) not in sanctioned]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)
    violations = collect_violations(ROOT)
    if args.json:
        print(json.dumps({"ok": not violations, "violations": [asdict(item) for item in violations]}, indent=2))
    elif violations:
        print("Unregistered direct devtools invocations:", file=sys.stderr)
        for item in violations:
            print(f"  {item.path}:{item.lineno}: {item.invocation}", file=sys.stderr)
    else:
        print("Catalog bypass scan OK: no undeclared direct devtools execution sites.")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
