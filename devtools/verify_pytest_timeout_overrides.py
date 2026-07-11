"""Verify explicit pytest timeout overrides remain bounded and reviewable.

The repository-wide pytest-timeout default lives in ``pyproject.toml``. This
gate only inspects explicit exceptions: test decorators and literal pytest
commands owned by ``devtools``. It deliberately parses Python ASTs rather than
searching source text, so prose and generated documentation are out of scope.

Marker aliases are deliberately fail-closed: imported, unresolved, cyclic, or
rebound names cannot prove the absence of a timeout override and are rejected.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from devtools import repo_root as _get_root

ROOT = _get_root()
MANIFEST_RELATIVE_PATH = Path("devtools/pytest_timeout_overrides.toml")


@dataclass(frozen=True, slots=True)
class TimeoutOverride:
    path: str
    line: int
    value: float
    source: str

    @property
    def manifest_key(self) -> tuple[str, float]:
        return (self.path, self.value)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    path: str
    value: float
    rationale: str

    @property
    def key(self) -> tuple[str, float]:
        return (self.path, self.value)


def _number_from_literal(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) and isinstance(node.value, (int, float)):
        value = float(node.value)
        return value if math.isfinite(value) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _number_from_literal(node.operand)
        if operand is not None:
            return -operand if isinstance(node.op, ast.USub) else operand
    return None


def _pytest_aliases(tree: ast.Module) -> tuple[set[str], set[str], set[str]]:
    """Return local names bound to ``pytest``, ``pytest.mark``, and ``pytest.param``."""
    pytest_names: set[str] = set()
    mark_names: set[str] = set()
    param_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    pytest_names.add(alias.asname or "pytest")
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                if alias.name == "mark":
                    mark_names.add(alias.asname or "mark")
                elif alias.name == "param":
                    param_names.add(alias.asname or "param")
    return pytest_names, mark_names, param_names


def _is_pytest_timeout_decorator(node: ast.expr, pytest_names: set[str], mark_names: set[str]) -> bool:
    func = node.func if isinstance(node, ast.Call) else node
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "timeout"
        and (
            (isinstance(func.value, ast.Name) and func.value.id in mark_names)
            or (
                isinstance(func.value, ast.Attribute)
                and func.value.attr == "mark"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id in pytest_names
            )
        )
    )


def _is_pytest_param_call(node: ast.Call, pytest_names: set[str], param_names: set[str]) -> bool:
    return (isinstance(node.func, ast.Name) and node.func.id in param_names) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "param"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in pytest_names
    )


def _parse_decorator_override(path: str, node: ast.Call) -> tuple[TimeoutOverride | None, str | None]:
    location = f"{path}:{node.lineno}"
    if len(node.args) > 2:
        return None, f"{location}: malformed pytest timeout decorator; too many positional arguments"
    timeout_argument: ast.expr | None = node.args[0] if node.args else None
    method_argument: ast.expr | None = node.args[1] if len(node.args) == 2 else None
    seen_keywords: set[str] = set()
    for keyword in node.keywords:
        if keyword.arg not in {"timeout", "method", "func_only"} or keyword.arg in seen_keywords:
            return None, f"{location}: malformed pytest timeout decorator keyword"
        seen_keywords.add(keyword.arg)
        if keyword.arg == "timeout":
            if timeout_argument is not None:
                return None, f"{location}: malformed pytest timeout decorator has multiple timeout values"
            timeout_argument = keyword.value
        elif keyword.arg == "method":
            if method_argument is not None:
                return None, f"{location}: malformed pytest timeout decorator has multiple method values"
            method_argument = keyword.value
        elif not (isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, bool)):
            return None, f"{location}: dynamic or malformed pytest timeout func_only option is forbidden"
    if timeout_argument is None:
        return None, f"{location}: malformed pytest timeout decorator is missing a timeout value"
    if method_argument is not None and (
        not isinstance(method_argument, ast.Constant)
        or not isinstance(method_argument.value, str)
        or method_argument.value not in {"signal", "thread"}
    ):
        return None, f"{location}: dynamic or malformed pytest timeout method option is forbidden"
    argument = timeout_argument
    if isinstance(argument, ast.Constant) and argument.value is None:
        return None, f"{location}: unbounded pytest timeout decorator is forbidden"
    value = _number_from_literal(argument)
    if value is None:
        return None, f"{location}: dynamic or malformed pytest timeout decorator is forbidden"
    if value <= 0:
        return None, f"{location}: pytest timeout must be positive, got {value:g}"
    return TimeoutOverride(path, node.lineno, value, "decorator"), None


def _is_pytest_execution_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "pytest_execution"


def _module_assignments(tree: ast.Module) -> tuple[dict[str, ast.expr], set[str]]:
    """Return only module bindings that have one unambiguous source assignment."""
    assignments: dict[str, ast.expr] = {}
    rebound: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr]
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets, value = [node.target], node.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id in assignments or target.id in rebound:
                assignments.pop(target.id, None)
                rebound.add(target.id)
            else:
                assignments[target.id] = value
    return assignments, rebound


def _resolve_alias(node: ast.expr, assignments: dict[str, ast.expr], seen: set[str] | None = None) -> ast.expr:
    if not isinstance(node, ast.Name) or node.id not in assignments:
        return node
    seen = set() if seen is None else seen
    if node.id in seen:
        return node
    seen.add(node.id)
    return _resolve_alias(assignments[node.id], assignments, seen)


def _flatten_command_expression(node: ast.expr, assignments: dict[str, ast.expr]) -> tuple[list[ast.expr], bool] | None:
    node = _resolve_alias(node, assignments)
    if isinstance(node, (ast.List, ast.Tuple)):
        items: list[ast.expr] = []
        dynamic = False
        for item in node.elts:
            if isinstance(item, ast.Starred):
                flattened = _flatten_command_expression(item.value, assignments)
                if flattened is None:
                    dynamic = True
                else:
                    nested_items, nested_dynamic = flattened
                    items.extend(nested_items)
                    dynamic = dynamic or nested_dynamic
            else:
                items.append(item)
        return items, dynamic
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _flatten_command_expression(node.left, assignments)
        right = _flatten_command_expression(node.right, assignments)
        if left is None or right is None:
            if left is None and right is None:
                return None
            return [*(left[0] if left is not None else []), *(right[0] if right is not None else [])], True
        return [*left[0], *right[0]], left[1] or right[1]
    return None


def _is_literal_pytest_command(nodes: list[ast.expr]) -> bool:
    return any(isinstance(node, ast.Constant) and node.value == "pytest" for node in nodes)


def _dynamic_string_fragments(node: ast.expr) -> tuple[str, ...]:
    """Return literal pieces embedded in a dynamic string expression."""
    if isinstance(node, ast.JoinedStr):
        return tuple(
            value.value for value in node.values if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (*_dynamic_string_fragments(node.left), *_dynamic_string_fragments(node.right))
    return ()


def _is_dynamic_timeout_option(node: ast.expr, assignments: dict[str, ast.expr]) -> bool:
    node = _resolve_alias(node, assignments)
    return not isinstance(node, ast.Constant) and any(
        "--timeout" in fragment for fragment in _dynamic_string_fragments(node)
    )


def _parse_command_overrides(
    path: str, line: int, nodes: list[ast.expr], assignments: dict[str, ast.expr]
) -> tuple[list[TimeoutOverride], list[str]]:
    overrides: list[TimeoutOverride] = []
    errors: list[str] = []
    for index, node in enumerate(nodes):
        resolved_node = _resolve_alias(node, assignments)
        if not isinstance(resolved_node, ast.Constant) or not isinstance(resolved_node.value, str):
            if _is_dynamic_timeout_option(node, assignments):
                location = f"{path}:{getattr(node, 'lineno', line)}"
                errors.append(f"{location}: dynamic or malformed pytest --timeout override is forbidden")
            continue
        token = resolved_node.value
        if token == "--timeout":
            location = f"{path}:{getattr(node, 'lineno', line)}"
            if index + 1 >= len(nodes):
                errors.append(f"{location}: unbounded pytest --timeout override is forbidden")
                continue
            value_node = nodes[index + 1]
            if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, str):
                errors.append(f"{location}: dynamic or malformed pytest --timeout override is forbidden")
                continue
            raw_value = value_node.value
        elif token.startswith("--timeout="):
            location = f"{path}:{getattr(node, 'lineno', line)}"
            raw_value = token.removeprefix("--timeout=")
            if not raw_value:
                errors.append(f"{location}: unbounded pytest --timeout override is forbidden")
                continue
        else:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{location}: malformed pytest --timeout override {raw_value!r}")
            continue
        if not math.isfinite(value):
            errors.append(f"{location}: malformed pytest --timeout override {raw_value!r}")
        elif value <= 0:
            errors.append(f"{location}: pytest timeout must be positive, got {value:g}")
        else:
            overrides.append(TimeoutOverride(path, getattr(node, "lineno", line), value, "command"))
    return overrides, errors


def _scan_timeout_marker(
    marker: ast.expr,
    *,
    path: str,
    pytest_names: set[str],
    mark_names: set[str],
) -> tuple[TimeoutOverride | None, str | None]:
    if not _is_pytest_timeout_decorator(marker, pytest_names, mark_names):
        return None, None
    if not isinstance(marker, ast.Call):
        return None, f"{path}:{marker.lineno}: malformed pytest timeout decorator is missing a timeout value"
    return _parse_decorator_override(path, marker)


def _flatten_marker_values(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    rebound: set[str],
    seen: set[str] | None = None,
) -> tuple[list[ast.expr], str | None]:
    """Resolve safe list/tuple marker aliases without evaluating Python."""
    if isinstance(node, ast.Name):
        if node.id in rebound:
            return [], "rebound pytest marker alias is forbidden"
        if node.id not in assignments:
            return [], "dynamic pytest marker alias is forbidden"
        seen = set() if seen is None else seen
        if node.id in seen:
            return [], "cyclic pytest marker alias is forbidden"
        seen.add(node.id)
        return _flatten_marker_values(assignments[node.id], assignments, rebound, seen)
    if isinstance(node, (ast.List, ast.Tuple)):
        markers: list[ast.expr] = []
        for item in node.elts:
            nested, error = _flatten_marker_values(item, assignments, rebound, set(seen or ()))
            if error is not None:
                return [], error
            markers.extend(nested)
        return markers, None
    return [node], None


def _scan_timeout_markers(
    marker_expression: ast.expr,
    *,
    path: str,
    assignments: dict[str, ast.expr],
    rebound: set[str],
    pytest_names: set[str],
    mark_names: set[str],
) -> tuple[list[TimeoutOverride], list[str]]:
    markers, flatten_error = _flatten_marker_values(marker_expression, assignments, rebound)
    if flatten_error is not None:
        return [], [f"{path}:{marker_expression.lineno}: {flatten_error}"]
    overrides: list[TimeoutOverride] = []
    errors: list[str] = []
    for marker in markers:
        override, error = _scan_timeout_marker(
            marker,
            path=path,
            pytest_names=pytest_names,
            mark_names=mark_names,
        )
        if override is not None:
            overrides.append(override)
        if error is not None:
            errors.append(error)
    return overrides, errors


def _scan_python(
    path: Path, root: Path, *, scan_decorators: bool, scan_commands: bool
) -> tuple[list[TimeoutOverride], list[str]]:
    relative = path.relative_to(root).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except SyntaxError as exc:
        return [], [f"{relative}:{exc.lineno or 0}: cannot parse Python source: {exc.msg}"]

    overrides: list[TimeoutOverride] = []
    errors: list[str] = []
    pytest_names, mark_names, param_names = _pytest_aliases(tree)
    assignments, rebound = _module_assignments(tree)
    if scan_decorators:
        for top_level in tree.body:
            if not isinstance(top_level, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = top_level.targets if isinstance(top_level, ast.Assign) else [top_level.target]
            value = top_level.value
            if value is None or not any(
                isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets
            ):
                continue
            found, found_errors = _scan_timeout_markers(
                value,
                path=relative,
                assignments=assignments,
                rebound=rebound,
                pytest_names=pytest_names,
                mark_names=mark_names,
            )
            overrides.extend(found)
            errors.extend(found_errors)
    for candidate in ast.walk(tree):
        if scan_decorators and isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in candidate.decorator_list:
                override, error = _scan_timeout_marker(
                    decorator,
                    path=relative,
                    pytest_names=pytest_names,
                    mark_names=mark_names,
                )
                if override is not None:
                    overrides.append(override)
                if error is not None:
                    errors.append(error)
        if (
            scan_decorators
            and isinstance(candidate, ast.Call)
            and _is_pytest_param_call(candidate, pytest_names, param_names)
        ):
            for keyword in candidate.keywords:
                if keyword.arg != "marks":
                    continue
                found, found_errors = _scan_timeout_markers(
                    keyword.value,
                    path=relative,
                    assignments=assignments,
                    rebound=rebound,
                    pytest_names=pytest_names,
                    mark_names=mark_names,
                )
                overrides.extend(found)
                errors.extend(found_errors)
        if not scan_commands:
            continue
        command_nodes: list[ast.expr] | None = None
        is_pytest_command = False
        line = getattr(candidate, "lineno", 0)
        if isinstance(candidate, (ast.List, ast.Tuple, ast.BinOp)):
            flattened = _flatten_command_expression(candidate, assignments)
            if flattened is not None:
                command_nodes, dynamic_expression = flattened
                is_pytest_command = _is_literal_pytest_command(command_nodes)
                if is_pytest_command and dynamic_expression and _is_dynamic_timeout_option(candidate, assignments):
                    errors.append(f"{relative}:{line}: dynamic managed pytest command expression is forbidden")
        elif isinstance(candidate, ast.Call) and _is_pytest_execution_call(candidate):
            flattened = _flatten_command_expression(ast.Tuple(elts=candidate.args, ctx=ast.Load()), assignments)
            if flattened is None:
                if any(
                    _is_dynamic_timeout_option(
                        argument.value if isinstance(argument, ast.Starred) else argument,
                        assignments,
                    )
                    for argument in candidate.args
                ):
                    errors.append(f"{relative}:{line}: dynamic managed pytest command expression is forbidden")
            else:
                command_nodes, dynamic_expression = flattened
                is_pytest_command = True
                if dynamic_expression and any(
                    _is_dynamic_timeout_option(
                        argument.value if isinstance(argument, ast.Starred) else argument,
                        assignments,
                    )
                    for argument in candidate.args
                ):
                    errors.append(f"{relative}:{line}: dynamic managed pytest command expression is forbidden")
        if command_nodes is not None and is_pytest_command:
            found, found_errors = _parse_command_overrides(relative, line, command_nodes, assignments)
            overrides.extend(found)
            errors.extend(found_errors)
    return overrides, errors


def _read_default_timeout(pyproject_path: Path) -> float:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    timeout: object = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("timeout")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("tool.pytest.ini_options.timeout must be a positive finite number")
    return float(timeout)


def _read_manifest(manifest_path: Path, root: Path) -> tuple[list[ManifestEntry], list[str]]:
    if not manifest_path.exists():
        return [], [f"missing timeout override manifest: {manifest_path}"]
    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [], [f"{manifest_path}: invalid TOML: {exc}"]
    raw_entries = data.get("exception", [])
    if not isinstance(raw_entries, list):
        return [], [f"{manifest_path}: exception must be an array of tables"]

    entries: list[ManifestEntry] = []
    errors: list[str] = []
    seen: set[tuple[str, float]] = set()
    for index, raw_entry in enumerate(raw_entries):
        label = f"{manifest_path}: exception[{index}]"
        if not isinstance(raw_entry, dict):
            errors.append(f"{label} must be a table")
            continue
        path = raw_entry.get("path")
        value = raw_entry.get("value")
        rationale = raw_entry.get("rationale")
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            errors.append(f"{label} path must be a repository-relative path")
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            errors.append(f"{label} value must be a positive finite number")
            continue
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"{label} rationale must be non-empty")
            continue
        entry = ManifestEntry(path, float(value), rationale.strip())
        if entry.key in seen:
            errors.append(f"{label} duplicates manifest entry for {path} value {entry.value:g}")
            continue
        seen.add(entry.key)
        entries.append(entry)
    return entries, errors


def check_timeout_overrides(
    root: Path, *, pyproject_path: Path | None = None, manifest_path: Path | None = None
) -> tuple[list[TimeoutOverride], list[str]]:
    """Return all valid overrides and every policy violation under ``root``."""
    root = root.resolve()
    pyproject_path = (pyproject_path or root / "pyproject.toml").resolve()
    manifest_path = (manifest_path or root / MANIFEST_RELATIVE_PATH).resolve()
    try:
        default_timeout = _read_default_timeout(pyproject_path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        return [], [f"{pyproject_path}: cannot read pytest timeout default: {exc}"]

    overrides: list[TimeoutOverride] = []
    errors: list[str] = []
    tests_dir = root / "tests"
    if tests_dir.exists():
        for path in sorted(tests_dir.rglob("*.py")):
            found, found_errors = _scan_python(path, root, scan_decorators=True, scan_commands=False)
            overrides.extend(found)
            errors.extend(found_errors)
    devtools_dir = root / "devtools"
    if devtools_dir.exists():
        for path in sorted(devtools_dir.rglob("*.py")):
            found, found_errors = _scan_python(path, root, scan_decorators=False, scan_commands=True)
            overrides.extend(found)
            errors.extend(found_errors)

    entries, manifest_errors = _read_manifest(manifest_path, root)
    errors.extend(manifest_errors)
    exceptional = {override.manifest_key for override in overrides if override.value > default_timeout}
    declared = {entry.key for entry in entries}
    for entry_path, value in sorted(exceptional - declared):
        errors.append(f"{entry_path}: timeout {value:g}s exceeds {default_timeout:g}s without a manifest rationale")
    for entry_path, value in sorted(declared - exceptional):
        errors.append(f"stale timeout override manifest entry: {entry_path} value {value:g}")
    return overrides, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to inspect.")
    parser.add_argument("--pyproject", type=Path, help="pytest configuration path (defaults to ROOT/pyproject.toml).")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Exception manifest path (defaults to ROOT/devtools/pytest_timeout_overrides.toml).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the policy result as JSON.")
    args = parser.parse_args(argv)
    overrides, errors = check_timeout_overrides(args.root, pyproject_path=args.pyproject, manifest_path=args.manifest)
    payload: dict[str, Any] = {
        "overrides": [
            {"path": item.path, "line": item.line, "value": item.value, "source": item.source} for item in overrides
        ],
        "errors": errors,
        "ok": not errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"pytest timeout overrides: {len(overrides)} explicit override(s), {len(errors)} violation(s)")
        for error in errors:
            print(f"  {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
