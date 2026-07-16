#!/usr/bin/env python3
"""Reproducible candidate census for Polylogue test-suite diet research.

This is a research query, not a lint and not a deletion oracle.  It emits the
test functions matching several deliberately broad implementation-coupling or
weak-oracle signals so a reviewer can inspect clusters instead of searching
one test at a time.

Run from the repository root:

    python .agent/scratch/testsuite_diet/census/test_coupling_census.py

Outputs are stable JSON and TSV files beside this script.  Update signal
definitions here, regenerate both files, and record the methodology change in
the directory README rather than silently comparing incompatible counts.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TEST_ROOT = ROOT / "tests"
OUT_DIR = Path(__file__).resolve().parent

SCHEMA_VERSION = 1

SOURCE_METHODS = {
    "read_text",
    "read_bytes",
    "getsource",
    "getsourcelines",
    "getsourcefile",
}
SOURCE_QUALIFIED_CALLS = {
    "ast.parse",
    "ast.walk",
    "ast.dump",
    "inspect.getsource",
    "inspect.getsourcelines",
    "inspect.getsourcefile",
    "tokenize.generate_tokens",
    "tokenize.tokenize",
}
MOCK_INTERACTION_NAMES = {
    "assert_any_await",
    "assert_any_call",
    "assert_awaited",
    "assert_awaited_once",
    "assert_awaited_once_with",
    "assert_awaited_with",
    "assert_called",
    "assert_called_once",
    "assert_called_once_with",
    "assert_called_with",
    "assert_has_awaits",
    "assert_has_calls",
    "assert_not_awaited",
    "assert_not_called",
    "await_args",
    "await_args_list",
    "await_count",
    "call_args",
    "call_args_list",
    "call_count",
    "method_calls",
    "mock_calls",
}
PATCH_METHODS = {
    "context",
    "delattr",
    "delenv",
    "setattr",
    "setenv",
    "setitem",
}


@dataclass(frozen=True)
class Finding:
    nodeid: str
    path: str
    line: int
    body_lines: int
    source_or_ast: bool
    source_literal_membership: bool
    mock_interaction: bool
    patches_collaborator: bool
    coarse_weak_assertions_only: bool
    evidence: tuple[str, ...]


def _qualified_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _literal_membership(assertion: ast.Assert) -> bool:
    for node in ast.walk(assertion.test):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            continue
        values = (node.left, *node.comparators)
        if any(isinstance(value, ast.Constant) and isinstance(value.value, str) for value in values):
            return True
    return False


def _source_signal(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, set[str]]:
    evidence: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        qualified = _qualified_name(node.func)
        leaf = qualified.rsplit(".", 1)[-1]
        if leaf in SOURCE_METHODS or qualified in SOURCE_QUALIFIED_CALLS:
            evidence.add(qualified or leaf)
    return bool(evidence), evidence


def _mock_signal(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, set[str]]:
    evidence: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and node.attr in MOCK_INTERACTION_NAMES:
            evidence.add(node.attr)
    return bool(evidence), evidence


def _patch_signal(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, set[str]]:
    evidence: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        qualified = _qualified_name(node.func)
        if (
            qualified in {"patch", "patch.object", "mock.patch", "mock.patch.object"}
            or qualified.startswith("monkeypatch.")
            and qualified.rsplit(".", 1)[-1] in PATCH_METHODS
        ):
            evidence.add(qualified)
    return bool(evidence), evidence


def _is_nullness(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    values = (test.left, *test.comparators)
    return any(isinstance(value, ast.Constant) and value.value is None for value in values)


def _is_permissive_numeric_bound(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if not all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in test.ops):
        return False
    values = (test.left, *test.comparators)
    return any(
        isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool)
        for value in values
    )


def _is_coarse_assertion(assertion: ast.Assert) -> bool:
    test = assertion.test
    if isinstance(test, ast.Call) and _qualified_name(test.func) in {"hasattr", "isinstance"}:
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Call):
        return _qualified_name(test.operand.func) in {"hasattr", "isinstance"}
    return _is_nullness(test) or _is_permissive_numeric_bound(test)


def _coarse_weak_only(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    assertions = [node for node in ast.walk(function) if isinstance(node, ast.Assert)]
    return bool(assertions) and all(_is_coarse_assertion(assertion) for assertion in assertions)


def _test_functions(tree: ast.Module) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            yield node


def collect() -> tuple[list[Finding], dict[str, str]]:
    findings: list[Finding] = []
    file_hashes: dict[str, str] = {}
    for path in sorted(TEST_ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        file_hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError:
            continue
        for function in _test_functions(tree):
            source_or_ast, source_evidence = _source_signal(function)
            mock_interaction, mock_evidence = _mock_signal(function)
            patches_collaborator, patch_evidence = _patch_signal(function)
            assertions = [node for node in ast.walk(function) if isinstance(node, ast.Assert)]
            source_literal_membership = source_or_ast and any(_literal_membership(item) for item in assertions)
            weak_only = _coarse_weak_only(function)
            if not (source_or_ast or mock_interaction or weak_only):
                continue
            end_line = function.end_lineno or function.lineno
            findings.append(
                Finding(
                    nodeid=f"{relative}::{function.name}",
                    path=relative,
                    line=function.lineno,
                    body_lines=end_line - function.lineno + 1,
                    source_or_ast=source_or_ast,
                    source_literal_membership=source_literal_membership,
                    mock_interaction=mock_interaction,
                    patches_collaborator=patches_collaborator,
                    coarse_weak_assertions_only=weak_only,
                    evidence=tuple(sorted(source_evidence | mock_evidence | patch_evidence)),
                )
            )
    return findings, file_hashes


def _summary(findings: list[Finding]) -> dict[str, int]:
    source = [item for item in findings if item.source_or_ast]
    source_literals = [item for item in findings if item.source_literal_membership]
    mock = [item for item in findings if item.mock_interaction]
    patched_mock = [item for item in mock if item.patches_collaborator]
    weak = [item for item in findings if item.coarse_weak_assertions_only]
    return {
        "candidate_functions_unique": len(findings),
        "source_or_ast_functions": len(source),
        "source_literal_membership_functions": len(source_literals),
        "mock_interaction_functions": len(mock),
        "mock_interaction_body_lines": sum(item.body_lines for item in mock),
        "mock_interaction_with_patch_functions": len(patched_mock),
        "coarse_weak_assertions_only_functions": len(weak),
    }


def write_outputs(findings: list[Finding], file_hashes: dict[str, str]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "tests/**/*.py test_ functions",
        "summary": _summary(findings),
        "input_sha256": file_hashes,
        "findings": [asdict(item) for item in findings],
    }
    json_path = OUT_DIR / "test-coupling-census.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tsv_path = OUT_DIR / "test-coupling-census.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "nodeid",
                "body_lines",
                "source_or_ast",
                "source_literal_membership",
                "mock_interaction",
                "patches_collaborator",
                "coarse_weak_assertions_only",
                "evidence",
            ]
        )
        for item in findings:
            writer.writerow(
                [
                    item.nodeid,
                    item.body_lines,
                    int(item.source_or_ast),
                    int(item.source_literal_membership),
                    int(item.mock_interaction),
                    int(item.patches_collaborator),
                    int(item.coarse_weak_assertions_only),
                    ",".join(item.evidence),
                ]
            )

    _write_rollup(OUT_DIR / "test-coupling-by-file.tsv", findings, key=lambda item: item.path)
    _write_rollup(OUT_DIR / "test-coupling-by-area.tsv", findings, key=_area_for)


def _area_for(item: Finding) -> str:
    parts = Path(item.path).parts
    if len(parts) >= 3 and parts[1] in {"unit", "integration"}:
        return "/".join(parts[:3])
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return item.path


def _write_rollup(
    path: Path,
    findings: list[Finding],
    *,
    key: Callable[[Finding], str],
) -> None:
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for item in findings:
        grouped[key(item)].append(item)
    rows: list[tuple[object, ...]] = []
    for group, items in grouped.items():
        mock_items = [item for item in items if item.mock_interaction]
        rows.append(
            (
                group,
                len(items),
                sum(item.source_or_ast for item in items),
                sum(item.source_literal_membership for item in items),
                len(mock_items),
                sum(item.body_lines for item in mock_items),
                sum(item.mock_interaction and item.patches_collaborator for item in items),
                sum(item.coarse_weak_assertions_only for item in items),
            )
        )
    rows.sort(key=lambda row: (-int(row[1]), str(row[0])))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "group",
                "candidate_functions",
                "source_or_ast",
                "source_literal_membership",
                "mock_interaction",
                "mock_interaction_body_lines",
                "mock_interaction_with_patch",
                "coarse_weak_assertions_only",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    findings, file_hashes = collect()
    write_outputs(findings, file_hashes)
    print(json.dumps(_summary(findings), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
