"""Audit coverage metadata against the pytest-testmon dependency graph.

This is an on-demand audit.  It consumes an existing coverage.py JSON report
and an existing pytest-testmon SQLite database; it never runs pytest and never
creates or refreshes coverage data.

Coverage metadata can know about a file that testmon does not fingerprint.
That absence is harmless for a declaration-only module, but it is a blind spot
for executable code.  The distinction is made from the source AST rather than
from the coverage statement count or the file name, so a stale or edited
fixture cannot turn executable validation code into a safe result.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from polylogue.storage.sqlite.connection_profile import open_readonly_connection

ASTClassification = Literal["declaration-only", "executable", "source-unreadable"]
FindingStatus = Literal[
    "fingerprinted",
    "declaration-only-unfingerprinted",
    "executable-validator-unfingerprinted",
    "source-unreadable",
]


@dataclass(frozen=True, slots=True)
class CoverageFile:
    path: str
    statements: int
    covered_lines: int


@dataclass(frozen=True, slots=True)
class BlindSpotFinding:
    path: str
    statements: int
    covered_lines: int
    ast_classification: ASTClassification
    testmon_fingerprinted: bool
    status: FindingStatus
    safe: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlindSpotReport:
    findings: tuple[BlindSpotFinding, ...]
    coverage_file_count: int
    testmon_fingerprint_count: int

    @property
    def risks(self) -> tuple[BlindSpotFinding, ...]:
        return tuple(finding for finding in self.findings if not finding.safe)

    def to_dict(self) -> dict[str, object]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "coverage_file_count": self.coverage_file_count,
            "testmon_fingerprint_count": self.testmon_fingerprint_count,
            "risk_count": len(self.risks),
            "safe": not self.risks,
        }


def _relative_path(filename: str, *, source_root: Path) -> str | None:
    root = source_root.resolve()
    candidate = Path(filename)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root)
        except ValueError:
            return None
    normalized = PurePosixPath(str(candidate).replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        return None
    return str(normalized)


def read_coverage_files(coverage_json_path: Path, *, source_root: Path) -> tuple[CoverageFile, ...]:
    """Read source-file metadata from an existing coverage.py JSON report."""
    payload = json.loads(coverage_json_path.read_text(encoding="utf-8"))
    raw_files = payload.get("files", {})
    if not isinstance(raw_files, dict):
        raise ValueError("coverage JSON has no files object")

    files: list[CoverageFile] = []
    for raw_filename, raw_filedata in raw_files.items():
        if not isinstance(raw_filename, str) or not isinstance(raw_filedata, dict):
            continue
        relative = _relative_path(raw_filename, source_root=source_root)
        if relative is None or not relative.endswith(".py"):
            continue
        summary = raw_filedata.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        files.append(
            CoverageFile(
                path=relative,
                statements=int(summary.get("num_statements", 0)),
                covered_lines=int(summary.get("covered_lines", 0)),
            )
        )
    return tuple(sorted(files, key=lambda item: item.path))


def read_testmon_fingerprints(testmon_db_path: Path, *, source_root: Path) -> frozenset[str]:
    """Read the file paths currently represented in testmon's dependency graph."""
    connection = open_readonly_connection(testmon_db_path)
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(file_fp)").fetchall()}
        path_column = "filename" if "filename" in columns else "path" if "path" in columns else None
        if path_column is None:
            raise ValueError("testmon database file_fp table has no filename/path column")
        rows = connection.execute(f"SELECT {path_column} FROM file_fp").fetchall()
    finally:
        connection.close()

    return frozenset(
        relative
        for row in rows
        if row and isinstance(row[0], str)
        if (relative := _relative_path(row[0], source_root=source_root)) is not None
    )


def _is_docstring(node: ast.stmt, *, first: bool) -> bool:
    return (
        first
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _statement_is_executable(node: ast.stmt) -> bool:
    """Return whether a statement can carry runtime validation behavior."""
    if isinstance(node, (ast.Pass, ast.Import, ast.ImportFrom)):
        return False
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and (isinstance(node.value.value, str) or node.value.value is Ellipsis)
    ):
        return False
    if isinstance(node, ast.AnnAssign) and node.value is None:
        return False
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.decorator_list:
            return True
        if node.args.defaults or any(default is not None for default in node.args.kw_defaults):
            return True
        return _body_is_executable(node.body)
    if isinstance(node, ast.ClassDef):
        if node.decorator_list:
            return True
        if node.bases or node.keywords:
            return True
        return _body_is_executable(node.body)
    if isinstance(node, ast.Assign):
        return _expression_is_runtime(node.value)
    if isinstance(node, ast.AnnAssign):
        return node.value is not None and _expression_is_runtime(node.value)
    return True


def _expression_is_runtime(node: ast.expr) -> bool:
    """Conservatively identify assignment expressions with runtime effects."""
    if isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
        return False
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_expression_is_runtime(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return any(
            (key is not None and _expression_is_runtime(key)) or _expression_is_runtime(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return True


def _body_is_executable(body: list[ast.stmt]) -> bool:
    for index, node in enumerate(body):
        if _is_docstring(node, first=index == 0):
            continue
        if _statement_is_executable(node):
            return True
    return False


def classify_source_ast(source_path: Path) -> ASTClassification:
    """Classify source by executable AST content, with read and parse failures as risk."""
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except OSError:
        return "source-unreadable"
    except (SyntaxError, UnicodeDecodeError):
        return "executable"
    return "executable" if _body_is_executable(tree.body) else "declaration-only"


def audit_blind_spots(
    *,
    coverage_json_path: Path,
    testmon_db_path: Path,
    source_root: Path,
) -> BlindSpotReport:
    """Compare coverage-known files with testmon fingerprints without mutation."""
    coverage_files = read_coverage_files(coverage_json_path, source_root=source_root)
    fingerprints = read_testmon_fingerprints(testmon_db_path, source_root=source_root)
    findings: list[BlindSpotFinding] = []
    for coverage_file in coverage_files:
        source_path = source_root / coverage_file.path
        readable = source_path.is_file()
        classification = classify_source_ast(source_path)
        fingerprinted = coverage_file.path in fingerprints
        if not readable or classification == "source-unreadable":
            status: FindingStatus = "source-unreadable"
            safe = False
        elif fingerprinted:
            status = "fingerprinted"
            safe = True
        elif classification == "declaration-only":
            status = "declaration-only-unfingerprinted"
            safe = True
        else:
            status = "executable-validator-unfingerprinted"
            safe = False
        findings.append(
            BlindSpotFinding(
                path=coverage_file.path,
                statements=coverage_file.statements,
                covered_lines=coverage_file.covered_lines,
                ast_classification=classification,
                testmon_fingerprinted=fingerprinted,
                status=status,
                safe=safe,
            )
        )
    return BlindSpotReport(
        findings=tuple(findings),
        coverage_file_count=len(coverage_files),
        testmon_fingerprint_count=len(fingerprints),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit existing coverage metadata against pytest-testmon fingerprints without running tests."
    )
    parser.add_argument("--coverage-json", type=Path, default=Path(".cache/coverage/coverage.json"))
    parser.add_argument("--testmon-db", type=Path, default=Path(".cache/testmon/testmondata"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="Emit the complete audit report as JSON.")
    args = parser.parse_args(argv)
    report = audit_blind_spots(
        coverage_json_path=args.coverage_json,
        testmon_db_path=args.testmon_db,
        source_root=args.source_root,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for finding in report.findings:
            print(f"{finding.status:>38}  {finding.path}")
        print(f"coverage files: {report.coverage_file_count}; testmon fingerprints: {report.testmon_fingerprint_count}")
        print(f"risk findings: {len(report.risks)}")
    return 1 if report.risks else 0


if __name__ == "__main__":
    raise SystemExit(main())
