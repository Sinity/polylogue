"""Verify ``tests/infra/`` helpers stay current with the live SQLite schema.

Background: ``tests/infra/storage_records.py`` and sibling helpers contain
hand-written SQL fragments that mirror production write paths (stats upsert,
identity-preserving repoint, etc.). When ``SCHEMA_VERSION`` bumps and new
tables are introduced (see #1208: v15 -> v16 -> v17 added
``user_marks`` / ``user_annotations``), helpers that reference those tables
silently break against any in-memory test connection that does not run the
full schema bootstrap. The breakage hides behind testmon selection until an
unrelated change invalidates the affected test set.

This lint closes that drift class:

1. Collect the set of tables declared by ``polylogue/storage/sqlite/schema_ddl_*.py``
   (the authoritative SCHEMA_VERSION-bound DDL surface).
2. Scan every ``tests/infra/*.py`` helper for SQL table references
   (``FROM <name>``, ``UPDATE <name>``, ``INSERT INTO <name>``,
   ``DELETE FROM <name>``).
3. Fail loudly when any referenced table is not present in the live schema.
   That asymmetric direction is the actionable one: a helper that targets a
   nonexistent / renamed table will crash at runtime against any DB built
   from the current schema, exactly the cliff #1208 documents.

The lint is intentionally narrow. It does NOT require every schema table
to appear in helpers — that direction would push us toward boilerplate
references for tables that test infra has no reason to touch.

This is a static check: no DB is opened, no Python imports beyond AST
parsing of the helper modules.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

from devtools import repo_root as _get_root

ROOT = _get_root()
SCHEMA_DDL_DIR = ROOT / "polylogue" / "storage" / "sqlite"
TEST_INFRA_DIR = ROOT / "tests" / "infra"

# Match CREATE TABLE [IF NOT EXISTS] <name>.
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
# Match CREATE VIRTUAL TABLE <name> USING fts5(...).
_CREATE_VTABLE_RE = re.compile(
    r"CREATE\s+VIRTUAL\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
# Table references in helper SQL. Keep these conservative — only match
# bare identifiers, not subqueries / quoted-identifier edge cases.
_REF_RES = (
    re.compile(r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE),
    re.compile(
        r"\bINSERT\s+(?:OR\s+(?:IGNORE|REPLACE|ABORT|FAIL|ROLLBACK)\s+)?INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE
    ),
    re.compile(r"\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE),
    re.compile(r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE),
)

# SQL keywords / common aliases that show up in match position but are not
# table names. Used to reduce false positives in the "FROM x" matcher.
_SQL_KEYWORDS = frozenset(
    {
        "select",
        "where",
        "values",
        "set",
        "into",
        "on",
        "as",
        "and",
        "or",
        "not",
        "null",
        "case",
        "when",
        "then",
        "else",
        "end",
    }
)

# Tables that exist outside the polylogue schema (SQLite built-ins, FTS5
# internals, etc.). Referencing them from helpers is legitimate.
_SQLITE_BUILTIN_TABLES = frozenset(
    {
        "sqlite_master",
        "sqlite_sequence",
        "sqlite_stat1",
        "sqlite_stat4",
        "sqlite_temp_master",
    }
)


def _collect_schema_tables() -> frozenset[str]:
    tables: set[str] = set()
    ddl_files = list(SCHEMA_DDL_DIR.glob("schema_ddl*.py"))
    # The archive DDL (source/index/embeddings/user/ops) is the active
    # storage shape; tests/infra helpers read those archive tables
    # (e.g. ``blocks``/``blocks_fts``), so the currency check must know them
    # alongside the top-level ``schema_ddl*.py`` surface.
    ddl_files.extend((SCHEMA_DDL_DIR / "archive_tiers").glob("*.py"))
    for ddl_file in ddl_files:
        text = ddl_file.read_text(encoding="utf-8")
        tables.update(name.lower() for name in _CREATE_TABLE_RE.findall(text))
        tables.update(name.lower() for name in _CREATE_VTABLE_RE.findall(text))
    return frozenset(tables)


_SQL_VERB_RE = re.compile(
    r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|REPLACE|UPSERT|WITH)\b",
    re.IGNORECASE,
)


def _looks_like_sql(literal: str) -> bool:
    """Heuristic: only scan literals that contain at least one SQL verb.

    Eliminates docstring / log-message false positives like "schema from".
    """
    return bool(_SQL_VERB_RE.search(literal))


def _iter_string_literals(text: str) -> list[tuple[str, int]]:
    """Return ``(literal_value, lineno)`` for every non-docstring string literal.

    Only string nodes are returned, so Python ``import`` statements and bare
    identifiers cannot generate false positives. SQL fragments live inside
    triple-quoted or single-line string literals in production helpers.

    Docstrings (the first ``Constant`` child of a module/class/function body)
    are skipped — they routinely contain English prose that happens to use
    SQL verbs like ``CREATE`` or ``WITH``.
    """
    results: list[tuple[str, int]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return results

    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", None) or []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_nodes:
                continue
            results.append((node.value, node.lineno))
    return results


def _collect_helper_table_refs() -> dict[str, set[tuple[Path, int]]]:
    """Return ``{table_name: {(helper_path, line_no), ...}}``.

    Helper modules express SQL as string literals (often triple-quoted). We
    parse each helper's AST, pull out string literals, and only then run the
    table-reference regex against those literals. That guarantees Python
    ``from x import y`` or ``UPDATE`` used as a method name never produces
    false positives.
    """
    refs: dict[str, set[tuple[Path, int]]] = {}
    for helper in sorted(TEST_INFRA_DIR.rglob("*.py")):
        if helper.name.startswith("test_"):
            # Tests under tests/infra/ are exercised by the suite itself;
            # the lint targets shared helper modules.
            continue
        text = helper.read_text(encoding="utf-8")
        for literal, lineno in _iter_string_literals(text):
            if not _looks_like_sql(literal):
                continue
            for matcher in _REF_RES:
                for name in matcher.findall(literal):
                    lowered = name.lower()
                    if lowered in _SQL_KEYWORDS:
                        continue
                    if lowered in _SQLITE_BUILTIN_TABLES:
                        continue
                    refs.setdefault(lowered, set()).add((helper, lineno))
    return refs


def _format_report(
    *,
    tables: frozenset[str],
    refs: dict[str, set[tuple[Path, int]]],
    missing: dict[str, set[tuple[Path, int]]],
) -> str:
    lines = [
        f"schema tables: {len(tables)}",
        f"helper table refs: {len(refs)}",
        f"helper refs without matching schema table: {len(missing)}",
    ]
    if missing:
        lines.append("")
        lines.append("Stale helper references:")
        for table, hits in sorted(missing.items()):
            for path, lineno in sorted(hits):
                rel = path.relative_to(ROOT)
                lines.append(f"  {rel}:{lineno} references unknown table {table!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    tables = _collect_schema_tables()
    refs = _collect_helper_table_refs()
    missing = {table: hits for table, hits in refs.items() if table not in tables}

    if args.json:
        payload = {
            "schema_tables": sorted(tables),
            "helper_table_refs": sorted(refs),
            "missing": {
                table: [{"path": str(path.relative_to(ROOT)), "line": lineno} for path, lineno in sorted(hits)]
                for table, hits in sorted(missing.items())
            },
            "ok": not missing,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(tables=tables, refs=refs, missing=missing))

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
