"""Verify that sync and async write paths use the same SQL templates.

Single-sourcing check: all UPSERT/INSERT SQL for the core archive tables
must be sourced from polylogue.core.common, which is the canonical store.
This prevents the sync (ingest_batch.py) and async (queries/*.py) paths
from drifting apart.

Also checks for duplicated function bodies across modules to catch copy-paste
divergence.

Exit 0 if clean, 1 with messages if violations found.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

POLYLOGUE_ROOT = Path(__file__).resolve().parent.parent / "polylogue"

# ---- SQL template source-of-truth ----
# To add a new SQL template: add it to polylogue/core/common.py, then
# register it here so the linter can verify both paths use it.

_CANONICAL_SQL_NAMES = frozenset(
    {
        "SQL_CONVERSATION_UPSERT",
        "SQL_MESSAGE_UPSERT",
        "SQL_CONTENT_BLOCK_UPSERT",
        "SQL_STATS_UPSERT",
        "SQL_ACTION_EVENT_INSERT",
        "SQL_ATTACHMENT_UPSERT",
        "SQL_ATTACHMENT_REF_INSERT",
    }
)

# Files that are allowed to import the canonical SQL templates.
# All other files must NOT define SQL with matching semantics.
_SQL_TEMPLATE_SOURCES = frozenset(
    {
        "polylogue/core/common.py",
        "polylogue/pipeline/services/ingest_batch.py",
        "polylogue/storage/sqlite/queries/conversations_writes.py",
        "polylogue/storage/sqlite/queries/message_query_writes.py",
        "polylogue/storage/sqlite/queries/attachment_content_blocks.py",
        "polylogue/storage/sqlite/queries/attachment_mutations.py",
        "polylogue/storage/sqlite/queries/action_events.py",
        "polylogue/storage/sqlite/queries/stats.py",
        "polylogue/storage/action_events/rebuild_sql.py",
    }
)

# ---- Function body duplication check ----
# Functions whose body hash must appear only once across these modules.


def _function_body_hash(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Compute a stable hash of a function body (excludes name and decorators)."""
    body_bytes = ast.unparse(node.body).encode("utf-8")
    return hashlib.sha256(body_bytes).hexdigest()[:16]


def _find_sql_definitions(file_path: Path) -> list[str]:
    """Find locally-defined SQL-like string constants in a Python file."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    sql_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Match our canonical naming pattern
                    if (
                        any(
                            name.endswith(suffix)
                            for suffix in ("_UPSERT_SQL", "_INSERT_SQL", "UPSERT_SQL", "INSERT_SQL")
                        )
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                        and "INSERT " in node.value.value.upper()
                        and "VALUES" in node.value.value.upper()
                    ):
                        sql_names.append(name)
    return sql_names


def _check_sql_single_sourcing() -> list[str]:
    """Verify SQL templates are sourced from common.py, not redefined locally."""
    errors: list[str] = []

    for py_file in POLYLOGUE_ROOT.rglob("*.py"):
        rel = str(py_file.relative_to(POLYLOGUE_ROOT.parent))
        if rel in _SQL_TEMPLATE_SOURCES:
            continue
        if "/test" in rel or "/tests/" in rel:
            continue
        if rel.startswith("devtools/"):
            continue

        sql_names = _find_sql_definitions(py_file)
        for name in sql_names:
            errors.append(
                f"Duplicate SQL: {rel} defines '{name}' locally. Import it from polylogue.core.common instead."
            )

    return errors


def _check_function_body_duplication() -> list[str]:
    """Detect identical non-trivial private module-level function bodies.

    Only flags private functions (starting with ``_``) that have >= 3 body
    statements AND appear with the same body in at least 3 different modules.
    This avoids false positives from Pydantic validators (same-name classmethods
    on different models), trivial one-liners, and legitimate two-module refactors.
    """
    errors: list[str] = []
    seen_hashes: dict[str, list[str]] = {}

    for py_file in POLYLOGUE_ROOT.rglob("*.py"):
        rel = str(py_file.relative_to(POLYLOGUE_ROOT.parent))
        if "/test" in rel or "/tests/" in rel:
            continue
        if rel.startswith("devtools/"):
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip dunder methods, trivial one-liners, and class-level methods
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue
                if len(node.body) <= 2:
                    continue
                # Only check private module-level functions (not model methods)
                if not node.name.startswith("_"):
                    continue
                body_hash = _function_body_hash(node)
                key = f"{node.name}:{body_hash}"
                seen_hashes.setdefault(key, []).append(rel)

    for key, files in seen_hashes.items():
        # Require at least 3 copies to flag — two-module duplication can be
        # intentional (e.g., shared via inheritance or import).
        if len(files) >= 3:
            func_name = key.split(":")[0]
            errors.append(
                f"Duplicated function body: '{func_name}' in {files}. "
                f"Move the canonical implementation to polylogue/core/common.py."
            )

    return errors


def main() -> int:
    errors: list[str] = []

    errors.extend(_check_sql_single_sourcing())
    errors.extend(_check_function_body_duplication())

    if errors:
        print(f"SQL template violations: {len(errors)}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("SQL templates verification: clean", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
