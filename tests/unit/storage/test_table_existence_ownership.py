"""One owner for "does this relation exist" (polylogue-grdt).

Sixteen named helpers each issued their own ``sqlite_master`` probe with five
different type-sets, so the same database answered differently depending on
which copy a caller reached: a view (``actions``, ``threads``) was present to
one and absent to another. ``core/sqlite_introspection`` now owns all three
questions -- table, table-or-view, view -- with one schema-attachment and
quoting policy behind them.

Anti-vacuity: reintroducing a module-level helper that runs its own
``sqlite_master``/``sqlite_schema`` type probe makes the census red; collapsing
``relation_exists`` or ``view_exists`` onto ``table_exists`` makes the
view-visibility assertions red.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import polylogue
from polylogue.core.sqlite_introspection import relation_exists, table_exists, trigger_exists, view_exists

_PROBE_MARKERS = ("sqlite_master WHERE type", "sqlite_schema WHERE type", "sqlite_temp_master")


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sessions (session_id TEXT)")
    conn.execute("CREATE VIEW actions AS SELECT session_id FROM sessions")
    return conn


def test_the_three_questions_have_three_distinct_answers() -> None:
    conn = _connection()
    assert table_exists(conn, "sessions") is True
    assert table_exists(conn, "actions") is False, "a view is not a table"
    assert relation_exists(conn, "actions") is True
    assert relation_exists(conn, "sessions") is True
    assert view_exists(conn, "actions") is True
    assert view_exists(conn, "sessions") is False, "a table is not a view"


def test_an_unattached_schema_is_absent_not_an_error() -> None:
    conn = _connection()
    assert table_exists(conn, "sessions", schema="nowhere") is False
    assert relation_exists(conn, "actions", schema="nowhere") is False
    assert view_exists(conn, "actions", schema="nowhere") is False


def test_triggers_are_asked_through_the_same_primitive() -> None:
    conn = _connection()
    assert trigger_exists(conn, "sessions_ai") is False
    conn.execute("CREATE TRIGGER sessions_ai AFTER INSERT ON sessions BEGIN SELECT 1; END")
    assert trigger_exists(conn, "sessions_ai") is True
    assert table_exists(conn, "sessions_ai") is False


def test_the_temp_schema_is_reachable_through_the_same_primitive() -> None:
    conn = _connection()
    assert table_exists(conn, "scratch", schema="temp") is False
    conn.execute("CREATE TEMP TABLE scratch (x)")
    assert table_exists(conn, "scratch", schema="temp") is True


def test_no_module_declares_its_own_relation_existence_probe() -> None:
    package_root = Path(polylogue.__file__).resolve().parent
    introspection = package_root / "core" / "sqlite_introspection.py"
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        if path == introspection:
            continue
        source = path.read_text(encoding="utf-8")
        if not any(marker in source for marker in _PROBE_MARKERS):
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "exists" not in node.name:
                continue
            body = ast.get_source_segment(source, node) or ""
            runs_query = any(
                isinstance(call.func, ast.Attribute) and call.func.attr in {"execute", "one"}
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
            if runs_query and any(marker in body for marker in _PROBE_MARKERS):
                offenders.append(f"{path.relative_to(package_root)}:{node.lineno} {node.name}")
    assert offenders == [], (
        "relation-existence probe re-declared; use polylogue.core.sqlite_introspection: " + ", ".join(offenders)
    )
