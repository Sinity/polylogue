"""Static guard: no user-input string interpolation in SQL execute() calls.

`polylogue/storage/` is the only module tree allowed to call `.execute()`
against SQLite connections. Every f-string, %-format, or `+` concatenation
inside an execute call risks SQL injection unless the interpolated values
are *trusted compile-time identifiers* (table names, savepoint counters,
schema-version literals).

This test:

1. Parses every Python file under ``polylogue/storage/`` with the standard
   ``ast`` module.
2. Visits every ``conn.execute(...)``/``conn.executemany(...)`` /
   ``cursor.execute(...)`` call (or ``await conn.execute(...)``).
3. Allows ``ast.Constant`` strings (the normal case) and an explicit
   whitelist of audited identifier-interpolation sites.
4. Fails for every f-string, ``%``-format, ``str.format``, or ``+``-concat
   first argument that is not on the whitelist — the test message points at
   the file, line, and the actual source so the offending site is obvious.

The whitelist (``_AUDITED_SITES``) is keyed by ``(relative_path, line_no)``
and records *why* a given interpolation is safe. New interpolations must be
audited and added explicitly; the gate forces that session rather than
allowing silent drift.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_STORAGE_ROOT: Final = Path("polylogue/storage")

# Audited identifier-interpolation sites. The values are not used at runtime;
# they document why the interpolation is safe and survive renames via grep.
#
# When adding an entry, paste the literal source line into the comment so a
# reviewer can confirm the interpolated value is a trusted identifier (table
# name from a closed set, schema-version literal, savepoint counter, etc.)
# and NOT user input that could ever flow through filters/CLI/MCP.
# Empty by default — the `_TRUSTED_IDENTIFIER_NAMES` mechanism handles every
# currently-safe interpolation. This dict exists for the rare future case
# where an interpolation site cannot be expressed as a single trusted-name
# reference (e.g. an inline expression). Each entry MUST carry a rationale.
_AUDITED_SITES: Final[dict[tuple[str, int], str]] = {
    # Chunked IN-clause: scoped_sql is a trusted compile-time SQL template (a
    # literal with a ``{}`` for the placeholder list); ``placeholders`` is a
    # generated ``?,?`` string sized to the bound ``chunk``; values flow through
    # bound params, never the format string.
    ("polylogue/storage/repository/archive/sessions.py", 300): (
        "chunked IN-clause: literal scoped_sql template + '?,?' placeholders, values bound"
    ),
    # #1743 readiness fallback degraded row count:
    #   degraded_row = self._conn.execute(f"SELECT ... FROM {table_name} ... WHERE ({any_terms}){clause}", ...)
    # ``table_name`` is a closed insight table name; ``any_terms`` is built
    # immediately above from closed ``(column, path)`` fallback specs; ``clause``
    # values are bound.
    ("polylogue/storage/sqlite/archive_tiers/archive.py", 2584): (
        "readiness fallback reason counts: closed insight-table + _INSIGHT_FALLBACK_PAYLOAD column/path; values bound"
    ),
    # #1743 readiness fallback reason breakdown:
    #   rows = self._conn.execute(f"... FROM {table_name} ... json_each(... '{path}') ...{clause}")
    # ``table_name``/``column``/``path`` are closed fallback specs; ``clause``
    # values are bound.
    ("polylogue/storage/sqlite/archive_tiers/archive.py", 2593): (
        "readiness fallback reason breakdown: closed insight-table + fallback payload column/path; values bound"
    ),
    # Terminal unit row readers:
    #   rows = self._conn.execute(f"""... WHERE ({clause}){session_clause} ...""", ...)
    # ``clause`` comes from the structural predicate lowerer, which returns SQL
    # fragments plus bound params from closed field metadata; ``session_clause``
    # comes from the shared session filter lowerer; pagination values are bound.
    ("polylogue/storage/sqlite/archive_tiers/archive.py", 3252): (
        "message query rows: structural predicate/session filter fragments from lowerers; values bound"
    ),
    ("polylogue/storage/sqlite/archive_tiers/archive.py", 3314): (
        "action query rows: structural predicate/session filter fragments from lowerers; values bound"
    ),
    ("polylogue/storage/sqlite/archive_tiers/archive.py", 3374): (
        "block query rows: structural predicate/session filter fragments from lowerers; values bound"
    ),
    ("polylogue/storage/sqlite/archive_tiers/archive.py", 3437): (
        "assertion query rows: structural predicate/session filter fragments from lowerers; values bound"
    ),
    # Assertion readers:
    #   rows = conn.execute(f"SELECT {_ASSERTION_COLUMNS} FROM assertions ...", ...)
    # ``_ASSERTION_COLUMNS`` is a module-level literal projection list; all
    # dynamic row values are passed through bound parameters.
    ("polylogue/storage/sqlite/archive_tiers/user_write.py", 657): (
        "assertion kind listing: literal projection column list; kind value bound"
    ),
    ("polylogue/storage/sqlite/archive_tiers/user_write.py", 671): (
        "assertion kind/key lookup: literal projection column list; values bound"
    ),
    ("polylogue/storage/sqlite/archive_tiers/user_write.py", 930): (
        "assertion id lookup: literal projection column list; assertion id bound"
    ),
    ("polylogue/storage/sqlite/archive_tiers/user_write.py", 947): (
        "assertions for target: literal projection column list; target ref bound"
    ),
    ("polylogue/storage/sqlite/archive_tiers/user_write.py", 952): (
        "assertions for target/kind: literal projection column list; values bound"
    ),
}


def _execute_call_target(node: ast.Call) -> str | None:
    """Return the canonical attribute name if ``node`` is a *.execute()/executemany() call."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in {"execute", "executemany", "executescript"}:
        return func.attr
    return None


class _SqlExecuteVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[tuple[str, int, str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        target = _execute_call_target(node)
        if target is not None and node.args:
            first = node.args[0]
            violation = _classify_first_arg(first)
            if violation is not None:
                rel = self.path.as_posix()
                if (rel, node.lineno) not in _AUDITED_SITES:
                    source_line = _read_line(self.path, node.lineno)
                    self.violations.append((rel, node.lineno, violation, source_line))
        self.generic_visit(node)


# Identifier names that, when used inside an f-string formatted-value, are
# treated as trusted compile-time identifiers rather than user input:
#
# * table / column / view / index / trigger names from closed-set constants
# * `placeholders` — always built as `", ".join("?" * n)` next to a `tuple`
#   of bound params, the canonical "variable IN (?, ?, ?)" pattern
# * savepoint / cursor counters (`sp_*`, `_depth`, `_counter`)
# * `SCHEMA_VERSION` literal
#
# A formatted-value referencing any OTHER name is flagged as a potential
# injection site. The intent is: every f-string SQL site must interpolate
# only audited identifier-shaped names; everything else must use `?`.
_TRUSTED_IDENTIFIER_NAMES: frozenset[str] = frozenset(
    {
        # container/self references — not data themselves, only path roots
        "self",
        "cls",
        "backend",
        # `placeholders` / `aid_placeholders` / etc. — `", ".join("?" * n)`
        "placeholders",
        "native_placeholders",
        "aid_placeholders",
        "values",
        "values_sql",
        "schema_version",
        "dimension",  # vec0 dimension is an int literal from config
        "table",
        "table_name",
        "status_table",
        "meta_table",
        "source_table",
        "freshness_table",
        "tablename",
        "column",
        "column_name",
        "columnname",
        "id_column",  # compile-time PK column name (e.g. "session_id")
        "key_column",
        "view",
        "view_name",
        "index",
        "index_name",
        "trigger",
        "trigger_name",
        "name",
        "sp_name",
        "sp_depth",
        "depth",
        "_transaction_depth",
        "transaction_depth",
        "counter",
        "fts_table",
        "stats_table",
        "fts_columns",
        "value_columns",
        "select_columns",
        "selected",
        "definition",
        # compile-time SELECT column-list constants (record projections)
        "_session_record_select",
        "_message_record_select",
        # cross-tier ATTACH schema name from a closed module-level tuple
        "schema_name",
        # archive correction filter clauses — all literal "col = ?" fragments,
        # values bound; built in feedback storage helper from a closed set
        "archive_clauses",
        # SQL fragments built by helper functions from closed-set inputs
        "set_clause",
        "set_clauses",
        "order_by",
        "where",
        "where_clause",
        "where_clauses",
        "clauses",
        "where_sql",
        "clause",
        "scope_clause",
        "scope_sql",
        "json_where",
        "join",  # attribute name of str.join() in `' AND '.join(where_clauses)`
        "join_clause",
        "filter_sql",
        "pagination",
        "bucket_format",
        "id_filter",
        "having_clause",
        "tag_clause",
        "effective_raw_provider_sql",
        "base_select",  # local literal SELECT template; dynamic values stay bound
        "quoted",  # double-quote escaped identifier from a closed table list
        "source_filter",  # local literal predicate fragment plus bound value
        "source_alias",  # attached schema alias returned by _ensure_source_tier_attached
        "schema",  # closed SQLite schema alias
        "tags_relation",  # archive-local table name or closed user/archive tag UNION relation
        "_tags_relation",  # instance copy of the same closed archive tag relation
        "target_table",  # closed user-state target mapping table name
        "target_column",  # closed user-state target mapping column name
        "spec",  # archive tier spec object; version is an internal int literal
        "version",
    }
)


class _NameCollector(ast.NodeVisitor):
    """Collect every ``Name.id`` and ``Attribute.attr`` reachable from a node."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, n: ast.Name) -> None:
        self.names.add(n.id.lower())

    def visit_Attribute(self, n: ast.Attribute) -> None:
        self.names.add(n.attr.lower())
        self.generic_visit(n)


def _collect_names(node: ast.expr) -> set[str]:
    walker = _NameCollector()
    walker.visit(node)
    return walker.names


def _formatted_value_uses_only_trusted_names(node: ast.JoinedStr) -> bool:
    """Every FormattedValue must reference only audited identifier-shaped names."""
    for part in node.values:
        if not isinstance(part, ast.FormattedValue):
            continue
        names = _collect_names(part.value)
        if not names:
            # purely computed expression — be conservative, flag it
            return False
        if not names <= _TRUSTED_IDENTIFIER_NAMES:
            return False
    return True


def _classify_first_arg(arg: ast.expr) -> str | None:
    """Return a violation kind for unsafe first-arg shapes, else None."""
    if isinstance(arg, ast.Constant):
        return None
    if isinstance(arg, ast.JoinedStr):
        if _formatted_value_uses_only_trusted_names(arg):
            return None
        return "f-string SQL"
    if isinstance(arg, ast.BinOp):
        if isinstance(arg.op, ast.Mod):
            return "%-format SQL"
        if isinstance(arg.op, ast.Add):
            return "string-concat SQL"
    if isinstance(arg, ast.Call):
        func = arg.func
        if isinstance(func, ast.Attribute) and func.attr == "format":
            # `<template>.format(name=value, ...)` — every keyword must be a
            # trusted identifier name, and every positional must be a literal
            # constant. Anything else is flagged.
            for kw in arg.keywords:
                if kw.arg is None:
                    return "str.format SQL with **kwargs"
                if kw.arg.lower() not in _TRUSTED_IDENTIFIER_NAMES:
                    return f"str.format SQL with non-trusted kwarg {kw.arg!r}"
            for pos in arg.args:
                if not (isinstance(pos, ast.Constant) and isinstance(pos.value, str)):
                    return "str.format SQL with non-literal positional arg"
            return None
    return None


def _read_line(path: Path, line_no: int) -> str:
    try:
        return path.read_text(encoding="utf-8").splitlines()[line_no - 1].strip()
    except (OSError, IndexError):
        return "<source unavailable>"


def test_no_unaudited_string_interpolated_sql() -> None:
    violations: list[tuple[str, int, str, str]] = []
    for path in sorted(_STORAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            raise AssertionError(f"{path} failed to parse: {exc}") from exc
        visitor = _SqlExecuteVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    assert not violations, (
        "Unaudited string-interpolated SQL in polylogue/storage/:\n"
        + "\n".join(f"  {rel}:{line} [{kind}]\n    {src}" for rel, line, kind, src in violations)
        + (
            "\n\nIf the interpolated value is a trusted compile-time identifier "
            "(table name, savepoint counter, schema-version literal), add an entry "
            "to _AUDITED_SITES in this file documenting why it is safe. Otherwise, "
            "rewrite the call to use bound parameters ('?')."
        )
    )


def test_audited_sites_are_real_violations() -> None:
    """Ensure _AUDITED_SITES does not develop stale entries.

    Every audited (path, line) must still parse to an execute() call with an
    unsafe first arg in the current tree; otherwise the whitelist is stale.
    """
    audited_by_path: dict[Path, set[int]] = {}
    for rel, line in _AUDITED_SITES:
        audited_by_path.setdefault(Path(rel), set()).add(line)

    stale: list[tuple[str, int]] = []
    for path, lines in audited_by_path.items():
        if not path.exists():
            stale.extend((path.as_posix(), line) for line in lines)
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _execute_call_target(node) is None or not node.args:
                continue
            if _classify_first_arg(node.args[0]) is not None:
                found.add(node.lineno)
        stale.extend((path.as_posix(), line) for line in lines if line not in found)

    assert not stale, (
        "Stale _AUDITED_SITES entries (no matching execute call at that line):\n"
        + "\n".join(f"  {rel}:{line}" for rel, line in stale)
        + "\nRemove the entry or update its line number after refactoring."
    )
