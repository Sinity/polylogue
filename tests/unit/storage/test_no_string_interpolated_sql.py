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
_AUDITED_SITES: Final[dict[tuple[str, int], str]] = {}


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
        "all_where",
        "clauses",
        "where_sql",
        "clause",
        "count_where",
        "any_terms",
        "session_clause",
        "group_expr",
        "from_sql",
        "order_clause",
        "order_direction",
        "path",
        "scope_clause",
        "scope_sql",
        "json_where",
        "join",  # attribute name of str.join() in `' AND '.join(where_clauses)`
        "join_clause",
        "filter_sql",
        "pagination",
        "bucket_format",
        "id_filter",
        "floor_filter",
        "pending_filter",
        "having_clause",
        "tag_clause",
        "limit_clause",
        "effective_raw_provider_sql",
        "base_select",  # local literal SELECT template; dynamic values stay bound
        "quoted",  # double-quote escaped identifier from a closed table list
        "quoted_schema",
        "source_filter",  # local literal predicate fragment plus bound value
        "source_where",
        "prefix_clause",  # local session-id prefix bounds predicate; values bound
        "prefix_sql",
        "source_alias",  # attached schema alias returned by _ensure_source_tier_attached
        "schema",  # closed SQLite schema alias
        "tags_relation",  # archive-local table name or closed user/archive tag UNION relation
        "_tags_relation",  # instance copy of the same closed archive tag relation
        "target_table",  # closed user-state target mapping table name
        "target_column",  # closed user-state target mapping column name
        "spec",  # archive tier spec object; version is an internal int literal
        "version",
        # schema-compatibility column names/projection fragments returned from
        # closed local helpers that inspect SQLite table shape.
        "origin_column",
        "native_id_column",
        "source_path_column",
        "source_index_column",
        "blob_size_column",
        "size_column",
        "parse_error_column",
        "validation_status_column",
        "acquired_at_column",
        "acquired_at_ms_column",
        "file_mtime_ms_column",
        "ref_id_column",
        "columns",
        "assignments",
        "raw_join",
        "alias_sql",
        "raw_filter",
        "origin_filter",
        "source_root_filter",
        "route_column",
        "route_value",
        "route_update",
        # embedding/search/read-model fragments from closed helper functions or
        # local literal templates; all external values stay bound.
        "status_select",
        "count_select",
        "messages_ref",
        "archive_messages_table_ref",
        "archive_embeddable_message_where",
        "materialization_selects",
        "tool_expr",
        "handler_expr",
        "status_expr",
        "_source_run_relation_sql",
        "_observed_event_relation_sql",
        "_source_context_snapshot_relation_sql",
        "_assertion_columns",
        "_work_event_select",
        "_work_event_from",
        "_phase_select",
        "_phase_from",
        "_where_origin",
        "select_parts",
        "total_cols",
        "total_predicate",
        "predicates",
        "limit",
        "_quote_identifier",
    }
)

_TRUSTED_SQL_HELPER_NAMES: frozenset[str] = frozenset(
    {
        "archive_embeddable_message_where",
        "archive_messages_table_ref",
        "_observed_event_relation_sql",
        "_where_origin",
        "format",
        "int",
        "max",
    }
)

_TRUSTED_SQL_HELPER_ARG_NAMES: frozenset[str] = frozenset(
    {
        "conn",
        "origin",
        "read_timeout",
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


def _expression_uses_only_trusted_sql_fragments(node: ast.expr) -> bool:
    """Return true when an interpolation expression references only trusted SQL fragments.

    Helper calls may mention ordinary values such as ``origin`` or ``conn`` as
    inputs to closed helpers, but those names are not trusted as direct SQL
    fragments. This keeps ``f"{origin}"`` flagged while allowing
    ``f"{_where_origin(origin)}"``.
    """

    names = _collect_names(node)
    if not names:
        return False
    if names <= _TRUSTED_IDENTIFIER_NAMES:
        return True
    allowed_arg_names = _TRUSTED_IDENTIFIER_NAMES | _TRUSTED_SQL_HELPER_ARG_NAMES | _TRUSTED_SQL_HELPER_NAMES
    if isinstance(node, ast.Subscript):
        return (
            _collect_names(node.value) <= _TRUSTED_IDENTIFIER_NAMES and _collect_names(node.slice) <= allowed_arg_names
        )
    if isinstance(node, ast.IfExp):
        return (
            _collect_names(node.test) <= allowed_arg_names
            and _expression_is_literal_or_trusted(node.body)
            and _expression_is_literal_or_trusted(node.orelse)
        )
    if isinstance(node, ast.Call):
        func_names = _collect_names(node.func)
        return (
            bool(func_names)
            and func_names <= (_TRUSTED_IDENTIFIER_NAMES | _TRUSTED_SQL_HELPER_NAMES)
            and all(_collect_names(arg) <= allowed_arg_names for arg in node.args)
            and all(kw.arg is not None and _collect_names(kw.value) <= allowed_arg_names for kw in node.keywords)
        )
    return False


def _expression_is_literal_or_trusted(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    return _expression_uses_only_trusted_sql_fragments(node)


def _formatted_value_uses_only_trusted_names(node: ast.JoinedStr) -> bool:
    """Every FormattedValue must reference only audited identifier-shaped names."""
    for part in node.values:
        if not isinstance(part, ast.FormattedValue):
            continue
        if not _expression_uses_only_trusted_sql_fragments(part.value):
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
                if isinstance(pos, ast.Constant) and isinstance(pos.value, str):
                    continue
                if isinstance(pos, ast.Name) and pos.id.lower() in _TRUSTED_IDENTIFIER_NAMES:
                    continue
                if _expression_uses_only_trusted_sql_fragments(pos):
                    continue
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
