"""Reconstruct an earlier durable source tier from the migration files.

Durable fixtures build an older schema by subtracting from the CURRENT DDL.
Every hand-written removal list went stale the moment a migration added a
table, index or column, so the set is derived here instead.

The archive format floor reset deleted the whole v2..v46 source chain, and
with it the two hand-maintained registries this module used to carry: the
retired tables introduced at slots 18-29, and the pre-v46
``raw_authority_blockers`` shape. Both could only re-create objects no shipped
migration introduces any more, which would have made a fixture staged at the
floor describe a schema this lineage never had. They are removed rather than
kept as no-ops; the derivation below is now the whole module.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

__all__ = ["reset_source_fixture_to_version"]


def reset_source_fixture_to_version(conn: sqlite3.Connection, version: int) -> None:
    """Remove every schema object a source migration above *version* creates.

    The fixtures build a "stale" tier from the CURRENT DDL, so they start out
    carrying tables, indexes and triggers that a later migration will create
    again. Hand-maintained removal lists went stale every time a migration was
    added; this derives the set from the migration files themselves.

    Only objects INTRODUCED above the version are removed: a later migration
    that rebuilds an existing table issues its own CREATE TABLE, and dropping
    that would delete a table the fixture must still have.
    """
    migrations = Path(__file__).parents[2] / "polylogue" / "storage" / "sqlite" / "migrations" / "source"
    table_pattern = re.compile(r"CREATE TABLE (?:IF NOT EXISTS )?([A-Za-z_][A-Za-z0-9_]*)")
    index_pattern = re.compile(r"CREATE (?:UNIQUE )?INDEX (?:IF NOT EXISTS )?([A-Za-z_][A-Za-z0-9_]*)")
    view_pattern = re.compile(r"CREATE VIEW (?:IF NOT EXISTS )?([A-Za-z_][A-Za-z0-9_]*)")
    rebuild_pattern = re.compile(
        r"(?:DROP TABLE (?:IF EXISTS )?([A-Za-z_][A-Za-z0-9_]*)"
        r"|ALTER TABLE ([A-Za-z_][A-Za-z0-9_]*)\s+RENAME)",
        re.I,
    )
    below: set[str] = set()
    above: list[tuple[str, str]] = []
    for path in sorted(migrations.glob("*.sql")):
        slot = int(path.name.split("_", 1)[0])
        text = path.read_text(encoding="utf-8")
        # A create-copy-drop-rename rebuild issues CREATE TABLE for a table it
        # does not introduce. Treat a name the same file also drops or renames
        # as a rebuild, not an introduction, or the reset deletes a table the
        # fixture is required to have (raw_sessions is rebuilt this way).
        rebuilt = {name for m in rebuild_pattern.finditer(text) for name in m.groups() if name}
        created = [("table", m.group(1)) for m in table_pattern.finditer(text) if m.group(1) not in rebuilt]
        created += [("index", m.group(1)) for m in index_pattern.finditer(text)]
        created += [("view", m.group(1)) for m in view_pattern.finditer(text)]
        if slot <= version:
            below.update(name for _kind, name in created)
        else:
            above.extend(created)
    # Columns a later migration adds are present in the current DDL too, so an
    # ALTER TABLE ADD COLUMN above the fixture version fails with "duplicate
    # column name" unless the column is removed first.
    column_pattern = re.compile(r"ALTER TABLE ([A-Za-z_][A-Za-z0-9_]*)\s+ADD COLUMN\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)
    columns_below: set[tuple[str, str]] = set()
    columns_above: list[tuple[str, str]] = []
    for path in sorted(migrations.glob("*.sql")):
        slot = int(path.name.split("_", 1)[0])
        found = [(m.group(1), m.group(2)) for m in column_pattern.finditer(path.read_text(encoding="utf-8"))]
        if slot <= version:
            columns_below.update(found)
        else:
            columns_above.extend(found)

    seen: set[str] = set()
    # A later migration may replace a view introduced by an earlier
    # migration.  Drop that current definition before removing columns it
    # references, then restore the latest historical definition below the
    # requested fixture version.
    replaced_views = {name for kind, name in above if kind == "view" and name in below}
    for kind, name in above:
        if kind == "view" and (name not in below or name in replaced_views) and name not in seen:
            seen.add(name)
            conn.execute(f"DROP VIEW IF EXISTS {name}")
    for kind, name in above:
        if name in below or name in seen:
            continue
        seen.add(name)
        if kind == "table":
            for dependent_kind in ("trigger", "index"):
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = ? AND sql LIKE ?", (dependent_kind, f"%{name}%")
                ).fetchall()
                for (object_name,) in rows:
                    conn.execute(f"DROP {dependent_kind.upper()} IF EXISTS {object_name}")
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        elif kind == "index":
            conn.execute(f"DROP INDEX IF EXISTS {name}")
    dropped_tables = {name for kind, name in above if kind == "table" and name not in below}
    for table, column in dict.fromkeys(columns_above):
        if (table, column) in columns_below or table in dropped_tables:
            continue
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column in existing:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    view_definition_pattern = re.compile(
        r"CREATE VIEW (?:IF NOT EXISTS )?([A-Za-z_][A-Za-z0-9_]*) AS\s+.*?;",
        re.I | re.S,
    )
    for view_name in sorted(replaced_views):
        historical_sql: str | None = None
        for path in sorted(migrations.glob("*.sql")):
            slot = int(path.name.split("_", 1)[0])
            if slot > version:
                continue
            for match in view_definition_pattern.finditer(path.read_text(encoding="utf-8")):
                if match.group(1) == view_name:
                    historical_sql = match.group(0)
        if historical_sql is None:
            raise AssertionError(f"no historical definition found for replaced source view: {view_name}")
        conn.executescript(historical_sql)
