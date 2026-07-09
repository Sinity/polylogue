"""Read-only SQLite space report for archive schema-size work (#1486)."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from polylogue.paths import active_index_db_path as default_db_path
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

REPORT_VERSION = 1


def _scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _object_types(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'index')
        """
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def _category(name: str, object_type: str) -> str:
    if object_type == "index":
        return "index"
    if name.startswith("sqlite_"):
        return "sqlite_internal"
    if name.startswith("raw_") or name in {"artifact_observations"}:
        return "raw"
    if "fts" in name or name.endswith("_docsize") or name.endswith("_config"):
        return "fts"
    if name.startswith(("session_", "work_", "cost_")):
        return "insights"
    if name.startswith(("message_embeddings", "embedding_")):
        return "embeddings"
    if name.startswith("gc_generations"):
        return "blob_gc"
    return "archive"


def _safe_dbstat_rows(conn: sqlite3.Connection) -> tuple[list[dict[str, object]], str | None]:
    object_types = _object_types(conn)
    try:
        rows = conn.execute(
            """
            SELECT name, SUM(pgsize) AS bytes, COUNT(*) AS pages
            FROM dbstat
            GROUP BY name
            ORDER BY bytes DESC, name
            """
        ).fetchall()
    except sqlite3.Error as exc:
        return [], f"dbstat_unavailable: {exc}"

    items: list[dict[str, object]] = []
    for row in rows:
        name = str(row[0])
        object_type = object_types.get(name, "sqlite_internal" if name.startswith("sqlite_") else "unknown")
        bytes_value = int(row[1] or 0)
        items.append(
            {
                "name": name,
                "type": object_type,
                "category": _category(name, object_type),
                "bytes": bytes_value,
                "mb": round(bytes_value / 1_048_576, 3),
                "pages": int(row[2] or 0),
            }
        )
    return items, None


def _int_value(value: object) -> int:
    return int(cast(int | float | str | bytes | bytearray, value))


def _category_totals(items: list[dict[str, object]]) -> dict[str, dict[str, int | float]]:
    totals: dict[str, dict[str, int | float]] = {}
    for item in items:
        category = str(item["category"])
        current = totals.setdefault(category, {"bytes": 0, "objects": 0})
        current["bytes"] = _int_value(current["bytes"]) + _int_value(item["bytes"])
        current["objects"] = _int_value(current["objects"]) + 1
    for value in totals.values():
        value["mb"] = round(_int_value(value["bytes"]) / 1_048_576, 3)
    return dict(sorted(totals.items(), key=lambda kv: (-_int_value(kv[1]["bytes"]), kv[0])))


def build_space_report(db: Path, *, limit: int = 25, include_objects: bool = False) -> dict[str, object]:
    """Return a read-only space report for ``db``."""

    if not db.exists():
        return {
            "ok": False,
            "report_version": REPORT_VERSION,
            "db_path": str(db),
            "error": "database_not_found",
        }
    with closing(open_readonly_connection(db)) as conn:
        page_size = _scalar_int(conn, "PRAGMA page_size")
        page_count = _scalar_int(conn, "PRAGMA page_count")
        freelist_count = _scalar_int(conn, "PRAGMA freelist_count")
        objects, dbstat_error = _safe_dbstat_rows(conn) if include_objects else ([], "dbstat_skipped")
    file_bytes = db.stat().st_size
    allocated_bytes = page_size * page_count
    freelist_bytes = page_size * freelist_count
    return {
        "ok": True,
        "report_version": REPORT_VERSION,
        "db_path": str(db),
        "file_bytes": file_bytes,
        "file_mb": round(file_bytes / 1_048_576, 3),
        "allocated_bytes": allocated_bytes,
        "allocated_mb": round(allocated_bytes / 1_048_576, 3),
        "freelist_bytes": freelist_bytes,
        "freelist_mb": round(freelist_bytes / 1_048_576, 3),
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "dbstat_available": include_objects and dbstat_error is None,
        "dbstat_error": dbstat_error,
        "object_scan_requested": include_objects,
        "category_totals": _category_totals(objects),
        "objects": objects[: max(1, limit)],
        "object_count": len(objects),
    }


def _print_human(report: dict[str, object]) -> None:
    if not report.get("ok"):
        print(f"archive-space-report: {report.get('error')} ({report.get('db_path')})")
        return
    print(f"Archive space report: {report['db_path']}")
    print(
        f"  file: {report['file_mb']} MiB; allocated: {report['allocated_mb']} MiB; "
        f"freelist: {report['freelist_mb']} MiB"
    )
    print("")
    print("Categories:")
    category_totals = cast(dict[str, dict[str, Any]], report["category_totals"])
    for category, data in category_totals.items():
        print(f"  {category:16s} {data['mb']:>10} MiB  {data['objects']:>4} objects")
    if not report["object_scan_requested"]:
        print("\nObject scan skipped. Re-run with --objects for dbstat table/index bytes.")
        return
    if not report["dbstat_available"]:
        print(f"\ndbstat unavailable: {report['dbstat_error']}")
        return
    print("")
    print("Largest objects:")
    objects = cast(list[dict[str, Any]], report["objects"])
    for item in objects:
        print(f"  {item['name'][:48]:48s} {item['mb']:>10} MiB  {item['type']}/{item['category']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report SQLite archive space by table/index using dbstat.")
    parser.add_argument("--db", type=Path, default=default_db_path(), help="Archive database path.")
    parser.add_argument("--limit", type=int, default=25, help="Largest object rows to include.")
    parser.add_argument("--objects", action="store_true", help="Run the dbstat table/index object scan.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = build_space_report(args.db, limit=args.limit, include_objects=args.objects)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
