"""Exact semantic comparison with two independently certified recipe columns.

Only sessions.parser_fingerprint and sessions.lowering_fingerprint differ
under this contract. Every row's stored identity must match the actual
candidate functions; every other field and FTS posting remains exact.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

RECIPE_COLUMNS = ("parser_fingerprint", "lowering_fingerprint")


def imported_recipe_binding() -> dict[str, Any]:
    """Called in the daemon at exit, using its imported candidate functions."""
    from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin

    candidate = Path(os.environ["POLYLOGUE_BENCH_CANDIDATE"]).resolve()
    functions = {}
    for function in (parser_fingerprint_for_origin, lowering_fingerprint):
        path = Path(function.__code__.co_filename).resolve()
        functions[function.__name__] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bound": path == candidate / "polylogue/sources/origin_specs.py",
        }
    if not all(row["bound"] for row in functions.values()):
        return {"outcome": "mismatch", "functions": functions}
    archive = Path(os.environ["POLYLOGUE_ARCHIVE_ROOT"])
    with closing(sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True)) as conn:
        origins = [row[0] for row in conn.execute("SELECT DISTINCT origin FROM sessions")]
    return {
        "outcome": "bound",
        "functions": functions,
        "parser_by_origin": {origin: parser_fingerprint_for_origin(origin) for origin in origins},
        "lowering": lowering_fingerprint(),
    }


def validate_recipe_rows(conn: sqlite3.Connection, binding: dict[str, Any]) -> bool:
    """Refuse null, stale or foreign identities before excluding two columns."""
    if binding.get("outcome") != "bound" or not binding.get("functions"):
        return False
    if set(binding["functions"]) != {"parser_fingerprint_for_origin", "lowering_fingerprint"}:
        return False
    if not all(row.get("bound") and row.get("sha256") for row in binding["functions"].values()):
        return False
    columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    if not {"origin", *RECIPE_COLUMNS} <= columns:
        return False
    for origin, parser, lowering in conn.execute(
        "SELECT DISTINCT origin, parser_fingerprint, lowering_fingerprint FROM sessions"
    ):
        if (
            not parser
            or parser != binding.get("parser_by_origin", {}).get(origin)
            or not lowering
            or lowering != binding.get("lowering")
        ):
            return False
    return True


def output_equivalence(before: dict[str, Any], after: dict[str, Any]) -> str | None:
    """Strict first; otherwise require the narrow certified semantic digests."""
    if before.get("digest") and before.get("digest") == after.get("digest"):
        return "exact"
    a, b = before.get("recipe_qualified", {}), after.get("recipe_qualified", {})
    if (
        all(row.get("validated") is True and row.get("columns") == list(RECIPE_COLUMNS) for row in (a, b))
        and a.get("digest")
        and a.get("digest") == b.get("digest")
    ):
        return "recipe-qualified-exact"
    return None
