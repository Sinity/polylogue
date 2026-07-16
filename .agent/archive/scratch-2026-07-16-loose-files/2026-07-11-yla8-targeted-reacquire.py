#!/usr/bin/env python3
"""Reacquire only still-missing yla8.6 repair cursors through the live batch route."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from polylogue import Polylogue
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import _PARSER_FINGERPRINT, default_sources


def _candidate_paths(census_path: Path) -> set[Path]:
    census: dict[str, Any] = json.loads(census_path.read_text(encoding="utf-8"))
    if census.get("schema") != "polylogue.yla8-production-census.v1":
        raise ValueError("unsupported census schema")
    return {
        Path(str(row["source_path"])) for key in ("cursor_ahead_rows", "broken_head_cursors") for row in census[key]
    }


async def _run(archive_root: Path, census_path: Path) -> dict[str, Any]:
    cursor = CursorStore(archive_root / "index.db")
    candidates = _candidate_paths(census_path)
    with sqlite3.connect(f"file:{cursor._ops_db_path.resolve()}?mode=ro", uri=True) as conn:
        existing = {Path(str(row[0])) for row in conn.execute("SELECT source_path FROM ingest_cursor")}
    pending = sorted(candidates - existing)
    missing_files = [str(path) for path in pending if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"repair candidates disappeared: {missing_files}")
    if not pending:
        return {"pending_count": 0, "paths": [], "metrics": None}

    async with Polylogue(archive_root=archive_root) as polylogue:
        processor = LiveBatchProcessor(
            polylogue,
            default_sources(),
            cursor=cursor,
            parser_fingerprint=_PARSER_FINGERPRINT,
        )
        metrics = await processor.ingest_files(pending, emit_event=False)

    records = {path: cursor.get_record(path) for path in pending}
    invalid = [
        str(path)
        for path, record in records.items()
        if record is None
        or record.content_fingerprint is None
        or record.failure_count != 0
        or bool(record.excluded)
        or record.byte_offset > path.stat().st_size
    ]
    if invalid:
        raise RuntimeError(f"targeted reacquisition left invalid cursors: {invalid}")
    return {
        "pending_count": len(pending),
        "paths": [str(path) for path in pending],
        "metrics": asdict(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args.archive_root.resolve(), args.census.resolve())), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
