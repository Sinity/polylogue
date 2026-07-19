"""Equivalence: parse-stage prefetch (flag on) vs. in-hold parse (flag off).

polylogue-m6tp phase (a). ``RawParsePrefetchCache`` is purely additive by
construction (see its docstring and the unit-level cache-hit/miss tests in
``tests/unit/sources/test_revision_backfill.py``); this test proves the
end-to-end claim against a real archive: running the SAME raw-materialization
convergence over the SAME fixture corpus, once with the daemon's
``DaemonParseStage`` warming the census parse ahead of time and once with no
prefetch cache at all, produces byte-identical durable archive content.

Production dependencies exercised: ``DaemonParseStage.warm`` (the actual
off-writer-hold pre-parse path) feeding ``polylogue.storage.repair.
repair_raw_materialization``'s ``prefetch_cache`` parameter (the actual
production plumbing the daemon conveyor uses), not a reimplementation of
either.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.daemon.parse_prefetch import DaemonParseStage
from polylogue.storage.repair import repair_raw_materialization
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

_VOLATILE_COLUMNS: dict[str, frozenset[str]] = {
    "raw_revision_heads": frozenset({"decided_at_ms"}),
    "raw_sessions": frozenset({"parsed_at_ms"}),
}


def _codex_session(native_id: str, messages: tuple[tuple[str, str], ...]) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-19T00:00:00Z"}}
    ]
    for position, (role, text) in enumerate(messages):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{position}",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text" if role == "user" else "output_text",
                            "text": text,
                        }
                    ],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _seed_corpus(root: Path) -> None:
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(4):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session(
                    f"session-{index}",
                    (("user", f"question {index}"), ("assistant", f"answer {index}")),
                ),
                source_path=f"corpus-{index}.jsonl",
                acquired_at_ms=index,
            )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_rows(conn: sqlite3.Connection, table: str) -> tuple[tuple[Any, ...], ...]:
    excluded = _VOLATILE_COLUMNS.get(table, frozenset())
    columns = tuple(
        row["name"] for row in conn.execute(f'PRAGMA table_xinfo("{table}")') if row["name"] not in excluded
    )
    quoted = ", ".join(f'"{column}"' for column in columns)
    rows = tuple(
        sorted(
            (
                tuple(bytes(value).hex() if isinstance(value, bytes) else value for value in row)
                for row in conn.execute(f'SELECT {quoted} FROM "{table}"')
            ),
            key=repr,
        )
    )
    return rows


def _canonical_snapshot(root: Path) -> dict[str, tuple[tuple[Any, ...], ...]]:
    snapshot: dict[str, tuple[tuple[Any, ...], ...]] = {}
    with _connect(root / "index.db") as conn:
        for table in ("sessions", "messages", "blocks", "raw_revision_heads"):
            snapshot[f"index.{table}"] = _table_rows(conn, table)
    with _connect(root / "source.db") as conn:
        for table in ("raw_sessions", "raw_authority_parser_census"):
            snapshot[f"source.{table}"] = _table_rows(conn, table)
    return snapshot


def test_flag_on_prefetch_and_flag_off_produce_identical_archive_content(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    prefetch_root = tmp_path / "prefetch"
    _seed_corpus(baseline_root)
    _seed_corpus(prefetch_root)

    # Flag OFF: parse happens entirely inside repair_raw_materialization,
    # exactly as production behaves today.
    baseline_result = repair_raw_materialization(
        _config(baseline_root),
        dry_run=False,
        raw_artifact_limit=100,
        max_payload_bytes=10_000_000,
    )
    assert baseline_result.success is True

    # Flag ON: warm the SAME candidates off any writer hold first, exactly as
    # the daemon's ``_maybe_warm_raw_materialization_parse_stage`` does, then
    # thread the warmed cache into the identical production entry point.
    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm(_config(prefetch_root), limit=100, max_payload_bytes=10_000_000)
        assert warmed == 4
        prefetch_result = repair_raw_materialization(
            _config(prefetch_root),
            dry_run=False,
            raw_artifact_limit=100,
            max_payload_bytes=10_000_000,
            prefetch_cache=stage.cache,
        )
    finally:
        stage.shutdown()
    assert prefetch_result.success is True
    # Every warmed entry was consumed by the census phase, not left stranded.
    assert len(stage.cache) == 0

    assert _canonical_snapshot(baseline_root) == _canonical_snapshot(prefetch_root)
    with _connect(baseline_root / "index.db") as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]) == 4
