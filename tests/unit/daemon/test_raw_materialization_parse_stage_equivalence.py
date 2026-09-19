"""Canonical preparation and retained prefetch produce equivalent replay content."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.parse_prefetch import DaemonParseStage
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.session_profile_composition import compose_session_profile_callback
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.intake_adapters import RawMaterializationDiscovery
from polylogue.operations.raw_observation_derivation import converge_raw_observations
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
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

    baseline_result = converge_raw_observations(
        baseline_root,
        source_roots=(),
        limit=100,
        max_payload_bytes=10_000_000,
    )
    assert baseline_result.failed == 0 and baseline_result.done == 4

    stage = DaemonParseStage(max_workers=2, max_inflight_bytes=10_000_000)
    try:
        warmed = stage.warm(_config(prefetch_root), limit=100, max_payload_bytes=10_000_000)
        assert warmed == 4
        backfill_historical_revision_evidence(
            prefetch_root,
            max_payload_bytes=10_000_000,
            prefetch_cache=stage.cache,
            pipeline_decode=False,
        )
    finally:
        stage.shutdown()
    # Every warmed entry was consumed by the census phase, not left stranded.
    assert len(stage.cache) == 0

    assert _canonical_snapshot(baseline_root) == _canonical_snapshot(prefetch_root)
    with _connect(baseline_root / "index.db") as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]) == 4


@pytest.mark.asyncio
async def test_raw_materialization_hands_current_output_to_the_canonical_session_derivation(tmp_path: Path) -> None:
    """Raw admission targets its actual output after releasing the writer lease.

    Anti-vacuity: removing the raw-to-session query or calling the profile
    owner before raw publication leaves the materialized session without its
    canonical profile partition. Repeating the handoff proves that inspection
    rather than the intake item is the source of idempotence.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session("raw-profile-handoff", (("user", "question"), ("assistant", "answer"))),
            source_path="raw-profile-handoff.jsonl",
            acquired_at_ms=1,
        )

    result = converge_raw_observations(
        archive_root,
        source_roots=(),
        limit=1,
        max_payload_bytes=10_000_000,
    )
    assert result.done == 1 and result.failed == 0
    session_ids = daemon_cli._raw_materialized_session_ids(archive_root, raw_id)
    assert session_ids == ("codex-session:raw-profile-handoff",)

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    try:
        composed = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        await daemon_cli._converge_raw_materialized_session_profiles(archive_root, raw_id, composed.callback)
        with _connect(archive_root / "index.db") as conn:
            first = tuple(
                conn.execute(
                    "SELECT session_id, input_content_hash FROM session_profiles WHERE session_id = ?",
                    session_ids,
                ).fetchall()
            )
        assert len(first) == 1

        await daemon_cli._converge_raw_materialized_session_profiles(archive_root, raw_id, composed.callback)
        with _connect(archive_root / "index.db") as conn:
            second = tuple(
                conn.execute(
                    "SELECT session_id, input_content_hash FROM session_profiles WHERE session_id = ?",
                    session_ids,
                ).fetchall()
            )
        assert second == first
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


def test_raw_materialized_session_ids_exclude_stale_component_sessions_without_current_heads(tmp_path: Path) -> None:
    """Raw-to-profile handoff follows authoritative heads, not residual session rows.

    Anti-vacuity: querying ``sessions`` by raw component alone includes the
    deliberately orphaned split member below and schedules a non-current
    session partition.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    payload: list[dict[str, object]] = [
        {
            "id": native_id,
            "title": native_id,
            "create_time": 1,
            "current_node": "m",
            "mapping": {
                "m": {
                    "id": "m",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "m",
                        "author": {"role": "user"},
                        "create_time": 1,
                        "content": {"content_type": "text", "parts": [native_id]},
                    },
                }
            },
        }
        for native_id in ("active", "stale")
    ]
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(payload).encode(),
            source_path="split.json",
            acquired_at_ms=1,
        )
    result = converge_raw_observations(
        archive_root,
        source_roots=(),
        limit=1,
        max_payload_bytes=10_000_000,
    )
    assert result.done == 1 and result.failed == 0
    with sqlite3.connect(archive_root / "index.db") as index:
        index.execute("DELETE FROM raw_revision_heads WHERE session_id = ?", ("chatgpt-export:stale",))
        index.commit()

    assert daemon_cli._raw_materialized_session_ids(archive_root, raw_id) == ("chatgpt-export:active",)


@pytest.mark.asyncio
async def test_whale_raw_materialization_hands_current_output_to_session_derivation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whale route gives canonical profile derivation its published raw output.

    Anti-vacuity: omitting the post-publication handoff leaves the real
    ``session_profiles`` relation empty even though the whale's canonical raw
    derivation wrote its session. This exercises the production whale owner,
    raw publication, output query, and session-profile adapter against one
    temporary SQLite archive.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session("whale-profile-handoff", (("user", "question"),)),
            source_path="whale-profile-handoff.jsonl",
            acquired_at_ms=1,
        )

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    try:
        bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())
        owner = RawObservationConvergenceOwner(
            archive_root,
            compute_adapter=compute,
            write_bridge=bridge,
            max_payload_bytes=1,
        )
        profiles = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=bridge,
            now=lambda: 0.0,
        )
        monkeypatch.setattr("polylogue.paths.archive_root", lambda: archive_root)
        monkeypatch.setattr(daemon_cli, "_RAW_MATERIALIZATION_DAEMON_BLOB_LIMIT_BYTES", 1)
        monkeypatch.setattr(daemon_cli, "_resolve_raw_materialization_whale_blob_limit_bytes", lambda: 1_000_000)
        monkeypatch.setattr(daemon_cli, "_drain_whale_receipt_outbox", lambda: asyncio.sleep(0))
        monkeypatch.setattr(daemon_cli, "_publish_whale_receipt", lambda **_kwargs: asyncio.sleep(0))

        assert await daemon_cli._maybe_run_raw_materialization_whale_pass(
            raw_observation_owner=owner,
            raw_intake_discovery=RawMaterializationDiscovery(archive_root, max_payload_bytes=1),
            session_profile_callback=profiles.callback,
        )
        with _connect(archive_root / "index.db") as conn:
            assert tuple(row[0] for row in conn.execute("SELECT session_id FROM sessions")) == (
                "codex-session:whale-profile-handoff",
            )
            assert tuple(row[0] for row in conn.execute("SELECT session_id FROM session_profiles")) == (
                "codex-session:whale-profile-handoff",
            )
            assert conn.execute("SELECT raw_id FROM sessions").fetchone()[0] == raw_id
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)
