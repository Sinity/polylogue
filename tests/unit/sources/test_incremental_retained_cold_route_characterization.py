"""Characterize the production intake, retained replay and cold-build routes.

This is deliberately a synthetic, one-file comparison. It fixes the observable
surface future route unification must compare without claiming those routes
already have identical publication behavior.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from polylogue.core.enums import Provider
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _payload() -> bytes:
    records = (
        {
            "type": "session_meta",
            "payload": {"id": "route-characterization", "timestamp": "2026-06-02T00:00:00Z"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "message-0",
                "role": "user",
                "content": [{"type": "input_text", "text": "synthetic route fixture"}],
            },
        },
    )
    return b"".join(json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records)


def _processor(archive_root: Path, source_root: Path) -> LiveBatchProcessor:
    index_db = archive_root / "index.db"
    return LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=source_root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="route-characterization-parser",
    )


def _rows(db_path: Path, query: str) -> tuple[tuple[object, ...], ...]:
    with sqlite3.connect(db_path) as conn:
        return tuple(tuple(row) for row in conn.execute(query))


def _snapshot(archive_root: Path, index_path: Path | None = None) -> dict[str, object]:
    index = index_path or archive_root / "index.db"
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        raw_id = str(archive.source_connection.execute("SELECT raw_id FROM raw_sessions").fetchone()[0])
        raw = archive.source_connection.execute(
            "SELECT origin, source_path, blob_hash, parse_error, validation_status FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
        raw_memberships = tuple(
            tuple(row)
            for row in archive.source_connection.execute(
                "SELECT logical_source_key, provider_session_id, source_revision, "
                "normalized_content_hash, message_count, predecessor_raw_id, decision, revision_authority "
                "FROM raw_session_memberships WHERE raw_id = ? ORDER BY logical_source_key",
                (raw_id,),
            )
        )
        census = tuple(
            tuple(row)
            for row in archive.source_connection.execute(
                "SELECT parser_fingerprint, status, member_count, detail FROM raw_membership_census WHERE raw_id = ?",
                (raw_id,),
            )
        )
        authority = tuple(
            tuple(row)
            for row in archive.source_connection.execute(
                "SELECT parser_fingerprint, status, logical_keys_json, detail "
                "FROM raw_authority_parser_census WHERE raw_id = ?",
                (raw_id,),
            )
        )
    return {
        "raw": tuple(raw) if raw is not None else None,
        "raw_terminal": tuple(
            tuple(row)
            for row in _rows(
                archive_root / "source.db",
                "SELECT parsed_at_ms IS NOT NULL, parse_error IS NOT NULL, "
                "validated_at_ms IS NOT NULL, validation_status FROM raw_sessions",
            )
        ),
        "raw_memberships": raw_memberships,
        "membership_census": census,
        "authority_census": authority,
        "sessions": _rows(
            index,
            "SELECT session_id, origin, native_id, title, content_hash, parent_session_id, root_session_id "
            "FROM sessions ORDER BY session_id",
        ),
        "messages": _rows(
            index,
            "SELECT message_id, session_id, native_id, role, position, content_hash, material_origin "
            "FROM messages ORDER BY message_id",
        ),
        "blocks": _rows(
            index,
            "SELECT block_id, message_id, position, block_type, search_text, content_hash "
            "FROM blocks ORDER BY block_id",
        ),
        "links": _rows(index, "SELECT * FROM session_links ORDER BY 1"),
    }


def test_live_retained_and_owned_cold_routes_characterize_one_synthetic_input(tmp_path: Path) -> None:
    """Compare real route owners and pin the current, concrete route delta.

    Anti-vacuity: each arm must actually produce a successful live admission,
    a valid retained observation, or a promoted owned generation. Message,
    block, link, and terminal-state assertions prevent a vacuous comparison.
    Session rows (title, content hash) must be identical across routes;
    dropping retained enrichment from the live worker makes the live title
    the native id again and turns this red. The retained membership census
    remains a pinned, observed route delta.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "one.jsonl"
    payload = _payload()
    source_path.write_bytes(payload)
    source_path_string = str(source_path)

    live_root = tmp_path / "live"
    bootstrap_archive_root(live_root)
    live_metrics = asyncio.run(_processor(live_root, source_root).ingest_files([source_path], emit_event=False))
    assert live_metrics.succeeded_file_count == 1, live_metrics
    live = _snapshot(live_root)

    retained_root = tmp_path / "retained"
    bootstrap_archive_root(retained_root)
    with ArchiveStore.open_existing(retained_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=source_path_string,
            acquired_at_ms=1,
        )
    derivation = RawObservationDerivation(retained_root)
    frame = raw_observation_frame(retained_root)
    replacement = derivation.compute(frame, raw_id)
    try:
        assert derivation.publish(frame, replacement)
    finally:
        if replacement.scratch_owner is not None:
            replacement.scratch_owner.cleanup()
    assert derivation.inspect(raw_observation_frame(retained_root), (raw_id,))[raw_id] == "valid"
    retained = _snapshot(retained_root)

    cold_root = tmp_path / "cold"
    bootstrap_archive_root(cold_root)
    generation = ColdBuildGeneration.begin(
        cold_root, reason="synthetic route characterization", sources=(WatchSource("codex", source_root),)
    )
    register_cold_build_generation(generation)
    try:
        cold_metrics = asyncio.run(_processor(cold_root, source_root).ingest_files([source_path], emit_event=False))
        assert cold_metrics.succeeded_file_count == 1, cold_metrics
        candidate = Path(generation.generation.index_path)
        cold_before_promotion = _snapshot(cold_root, candidate)
        assert generation.session_count() == 1
        with ArchiveStore.open_existing(cold_root, read_only=True) as reader:
            assert reader._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        generation.promote()
    finally:
        clear_cold_build_generation()
        if not generation.settled:
            generation.discard()
    cold = _snapshot(cold_root)

    assert live == cold
    assert cold_before_promotion == cold
    assert live["raw"] == retained["raw"] == cold["raw"]
    assert live["raw_terminal"] == retained["raw_terminal"] == cold["raw_terminal"]
    for relation in ("messages", "blocks", "links"):
        assert live[relation] == retained[relation] == cold[relation]

    # One interpretation: live intake enriches from retained archive evidence
    # exactly as retained replay does, so title and content hash agree.
    assert live["sessions"] == retained["sessions"] == cold["sessions"]
    assert cast(tuple[tuple[object, ...], ...], live["sessions"])[0][3] == "synthetic route fixture"
    assert live["raw_memberships"] == cold["raw_memberships"] == ()
    assert len(cast(tuple[object, ...], retained["raw_memberships"])) == 1
    assert live["membership_census"] == cold["membership_census"] == ()
    assert len(cast(tuple[object, ...], retained["membership_census"])) == 1
