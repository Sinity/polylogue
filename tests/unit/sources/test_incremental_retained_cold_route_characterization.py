"""Compare the production intake, retained replay and cold-build routes.

A synthetic, one-file comparison: given the same acquired raw (bytes and
revision identity), every route must publish the same source-tier governance
rows and the same index rows.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.compute import BoundedComputeAdapter
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
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.raw_owner_routes import ingest_files_with_owners


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
            "SELECT origin, source_path, blob_hash, parse_error, validation_status, logical_source_key, "
            "revision_kind, source_revision, acquisition_generation, revision_authority "
            "FROM raw_sessions WHERE raw_id = ?",
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


def _replay_until_valid(archive_root: Path, raw_id: str) -> None:
    """Drive retained replay the way the derivation kernel does.

    A byte-revision raw first owes its source classification; the next pass
    publishes the prepared carrier.
    """

    def replay(compute_adapter: BoundedComputeAdapter) -> bool:
        derivation = RawObservationDerivation(archive_root, compute_adapter=compute_adapter)
        for _attempt in range(3):
            frame = raw_observation_frame(archive_root)
            replacement = derivation.compute(frame, raw_id)
            try:
                with write_lease("test.retained-cold-route.publish", archive_root=archive_root):
                    derivation.publish(frame, replacement)
            finally:
                if replacement.scratch_owner is not None:
                    replacement.scratch_owner.cleanup()
            if derivation.inspect(raw_observation_frame(archive_root), (raw_id,))[raw_id] == "valid":
                return True
        return False

    async def run() -> bool:
        async with prepared_live_convergence_owner(archive_root) as owner:
            return await owner.run_convergence_sync("test.retained-cold-route", replay, owner._compute_adapter)

    if not asyncio.run(run()):
        raise AssertionError(f"retained replay did not converge for {raw_id}")


def test_live_retained_and_owned_cold_routes_publish_one_interpretation(tmp_path: Path) -> None:
    """One acquired raw publishes the same rows through every route owner.

    The retained arm acquires the raw with the revision identity live intake
    assigned, because governance (byte revision or membership census) follows
    that identity, not the route.

    Anti-vacuity: each arm must actually produce a successful live admission,
    a valid retained observation, or a promoted owned generation. Dropping
    retained enrichment from the live worker makes the live title the native
    id again; a route that writes membership governance for a byte-proven raw
    (or omits the authority census) breaks the source-row equality.
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "one.jsonl"
    payload = _payload()
    source_path.write_bytes(payload)
    source_path_string = str(source_path)

    live_root = tmp_path / "live"
    bootstrap_archive_root(live_root)
    live_metrics = asyncio.run(
        ingest_files_with_owners(_processor(live_root, source_root), [source_path], emit_event=False)
    )
    assert live_metrics.succeeded_file_count == 1, live_metrics
    live = _snapshot(live_root)

    live_raw = cast(tuple[object, ...], live["raw"])
    logical_key, revision_kind, source_revision, generation_number, authority = live_raw[5:10]
    assert revision_kind == RawRevisionKind.FULL.value
    retained_root = tmp_path / "retained"
    bootstrap_archive_root(retained_root)
    with ArchiveStore.open_existing(retained_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=source_path_string,
            canonical_source_path=source_path_string,
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                str(logical_key),
                RawRevisionKind(str(revision_kind)),
                str(source_revision),
                int(cast(int, generation_number)),
                authority=RawRevisionAuthority(str(authority)),
            ),
        )
    _replay_until_valid(retained_root, raw_id)
    retained = _snapshot(retained_root)

    cold_root = tmp_path / "cold"
    bootstrap_archive_root(cold_root)
    generation = ColdBuildGeneration.begin(
        cold_root,
        reason="synthetic route characterization",
        observed=ColdBuildGeneration.observe_source_baseline((WatchSource("codex", source_root),)),
    )
    register_cold_build_generation(generation)
    try:
        cold_metrics = asyncio.run(
            ingest_files_with_owners(_processor(cold_root, source_root), [source_path], emit_event=False)
        )
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

    assert live == cold, {key: (live[key], cold[key]) for key in live if live[key] != cold[key]}
    # A cold build has no live Source authority: its replay leaves the parse
    # unacknowledged, and promotion, its commit point, acknowledges it.
    # Anti-vacuity: drop the promotion stamp and ``cold`` keeps the
    # unparsed raw, so the live/cold equality above goes red.
    assert cold_before_promotion["raw_terminal"] == ((0, 0, 0, None),)
    assert {key: value for key, value in cold_before_promotion.items() if key != "raw_terminal"} == {
        key: value for key, value in cold.items() if key != "raw_terminal"
    }
    # One interpretation: live intake enriches from retained archive evidence
    # exactly as retained replay does, so title and content hash agree, and a
    # byte-proven raw is governed by its revision on every route.
    assert live == retained
    assert cast(tuple[tuple[object, ...], ...], live["sessions"])[0][3] == "synthetic route fixture"
    assert live["messages"] and live["blocks"]
    assert live["raw_memberships"] == ()
    assert live["membership_census"] == ()
    assert live["authority_census"]
