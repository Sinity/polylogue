"""Raw materialization hands its current output to canonical session derivation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.session_profile_composition import compose_session_profile_callback
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.raw_owner_routes import converge_pending_raws_with_owner


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


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.mark.asyncio
async def test_raw_materialization_hands_current_output_to_the_canonical_session_derivation(tmp_path: Path) -> None:
    """Raw admission targets its actual output after releasing the writer lease.

    Anti-vacuity: removing the raw-to-session query or calling the profile
    owner before raw publication leaves the materialized session without its
    canonical profile partition. Repeating the handoff proves that inspection
    rather than the intake item is the source of idempotence.
    """
    archive_root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, archive_root)

    def acquire() -> str:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session("raw-profile-handoff", (("user", "question"), ("assistant", "answer"))),
                source_path="raw-profile-handoff.jsonl",
                canonical_source_path="raw-profile-handoff.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await asyncio.to_thread(acquire)
    result = await asyncio.to_thread(
        converge_pending_raws_with_owner,
        archive_root,
        limit=1,
    )
    assert result.done == 1 and result.failed == 0
    session_ids = daemon_cli._raw_materialized_session_ids(archive_root, raw_id)
    assert session_ids == ("codex-session:raw-profile-handoff",)

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
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


@pytest.mark.asyncio
async def test_two_accepted_revisions_survive_one_periodic_profile_pass(tmp_path: Path) -> None:
    """Accepted R1/R2 marker inputs survive a coalesced profile publication.

    Two retained revisions of one Codex session each carry a note marker. They
    publish through the canonical raw-observation route in arrival order.
    """
    archive_root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, archive_root)
    from tests.infra.retained_jsonl import acquire_full_revision

    source_path = tmp_path / "coalesced-profile.jsonl"
    for revision, note in ((1, "first retained note"), (2, "second retained note")):

        def acquire(revision: int = revision, note: str = note) -> str:
            with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
                return acquire_full_revision(
                    archive,
                    provider=Provider.CODEX,
                    source_path=source_path,
                    payload=_codex_session(
                        "coalesced-profile", (("user", "question"), ("assistant", f"::note: {note}"))
                    ),
                    native_id="coalesced-profile",
                    generation=revision - 1,
                    acquired_at_ms=revision,
                )

        await asyncio.to_thread(acquire)
        # Each call is a fresh pass with no carried cursor, so its discovery
        # bound must reach past the already-current earlier revision.
        result = await asyncio.to_thread(
            converge_pending_raws_with_owner,
            archive_root,
            limit=revision,
        )
        assert result.done == 1 and result.failed == 0

    with sqlite3.connect(archive_root / "source.db") as source:
        retained = source.execute("SELECT sequence, raw_id FROM accepted_marker_inputs ORDER BY sequence").fetchall()
        assert len(retained) == 2
        assert retained[0][0] < retained[1][0]

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    try:
        composed = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        periodic = await composed.callback(None)
        assert periodic.outcomes
        # Marker delivery intentionally advances one accepted source batch per
        # transaction. The next periodic tick consumes R2 without republishing
        # the already-current profile of the coalesced index revision.
        second = await composed.callback(None)
        assert second.outcomes
        assert all(item.key.domain != "session_profile" for item in second.outcomes)
        with sqlite3.connect(archive_root / "index.db") as index:
            assert index.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", ("codex-session:coalesced-profile",)
            ).fetchone() == (1,)
            assert (
                index.execute(
                    "SELECT 1 FROM session_profile_demand WHERE session_id = ?", ("codex-session:coalesced-profile",)
                ).fetchone()
                is None
            )
        with sqlite3.connect(archive_root / "user.db") as user:
            bodies = [str(row[0]) for row in user.execute("SELECT body_text FROM assertions ORDER BY body_text")]
            assert any("first retained note" in body for body in bodies)
            assert any("second retained note" in body for body in bodies)
        unchanged = await composed.callback(None)
        assert unchanged.made_no_publication_attempts
        assert unchanged.work.inspected == 0
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
            canonical_source_path="split.json",
            acquired_at_ms=1,
        )
    result = converge_pending_raws_with_owner(archive_root, limit=1)
    assert result.done == 1 and result.failed == 0
    with sqlite3.connect(archive_root / "index.db") as index:
        index.execute("DELETE FROM raw_revision_heads WHERE session_id = ?", ("chatgpt-export:stale",))
        index.commit()

    assert daemon_cli._raw_materialized_session_ids(archive_root, raw_id) == ("chatgpt-export:active",)
