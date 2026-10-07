"""Codex state exports must materialize completely with bounded working sets.

Two defects, one logical export (polylogue-xrba4):

1. State rows and long text need bounded pages with addressable continuation.
2. The materials that route produces carry ``referrer_ref =
   codex-session:<thread>`` and own their bytes through
   ``material_observations.blob_hash`` alone -- no ``sessions`` row, no
   ``raw_sessions`` row, no ``blob_refs`` row. ``excise --session`` resolved
   raw targets only from ``sessions.raw_id``, so it left that content
   readable and re-admissible.

Fixtures are synthetic: invented thread uuids, invented objectives, no
operator paths and no transcript bytes.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tracemalloc
from builtins import BaseExceptionGroup
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.sql_settlement import retain_native_sql_lifetimes
from polylogue.core.stage_admission import admit_stage_write
from polylogue.sources.codex_state_evidence import CodexStateMaterializationReceipt, _materialize_codex_state_content
from polylogue.sources.parsers.codex_state import CODEX_STATE_MAX_TEXT_CHARS
from polylogue.sources.prepared_jsonl import PreparedJsonl, _prepare_codex_state_blob
from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived.raw import _cleanup_scratch
from polylogue.storage.index_generation import ActiveWriterLease
from polylogue.storage.materials import MaterialObservation, list_materials, list_materials_page, read_material
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.revision_governance import _PreparedSourceProducer
from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.excision import (
    plan_session_excision_from_root,
    resolve_session_excision_target_from_root,
)
from tests.infra.excision_execution import execute_excision
from tests.infra.live_ingest import prepared_live_convergence_owner

_THREAD_A = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
_THREAD_B = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
_SESSION_A = f"codex-session:{_THREAD_A}"
_SESSION_B = f"codex-session:{_THREAD_B}"


def _write_goals_db(path: Path, goals: list[tuple[str, str, str]]) -> None:
    """Write a synthetic ``goals_1.sqlite`` holding ``(thread, goal, objective)``."""
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE thread_goals (
                thread_id TEXT NOT NULL,
                goal_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                status TEXT NOT NULL,
                token_budget INTEGER,
                tokens_used INTEGER NOT NULL,
                time_used_seconds INTEGER NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO thread_goals VALUES (?, ?, ?, 'active', 100, 1, 2, 1000, 2000)",
            goals,
        )
        conn.commit()


async def _materialize_async(root: Path, state_path: Path, **limits: int) -> CodexStateMaterializationReceipt | None:
    async with prepared_live_convergence_owner(root) as owner:
        retained: list[PreparedIndexMutation] = []

        def prepare() -> CodexStateMaterializationReceipt | None:
            admit_stage_write("fixture.codex-material.bootstrap", lambda: bootstrap_archive_root(root))
            store = BlobStore(root / "blob")
            captured = snapshot_sqlite_to_blob(state_path, store)
            publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
            scratch = TemporaryDirectory(dir=store._ensure_private_staging_root(), prefix=".state-prepared-")
            prepared: PreparedJsonl | None = None
            seal: PreparedIndexMutation | None = None
            lease: ActiveWriterLease | None = None
            bound = False
            kind = "memories" if state_path.name == "memories_1.sqlite" else "goals"
            with retain_native_sql_lifetimes(scratch):

                def close_payload() -> None:
                    if prepared is not None:
                        prepared.discard()
                    publisher.discard_pending()
                    assert not retained_native_sql_owners_for_lifetime(scratch)
                    _cleanup_scratch(scratch)

                try:
                    seal = PreparedIndexMutation.source_only(archive_root=root)
                    retained.append(seal)
                    lease = ActiveWriterLease(root)
                    lease.acquire()
                    seal.retain_publication_lifetime(lease, close_payload)
                    bound = True
                    prepared = _prepare_codex_state_blob(
                        store.blob_path(captured.blob_hash),
                        Path(scratch.name),
                        state_kind=kind,
                        source_hash=captured.blob_hash,
                        semantic_source_path=f"/synthetic/codex/{state_path.name}",
                        text_chars=limits.pop("text_char_limit", CODEX_STATE_MAX_TEXT_CHARS),
                        publication_publisher=publisher,
                        captured_profile_key=captured.captured_profile_key,
                    )
                    prepared.publish_blobs(reference_seal=seal)
                    with seal.original_read_snapshot(), seal.source_producer():
                        receipt = _materialize_codex_state_content(
                            _PreparedSourceProducer(seal),
                            "raw-codex-state",
                            prepared_state=prepared,
                            source_path=f"/synthetic/codex/{state_path.name}",
                            state_kind=kind,
                            acquired_at_ms=5_000,
                            settle_page=check_compute_cancelled,
                            **limits,
                        )
                    permit = seal.prepare_source_mutation()

                    def publish() -> None:
                        with permit.hold_authority(), permit.mutation_connection() as connection:
                            with closing(connection.execute("BEGIN IMMEDIATE")):
                                pass
                            permit.apply_source_statements(connection)
                            permit.allow_commit(connection)
                            connection.commit()
                            assert seal is not None
                            seal.accept_known_tier_commit(permit.committed())

                    admit_stage_write("fixture.codex-material.publish", publish)
                    return receipt
                finally:
                    primary = sys.exception()
                    cleanup: list[BaseException] = []
                    actions = (
                        (seal.close,)
                        if bound and seal is not None
                        else (
                            *((seal.close,) if seal is not None else ()),
                            close_payload,
                            *((lease.close,) if lease is not None else ()),
                        )
                    )
                    for action in actions:
                        try:
                            action()
                        except BaseException as failure:
                            cleanup.append(failure)
                    if cleanup:
                        if primary is not None:
                            cleanup.insert(0, primary)
                        raise BaseExceptionGroup("material preparation and physical cleanup failed", cleanup)
                    if seal is not None:
                        retained.remove(seal)

        return await owner.run_prepared_sync(
            "fixture.codex-material.prepare",
            prepare,
            settlement_owners=lambda: tuple(retained),
            estimated_bytes=state_path.stat().st_size,
        )


def _materialize(root: Path, state_path: Path, **limits: int) -> CodexStateMaterializationReceipt | None:
    return asyncio.run(_materialize_async(root, state_path, **limits))


def test_state_materialization_continues_past_each_work_window(tmp_path: Path) -> None:
    """A small row and byte window must still publish every row and text part."""
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [(f"thread-{index:02d}", f"goal-{index:02d}", "x" * 500) for index in range(6)],
    )

    receipt = _materialize(root, goals_path, row_limit=2, text_char_limit=16, aggregate_byte_limit=1024)
    assert receipt is not None
    assert receipt.rows_available == 6
    assert receipt.rows_materialized == 6
    assert receipt.rows_declined_row_cap == 0
    assert receipt.rows_declined_byte_cap == 0
    assert receipt.clipped_item_ids == ()
    assert receipt.bounded is False
    assert receipt.bytes_materialized > receipt.aggregate_byte_cap
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        cursor = None
        observed: list[MaterialObservation] = []
        while True:
            page = list_materials_page(conn, after=cursor, limit=3)
            observed.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        assert len(observed) == 6 * (1 + 31)
        chunks = [json.loads(read_material(conn, item.material_id)) for item in observed]
        assert any(part.get("item_id") == "goal-05" and part.get("offset_chars") == 496 for part in chunks)
        assert all(part.get("text_continuation") or part.get("record_type") == "goals" for part in chunks)


@pytest.mark.timeout(0)
def test_large_goal_export_last_row_is_reachable(tmp_path: Path) -> None:
    """A valid row after the old 10,000-row limit survives materialization.

    This scale assertion opts out of a wall-clock kill: the materializer owns
    cooperative cancellation and reports row/byte progress while it runs.
    """
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [(f"thread-{index:05d}", f"goal-{index:05d}", "objective") for index in range(10_001)],
    )
    receipt = _materialize(root, goals_path)
    assert receipt is not None and receipt.rows_materialized == 10_001
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        last = list_materials_page(conn, evidence_ref="codex-session:thread-10000", limit=2)
        assert len(last.items) == 1
        assert json.loads(read_material(conn, last.items[0].material_id))["goal_id"] == "goal-10000"


def test_long_memory_text_reassembles_from_material_pages(tmp_path: Path) -> None:
    """Every character after the old clip is addressable through material reads."""
    root = tmp_path / "archive"
    root.mkdir()
    memory_path = tmp_path / "memories_1.sqlite"
    memory = "Ω" * 65_001
    with closing(sqlite3.connect(memory_path)) as conn, conn:
        conn.execute(
            "CREATE TABLE stage1_outputs (thread_id TEXT PRIMARY KEY, source_updated_at INTEGER, "
            "generated_at INTEGER, raw_memory TEXT, rollout_summary TEXT, usage_count INTEGER, "
            "rollout_slug TEXT, selected_for_phase2 INTEGER)"
        )
        conn.execute(
            "INSERT INTO stage1_outputs VALUES (?, 1, 2, ?, 'summary', 3, 'slug', 1)", ("thread-memory", memory)
        )
    receipt = _materialize(root, memory_path)
    assert receipt is not None and receipt.rows_materialized == 1 and not receipt.bounded
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        page = list_materials_page(conn, evidence_ref="codex-session:thread-memory", limit=2)
        assert len(page.items) == 2
        parts = [json.loads(read_material(conn, item.material_id)) for item in page.items]
    header = next(part for part in parts if "raw_memory" in part)
    suffix = next(part for part in parts if part.get("field") == "raw_memory")
    assert header["raw_memory"] + suffix["text"] == memory
    assert suffix["offset_chars"] == 64_000 and suffix["final"] is True


def test_parser_row_pages_do_not_retain_the_export(tmp_path: Path) -> None:
    """A larger export must not become one tuple or one retained payload list."""
    from polylogue.sources.parsers.codex_state import iter_codex_state_parts

    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [(f"thread-{i:04d}", f"goal-{i:04d}", "x" * 10_000) for i in range(1000)],
    )
    tracemalloc.start()
    try:
        count = sum(1 for _ in iter_codex_state_parts(goals_path, state_kind="goals", page_size=8))
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == 1000
    assert peak < 8 * 1024 * 1024


def test_invalid_goal_row_is_named_as_partial(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(goals_path, [("valid-thread", "valid-goal", "objective"), ("", "invalid-goal", "objective")])
    receipt = _materialize(root, goals_path, row_limit=1)
    assert receipt is not None
    assert receipt.rows_available == 2
    assert receipt.rows_materialized == 1
    assert receipt.rows_skipped_invalid == 1
    assert receipt.bounded
    assert "partial goal materialization" in receipt.as_detail()


def test_interrupted_state_projection_resumes_without_duplicate_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed preparation accepts no materials; complete replay publishes each row once."""
    from polylogue.sources import codex_state_evidence

    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(goals_path, [(f"thread-{i}", f"goal-{i}", "objective") for i in range(8)])
    original = codex_state_evidence._upsert_codex_material_source
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("synthetic interruption")
        original(*args, **kwargs)

    monkeypatch.setattr(codex_state_evidence, "_upsert_codex_material_source", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _materialize(root, goals_path, row_limit=2)
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        assert conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 0
    monkeypatch.setattr(codex_state_evidence, "_upsert_codex_material_source", original)
    receipt = _materialize(root, goals_path, row_limit=2)
    assert receipt is not None and receipt.rows_materialized == 8
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        assert conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 8


def test_excising_a_thread_removes_its_codex_state_materials(tmp_path: Path) -> None:
    """``excise --session`` must reach state materials retained for that thread.

    Anti-vacuity: removing the ``_session_material_targets`` call from
    ``resolve_session_excision_target`` (or the material deletion from
    the audited Excision operation) leaves thread A's material row, its blob hash
    absent from ``excised_content``, and its objective text still readable
    through ``list_materials``. Thread B's material is asserted untouched, so
    a resolver that simply deleted every material fails too.
    """
    root = tmp_path / "archive"
    root.mkdir()
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [
            (_THREAD_A, "goal-a", "synthetic objective for thread a"),
            (_THREAD_B, "goal-b", "synthetic objective for thread b"),
        ],
    )
    _materialize(root, goals_path)

    # Precondition: both threads' state evidence is retained and readable.
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        assert len(list_materials(conn, evidence_ref=_SESSION_A)) == 1
        assert len(list_materials(conn, evidence_ref=_SESSION_B)) == 1
        hash_a = bytes(
            conn.execute(
                "SELECT blob_hash FROM material_observations WHERE referrer_ref = ?",
                (_SESSION_A,),
            ).fetchone()[0]
        )

    # A session with no index row at all is still a real excision target when
    # state materials name it.
    target = resolve_session_excision_target_from_root(root, _SESSION_A)
    assert target.session_exists is False
    assert target.found is True
    assert len(target.material_ids) == 1

    plan = plan_session_excision_from_root(root, _SESSION_A)
    assert plan.source_materials == 1

    receipt = execute_excision(root, _SESSION_A, reason="operator request", actor="tests")
    assert receipt["found"] is True
    assert receipt["counts"]["source_materials"] == 1
    assert hash_a.hex() in receipt["removed_blob_hashes"]

    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        assert list_materials(conn, evidence_ref=_SESSION_A) == []
        # The other thread's evidence, in the same export, is untouched.
        assert len(list_materials(conn, evidence_ref=_SESSION_B)) == 1
        # The bytes are durably refused on re-acquisition, not merely unlinked.
        excised = {bytes(row[0]) for row in conn.execute("SELECT removed_hash FROM excised_content")}
        assert hash_a in excised


@pytest.mark.asyncio
async def test_retained_codex_goals_are_readable_through_every_public_session_route(tmp_path: Path) -> None:
    """A goal's objective and status are read back through CLI, API and MCP.

    The ``read --view materials`` CLI view lowers to ``session.read`` with
    ``kind="materials"``, executed here on the pinned archive; the facade and
    MCP ``get(projection="materials")`` must answer with the same rows.
    Anti-vacuity: drop ``"materials"`` from ``SESSION_EVIDENCE_PAGE_READERS`` and
    ``session.read`` refuses the kind; drop the ``referrer_ref`` filter from
    ``_session_material_rows`` and thread B's objective leaks into thread A;
    stop decoding the retained bytes and the objective text is absent.
    """
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import patch

    from polylogue import Polylogue
    from polylogue.core.enums import Provider, Role
    from polylogue.mcp.server import build_server
    from polylogue.operations.daemon_reads import execute_read_operation
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async

    root = tmp_path / "archive"
    root.mkdir()
    from tests.infra.archive_templates import run_archive_fixture_prepare
    from tests.infra.live_ingest import write_session_sync

    await run_archive_fixture_prepare(
        lambda: write_session_sync(
            root / "index.db",
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=_THREAD_A,
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="start")],
            ),
            archive_root=root,
        )
    )
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [
            (_THREAD_A, "goal-a", "synthetic objective for thread a"),
            (_THREAD_B, "goal-b", "synthetic objective for thread b"),
        ],
    )
    await _materialize_async(root, goals_path)

    walked: list[dict[str, Any]] = []
    with ArchiveStore.open_existing(root) as archive:
        page = execute_read_operation(
            "session.read",
            {"ref": f"session:{_SESSION_A}", "kind": "materials", "limit": 1, "offset": 0},
            archive=archive,
            serving_identity="test",
        )
        while True:
            window = cast("dict[str, Any]", page["evidence_window"])
            walked.extend(window["rows"])
            if window["continuation"] is None:
                assert window["complete"] is True
                assert window["total"] == len(walked)
                break
            page = execute_read_operation(
                "session.read",
                {"ref": f"session:{_SESSION_A}", "kind": "materials", "continuation": window["continuation"]},
                archive=archive,
                serving_identity="test",
            )

    goals = [row["content"] for row in walked if "goal_id" in row["content"]]
    assert [(goal["goal_id"], goal["objective"], goal["status"]) for goal in goals] == [
        ("goal-a", "synthetic objective for thread a", "active")
    ]
    assert "thread b" not in json.dumps(walked)

    owner = Polylogue(archive_root=root)
    assert await owner.get_session_materials(_SESSION_A) == walked
    assert await owner.get_session_materials("codex-session:missing") is None

    server = cast(MCPServerUnderTest, build_server())
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=owner),
    ):
        result = json.loads(
            await invoke_surface_async(
                server._tool_manager._tools["get"].fn, ref=f"session:{_SESSION_A}", projection="materials"
            )
        )
    assert result["materials"] == walked
    assert result["total"] == len(walked)


def _archive_with_goals(tmp_path: Path, goal_count: int) -> Path:
    """One ingested thread-A session plus ``goal_count`` retained goals naming it."""
    from polylogue.core.enums import Provider, Role
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession

    root = tmp_path / "archive"
    root.mkdir()
    from tests.infra.live_ingest import write_session_sync

    write_session_sync(
        root / "index.db",
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=_THREAD_A,
            messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="start")],
        ),
        archive_root=root,
    )
    goals_path = tmp_path / "goals_1.sqlite"
    _write_goals_db(
        goals_path,
        [(_THREAD_A, f"goal-{index}", f"synthetic objective {index}") for index in range(goal_count)],
    )
    _materialize(root, goals_path)
    return root


def _admit_session_material(root: Path, *, source_uri: str, payload: bytes, observed_at_ms: int) -> str:
    """Admit the synthetic bytes through one original prepared Source parent."""
    from polylogue.storage.materials import (
        PreparedMaterial,
        _admit_material,
        _link_material,
        prepare_material,
        publish_prepared_materials,
    )

    async def run() -> str:
        async with prepared_live_convergence_owner(root) as owner:
            retained: list[PreparedIndexMutation] = []

            def prepare() -> str:
                publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
                prepared: PreparedMaterial | None = None
                seal: PreparedIndexMutation | None = None
                lease: ActiveWriterLease | None = None
                bound = False

                def close_payload() -> None:
                    if prepared is not None:
                        prepared.discard()
                    publisher.discard_pending()

                with retain_native_sql_lifetimes(publisher):
                    try:
                        seal = PreparedIndexMutation.source_only(archive_root=root)
                        retained.append(seal)
                        lease = ActiveWriterLease(root)
                        lease.acquire()
                        seal.retain_publication_lifetime(lease, close_payload)
                        bound = True
                        prepared = prepare_material(
                            blob_store=publisher,
                            source_uri=source_uri,
                            referrer_ref=_SESSION_A,
                            payload=payload,
                            media_type="application/json",
                        )
                        publish_prepared_materials((prepared,), reference_seal=seal)
                        with seal.original_read_snapshot(), seal.source_producer():
                            producer = _PreparedSourceProducer(seal)
                            material = _admit_material(producer, prepared=prepared, observed_at_ms=observed_at_ms)
                            _link_material(
                                producer,
                                material.material_id,
                                _SESSION_A,
                                relation="refers_to",
                                authority="provider",
                                observed_at_ms=observed_at_ms,
                            )
                        permit = seal.prepare_source_mutation()

                        def publish() -> None:
                            with permit.hold_authority(), permit.mutation_connection() as connection:
                                with closing(connection.execute("BEGIN IMMEDIATE")):
                                    pass
                                permit.apply_source_statements(connection)
                                permit.allow_commit(connection)
                                connection.commit()
                                assert seal is not None
                                seal.accept_known_tier_commit(permit.committed())

                        admit_stage_write("fixture.session-material.publish", publish)
                        return material.material_id
                    finally:
                        primary = sys.exception()
                        failures: list[BaseException] = []
                        actions = (
                            (seal.close,)
                            if bound and seal is not None
                            else (
                                *((seal.close,) if seal is not None else ()),
                                close_payload,
                                *((lease.close,) if lease is not None else ()),
                            )
                        )
                        for action in actions:
                            try:
                                action()
                            except BaseException as failure:
                                failures.append(failure)
                        if failures:
                            raise BaseExceptionGroup(
                                "session material preparation and physical cleanup failed",
                                ([primary] if primary is not None else []) + failures,
                            )
                        if seal is not None:
                            retained.remove(seal)

            return await owner.run_prepared_sync(
                "fixture.session-material.prepare",
                prepare,
                settlement_owners=lambda: tuple(retained),
                estimated_bytes=len(payload),
            )

    return asyncio.run(run())


@pytest.mark.asyncio
async def test_mcp_materials_view_is_paged_by_the_evidence_window(tmp_path: Path) -> None:
    """MCP ``read(view="materials", limit=1)`` returns one row and a continuation.

    Anti-vacuity: route the ``materials`` projection back through the whole
    ``get_session_materials`` list and the first call returns every retained
    material with no continuation, whatever ``limit`` says.
    """
    from functools import partial
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import patch

    from polylogue import Polylogue
    from polylogue.mcp.server import build_server
    from tests.infra.archive_templates import run_archive_fixture_prepare
    from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async

    root = await run_archive_fixture_prepare(partial(_archive_with_goals, tmp_path, 3))
    owner = Polylogue(archive_root=root)
    whole = await owner.get_session_materials(_SESSION_A)
    assert whole is not None and len(whole) >= 3

    server = cast(MCPServerUnderTest, build_server())
    read = server._tool_manager._tools["read"].fn
    walked: list[dict[str, Any]] = []
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=owner),
    ):
        page = json.loads(await invoke_surface_async(read, ref=f"session:{_SESSION_A}", view="materials", limit=1))
        while True:
            assert page.get("is_error") is not True, page
            assert page["total"] == len(whole)
            assert len(page["materials"]) == 1
            walked.extend(page["materials"])
            if page["continuation"] is None:
                assert page["complete"] is True
                break
            assert page["complete"] is False
            page = json.loads(
                await invoke_surface_async(
                    read,
                    ref=f"session:{_SESSION_A}",
                    view="materials",
                    limit=1,
                    continuation=page["continuation"],
                )
            )
    assert walked == whole


def test_a_malformed_json_material_is_returned_with_its_state(tmp_path: Path) -> None:
    """Retained JSON bytes that do not parse are a row, not a failed page.

    Anti-vacuity: decode ``application/json`` with an unguarded
    ``json.loads`` and the page read raises ``JSONDecodeError`` instead of
    returning the ``malformed`` observation.
    """
    from polylogue.operations.session_evidence import read_session_materials_page

    root = _archive_with_goals(tmp_path, 1)
    material_id = _admit_session_material(
        root, source_uri="codex://state/goals/broken", payload=b'{"objective": ', observed_at_ms=9_000
    )

    with ArchiveStore.open_existing(root) as archive:
        rows, total = read_session_materials_page(archive, _SESSION_A, limit=50, offset=0)

    assert total == len(rows)
    broken = next(row for row in rows if row["material_id"] == material_id)
    assert broken["acquisition_state"] == "malformed"
    assert broken["content_form"] == "text"
    assert broken["content"] == '{"objective": '
    assert all(row["content_form"] == "json" for row in rows if row["material_id"] != material_id)


def test_session_materials_page_never_scans_the_archive_wide_table(tmp_path: Path) -> None:
    """Every statement a materials page issues is an indexed search.

    Anti-vacuity: select the page by ``material_observations.referrer_ref``
    again -- a column no index covers -- and the plan for the count and page
    statements reads ``SCAN``.
    """
    from polylogue.operations.session_evidence import read_session_materials_page, session_materials_source_epoch

    root = _archive_with_goals(tmp_path, 2)
    statements: list[str] = []
    with ArchiveStore.open_existing(root) as archive:
        conn = archive.source_connection
        conn.set_trace_callback(statements.append)
        try:
            read_session_materials_page(archive, _SESSION_A, limit=1, offset=1)
            session_materials_source_epoch(archive, _SESSION_A)
        finally:
            conn.set_trace_callback(None)
        selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT") and "material_" in sql]
        assert selects
        for sql in selects:
            plan = [str(row[3]) for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
            assert not any(step.startswith("SCAN") for step in plan), (sql, plan)


def test_a_materials_continuation_is_stale_after_an_earlier_admission(tmp_path: Path) -> None:
    """A material admitted ahead of the offset makes the continuation stale.

    The relation lives in ``source.db``, outside the index/user frame the
    token also carries. Anti-vacuity: drop the ``source_epoch`` binding from
    ``read_session_evidence_window`` and the resume is accepted, returning
    the row page one already delivered.
    """
    from polylogue.archive.query.transaction import QueryContinuationStaleError
    from polylogue.operations.session_evidence import read_session_evidence_window

    root = _archive_with_goals(tmp_path, 2)
    ref = f"session:{_SESSION_A}"
    with ArchiveStore.open_existing(root) as archive:
        first = read_session_evidence_window(archive, "materials", ref=ref, limit=1, offset=0, continuation=None)
        assert first is not None and first["continuation"] is not None
        unchanged = read_session_evidence_window(
            archive, "materials", ref=ref, limit=1, offset=0, continuation=str(first["continuation"])
        )
        assert unchanged is not None and unchanged["offset"] == 1

    _admit_session_material(root, source_uri="codex://state/goals/earlier", payload=b"{}", observed_at_ms=1)

    with ArchiveStore.open_existing(root) as archive, pytest.raises(QueryContinuationStaleError):
        read_session_evidence_window(
            archive, "materials", ref=ref, limit=1, offset=0, continuation=str(first["continuation"])
        )
