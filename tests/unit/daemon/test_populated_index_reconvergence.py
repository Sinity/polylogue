"""Ordinary startup derives a successor from retained evidence under schema drift."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.daemon import cli as daemon_cli
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported, open_readonly_connection
from tests.infra.populated_managed_index import logical_rows, make_populated_stale_index


@pytest.mark.parametrize(
    "control", ["owned-empty", "foreign-owner", "unknown-populated", "changed-control-shape", "unreadable-view"]
)
def test_startup_reconstructs_owned_empty_predecessor_with_changed_ddl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from tests.infra.retained_replay import publish_retained_payload

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    store = IndexGenerationStore.for_archive_root(root)
    parent = store.create(
        owner_id="foreign-owner" if control == "foreign-owner" else "daemon:empty-index-startup",
        source_snapshot="neutral-empty-bootstrap",
    )
    store.promote(parent)
    old = Path(parent.index_path)
    source = tmp_path / "external.jsonl"
    source.write_bytes(
        b'{"type":"session_meta","payload":{"id":"neutral-retained"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"neutral-message",'
        b'"role":"user","content":[{"type":"input_text","text":"neutral retained prose"}]}}\n'
    )
    raw_id, session_ids = asyncio.run(
        publish_retained_payload(
            root, provider=Provider.CODEX, payload=source.read_bytes(), source_path=str(source), acquired_at_ms=0
        )
    )
    assert raw_id and len(session_ids) == 1
    source.unlink()
    # An earlier empty bootstrap may have a different rebuildable schema.
    # Retained Source is already positive, so the empty-archive route cannot replace it.
    with closing(sqlite3.connect(old)) as conn, conn:
        triggers = conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'").fetchall()
        for name, _sql in triggers:
            conn.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        controls = {
            "schema_identity",
            "query_unit_frame_state",
            "raw_existence_journal_control",
            "session_profile_demand_state",
        }
        for schema, name, kind, *_ in conn.execute("PRAGMA table_list").fetchall():
            if (
                schema == "main"
                and kind in {"table", "virtual"}
                and not name.startswith("sqlite_")
                and name not in controls
            ):
                conn.execute('DELETE FROM "' + name.replace('"', '""') + '"')
        for _name, sql in triggers:
            conn.execute(sql)
        conn.execute("DROP INDEX idx_messages_source_native")
        conn.execute("ALTER TABLE messages RENAME COLUMN source_native_id_json TO predecessor_native")
        conn.execute("ALTER TABLE messages ADD COLUMN predecessor_only TEXT")
        conn.execute("CREATE TABLE retired_material(value TEXT)")
        if control == "changed-control-shape":
            conn.execute("ALTER TABLE query_unit_frame_state ADD COLUMN unknown_state TEXT")
        if control == "unknown-populated":
            conn.execute("INSERT INTO retired_material VALUES ('neutral')")
        conn.execute("UPDATE schema_identity SET identity='neutral-predecessor-runtime' WHERE tier='index'")
        nonempty = {}
        for schema, name, kind, *_ in conn.execute("PRAGMA table_list").fetchall():
            if schema == "main" and not name.startswith("sqlite_") and name not in controls and kind != "shadow":
                quoted = '"' + name.replace('"', '""') + '"'
                count = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
                if count:
                    nonempty[name] = count
        assert nonempty == ({"retired_material": 1} if control == "unknown-populated" else {}), nonempty
        if control == "unreadable-view":
            conn.execute("CREATE VIEW unreadable_material AS SELECT missing_column FROM messages")
    before = {tier: logical_rows(root / f"{tier}.db") for tier in ("user", "audit", "embeddings")}
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        active = ArchiveLocation.resolve(root).active_index_path.resolve(strict=True)
        if control != "owned-empty":
            assert active == old
        else:
            assert active != old and old.exists() and not source.exists()
            with closing(open_readonly_connection(active)) as conn:
                assert_tier_schema_supported(conn, active)
                assert tuple(row[0] for row in conn.execute("SELECT session_id FROM sessions")) == session_ids
                assert conn.execute("SELECT text FROM blocks").fetchone()[0] == "neutral retained prose"
        assert {tier: logical_rows(root / f"{tier}.db") for tier in before} == before
        raise PreflightReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    for _ in range(2 if control == "owned-empty" else 1):
        source_before = logical_rows(root / "source.db")
        with pytest.raises(PreflightReachedError):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                )
            )
        assert logical_rows(root / "source.db") == source_before


def test_startup_defers_bulk_read_models_then_publishes_canonical_equivalents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual cohort replay skips per-session work; readiness closes the gap."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.empty_managed_index import make_empty_managed_index, mutate_fixture_database
    from tests.infra.retained_replay import publish_retained_payload

    root = tmp_path / "archive"
    old = make_empty_managed_index(root, stale=False)
    parent = [
        {"type": "session_meta", "payload": {"id": "neutral-parent", "timestamp": "2026-06-02T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": "fc",
                "call_id": "call",
                "name": "exec_command",
                "arguments": '{"cmd":"printf neutral"}',
            },
        },
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "call", "output": "neutral"}},
    ]
    child = [
        {
            "type": "session_meta",
            "payload": {
                "id": "neutral-child",
                "timestamp": "2026-06-02T00:00:01Z",
                "forked_from_id": "neutral-parent",
                "source": {"subagent": {"thread_spawn": True}},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "child-message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "neutral child"}],
            },
        },
    ]
    for name, records in (("parent", parent), ("child", child)):
        asyncio.run(
            publish_retained_payload(
                root,
                provider=Provider.CODEX,
                payload=("\n".join(json.dumps(record) for record in records) + "\n").encode(),
                source_path=str(tmp_path / f"{name}.jsonl"),
                acquired_at_ms=0,
            )
        )

    def read_models(conn: sqlite3.Connection) -> dict[str, tuple[tuple[object, ...], ...]]:
        from polylogue.storage.derived.session.summary import SESSION_SUMMARY_MEASURES

        models = {
            table: tuple(sorted(tuple(row) for row in conn.execute(f"SELECT * FROM {table}")))
            for table in ("action_pairs", "delegation_facts")
        }
        columns = ",".join(measure.column for measure in SESSION_SUMMARY_MEASURES)
        models["session_summary"] = tuple(
            tuple(row) for row in conn.execute(f"SELECT session_id,{columns} FROM sessions ORDER BY session_id")
        )
        return models

    with closing(open_readonly_connection(old)) as conn:
        expected = read_models(conn)
        assert expected["action_pairs"] and expected["delegation_facts"] and expected["session_summary"]
    mutate_fixture_database(old, "UPDATE schema_identity SET identity='prior-runtime' WHERE tier='index'")
    original_readiness = ArchiveStore.run_generation_readiness_pass
    readiness_calls = []

    def readiness(self: ArchiveStore) -> None:
        readiness_calls.append(self.index_db_path)
        assert self.owns_inactive_generation
        assert self._conn.execute("SELECT COUNT(*) FROM action_pairs").fetchone()[0] == 0
        assert self._conn.execute("SELECT COUNT(*) FROM delegation_facts").fetchone()[0] == 0
        assert self._conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0] == 0
        assert self._conn.execute("SELECT 1 FROM sqlite_master WHERE name='idx_messages_role'").fetchone() is None
        original_readiness(self)
        assert read_models(self._conn) == expected
        assert self._conn.execute("SELECT COUNT(*) FROM delegation_refresh_scope").fetchone()[0] == 0
        assert self._conn.execute("SELECT COUNT(*) FROM derived_refresh_guard").fetchone()[0] == 0

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        active = ArchiveLocation.resolve(root).active_index_path.resolve(strict=True)
        assert active != old and old.exists() and len(readiness_calls) == 1
        with closing(open_readonly_connection(active)) as conn:
            assert_tier_schema_supported(conn, active)
            assert read_models(conn) == expected
            assert (
                conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'neutral'").fetchone()[0] > 0
            )
        raise PreflightReachedError

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    monkeypatch.setattr(ArchiveStore, "run_generation_readiness_pass", readiness)
    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    with pytest.raises(PreflightReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )


@pytest.mark.parametrize(
    ("multi_session", "include_history", "include_codex_materials", "missing_history_membership"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, True, True, False),
        pytest.param(False, True, False, True, id="missing-history-membership"),
    ],
)
def test_actual_startup_replays_populated_source_after_original_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multi_session: bool,
    include_history: bool,
    include_codex_materials: bool,
    missing_history_membership: bool,
) -> None:
    root = tmp_path / "archive"
    source = tmp_path / "external" / ("bundle.json" if multi_session else "session.jsonl")
    old, raw_id, session_ids = make_populated_stale_index(
        root,
        source,
        multi_session=multi_session,
        include_history=include_history,
        include_codex_materials=include_codex_materials,
    )
    assert raw_id and len(session_ids) == (2 if multi_session else 1)
    if include_history:
        from polylogue.storage.blob_store import BlobStore
        from polylogue.storage.sqlite.archive_tiers.revision_governance import prepared_parser_census_is_current
        from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead
        from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

        with closing(open_readonly_connection(root / "source.db")) as conn:
            history_id = str(
                conn.execute(
                    "SELECT raw_id FROM raw_sessions WHERE source_path LIKE '%/.claude/history.jsonl'"
                ).fetchone()[0]
            )
            if missing_history_membership:
                from tests.infra.empty_managed_index import mutate_fixture_database

                mutate_fixture_database(
                    root / "source.db", "DELETE FROM raw_membership_census WHERE raw_id=?", (history_id,)
                )
            history_binding = tuple(
                conn.execute(
                    "SELECT logical_source_key,revision_kind,revision_authority,source_revision,"
                    "predecessor_raw_id,baseline_raw_id FROM raw_sessions WHERE raw_id=?",
                    (history_id,),
                ).fetchone()
            )
        with PreparedIndexMutation(old, archive_root=root) as seal:
            with seal.original_read_snapshot(), seal.source_producer():
                read = PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob"))
                assert prepared_parser_census_is_current(seal, history_id)
                assert not read.raw_parser_confirmed_non_session(history_id)
        from polylogue.operations.raw_observation_derivation import raw_observation_inspection_frame
        from polylogue.storage.derived.raw import RawObservationInspection

        inspection = RawObservationInspection(root, index_db_path=old)
        assert inspection.inspect(raw_observation_inspection_frame(root, index_db_path=old), (history_id,)) == {
            history_id: "stale"
        }
    if include_codex_materials:
        with closing(open_readonly_connection(root / "source.db")) as conn:
            materials_before = tuple(
                conn.execute("SELECT material_id,blob_hash FROM material_observations ORDER BY material_id")
            )
            assert len(materials_before) == 2
    continued_material_counts: list[tuple[int, int]] = []
    if include_codex_materials:
        from polylogue.storage.derived.raw import (
            RawFrame,
            RawObservationDerivation,
            RawObservationReplacement,
            _PreparationCarry,
        )
        from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead

        original_continue = RawObservationDerivation._continue_after_phase

        def observe_continue(
            self: RawObservationDerivation,
            frame: RawFrame,
            replacement: RawObservationReplacement,
            carry: _PreparationCarry,
        ) -> _PreparationCarry | BaseException | None:
            # The production handoff advertises these captured carriers for
            # reuse. Every carried claim must remain consumable on its seal.
            before_count = sum(a.codex_state_kind in {"goals", "memories"} for a in carry.artifacts.values())
            result = original_continue(self, frame, replacement, carry)
            if result is carry and replacement.needs_source_census:
                state_carriers = [a for a in carry.artifacts.values() if a.codex_state_kind in {"goals", "memories"}]
                continued_material_counts.append((before_count, len(state_carriers)))
                seal = replacement.reference_seal
                assert seal is not None
                with seal.original_read_snapshot(), seal.source_producer():
                    for artifact in state_carriers:
                        publisher = artifact.publication_publisher
                        assert publisher is not None
                        read = PreparedSessionSourceRead(seal, blob_store=publisher)
                        for *_coordinate, material in artifact.iter_codex_state_material():
                            if material is not None:
                                assert material.publication_claim is not None
                                publisher.validate_published_claim(
                                    read, material.publication_claim, source_path=material.source_uri
                                )
            return result

        monkeypatch.setattr(RawObservationDerivation, "_continue_after_phase", observe_continue)
    before = {tier: logical_rows(root / f"{tier}.db") for tier in ("user", "audit", "embeddings")}
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        active = ArchiveLocation.resolve(root).active_index_path.resolve(strict=True)
        with closing(open_readonly_connection(active)) as conn:
            assert_tier_schema_supported(conn, active)
            assert (
                tuple(row[0] for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"))
                == session_ids
            )
            assert conn.execute("SELECT text FROM blocks").fetchone()[0] == "neutral retained prose"
        assert active != old
        assert old.exists() and not source.exists()
        assert {tier: logical_rows(root / f"{tier}.db") for tier in before} == before
        if include_history:
            from polylogue.archive.revision_authority import raw_authority_parser_fingerprint

            with closing(open_readonly_connection(root / "source.db")) as conn:
                assert conn.execute(
                    "SELECT c.status,c.parser_fingerprint,r.validation_mode FROM raw_membership_census c "
                    "JOIN raw_sessions r USING(raw_id) WHERE r.source_path LIKE '%/.claude/history.jsonl'"
                ).fetchall() == [("non_session", raw_authority_parser_fingerprint(), None)]
                assert (
                    tuple(
                        conn.execute(
                            "SELECT logical_source_key,revision_kind,revision_authority,source_revision,"
                            "predecessor_raw_id,baseline_raw_id FROM raw_sessions WHERE raw_id=?",
                            (history_id,),
                        ).fetchone()
                    )
                    == history_binding
                )
        if include_codex_materials:
            with closing(open_readonly_connection(root / "source.db")) as conn:
                assert (
                    tuple(conn.execute("SELECT material_id,blob_hash FROM material_observations ORDER BY material_id"))
                    == materials_before
                )
        raise PreflightReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    with pytest.raises(PreflightReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )

    if include_codex_materials:
        assert (2, 0) in continued_material_counts
        source_before_restart = logical_rows(root / "source.db")
        with pytest.raises(PreflightReachedError):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                )
            )
        assert logical_rows(root / "source.db") == source_before_restart


@pytest.mark.parametrize("phase", ["replay", "readiness"])
def test_cancelled_startup_keeps_predecessor_and_restarts_from_retained_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from polylogue.operations.raw_observation_owner import RawObservationArchiveWork
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    source = tmp_path / "external" / "session.jsonl"
    old, _raw_id, session_ids = make_populated_stale_index(root, source)
    before = {tier: logical_rows(root / f"{tier}.db") for tier in ("source", "user", "audit", "embeddings")}
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    def cancel(*_args: object, **_kwargs: object) -> object:
        raise asyncio.CancelledError()

    with monkeypatch.context() as control:
        if phase == "replay":
            control.setattr(RawObservationArchiveWork, "retained_replay_operation", cancel)
        else:
            control.setattr(ArchiveStore, "run_generation_readiness_pass", cancel)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                )
            )
    assert ArchiveLocation.resolve(root).active_index_path.resolve() == old
    assert {tier: logical_rows(root / f"{tier}.db") for tier in before} == before
    abandoned = tuple((root / ".index-generations").glob("gen-*/generation.json"))
    assert len(abandoned) == 2

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        active = ArchiveLocation.resolve(root).active_index_path.resolve()
        with closing(open_readonly_connection(active)) as conn:
            assert tuple(row[0] for row in conn.execute("SELECT session_id FROM sessions")) == session_ids
        assert active != old and not source.exists()
        raise PreflightReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    with pytest.raises(PreflightReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )
    current_metadata = tuple((root / ".index-generations").glob("gen-*/generation.json"))
    assert len(current_metadata) == 2
    assert any(not item.exists() for item in abandoned)
    # A normal current-runtime restart must not reconstruct or rewrite Source acknowledgement.
    source_before_restart = logical_rows(root / "source.db")
    with pytest.raises(PreflightReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )
    assert tuple((root / ".index-generations").glob("gen-*/generation.json")) == current_metadata
    assert logical_rows(root / "source.db") == source_before_restart


@pytest.mark.parametrize(
    ("tier", "sql", "reason"),
    [
        ("index", "CREATE TABLE unknown_material(value TEXT)", "unprovable_index_shape"),
        ("embeddings", "PRAGMA user_version=99", "unsupported_embeddings_schema"),
    ],
)
def test_actual_startup_refuses_ineligible_populated_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str, sql: str, reason: str
) -> None:
    from polylogue.logging import capture
    from tests.infra.empty_managed_index import mutate_fixture_database

    root = tmp_path / "archive"
    old, _raw_id, _session_ids = make_populated_stale_index(root, tmp_path / "external" / "session.jsonl")
    mutate_fixture_database(old if tier == "index" else root / f"{tier}.db", sql)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    before = tuple((root / ".index-generations").glob("gen-*/generation.json"))

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        assert ArchiveLocation.resolve(root).active_index_path.resolve() == old
        raise PreflightReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    with capture() as records, pytest.raises(PreflightReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )
    assert any(
        row["event"] == "daemon.index_reconvergence.startup"
        and row.get("outcome") == "refused"
        and row.get("reason") == reason
        for row in records
    )
    assert tuple((root / ".index-generations").glob("gen-*/generation.json")) == before


@pytest.mark.parametrize("leftover_temporary", [False, True])
def test_pointer_swapped_recovery_finishes_owned_promotion_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leftover_temporary: bool
) -> None:
    from polylogue.storage.index_generation import IndexGeneration, IndexGenerationStore
    from tests.infra.empty_managed_index import mutate_fixture_database

    root = tmp_path / "archive"
    old, raw_id, session_ids = make_populated_stale_index(root, tmp_path / "external" / "session.jsonl")
    mutate_fixture_database(root / "source.db", "UPDATE raw_sessions SET parsed_at_ms=NULL WHERE raw_id=?", (raw_id,))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    class InterruptedPromotionError(Exception):
        pass

    original_write = IndexGenerationStore._write

    def stop_activation(self: IndexGenerationStore, generation: IndexGeneration) -> None:
        if generation.owner_id == "daemon:retained-index-startup" and generation.state == "active":
            raise InterruptedPromotionError
        original_write(self, generation)

    with monkeypatch.context() as control:
        control.setattr(IndexGenerationStore, "_write", stop_activation)
        with pytest.raises(InterruptedPromotionError):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                )
            )
    active = ArchiveLocation.resolve(root).active_index_path.resolve()
    assert active != old and old.exists()
    metadata = tuple((root / ".index-generations").glob("gen-*/generation.json"))
    temporary = active.parent / "generation.json.tmp"
    temporary_identity = None
    if leftover_temporary:
        temporary.write_bytes(b"neutral interrupted atomic write")
        temporary_identity = temporary.stat()

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        assert ArchiveLocation.resolve(root).active_index_path.resolve() == active
        with closing(open_readonly_connection(active)) as conn:
            assert tuple(row[0] for row in conn.execute("SELECT session_id FROM sessions")) == session_ids
        with closing(open_readonly_connection(root / "source.db")) as conn:
            assert conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()[0] is None
        raise PreflightReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    with pytest.raises(PreflightReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )
    assert tuple((root / ".index-generations").glob("gen-*/generation.json")) == metadata
    assert IndexGenerationStore.for_archive_root(root).load(active.parent.name).state == "active"

    if leftover_temporary:
        assert temporary.read_bytes() == b"neutral interrupted atomic write"
        assert temporary_identity is not None
        assert temporary.stat().st_ino == temporary_identity.st_ino


def test_current_startup_preserves_newer_source_retry_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.core.enums import Provider
    from polylogue.core.errors import RawCASFrontierError
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.empty_managed_index import mutate_fixture_database

    root = tmp_path / "archive"
    old, raw_id, _session_ids = make_populated_stale_index(root, tmp_path / "external" / "session.jsonl")
    mutate_fixture_database(root / "source.db", "UPDATE raw_sessions SET parsed_at_ms=NULL WHERE raw_id=?", (raw_id,))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    class PreflightReachedError(Exception):
        pass

    def stop_preflight() -> None:
        raise PreflightReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", stop_preflight)

    def restart() -> None:
        with pytest.raises(PreflightReachedError):
            asyncio.run(
                daemon_cli.run_daemon_services(
                    sources=(),
                    enable_watch=False,
                    enable_browser_capture=False,
                    browser_capture_host="127.0.0.1",
                    browser_capture_port=8765,
                )
            )

    restart()
    active = ArchiveLocation.resolve(root).active_index_path.resolve()
    assert active != old
    with closing(open_readonly_connection(active)) as index:
        assert index.execute("SELECT COUNT(*) FROM raw_revision_applications WHERE raw_id=?", (raw_id,)).fetchone()[0]

    def refuse_current() -> None:
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            archive.mark_raw_parse_failed(
                raw_id, provider=Provider.CODEX, error=RawCASFrontierError("new unresolved frontier")
            )

    asyncio.run(run_archive_fixture_write(root, refuse_current))
    before = logical_rows(root / "source.db")
    with closing(open_readonly_connection(root / "source.db")) as source:
        assert source.execute(
            "SELECT parsed_at_ms,parse_error FROM raw_sessions WHERE raw_id=?", (raw_id,)
        ).fetchone() == (
            None,
            "RawCASFrontierError: new unresolved frontier",
        )
        assert source.execute(
            "SELECT artifact_kind FROM raw_artifacts WHERE raw_id=? AND artifact_kind='deferred_cas_frontier'",
            (raw_id,),
        ).fetchall() == [
            ("deferred_cas_frontier",),
        ]
    restart()
    assert ArchiveLocation.resolve(root).active_index_path.resolve() == active
    assert logical_rows(root / "source.db") == before


@pytest.mark.parametrize("provider", ["codex", "unknown"])
def test_unparsed_retained_source_is_included_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.archive_templates import run_archive_fixture_write

    root = tmp_path / "archive"
    old, _raw_id, _session_ids = make_populated_stale_index(root, tmp_path / "external" / "session.jsonl")
    payload = b'{"type":"session_meta","payload":{"id":"not-yet-parsed","timestamp":"2026-06-02T00:00:00Z"}}\n{"type":"response_item","payload":{"type":"message","id":"second-message","role":"user","content":[{"type":"input_text","text":"second retained prose"}]}}\n'

    def acquire() -> str:
        with ArchiveStore.open_source_tier_acquisition(root) as source:
            return source.write_raw_payload(
                provider=Provider(provider),
                payload=payload,
                source_path=str(tmp_path / "gone.jsonl"),
                canonical_source_path=str(tmp_path / "gone.jsonl"),
                acquired_at_ms=0,
            )

    asyncio.run(run_archive_fixture_write(root, acquire))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        active = ArchiveLocation.resolve(root).active_index_path.resolve()
        assert active != old
        with closing(open_readonly_connection(active)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
            assert {row[0] for row in conn.execute("SELECT text FROM blocks")} == {
                "neutral retained prose",
                "second retained prose",
            }
        raise PreflightReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    with pytest.raises(PreflightReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )


@pytest.mark.parametrize(
    ("tier", "sql", "reason"),
    [
        ("source", "UPDATE raw_sessions SET native_id='changed-native'", "changed_source_evidence"),
        ("source", "UPDATE raw_sessions SET source_path='changed-coordinate'", "changed_source_evidence"),
        (
            "source",
            "UPDATE blob_refs SET source_path='changed-claim' WHERE ref_type='raw_payload'",
            "changed_source_evidence",
        ),
        ("user", "UPDATE assertions SET body_text='changed note'", "changed_user_custody"),
        ("user", "replace_leaf", "changed_user_binding"),
    ],
)
def test_changed_acquisition_custody_refuses_startup_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str, sql: str, reason: str
) -> None:
    from polylogue.logging import capture
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.empty_managed_index import mutate_fixture_database

    root = tmp_path / "archive"
    old, _raw_id, _session_ids = make_populated_stale_index(root, tmp_path / "external" / "session.jsonl")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    readiness = ArchiveStore.run_generation_readiness_pass

    def change_custody(candidate: ArchiveStore) -> None:
        readiness(candidate)
        if sql == "replace_leaf":
            replacement = root / "replacement-user.db"
            target = root / "user.db"
            replacement.write_bytes(target.read_bytes())
            replacement.replace(target)
        else:
            mutate_fixture_database(root / f"{tier}.db", sql)

    monkeypatch.setattr(ArchiveStore, "run_generation_readiness_pass", change_custody)

    class PreflightReachedError(Exception):
        pass

    def preflight() -> None:
        assert ArchiveLocation.resolve(root).active_index_path.resolve() == old
        raise PreflightReachedError

    monkeypatch.setattr(daemon_cli, "_check_schema_version_fast", preflight)
    with capture() as records, pytest.raises(PreflightReachedError):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )
    assert any(
        row["event"] == "daemon.index_reconvergence.startup"
        and row.get("outcome") == "refused"
        and row.get("reason") == reason
        for row in records
    )
