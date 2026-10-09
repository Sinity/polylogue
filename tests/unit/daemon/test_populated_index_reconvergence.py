"""Ordinary startup derives a successor from retained evidence under schema drift."""

from __future__ import annotations

import asyncio
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.daemon import cli as daemon_cli
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported, open_readonly_connection
from tests.infra.populated_managed_index import logical_rows, make_populated_stale_index


@pytest.mark.parametrize(
    ("multi_session", "include_history", "include_codex_materials"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_actual_startup_replays_populated_source_after_original_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multi_session: bool,
    include_history: bool,
    include_codex_materials: bool,
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


def test_pointer_swapped_recovery_finishes_owned_promotion_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
