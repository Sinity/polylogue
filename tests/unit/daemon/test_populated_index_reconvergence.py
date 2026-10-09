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


def test_actual_startup_replays_populated_source_after_original_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    source = tmp_path / "external" / "session.jsonl"
    old, raw_id, session_ids = make_populated_stale_index(root, source)
    assert raw_id and len(session_ids) == 1
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


@pytest.mark.parametrize("phase", ["activation", "acknowledgement"])
def test_pointer_swapped_recovery_finishes_owned_promotion_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from polylogue.operations import index_reconvergence_startup as startup
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

    def stop_acknowledgement(*_args: object) -> None:
        raise InterruptedPromotionError

    with monkeypatch.context() as control:
        if phase == "activation":
            control.setattr(IndexGenerationStore, "_write", stop_activation)
        else:
            control.setattr(startup, "_acknowledge_promoted", stop_acknowledgement)
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
            assert (
                conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()[0]
                is not None
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
    assert tuple((root / ".index-generations").glob("gen-*/generation.json")) == metadata
    assert IndexGenerationStore.for_archive_root(root).load(active.parent.name).state == "active"


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
