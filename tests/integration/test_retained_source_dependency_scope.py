"""Original generation and Index evidence survive preparatory Source phases."""

from __future__ import annotations

import faulthandler
import json
import sys
from builtins import BaseExceptionGroup
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.core.enums import Provider, TitleSource
from polylogue.core.stage_admission import admit_stage_write
from polylogue.daemon.api_auth import resolve_api_auth_token
from polylogue.daemon.services import ServiceCapability, ServiceProfile
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.sources.acquisition_boundary import bound_profile_identity, bound_source_observation, open_bound_path
from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived.raw import RawObservationReplacement
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.daemon_service_harness import ServiceHarness
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.retained_parser_payloads import _chatgpt_session, _codex_thread_state_snapshot_bytes


@pytest.mark.timeout(0)
async def test_generation_owned_cohort_spans_input_pages(tmp_path: Path) -> None:
    """Actual first and 257th input members both participate in one accepted cohort."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    empty_zip = b"PK\x05\x06" + b"\x00" * 18
    for ordinal in range(1, 256):
        (inputs / f"input-{ordinal:03d}.zip").write_bytes(empty_zip)
    for ordinal, texts in ((0, ("one",)), (256, ("one", "two"))):
        (inputs / f"input-{ordinal:03d}.json").write_text(json.dumps(_chatgpt_session("cross-page", *texts)))
    root = tmp_path / "archive"
    await run_archive_fixture_write(root, lambda: bootstrap_archive_root(root))
    archive = Polylogue(archive_root=root, db_path=root / "index.db")
    harness = ServiceHarness(profile=ServiceProfile.SURFACES, capabilities={ServiceCapability.API})
    try:
        server = harness.api_server(root)
        uds = harness.uds_server(root, api_server=server, auth_token=resolve_api_auth_token(None))
        _api_task = harness.start_server("api_server", server)
        _uds_task = harness.start_server("uds_server", uds)
        with (tmp_path / "generation-worker-stacks.txt").open("w") as diagnostic:
            faulthandler.dump_traceback_later(45, file=diagnostic)
            try:
                result = await archive.parse_file(inputs, source_name="machine-ingest")
            finally:
                faulthandler.cancel_dump_traceback_later()
        assert result.parse_failures == 0
    finally:
        try:
            await harness.close()
        finally:
            await archive.close()
    with ArchiveStore.open_existing(root, read_only=True) as retained:
        source = retained.source_connection
        assert source is not None
        members = source.execute(
            "SELECT i.logical_coordinate,m.raw_id,m.source_generation_id FROM source_items i "
            "JOIN source_item_raw_members m USING(source_generation_id,source_item_id) "
            "ORDER BY i.logical_coordinate"
        ).fetchall()
        assert [row[0] for row in members] == ["input-000.json", "input-256.json"]
        assert len({row[2] for row in members}) == 1
        generation = members[0][2]
        assert (
            source.execute(
                "SELECT item_count FROM source_generations WHERE source_generation_id=?", (generation,)
            ).fetchone()[0]
            == 257
        )
        ids = {row[1] for row in members}
        assert {
            row[0]
            for row in source.execute(
                "SELECT raw_id FROM raw_session_memberships WHERE logical_source_key='chatgpt-export:cross-page'"
            )
        } == ids
        index = retained.index_connection
        assert index is not None
        assert (
            index.execute("SELECT message_count FROM sessions WHERE session_id='chatgpt-export:cross-page'").fetchone()[
                0
            ]
            == 2
        )
        assert {row[0] for row in index.execute("SELECT text FROM blocks WHERE block_type='text'")} == {"one", "two"}

    retained_outputs: list[RawObservationReplacement] = []
    async with prepared_live_convergence_owner(root) as owner:

        def select_first_original() -> None:
            adapter, index_path, _index_destination = owner._archive.destination_adapter()
            frame = raw_observation_frame(root, raw_ids=(str(members[0][1]),), index_db_path=index_path)
            replacement = adapter.compute(frame, str(members[0][1]), replay_current=True)
            retained_outputs.append(replacement)
            try:
                assert set(replacement.raw_ids) == ids
            finally:
                primary = sys.exception()
                try:
                    replacement.close()
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup(
                            "generation selection and close failed", [primary, cleanup]
                        ) from primary
                    raise
                retained_outputs.remove(replacement)

        await owner.run_prepared_sync(
            "fixture.generation.complete-selection",
            select_first_original,
            settlement_owners=lambda: tuple(retained_outputs),
            estimated_bytes=sum((inputs / name).stat().st_size for name in ("input-000.json", "input-256.json")),
        )


async def test_source_census_codex_artifact_enrolls_original_projected_titles(tmp_path: Path) -> None:
    """The first retained preparation of a rollout already uses its original Index title evidence."""
    root = tmp_path / "archive"
    title = "curated title"
    _codex_thread_state_snapshot_bytes(tmp_path, title)
    state_path = tmp_path / title / "state_5.sqlite"
    rollout_path = state_path.parent / "sessions" / "rollout-codex-state-thread.jsonl"
    payload = b' {"type":"session_meta","payload":{"id":"codex-state-thread","timestamp":"2026-01-01T00:00:00Z"}}\n{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"heuristic prompt"}]}}\n'

    def acquire_state() -> str:
        bootstrap_archive_root(root)
        blobs = BlobStore(root / "blob")
        snapshot = snapshot_sqlite_to_blob(state_path, blobs)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=blobs.blob_path(snapshot.blob_hash).read_bytes(),
                source_path=str(state_path),
                canonical_source_path=str(state_path),
                captured_profile_key=snapshot.captured_profile_key,
                acquired_at_ms=1,
            )

    state_raw = await run_archive_fixture_write(root, acquire_state)
    async with prepared_live_convergence_owner(root) as owner:
        (await owner.replay_retained_raw_ids((state_raw,))).require_complete()

        def acquire_rollout() -> str:
            rollout_path.parent.mkdir()
            rollout_path.write_bytes(payload)
            with open_bound_path(rollout_path, Provider.CODEX) as original:
                profile = bound_profile_identity(original)
                canonical, observation = bound_source_observation(original)
                assert profile is not None and canonical is not None and observation is not None
                with ArchiveStore.open_existing(root, read_only=False) as archive:
                    return archive.write_raw_payload(
                        provider=Provider.CODEX,
                        payload=original.read(),
                        source_path=str(rollout_path),
                        canonical_source_path=canonical,
                        captured_profile_key=profile.key,
                        acquired_at_ms=2,
                    )

        raw_id = await run_archive_fixture_write(root, acquire_rollout)
        retained: list[RawObservationReplacement] = []

        def census() -> None:
            adapter, index_path, _index_destination = owner._archive.destination_adapter()
            frame = raw_observation_frame(root, raw_ids=(raw_id,), index_db_path=index_path)
            replacement = adapter.compute(frame, raw_id)
            retained.append(replacement)
            try:
                # Acquisition already records the rollout's parser census, so
                # its first preparation is the replay itself; the original
                # artifact it prepares must already carry the projected title.
                assert replacement.prepared_source_census is None
                assert replacement.prepared_inputs is not None
                artifact = replacement.prepared_inputs[raw_id].prepared_artifact
                assert artifact is not None
                assert artifact.enrichment_digest is None and artifact.enrichment_index_path is None
                with closing(artifact.iter_sessions()) as sessions:
                    session = next(sessions)
                    assert (session.title, session.title_source) == (title, TitleSource.ORIGIN)
                    assert session.enrichment_evidence_key is not None
                admit_stage_write("fixture.source-title.publish", lambda: adapter.publish(frame, replacement))
            finally:
                primary = sys.exception()
                try:
                    replacement.close()
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup(
                            "Source title preparation and close failed", [primary, cleanup]
                        ) from primary
                    raise
                retained.remove(replacement)

        await owner.run_prepared_sync(
            "fixture.source-title", census, settlement_owners=lambda: tuple(retained), estimated_bytes=len(payload)
        )
        with ArchiveStore.open_existing(root, read_only=True) as observed:
            source = observed.source_connection
            index = observed.index_connection
            assert index is not None
            census_receipt = source.execute(
                "SELECT status,logical_keys_json FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)
            ).fetchone()
            assert census_receipt is not None
            assert census_receipt[0] == "complete"
            assert json.loads(census_receipt[1]) == ["codex-session:codex-state-thread"]
            row = index.execute(
                "SELECT title,title_source FROM sessions WHERE native_id='codex-state-thread'"
            ).fetchone()
            assert row is not None and tuple(row) == (title, TitleSource.ORIGIN.value)
        (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        row = index.execute("SELECT title,title_source FROM sessions WHERE native_id='codex-state-thread'").fetchone()
        assert row is not None and tuple(row) == (title, TitleSource.ORIGIN.value)
