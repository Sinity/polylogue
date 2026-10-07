"""Work events retain the target's actual acquisition provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.coordination.work_events import WorkEventProvenanceRefusedError
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner, write_index_session


@pytest.mark.parametrize("provider", [Provider.GEMINI, Provider.DRIVE])
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_work_event_requires_actual_unambiguous_target_acquisition(
    tmp_path: Path, provider: Provider, ambiguous: bool
) -> None:
    payload = json.dumps(
        {"id": "neutral-target", "chunkedPrompt": {"chunks": [{"role": "user", "text": "neutral input"}]}},
        separators=(",", ":"),
    ).encode()
    source_path = tmp_path / "capture.json"
    source_path.write_bytes(payload)

    def acquire(mode: Provider) -> str:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=provider,
                capture_mode=mode,
                payload=payload,
                source_path=str(source_path),
                canonical_source_path=str(source_path),
                acquired_at_ms=1_767_000_000_000,
                file_mtime_ms=1_767_000_000_000,
            )
            archive.commit()
            return raw_id

    raw_id = await run_archive_fixture_write(tmp_path, lambda: acquire(provider))
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
    changed = {session_id for receipt in receipts for session_id in receipt.changed_session_ids}
    assert len(changed) == 1
    session_id = next(iter(changed))
    if ambiguous:
        other = Provider.DRIVE if provider is Provider.GEMINI else Provider.GEMINI
        assert await run_archive_fixture_write(tmp_path, lambda: acquire(other)) == raw_id

    def admit() -> dict[str, object]:
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.admit_work_event(
                session_id=session_id,
                event_type="decision",
                event_id="neutral-event",
                summary="neutral decision",
                payload={},
            )

    if ambiguous:
        with pytest.raises(WorkEventProvenanceRefusedError) as refused:
            await run_archive_fixture_write(tmp_path, admit)
        assert refused.value.session_id == session_id
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert (
                archive._ensure_source_conn()
                .execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id LIKE 'agent-work-event:%'")
                .fetchone()[0]
                == 0
            )
        return
    admitted = await run_archive_fixture_write(tmp_path, admit)
    event_raw_id = admitted["raw_id"]
    assert isinstance(event_raw_id, str)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        event_provider, event_payload, _path, _kind = archive.raw_revision_material(event_raw_id)
    assert event_provider is provider
    assert json.loads(event_payload)["provider"] == provider.value
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((event_raw_id,))).require_complete()
    assert event_raw_id in {raw for receipt in receipts for raw in receipt.writer_changed_raw_ids}


async def test_work_event_refuses_ambiguous_origin_without_acquisition(tmp_path: Path) -> None:
    def seed() -> str:
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return write_index_session(
                archive, ParsedSession(source_name=Provider.DRIVE, provider_session_id="index-only", messages=[])
            )

    session_id = await run_archive_fixture_write(tmp_path, seed)

    def admit() -> dict[str, object]:
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            return archive.admit_work_event(
                session_id=session_id, event_type="decision", event_id="event", summary="neutral", payload={}
            )

    with pytest.raises(WorkEventProvenanceRefusedError) as refused:
        await run_archive_fixture_write(tmp_path, admit)
    assert refused.value.session_id == session_id
