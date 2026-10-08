"""Original prepared deferral preserves an incomparable ungoverned Index head."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.dispatch import parse_payload
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.retained_parser_payloads import _chatgpt_session


@pytest.mark.asyncio
async def test_original_prepared_deferral_keeps_incomparable_index_and_no_accepted_head(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    acquired_payload = json.dumps([_chatgpt_session("same-key", "new different content")]).encode()
    existing_payload = [_chatgpt_session("same-key", "existing different content")]

    def acquire() -> tuple[str, str, object]:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            session = parse_payload(Provider.CHATGPT, existing_payload, "same-key")[0]
            index = archive.index_connection
            assert index is not None
            session_id = write_fixture_index_session(index, session, archive_root=root)
            original_hash = index.execute(
                "SELECT content_hash FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=acquired_payload,
                source_path="same-key.json",
                canonical_source_path="same-key.json",
                acquired_at_ms=1,
            )
            return raw_id, session_id, original_hash

    raw_id, session_id, original_hash = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(receipt.written_session_ids) for receipt in receipts) == 0
        assert sum(receipt.written_message_count for receipt in receipts) == 0
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        assert (
            index.execute("SELECT content_hash FROM sessions WHERE session_id=?", (session_id,)).fetchone()[0]
            == original_hash
        )
        assert (
            index.execute("SELECT COUNT(*) FROM raw_revision_heads WHERE session_id=?", (session_id,)).fetchone()[0]
            == 0
        )
        rows = index.execute("SELECT raw_id, session_id, decision FROM raw_revision_applications").fetchall()
        assert [tuple(row) for row in rows] == [(raw_id, session_id, "deferred")]


@pytest.mark.asyncio
async def test_original_replay_validation_failure_retires_carrier_before_same_owner_retry(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    payload = json.dumps([_chatgpt_session("same-owner", "neutral content")]).encode()

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="same-owner.json",
                canonical_source_path="same-owner.json",
                acquired_at_ms=1,
            )

    raw_id = await run_archive_fixture_write(root, acquire)
    original = RuntimeError("original caller validation failure")

    def refuse() -> None:
        raise original

    async with prepared_live_convergence_owner(root) as owner:
        with pytest.raises(RuntimeError) as raised:
            (await owner.replay_retained_raw_ids((raw_id,), before_publication=refuse)).require_complete()
        assert raised.value is original
        with ArchiveStore.open_existing(root, read_only=True) as archive:
            index = archive.index_connection
            assert index is not None
            assert index.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        receipts = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(receipt.written_session_ids) for receipt in receipts) == 1
        assert sum(receipt.written_message_count for receipt in receipts) == 1


@pytest.mark.asyncio
async def test_original_suppressed_byte_outcome_needs_no_membership_plan(tmp_path: Path) -> None:
    from polylogue.pipeline.ids import session_id as make_session_id
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_suppression

    root = tmp_path / "archive"
    payload = json.dumps([_chatgpt_session("suppressed-key", "retained neutral content")]).encode()
    session_id = str(make_session_id(Provider.CHATGPT, "suppressed-key"))

    def acquire() -> str:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=payload,
                source_path="suppressed-key.json",
                canonical_source_path="suppressed-key.json",
                acquired_at_ms=1,
            )
            user = archive._open_user_write_connection(initialize=False)
            try:
                upsert_suppression(user, session_id, "neutral suppression", mode="hide", now_ms=1)
                user.commit()
            finally:
                archive._close_user_connection(user)
            return raw_id

    raw_id = await run_archive_fixture_write(root, acquire)
    async with prepared_live_convergence_owner(root) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(receipt.written_session_ids) for receipt in receipts) == 0
        assert sum(receipt.written_message_count for receipt in receipts) == 0
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert index.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert index.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone()[0] == 0
        assert index.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 0
        source = archive.source_connection
        assert source is not None
        # An export raw is censused into its session memberships; the
        # suppressed session's membership is acknowledged as deferred, so
        # Source and Index agree that nothing was applied.
        decisions = source.execute(
            "SELECT decision FROM raw_session_memberships WHERE raw_id=? AND logical_source_key=?",
            (raw_id, "chatgpt-export:suppressed-key"),
        ).fetchall()
        assert [row[0] for row in decisions] == ["deferred"]
        assert (
            source.execute("SELECT status FROM raw_authority_parser_census WHERE raw_id=?", (raw_id,)).fetchone()[0]
            == "complete"
        )
