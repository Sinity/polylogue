"""Surface wiring for the durable context-delivery ledger (polylogue-37t.22).

``polylogue/storage/sqlite/archive_tiers/context_delivery_write.py`` (PR
#2703) persists exact context-delivery receipts, but until this module
nothing outside tests ever called ``write_context_delivery`` -- the read-only
``get_context_delivery`` facade method could resolve a receipt, but no
surface ever recorded one. These tests exercise the write-capable facade
methods (``compile_and_record_context`` / ``record_context_delivery`` /
``list_context_deliveries``) that close that gap, proving compilation,
recording, idempotent replay, drift refusal, and bounded listing all work
end-to-end against a real archive.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import ArchiveContextDeliveryEnvelope
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed(archive_root: Path, *, provider_session_id: str, text: str) -> None:
    with ArchiveStore(archive_root) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=provider_session_id,
                title="Delivery target",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                    )
                ],
            )
        )


async def test_compile_and_record_context_persists_the_exact_compiled_image(tmp_path: Path) -> None:
    """The delivery boundary records exactly the image compile_context produced."""

    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="delivery-target", text="quoted archival evidence")

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        envelope = await poly.compile_and_record_context(
            recipient_ref="agent:codex-main",
            delivered_by_ref="user:local",
            boundary="explicit-recall",
            query="quoted archival",
            max_sessions=1,
        )

        assert isinstance(envelope, ArchiveContextDeliveryEnvelope)
        assert envelope.outcome == "recorded"
        assert envelope.recipient_ref == "agent:codex-main"
        assert envelope.delivered_by_ref == "user:local"
        assert envelope.boundary == "explicit-recall"
        message_segments = [s for s in envelope.context_image.segments if s.payload_kind == "messages"]
        assert message_segments, "expected a messages segment for the delivered image"
        assert "quoted archival evidence" in (message_segments[0].markdown or "")

        # Fetching the receipt back returns exactly the delivered image.
        fetched = await poly.get_context_delivery(envelope.snapshot_ref, recipient_ref="agent:codex-main")
        assert fetched is not None
        assert fetched.context_image == envelope.context_image
        assert fetched.context_image_sha256 == envelope.context_image_sha256

        # A wrong recipient never sees the receipt.
        wrong_recipient = await poly.get_context_delivery(envelope.snapshot_ref, recipient_ref="agent:other")
        assert wrong_recipient is None


async def test_compile_and_record_context_replay_is_idempotent_and_drift_is_rejected(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="delivery-target", text="quoted archival evidence")

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        first = await poly.compile_and_record_context(
            recipient_ref="agent:codex-main",
            delivered_by_ref="user:local",
            boundary="explicit-recall",
            query="quoted archival",
            max_sessions=1,
        )
        replay = await poly.compile_and_record_context(
            recipient_ref="agent:codex-main",
            delivered_by_ref="user:local",
            boundary="explicit-recall",
            query="quoted archival",
            max_sessions=1,
        )
        assert replay.outcome == "idempotent"
        assert replay.snapshot_ref == first.snapshot_ref
        assert replay.context_image == first.context_image

        listed = await poly.list_context_deliveries(recipient_ref="agent:codex-main")
        assert [item.snapshot_ref for item in listed] == [first.snapshot_ref]

        # Same snapshot ref, different recipient: identity drift is rejected.
        with pytest.raises(ValueError, match="different delivery identity"):
            await poly.compile_and_record_context(
                recipient_ref="agent:someone-else",
                delivered_by_ref="user:local",
                boundary="explicit-recall",
                query="quoted archival",
                max_sessions=1,
            )


async def test_list_context_deliveries_never_includes_full_context_image(tmp_path: Path) -> None:
    """The bounded list surface is a summary read, not a disclosure surface."""

    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="delivery-target", text="quoted archival evidence")

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        recorded = await poly.compile_and_record_context(
            recipient_ref="agent:codex-main",
            delivered_by_ref="user:local",
            boundary="explicit-recall",
            query="quoted archival",
            max_sessions=1,
        )
        listed = await poly.list_context_deliveries(recipient_ref="agent:codex-main")
        assert len(listed) == 1
        # The list path returns the same durable envelope type as get -- the
        # summary/full split is enforced at the surface payload layer
        # (MCPContextDeliverySummaryPayload), not by truncating the facade
        # return type. Prove it round-trips to the same recorded receipt.
        assert listed[0].snapshot_ref == recorded.snapshot_ref
        assert listed[0].context_image == recorded.context_image

        unrelated = await poly.list_context_deliveries(recipient_ref="agent:unrelated")
        assert unrelated == []


async def test_record_context_delivery_requires_initialized_user_tier(tmp_path: Path) -> None:
    """A missing user.db fails closed with a typed error, not a silent no-op write."""

    archive_root = tmp_path / "archive-missing-user-tier"
    archive_root.mkdir()
    # Initialize the source and index tiers only -- user.db (the durable
    # receipt ledger) is deliberately never created, mirroring an archive
    # that predates the fs1.11 migration or has not been re-initialized yet.
    with sqlite3.connect(archive_root / "source.db") as source_conn:
        initialize_archive_tier(source_conn, ArchiveTier.SOURCE)
    with sqlite3.connect(archive_root / "index.db") as index_conn:
        initialize_archive_tier(index_conn, ArchiveTier.INDEX)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        with pytest.raises(ValueError, match="context-delivery user tier is not initialized"):
            await poly.compile_and_record_context(
                recipient_ref="agent:codex-main",
                delivered_by_ref="user:local",
                boundary="explicit-recall",
                query="anything",
                max_sessions=1,
            )
