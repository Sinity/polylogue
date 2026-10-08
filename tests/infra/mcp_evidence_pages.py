"""Synthetic archive construction for MCP evidence-page contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.context_models import ContextImage, ContextSpec
from polylogue.context.compiler import context_snapshot_record_from_image
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import write_context_delivery
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.live_ingest import write_index_session


def seed_evidence_pages(root: Path, *, children: int = 12, receipts: int = 0) -> tuple[str, tuple[str, ...]]:
    """Seed off any running event loop; synchronous callers run directly."""
    return run_off_event_loop(lambda: _seed_evidence_pages(root, children=children, receipts=receipts))


def _seed_evidence_pages(root: Path, *, children: int, receipts: int) -> tuple[str, tuple[str, ...]]:
    with ArchiveStore(root) as archive:
        identifiers = tuple(
            write_index_session(
                archive, ParsedSession(source_name=Provider.CODEX, provider_session_id=f"evidence-{i:03}", messages=[])
            )
            for i in range(children + 1)
        )
        for index, child in enumerate(identifiers[1:]):
            archive._conn.execute(
                """INSERT INTO delegation_facts(delegation_id, parent_session_id, child_session_id,
                    mapping_state, result_status, parent_origin) VALUES (?, ?, ?, 'resolved', 'unknown', 'codex-session')""",
                (f"fixture-edge-{index:03}", identifiers[0], child),
            )
        archive._conn.commit()
    with sqlite3.connect(root / "user.db") as conn:
        for index in range(receipts):
            image = ContextImage(spec=ContextSpec(seed_query=f"receipt-{index:03}", read_views=()), segments=())
            record = context_snapshot_record_from_image(image, boundary="explicit-recall", run_ref="run:fixture")
            write_context_delivery(
                conn,
                image=image,
                record=record,
                recipient_ref="agent:fixture",
                delivered_by_ref="user:fixture",
                delivered_at_ms=1000 + index,
            )
    return identifiers[0], identifiers
