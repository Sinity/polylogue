"""Prepare a neutral publication using actual retained intake and Source custody."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import polylogue.config as config
from polylogue.core.enums import Provider
from polylogue.sinex.models import PublicationObligation, PublicationPayload
from polylogue.sinex.obligations import list_obligations, load_payload
from tests.infra.live_provider_proof import native_proof_artifact
from tests.infra.retained_replay import publish_retained_payload

if TYPE_CHECKING:
    from polylogue.sinex.service import DrainSummary, PublicationService

_T = TypeVar("_T")


async def drain_publication(service: PublicationService, *, redrive: bool = False) -> DrainSummary:
    """Run the actual service with daemon admission for its short writes."""
    from polylogue.core.stage_admission import admit_stage_write, stage_write_admission
    from polylogue.core.write_lease import adopt_write_lease
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge

    coordinator = DaemonWriteCoordinator(archive_root=service.source_db_path.parent)
    bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())

    def admission(actor: str, work: Callable[[], _T]) -> _T:
        with bridge.hold(actor) as delegation, adopt_write_lease(delegation):
            return work()

    def drain() -> DrainSummary:
        with stage_write_admission(admission):
            if redrive:
                admit_stage_write("convergence.stage.sinex_publication.fixture_redrive", service.reset_retryable)
            return service.drain_once()

    task = asyncio.create_task(asyncio.to_thread(drain))
    try:
        return await asyncio.shield(task)
    finally:
        # The network worker must finish before its writer coordinator retires.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        if not await coordinator.shutdown(timeout=float("inf")):
            raise RuntimeError("publication fixture writer did not settle")


async def acquired_publication(
    root: Path, *, extra_grok_responses: int = 0
) -> tuple[PublicationObligation, PublicationPayload]:
    """Return the committed production outbox, including a real attachment."""
    root.mkdir(parents=True, exist_ok=True)
    original_config = config.load_polylogue_config
    configured = replace(
        original_config(), _data={**original_config()._data, "sinex_mode": "mirror", "archive_root": str(root)}
    )
    config.load_polylogue_config = lambda **_kwargs: configured
    try:
        envelope, _, _ = native_proof_artifact(
            root,
            "native-inline-attachment-v1.json",
            Provider.GROK,
            extra_grok_responses=extra_grok_responses,
        )
        payload = json.dumps(envelope).encode()
        revision = None
        if extra_grok_responses:
            from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind

            revision = RawRevisionEnvelope(
                f"grok-export:{envelope['session']['provider_session_id']}",
                RawRevisionKind.FULL,
                hashlib.sha256(payload).hexdigest(),
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            )
        await publish_retained_payload(
            root,
            provider=Provider.GROK,
            payload=payload,
            source_path="/synthetic/native-grok-capture.json",
            acquired_at_ms=1,
            revision=revision,
        )
    finally:
        config.load_polylogue_config = original_config
    with sqlite3.connect(root / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = list_obligations(conn)
        if len(rows) != 1:
            raise AssertionError("synthetic publication has exactly one committed obligation")
        return rows[0], load_payload(conn, rows[0])
