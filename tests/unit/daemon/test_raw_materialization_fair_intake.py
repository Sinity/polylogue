"""The ordinary fair intake admits a retained component above its cache budget."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.intake import AdmissionOutcome, AdmissionResult, FairIntakeDispatcher, IntakeClassSpec
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.session_profile_composition import compose_session_profile_callback
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.intake_adapters import RawMaterializationDiscovery, RawMaterializationIntakeAdapter
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _codex_session(native_id: str, messages: tuple[tuple[str, str], ...]) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-19T00:00:00Z"}}
    ]
    for position, (role, text) in enumerate(messages):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{position}",
                    "role": role,
                    "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _blob_sizes(archive_root: Path) -> list[int]:
    with sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True) as conn:
        return [int(row[0]) for row in conn.execute("SELECT blob_size FROM raw_sessions")]


@pytest.mark.asyncio
async def test_fair_intake_converges_component_above_cache_budget_with_profiles(tmp_path: Path) -> None:
    """Fair intake retries a component larger than its cache budget.

    Anti-vacuity: restoring the aggregate payload refusal leaves the index
    without sessions or profiles. Omitting the post-publication handoff leaves
    the profiles empty after the raw session rows appear.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for index in range(2):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session(
                    f"large-component-{index}",
                    (("user", f"question {index}"), ("assistant", f"answer {index}")),
                ),
                source_path="large-component.jsonl",
                acquired_at_ms=index + 1,
            )

    sizes = _blob_sizes(archive_root)
    assert len(sizes) == 2
    cache_bytes = max(sizes)
    assert sum(sizes) > cache_bytes, "the fixture must carry the component/seed skew"

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    try:
        bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())
        owner = RawObservationConvergenceOwner(
            archive_root,
            compute_adapter=compute,
            write_bridge=bridge,
            max_payload_bytes=cache_bytes,
        )
        profiles = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=bridge,
            now=lambda: 0.0,
        )
        discovery = RawMaterializationDiscovery(archive_root, max_payload_bytes=cache_bytes)

        async def admit(raw_id: str) -> AdmissionResult:
            report = await owner.converge_raw_id(raw_id)
            if report.failed or report.pending:
                return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="raw observation pending")
            if report.done:
                await daemon_cli._converge_raw_materialized_session_profiles(archive_root, raw_id, profiles.callback)
            return AdmissionResult(AdmissionOutcome.ADMITTED if report.done else AdmissionOutcome.DUPLICATE)

        dispatcher = FairIntakeDispatcher(
            [
                IntakeClassSpec(
                    name="raw_materialization",
                    adapter=RawMaterializationIntakeAdapter(discovery.discover_pending_raw_ids, admit),
                    page_size=1,
                )
            ]
        )
        passes = [await dispatcher.run_once(budget=cache_bytes) for _ in range(4)]
        assert any(passed.progressed for passed in passes)
        with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
            published = sorted(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions"))
            profiled = sorted(str(row[0]) for row in conn.execute("SELECT session_id FROM session_profiles"))
        assert published == ["codex-session:large-component-0", "codex-session:large-component-1"]
        assert profiled == published
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)
