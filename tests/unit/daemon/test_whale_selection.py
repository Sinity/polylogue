"""Whale escalation selects on the budget the ordinary admission actually uses.

``RawObservationDerivation.compute`` expands a raw's membership and budgets
``sum(sizes.values())`` over the whole component, while
``RawMaterializationDiscovery`` reports each seed row's own size. Selecting the
whale candidate on the seed size therefore left a component whose members are
each inside the ordinary limit -- but whose total is not -- permanently stuck:
too large for every ordinary admission, never offered to the whale owner.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.operations.intake_adapters import RawMaterializationDiscovery
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
async def test_whale_selection_measures_component(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two revisions of one source path, each under the limit, together over it.

    That skew is the whole point of the fixture: a seed that alone exceeds the
    ordinary limit is selected by the per-raw predicate and the component
    predicate alike, so it could not separate them.

    Anti-vacuity: restore ``payload_bytes > _RAW_MATERIALIZATION_DAEMON_BLOB_
    LIMIT_BYTES`` as the whole predicate and this returns ``False`` with no
    session published.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for index in range(2):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session(
                    f"whale-component-{index}",
                    (("user", f"question {index}"), ("assistant", f"answer {index}")),
                ),
                source_path="whale-component.jsonl",
                acquired_at_ms=index + 1,
            )

    sizes = _blob_sizes(archive_root)
    assert len(sizes) == 2
    ordinary_limit = max(sizes)
    assert sum(sizes) > ordinary_limit, "the fixture must carry the component/seed skew"

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    try:
        bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())
        owner = RawObservationConvergenceOwner(
            archive_root,
            compute_adapter=compute,
            write_bridge=bridge,
            max_payload_bytes=ordinary_limit,
        )
        monkeypatch.setattr("polylogue.paths.archive_root", lambda: archive_root)
        monkeypatch.setattr(daemon_cli, "_RAW_MATERIALIZATION_DAEMON_BLOB_LIMIT_BYTES", ordinary_limit)
        monkeypatch.setattr(daemon_cli, "_resolve_raw_materialization_whale_blob_limit_bytes", lambda: 1_000_000)
        monkeypatch.setattr(daemon_cli, "_drain_whale_receipt_outbox", lambda: asyncio.sleep(0))
        monkeypatch.setattr(daemon_cli, "_publish_whale_receipt", lambda **_kwargs: asyncio.sleep(0))

        assert await daemon_cli._maybe_run_raw_materialization_whale_pass(
            raw_observation_owner=owner,
            raw_intake_discovery=RawMaterializationDiscovery(archive_root, max_payload_bytes=ordinary_limit),
        )
        with sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True) as conn:
            published = sorted(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions"))
        assert published == ["codex-session:whale-component-0", "codex-session:whale-component-1"]
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)
