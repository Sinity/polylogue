"""The daemon's ingest owner re-drives accepted generations a dead process left unfinished."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from polylogue.api import Polylogue
from polylogue.daemon.api_auth import resolve_api_auth_token
from polylogue.daemon.services import ServiceCapability, ServiceProfile
from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.daemon_client import DaemonClient
from polylogue.operations.audit import AuditRepository
from polylogue.operations.daemon_ingest import IngestExecution
from polylogue.operations.daemon_protocol import daemon_operation_spec
from polylogue.operations.ingest_acceptance import INGEST_OPERATION
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.machine_receipts import IngestHistoricalReceiptV2
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.daemon_service_harness import ServiceHarness

_DEAD_OWNER = "pid:999999999:0"


def _chatgpt_export(path: Path) -> None:
    export = {
        "title": "Retained Redrive",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["msg_1"]},
            "msg_1": {
                "id": "msg_1",
                "message": {
                    "id": "msg_1",
                    "author": {"role": "user"},
                    "create_time": 1234567890.0,
                    "content": {"content_type": "text", "parts": ["What survives a restart?"]},
                },
                "parent": "root",
                "children": ["msg_2"],
            },
            "msg_2": {
                "id": "msg_2",
                "message": {
                    "id": "msg_2",
                    "author": {"role": "assistant"},
                    "create_time": 1234567891.0,
                    "content": {"content_type": "text", "parts": ["The accepted generation does."]},
                },
                "parent": "msg_1",
                "children": [],
            },
        },
    }
    path.write_text(json.dumps(export), encoding="utf-8")


@contextmanager
def _serving(archive_root: Path) -> Iterator[tuple[ServiceHarness, Any]]:
    harness = ServiceHarness(profile=ServiceProfile.SURFACES, capabilities={ServiceCapability.API})
    api_server = harness.api_server(archive_root)
    uds_server = harness.uds_server(archive_root, api_server=api_server, auth_token=resolve_api_auth_token(None))
    _api_task = harness.start_server("api_server", api_server)
    _uds_task = harness.start_server("uds_server", uds_server)
    yield harness, api_server


async def _die_after_acceptance(archive_root: Path, source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept the source, then end the attempt as a killed process would: no finalization, no fence."""

    async def killed(self: IngestExecution, *_args: object, **_kwargs: object) -> tuple[()]:
        with sqlite3.connect(archive_root / "audit.db") as audit:
            audit.execute("UPDATE operation_attempts SET worker_id = ? WHERE state = 'running'", (_DEAD_OWNER,))
        raise RuntimeError("process killed after acceptance")

    async def nothing(self: IngestExecution, *_args: object) -> None:
        return None

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    with monkeypatch.context() as patch:
        patch.setattr(IngestExecution, "input_page", killed)
        patch.setattr(IngestExecution, "mark_unknown", nothing)
        patch.setattr(IngestExecution, "fence", nothing)
        with _serving(archive_root) as (harness, _api_server):
            try:
                with pytest.raises(RuntimeError, match="did not complete"):
                    await archive.parse_file(source, source_name="redrive")
            finally:
                try:
                    await harness.close()
                finally:
                    await archive.close()


async def _restart_and_settle(archive_root: Path) -> None:
    """Start the ingest owner again and wait for its re-drive to settle."""
    with _serving(archive_root) as (harness, api_server):
        try:
            redrive = api_server.operation_runtime._redrive
            assert redrive is not None
            await asyncio.wrap_future(redrive)
        finally:
            await harness.close()


def _only_request(archive_root: Path) -> tuple[str, dict[str, object]]:
    with sqlite3.connect(archive_root / "audit.db") as audit:
        audit.row_factory = sqlite3.Row
        record = dict(audit.execute("SELECT * FROM machine_requests").fetchone())
        operation_id = str(
            audit.execute(
                "SELECT operation_id FROM operation_runs WHERE operation_name = ?", (INGEST_OPERATION,)
            ).fetchone()[0]
        )
    return operation_id, record


def _session_titles(archive_root: Path) -> list[str]:
    index = ArchiveLocation.resolve(archive_root).active_index_path
    with sqlite3.connect(index) as conn:
        return [str(row[0]) for row in conn.execute("SELECT title FROM sessions ORDER BY title")]


def _archive(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "inputs" / "export.json"
    source.parent.mkdir()
    _chatgpt_export(source)
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root):
        pass
    return archive_root, source


@pytest.mark.timeout(300)
async def test_accepted_generation_materializes_after_restart_without_its_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill after acceptance, then the input removed: the restarted owner still materializes it.

    Anti-vacuity: restore ``IngestRecovery`` to terminalize every interrupted
    ingest as not replayable, or drop ``start_accepted_ingest_redrive`` from
    the HTTP server, and no session appears while the request stays failed or
    indeterminate.
    """
    archive_root, source = _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    assert _session_titles(archive_root) == []
    source.unlink()

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == ["Retained Redrive"]
    operation_id, record = _only_request(archive_root)
    audit = AuditRepository.for_archive_root(archive_root)
    with audit.settled_machine_read():
        state = machine_request_state(audit, record)
        history = audit.historical_machine_receipt(operation_id)
        run = audit.get_operation(operation_id)
    assert run is not None and run["status"] == "completed"
    assert state["outcome"] in {"completed", "degraded"}
    assert isinstance(history, IngestHistoricalReceiptV2)
    assert history.summary.enumeration_complete and history.summary.source_complete
    assert history.input_count == 1

    # The original request reference now answers with its terminal receipt.
    spec = daemon_operation_spec("ingest")
    assert spec is not None
    with _serving(archive_root) as (harness, _api_server):
        try:
            client = DaemonClient(
                daemon_socket_path(archive_root),
                timeout_s=spec.deadline_s,
                auth_token=resolve_api_auth_token(None),
            )
            path = str(source.resolve())
            envelope = await asyncio.to_thread(
                client.operation_to_completion,
                "ingest",
                {"path": path, "source_path": path, "source_name": "redrive", "idempotency_key": None},
                archive_root=str(archive_root),
                request_id=str(record["request_id"]),
            )
        finally:
            await harness.close()
    assert envelope is not None
    assert envelope["outcome"] == state["outcome"]
    assert envelope["result"]["historical_receipt"]["source_generation_id"] == record["artifact_ref"]


@pytest.mark.timeout(300)
async def test_stopped_request_is_not_redriven(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request whose stop was fenced before the process died keeps that decision.

    Anti-vacuity: drop the ``stop_reason IS NULL`` condition from the owner's
    discovery and claim, and the cancelled request's sessions materialize.
    """
    archive_root, source = _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    with sqlite3.connect(archive_root / "audit.db") as audit:
        audit.execute("UPDATE machine_requests SET stop_reason = 'cancelled', stopped_at_ms = 1")

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == []
    operation_id, _record = _only_request(archive_root)
    with sqlite3.connect(archive_root / "audit.db") as audit:
        assert audit.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("failed", "recovery_not_replayable")


@pytest.mark.timeout(300)
async def test_undrivable_generation_fails_instead_of_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A generation whose enumeration decoder is gone ends failed, not interrupted forever.

    Anti-vacuity: leave a failed re-drive at ``mark_unknown`` and the run
    stays ``interrupted`` for every later start.
    """
    archive_root, source = _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    monkeypatch.setattr(
        "polylogue.operations.daemon_ingest.retained_enumeration_fingerprint", lambda: "retired-decoder"
    )

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == []
    operation_id, record = _only_request(archive_root)
    audit = AuditRepository.for_archive_root(archive_root)
    with audit.settled_machine_read():
        state = machine_request_state(audit, record)
        run = audit.get_operation(operation_id)
    assert run is not None and (run["status"], run["terminal_reason"]) == ("failed", "domain_failure")
    assert state["outcome"] == "failed"
