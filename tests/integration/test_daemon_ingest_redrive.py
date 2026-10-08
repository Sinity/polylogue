"""The daemon's ingest owner re-drives accepted generations a dead process left unfinished."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import pytest

from polylogue.api import Polylogue
from polylogue.core.enums import ValidationMode
from polylogue.daemon.api_auth import resolve_api_auth_token
from polylogue.daemon.operation_runtime import DaemonOperationRuntime
from polylogue.daemon.services import ServiceCapability, ServiceProfile
from polylogue.daemon.socket_path import daemon_socket_path
from polylogue.daemon_client import DaemonClient
from polylogue.operations.audit import AuditRepository
from polylogue.operations.daemon_ingest import IngestExecution, SourceReceiptSpool
from polylogue.operations.daemon_protocol import daemon_operation_spec
from polylogue.operations.ingest_acceptance import INGEST_OPERATION
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.machine_receipts import IngestHistoricalReceiptV2
from polylogue.operations.operation_context import PinnedOperationRead
from polylogue.operations.raw_observation_owner import RetainedMaterializationResult
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.prepared_jsonl import PreparedJsonl
from polylogue.sources.revision_backfill import RetainedSessionRead
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
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


async def _archive(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "inputs" / "export.json"
    source.parent.mkdir()
    _chatgpt_export(source)
    archive_root = tmp_path / "archive"
    await run_archive_fixture_write(archive_root, lambda: bootstrap_archive_root(archive_root))
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
    archive_root, source = await _archive(tmp_path)
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
    archive_root, source = await _archive(tmp_path)
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
    archive_root, source = await _archive(tmp_path)
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


def _run_and_state(archive_root: Path) -> tuple[dict[str, object] | None, dict[str, object]]:
    operation_id, record = _only_request(archive_root)
    audit = AuditRepository.for_archive_root(archive_root)
    with audit.settled_machine_read():
        return audit.get_operation(operation_id), machine_request_state(audit, record)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("transient", ["reprepare", "backpressure"])
async def test_a_transient_refusal_retries_the_claimed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transient: str
) -> None:
    """A moved generation or full admission retries the same claimed run to its receipt.

    Anti-vacuity (Codex P1, #5717): leaving a reprepare-required run as it is
    keeps it ``running`` under the live owner, which discovery never returns
    again, and settling backpressure as failed terminalizes a run a retry
    completes; either way no session materializes. Retrying under the old
    pinned identity fails the moved generation as ``archive_identity_stale``,
    and cleaning up through the saturated admission class escapes the retry.
    """
    from polylogue.core.compute import DaemonBackpressureError
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.operations.daemon_ingest import IngestReprepareRequiredError

    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    # The run's archive publication is ``materialize`` (the resident Raw
    # owner); the refusal lands where publication starts.
    original = IngestExecution.materialize
    original_compute = DaemonOperationRuntime.compute_phase
    refusals = {"left": 1, "saturated": 0}

    async def refuse_once(self: IngestExecution, generation_id: str) -> Any:
        if refusals["left"]:
            refusals["left"] -= 1
            refusals["saturated"] = 1
            if transient == "reprepare":
                # The promotion moved the archive identity this run pinned.
                self.observed_identity = "a-generation-since-promoted"
                raise IngestReprepareRequiredError("ingest publication generation changed; reprepare required")
            raise DaemonBackpressureError("control admission is full")
        return await original(self, generation_id)

    async def saturated(self: DaemonOperationRuntime, work: Any) -> Any:
        # Admission stays full for the next submission after a refusal.
        if refusals["saturated"]:
            refusals["saturated"] = 0
            raise DaemonBackpressureError("control admission is still full")
        return await original_compute(self, work)

    monkeypatch.setattr(IngestExecution, "materialize", refuse_once)
    monkeypatch.setattr(DaemonOperationRuntime, "compute_phase", saturated)
    await _restart_and_settle(archive_root)

    assert refusals["left"] == 0
    assert _session_titles(archive_root) == ["Retained Redrive"]
    run, state = _run_and_state(archive_root)
    assert run is not None and run["status"] == "completed", run
    assert state["outcome"] in {"completed", "degraded"}


@pytest.mark.timeout(300)
async def test_a_redrive_honors_the_accepted_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request whose accepted deadline passed while the process was down is not materialized.

    Anti-vacuity (Codex P1, #5717): consult only the owner's stop and
    cancellation and the expired request completes as a mutation.
    """
    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    with sqlite3.connect(archive_root / "audit.db") as audit:
        audit.execute("UPDATE machine_requests SET accepted_deadline_unix_ms = 1")

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == []
    run, state = _run_and_state(archive_root)
    assert run is not None and run["status"] != "completed", run
    assert state["outcome"] not in {"completed", "degraded"}


@pytest.mark.timeout(300)
async def test_a_redrive_after_partial_publication_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead attempt that already published leaves a re-drive that cannot count it.

    Anti-vacuity (Codex P1, #5717): finalize the re-drive as applied and its
    receipt reports zero changed sessions although this request wrote one.
    """
    archive_root, source = await _archive(tmp_path)

    async def killed(self: IngestExecution, *_args: object, **_kwargs: object) -> None:
        with sqlite3.connect(archive_root / "audit.db") as audit:
            audit.execute("UPDATE operation_attempts SET worker_id = ? WHERE state = 'running'", (_DEAD_OWNER,))
        raise RuntimeError("process killed before finalization")

    async def nothing(self: IngestExecution, *_args: object) -> None:
        return None

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    with monkeypatch.context() as patch:
        patch.setattr(IngestExecution, "finalize", killed)
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
    assert _session_titles(archive_root) == ["Retained Redrive"]

    await _restart_and_settle(archive_root)

    _run, state = _run_and_state(archive_root)
    assert state["outcome"] == "indeterminate", state
    # Fenced: a later start neither re-drives nor reclaims it (Codex P1, #5717).
    with sqlite3.connect(archive_root / "audit.db") as audit:
        assert audit.execute("SELECT stop_reason FROM machine_requests").fetchone()[0] is not None
        attempts = audit.execute("SELECT COUNT(*) FROM operation_attempts").fetchone()[0]
    await _restart_and_settle(archive_root)
    with sqlite3.connect(archive_root / "audit.db") as audit:
        assert audit.execute("SELECT COUNT(*) FROM operation_attempts").fetchone()[0] == attempts
    _run, state = _run_and_state(archive_root)
    assert state["outcome"] == "indeterminate", state


@pytest.mark.timeout(300)
async def test_a_stopped_partial_ingest_stays_indeterminate_across_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request stopped after its content was published keeps its indeterminate outcome.

    Anti-vacuity (Codex P1, #5717): terminalize every stopped ingest as not
    replayable and the next startup rewrites it as failed with no effect,
    although its sessions are in the archive.
    """
    archive_root, source = await _archive(tmp_path)

    async def killed(self: IngestExecution, *_args: object, **_kwargs: object) -> None:
        with sqlite3.connect(archive_root / "audit.db") as audit:
            audit.execute("UPDATE operation_attempts SET worker_id = ? WHERE state = 'running'", (_DEAD_OWNER,))
            audit.execute("UPDATE machine_requests SET stop_reason = 'cancelled', stopped_at_ms = 1")
        raise RuntimeError("process killed after the cancel fence")

    async def nothing(self: IngestExecution, *_args: object) -> None:
        return None

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    with monkeypatch.context() as patch:
        patch.setattr(IngestExecution, "finalize", killed)
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
    assert _session_titles(archive_root) == ["Retained Redrive"]

    await _restart_and_settle(archive_root)

    run, state = _run_and_state(archive_root)
    assert run is not None and run["status"] != "failed", run
    assert state["outcome"] == "indeterminate", state


@pytest.mark.timeout(300)
async def test_runs_are_claimed_before_the_listeners_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The server exposes its listeners only after the owner claimed every interrupted run.

    Anti-vacuity (Codex P1, #5717): schedule the re-drive fire-and-forget and
    a slow claim leaves the run on its dead attempt when the listeners open,
    so an immediate resend reads it as ``interrupted``.
    """
    from polylogue.operations import daemon_ingest

    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    original_claim = daemon_ingest.claim_interrupted_ingest

    async def slow_claim(*args: Any) -> bool:
        await asyncio.sleep(1.0)
        return await original_claim(*args)

    monkeypatch.setattr(daemon_ingest, "claim_interrupted_ingest", slow_claim)
    with _serving(archive_root) as (harness, api_server):
        try:
            with sqlite3.connect(archive_root / "audit.db") as audit:
                owners = [row[0] for row in audit.execute("SELECT worker_id FROM operation_attempts")]
            assert any(owner != _DEAD_OWNER for owner in owners), owners
            redrive = api_server.operation_runtime._redrive
            assert redrive is not None
            await asyncio.wrap_future(redrive)
        finally:
            await harness.close()


@pytest.mark.timeout(300)
async def test_a_retry_after_this_attempts_materialization_still_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backpressure after the re-drive published its sessions does not make its retry indeterminate.

    Anti-vacuity (Codex P1, #5717): reclassify prior materialization on every
    retry, or count this attempt's own re-publication as unchanged, and the
    retry sees its own sessions as a dead attempt's and finalizes indeterminate.
    """
    from polylogue.core.compute import DaemonBackpressureError

    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    original = IngestExecution.converge_profiles
    refusals = {"left": 1}

    async def refuse_once(self: IngestExecution, receipt: Any) -> Any:
        if refusals["left"]:
            refusals["left"] -= 1
            raise DaemonBackpressureError("control admission is full")
        return await original(self, receipt)

    monkeypatch.setattr(IngestExecution, "converge_profiles", refuse_once)
    await _restart_and_settle(archive_root)

    assert refusals["left"] == 0
    run, state = _run_and_state(archive_root)
    assert run is not None and run["status"] == "completed", run
    assert state["outcome"] in {"completed", "degraded"}, state


@pytest.mark.timeout(300)
async def test_shutdown_releases_every_claimed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An owner shut down mid re-drive hands every claimed run back, so a server in the same process reclaims them.

    The stop is the production one: the server's shutdown closes its operation
    runtime while the first of two claimed runs is being driven.

    Anti-vacuity (Codex P1, #5717): release only the active run and the later
    claimed one keeps a ``running`` attempt owned by this live process, which
    discovery never returns, so it never completes.
    """
    import threading

    archive_root = await _two_interrupted_ingests(tmp_path, monkeypatch)
    entered = threading.Event()
    original = IngestExecution.input_page

    async def held_until_shutdown(self: IngestExecution, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        while self.stop_reason() is None:
            await asyncio.sleep(0.01)
        return await original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(IngestExecution, "input_page", held_until_shutdown)
        with _serving(archive_root) as (harness, _api_server):
            try:
                assert await asyncio.to_thread(entered.wait, 60), "the re-drive never started driving a run"
            finally:
                await harness.close()
    with sqlite3.connect(archive_root / "audit.db") as audit:
        assert audit.execute("SELECT COUNT(*) FROM operation_attempts WHERE state = 'running'").fetchone() == (0,)
    # The shutdown interrupted the drive: neither generation was materialized.
    assert _session_titles(archive_root) == []

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == ["Retained Redrive", "Second Redrive"]


@pytest.mark.timeout(300)
async def test_a_transient_refusal_of_a_fresh_ingest_stays_redrivable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh request's post-acceptance backpressure leaves its generation for the ingest owner.

    Anti-vacuity (Codex P1, #5717): fence it as ``refused`` like a permanent
    failure and discovery excludes the stopped request, so it never materializes.
    """
    from polylogue.core.compute import DaemonBackpressureError

    archive_root, source = await _archive(tmp_path)
    original = IngestExecution.materialize
    refusals = {"left": 1}

    async def refuse_once(self: IngestExecution, generation_id: str) -> Any:
        if refusals["left"]:
            refusals["left"] -= 1
            raise DaemonBackpressureError("control admission is full")
        return await original(self, generation_id)

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    with monkeypatch.context() as patch:
        patch.setattr(IngestExecution, "materialize", refuse_once)
        with _serving(archive_root) as (harness, _api_server):
            try:
                with pytest.raises(Exception):  # noqa: B017 - the surface's error type is not this contract
                    await archive.parse_file(source, source_name="redrive")
            finally:
                try:
                    await harness.close()
                finally:
                    await archive.close()
    _operation_id, record = _only_request(archive_root)
    assert record["stop_reason"] is None

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == ["Retained Redrive"]


@pytest.mark.timeout(60)
async def test_starting_the_redrive_on_its_owner_loop_does_not_block_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``polylogued run`` constructs its server on the loop the re-drive runs on.

    Anti-vacuity (Codex P1, #5717): wait for the claim inside the start call
    and the loop that must run the claim is blocked, so this test times out.
    """
    from typing import cast

    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.operations import daemon_ingest

    claimed: list[bool] = []

    async def redrive(*_args: object, on_claimed: Any = None, **_kwargs: object) -> None:
        await asyncio.sleep(0.05)
        claimed.append(True)
        on_claimed()

    monkeypatch.setattr(daemon_ingest, "redrive_accepted_ingests", redrive)
    runtime = DaemonOperationRuntime(
        tmp_path,
        write_bridge=cast(Any, object()),
        execution_kernel=cast(Any, object()),
        raw_observation_owner=cast(Any, object()),
        owner_loop=asyncio.get_running_loop(),
        session_maintenance=cast(Any, object()),
    )
    runtime.start_accepted_ingest_redrive()
    assert claimed == []
    await runtime.accepted_ingest_redrive_claimed()
    assert claimed == [True]


async def _two_interrupted_ingests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime

    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    second = tmp_path / "inputs" / "second.json"
    export = json.loads(source.read_text(encoding="utf-8"))
    export["title"] = "Second Redrive"
    second.write_text(json.dumps(export), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(DaemonOperationRuntime, "start_accepted_ingest_redrive", lambda self: None)
        await _die_after_acceptance(archive_root, second, monkeypatch)
    return archive_root


@pytest.mark.timeout(300)
async def test_shutdown_during_the_claim_phase_releases_earlier_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owner stopping between two claims hands the first claim back.

    Anti-vacuity (Codex P1, #5717): return from the claim loop without
    releasing and the first run keeps a ``running`` attempt owned by this
    live process, which no later owner in it can reclaim.
    """
    from polylogue.operations import daemon_ingest

    archive_root = await _two_interrupted_ingests(tmp_path, monkeypatch)
    original_claim = daemon_ingest.claim_interrupted_ingest

    async def claim_then_stop(runtime: Any, audit: Any, operation_id: str) -> bool:
        taken = await original_claim(runtime, audit, operation_id)
        with runtime._condition:
            runtime._closing = True
        return taken

    with monkeypatch.context() as patch:
        patch.setattr(daemon_ingest, "claim_interrupted_ingest", claim_then_stop)
        await _restart_and_settle(archive_root)
    with sqlite3.connect(archive_root / "audit.db") as audit:
        assert audit.execute("SELECT COUNT(*) FROM operation_attempts WHERE state = 'running'").fetchone() == (0,)

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == ["Retained Redrive", "Second Redrive"]


@pytest.mark.timeout(300)
async def test_a_cancel_committed_before_finalization_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stop fenced after the last stop check still refuses the applied receipt.

    Anti-vacuity (Codex P1, #5717): finalize without rechecking the durable
    stop under the writer and the cancelled request completes as applied.
    """
    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    original = IngestExecution.historical_receipt

    async def cancel_then_receipt(self: IngestExecution, *args: Any, **kwargs: Any) -> Any:
        with sqlite3.connect(archive_root / "audit.db") as audit:
            audit.execute("UPDATE machine_requests SET stop_reason = 'cancelled', stopped_at_ms = 1")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(IngestExecution, "historical_receipt", cancel_then_receipt)
    await _restart_and_settle(archive_root)

    run, state = _run_and_state(archive_root)
    assert run is not None and run["status"] != "completed", run
    assert state["outcome"] not in {"completed", "degraded"}, state


@pytest.mark.timeout(300)
async def test_an_identity_moved_between_reads_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A generation promoted between two reads of one drive retries instead of failing.

    Anti-vacuity (Codex P1, #5717): raise the stale identity as a plain
    ``ValueError`` and the re-drive settles the accepted generation as failed.
    """
    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    original = IngestExecution.input_page
    moves = {"left": 1}

    async def moved_once(self: IngestExecution, *args: Any, **kwargs: Any) -> Any:
        if moves["left"]:
            moves["left"] -= 1
            self.observed_identity = "a-generation-promoted-since"
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(IngestExecution, "input_page", moved_once)
    await _restart_and_settle(archive_root)

    assert moves["left"] == 0
    run, state = _run_and_state(archive_root)
    assert run is not None and run["status"] == "completed", run
    assert state["outcome"] in {"completed", "degraded"}, state


@pytest.mark.timeout(300)
async def test_a_failed_claim_phase_releases_earlier_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A claim that raises hands the claims taken before it back.

    Anti-vacuity (Codex P1, #5717): unwind the claim loop without releasing
    and the first run keeps a ``running`` attempt owned by this live process.
    """
    from polylogue.operations import daemon_ingest

    archive_root = await _two_interrupted_ingests(tmp_path, monkeypatch)
    original_claim = daemon_ingest.claim_interrupted_ingest
    claims = {"n": 0}

    async def fail_second(runtime: Any, audit: Any, operation_id: str) -> bool:
        claims["n"] += 1
        if claims["n"] == 2:
            raise sqlite3.OperationalError("disk I/O error")
        return await original_claim(runtime, audit, operation_id)

    with monkeypatch.context() as patch:
        patch.setattr(daemon_ingest, "claim_interrupted_ingest", fail_second)
        await _restart_and_settle(archive_root)
    with sqlite3.connect(archive_root / "audit.db") as audit:
        assert audit.execute("SELECT COUNT(*) FROM operation_attempts WHERE state = 'running'").fetchone() == (0,)

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == ["Retained Redrive", "Second Redrive"]


@pytest.mark.timeout(300)
async def test_a_transient_storage_fault_retries_the_redrive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A locked database during a re-drive is retried, not settled as failed.

    Anti-vacuity (Codex P1, #5717): retry only the three typed transients and
    ``database is locked`` permanently fails the accepted generation.
    """
    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    original = IngestExecution.input_page
    faults = {"left": 1}

    async def locked_once(self: IngestExecution, *args: Any, **kwargs: Any) -> Any:
        if faults["left"]:
            faults["left"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(IngestExecution, "input_page", locked_once)
    await _restart_and_settle(archive_root)

    run, state = _run_and_state(archive_root)
    assert run is not None and run["status"] == "completed", run
    assert state["outcome"] in {"completed", "degraded"}, state


@pytest.mark.timeout(300)
async def test_an_accepted_deadline_passing_before_finalization_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline passing after the last stop check still refuses the applied receipt.

    Anti-vacuity (Codex P1, #5717): recheck only a recorded ``stop_reason`` in
    the final writer and the expired request completes.
    """
    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    original = IngestExecution.historical_receipt

    async def expire_then_receipt(self: IngestExecution, *args: Any, **kwargs: Any) -> Any:
        assert self.record is not None
        self.record = {**self.record, "accepted_deadline_unix_ms": 1}
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(IngestExecution, "historical_receipt", expire_then_receipt)
    await _restart_and_settle(archive_root)

    run, state = _run_and_state(archive_root)
    assert run is not None and run["status"] != "completed", run
    assert state["outcome"] not in {"completed", "degraded"}, state


@pytest.mark.timeout(300)
async def test_profile_receipts_survive_a_transient_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry after profiles converged reuses their receipts instead of converging again.

    Anti-vacuity (Codex P1, #5717): reconverge on the retry and the published
    profiles come back ``already_satisfied``, conflicting with persisted pages.
    """
    from polylogue.core.compute import DaemonBackpressureError

    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    original_converge = IngestExecution.converge_profiles
    original_receipt = IngestExecution.historical_receipt
    converged = {"n": 0}
    refusals = {"left": 1}

    async def counting(self: IngestExecution, receipt: Any) -> Any:
        converged["n"] += 1
        return await original_converge(self, receipt)

    async def refuse_once(self: IngestExecution, *args: Any, **kwargs: Any) -> Any:
        if refusals["left"]:
            refusals["left"] -= 1
            raise DaemonBackpressureError("control admission is full")
        return await original_receipt(self, *args, **kwargs)

    monkeypatch.setattr(IngestExecution, "converge_profiles", counting)
    monkeypatch.setattr(IngestExecution, "historical_receipt", refuse_once)
    await _restart_and_settle(archive_root)

    assert refusals["left"] == 0
    assert converged["n"] == 1
    run, _state = _run_and_state(archive_root)
    assert run is not None and run["status"] == "completed", run


def test_a_refusal_seen_by_two_drives_is_one_refusal(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5717): append refusals without a key and a
    retried drive reports the same refused membership twice."""
    from typing import cast

    from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind, RetainedRawDecodeRefusalError

    execution = IngestExecution.__new__(IngestExecution)
    execution._setup(cast(Any, None), tmp_path, cast(Any, None))
    refusal = RetainedRawDecodeRefusalError(
        "raw-1", RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT, "neutral corrupt input"
    )
    execution.record_refusal("key", refusal)
    execution.record_refusal("key", refusal)
    with sqlite3.connect(execution.state_path) as state:
        assert state.execute("SELECT COUNT(*) FROM refusals").fetchone() == (1,)
    execution.state_path.unlink(missing_ok=True)


@pytest.mark.timeout(300)
async def test_a_refusal_before_authority_loads_leaves_no_running_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backpressure right after acceptance settles the attempt, then a restart re-drives it.

    Anti-vacuity (Codex P1, #5717): ``mark_unknown`` returning while the
    authority is unloaded leaves the attempt ``running`` under this live
    process, which no owner reclaims.
    """
    from polylogue.core.compute import DaemonBackpressureError
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime

    archive_root, source = await _archive(tmp_path)
    original_compute = DaemonOperationRuntime.compute_phase
    refusals = {"left": 1}

    async def refuse_authority_load(self: DaemonOperationRuntime, work: Any) -> Any:
        if refusals["left"] and getattr(work, "__name__", "") == "load_started":
            refusals["left"] -= 1
            raise DaemonBackpressureError("control admission is full")
        return await original_compute(self, work)

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    with monkeypatch.context() as patch:
        patch.setattr(DaemonOperationRuntime, "compute_phase", refuse_authority_load)
        with _serving(archive_root) as (harness, _api_server):
            try:
                with pytest.raises(Exception):  # noqa: B017 - the surface's error type is not this contract
                    await archive.parse_file(source, source_name="redrive")
            finally:
                try:
                    await harness.close()
                finally:
                    await archive.close()
    assert refusals["left"] == 0
    with sqlite3.connect(archive_root / "audit.db") as audit:
        assert audit.execute("SELECT COUNT(*) FROM operation_attempts WHERE state = 'running'").fetchone() == (0,)

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == ["Retained Redrive"]


async def test_a_failed_redrive_construction_releases_every_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-drive whose scratch state cannot be created hands its claims back.

    Anti-vacuity (Codex P1, #5717): build the execution outside the cleanup
    path and both runs keep a ``running`` attempt owned by this live process.
    """
    import errno

    from polylogue.operations import daemon_ingest

    archive_root = await _two_interrupted_ingests(tmp_path, monkeypatch)

    def no_scratch(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(daemon_ingest, "IngestRedrive", no_scratch)
        await _restart_and_settle(archive_root)
    with sqlite3.connect(archive_root / "audit.db") as audit:
        assert audit.execute("SELECT COUNT(*) FROM operation_attempts WHERE state = 'running'").fetchone() == (0,)

    await _restart_and_settle(archive_root)

    assert _session_titles(archive_root) == ["Retained Redrive", "Second Redrive"]


async def test_a_watcher_only_daemon_redrives_accepted_ingests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ingest owner a no-API daemon composes materializes an accepted generation.

    Anti-vacuity (Codex P1, #5717): compose the ingest owner only inside the
    API server and ``polylogued run --no-api`` has no runtime to re-drive the
    run, so no session appears (the composition-route test pins that the
    watcher-only daemon starts this owner's re-drive).
    """
    from polylogue.daemon.cli import compose_ingest_owner
    from polylogue.daemon.http import _StandaloneWriteRuntime

    archive_root, source = await _archive(tmp_path)
    await _die_after_acceptance(archive_root, source, monkeypatch)
    source.unlink()

    from polylogue.core.compute import BoundedComputeAdapter

    kernel = BoundedComputeAdapter(max_workers=1, queue_units=1, queue_bytes=0)
    writer = _StandaloneWriteRuntime(archive_root, compute_adapter=kernel)
    runtime, _profiles = compose_ingest_owner(
        archive_root,
        writer.bridge,
        compute_adapter=kernel,
        validation_mode=ValidationMode.ADVISORY,
    )
    try:
        await asyncio.to_thread(runtime.start_accepted_ingest_redrive)
        redrive = runtime._redrive
        assert redrive is not None
        await asyncio.wrap_future(redrive)
    finally:
        await asyncio.to_thread(writer.close, before_drain=runtime.shutdown, after_drain=kernel.shutdown)

    assert _session_titles(archive_root) == ["Retained Redrive"]


async def test_ingest_counts_sessions_convergence_published_before_its_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convergence publishing an accepted raw first still counts it for the ingest.

    Single-pass convergence lets the daemon's raw owner publish the ingest's
    raw before the ingest materializes it; the ingest then writes nothing, yet
    its receipt reports what the archive holds for the input it introduced.
    """
    archive_root, source = await _archive(tmp_path)
    original = IngestExecution.materialize
    reached: list[str] = []

    async def materialize(self: IngestExecution, generation_id: str) -> SourceReceiptSpool:
        receipt = await self.receipt(generation_id)
        try:
            raw_ids = receipt.raw_page()
        finally:
            receipt.close()
        assert len(raw_ids) == 1
        runtime = self.runtime
        assert isinstance(runtime, DaemonOperationRuntime)
        owner = runtime.raw_observation_owner
        await owner.converge_raw_id(raw_ids[0])

        def original_state(pinned: PinnedOperationRead) -> tuple[int, int]:
            source_conn = pinned.archive.source_connection
            index_conn = pinned.archive.index_connection
            assert index_conn is not None
            census = source_conn.execute(
                "SELECT COUNT(*) FROM raw_authority_parser_census WHERE raw_id=?", raw_ids
            ).fetchone()[0]
            return census, index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        assert await self.read(original_state) == (1, 1)
        reached.append(raw_ids[0])
        return await original(self, generation_id)

    monkeypatch.setattr(IngestExecution, "materialize", materialize)
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    with _serving(archive_root) as (harness, _server):
        try:
            result = await archive.parse_file(source, source_name="neutral-original-counts")
        finally:
            await harness.close()
            await archive.close()
    assert len(reached) == 1
    assert result.counts["sessions"] == result.changed_counts["sessions"] == 1
    assert result.counts["messages"] == result.changed_counts["messages"] == 2
    assert len(result.processed_ids) == 1
    operation_id, record = _only_request(archive_root)
    audit = AuditRepository.for_archive_root(archive_root)
    with audit.settled_machine_read():
        history = audit.historical_machine_receipt(operation_id)
    assert isinstance(history, IngestHistoricalReceiptV2)
    assert history.summary.parse_projection_known
    assert history.summary.changed_session_count == 1
    assert history.summary.changed_message_count == history.summary.processed_message_count == 2
    generation_id = str(record["artifact_ref"])
    with sqlite3.connect(archive_root / "source.db") as source_db:
        settled = source_db.execute(
            "SELECT disposition, outcome_code, stage FROM source_items WHERE source_generation_id=?",
            (generation_id,),
        ).fetchall()
        census = source_db.execute(
            "SELECT pending, admitted, deliberate, unknown_blocking, sealable "
            "FROM source_item_reconciliation WHERE source_generation_id=?",
            (generation_id,),
        ).fetchone()
    assert settled == [("admitted", "success", "materialization")]
    assert census == (0, 1, 0, 0, 1)


@pytest.mark.timeout(300)
async def test_ingest_settles_source_item_from_owned_cold_candidate_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The receipt reads candidate rows while the active Index stays empty."""
    from polylogue.storage.archive_identity import resolve_active_index_path

    archive_root, source = await _archive(tmp_path)

    def begin_candidate() -> ColdBuildGeneration:
        return ColdBuildGeneration.begin(
            archive_root,
            reason="test-ingest-source-receipt",
            observed=ColdBuildGeneration.observe_source_baseline(
                (WatchSource("fixture", archive_root / "absent-source"),)
            ),
        )

    candidate = await run_archive_fixture_write(archive_root, begin_candidate)
    active_path = resolve_active_index_path(archive_root)
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    materializations: list[RetainedMaterializationResult] = []
    original_materialize = DaemonOperationRuntime.materialize_retained_raw_ids

    async def capture_destination(
        self: DaemonOperationRuntime, *args: Any, **kwargs: Any
    ) -> RetainedMaterializationResult:
        result = await original_materialize(self, *args, **kwargs)
        materializations.append(result)
        return result

    monkeypatch.setattr(DaemonOperationRuntime, "materialize_retained_raw_ids", capture_destination)
    try:
        with _serving(archive_root) as (harness, _server):
            try:
                register_cold_build_generation(candidate)
                await archive.parse_file(source, source_name="neutral-cold-receipt")
            finally:
                await harness.close()

        _operation_id, record = _only_request(archive_root)
        generation_id = str(record["artifact_ref"])
        with sqlite3.connect(archive_root / "source.db") as source_db:
            items = source_db.execute(
                "SELECT disposition, outcome_code, stage FROM source_items WHERE source_generation_id=?",
                (generation_id,),
            ).fetchall()
            census = source_db.execute(
                "SELECT pending, admitted, deliberate, unknown_blocking, sealable "
                "FROM source_item_reconciliation WHERE source_generation_id=?",
                (generation_id,),
            ).fetchone()
        assert items == [("admitted", "success", "materialization")]
        assert census == (0, 1, 0, 0, 1)
        assert materializations and materializations[0].index_destination is not None
        assert materializations[0].index_destination.index_path == Path(candidate.generation.index_path)
        assert materializations[0].outcome.receipts
        assert any(receipt.written_session_ids for receipt in materializations[0].outcome.receipts)
        assert resolve_active_index_path(archive_root) == active_path
        with sqlite3.connect(active_path) as active:
            assert active.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        with ArchiveStore.open_owned_inactive_read(candidate.generation) as reader:
            candidate_sessions = reader._conn.execute("SELECT session_id FROM sessions").fetchall()
        candidate_session_ids = {str(row[0]) for row in candidate_sessions}
        written_session_ids = {
            session_id for receipt in materializations[0].outcome.receipts for session_id in receipt.written_session_ids
        }
        assert written_session_ids and written_session_ids <= candidate_session_ids
    finally:
        await archive.close()
        clear_cold_build_generation()
        await run_archive_fixture_write(archive_root, candidate.discard)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("refused_payload", "validation_mode", "parse_error", "expected_refusal"),
    [
        pytest.param(
            b'{"fixture":"neutral schema rejection"}',
            "strict",
            None,
            ("pending", "validation_rejected", "validation", 0),
            id="strict-schema-refusal",
        ),
        pytest.param(
            b"not json",
            "strict",
            "synthetic decoder failure",
            ("pending", "interrupted", "manifest", None),
            id="strict-decode-failure-is-not-schema-refusal",
        ),
        pytest.param(
            b"not json",
            "advisory",
            "synthetic decoder failure",
            ("pending", "interrupted", "manifest", None),
            id="advisory-decode-failure-is-not-schema-refusal",
        ),
    ],
)
async def test_materialize_keeps_valid_subject_after_original_decode_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refused_payload: bytes,
    validation_mode: str,
    parse_error: str | None,
    expected_refusal: tuple[str, str, str, int | None],
) -> None:
    archive_root, source = await _archive(tmp_path)
    (source.parent / "refused.json").write_bytes(refused_payload)
    original = IngestExecution.materialize
    reached: list[tuple[int, int]] = []
    failures: list[BaseException] = []

    async def materialize(self: IngestExecution, generation_id: str) -> SourceReceiptSpool:
        receipt = await self.receipt(generation_id)
        try:
            raw_ids = receipt.raw_page()
        finally:
            receipt.close()
        assert len(raw_ids) == 2
        try:
            result = await original(self, generation_id)
        except BaseException as failure:
            failures.append(failure)
            raise
        reached.append((len(raw_ids), len(result.session_page())))

        def reject_strictly(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT raw_id FROM raw_sessions WHERE source_path LIKE ? ORDER BY raw_id LIMIT 1",
                ("%/refused.json",),
            ).fetchone()
            assert row is not None
            connection.execute(
                "UPDATE raw_sessions SET validation_status='failed', validation_mode=?, parse_error=?, "
                "validation_error='synthetic typed validation failure', validated_at_ms=1 WHERE raw_id=?",
                (validation_mode, parse_error, row[0]),
            )

        # Inject the durable typed refusal through the same admitted Source
        # writer used by production validation, before finalization consumes it.
        await self.source_write(reject_strictly)
        return result

    monkeypatch.setattr(IngestExecution, "materialize", materialize)
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    with _serving(archive_root) as (harness, _server):
        try:
            try:
                result = await archive.parse_file(source.parent, source_name="neutral-mixed")
            except Exception as failure:
                for original_failure in failures:
                    failure.add_note(f"original materialize failure: {original_failure!r}")
                raise
        finally:
            await harness.close()
            await archive.close()
    assert reached == [(2, 1)]
    assert result.changed_counts["sessions"] == 1
    assert result.changed_counts["messages"] == 2
    assert _session_titles(archive_root) == ["Retained Redrive"]
    operation_id, record = _only_request(archive_root)
    generation_id = str(record["artifact_ref"])
    with sqlite3.connect(archive_root / "source.db") as source_db:
        items = source_db.execute(
            "SELECT disposition, outcome_code, stage, retryable, evidence_ref FROM source_items "
            "WHERE source_generation_id=? ORDER BY logical_coordinate",
            (generation_id,),
        ).fetchall()
        census = source_db.execute(
            "SELECT pending, admitted, deliberate, unknown_blocking, sealable "
            "FROM source_item_reconciliation WHERE source_generation_id=?",
            (generation_id,),
        ).fetchone()
    assert items[0] == ("admitted", "success", "materialization", 0, None)
    assert items[1][:4] == expected_refusal
    if expected_refusal[1] == "validation_rejected":
        assert items[1][4].startswith("raw:")
    else:
        assert items[1][4] is None
    assert census == (1, 1, 0, 0, 0)
    audit = AuditRepository.for_archive_root(archive_root)
    with audit.settled_machine_read():
        history = audit.historical_machine_receipt(operation_id)
        state = machine_request_state(audit, record)
    assert isinstance(history, IngestHistoricalReceiptV2)
    assert history.summary.source_complete is False
    assert state["outcome"] == "degraded"


@pytest.mark.asyncio
async def test_materialize_uses_original_non_json_carrier_and_writer_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources import revision_backfill

    archive_root, source = await _archive(tmp_path)
    captured = source.with_suffix(".capture")
    source.rename(captured)
    original = revision_backfill.prepare_retained_non_json_artifact
    reached: list[str] = []

    def prepare(
        reader: RetainedSessionRead,
        raw_id: str,
        *,
        directory: Path,
        validation_mode: ValidationMode = ValidationMode.ADVISORY,
    ) -> PreparedJsonl:
        descriptor = reader.raw_revision_descriptor(raw_id)
        assert Path(descriptor[2]).suffix == ".capture"
        artifact = original(reader, raw_id, directory=directory, validation_mode=validation_mode)
        verdict = artifact.validation_verdict
        assert verdict is not None
        assert verdict.raw_id == raw_id
        assert verdict.revision_sha256 == descriptor[1]
        assert verdict.mode is validation_mode
        reached.append(raw_id)
        return artifact

    monkeypatch.setattr(revision_backfill, "prepare_retained_non_json_artifact", prepare)
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    with _serving(archive_root) as (harness, _server):
        try:
            result = await archive.parse_file(captured, source_name="neutral-original-carrier")
        finally:
            await harness.close()
            await archive.close()
    assert reached
    assert result.changed_counts["sessions"] == 1
    assert result.changed_counts["messages"] == 2
    assert _session_titles(archive_root) == ["Retained Redrive"]
    with sqlite3.connect(archive_root / "source.db") as source_db:
        assert source_db.execute("SELECT validation_status, validation_mode FROM raw_sessions").fetchall() == [
            ("passed", "advisory")
        ]
