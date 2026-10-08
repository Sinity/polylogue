"""Resident frozen selection, bounded transport and exact reset authority."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from time import time
from typing import Any, cast

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.operations.daemon_protocol import DAEMON_OPERATION_SPECS, DaemonOperationRequest
from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor
from tests.infra.daemon_operations import cli_daemon_archive, running_daemon_operations
from tests.infra.identity_reset import seed_identity_reset_sources

pytestmark = pytest.mark.uses_real_clock("starts the real UDS listener and coordinator loop")


def _principal() -> MutationPrincipal:
    return MutationPrincipal(
        actor_ref="test:neutral-owner",
        surface="cli",
        role_label="operator",
        capabilities=frozenset(spec.capability for spec in DAEMON_OPERATION_SPECS),
    )


def _runtime_complete(
    stack: Any, request: DaemonOperationRequest, principal: MutationPrincipal, **kwargs: Any
) -> dict[str, Any]:
    response = stack.runtime.call(request, principal, **kwargs)
    sequence = 0
    while response["outcome"] in {"accepted", "running"}:
        waited = stack.runtime.call(
            DaemonOperationRequest(
                operation="operation.await",
                request_id=f"wait-{request.request_id}-{sequence}",
                payload={"request_id": request.request_id, "after_sequence": sequence, "timeout_ms": 30_000},
            ),
            principal,
        )
        state = waited["result"]
        assert isinstance(state, dict), waited
        sequence = state["sequence"]
        response = {**waited, "outcome": state["outcome"], "result": state.get("result", state)}
    return cast(dict[str, Any], response)


def test_source_reset_preview_pages_then_applies_only_frozen_targets(tmp_path: Path) -> None:
    with running_daemon_operations(
        tmp_path / "archive", seed_archive=lambda root: seed_identity_reset_sources(root, 513)
    ) as stack:
        prepared = stack.client.operation_to_completion(
            "mutation.identity-reset.preview",
            {"source_path": "/neutral/selected", "reason": "neutral reset"},
            archive_root=str(stack.archive_root),
        )
        assert prepared is not None and prepared["outcome"] == "completed", prepared
        summary = prepared["result"]
        assert summary["session_count"] == 513 and "session_ids" not in summary
        preview_request_id = summary["reference"]["request_id"]
        past_end = stack.client.operation(
            "session.identity-reset.targets",
            {"preview_request_id": preview_request_id, "offset": 10**100, "page_size": 10**100},
        )
        assert past_end is not None and past_end["outcome"] == "completed", past_end
        assert past_end["result"]["session_ids"] == []
        # Matching arrivals after preview must not be absorbed by confirmation.
        seed_identity_reset_sources(stack.archive_root, 1, start=513)
        seen = []
        offset = 0
        while True:
            response = stack.client.operation(
                "session.identity-reset.targets",
                {"preview_request_id": preview_request_id, "offset": offset, "page_size": 17},
            )
            assert response is not None and response["outcome"] == "completed", response
            page = response["result"]
            assert page["total"] == 513 and len(page["session_ids"]) <= 17
            seen.extend(page["session_ids"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        assert seen == [f"codex-session:neutral-reset-{index:04d}" for index in range(513)]
        authorized = stack.client.operation_to_completion(
            "mutation.identity-reset.authorize",
            {"preview_request_id": preview_request_id, "confirm": True},
            archive_root=str(stack.archive_root),
        )
        assert authorized is not None and authorized["outcome"] == "completed", authorized
        applied = stack.client.operation_to_completion(
            "mutation.identity-reset",
            {"authorization_request_id": authorized["result"]["reference"]["request_id"]},
            archive_root=str(stack.archive_root),
        )
        assert applied is not None and applied["outcome"] == "completed", applied
        assert applied["result"]["affected_count"] == 513
        assert stack.session_exists("codex-session:neutral-reset-0513")
        with sqlite3.connect(stack.archive_root / "user.db") as user:
            assert user.execute("SELECT COUNT(*) FROM assertions WHERE kind='suppression'").fetchone()[0] == 513


@pytest.mark.parametrize("fault", ["wrong-principal", "no-capability", "stale", "cancelled-origin", "forged-reference"])
def test_source_reset_preview_refuses_changed_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    now = [int(time() * 1000)]

    class ClockedExecutor(OperationExecutor):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs, now_ms=lambda: now[0])

    monkeypatch.setattr("polylogue.operations.daemon_mutations.OperationExecutor", ClockedExecutor)
    with running_daemon_operations(
        tmp_path / "archive", seed_archive=lambda root: seed_identity_reset_sources(root, 1)
    ) as stack:
        principal = _principal()
        request = DaemonOperationRequest(
            operation="mutation.identity-reset.preview",
            request_id="neutral-preview",
            payload={"source_path": "/neutral/selected", "reason": "neutral"},
            deadline_ms=300_000,
        )
        prepared = _runtime_complete(stack, request, principal)
        assert prepared["outcome"] == "completed", prepared
        result = prepared["result"]
        assert isinstance(result, dict)
        summary = result
        if fault == "wrong-principal":
            principal = replace(principal, actor_ref="test:neutral-other")
        elif fault == "no-capability":
            principal = replace(principal, capabilities=frozenset())
        elif fault == "cancelled-origin":
            from polylogue.operations.audit import AuditRepository, MachineRequestBinding

            audit = AuditRepository.for_archive_root(stack.archive_root)
            audit.stop_machine_batch(
                MachineRequestBinding(
                    **{
                        key: summary["reference"][key]
                        for key in ("archive_identity", "request_id", "principal_ref", "fingerprint", "operation_name")
                    }
                ),
                "cancelled",
            )
        elif fault == "forged-reference":
            summary["reference"]["request_id"] = "neutral-forged"
        else:
            from polylogue.storage.archive_identity import ArchiveLocation

            with sqlite3.connect(ArchiveLocation.resolve(stack.archive_root).active_index_path) as index:
                index.execute("DELETE FROM sessions")
        authorized = _runtime_complete(
            stack,
            DaemonOperationRequest(
                operation="mutation.identity-reset.authorize",
                request_id="neutral-authorize",
                payload={"preview_request_id": summary["reference"]["request_id"], "confirm": True},
            ),
            principal,
        )
        applied = authorized
        if authorized["outcome"] == "completed":
            applied = _runtime_complete(
                stack,
                DaemonOperationRequest(
                    operation="mutation.identity-reset",
                    request_id="neutral-apply",
                    payload={"authorization_request_id": "neutral-authorize"},
                ),
                principal,
            )
        assert applied["outcome"] != "completed", applied
        with sqlite3.connect(stack.archive_root / "user.db") as user:
            assert user.execute("SELECT COUNT(*) FROM assertions WHERE kind='suppression'").fetchone()[0] == 0
        if fault in {"wrong-principal", "no-capability"}:
            page = stack.runtime.call(
                DaemonOperationRequest(
                    operation="session.identity-reset.targets",
                    request_id="neutral-targets",
                    payload={"preview_request_id": summary["reference"]["request_id"]},
                ),
                principal,
            )
            assert page["outcome"] != "completed", page
            assert page["result"] is None


@pytest.mark.parametrize("mode", ["--dry-run", "--yes"])
def test_source_reset_cli_streams_complete_json_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    with cli_daemon_archive(
        tmp_path / "archive", monkeypatch, seed_archive=lambda root: seed_identity_reset_sources(root, 513)
    ) as stack:
        result = CliRunner().invoke(cli, ["--plain", "ops", "reset", "--source", "/neutral/selected", mode, "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["session_count"] == 513
        assert payload["session_ids"] == [f"codex-session:neutral-reset-{index:04d}" for index in range(513)]
        with sqlite3.connect(stack.archive_root / "user.db") as user:
            assert user.execute("SELECT COUNT(*) FROM assertions WHERE kind='suppression'").fetchone()[0] == (
                513 if mode == "--yes" else 0
            )


def test_identity_reset_target_page_cancellation_precedes_audit_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.archive.query.execution_control import QueryExecutionContext
    from polylogue.operations.daemon_execution import execute_operation
    from polylogue.operations.operation_context_types import OperationContext

    with running_daemon_operations(tmp_path / "archive") as stack:
        control = QueryExecutionContext(call_id="neutral-cancel", query_ref="neutral-page")
        control.cancel()

        def must_not_read(*_args: object, **_kwargs: object) -> None:
            pytest.fail("cancelled target page reached Audit selection")

        monkeypatch.setattr(
            "polylogue.operations.audit.AuditRepository.identity_reset_preview_target_page", must_not_read
        )
        result = execute_operation(
            DaemonOperationRequest(
                operation="session.identity-reset.targets",
                request_id="neutral-cancel",
                payload={"preview_request_id": "neutral-preview"},
            ),
            OperationContext(stack.archive_root, _principal(), "daemon", stack.runtime, read_control=control),
        )
        assert result.outcome == "cancelled"
        assert result.result is None


@pytest.mark.parametrize(
    "payload",
    [
        {"session": "neutral", "source_path": "/neutral", "reason": "neutral"},
        {"reason": "neutral"},
        {"session_ids": ["neutral"], "confirm": True},
    ],
)
def test_identity_reset_rejects_ambiguous_or_retired_upload_shape(payload: dict[str, object]) -> None:
    from pydantic import ValidationError

    from polylogue.operations.daemon_protocol import IdentityResetPreviewRequest, IdentityResetRequest

    model = IdentityResetRequest if "session_ids" in payload else IdentityResetPreviewRequest
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_reset_existing_id_probe_pages_actual_sqlite_binding_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations import mutation_actuators
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    with running_daemon_operations(
        tmp_path / "archive", seed_archive=lambda root: seed_identity_reset_sources(root, 13)
    ) as stack:
        original_open = open_readonly_connection

        def limited_open(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            connection = original_open(*args, **kwargs)
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 7)
            return connection

        monkeypatch.setattr(mutation_actuators, "open_readonly_connection", limited_open)
        ids = tuple(f"codex-session:neutral-reset-{index:04d}" for index in range(13))
        assert mutation_actuators._resolve_existing_session_ids(stack.archive_root, ids) == ids


def test_reset_preview_has_durable_custody_without_implicit_deadline(tmp_path: Path) -> None:
    from time import monotonic

    from polylogue.core.compute import CancellationHandle
    from polylogue.operations.daemon_protocol import daemon_operation_spec

    spec = daemon_operation_spec("mutation.identity-reset.preview")
    assert spec is not None and spec.deadline_s is None and spec.accepted_reference
    with running_daemon_operations(
        tmp_path / "archive", seed_archive=lambda root: seed_identity_reset_sources(root, 2)
    ) as stack:
        principal = _principal()
        request = DaemonOperationRequest(
            operation="mutation.identity-reset.preview",
            request_id="neutral-durable-preview",
            payload={"source_path": "/neutral/selected", "reason": "neutral" * 1000},
        )
        # A valid selection started beyond the predecessor's five-minute budget
        # must still complete and retain its exact preview under the request ID.
        prepared = _runtime_complete(stack, request, principal, started_at=monotonic() - 301)
        assert prepared["outcome"] == "completed", prepared
        prepared_result = prepared["result"]
        assert isinstance(prepared_result, dict)
        summary = prepared_result
        from polylogue.operations.audit import AuditRepository

        audit = AuditRepository.for_archive_root(stack.archive_root)
        with sqlite3.connect(stack.archive_root / "audit.db") as connection:
            preview_ref = connection.execute("SELECT preview_id FROM operation_previews").fetchone()[0]
        stored = audit.preview_for_principal(preview_ref, principal)
        assert stored.plan.context["reason"] == "neutral" * 1000
        seed_identity_reset_sources(stack.archive_root, 1, start=2)
        cancelled = CancellationHandle()
        cancelled.cancel()
        replayed = stack.runtime.call(request, principal, client_disconnect=cancelled)
        assert replayed["outcome"] == "completed", replayed
        replayed_result = replayed["result"]
        assert isinstance(replayed_result, dict)
        assert replayed_result == summary
        assert summary["session_count"] == 2
        authorized = _runtime_complete(
            stack,
            DaemonOperationRequest(
                operation="mutation.identity-reset.authorize",
                request_id="neutral-durable-authorize",
                payload={"preview_request_id": summary["reference"]["request_id"], "confirm": True},
            ),
            principal,
        )
        assert authorized["outcome"] == "completed", authorized
        apply_request = DaemonOperationRequest(
            operation="mutation.identity-reset",
            request_id="neutral-durable-apply",
            payload={"authorization_request_id": "neutral-durable-authorize"},
        )
        accepted = stack.runtime.call(apply_request, principal, started_at=monotonic() - 301)
        assert accepted["outcome"] in {"accepted", "completed"}, accepted
        applied = accepted
        sequence = 0
        while applied["outcome"] in {"accepted", "running"}:
            waited = stack.runtime.call(
                DaemonOperationRequest(
                    operation="operation.await",
                    request_id=f"neutral-wait-{sequence}",
                    payload={
                        "request_id": str(apply_request.request_id),
                        "after_sequence": sequence,
                        "timeout_ms": 30_000,
                    },
                ),
                principal,
            )
            state = waited["result"]
            assert isinstance(state, dict), waited
            sequence = int(state["sequence"])
            applied = {"outcome": state["outcome"], "result": state.get("result", state)}
        assert applied["outcome"] == "completed", applied
        repeated = stack.runtime.call(apply_request, principal, client_disconnect=cancelled)
        assert repeated["outcome"] == "completed", repeated
        apply_result = applied["result"]
        repeated_result = repeated["result"]
        assert isinstance(apply_result, dict) and isinstance(repeated_result, dict)
        assert (
            apply_result["result"]
            == repeated_result["result"]
            == {
                "count_scope": "completing-apply",
                "suppressed_count": 2,
                "deleted_archive_rows": 2,
                "tombstoned_without_index_row_count": 0,
            }
        )
        assert stack.session_exists("codex-session:neutral-reset-0002")
        other = stack.runtime.call(request, replace(principal, actor_ref="test:neutral-other"))
        assert other["outcome"] == "rejected", other
        pending = DaemonOperationRequest(
            operation=request.operation,
            request_id="neutral-cancelled-preview",
            payload=request.payload,
        )
        refused = stack.runtime.call(pending, principal, client_disconnect=cancelled)
        assert refused["outcome"] in {"cancelled", "disconnected-before-acceptance"}, refused
        with sqlite3.connect(stack.archive_root / "audit.db") as connection:
            assert connection.execute("SELECT COUNT(*) FROM operation_previews").fetchone()[0] == 1


def test_reset_overlap_probe_pages_actual_sqlite_binding_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    from polylogue.operations.audit import AuditRepository

    with running_daemon_operations(
        tmp_path / "archive", seed_archive=lambda root: seed_identity_reset_sources(root, 13)
    ) as stack:
        audit = AuditRepository.for_archive_root(stack.archive_root)
        original = audit._connection

        @contextmanager
        def limited() -> Iterator[sqlite3.Connection]:
            with original() as conn:
                conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 7)
                yield conn

        monkeypatch.setattr(audit, "_connection", limited)
        targets = tuple(f"session:codex-session:neutral-reset-{index:04d}" for index in range(13))
        assert audit.nonterminal_operations_overlapping(targets) == ()


def test_reset_spills_more_than_one_authority_page_and_survives_delayed_confirmation_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations import daemon_mutations, mutation_transaction
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.mutation_actuators import IdentityResetActuator
    from polylogue.operations.mutation_transaction import TokenExpiredError

    # Small plan pages discriminate the >40-part transport path with a neutral
    # fixture, without making test cost proportional to production's page size.
    monkeypatch.setattr(mutation_transaction, "MUTATION_PLAN_PAGE_SIZE", 3)
    monkeypatch.setattr(daemon_mutations, "MUTATION_PLAN_PAGE_SIZE", 3)
    now = [int(time() * 1000)]

    class ClockedExecutor(OperationExecutor):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs, now_ms=lambda: now[0])

    monkeypatch.setattr(daemon_mutations, "OperationExecutor", ClockedExecutor)
    monkeypatch.setattr(daemon_mutations, "time", lambda: now[0] / 1000)
    monkeypatch.setattr("polylogue.operations.audit.time.time", lambda: now[0] / 1000)
    original_prepare = IdentityResetActuator.prepare
    original_batch = AuditRepository.create_preview_batch
    pages: list[int] = []

    def bounded_prepare(self: IdentityResetActuator, args: Any) -> Any:
        assert len(args.session_ids) <= 3
        return original_prepare(self, args)

    def bounded_batch(self: AuditRepository, plans: Any, principal: MutationPrincipal) -> Any:
        assert len(plans) <= 40
        pages.append(len(plans))
        return original_batch(self, plans, principal)

    monkeypatch.setattr(IdentityResetActuator, "prepare", bounded_prepare)
    monkeypatch.setattr(AuditRepository, "create_preview_batch", bounded_batch)
    root = tmp_path / "archive"
    with running_daemon_operations(
        root, seed_archive=lambda archive: seed_identity_reset_sources(archive, 125)
    ) as stack:
        prepared = stack.client.operation_to_completion(
            "mutation.identity-reset.preview",
            {"source_path": "/neutral/selected", "reason": "neutral"},
            archive_root=str(root),
            request_id="neutral-spilled-preview",
        )
        assert prepared is not None and prepared["outcome"] == "completed", prepared
        assert pages == [40, 2]
        assert prepared["result"]["reference"]["part_count"] == 42
        with sqlite3.connect(root / "user.db") as user:
            assert user.execute("SELECT count(*) FROM assertions").fetchone()[0] == 0
        with sqlite3.connect(root / "audit.db") as connection:
            preview_ref = connection.execute("SELECT preview_id FROM operation_previews LIMIT 1").fetchone()[0]
        principal = MutationPrincipal(
            actor_ref=prepared["result"]["reference"]["principal_ref"],
            surface="cli",
            capabilities=frozenset(spec.capability for spec in DAEMON_OPERATION_SPECS),
        )
        preview = AuditRepository.for_archive_root(root).preview_for_principal(preview_ref, principal)
        now[0] = preview.plan.expires_at_ms + 120_000
        from polylogue.operations.bindings import runtime_operation_binding

        with pytest.raises(TokenExpiredError):
            ClockedExecutor().authorize_bound(runtime_operation_binding(IdentityResetActuator()), preview, principal)
    # The accepted preview survives process restart and a human delay longer
    # than the old lease; later matching arrivals never join its authority.
    with running_daemon_operations(root) as stack:
        seed_identity_reset_sources(root, 1, start=125)
        authorized = stack.client.operation_to_completion(
            "mutation.identity-reset.authorize",
            {"preview_request_id": "neutral-spilled-preview", "confirm": True},
            archive_root=str(root),
            request_id="neutral-spilled-authorize",
        )
        assert authorized is not None and authorized["outcome"] == "completed", authorized
        with sqlite3.connect(root / "audit.db") as connection:
            assert (
                connection.execute(
                    "SELECT count(*) FROM operation_authorizations WHERE expires_at_ms=issued_at_ms"
                ).fetchone()[0]
                == 42
            )
        applied = stack.client.operation_to_completion(
            "mutation.identity-reset",
            {"authorization_request_id": "neutral-spilled-authorize"},
            archive_root=str(root),
            request_id="neutral-spilled-apply",
        )
        assert applied is not None and applied["outcome"] == "completed", applied
        assert applied["result"]["result"]["suppressed_count"] == 125
        assert applied["result"]["result"]["deleted_archive_rows"] == 125
        assert stack.session_exists("codex-session:neutral-reset-0125")
        repeated = stack.client.operation_to_completion(
            "mutation.identity-reset",
            {"authorization_request_id": "neutral-spilled-authorize"},
            archive_root=str(root),
            request_id="neutral-spilled-apply",
        )
        assert repeated is not None and repeated["result"]["result"] == applied["result"]["result"]


def test_reset_cancelled_after_first_part_reports_only_the_committed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.operations import daemon_mutations, mutation_transaction
    from polylogue.operations.mutation_actuators import IdentityResetActuator

    monkeypatch.setattr(mutation_transaction, "MUTATION_PLAN_PAGE_SIZE", 3)
    monkeypatch.setattr(daemon_mutations, "MUTATION_PLAN_PAGE_SIZE", 3)
    original_apply = IdentityResetActuator.apply
    original_stop = DaemonOperationRuntime.stop_reason
    committed = [False]

    def apply(self: IdentityResetActuator, plan: Any, args: Any) -> Any:
        receipt = original_apply(self, plan, args)
        committed[0] = True
        return receipt

    def stop(self: DaemonOperationRuntime, request: DaemonOperationRequest) -> Any:
        if request.operation == "mutation.identity-reset" and committed[0]:
            return "cancelled"
        return original_stop(self, request)

    monkeypatch.setattr(IdentityResetActuator, "apply", apply)
    monkeypatch.setattr(DaemonOperationRuntime, "stop_reason", stop)
    with running_daemon_operations(
        tmp_path / "archive", seed_archive=lambda root: seed_identity_reset_sources(root, 8)
    ) as stack:
        prepared = stack.client.operation_to_completion(
            "mutation.identity-reset.preview",
            {"source_path": "/neutral/selected", "reason": "neutral"},
            archive_root=str(stack.archive_root),
            request_id="neutral-prefix-preview",
        )
        assert prepared is not None and prepared["outcome"] == "completed", prepared
        authorized = stack.client.operation_to_completion(
            "mutation.identity-reset.authorize",
            {"preview_request_id": "neutral-prefix-preview", "confirm": True},
            archive_root=str(stack.archive_root),
            request_id="neutral-prefix-authorize",
        )
        assert authorized is not None and authorized["outcome"] == "completed", authorized
        applied = stack.client.operation_to_completion(
            "mutation.identity-reset",
            {"authorization_request_id": "neutral-prefix-authorize"},
            archive_root=str(stack.archive_root),
            request_id="neutral-prefix-apply",
        )
        assert applied is not None and applied["outcome"] == "cancelled", applied
        status = stack.client.operation(
            "operation.status", {"request_id": "neutral-prefix-apply"}, archive_root=str(stack.archive_root)
        )
        assert status is not None and isinstance(status["result"], dict), status
        state = status["result"]
        assert state["outcome"] == "cancelled" and state["not_attempted_count"] == 2
        assert state["result"]["result"]["suppressed_count"] == 3
        with sqlite3.connect(stack.archive_root / "user.db") as connection:
            assert connection.execute("SELECT count(*) FROM assertions WHERE kind='suppression'").fetchone()[0] == 3
        assert stack.session_exists("codex-session:neutral-reset-0003")


def test_reset_authorization_replays_its_outer_confirmed_custody_without_child_machine_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.operations.audit import AuditRepository, MachineRequestBinding
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.operations.mutation_actuators import IdentityResetActuator
    from polylogue.storage.sqlite.audit_continuity import AuditContinuityCoordinator, AuditMutation

    root = tmp_path / "archive"
    principal = _principal()
    with running_daemon_operations(root, seed_archive=lambda archive: seed_identity_reset_sources(archive, 2)) as stack:
        prepared = _runtime_complete(
            stack,
            DaemonOperationRequest(
                operation="mutation.identity-reset.preview",
                request_id="neutral-replay-preview",
                payload={"source_path": "/neutral/selected", "reason": "neutral"},
            ),
            principal,
        )
        assert prepared["outcome"] == "completed", prepared
        reference = prepared["result"]["reference"]
    audit = AuditRepository.for_archive_root(root)
    request = DaemonOperationRequest(
        operation="mutation.identity-reset.authorize",
        request_id="neutral-replay-authorize",
        payload={"preview_request_id": "neutral-replay-preview", "confirm": True},
    )
    binding = MachineRequestBinding(
        reference["archive_identity"],
        str(request.request_id),
        principal.actor_ref,
        request.fingerprint,
        request.operation,
    )
    preview = audit.preview_for_principal(reference["artifact_ref"], principal)
    original_phase = AuditContinuityCoordinator._phase

    def crash(self: AuditContinuityCoordinator, phase: str, mutation: AuditMutation) -> None:
        if mutation.kind == "issue_authorization_batch" and phase == "after_source_prepare":
            raise RuntimeError("neutral source-WAL crash")
        original_phase(self, phase, mutation)

    with audit.bind_identity_reset_request(binding, request, transition="issue_authorization_batch", page=(0, True)):
        proof = audit.require_accepted_identity_reset_custody(
            preview, principal, ordinal=0, issued_at_ms=int(time() * 1000)
        )
        authorization = OperationExecutor().authorize_bound(
            runtime_operation_binding(IdentityResetActuator()),
            preview,
            principal,
            confirmation_strength="bound_token",
            identity_reset_custody=proof,
        )
        with monkeypatch.context() as patch:
            patch.setattr(AuditContinuityCoordinator, "_phase", crash)
            with pytest.raises(RuntimeError, match="source-WAL crash"):
                audit.issue_authorization_batch((preview,), principal, (authorization,))
    with sqlite3.connect(root / "source.db") as connection:
        command = json.loads(
            connection.execute("SELECT pending_payload_json FROM audit_continuity_control").fetchone()[0]
        )["command"]
    assert command["payload"]["identity_reset_intent"]["request_intent"]["payload"]["confirm"] is True
    assert "machine_request" not in command["payload"]["commands"][0]["payload"]
    restarted = AuditRepository.for_archive_root(root)
    restarted.reconcile_continuity()
    with sqlite3.connect(root / "audit.db") as connection:
        assert connection.execute("SELECT count(*) FROM machine_requests").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT count(*) FROM operation_authorizations WHERE expires_at_ms=issued_at_ms"
            ).fetchone()[0]
            == 1
        )
    with running_daemon_operations(root) as stack:
        applied = _runtime_complete(
            stack,
            DaemonOperationRequest(
                operation="mutation.identity-reset",
                request_id="neutral-replay-apply",
                payload={"authorization_request_id": "neutral-replay-authorize"},
            ),
            principal,
        )
        assert applied["outcome"] == "completed", applied
        assert applied["result"]["result"]["suppressed_count"] == 2


@pytest.mark.parametrize("fault", ["wrong-part", "wrong-batch", "unconfirmed", "wrong-operation"])
def test_reset_custody_rejects_detached_confirmation_and_part_provenance(tmp_path: Path, fault: str) -> None:
    from polylogue.operations.audit import AuditRepository, MachineRequestBinding
    from polylogue.operations.mutation_transaction import AuthorizationMismatchError

    principal = _principal()
    root = tmp_path / "archive"
    with running_daemon_operations(
        root, seed_archive=lambda archive: seed_identity_reset_sources(archive, 257)
    ) as stack:
        prepared = _runtime_complete(
            stack,
            DaemonOperationRequest(
                operation="mutation.identity-reset.preview",
                request_id="neutral-proof-preview",
                payload={"source_path": "/neutral/selected", "reason": "neutral"},
            ),
            principal,
        )
        assert prepared["outcome"] == "completed", prepared
        reference = prepared["result"]["reference"]
        other = _runtime_complete(
            stack,
            DaemonOperationRequest(
                operation="mutation.identity-reset.preview",
                request_id="neutral-other-preview",
                payload={"source_path": "/neutral/selected", "reason": "neutral"},
            ),
            principal,
        )
        assert other["outcome"] == "completed", other
    audit = AuditRepository.for_archive_root(root)
    preview = audit.preview_for_principal(reference["artifact_ref"], principal)
    request = DaemonOperationRequest(
        operation="mutation.session.delete.authorize"
        if fault == "wrong-operation"
        else "mutation.identity-reset.authorize",
        request_id="neutral-proof-authorize",
        payload={
            "preview_request_id": "neutral-other-preview" if fault == "wrong-batch" else "neutral-proof-preview",
            "confirm": fault != "unconfirmed",
        },
    )
    binding = MachineRequestBinding(
        reference["archive_identity"],
        str(request.request_id),
        principal.actor_ref,
        request.fingerprint,
        request.operation,
    )
    with pytest.raises(AuthorizationMismatchError):
        with audit.bind_identity_reset_request(
            binding, request, transition="issue_authorization_batch", page=(0, True)
        ):
            audit.require_accepted_identity_reset_custody(
                preview, principal, ordinal=1 if fault == "wrong-part" else 0, issued_at_ms=int(time() * 1000)
            )
    with sqlite3.connect(root / "audit.db") as connection:
        assert connection.execute("SELECT count(*) FROM operation_authorizations").fetchone()[0] == 0


def test_empty_reset_selection_returns_an_authenticated_terminal_target_page(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive") as stack:
        prepared = stack.client.operation_to_completion(
            "mutation.identity-reset.preview",
            {"source_path": "/neutral/selected", "reason": "neutral"},
            archive_root=str(stack.archive_root),
            request_id="neutral-empty-preview",
        )
        assert prepared is not None and prepared["outcome"] == "completed", prepared
        assert prepared["result"]["session_count"] == 0
        assert prepared["result"]["reference"]["part_count"] == 1
        response = stack.client.operation(
            "session.identity-reset.targets",
            {"preview_request_id": "neutral-empty-preview"},
            archive_root=str(stack.archive_root),
        )
        assert response is not None and response["outcome"] == "completed", response
        assert response["result"]["session_ids"] == []
        assert response["result"]["total"] == 0 and response["result"]["next_offset"] is None
        assert response["result"]["outcome"]["state"] == "empty"
