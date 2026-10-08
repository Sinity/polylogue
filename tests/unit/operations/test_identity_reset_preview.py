"""Resident frozen selection, bounded transport and exact reset authority."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from time import time
from typing import Any

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


def test_source_reset_preview_pages_then_applies_only_frozen_targets(tmp_path: Path) -> None:
    with running_daemon_operations(
        tmp_path / "archive", seed_archive=lambda root: seed_identity_reset_sources(root, 513)
    ) as stack:
        prepared = stack.client.operation(
            "mutation.identity-reset.preview", {"source_path": "/neutral/selected", "reason": "neutral reset"}
        )
        assert prepared is not None and prepared["outcome"] == "completed", prepared
        summary = prepared["result"]["result"]
        assert summary["session_count"] == 513 and "session_ids" not in summary
        preview_ref = summary["preview_ref"]
        past_end = stack.client.operation(
            "session.identity-reset.targets", {"preview_ref": preview_ref, "offset": 10**100, "page_size": 10**100}
        )
        assert past_end is not None and past_end["outcome"] == "completed", past_end
        assert past_end["result"]["session_ids"] == []
        # Matching arrivals after preview must not be absorbed by confirmation.
        seed_identity_reset_sources(stack.archive_root, 1, start=513)
        seen = []
        offset = 0
        while True:
            response = stack.client.operation(
                "session.identity-reset.targets", {"preview_ref": preview_ref, "offset": offset, "page_size": 17}
            )
            assert response is not None and response["outcome"] == "completed", response
            page = response["result"]
            assert page["total"] == 513 and len(page["session_ids"]) <= 17
            seen.extend(page["session_ids"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        assert seen == [f"codex-session:neutral-reset-{index:04d}" for index in range(513)]
        applied = stack.client.operation_to_completion(
            "mutation.identity-reset",
            {"preview_ref": preview_ref, "confirm": True},
            archive_root=str(stack.archive_root),
        )
        assert applied is not None and applied["outcome"] == "completed", applied
        assert applied["result"]["affected_count"] == 513
        assert stack.session_exists("codex-session:neutral-reset-0513")
        with sqlite3.connect(stack.archive_root / "user.db") as user:
            assert user.execute("SELECT COUNT(*) FROM assertions WHERE kind='suppression'").fetchone()[0] == 513


@pytest.mark.parametrize("fault", ["wrong-principal", "no-capability", "expired", "stale"])
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
        prepared = stack.runtime.call(request, principal)
        assert prepared["outcome"] == "completed", prepared
        result = prepared["result"]
        assert isinstance(result, dict)
        summary = result["result"]
        if fault == "wrong-principal":
            principal = replace(principal, actor_ref="test:neutral-other")
        elif fault == "no-capability":
            principal = replace(principal, capabilities=frozenset())
        elif fault == "expired":
            now[0] = summary["expires_at_ms"] + 1
        else:
            from polylogue.storage.archive_identity import ArchiveLocation

            with sqlite3.connect(ArchiveLocation.resolve(stack.archive_root).active_index_path) as index:
                index.execute("DELETE FROM sessions")
        applied = stack.runtime.call(
            DaemonOperationRequest(
                operation="mutation.identity-reset",
                request_id="neutral-apply",
                payload={"preview_ref": summary["preview_ref"], "confirm": True},
                deadline_ms=300_000,
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
                    payload={"preview_ref": summary["preview_ref"]},
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
                payload={"preview_ref": "preview:neutral"},
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
        prepared = stack.runtime.call(request, principal, started_at=monotonic() - 301)
        assert prepared["outcome"] == "completed", prepared
        prepared_result = prepared["result"]
        assert isinstance(prepared_result, dict)
        summary = prepared_result["result"]
        from polylogue.operations.audit import AuditRepository

        stored = AuditRepository.for_archive_root(stack.archive_root).preview_for_principal(
            str(summary["preview_ref"]), principal
        )
        assert stored.plan.context["reason"] == "neutral" * 1000
        seed_identity_reset_sources(stack.archive_root, 1, start=2)
        cancelled = CancellationHandle()
        cancelled.cancel()
        replayed = stack.runtime.call(request, principal, client_disconnect=cancelled)
        assert replayed["outcome"] == "completed", replayed
        replayed_result = replayed["result"]
        assert isinstance(replayed_result, dict)
        assert replayed_result["result"] == summary
        assert summary["session_count"] == 2
        apply_request = DaemonOperationRequest(
            operation="mutation.identity-reset",
            request_id="neutral-durable-apply",
            payload={"preview_ref": summary["preview_ref"], "confirm": True},
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
        with sqlite3.connect(stack.archive_root / "audit.db") as audit:
            assert audit.execute("SELECT COUNT(*) FROM operation_previews").fetchone()[0] == 1


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
