"""Cross-surface parity for the shared MaintenanceOperationEnvelope (#1149).

The envelope is the single typed shape returned by:

* CLI ``polylogue ops maintenance plan --output-format json`` /
  ``polylogue ops maintenance run --output-format json``,
* daemon HTTP ``POST /api/maintenance/plan`` and
  ``POST /api/maintenance/run``,
* MCP ``maintenance_preview`` and ``maintenance_execute``.

These tests pin (1) the envelope itself — its fields, frozen-ness,
``origin``/``mode`` validation — and (2) that all three surfaces emit
byte-equal JSON for the same planner result. ``operation_id`` /
timestamps are derived from each surface's own planner call so the
parity assertion strips them before comparing; every other field is
required to match exactly.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from polylogue.cli.commands.maintenance import maintenance_group
from polylogue.config import Config
from polylogue.maintenance.envelope import (
    EnvelopeMode,
    EnvelopeOrigin,
    MaintenanceOperationEnvelope,
    envelope_from_operation,
    envelope_keys,
)
from polylogue.maintenance.planner import (
    BackfillKind,
    BackfillOperation,
    BackfillStatus,
    BoundedFailureSamples,
    FailureSample,
    MaintenanceScope,
)

# ---------------------------------------------------------------------------
# Envelope shape pinning
# ---------------------------------------------------------------------------


EXPECTED_ENVELOPE_KEYS = frozenset(
    {
        "operation_id",
        "kind",
        "mode",
        "origin",
        "status",
        "targets",
        "scope",
        "progress",
        "started_at",
        "completed_at",
        "error",
        "affected_rows",
        "estimated_time_s",
        "results",
        "reason",
        "resume_cursor",
        "failure_samples",
        "metrics",
    }
)


class TestEnvelopeShape:
    """Pin the typed envelope fields, frozen-ness, and validation rules."""

    def test_envelope_keys_pinned(self) -> None:
        assert envelope_keys() == EXPECTED_ENVELOPE_KEYS

    def test_envelope_is_frozen(self) -> None:
        operation = _example_operation()
        envelope = envelope_from_operation(operation, origin="cli", mode="preview")
        with pytest.raises(ValidationError):
            envelope.operation_id = "mutated"

    def test_envelope_forbids_extra_fields(self) -> None:
        from polylogue.maintenance.envelope import (
            MaintenanceFailureSamplesPayload,
            MaintenanceScopePayload,
        )

        scope = MaintenanceScopePayload(targets=(), filter={})
        failure_samples = MaintenanceFailureSamplesPayload(samples=(), truncated=False)
        with pytest.raises(ValidationError):
            MaintenanceOperationEnvelope(
                operation_id="op",
                kind="derived-rebuild",
                mode="preview",
                origin="cli",
                status="pending",
                targets=(),
                scope=scope,
                progress=0.0,
                started_at=None,
                completed_at=None,
                error=None,
                affected_rows=0,
                estimated_time_s=0.0,
                results=(),
                reason=None,
                resume_cursor=None,
                failure_samples=failure_samples,
                metrics={},
                surprise_field="boom",  # type: ignore[call-arg]
            )

    @pytest.mark.parametrize("origin", ["cli", "daemon", "mcp"])
    def test_envelope_accepts_known_origins(self, origin: EnvelopeOrigin) -> None:
        operation = _example_operation()
        envelope = envelope_from_operation(operation, origin=origin, mode="preview")
        assert envelope.origin == origin

    @pytest.mark.parametrize("mode", ["preview", "execute"])
    def test_envelope_accepts_known_modes(self, mode: EnvelopeMode) -> None:
        operation = _example_operation()
        envelope = envelope_from_operation(operation, origin="cli", mode=mode)
        assert envelope.mode == mode

    def test_envelope_rejects_unknown_origin(self) -> None:
        operation = _example_operation()
        with pytest.raises(ValidationError):
            envelope_from_operation(operation, origin="webhook", mode="preview")  # type: ignore[arg-type]

    def test_envelope_rejects_unknown_mode(self) -> None:
        operation = _example_operation()
        with pytest.raises(ValidationError):
            envelope_from_operation(operation, origin="cli", mode="rollback")  # type: ignore[arg-type]

    def test_failure_samples_round_trip(self) -> None:
        samples = BoundedFailureSamples.from_samples(
            [FailureSample(kind="RuntimeError", locator="x.y", message="boom")]
        )
        operation = _example_operation(failure_samples=samples)
        envelope = envelope_from_operation(operation, origin="daemon", mode="execute")
        payload = envelope.to_dict()
        samples_payload: Any = payload["failure_samples"]
        assert samples_payload["truncated"] is False
        assert len(samples_payload["samples"]) == 1
        assert samples_payload["samples"][0] == {
            "kind": "RuntimeError",
            "locator": "x.y",
            "message": "boom",
        }


# ---------------------------------------------------------------------------
# Cross-surface parity — identical planner result through CLI, daemon, MCP.
# ---------------------------------------------------------------------------


@contextmanager
def _patched_preview(operation: BackfillOperation) -> Iterator[None]:
    """Replace preview_backfill at every surface entry point."""
    with (
        patch(
            "polylogue.cli.commands.maintenance._plan.preview_backfill",
            return_value=operation,
        ),
        patch(
            "polylogue.maintenance.planner.preview_backfill",
            return_value=operation,
        ),
    ):
        yield


def _strip_volatile_fields(envelope: dict[str, Any]) -> dict[str, Any]:
    """Remove fields that each surface derives independently.

    ``operation_id`` and timestamps are minted per call; we are pinning
    *structural* parity, not call-time identity. ``origin`` is the one
    field that legitimately differs between surfaces.
    """
    stripped = dict(envelope)
    stripped.pop("operation_id", None)
    stripped.pop("started_at", None)
    stripped.pop("completed_at", None)
    stripped.pop("origin", None)
    return stripped


class TestCrossSurfaceParity:
    """The same planner result must produce structurally identical envelopes."""

    def test_preview_parity_across_cli_daemon_mcp(self, tmp_path: Path) -> None:
        operation = _example_operation()

        # --- CLI ---
        cli_payload = _capture_cli_preview(operation, tmp_path)

        # --- Daemon ---
        daemon_envelope = envelope_from_operation(operation, origin="daemon", mode="preview")
        daemon_payload = daemon_envelope.to_dict()

        # --- MCP ---
        mcp_envelope = envelope_from_operation(operation, origin="mcp", mode="preview")
        mcp_payload = mcp_envelope.to_dict()

        cli_stripped = _strip_volatile_fields(cli_payload)
        daemon_stripped = _strip_volatile_fields(daemon_payload)
        mcp_stripped = _strip_volatile_fields(mcp_payload)

        assert cli_stripped == daemon_stripped == mcp_stripped
        # Origin tagged per surface.
        assert cli_payload["origin"] == "cli"
        assert daemon_payload["origin"] == "daemon"
        assert mcp_payload["origin"] == "mcp"
        # Mode is preview at every surface.
        assert cli_payload["mode"] == daemon_payload["mode"] == mcp_payload["mode"] == "preview"

    def test_envelope_keys_identical_across_surfaces(self) -> None:
        operation = _example_operation()
        cli_keys = set(envelope_from_operation(operation, origin="cli", mode="preview").to_dict().keys())
        daemon_keys = set(envelope_from_operation(operation, origin="daemon", mode="preview").to_dict().keys())
        mcp_keys = set(envelope_from_operation(operation, origin="mcp", mode="preview").to_dict().keys())
        assert cli_keys == daemon_keys == mcp_keys == EXPECTED_ENVELOPE_KEYS


# ---------------------------------------------------------------------------
# MCP tool registration & invocation
# ---------------------------------------------------------------------------


class TestMaintenanceMCPTools:
    """The MCP tools register and produce the shared envelope."""

    def test_maintenance_tools_registered_on_admin_server(self) -> None:
        from polylogue.mcp.server import build_server

        server = build_server(role="admin")
        tools = server._tool_manager._tools
        assert "maintenance_preview" in tools
        assert "maintenance_execute" in tools

    def test_maintenance_tools_absent_from_read_server(self) -> None:
        from polylogue.mcp.server import build_server

        server = build_server(role="read")
        tools = server._tool_manager._tools
        assert "maintenance_preview" not in tools
        assert "maintenance_execute" not in tools

    def test_maintenance_preview_returns_envelope(self) -> None:
        from polylogue.mcp.server import build_server

        operation = _example_operation()
        server = build_server(role="admin")
        fn = server._tool_manager._tools["maintenance_preview"].fn

        with _patched_preview(operation):
            result = asyncio.run(fn())

        payload = json.loads(result)
        assert set(payload.keys()) == EXPECTED_ENVELOPE_KEYS
        assert payload["origin"] == "mcp"
        assert payload["mode"] == "preview"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _example_operation(
    *,
    failure_samples: BoundedFailureSamples | None = None,
) -> BackfillOperation:
    return BackfillOperation(
        operation_id="op-1",
        kind=BackfillKind.DERIVED_REBUILD,
        targets=("session_insights",),
        status=BackfillStatus.PENDING,
        progress=0.0,
        started_at=None,
        completed_at=None,
        error=None,
        affected_rows=42,
        estimated_time_s=1.5,
        results=[],
        scope=MaintenanceScope(targets=("session_insights",)),
        reason=None,
        resume_cursor=None,
        failure_samples=failure_samples if failure_samples is not None else BoundedFailureSamples(),
        metrics={},
    )


def _capture_cli_preview(operation: BackfillOperation, tmp_path: Path) -> dict[str, Any]:
    archive = tmp_path / "archive"
    render = tmp_path / "render"
    archive.mkdir(parents=True, exist_ok=True)
    render.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()

    config_obj = Config(archive_root=archive, render_root=render, sources=[])

    with (
        _patched_preview(operation),
        patch("polylogue.cli.commands.maintenance._plan.archive_root", return_value=archive),
        patch("polylogue.cli.commands.maintenance._plan.render_root", return_value=render),
    ):
        result = runner.invoke(
            maintenance_group,
            ["plan", "--target", "session_insights", "--output-format", "json"],
            obj=config_obj,
        )

    assert result.exit_code == 0, result.output
    payload: dict[str, Any] = json.loads(result.output)
    return payload
