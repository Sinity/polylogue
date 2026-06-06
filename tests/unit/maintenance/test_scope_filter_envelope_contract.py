"""Cross-surface envelope contract for the typed maintenance scope filter (#1303).

One typed :class:`MaintenanceScopeFilter` must round-trip identically
through every surface — CLI ``polylogue maintenance plan`` flags, the
daemon ``POST /api/maintenance/plan`` JSON body, and MCP
``maintenance_preview`` typed parameters — and every surface must echo
it back as the same envelope JSON. The test pins:

* CLI argv → planner ``scope_filter`` (every dimension)
* HTTP POST body → planner ``scope_filter`` (every dimension; nested
  ``{"scope":{"filter":{...}}}`` envelope and flat top-level form both
  accepted)
* MCP typed args → planner ``scope_filter`` (every dimension)
* All three surfaces serialize the same operation back to a byte-equal
  ``scope.filter`` envelope payload.

This is the cross-surface coherence assertion that prevents #1196 from
drifting if any one surface gains a new dimension without the other
two.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.cli.commands.maintenance import maintenance_group
from polylogue.config import Config
from polylogue.maintenance.envelope import envelope_from_operation
from polylogue.maintenance.planner import (
    BackfillKind,
    BackfillOperation,
    BackfillStatus,
    MaintenanceScope,
)
from polylogue.maintenance.scope import MaintenanceScopeFilter

# ---------------------------------------------------------------------------
# Canonical full filter — covers every dimension in one payload
# ---------------------------------------------------------------------------


_SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)
_UNTIL = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _canonical_filter() -> MaintenanceScopeFilter:
    return MaintenanceScopeFilter(
        session_ids=("c1", "c2"),
        provider="claude-code",
        source_family="claude-code-session",
        source_root=Path("/data/claude"),
        raw_artifact_id="raw-1",
        time_range=(_SINCE, _UNTIL),
        failure_kind="ValidationError",
        parser_version="v3",
    )


def _operation_for(scope_filter: MaintenanceScopeFilter) -> BackfillOperation:
    return BackfillOperation(
        operation_id="op-fixed",
        kind=BackfillKind.DERIVED_REBUILD,
        targets=("session_insights",),
        status=BackfillStatus.PENDING,
        scope=MaintenanceScope(targets=("session_insights",), filter=scope_filter),
    )


def _make_config(tmp_path: Path) -> Config:
    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    return Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[],
        db_path=tmp_path / "archive.db",
    )


# ---------------------------------------------------------------------------
# CLI surface — argv to planner call to envelope
# ---------------------------------------------------------------------------


def _invoke_cli_plan(operation: BackfillOperation, tmp_path: Path) -> tuple[dict[str, Any], MaintenanceScopeFilter]:
    """Run ``polylogue maintenance plan`` and return (envelope JSON, captured filter)."""
    captured: dict[str, MaintenanceScopeFilter] = {}

    def _capture(_config: Any, *, targets: tuple[str, ...], scope_filter: MaintenanceScopeFilter) -> BackfillOperation:
        captured["filter"] = scope_filter
        return operation.__class__(
            operation_id=operation.operation_id,
            kind=operation.kind,
            targets=operation.targets,
            status=operation.status,
            scope=MaintenanceScope(targets=targets, filter=scope_filter),
        )

    runner = CliRunner()
    archive = tmp_path / "archive"
    render = tmp_path / "render"
    archive.mkdir(parents=True, exist_ok=True)
    render.mkdir(parents=True, exist_ok=True)
    config_obj = Config(archive_root=archive, render_root=render, sources=[])

    with (
        patch("polylogue.cli.commands.maintenance.preview_backfill", side_effect=_capture),
        patch("polylogue.cli.commands.maintenance.archive_root", return_value=archive),
        patch("polylogue.cli.commands.maintenance.render_root", return_value=render),
    ):
        result = runner.invoke(
            maintenance_group,
            [
                "plan",
                "--target",
                "session_insights",
                "--session-id",
                "c1",
                "--session-id",
                "c2",
                "--origin",
                "claude-code-session",
                "--source-family",
                "claude-code-session",
                "--source-root",
                "/data/claude",
                "--raw-artifact",
                "raw-1",
                "--since",
                "2026-01-01T00:00:00Z",
                "--until",
                "2026-02-01T00:00:00Z",
                "--failure-kind",
                "ValidationError",
                "--parser-version",
                "v3",
                "--output-format",
                "json",
            ],
            obj=config_obj,
        )

    assert result.exit_code == 0, result.output
    payload: dict[str, Any] = json.loads(result.output)
    return payload, captured["filter"]


# ---------------------------------------------------------------------------
# HTTP surface — POST body to planner call to envelope
# ---------------------------------------------------------------------------


def _invoke_daemon_plan(body: dict[str, Any]) -> tuple[dict[str, Any], MaintenanceScopeFilter]:
    """Drive ``_handle_maintenance_plan`` with ``body`` and return (envelope, filter)."""
    from tests.unit.daemon.test_maintenance_endpoints import MockHeaders, _make_handler

    captured: dict[str, MaintenanceScopeFilter] = {}

    def _capture(_config: Any, *, targets: tuple[str, ...], scope_filter: MaintenanceScopeFilter) -> BackfillOperation:
        captured["filter"] = scope_filter
        return _operation_for(scope_filter)

    body_raw = json.dumps(body).encode("utf-8")
    handler = _make_handler("/api/maintenance/plan")
    cast(MockHeaders, handler.headers)._headers["Content-Length"] = str(len(body_raw))
    handler.rfile = BytesIO(body_raw)

    sent: dict[str, Any] = {}

    def _capture_send(status: Any, payload: Any) -> None:
        sent["status"] = status
        sent["payload"] = payload

    with (
        patch("polylogue.maintenance.planner.preview_backfill", side_effect=_capture),
        patch.object(handler, "_send_json", side_effect=_capture_send),
    ):
        handler._handle_maintenance_plan()

    return cast(dict[str, Any], sent["payload"]), captured["filter"]


# ---------------------------------------------------------------------------
# MCP surface — typed args to planner call to envelope
# ---------------------------------------------------------------------------


def _invoke_mcp_preview() -> tuple[dict[str, Any], MaintenanceScopeFilter]:
    """Drive ``maintenance_preview`` with the canonical filter args."""
    from polylogue.mcp.server import build_server

    captured: dict[str, MaintenanceScopeFilter] = {}

    def _capture(_config: Any, *, targets: tuple[str, ...], scope_filter: MaintenanceScopeFilter) -> BackfillOperation:
        captured["filter"] = scope_filter
        return _operation_for(scope_filter)

    server = build_server(role="admin")
    fn = server._tool_manager._tools["maintenance_preview"].fn

    with patch("polylogue.maintenance.planner.preview_backfill", side_effect=_capture):
        result = asyncio.run(
            fn(
                targets=["session_insights"],
                session_ids=["c1", "c2"],
                origin="claude-code-session",
                source_family="claude-code-session",
                source_root="/data/claude",
                raw_artifact_id="raw-1",
                since="2026-01-01T00:00:00Z",
                until="2026-02-01T00:00:00Z",
                failure_kind="ValidationError",
                parser_version="v3",
            )
        )

    payload: dict[str, Any] = json.loads(result)
    return payload, captured["filter"]


# ---------------------------------------------------------------------------
# Single-dimension parity tests — one assertion per scope dimension
# ---------------------------------------------------------------------------


class TestCLIArgsBuildScopeFilter:
    """Each CLI flag round-trips into the matching scope-filter field."""

    def test_every_dimension_reaches_the_planner(self, tmp_path: Path) -> None:
        _, captured = _invoke_cli_plan(_operation_for(_canonical_filter()), tmp_path)
        assert captured == _canonical_filter()


class TestDaemonBodyBuildsScopeFilter:
    """HTTP body fields round-trip into the matching scope-filter field."""

    def test_flat_body_every_dimension_reaches_the_planner(self) -> None:
        body = {
            "targets": ["session_insights"],
            "session_ids": ["c1", "c2"],
            "provider": "claude-code",
            "source_family": "claude-code-session",
            "source_root": "/data/claude",
            "raw_artifact_id": "raw-1",
            "time_range": [
                "2026-01-01T00:00:00+00:00",
                "2026-02-01T00:00:00+00:00",
            ],
            "failure_kind": "ValidationError",
            "parser_version": "v3",
        }
        _, captured = _invoke_daemon_plan(body)
        assert captured == _canonical_filter()

    def test_nested_scope_filter_body_is_accepted(self) -> None:
        """``{"scope":{"filter":{...}}}`` body shape is honored too.

        The daemon HTTP handler accepts both the flat top-level shape and
        the nested envelope shape, so a client that echoes a previous
        envelope's ``scope.filter`` payload still drives the planner
        with the same filter.
        """
        body = {
            "targets": ["session_insights"],
            "scope": {"filter": _canonical_filter().to_dict()},
        }
        _, captured = _invoke_daemon_plan(body)
        assert captured == _canonical_filter()


class TestMCPArgsBuildScopeFilter:
    """MCP typed parameters round-trip into the matching scope-filter field."""

    def test_every_dimension_reaches_the_planner(self) -> None:
        _, captured = _invoke_mcp_preview()
        assert captured == _canonical_filter()


# ---------------------------------------------------------------------------
# Cross-surface envelope parity
# ---------------------------------------------------------------------------


class TestEnvelopeFilterParity:
    """All three surfaces emit the same ``scope.filter`` envelope JSON."""

    def test_envelope_filters_match_across_surfaces(self, tmp_path: Path) -> None:
        # Same operation, three different surfaces. The envelope is the
        # contract — every surface must produce a byte-equal ``scope.filter``.
        operation = _operation_for(_canonical_filter())

        cli_payload, cli_filter = _invoke_cli_plan(operation, tmp_path)
        daemon_payload, daemon_filter = _invoke_daemon_plan(
            {
                "targets": ["session_insights"],
                **_canonical_filter().to_dict(),
            }
        )
        mcp_payload, mcp_filter = _invoke_mcp_preview()

        # 1. Every surface reconstructed the same typed filter.
        assert cli_filter == daemon_filter == mcp_filter == _canonical_filter()

        # 2. Every surface echoed the same envelope ``scope.filter`` payload.
        cli_envelope_filter = cli_payload["scope"]["filter"]
        daemon_envelope_filter = daemon_payload["scope"]["filter"]
        mcp_envelope_filter = mcp_payload["scope"]["filter"]
        assert cli_envelope_filter == daemon_envelope_filter == mcp_envelope_filter

        # 3. That envelope payload is exactly the canonical filter's
        #    JSON-shape — no surface added or dropped dimensions.
        assert cli_envelope_filter == _canonical_filter().to_dict()

    def test_envelope_filter_round_trips_back_to_typed_filter(self, tmp_path: Path) -> None:
        """Echoing the envelope ``scope.filter`` payload reconstructs the typed filter."""
        cli_payload, _ = _invoke_cli_plan(_operation_for(_canonical_filter()), tmp_path)
        recovered = MaintenanceScopeFilter.from_dict(cli_payload["scope"]["filter"])
        assert recovered == _canonical_filter()


# ---------------------------------------------------------------------------
# Empty-filter parity — the default "no narrowing" case
# ---------------------------------------------------------------------------


class TestEmptyFilterParity:
    """An empty filter must serialize identically across surfaces too."""

    def test_envelope_from_empty_operation_matches_canonical_empty(self) -> None:
        envelope = cast(
            dict[str, Any],
            envelope_from_operation(
                _operation_for(MaintenanceScopeFilter()),
                origin="cli",
                mode="preview",
            ).to_dict(),
        )
        assert envelope["scope"]["filter"] == MaintenanceScopeFilter().to_dict()
