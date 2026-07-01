"""Typed maintenance scope-filter contract (issue #1196).

Pins:

* the :class:`MaintenanceScopeFilter` model — fields, frozen-ness,
  extra-field rejection, ``is_empty`` / ``to_dict`` / ``from_dict``
  round-trip;
* the planner contract — :func:`preview_backfill` and
  :func:`execute_replay` accept the typed filter, surface it on the
  returned :class:`BackfillOperation.scope`, and shrink
  ``affected_rows`` when ``session_ids`` narrows the scope;
* cross-surface parity — CLI ``polylogue ops maintenance plan``, daemon
  ``POST /api/maintenance/plan``, and MCP ``maintenance_preview``
  all serialize an equivalent filter into the same envelope payload.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from polylogue.cli.commands.maintenance import maintenance_group
from polylogue.config import Config
from polylogue.maintenance import replay as replay_mod
from polylogue.maintenance.envelope import envelope_from_operation
from polylogue.maintenance.planner import (
    BackfillKind,
    BackfillOperation,
    BackfillStatus,
    MaintenanceScope,
    preview_backfill,
)
from polylogue.maintenance.replay import execute_replay
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.storage import repair as repair_mod


class TestMaintenanceScopeFilterShape:
    """Pin the typed-filter Pydantic contract."""

    def test_default_filter_is_empty(self) -> None:
        f = MaintenanceScopeFilter()
        assert f.is_empty()
        assert f.session_ids is None
        assert f.provider is None
        assert f.source_root is None
        assert f.time_range is None

    def test_filter_is_frozen(self) -> None:
        f = MaintenanceScopeFilter(provider="claude-code")
        with pytest.raises(ValidationError):
            f.provider = "chatgpt"

    def test_filter_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            MaintenanceScopeFilter(unknown_dimension="boom")  # type: ignore[call-arg]

    def test_session_ids_coerce_list_to_tuple(self) -> None:
        f = MaintenanceScopeFilter(session_ids=["c1", "c2"])  # type: ignore[arg-type]
        assert f.session_ids == ("c1", "c2")

    def test_session_ids_coerce_single_string(self) -> None:
        f = MaintenanceScopeFilter(session_ids="c1")  # type: ignore[arg-type]
        assert f.session_ids == ("c1",)

    def test_time_range_accepts_iso_strings(self) -> None:
        f = MaintenanceScopeFilter(time_range=("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"))  # type: ignore[arg-type]
        assert f.time_range is not None
        since, until = f.time_range
        assert since == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert until == datetime(2026, 2, 1, tzinfo=timezone.utc)

    def test_time_range_rejects_single_value(self) -> None:
        with pytest.raises(ValidationError):
            MaintenanceScopeFilter(time_range=("2026-01-01T00:00:00Z",))  # type: ignore[arg-type]

    def test_source_root_coerces_string_to_path(self) -> None:
        f = MaintenanceScopeFilter(source_root="/var/tmp")  # type: ignore[arg-type]
        assert f.source_root == Path("/var/tmp")


class TestMaintenanceScopeFilterRoundTrip:
    """from_dict(to_dict(f)) == f for every dimension combination."""

    @pytest.mark.parametrize(
        "filter_kwargs",
        [
            {},
            {"session_ids": ("c1",)},
            {"session_ids": ("c1", "c2", "c3")},
            {"provider": "claude-code"},
            {"source_family": "claude-code-session"},
            {"source_root": Path("/data/claude")},
            {"raw_artifact_id": "raw-abc"},
            {"time_range": (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 2, 1, tzinfo=timezone.utc))},
            {"failure_kind": "ValidationError"},
            {"parser_version": "v3"},
            {
                "session_ids": ("c1",),
                "provider": "claude-code",
                "source_family": "claude-code-session",
                "raw_artifact_id": "raw-1",
            },
        ],
    )
    def test_round_trip(self, filter_kwargs: dict[str, Any]) -> None:
        original = MaintenanceScopeFilter(**filter_kwargs)
        payload = original.to_dict()
        # Payload must be a plain dict with every known dimension.
        assert isinstance(payload, dict)
        assert "session_ids" in payload
        assert "time_range" in payload
        recovered = MaintenanceScopeFilter.from_dict(payload)
        assert recovered == original

    def test_from_dict_none_is_empty(self) -> None:
        assert MaintenanceScopeFilter.from_dict(None).is_empty()

    def test_from_dict_empty_is_empty(self) -> None:
        assert MaintenanceScopeFilter.from_dict({}).is_empty()

    def test_from_dict_ignores_absent_dimensions(self) -> None:
        f = MaintenanceScopeFilter.from_dict({"provider": "claude-code"})
        assert f.provider == "claude-code"
        assert f.session_ids is None


class TestPlannerHonorsFilter:
    """The planner threads the filter onto the returned scope and narrows preview counts."""

    def test_preview_attaches_filter_to_scope(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        scope_filter = MaintenanceScopeFilter(session_ids=("c1", "c2"))
        op = preview_backfill(config, targets=("session_insights",), scope_filter=scope_filter)
        assert op.scope is not None
        assert op.scope.filter == scope_filter

    def test_preview_narrows_affected_rows_for_session_ids(self, tmp_path: Path) -> None:
        from polylogue.maintenance.models import DerivedModelStatus

        config = _make_config(tmp_path)
        # Patch the debt collector to advertise 1000 pending insights;
        # a one-session filter must clamp that down to 1.
        fake_status = DerivedModelStatus(
            name="session_insights",
            ready=False,
            detail="1000 pending",
            source_documents=1000,
            materialized_documents=0,
            stale_rows=1000,
            missing_provenance_rows=0,
        )

        def _fake_collect(*_a: Any, **_kw: Any) -> dict[str, DerivedModelStatus]:
            return {"session_insights": fake_status}

        def _fake_counts(_statuses: dict[str, DerivedModelStatus]) -> dict[str, int]:
            return {"session_insights": 1000}

        with (
            patch("polylogue.storage.repair.collect_archive_debt_statuses_sync", _fake_collect),
            patch("polylogue.storage.repair.preview_counts_from_archive_debt", _fake_counts),
        ):
            broad = preview_backfill(config, targets=("session_insights",))
            narrow = preview_backfill(
                config,
                targets=("session_insights",),
                scope_filter=MaintenanceScopeFilter(session_ids=("only-one",)),
            )

        assert broad.affected_rows == 1000
        assert narrow.affected_rows == 1
        assert narrow.scope is not None
        assert narrow.scope.filter.session_ids == ("only-one",)

    def test_execute_replay_passes_raw_artifact_id_to_raw_materialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        captured: dict[str, object] = {}

        def fake_repair_raw_materialization(
            _config: Config,
            dry_run: bool = False,
            *,
            raw_artifact_id: str | None = None,
            provider: str | None = None,
            source_family: str | None = None,
            source_root: Path | None = None,
        ) -> repair_mod.RepairResult:
            captured["dry_run"] = dry_run
            captured["raw_artifact_id"] = raw_artifact_id
            captured["provider"] = provider
            captured["source_family"] = source_family
            captured["source_root"] = source_root
            return repair_mod._repair_result(
                "raw_materialization",
                repaired_count=1,
                success=True,
                detail="scoped replay",
            )

        monkeypatch.setattr("polylogue.maintenance.replay.repair_raw_materialization", fake_repair_raw_materialization)
        monkeypatch.setitem(replay_mod._REPLAY_DISPATCH, "raw_materialization", fake_repair_raw_materialization)

        operation = execute_replay(
            config,
            targets=("raw_materialization",),
            persist_state=False,
            scope_filter=MaintenanceScopeFilter(
                provider="claude-code",
                source_family="claude-code-session",
                source_root=tmp_path / "sources",
                raw_artifact_id="raw-1",
            ),
        )

        assert operation.status is BackfillStatus.COMPLETED
        assert captured == {
            "dry_run": False,
            "provider": "claude-code",
            "source_family": "claude-code-session",
            "source_root": tmp_path / "sources",
            "raw_artifact_id": "raw-1",
        }
        assert operation.scope is not None
        assert operation.scope.filter.raw_artifact_id == "raw-1"
        assert operation.scope.filter.source_family == "claude-code-session"


class TestCrossSurfaceFilterParity:
    """The same typed filter must serialize identically across CLI / daemon / MCP."""

    def test_filter_serializes_identically_across_surfaces(self, tmp_path: Path) -> None:
        operation = _example_operation_with_filter(
            MaintenanceScopeFilter(
                session_ids=("c1", "c2"),
                provider="claude-code",
                source_family="claude-code-session",
            )
        )

        # --- CLI ---
        cli_payload = _capture_cli_preview(operation, tmp_path)

        # --- daemon (direct envelope, same as the HTTP handler emits) ---
        daemon_payload = cast(
            dict[str, Any],
            envelope_from_operation(operation, origin="daemon", mode="preview").to_dict(),
        )

        # --- MCP ---
        mcp_payload = _capture_mcp_preview(operation)

        cli_filter = cli_payload["scope"]["filter"]
        daemon_filter = daemon_payload["scope"]["filter"]
        mcp_filter = mcp_payload["scope"]["filter"]

        assert cli_filter == daemon_filter == mcp_filter
        assert cli_filter["session_ids"] == ["c1", "c2"]
        assert cli_filter["provider"] == "claude-code"
        assert cli_filter["source_family"] == "claude-code-session"

    def test_daemon_http_parses_filter_body(self) -> None:
        """``POST /api/maintenance/plan`` parses the typed-filter body fields.

        The daemon handler accepts both a nested ``{"scope":{"filter":{...}}}``
        envelope and a flat top-level shape; both round-trip through the
        same :class:`MaintenanceScopeFilter`.
        """
        from io import BytesIO
        from typing import cast

        from tests.unit.daemon.test_maintenance_endpoints import MockHeaders, _make_handler

        captured: dict[str, Any] = {}

        def _capture(
            config: Any, *, targets: tuple[str, ...], scope_filter: MaintenanceScopeFilter
        ) -> BackfillOperation:
            captured["targets"] = targets
            captured["filter"] = scope_filter
            return _example_operation_with_filter(scope_filter)

        body = {
            "targets": ["session_insights"],
            "session_ids": ["c1", "c2"],
            "provider": "claude-code",
            "source_family": "claude-code-session",
        }
        body_raw = json.dumps(body).encode("utf-8")
        handler = _make_handler("/api/maintenance/plan")
        cast(MockHeaders, handler.headers)._headers["Content-Length"] = str(len(body_raw))
        handler.rfile = BytesIO(body_raw)

        with (
            patch("polylogue.maintenance.planner.preview_backfill", side_effect=_capture),
            patch.object(handler, "_send_json"),
        ):
            handler._handle_maintenance_plan()

        assert captured["filter"].session_ids == ("c1", "c2")
        assert captured["filter"].provider == "claude-code"
        assert captured["filter"].source_family == "claude-code-session"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _example_operation_with_filter(scope_filter: MaintenanceScopeFilter) -> BackfillOperation:
    return BackfillOperation(
        operation_id="op-1",
        kind=BackfillKind.DERIVED_REBUILD,
        targets=("session_insights",),
        status=BackfillStatus.PENDING,
        scope=MaintenanceScope(targets=("session_insights",), filter=scope_filter),
    )


def _capture_cli_preview(operation: BackfillOperation, tmp_path: Path) -> dict[str, Any]:
    archive = tmp_path / "archive"
    render = tmp_path / "render"
    archive.mkdir(parents=True, exist_ok=True)
    render.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    config_obj = Config(archive_root=archive, render_root=render, sources=[])

    with (
        patch("polylogue.cli.commands.maintenance.preview_backfill", return_value=operation),
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
                "--output-format",
                "json",
            ],
            obj=config_obj,
        )

    assert result.exit_code == 0, result.output
    payload: dict[str, Any] = json.loads(result.output)
    return payload


def _capture_mcp_preview(operation: BackfillOperation) -> dict[str, Any]:
    from polylogue.mcp.server import build_server

    server = build_server(role="admin")
    fn = server._tool_manager._tools["maintenance_preview"].fn

    with patch("polylogue.maintenance.planner.preview_backfill", return_value=operation):
        result = asyncio.run(
            fn(
                session_ids=["c1", "c2"],
                origin="claude-code-session",
                source_family="claude-code-session",
            )
        )
    payload: dict[str, Any] = json.loads(result)
    return payload
