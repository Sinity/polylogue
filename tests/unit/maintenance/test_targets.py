"""Catalog-owns-replay contract tests (polylogue-71ey).

Two concrete bugs motivated this file:

1. The canonical maintenance target catalog
   (:mod:`polylogue.maintenance.targets`) advertised seven targets, but
   the resumable replay executor kept a private, hand-maintained
   dispatch table that omitted ``superseded_raw_snapshots``. The
   generated CLI accepted the target (``click.Choice`` is built from the
   catalog) and then failed at runtime with
   :class:`~polylogue.maintenance.replay.UnsupportedReplayTargetError`.
2. ``polylogue ops maintenance run`` with no ``--target`` (the
   documented "no scope — full inventory" targetless path) resolved to
   zero targets and returned ``status=failed`` with exit code 0.

``TestCatalogReplayEquality`` proves every catalog target is either
replay-capable (with a real handler in
:data:`polylogue.storage.repair.REPAIR_HANDLERS`) or explicitly
declared non-replayable with a surface-visible reason -- derived from
the catalog and the real handler dict, not a hardcoded list mirroring
either. ``TestRealAdapterTargetlessRunAll`` and
``TestExplicitSupersededRawSnapshotsRoute`` exercise the three real
adapters (CLI, MCP, HTTP) end to end rather than rendering a prebuilt
envelope. ``TestFailedEnvelopeSurfaces`` pins AC 4: a failed maintenance
envelope is a non-zero CLI exit and a typed HTTP/MCP failure, not a
200-shaped success body.
"""

from __future__ import annotations

import asyncio
import io
import json
from email.message import Message
from http import HTTPStatus
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.daemon.http import DaemonAPIHandler
from polylogue.maintenance.targets import build_maintenance_target_catalog
from polylogue.storage import repair as repair_module


def _extract_json_envelope(output: str) -> dict[str, object]:
    """Pull the trailing ``--output-format json`` envelope out of CLI output.

    ``CliRunner`` (Click 8.2+) merges stdout and stderr into one stream, so
    ``result.output`` also carries the ``--dry-run`` progress lines this
    command writes to stderr before the JSON envelope. Those lines never
    contain ``{``, so decoding from the first ``{`` in the stream isolates
    the JSON payload.
    """
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(output, output.index("{"))
    assert isinstance(payload, dict)
    return cast("dict[str, object]", payload)


def _targets(payload: dict[str, object]) -> set[str]:
    return set(cast("list[str]", payload["targets"]))


def _always_raises(_config: object, _dry_run: bool) -> object:
    raise RuntimeError("simulated repair failure (polylogue-71ey AC 4 fixture)")


def _http_headers(body: bytes) -> Message:
    headers = Message()
    headers["Content-Length"] = str(len(body))
    return headers


def _build_run_handler(body: bytes) -> tuple[DaemonAPIHandler, list[tuple[object, dict[str, object]]]]:
    """Construct a bare ``DaemonAPIHandler`` and drive its real POST body.

    Mirrors the pattern in ``tests/unit/daemon/test_http_write_coordination.py``:
    ``object.__new__`` skips socket setup, and ``_send_json``/``_send_error``
    are replaced with recording stubs so the real ``_handle_maintenance_run``
    body runs to completion without a live connection.
    """
    handler = object.__new__(DaemonAPIHandler)
    handler.headers = _http_headers(body)
    handler.rfile = io.BytesIO(body)
    calls: list[tuple[object, dict[str, object]]] = []
    handler._send_json = lambda status, payload, **_kw: calls.append((status, payload))  # type: ignore[method-assign]
    handler._send_error = lambda *a, **kw: calls.append(("error", {"args": a, "kwargs": kw}))  # type: ignore[method-assign]
    return handler, calls


# ---------------------------------------------------------------------------
# AC 1: catalog equality -- every target is replayable-with-handler or
# explicitly non-replayable-with-reason.
# ---------------------------------------------------------------------------


class TestCatalogReplayEquality:
    def test_every_replayable_target_has_a_real_handler(self) -> None:
        """A target claiming ``replayable=True`` must have a real handler.

        Deleting a :data:`polylogue.storage.repair.REPAIR_HANDLERS` entry
        for any replayable catalog target must fail this test -- it is
        derived from the live catalog and the live handler dict, not a
        name list copied from either.
        """
        catalog = build_maintenance_target_catalog()
        missing_handlers = [
            spec.name for spec in catalog.specs if spec.replayable and spec.name not in repair_module.REPAIR_HANDLERS
        ]
        assert missing_handlers == [], (
            f"catalog declares these targets replayable but REPAIR_HANDLERS has no entry for them: {missing_handlers}"
        )

    def test_every_non_replayable_target_has_a_visible_reason(self) -> None:
        catalog = build_maintenance_target_catalog()
        unexplained = [spec.name for spec in catalog.specs if not spec.replayable and not spec.non_replayable_reason]
        assert unexplained == [], f"non-replayable targets missing a surface-visible reason: {unexplained}"

    def test_non_replayable_reason_surfaces_in_to_dict(self) -> None:
        catalog = build_maintenance_target_catalog()
        for spec in catalog.specs:
            payload = spec.to_dict()
            assert payload["replayable"] == spec.replayable
            assert payload["non_replayable_reason"] == spec.non_replayable_reason

    def test_supported_replay_targets_matches_catalog_replayable_set(self) -> None:
        """``supported_replay_targets()`` derives from the catalog + REPAIR_HANDLERS.

        This is the anti-vacuity check for bug 1: it fails if the two
        ever diverge again, without hardcoding either side's target
        names.
        """
        from polylogue.maintenance.replay import supported_replay_targets

        catalog = build_maintenance_target_catalog()
        expected = {
            spec.name for spec in catalog.specs if spec.replayable and spec.name in repair_module.REPAIR_HANDLERS
        }
        assert set(supported_replay_targets()) == expected

    def test_superseded_raw_snapshots_is_declared_replayable(self) -> None:
        """Regression pin for bug 1: this target was silently excluded before."""
        catalog = build_maintenance_target_catalog()
        spec = catalog.resolve_name("superseded_raw_snapshots")
        assert spec is not None
        assert spec.replayable is True
        assert "superseded_raw_snapshots" in repair_module.REPAIR_HANDLERS


# ---------------------------------------------------------------------------
# AC 1 (route proof) + AC 3 (CLI real route): explicit --target
# superseded_raw_snapshots succeeds through the real CLI, not a
# hand-rolled call into execute_replay.
# ---------------------------------------------------------------------------


class TestExplicitSupersededRawSnapshotsRoute:
    def test_cli_run_explicit_target_succeeds(self, cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "run",
                "--target",
                "superseded_raw_snapshots",
                "--dry-run",
                "--output-format",
                "json",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = _extract_json_envelope(result.output)
        assert payload["status"] == "completed"
        assert _targets(payload) == {"superseded_raw_snapshots"}
        results = cast("list[dict[str, object]]", payload["results"])
        assert results[0]["name"] == "superseded_raw_snapshots"
        assert results[0]["success"] is True


# ---------------------------------------------------------------------------
# AC 2 + AC 3: targetless run-all through the three real adapters.
# ---------------------------------------------------------------------------


def _catalog_names() -> frozenset[str]:
    return frozenset(build_maintenance_target_catalog().names())


class TestRealAdapterTargetlessRunAll:
    """``--dry-run`` with no ``--target`` must expand to the documented
    run-all set (every catalog target) and succeed, through each of the
    three real request adapters -- not a synthetic ``BackfillOperation``
    rendered through ``envelope_from_operation`` in isolation.
    """

    def test_cli_targetless_dry_run_expands_to_catalog_and_succeeds(
        self, cli_workspace: dict[str, Path], cli_runner: CliRunner
    ) -> None:
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "run", "--dry-run", "--output-format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = _extract_json_envelope(result.output)
        assert payload["status"] == "completed"
        assert _targets(payload) == _catalog_names()

    def test_mcp_targetless_execute_dry_run_expands_to_catalog_and_succeeds(
        self, cli_workspace: dict[str, Path]
    ) -> None:
        from polylogue.mcp.declarations.models import MCPCapabilities
        from polylogue.mcp.server import build_server

        server = build_server(capabilities=MCPCapabilities(maintenance=True))
        fn = server._tool_manager._tools["maintenance"].fn

        result = asyncio.run(fn(operation="execute", dry_run=True))
        payload = cast("dict[str, object]", json.loads(result))
        assert "ok" not in payload  # the typed error envelope shape, absent on success
        assert payload["status"] == "completed"
        assert _targets(payload) == _catalog_names()

    def test_http_targetless_run_dry_run_expands_to_catalog_and_succeeds(self, cli_workspace: dict[str, Path]) -> None:
        body = json.dumps({"dry_run": True}).encode("utf-8")
        handler, calls = _build_run_handler(body)

        handler._handle_maintenance_run()

        assert len(calls) == 1
        status, payload = calls[0]
        assert status == HTTPStatus.OK
        assert payload["status"] == "completed"
        assert _targets(payload) == _catalog_names()

    def test_three_adapters_agree_on_targetless_target_set(
        self, cli_workspace: dict[str, Path], cli_runner: CliRunner
    ) -> None:
        """The one property the three real adapters must share: what
        "no explicit target" resolves to. Deleting the shared
        ``MaintenanceTargetCatalog.resolve_or_default`` default-expansion
        behavior (reverting to the old "empty targets -> failed" path in
        any of the three call sites) makes this test fail.
        """
        cli_result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "run", "--dry-run", "--output-format", "json"],
            catch_exceptions=False,
        )
        cli_targets = _targets(_extract_json_envelope(cli_result.output))

        from polylogue.mcp.declarations.models import MCPCapabilities
        from polylogue.mcp.server import build_server

        server = build_server(capabilities=MCPCapabilities(maintenance=True))
        fn = server._tool_manager._tools["maintenance"].fn
        mcp_payload = cast("dict[str, object]", json.loads(asyncio.run(fn(operation="execute", dry_run=True))))
        mcp_targets = _targets(mcp_payload)

        body = json.dumps({"dry_run": True}).encode("utf-8")
        handler, calls = _build_run_handler(body)
        handler._handle_maintenance_run()
        assert calls[0][0] == HTTPStatus.OK
        http_targets = _targets(calls[0][1])

        assert cli_targets == mcp_targets == http_targets == _catalog_names()


# ---------------------------------------------------------------------------
# AC 4: a failed maintenance envelope is a non-zero CLI exit and a typed
# HTTP/MCP failure -- not a 200/exit-0 body that happens to say "failed".
# ---------------------------------------------------------------------------


class TestFailedEnvelopeSurfaces:
    """Drive a deterministic failure (the ``session_insights`` handler
    raises) through each real adapter and assert the failure is visible
    without parsing the response body -- exit code / HTTP status /
    typed MCP error, not merely a ``"status": "failed"`` string a
    caller has to notice.

    ``dry_run`` must stay ``False`` here: the offline daemon-PID guard
    short-circuits to "no block" whenever ``dry_run=True``
    (``offline_maintenance_block_reason``), so a dry-run request cannot
    exercise this failure path -- only a raising handler can.
    """

    def test_cli_run_exits_non_zero_on_failure(
        self, cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(repair_module.REPAIR_HANDLERS, "session_insights", _always_raises)
        result = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "run",
                "--target",
                "session_insights",
                "--output-format",
                "json",
            ],
        )
        payload = _extract_json_envelope(result.output)
        assert payload["status"] == "failed"
        assert result.exit_code != 0

    def test_http_run_returns_422_on_failure(
        self, cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(repair_module.REPAIR_HANDLERS, "session_insights", _always_raises)
        body = json.dumps({"targets": ["session_insights"]}).encode("utf-8")
        handler, calls = _build_run_handler(body)

        handler._handle_maintenance_run()

        assert len(calls) == 1
        status, payload = calls[0]
        assert status == HTTPStatus.UNPROCESSABLE_ENTITY
        assert payload["status"] == "failed"

    def test_mcp_execute_returns_typed_error_on_failure(
        self, cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(repair_module.REPAIR_HANDLERS, "session_insights", _always_raises)
        from polylogue.mcp.declarations.models import MCPCapabilities
        from polylogue.mcp.server import build_server

        server = build_server(capabilities=MCPCapabilities(maintenance=True))
        fn = server._tool_manager._tools["maintenance"].fn

        result = asyncio.run(fn(operation="execute", targets=["session_insights"]))
        payload = json.loads(result)
        assert payload["ok"] is False
        assert payload["code"] == "maintenance_execute_failed"
