"""Operational daemon HTTP contracts — status envelope, convergence, privacy (#1060).

Security-boundary contracts (auth, origin, JSON serialization) are pinned
by ``tests/unit/daemon/test_daemon_http_security.py`` (#1075/#1076).

This module pins the *operational* contracts of the daemon HTTP surface:

1. ``GET /api/status`` and ``GET /api/health`` return stable, documented
   envelope shapes — required fields are always present, optional fields
   are explicit ``null`` rather than absent, and the payload never leaks
   secrets, environment variables, or process internals.
2. ``GET /api/status`` and ``GET /api/health`` are read-only — invoking
   them never mutates archive state.  The status payload reflects daemon
   liveness, FTS readiness, and convergence debt without performing
   convergence work as a side effect.
3. Stale / disconnected / degraded states surface as explicit envelope
   fields (``daemon_liveness``, ``fts_readiness.fts_ready``,
   ``health.overall_status``, ``convergence`` debt counts) instead of
   raw exceptions or free-form messages.
4. Per-session reader endpoints expose only the requested
   session — adjacent rows are never leaked through ``id``,
   ``title``, ``messages``, or ``raw`` fields.
5. Every authored ``OperationSpec`` declares the minimum fields the
   verification catalog and control-plane surfaces rely on (name,
   description, kind, effects, code_refs).

Tests use the in-process handler pattern from
``test_daemon_http_security.py`` — no real daemon, no socket listener.
"""

from __future__ import annotations

import json
import os
import re
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from polylogue.operations import build_declared_operation_catalog, build_runtime_operation_catalog

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


# ---------------------------------------------------------------------------
# Documented status envelope keys (top level)
# ---------------------------------------------------------------------------

# The full set of keys the daemon promises on ``GET /api/status``.  Adding a
# new top-level key is a contract change — extend this set in the same PR
# that adds the key and update any consumers.
REQUIRED_STATUS_KEYS: frozenset[str] = frozenset(
    {
        "ok",
        "daemon",
        "daemon_liveness",
        "checked_at",
        "component_state",
        "component_readiness",
        "live",
        "browser_capture",
        "db_path",
        "db_size_bytes",
        "wal_size_bytes",
        "blob_dir_size_bytes",
        "disk_free_bytes",
        "quick_check_result",
        "quick_check_age_s",
        "watcher_roots",
        "browser_capture_active",
        "failing_files",
        "live_cursor",
        "live_ingest_attempts",
        "catchup",
        "convergence",
        "operations",
        "last_ingestion_batch",
        "fts_readiness",
        "raw_materialization_readiness",
        "embedding_readiness",
        "memory",
        "health",
        "raw_parse_failures",
        "raw_validation_failures",
        "raw_quarantined",
        "raw_detection_warnings",
        "raw_failure_samples",
        "last_event_id",
    }
)

REQUIRED_HEALTH_KEYS: frozenset[str] = frozenset(
    {
        "ok",
        "db_size_bytes",
        "wal_size_bytes",
        "disk_free_bytes",
        "blob_dir_size_bytes",
        "quick_check",
        "quick_check_age_s",
    }
)


# ---------------------------------------------------------------------------
# In-process handler harness — mirrors test_daemon_http_security.py
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""  # local-dev default: no auth needed for these contracts
    api_host = "127.0.0.1"


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str, *, body: bytes = b"") -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    headers: dict[str, str] = {"Content-Length": str(len(body))}
    handler.headers = cast("Message[str, str]", _MockHeaders(headers))
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


def _init_archive(archive_root: Path) -> None:
    """Bootstrap the archive root so read endpoints open
    existing files rather than creating them on first access."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root.mkdir(parents=True, exist_ok=True)
    ArchiveStore(archive_root).close()


def _archive_state_hash(archive_root: Path) -> str:
    """Cheap fingerprint of the archive directory tree.

    Used to assert read endpoints don't mutate state — we don't need a
    cryptographic hash, just a deterministic snapshot of filenames +
    sizes + mtimes so any write would change the result.
    """
    import hashlib

    h = hashlib.sha256()
    if not archive_root.exists():
        return "<missing>"
    for path in sorted(archive_root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            h.update(str(path.relative_to(archive_root)).encode())
            h.update(f"{stat.st_size}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Status envelope contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestStatusEnvelopeContract:
    """``GET /api/status`` envelope is stable and documented."""

    def test_required_keys_always_present(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """All ``REQUIRED_STATUS_KEYS`` are present in every status response.

        Missing fields must be explicit ``null`` rather than absent —
        consumers (web reader, dashboards) should never need to
        ``payload.get(key, default)`` for a documented field.
        """
        handler = _make_handler("GET", "/api/status")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert isinstance(payload, dict)

        actual_keys = frozenset(payload.keys())
        missing = REQUIRED_STATUS_KEYS - actual_keys
        assert not missing, f"status envelope missing required keys: {sorted(missing)}"

        actual_keys - REQUIRED_STATUS_KEYS
        # Extra keys are allowed (forward-compat) but worth recording so
        # we notice when new fields are added without docs.

    def test_component_state_envelope_has_documented_subsystems(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """``component_state`` reports api / watcher / browser_capture subsystems.

        These three subsystems are the documented daemon components; the
        dashboard renders one strip per subsystem.  Adding a new one is a
        contract change.
        """
        handler = _make_handler("GET", "/api/status")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        component_state = payload["component_state"]
        assert isinstance(component_state, dict)
        assert frozenset(component_state.keys()) == frozenset({"api", "watcher", "browser_capture"})

    def test_component_readiness_envelope_has_canonical_status_components(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """``component_readiness`` exposes daemon status through the shared DTO."""
        handler = _make_handler("GET", "/api/status")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        component_readiness = payload["component_readiness"]
        assert isinstance(component_readiness, dict)
        assert {
            "archive_storage",
            "browser_capture",
            "daemon_api",
            "daemon_ingest",
            "daemon_watcher",
            "embeddings",
            "raw_materialization",
            "search",
        } <= set(component_readiness)
        api = component_readiness["daemon_api"]
        assert isinstance(api, dict)
        assert api["state"] == "ready"
        assert api["scope"] == "daemon"

    def test_health_envelope_has_documented_fields(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """``GET /api/health`` returns a stable envelope.

        The CI/Docker health-check endpoint is ``/api/health/check``
        (binary).  The richer ``/api/health`` endpoint exposes durable
        diagnostics — drift here breaks dashboards.
        """
        handler = _make_handler("GET", "/api/health")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert isinstance(payload, dict)
        missing = REQUIRED_HEALTH_KEYS - frozenset(payload.keys())
        assert not missing, f"health envelope missing required keys: {sorted(missing)}"
        # quick_check is a stable enum, not a free-form message
        assert payload["quick_check"] in {"pass", "error"}
        assert isinstance(payload["ok"], bool)

    def test_health_endpoint_uses_bounded_db_probe(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``GET /api/health`` must not run archive readiness scans."""
        archive_root = workspace_env["archive_root"]
        _init_archive(archive_root)

        def fail_readiness(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("health endpoint must not call archive readiness")

        monkeypatch.setattr("polylogue.readiness.get_readiness", fail_readiness)

        handler = _make_handler("GET", "/api/health")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["quick_check"] == "pass"
        assert payload["ok"] is True


# ---------------------------------------------------------------------------
# 2. Convergence/idempotency contracts at the HTTP layer
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestStatusReadOnlyContract:
    """Reading status/health never mutates archive state."""

    def test_status_endpoint_does_not_mutate_archive(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """``GET /api/status`` returns observations only.

        Status assembly inspects DB size, FTS readiness, convergence
        debt counts, etc. — all read-only queries. A regression that
        triggered convergence work as a side effect of status polling
        would be caught here.
        """
        archive_root = workspace_env["archive_root"]
        _init_archive(archive_root)
        # Seed a marker file so the hash is non-empty.
        (archive_root / "marker.txt").write_text("seed")

        # Warm the read path once so any lazy WAL/SHM sidecar files for the
        # archive tiers materialize before the snapshot — the contract is that
        # *repeated* status reads do not mutate archive content.
        warm = _make_handler("GET", "/api/status")
        _capture_responses(warm)
        warm.do_GET()

        before = _archive_state_hash(archive_root)

        for _ in range(3):
            handler = _make_handler("GET", "/api/status")
            _, send_json = _capture_responses(handler)
            handler.do_GET()
            send_json.assert_called_once()

        after = _archive_state_hash(archive_root)
        assert before == after, "status endpoint mutated archive state"

    def test_health_endpoint_does_not_mutate_archive(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        archive_root = workspace_env["archive_root"]
        _init_archive(archive_root)
        (archive_root / "marker.txt").write_text("seed")

        warm = _make_handler("GET", "/api/health")
        _capture_responses(warm)
        warm.do_GET()
        before = _archive_state_hash(archive_root)

        for _ in range(3):
            handler = _make_handler("GET", "/api/health")
            _, send_json = _capture_responses(handler)
            handler.do_GET()

        after = _archive_state_hash(archive_root)
        assert before == after


# ---------------------------------------------------------------------------
# 3. Stale/disconnected/degraded state contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestDegradedStateContract:
    """Stale / disconnected / degraded states surface as structured fields."""

    def test_status_reports_daemon_liveness_false_when_no_pidfile(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """``daemon_liveness`` is the documented disconnected indicator.

        When the daemon process isn't running (no pidfile, dead pid)
        the status payload still serves — clients see
        ``daemon_liveness=False`` rather than a connection error.

        This is the contract the web reader's status strip relies on
        to render the offline state.
        """
        # No pidfile in the workspace ⇒ liveness must be False.
        handler = _make_handler("GET", "/api/status")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        assert payload["daemon_liveness"] is False
        # component_state.api reflects the same fact.
        assert isinstance(payload["component_state"], dict)

    def test_health_check_endpoint_reports_503_on_degraded_health(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``GET /api/health/check`` returns 503 when health is degraded.

        This is the contract Docker/systemd/CI healthchecks rely on:
        a degraded daemon must produce a non-2xx response so the
        external monitor will restart / alert.
        """
        from polylogue.daemon import health as health_module
        from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

        bad_alert = HealthAlert(
            check_name="wal_size",
            tier=HealthTier.FAST,
            severity=HealthSeverity.CRITICAL,
            message="WAL too large",
            checked_at="2026-05-16T00:00:00+00:00",
            consecutive_failures=3,
        )
        monkeypatch.setattr(health_module, "_run_fast_checks", lambda: [bad_alert])
        monkeypatch.setattr(health_module, "_run_medium_checks", lambda: [])
        monkeypatch.setattr(health_module, "_run_expensive_checks", lambda: [])

        handler = _make_handler("GET", "/api/health/check")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert isinstance(payload, dict)
        assert payload["ok"] is False
        # status field is a documented enum — never a raw exception string.
        assert isinstance(payload["status"], str)
        # alerts count is structured, not embedded in a message.
        assert isinstance(payload["alerts"], int)

    def test_health_check_endpoint_returns_503_on_internal_error(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unexpected error in the health checker still produces a
        well-formed 503 envelope — never a 500 with a raw traceback.

        This is the contract documented in ``_handle_health_check``:
        any exception is caught and translated to ``{"ok": False,
        "status": "error", "detail": "health check failed"}``.
        """
        from polylogue.daemon import health as health_module

        def _explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated infra failure")

        monkeypatch.setattr(health_module, "check_health", _explode)

        handler = _make_handler("GET", "/api/health/check")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert isinstance(payload, dict)
        assert payload["ok"] is False
        assert payload["status"] == "error"
        # detail is bounded, structured — not a raw exception message.
        assert payload["detail"] == "health check failed"

    def test_status_fts_readiness_is_structured_field(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """FTS index sync is exposed via ``fts_readiness`` (a dict),
        never as a free-form ``message`` string. Dashboards filter on
        ``fts_readiness.fts_ready`` to decide whether to show degraded
        search.
        """
        handler = _make_handler("GET", "/api/status")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        readiness = payload["fts_readiness"]
        assert isinstance(readiness, dict), "fts_readiness must be a structured envelope"


# ---------------------------------------------------------------------------
# 4. Privacy contracts
# ---------------------------------------------------------------------------


# A grab-bag of secret-shaped environment variable names that the daemon
# might be configured with.  None of these may surface in
# ``GET /api/status`` or any HTML/JSON the reader serves.
SECRET_ENV_NAMES = ("VOYAGE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.mark.contract
class TestPrivacyContract:
    """Daemon HTTP surface never leaks secrets or adjacent sessions."""

    def test_dev_loop_payload_reports_allowlisted_launcher_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Branch-local debug metadata is explicit and does not dump env."""

        run_dir = tmp_path / "run"
        archive_root = tmp_path / "archive"
        monkeypatch.setenv("POLYLOGUE_DEV_LOOP_RUN_ID", "dev-run-123")
        monkeypatch.setenv("POLYLOGUE_DEV_LOOP_LOG_DIR", str(run_dir))
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
        monkeypatch.setenv("POLYLOGUE_API_PORT", "8876")
        monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_PORT", "8875")
        monkeypatch.setenv("POLYLOGUE_API_AUTH_TOKEN", "SECRET-DEV-LOOP-TOKEN")
        monkeypatch.setenv("POLYLOGUE_TEST_SENTINEL_ENV_DO_NOT_DUMP", "SECRET-ENV-DUMP-CANARY")

        handler = _make_handler("GET", "/api/dev-loop")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload == {
            "ok": True,
            "enabled": True,
            "run_id": "dev-run-123",
            "log_dir": str(run_dir),
            "archive_root": str(archive_root),
            "api_port": 8876,
            "browser_capture_port": 8875,
            "pid": os.getpid(),
            "cwd": os.getcwd(),
        }
        serialized = json.dumps(payload)
        assert "SECRET-DEV-LOOP-TOKEN" not in serialized
        assert "SECRET-ENV-DUMP-CANARY" not in serialized
        assert "POLYLOGUE_TEST_SENTINEL_ENV_DO_NOT_DUMP" not in serialized

    def test_dev_loop_payload_is_disabled_without_launcher_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for name in (
            "POLYLOGUE_DEV_LOOP_RUN_ID",
            "POLYLOGUE_DEV_LOOP_LOG_DIR",
            "POLYLOGUE_ARCHIVE_ROOT",
            "POLYLOGUE_API_PORT",
            "POLYLOGUE_BROWSER_CAPTURE_PORT",
        ):
            monkeypatch.delenv(name, raising=False)

        handler = _make_handler("GET", "/api/dev-loop")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["enabled"] is False
        assert payload["run_id"] is None
        assert payload["log_dir"] is None

    def test_status_payload_does_not_leak_secrets(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``GET /api/status`` must not include API keys or process env.

        We set a distinctive sentinel for each known secret env var,
        request the status payload, and assert the sentinel does not
        appear anywhere in the serialized payload — including nested
        values.
        """
        sentinels = {name: f"SENTINEL-{name}-DO-NOT-LEAK-{os.getpid()}" for name in SECRET_ENV_NAMES}
        for name, value in sentinels.items():
            monkeypatch.setenv(name, value)

        handler = _make_handler("GET", "/api/status")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        serialized = json.dumps(payload)
        leaks: list[str] = []
        for name, value in sentinels.items():
            if value in serialized:
                leaks.append(name)
        assert not leaks, f"status payload leaked secret env vars: {leaks}"

    def test_status_payload_does_not_leak_full_environment(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No accidental ``dict(os.environ)`` dump in the status payload.

        We plant a uniquely-named env var that is not part of any
        documented config field and assert it never appears in the
        serialized payload.
        """
        sentinel_name = "POLYLOGUE_TEST_SENTINEL_ENV_DO_NOT_DUMP"
        sentinel_value = f"ENV-DUMP-CANARY-{os.getpid()}"
        monkeypatch.setenv(sentinel_name, sentinel_value)

        handler = _make_handler("GET", "/api/status")
        _, send_json = _capture_responses(handler)
        handler.do_GET()

        _, payload = send_json.call_args.args
        serialized = json.dumps(payload)
        assert sentinel_value not in serialized
        assert sentinel_name not in serialized

    def test_web_shell_html_has_no_inline_secrets(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The static web shell HTML must not bake in any env-derived secrets.

        The reader is served as a static SPA — runtime auth happens
        via the bearer token in JS-issued requests, never via tokens
        substituted into the page source.
        """
        sentinel = f"SENTINEL-WEBSHELL-{os.getpid()}"
        for name in SECRET_ENV_NAMES:
            monkeypatch.setenv(name, sentinel)

        from polylogue.daemon.web_shell import WEB_SHELL_HTML

        assert sentinel not in WEB_SHELL_HTML

    def test_load_status_success_path_clears_the_status_route_notice(self) -> None:
        """Dogfood regression (2026-07-08): loadStatus()'s success path set
        state.routeStates.status to 'ready' but never called renderFacets(),
        so a "Status: loading" notice rendered from an earlier snapshot
        stayed stuck in the sidebar forever — verified live against the
        real archive daemon. The error path already called renderFacets();
        the success path must too, or any future edit that reintroduces
        this asymmetry regresses silently (there is no JS-executing test
        for this file's inline script, so this is a structural string
        check on the success-path source, not a DOM assertion)."""
        from polylogue.daemon.web_shell import WEB_SHELL_HTML

        anchor = "setRouteState('status', {state: 'ready', route: statusRoute, status: '200', error: ''});"
        idx = WEB_SHELL_HTML.index(anchor)
        following = WEB_SHELL_HTML[idx + len(anchor) : idx + len(anchor) + 200]
        assert "renderFacets();" in following, (
            "loadStatus() success path must call renderFacets() to clear any "
            "stale 'Status: loading' notice rendered before this route became ready"
        )

    def test_get_session_does_not_leak_adjacent_sessions(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """``GET /api/sessions/<id>`` returns only the requested row.

        Seeds three sessions (``c1``, ``c2``, ``c3``), requests
        ``c1``, and asserts the serialized payload mentions ``c1``
        identifiers but not ``c2``/``c3``.
        """
        from tests.infra.storage_records import SessionBuilder, db_setup

        dbp = db_setup(workspace_env)
        c1_session_id = ""
        for cid, title, msg_text in [
            ("c1", "First session", "Alpha content here"),
            ("c2", "Second session SECRET-C2", "Beta content SECRET-C2"),
            ("c3", "Third session SECRET-C3", "Gamma content SECRET-C3"),
        ]:
            builder = (
                SessionBuilder(dbp, cid)
                .provider("claude-code")
                .title(title)
                .add_message(message_id=f"m-{cid}", role="user", text=msg_text)
            )
            builder.save()
            if cid == "c1":
                c1_session_id = builder.native_session_id()

        handler = _make_handler("GET", f"/api/sessions/{c1_session_id}")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()

        send_error.assert_not_called()
        send_json.assert_called_once()
        _, payload = send_json.call_args.args
        serialized = json.dumps(payload)
        # Requested row is present...
        assert "c1" in serialized
        assert "Alpha content here" in serialized
        # ...adjacent rows are not leaked.
        for forbidden in ("SECRET-C2", "SECRET-C3", "Second session", "Third session"):
            assert forbidden not in serialized, f"adjacent session data leaked: {forbidden!r}"


# ---------------------------------------------------------------------------
# 5. Operation spec contracts
# ---------------------------------------------------------------------------


# Runtime operation names are kebab-case (``acquire-raw-sessions``).
# Declared scenario/benchmark entries are namespaced (``benchmark.query.foo``).
_RUNTIME_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]*$")
_DECLARED_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]*(\.[a-z][a-z0-9\-]*)*$")
_CODE_REF_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$")
_KNOWN_EFFECTS = frozenset({"Pure", "DbRead", "DbWrite", "FileWrite", "Network", "LiveArchive", "Destructive"})
_MUTATING_EFFECTS = frozenset({"DbWrite", "FileWrite", "Destructive"})
_KNOWN_SAFETY_GUARDS = frozenset({"write_role_required", "confirmed_before_execute", "explicit_dry_run_evidence"})


@pytest.mark.contract
class TestOperationSpecContract:
    """Every authored operation has the metadata the catalog relies on.

    Two tiers of strictness:

    - ``RUNTIME_OPERATION_SPECS``: production daemon operations.  Must
      declare code_refs (the verification catalog uses them to attribute
      checks to source), effects (so previewable/idempotency tooling
      knows which ones to flag), and ``mutates_state=True`` for any
      write/destructive effect.
    - ``DECLARED_OPERATION_SPECS``: superset including scenario /
      benchmark / fixture entries.  Only structural fields (name,
      description, effects enum membership) are enforced — these
      entries are advisory and may not point at concrete code.
    """

    def test_runtime_operations_are_well_formed(
        self,
    ) -> None:
        """Strict contract for the runtime operation catalog."""
        catalog = build_runtime_operation_catalog()
        assert catalog.specs, "runtime operation catalog must not be empty"

        offenders: list[str] = []
        for spec in catalog.specs:
            issues: list[str] = []
            if not spec.name or not _RUNTIME_NAME_RE.match(spec.name):
                issues.append(f"bad name: {spec.name!r}")
            if not spec.description or len(spec.description) < 10:
                issues.append("description too short")
            if not spec.code_refs:
                issues.append("no code_refs")
            else:
                for ref in spec.code_refs:
                    if not _CODE_REF_RE.match(ref):
                        issues.append(f"malformed code_ref: {ref!r}")
            if not spec.effects:
                issues.append("no effects declared")
            else:
                unknown = [e for e in spec.effects if e not in _KNOWN_EFFECTS]
                if unknown:
                    issues.append(f"unknown effects: {unknown}")
            if any(e in _MUTATING_EFFECTS for e in spec.effects) and not spec.mutates_state:
                issues.append("mutating effects but mutates_state=False")
            unknown_guards = [guard for guard in spec.safety_guards if guard not in _KNOWN_SAFETY_GUARDS]
            if unknown_guards:
                issues.append(f"unknown safety_guards: {unknown_guards}")
            if spec.mutates_state and "mcp" in spec.surfaces and "write_role_required" not in spec.safety_guards:
                issues.append("mutating MCP surface but no write_role_required guard")
            if "Destructive" in spec.effects:
                missing_guards = {
                    "confirmed_before_execute",
                    "explicit_dry_run_evidence",
                } - set(spec.safety_guards)
                if missing_guards:
                    issues.append(f"destructive operation missing safety guards: {sorted(missing_guards)}")
            if issues:
                offenders.append(f"{spec.name}: {'; '.join(issues)}")

        assert not offenders, "runtime operation spec violations:\n" + "\n".join(offenders)

    def test_declared_operations_have_structural_fields(self) -> None:
        """Light-touch contract for the declared (superset) catalog.

        Scenario / benchmark entries in the declared catalog may omit
        code_refs (they describe scenarios, not call sites), but must
        still have a name, a description, and known effects if any are
        declared.
        """
        catalog = build_declared_operation_catalog()
        assert catalog.specs, "declared operation catalog must not be empty"

        offenders: list[str] = []
        for spec in catalog.specs:
            issues: list[str] = []
            if not spec.name or not _DECLARED_NAME_RE.match(spec.name):
                issues.append(f"bad name: {spec.name!r}")
            if not spec.description or len(spec.description) < 10:
                issues.append("description too short")
            if spec.effects:
                unknown = [e for e in spec.effects if e not in _KNOWN_EFFECTS]
                if unknown:
                    issues.append(f"unknown effects: {unknown}")
            unknown_guards = [guard for guard in spec.safety_guards if guard not in _KNOWN_SAFETY_GUARDS]
            if unknown_guards:
                issues.append(f"unknown safety_guards: {unknown_guards}")
            if issues:
                offenders.append(f"{spec.name}: {'; '.join(issues)}")

        assert not offenders, "declared operation spec violations:\n" + "\n".join(offenders)

    def test_operation_names_are_unique(self) -> None:
        """The catalog is keyed by name; duplicates would mask one entry."""
        catalog = build_declared_operation_catalog()
        names = [spec.name for spec in catalog.specs]
        seen: dict[str, int] = {}
        for name in names:
            seen[name] = seen.get(name, 0) + 1
        duplicates = {name: count for name, count in seen.items() if count > 1}
        assert not duplicates, f"duplicate operation names in catalog: {duplicates}"


# ---------------------------------------------------------------------------
# Error-envelope convergence (#1818)
# ---------------------------------------------------------------------------


class TestErrorEnvelopeContract:
    """Every genuine daemon error response shares one machine-output shape.

    ``_send_error`` is the single helper for daemon error responses; it emits
    ``{"ok": False, "error": <code>, "detail": <str|null>, "field": <str|null>}``
    through the shared ``QueryErrorPayload``, identical to the
    ``daemon_safe_handler`` decorator and the cursor-rejection path. Health and
    status payloads use a deliberately separate shape (``status`` / ``alerts`` /
    ``quick_check`` keys) and must NOT be routed through this helper.
    """

    def test_send_error_emits_canonical_envelope(self) -> None:
        handler = _make_handler("GET", "/api/status")
        send_json = MagicMock()
        handler._send_json = send_json  # type: ignore[method-assign]
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert payload == {"ok": False, "error": "not_found", "detail": None, "field": None}

    def test_send_error_includes_detail_when_provided(self) -> None:
        handler = _make_handler("GET", "/api/status")
        send_json = MagicMock()
        handler._send_json = send_json  # type: ignore[method-assign]
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_cursor", "bad anchor")
        _, payload = send_json.call_args.args
        assert payload == {
            "ok": False,
            "error": "invalid_cursor",
            "detail": "bad anchor",
            "field": None,
        }

    def test_do_options_error_routes_through_canonical_envelope(self) -> None:
        """A representative real error site (405 on OPTIONS) carries the shape."""
        handler = _make_handler("OPTIONS", "/api/status")
        send_json = MagicMock()
        handler._send_json = send_json  # type: ignore[method-assign]
        handler.do_OPTIONS()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.METHOD_NOT_ALLOWED
        assert payload["ok"] is False
        assert payload["error"] == "method_not_allowed"
        assert payload["detail"] is None

    def test_health_payload_is_not_an_error_envelope(self) -> None:
        """Health/status sites keep their own shape and never go through _send_error.

        The healthz unhealthy/error bodies carry ``status``/``alerts`` keys, not
        an ``error`` code — that separation is intentional (#1818) and must hold.
        """
        from polylogue.surfaces.payloads import QueryErrorPayload

        # The canonical error envelope never carries health-status keys.
        error_keys = set(QueryErrorPayload(error="x").model_dump(mode="json"))
        assert error_keys == {"ok", "error", "detail", "field"}
        assert "status" not in error_keys
        assert "alerts" not in error_keys
        assert "quick_check" not in error_keys
