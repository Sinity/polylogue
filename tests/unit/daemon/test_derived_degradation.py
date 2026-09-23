from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.core.errors import SchemaSkew, SchemaVersionMismatchError
from polylogue.daemon.derived_degradation import schema_refusal_details
from polylogue.daemon.health import HealthSeverity, _check_schema_version_fast
from polylogue.daemon.status import _derived_identity_mismatches
from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL, DaemonOperationRequest
from polylogue.operations.mutation_transaction import MutationPrincipal
from polylogue.operations.operation_context import OperationContext


def test_schema_skew_details_preserve_provenance_and_unknown_progress() -> None:
    refusal = SchemaSkew("index", "expected-identity", "actual-identity")

    details = schema_refusal_details(refusal)

    assert details["tier"] == "index"
    assert details["expected_identity"] == "expected-identity"
    assert details["actual_identity"] == "actual-identity"
    assert details["route"] == "daemon_convergence"
    assert details["progress"] == {
        "state": "unknown",
        "completed": None,
        "total": None,
        "percent": None,
        "reason": "no convergence progress measurement was available at read refusal",
    }
    assert details["completion_estimate"]["state"] == "unknown"  # type: ignore[index]
    assert details["convergence_rate"]["value"] is None  # type: ignore[index]


def test_schema_version_details_keep_measured_versions() -> None:
    refusal = SchemaVersionMismatchError(
        "schema mismatch", current_version=12, expected_version=13, generation_id="gen-1"
    )

    details = schema_refusal_details(refusal)

    assert details["actual_version"] == 12
    assert details["expected_version"] == 13
    assert details["expected_identity"] is None
    assert details["actual_identity"] is None


def test_operation_refusal_is_degraded_and_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.operations import daemon_execution

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise SchemaSkew("index", "expected-identity", "actual-identity")

    monkeypatch.setattr(daemon_execution, "open_operation_read", refuse)
    request = DaemonOperationRequest.from_dict(
        {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": "status",
            "payload": {},
            "archive_root": str(tmp_path),
            "request_id": "request-schema-skew",
        }
    )

    daemon_runtime = SimpleNamespace(publication_guard=lambda: nullcontext())
    context = OperationContext(
        tmp_path,
        MutationPrincipal("test-daemon", frozenset({"read"}), "daemon"),
        "daemon",
        daemon_runtime,
    )
    envelope = daemon_execution.execute_operation(request, context).to_dict()

    assert envelope["outcome"] == "degraded"
    assert envelope["readiness"] == {
        "ready": False,
        "state": "degraded",
        "degraded_components": ["derived_schema:index"],
    }
    assert envelope["progress"]["state"] == "degraded"  # type: ignore[index]
    assert envelope["progress"]["expected_identity"] == "expected-identity"  # type: ignore[index]
    assert envelope["error"]["code"] == "schema_skew"  # type: ignore[index]
    assert envelope["error"]["data"]["affected_tier"] == "index"  # type: ignore[index]


def test_fast_health_marks_same_version_identity_mismatch_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "index.db"
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION

    with sqlite3.connect(index) as connection:
        connection.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
        connection.execute("CREATE TABLE schema_identity (tier TEXT PRIMARY KEY, identity TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_identity VALUES ('index', 'stale-identity')")

    monkeypatch.setattr("polylogue.daemon.health._active_health_db_path", lambda: index)
    alert = _check_schema_version_fast()

    assert alert.severity == HealthSeverity.CRITICAL
    assert "index.db:identity" in alert.message
    assert "stale-identity" in alert.message


def test_status_identity_probe_names_stale_derived_tier(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE schema_identity (tier TEXT PRIMARY KEY, identity TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_identity VALUES ('index', 'stale-identity')")

    assert _derived_identity_mismatches({"index": index, "ops": tmp_path / "ops.db"}) == ["index"]


def test_derived_tier_refusal_reaches_daemon_status_and_readiness(tmp_path: Path) -> None:
    """polylogue-b5l AC2: a refused derived tier is visible in status, not only the log.

    ``schema_refusal_status_component`` had no caller anywhere, so a rebuilding
    derived tier reached the read path's typed refusal and nothing else. The
    ops tier is the reachable case: it is attached without identity validation,
    so status observes the skew instead of failing to open.

    Anti-vacuity: drop the ``_derived_tier_refusals`` wiring from
    ``produce_direct_status`` (or let it report the tier as merely
    version-mismatched) and the ``derived:ops`` component, the
    ``derived_degradation`` evidence, and the false ``ok`` all disappear.
    """

    from polylogue.operations.daemon_status import produce_direct_status
    from polylogue.operations.operation_context import open_operation_read
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "ops.db") as connection:
        expected = str(connection.execute("SELECT identity FROM schema_identity WHERE tier = 'ops'").fetchone()[0])
        connection.execute("UPDATE schema_identity SET identity = 'stale-identity' WHERE tier = 'ops'")

    with open_operation_read(tmp_path) as pinned:
        payload = produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000)

    degradation = payload["derived_degradation"]
    assert isinstance(degradation, list) and len(degradation) == 1
    details = degradation[0]
    assert details["code"] == "schema_skew"
    assert details["affected_tier"] == "ops"
    assert details["expected_identity"] == expected
    assert details["actual_identity"] == "stale-identity"
    assert details["route"] == "daemon_convergence"
    # The refusal never fabricates progress it could not measure.
    assert details["completion_estimate"]["state"] == "unknown"

    components = payload["component_readiness"]
    assert isinstance(components, dict)
    component = components["derived:ops"]
    assert component["state"] == "degraded"
    assert component["repair_hint"] == "daemon convergence"
    assert payload["ok"] is False


def test_healthy_derived_tiers_report_no_refusal(tmp_path: Path) -> None:
    """The refusal branch must stay silent on an intact archive.

    Anti-vacuity: report every tier as refused and this fails, proving the test
    above does not pass for a blanket reason.
    """

    from polylogue.operations.daemon_status import produce_direct_status
    from polylogue.operations.operation_context import open_operation_read
    from tests.infra.archive_templates import bootstrap_archive_root

    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        payload = produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000)

    assert payload["derived_degradation"] == []
    components = payload["component_readiness"]
    assert isinstance(components, dict)
    assert not [name for name in components if name.startswith("derived:")]
