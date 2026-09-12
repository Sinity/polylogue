from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.errors import SchemaSkew, SchemaVersionMismatchError
from polylogue.daemon.derived_degradation import schema_refusal_details
from polylogue.daemon.health import HealthSeverity, _check_schema_version_fast
from polylogue.daemon.status import _derived_identity_mismatches
from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL, DaemonOperationRequest
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

    envelope = daemon_execution.execute_operation(request, OperationContext.direct_read(tmp_path)).to_dict()

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
