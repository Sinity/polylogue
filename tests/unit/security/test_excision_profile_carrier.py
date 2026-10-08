"""Source profile receipts follow their exact raw acquisition cascade."""

from pathlib import Path

import pytest

from polylogue.security.excision_carriers import SESSION_CARRIERS, CarrierReach, audit_session_carriers
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.connection_profile import readonly_connection_context
from polylogue.storage.sqlite.managed_connection import sqlite_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root


def test_runtime_source_profile_receipt_has_verified_cascade(tmp_path: Path) -> None:
    with write_lease("test.profile-carrier-runtime", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    with readonly_connection_context(tmp_path / "source.db") as source:
        audit = audit_session_carriers(source)
    assert audit.ok
    assert "raw_profile_identity_receipts" in audit.declared
    assert SESSION_CARRIERS["raw_profile_identity_receipts"].reach is CarrierReach.RAW_CASCADE


@pytest.mark.parametrize("cascade", [True, False])
def test_profile_receipt_exact_baseline_fk_erases_only_target(cascade: bool) -> None:
    # Use the shipped Source baseline table statement; the negative control
    # changes only its FK action, which the live registry must detect.
    start = SOURCE_DDL.index("CREATE TABLE IF NOT EXISTS raw_profile_identity_receipts")
    statement = SOURCE_DDL[start : SOURCE_DDL.index(";", start)]
    if not cascade:
        statement = statement.replace("ON DELETE CASCADE", "ON DELETE NO ACTION")
    with sqlite_connection(":memory:") as source:
        with connection_cursor(source, "PRAGMA foreign_keys=ON"):
            pass
        with connection_cursor(source, "CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY) STRICT"):
            pass
        with connection_cursor(source, statement):
            pass
        audit = audit_session_carriers(source)
        assert audit.misdeclared == (() if cascade else ("raw_profile_identity_receipts",))
        if not cascade:
            return
        for raw_id in ("target", "survivor"):
            with connection_cursor(source, "INSERT INTO raw_sessions(raw_id) VALUES (?)", (raw_id,)):
                pass
            with connection_cursor(
                source,
                "INSERT INTO raw_profile_identity_receipts(raw_id,profile_key) VALUES (?,?)",
                (raw_id, "a" * 12),
            ):
                pass
        with connection_cursor(source, "DELETE FROM raw_sessions WHERE raw_id=?", ("target",)):
            pass
        with connection_cursor(source, "SELECT raw_id,profile_key FROM raw_profile_identity_receipts") as cursor:
            assert [tuple(row) for row in cursor] == [("survivor", "a" * 12)]
        with connection_cursor(source, "PRAGMA foreign_key_check") as cursor:
            assert cursor.fetchone() is None
