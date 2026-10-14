"""Actual daemon ingest retains productive conflicts across repeated delivery."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.daemon.api_auth import resolve_api_auth_token
from polylogue.daemon.services import ServiceCapability, ServiceProfile
from polylogue.operations.audit import AuditRepository
from polylogue.operations.ingest_acceptance import INGEST_OPERATION
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.machine_receipts import IngestHistoricalReceiptV2
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.daemon_service_harness import ServiceHarness
from tests.infra.retained_parser_payloads import _chatgpt_session


@pytest.mark.timeout(300)
async def test_daemon_first_and_duplicate_ingest_publish_rows_and_report_conflict(tmp_path: Path) -> None:
    source = tmp_path / "conversations.json"
    source.write_text(
        json.dumps(
            [
                _chatgpt_session("conflict", "first"),
                _chatgpt_session("conflict", "second"),
                _chatgpt_session("conflict", "second"),
            ]
        )
    )
    root = tmp_path / "archive"
    await run_archive_fixture_write(root, lambda: bootstrap_archive_root(root))
    archive = Polylogue(archive_root=root, db_path=root / "index.db")
    harness = ServiceHarness(profile=ServiceProfile.SURFACES, capabilities={ServiceCapability.API})
    try:
        api = harness.api_server(root)
        uds = harness.uds_server(root, api_server=api, auth_token=resolve_api_auth_token(None))
        harness.start_server("api_server", api)
        harness.start_server("uds_server", uds)
        first = await archive.parse_file(source, source_name="neutral-occurrence")
        repeated = await archive.parse_file(source, source_name="neutral-occurrence")
        assert first.counts["sessions"] == first.changed_counts["sessions"] == 1
        assert first.counts["messages"] == first.changed_counts["messages"] == 1
        assert repeated.counts["sessions"] == repeated.changed_counts["sessions"] == 0
        assert first.counts["membership_conflicts"] == repeated.counts["membership_conflicts"] == 1
        assert first.parse_failures == repeated.parse_failures == 0
    finally:
        try:
            await harness.close()
        finally:
            await archive.close()
    with sqlite3.connect(root / "audit.db") as connection:
        operations = [
            str(row[0])
            for row in connection.execute(
                "SELECT operation_id FROM operation_runs WHERE operation_name=? ORDER BY rowid", (INGEST_OPERATION,)
            )
        ]
        connection.row_factory = sqlite3.Row
        requests = [dict(row) for row in connection.execute("SELECT * FROM machine_requests ORDER BY rowid")]
    assert len(operations) == 2
    audit = AuditRepository.for_archive_root(root)
    with audit.settled_machine_read():
        assert [machine_request_state(audit, request)["outcome"] for request in requests] == ["degraded", "degraded"]
        histories: list[IngestHistoricalReceiptV2] = []
        for operation in operations:
            history = audit.historical_machine_receipt(operation)
            assert isinstance(history, IngestHistoricalReceiptV2)
            histories.append(history)
    for history in histories:
        assert history.summary.conflicting_membership_count == 1
        assert history.summary.refused_membership_count == 0
        assert history.summary.unresolved_raw_count == 0
        assert not history.summary.source_complete
        assert history.summary.profile_convergence_complete
    assert histories[0].summary.changed_session_count == 1
    assert histories[1].summary.changed_session_count == 0
