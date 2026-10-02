"""Machine ingest with an input denominator above the old receipt ceiling."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.daemon.api_auth import resolve_api_auth_token
from polylogue.daemon.services import ServiceCapability, ServiceProfile
from polylogue.operations.audit import AuditRepository
from polylogue.operations.ingest_acceptance import INGEST_OPERATION
from polylogue.operations.ingest_inputs import retain_input_page
from polylogue.operations.machine_receipts import IngestHistoricalReceiptV2
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.source_items import FrozenSourceInput
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.daemon_service_harness import ServiceHarness


@pytest.mark.timeout(300)
async def test_machine_ingest_10241_files_has_complete_historical_pages(tmp_path: Path) -> None:
    """A 10,000-input cap or forty-page receipt limit makes this route fail."""
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    empty_zip = b"PK\x05\x06" + b"\x00" * 18
    for ordinal in range(10_241):
        (source_dir / f"input-{ordinal:05d}.zip").write_bytes(empty_zip)

    archive_root = tmp_path / "archive"
    await run_archive_fixture_write(archive_root, lambda: bootstrap_archive_root(archive_root))
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    harness = ServiceHarness(profile=ServiceProfile.SURFACES, capabilities={ServiceCapability.API})
    try:
        api_server = harness.api_server(archive_root)
        uds_server = harness.uds_server(archive_root, api_server=api_server, auth_token=resolve_api_auth_token(None))
        _api_task = harness.start_server("api_server", api_server)
        _uds_task = harness.start_server("uds_server", uds_server)
        result = await archive.parse_file(source_dir, source_name="machine-ingest")
        assert result.parse_failures == 0
        with sqlite3.connect(archive_root / "audit.db") as audit_conn:
            operation_id = str(
                audit_conn.execute(
                    "SELECT operation_id FROM operation_runs WHERE operation_name=?", (INGEST_OPERATION,)
                ).fetchone()[0]
            )
    finally:
        try:
            await harness.close()
        finally:
            await archive.close()

    audit = AuditRepository.for_archive_root(archive_root)
    with audit.settled_machine_read():
        receipt = audit.historical_machine_receipt(operation_id)
    assert isinstance(receipt, IngestHistoricalReceiptV2)
    assert receipt.input_count == 10_241
    assert receipt.input_page_count == 41
    (archive_root / "source.db").rename(archive_root / "source.unavailable")
    (archive_root / "index.db").unlink(missing_ok=True)
    pages = list(audit.iter_ingest_input_pages(receipt))
    assert len(pages) == 41
    assert sum(len(page.items) for page in pages) == 10_241
    assert pages[0].items[0].logical_coordinate == "input-00000.zip"
    assert pages[-1].items[-1].logical_coordinate == "input-10240.zip"


async def test_later_input_page_failure_aborts_preaccept_source_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File 257 disappearing leaves no prepared rows or GC-protecting receipts."""
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    empty_zip = b"PK\x05\x06" + b"\x00" * 18
    for ordinal in range(257):
        (source_dir / f"input-{ordinal:05d}.zip").write_bytes(empty_zip)

    archive_root = tmp_path / "archive"
    await run_archive_fixture_write(archive_root, lambda: bootstrap_archive_root(archive_root))
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    harness = ServiceHarness(profile=ServiceProfile.SURFACES, capabilities={ServiceCapability.API})
    calls = 0

    def fail_later_page(
        spool: Path,
        *,
        after_coordinate: str | None,
        publisher: ArchiveBlobPublisher,
        check_stop: Callable[[], None],
    ) -> tuple[FrozenSourceInput, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            (source_dir / "input-00256.zip").unlink()
        return retain_input_page(
            spool,
            after_coordinate=after_coordinate,
            publisher=publisher,
            check_stop=check_stop,
        )

    monkeypatch.setattr("polylogue.operations.daemon_ingest.retain_input_page", fail_later_page)
    try:
        api_server = harness.api_server(archive_root)
        uds_server = harness.uds_server(archive_root, api_server=api_server, auth_token=resolve_api_auth_token(None))
        _api_task = harness.start_server("api_server", api_server)
        _uds_task = harness.start_server("uds_server", uds_server)
        with pytest.raises(RuntimeError):
            await archive.parse_file(source_dir, source_name="machine-ingest")
    finally:
        try:
            await harness.close()
        finally:
            await archive.close()

    assert calls == 2
    with sqlite3.connect(archive_root / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM prepared_source_manifests").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM prepared_source_manifest_members").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM source_generations").fetchone() == (0,)
