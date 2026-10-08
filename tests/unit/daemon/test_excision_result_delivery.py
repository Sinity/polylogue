"""Original Excision product transfer through the resident operation owner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.operation_runtime import _Exchange
from polylogue.daemon_client import DaemonOperationProtocolError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import readonly_connection_context
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.live_ingest import prepared_live_convergence_owner


def _seed_target(root: Path) -> None:
    """Acquire genuine original Codex bytes, then use canonical retained replay."""
    records = [
        {"type": "session_meta", "payload": {"id": "receipt-target", "timestamp": "2026-06-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "receipt-message",
                "role": "user",
                "content": [{"type": "input_text", "text": "neutral receipt target"}],
            },
        },
    ]
    data = "".join(json.dumps(record) + "\n" for record in records).encode()
    with write_lease("test.excision-result-acquisition", archive_root=root):
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=data,
                source_path="neutral/receipt-target.jsonl",
                canonical_source_path="neutral/receipt-target.jsonl",
                acquired_at_ms=1,
                native_id="receipt-target",
            )

    async def replay() -> None:
        async with prepared_live_convergence_owner(root) as owner:
            (await owner.replay_retained_raw_ids((raw_id,))).require_complete()

    asyncio.run(replay())


@pytest.mark.parametrize("found", [False, True])
def test_resident_excision_retains_and_delivers_original_complete_receipt(tmp_path: Path, found: bool) -> None:
    root = tmp_path / "archive"
    with running_daemon_operations(root, seed_archive=_seed_target if found else None) as daemon:
        result = daemon.client.operation_to_completion(
            "mutation.session.excision",
            {
                "session_id": "codex-session:receipt-target",
                "reason": "neutral removal",
                "actor": "user:local",
                "confirm": True,
            },
            archive_root=str(root),
        )
        assert result is not None
        assert result["outcome"] == "completed", result
        payload = result["result"]
        document = payload["result_document"]
        page = daemon.client.operation("operation.result", {**document, "offset": 0}, archive_root=str(root))
        assert page is not None and page["outcome"] == "completed", page
        data = b"".join(daemon.client.iter_operation_result(document, archive_root=str(root)))
        assert len(data) == document["byte_length"]
        assert hashlib.sha256(data).hexdigest() == document["sha256"]
        receipt = json.loads(data)
        assert data == json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        assert receipt["found"] is found
        assert receipt["session_id"] == "codex-session:receipt-target"
        assert receipt["reason"] == "neutral removal"
        assert receipt["actor"] == "user:local"
        for field in (
            "removed_blob_hashes",
            "shared_blob_hashes",
            "marker_input_digests",
            "cascaded_session_ids",
            "retained_hook_events",
            "retained_source_containers",
        ):
            assert payload["result"][field + "_count"] == len(receipt[field])
            assert field not in payload["result"]
        assert bool(receipt["removed_blob_hashes"]) is found
        assert payload["effect"] == ("committed" if found else "no-effect")
        assert payload["receipt_ref"].startswith("mutation-operation:")
        altered = {**document, "sha256": "0" * 64}
        with pytest.raises(DaemonOperationProtocolError) as refused:
            list(daemon.client.iter_operation_result(altered, archive_root=str(root)))
        assert refused.value.outcome == "rejected"
        assert refused.value.error_code == "operation_result_document_identity_mismatch"
        assert not daemon.session_exists("codex-session:receipt-target")


@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_original_document_delivery_fault_preserves_completed_mutation(tmp_path: Path, fault: str) -> None:
    root = tmp_path / "archive"
    with running_daemon_operations(root) as daemon:
        result = daemon.client.operation_to_completion(
            "mutation.session.excision",
            {"session_id": "codex-session:absent", "reason": "neutral removal", "actor": "user:local", "confirm": True},
            archive_root=str(root),
        )
        assert result is not None and result["outcome"] == "completed", result
        assert result["result"]["effect"] == "no-effect"
        document = result["result"]["result_document"]
        path = daemon.runtime._terminal_path(document["request_id"])
        assert path is not None
        product = path.with_suffix(".product.json")
        if fault == "missing":
            product.unlink()
        else:
            product.write_bytes(b"invalid")
        with pytest.raises(DaemonOperationProtocolError) as refused:
            list(daemon.client.iter_operation_result(document, archive_root=str(root)))
        assert refused.value.outcome == "rejected"
        assert refused.value.error_code == "operation_result_document_" + (
            "missing" if fault == "missing" else "corrupt"
        )
        assert result["result"]["receipt_ref"].startswith("mutation-operation:")


def test_terminal_metadata_transfer_failure_keeps_original_bound_document_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    with running_daemon_operations(root) as daemon:
        original = daemon.runtime._retain_terminal
        reached: list[str] = []

        def fail_transfer_once(exchange: _Exchange) -> None:
            if not reached:
                reached.append("terminal-transfer")
                raise OSError("synthetic terminal metadata fault")
            original(exchange)

        monkeypatch.setattr(daemon.runtime, "_retain_terminal", fail_transfer_once)
        result = daemon.client.operation_to_completion(
            "mutation.session.excision",
            {"session_id": "codex-session:absent", "reason": "neutral removal", "actor": "user:local", "confirm": True},
            archive_root=str(root),
        )
        assert result is not None and result["outcome"] == "completed", result
        assert reached == ["terminal-transfer"]
        document = result["result"]["result_document"]
        data = b"".join(daemon.client.iter_operation_result(document, archive_root=str(root)))
        assert hashlib.sha256(data).hexdigest() == document["sha256"]
        assert len(data) == document["byte_length"]
        assert json.loads(data)["found"] is False


def test_pre_document_sink_failure_preserves_original_deletion_and_terminal_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    root = tmp_path / "archive"
    with running_daemon_operations(root, seed_archive=_seed_target) as daemon:
        original = tempfile.mkstemp
        reached: list[str] = []

        def fail_product_file(
            suffix: str | None = None, prefix: str | None = None, dir: str | None = None, text: bool = False
        ) -> tuple[int, str]:
            if prefix == ".product-":
                reached.append("original-product-sink")
                raise OSError("synthetic product creation fault")
            return original(suffix=suffix, prefix=prefix, dir=dir, text=text)

        monkeypatch.setattr(tempfile, "mkstemp", fail_product_file)
        result = daemon.client.operation(
            "mutation.session.excision",
            {
                "session_id": "codex-session:receipt-target",
                "reason": "neutral removal",
                "actor": "user:local",
                "confirm": True,
            },
            archive_root=str(root),
        )
        assert reached == ["original-product-sink"]
        assert result is not None and result["outcome"] == "indeterminate", result
        assert result["error"]["code"] == "OSError"
        assert not daemon.session_exists("codex-session:receipt-target")
        status = daemon.client.operation(
            "operation.status", {"request_id": result["request_id"]}, archive_root=str(root)
        )
        assert status is not None and status["outcome"] == "completed", status
        assert status["result"]["outcome"] == "indeterminate"
        assert status["result"]["error"]["code"] == "OSError"
        assert "result_document" not in status["result"]
        with readonly_connection_context(root / "audit.db") as audit:
            original_attempts = audit.execute(
                "SELECT r.operation_id,a.attempt_id,a.state,a.unknown_reason FROM operation_runs r "
                "JOIN operation_attempts a USING(operation_id) WHERE r.operation_name='mutate-session-excision'"
            ).fetchall()
        assert len(original_attempts) == 1
        assert original_attempts[0][0] and original_attempts[0][1]
        assert original_attempts[0][2] == "unknown"
        assert original_attempts[0][3] == "actuator exception after durable intent"
