"""Resident descriptive packets keep the complete original evidence population."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from polylogue.operations.daemon_protocol import (
    FablePacketWireRequest,
    FablePacketWireResult,
    OperationResultContractError,
    validate_operation_result,
)
from polylogue.operations.fable_packet_contracts import (
    FablePacketRequest,
    FablePacketResult,
    decode_fable_packet_result,
)
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.delegation_packets import seed_delegations


def test_resident_packet_reads_every_page_on_its_original_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    seed_delegations(root, count=257, annotation_count=257)
    original = ArchiveStore.query_delegations
    returned: list[int] = []
    epochs: list[str] = []
    assertion_returned: list[int] = []
    original_assertions = ArchiveStore.query_assertions

    def record(self: ArchiveStore, *args: Any, **kwargs: Any) -> Any:
        from polylogue.archive.query.transaction import archive_snapshot_epoch

        epochs.append(archive_snapshot_epoch(self))
        rows = original(self, *args, **kwargs)
        returned.append(len(rows))
        return rows

    def record_assertions(self: ArchiveStore, *args: Any, **kwargs: Any) -> Any:
        rows = original_assertions(self, *args, **kwargs)
        assertion_returned.append(len(rows))
        return rows

    monkeypatch.setattr(ArchiveStore, "query_delegations", record)
    monkeypatch.setattr(ArchiveStore, "query_assertions", record_assertions)
    with running_daemon_operations(root) as stack:
        envelope = stack.client.operation(
            "insights.fable_packet", {"seed": "neutral", "requested_size": 1, "schema_id": "neutral.absent"}
        )
        assert envelope is not None and envelope["outcome"] == "completed", envelope
        result = decode_fable_packet_result(envelope["result"])
        assert result.packet.population_count == 257
        assert result.packet.action_observed_count == 257
        assert len(result.packet.selected_refs) == 1
        assert result.packet.manifest is not None
        assert result.packet.manifest.population_count == 257
        assert result.packet.manifest.spec.archive_cursor == epochs[0]
        assert len(set(epochs)) == 1
        assert result.packet.status == "not_supported"
        assert "missing_annotation_schema" in result.packet.not_supported_reasons
        assert result.outcome.state == "degraded"
    assert sum(returned) == 514
    assert returned.count(0) == 2
    assert len([size for size in returned if size]) >= 2
    assert returned[-1] == 0
    assert assertion_returned == [256, 1, 0]


def test_packet_wire_matches_canonical_schema_and_refuses_invalid_nested_fields(tmp_path: Path) -> None:
    assert FablePacketWireRequest.model_json_schema() == FablePacketRequest.model_json_schema()
    assert FablePacketWireResult.model_json_schema() == FablePacketResult.model_json_schema()
    with running_daemon_operations(tmp_path / "archive") as stack:
        envelope = stack.client.operation("insights.fable_packet", {"seed": "neutral", "requested_size": 0})
        assert envelope is not None and envelope["outcome"] == "completed", envelope
        value = envelope["result"]
        validate_operation_result("insights.fable_packet", value)
        assert decode_fable_packet_result(value).packet.population_count == 0
        for changed in ({"population_count": True}, {"unexpected": "value"}):
            with pytest.raises(OperationResultContractError):
                validate_operation_result("insights.fable_packet", {**value, "packet": {**value["packet"], **changed}})


def test_packet_cancellation_reaches_original_population_read(tmp_path: Path) -> None:
    from polylogue.analysis.fable_packet import regenerate_private_fable_packet
    from polylogue.operations.operation_context import open_operation_read

    seed_delegations(tmp_path, count=2)

    class CancelledError(RuntimeError):
        pass

    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise CancelledError()

    with open_operation_read(tmp_path) as pinned:
        with pytest.raises(CancelledError):
            regenerate_private_fable_packet(pinned.archive, seed="neutral", requested_size=1, checkpoint=checkpoint)
    assert calls == 3
