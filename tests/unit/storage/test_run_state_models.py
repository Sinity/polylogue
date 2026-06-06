from __future__ import annotations

import pytest

from polylogue.storage.run_state import DriftBucket, PlanCounts, PlanResult, RunCounts, RunDrift
from polylogue.types import PlanStage


def test_plan_result_coerces_stage_sequence_counts_details_and_cursors() -> None:
    result = PlanResult.model_validate(
        {
            "timestamp": 123,
            "stage": "parse",
            "stage_sequence": ("acquire", "parse"),
            "counts": {"scan": 1, "store_raw": 2, "validate": 3, "parse": 4, "materialize": 5, "index": 6},
            "details": {
                "new_raw": 1,
                "existing_raw": 2,
                "duplicate_raw": 3,
                "backlog_validate": 4,
                "backlog_parse": 5,
                "preview_invalid": 6,
                "preview_skipped_no_schema": 7,
            },
            "sources": ["inbox"],
            "cursors": {
                "inbox": {
                    "file_count": 3,
                    "error_count": 2,
                    "failed_count": 1,
                    "latest_mtime": 1,
                    "latest_file_name": "latest.json",
                    "latest_path": "/tmp/latest.json",
                    "latest_file_id": "drive-id",
                    "latest_error": "bad",
                    "latest_error_file": "bad.json",
                    "failed_files": [{"path": "bad.json", "error": "bad"}, {"path": 1}, "bad"],
                },
                7: {"file_count": 1},
                "invalid": "cursor",
            },
        }
    )

    assert result.stage is PlanStage.PARSE
    assert result.stage_sequence == [PlanStage.ACQUIRE, PlanStage.PARSE]
    assert result.counts.to_payload()["validate"] == 3
    assert result.details.to_payload()["preview_skipped_no_schema"] == 7
    assert result.cursors["inbox"]["file_count"] == 3
    assert result.cursors["inbox"]["latest_mtime"] == 1.0
    assert result.cursors["inbox"]["failed_files"] == [{"path": "bad.json", "error": "bad"}]
    assert (
        PlanResult.model_validate(
            {"timestamp": 1, "stage": "all", "stage_sequence": [], "sources": [], "cursors": "bad"}
        ).cursors
        == {}
    )
    assert (
        PlanResult.model_validate(
            {"timestamp": 1, "stage": "all", "stage_sequence": None, "sources": []}
        ).stage_sequence
        == []
    )


def test_int_payload_models_expose_mapping_contracts() -> None:
    counts = PlanCounts(scan=1, store_raw=2, validate=3, index=4)

    assert counts.to_dict() == {"scan": 1, "store_raw": 2, "validate": 3, "index": 4}
    assert counts["scan"] == 1
    assert counts.get("missing", 9) == 9
    assert list(counts.items()) == [("scan", 1), ("store_raw", 2), ("validate", 3), ("index", 4)]
    assert list(counts.keys()) == ["scan", "store_raw", "validate", "index"]
    assert list(counts.values()) == [1, 2, 3, 4]
    assert "scan" in counts
    assert counts == {"scan": 1, "store_raw": 2, "validate": 3, "index": 4}
    assert counts.int_value("index") == 4
    with pytest.raises(KeyError):
        counts["missing"]


def test_run_counts_and_drift_payloads_cover_all_fields() -> None:
    run_counts = RunCounts(
        sessions=1,
        messages=2,
        attachments=3,
        skipped_sessions=4,
        skipped_messages=5,
        skipped_attachments=6,
        acquired=7,
        skipped=8,
        acquire_errors=9,
        validated=10,
        validation_invalid=11,
        validation_drift=12,
        validation_skipped_no_schema=13,
        validation_errors=14,
        materialized=15,
        parse_failures=16,
        schemas_generated=17,
        schemas_failed=18,
        new_sessions=19,
        changed_sessions=20,
    )
    assert run_counts.to_payload()["changed_sessions"] == 20

    drift = RunDrift.model_validate(
        {
            "new": DriftBucket(sessions=1, messages=2, attachments=3),
            "removed": {"sessions": 4, "messages": 5, "attachments": 6},
            "changed": {"sessions": 7, "messages": 8, "attachments": 9},
        }
    )
    assert drift.to_payload()["new"] == {"sessions": 1, "messages": 2, "attachments": 3}
    assert drift["removed"].messages == 5
    assert drift.get("missing") is None
    assert [(name, bucket.sessions) for name, bucket in drift.items()] == [
        ("new", 1),
        ("removed", 4),
        ("changed", 7),
    ]
    assert list(drift.keys()) == ["new", "removed", "changed"]
    assert [bucket.sessions for bucket in drift.values()] == [1, 4, 7]
    assert "changed" in drift
    assert drift == drift.to_payload()
    with pytest.raises(KeyError):
        drift["missing"]
