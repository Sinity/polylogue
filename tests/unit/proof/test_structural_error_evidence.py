from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from polylogue.config import get_config
from polylogue.lib.json import JSONDocument, require_json_document
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.proof.catalog import build_verification_catalog
from polylogue.proof.models import ProofObligation
from polylogue.proof.runners import (
    ErrorContextObservation,
    MaintenanceRepairObservation,
    QuarantineErrorObservation,
    run_artifact_path_evidence,
    run_error_context_evidence,
    run_maintenance_repair_state_evidence,
    run_quarantine_error_evidence,
)
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.repair import count_orphaned_messages_sync, repair_orphaned_messages
from polylogue.storage.runtime import RawConversationRecord
from polylogue.surfaces.payloads import MachineErrorPayload
from tests.infra.storage_records import ConversationBuilder, db_setup

PRIVATE_PAYLOAD_FRAGMENT = "private payload transcript fragment"


def _obligation(claim_id: str, *, subject_id: str | None = None) -> ProofObligation:
    catalog = build_verification_catalog()
    for obligation in catalog.obligations:
        if obligation.claim.id != claim_id:
            continue
        if subject_id is not None and obligation.subject.id != subject_id:
            continue
        return obligation
    raise AssertionError(f"missing obligation for claim={claim_id!r} subject={subject_id!r}")


def test_artifact_path_runner_emits_structural_closure_evidence() -> None:
    obligation = _obligation(
        "artifact.path.dependency_closure",
        subject_id="artifact.path.raw-session-product-repair-loop",
    )

    envelope = run_artifact_path_evidence(obligation)

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["path_name"] == "raw-session-product-repair-loop"
    assert envelope.evidence["missing_dependencies"] == []
    nodes = envelope.evidence["nodes"]
    assert isinstance(nodes, list)
    assert {"raw_validation_state", "archive_conversation_rows", "session_product_rows"}.issubset(nodes)
    layers = envelope.evidence["layers"]
    assert isinstance(layers, dict)
    assert "source" in set(layers.values())
    assert "durable" in set(layers.values())
    assert "derived" in set(layers.values())
    assert "index" in set(layers.values())
    assert "projection" in set(layers.values())
    assert envelope.counterexample is None


def test_maintenance_repair_runner_exercises_synthetic_archive_transition(
    workspace_env: Mapping[str, Path],
) -> None:
    db_path = _archive_with_orphaned_messages(workspace_env)
    config = get_config()

    before_count = _orphaned_message_count(db_path)
    preview = repair_orphaned_messages(config, dry_run=True)
    after_dry_run_count = _orphaned_message_count(db_path)
    repaired = repair_orphaned_messages(config, dry_run=False)
    after_count = _orphaned_message_count(db_path)
    second_repair = repair_orphaned_messages(config, dry_run=False)

    observation = MaintenanceRepairObservation(
        target="orphaned_messages",
        before_count=before_count,
        preview_repaired_count=preview.repaired_count,
        after_dry_run_count=after_dry_run_count,
        repaired_count=repaired.repaired_count,
        after_count=after_count,
        second_repair_count=second_repair.repaired_count,
        state_effect="changed",
        destructive=repaired.destructive,
        result_success=repaired.success,
        failure_state=None,
        operation="polylogue doctor --cleanup --target orphaned_messages",
    )
    obligation = _obligation(
        "maintenance.repair.crash_consistency",
        subject_id="maintenance.target.orphaned_messages",
    )

    envelope = run_maintenance_repair_state_evidence(obligation, observation)

    assert before_count == 2
    assert preview.repaired_count == 2
    assert after_dry_run_count == 2
    assert repaired.repaired_count == 2
    assert after_count == 0
    assert second_repair.repaired_count == 0
    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["target"] == "orphaned_messages"
    assert envelope.evidence["state_effect"] == "changed"
    assert envelope.counterexample is None


def test_maintenance_repair_runner_rejects_ambiguous_failure_state() -> None:
    obligation = _obligation(
        "maintenance.repair.crash_consistency",
        subject_id="maintenance.target.orphaned_messages",
    )
    observation = MaintenanceRepairObservation(
        target="orphaned_messages",
        before_count=2,
        preview_repaired_count=2,
        after_dry_run_count=2,
        repaired_count=0,
        after_count=2,
        second_repair_count=0,
        state_effect="ambiguous",
        destructive=True,
        result_success=False,
        failure_state=None,
        operation="polylogue doctor --cleanup --target orphaned_messages",
    )

    envelope = run_maintenance_repair_state_evidence(obligation, observation)

    assert envelope.status is OutcomeStatus.ERROR
    assert envelope.counterexample is not None


def test_maintenance_repair_runner_accepts_explicit_rollback_failure_state() -> None:
    obligation = _obligation(
        "maintenance.repair.crash_consistency",
        subject_id="maintenance.target.orphaned_messages",
    )
    observation = MaintenanceRepairObservation(
        target="orphaned_messages",
        before_count=2,
        preview_repaired_count=2,
        after_dry_run_count=2,
        repaired_count=0,
        after_count=2,
        second_repair_count=0,
        state_effect="rolled_back",
        destructive=True,
        result_success=False,
        failure_state="rolled_back",
        operation="polylogue doctor --cleanup --target orphaned_messages",
    )

    envelope = run_maintenance_repair_state_evidence(obligation, observation)

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["failure_state"] == "rolled_back"
    assert envelope.counterexample is None


def test_quarantine_runner_preserves_context_without_payload_leak(
    workspace_env: Mapping[str, Path],
) -> None:
    record = _make_raw_record(
        _codex_malformed_jsonl_bytes(),
        provider="codex",
        path="/exports/codex-session.jsonl",
    )
    result = ingest_record(record, str(workspace_env["archive_root"]), "strict")
    parse_error = result.parse_error or result.error or ""
    machine_payload = _machine_error_payload(
        code="parser.quarantine",
        message="Raw payload was quarantined",
        details={
            "provider": "codex",
            "source_path": record.source_path,
            "raw_id": record.raw_id,
            "parse_error": parse_error,
        },
    )
    observation = QuarantineErrorObservation(
        provider="codex",
        source_path=record.source_path,
        raw_id=record.raw_id,
        parse_error=parse_error,
        machine_payload=machine_payload,
        user_message=f"codex raw {record.source_path} was quarantined as {record.raw_id}",
        payload_fragments=(PRIVATE_PAYLOAD_FRAGMENT,),
        validation_status=result.validation_status,
    )
    obligation = _obligation(
        "parser.quarantine.context_redaction",
        subject_id="error.surface.parser_quarantine",
    )

    envelope = run_quarantine_error_evidence(obligation, observation)

    assert result.error is not None
    assert parse_error
    assert PRIVATE_PAYLOAD_FRAGMENT not in parse_error
    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["provider"] == "codex"
    assert envelope.evidence["source_path"] == record.source_path
    assert envelope.evidence["payload_leak_detected"] is False
    assert envelope.counterexample is None


def test_error_context_runner_checks_machine_and_user_context() -> None:
    machine_payload = _machine_error_payload(
        code="maintenance.failed",
        message="Maintenance repair failed",
        details={
            "target": "orphaned_messages",
            "state_effect": "rolled_back",
            "operation": "polylogue doctor --cleanup --target orphaned_messages",
        },
    )
    observation = ErrorContextObservation(
        error_family="maintenance-failure",
        machine_payload=machine_payload,
        user_message=(
            "Maintenance target orphaned_messages failed; state_effect=rolled_back; "
            "operation=polylogue doctor --cleanup --target orphaned_messages"
        ),
        payload_fragments=(PRIVATE_PAYLOAD_FRAGMENT,),
    )
    obligation = _obligation(
        "error.machine_user_context",
        subject_id="error.surface.maintenance_failure",
    )

    envelope = run_error_context_evidence(obligation, observation)

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["user_context_checks"] == {
        "target": True,
        "state_effect": True,
        "operation": True,
    }
    assert envelope.evidence["privacy_checks"] == {
        "payload_fragments_redacted": True,
        "payload_leak_detected": False,
    }
    assert envelope.counterexample is None


def _archive_with_orphaned_messages(workspace_env: Mapping[str, Path]) -> Path:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "healthy-1").provider("chatgpt").title("Healthy").add_message(
        role="user",
        text="A valid message",
    ).save()

    with open_connection(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version, "
            "provider_name, word_count, has_tool_use, has_thinking) "
            "VALUES ('orphan-m1', 'deleted-conv', 'user', 'orphan text', 'oh1', 1, 'test', 2, 0, 0)"
        )
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version, "
            "provider_name, word_count, has_tool_use, has_thinking) "
            "VALUES ('orphan-m2', 'deleted-conv', 'assistant', 'orphan reply', 'oh2', 1, 'test', 2, 0, 0)"
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")
    return db_path


def _orphaned_message_count(db_path: Path) -> int:
    with open_connection(db_path) as conn:
        return count_orphaned_messages_sync(conn)


def _make_raw_record(content: bytes, *, provider: str, path: str) -> RawConversationRecord:
    raw_id, size = get_blob_store().write_from_bytes(content)
    now = datetime.now(timezone.utc).isoformat()
    return RawConversationRecord(
        raw_id=raw_id,
        provider_name=provider,
        source_name="proof-quarantine-fixture",
        source_path=path,
        source_index=None,
        blob_size=size,
        acquired_at=now,
        file_mtime=now,
    )


def _codex_malformed_jsonl_bytes() -> bytes:
    meta = b'{"type":"session_meta","payload":{"id":"session-x","timestamp":"2025-01-01T00:00:00Z"}}'
    good = (
        b'{"type":"message","id":"msg-1","role":"user",'
        b'"timestamp":"2025-01-01T00:00:01Z",'
        b'"content":[{"type":"input_text","text":"ping"}]}'
    )
    bad = (
        b'{"type":"message","id":"msg-2","role":"assistant",'
        b'"timestamp":"2025-01-01T00:00:02Z",'
        b'"content":[{"type":"output_text","text":"' + PRIVATE_PAYLOAD_FRAGMENT.encode() + b'"}]'
    )
    return meta + b"\n" + good + b"\n" + bad + b"\n"


def _machine_error_payload(*, code: str, message: str, details: Mapping[str, object]) -> JSONDocument:
    return require_json_document(
        MachineErrorPayload(code=code, message=message, details=details).to_dict(),
        context="proof machine error payload",
    )
