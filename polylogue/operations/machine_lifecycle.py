"""Derive machine lifecycle from durable references and existing domain receipts."""

from __future__ import annotations

from polylogue.operations.audit import AuditRepository, MachineRequestBinding


def machine_request_state(audit: AuditRepository, record: dict[str, object]) -> dict[str, object]:
    binding = MachineRequestBinding(
        **{
            key: str(record[key])
            for key in (
                "archive_identity",
                "request_id",
                "principal_ref",
                "fingerprint",
                "operation_name",
            )
        }
    )
    parts = audit.machine_parts(binding)
    kind = str(record["artifact_kind"])
    state: dict[str, object] = {"reference": record, "sequence": 1, "outcome": "completed"}
    if kind not in {"operation", "execution-batch"}:
        refs = [part["artifact_ref"] for part in parts] or [record["artifact_ref"]]
        state["artifact_refs"] = refs
        if kind == "preview-batch":
            state["result"] = audit.machine_preview_summary(binding)
        elif kind == "authorization-batch":
            state["result"] = {"status": "authorized", "authorization_ref": refs[0], "authorization_refs": refs}
        elif kind == "cancelled-preview-batch":
            state["result"] = {"status": "cancelled", "preview_refs": refs}
        return state
    if kind == "operation":
        parts = [{"ordinal": 0, "operation_id": record["artifact_ref"]}]
    sequence = 1 + (1 if record.get("stop_reason") else 0)
    completed = affected = 0
    attempted: list[dict[str, object]] = []
    unattempted: list[int] = []
    outcomes: list[str] = []
    for part in parts:
        ordinal, operation_id = int(part["ordinal"]), part["operation_id"]
        if operation_id is None:
            unattempted.append(ordinal)
            continue
        run = audit.get_operation(str(operation_id))
        events = audit.list_events(str(operation_id))
        sequence += 1 + (int(events[-1]["sequence"]) if events else 0)
        if run is None:
            outcome = "indeterminate"
        else:
            affected += int(run["affected_count"])
            if int(run["unknown_count"]):
                outcome = "indeterminate"
            elif run["status"] == "completed":
                completed += 1
                outcome = "completed"
            elif run["status"] == "running":
                outcome = "running" if audit.attempt_owner_liveness(str(operation_id)) == "live" else "indeterminate"
            else:
                outcome = "failed"
        outcomes.append(outcome)
        attempted.append({"ordinal": ordinal, "operation_id": operation_id, "outcome": outcome, "receipt": run})
    if "indeterminate" in outcomes:
        outcome = "indeterminate"
    elif "running" in outcomes:
        outcome = "running"
    elif "failed" in outcomes:
        outcome = "failed"
    elif record.get("stop_reason"):
        outcome = "cancelled" if record["stop_reason"] == "cancelled" else "failed"
    elif unattempted:
        outcome = "accepted"
    else:
        outcome = "completed"
    return {
        **state,
        "sequence": sequence,
        "outcome": outcome,
        "effect": "indeterminate"
        if outcome in {"indeterminate", "running", "accepted"}
        else ("committed" if affected else "no-effect"),
        "completed_chunks": completed,
        "affected_count": affected,
        "not_attempted": unattempted,
        "parts": attempted,
        "stop_reason": record.get("stop_reason"),
    }
