"""Derive machine lifecycle from durable references and existing domain receipts."""

from __future__ import annotations

import json
import math

from polylogue.operations.audit import AuditRepository, MachineRequestBinding
from polylogue.operations.daemon_protocol import AcceptedOperationReference
from polylogue.operations.machine_receipts import encode_machine_receipt

_RICH_HISTORICAL_RECEIPT_OPERATIONS = frozenset({"ingest", "maintenance.insights.rebuild"})


def _operation_requires_rich_historical_receipt(operation_name: object) -> bool:
    """Whether recovery needs a closed domain receipt rather than terminal audit facts.

    Ingest and insight rebuild replies carry bounded, domain-specific results
    that cannot be reconstructed from generic operation counters. Ordinary
    mutations have their terminal effect recorded by ``operation_runs`` and
    the final audit event, so they must not be demoted merely because they do
    not opt into a rich receipt.
    """
    return operation_name in _RICH_HISTORICAL_RECEIPT_OPERATIONS


def _audit_int(value: object, *, field: str) -> int:
    """Reject malformed audit scalars rather than silently coercing receipt facts."""

    if type(value) is not int:
        raise ValueError(f"audit {field} is not an integer")
    return value


def _embedding_terminal_receipt(raw: object) -> dict[str, object] | None:
    """Decode the bounded domain summary stored in the terminal audit event."""
    if not isinstance(raw, str) or not raw.startswith("embedding_receipt:"):
        return None
    try:
        value = json.loads(raw.removeprefix("embedding_receipt:"))
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("operation") != "maintenance.embeddings.backfill":
        return None
    if value.get("outcome") not in {"completed", "stopped", "cancelled", "failed"}:
        return None
    if type(value.get("sequence")) is not int or value["sequence"] < 1:
        return None
    progress = value.get("progress")
    result = value.get("result")
    if not isinstance(progress, dict) or not isinstance(result, dict):
        return None
    for field in ("computed", "failed", "done", "pending"):
        container = progress if field in {"computed", "failed"} else result
        if type(container.get(field)) is not int or container[field] < 0:
            return None
    cost = progress.get("cost_usd")
    if type(cost) not in {int, float} or not math.isfinite(float(cost)) or cost < 0:
        return None
    return value


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
    state: dict[str, object] = {
        "reference": AcceptedOperationReference.from_record(record).to_dict(),
        "sequence": 1,
        "outcome": "completed",
    }
    if kind == "source-generation":
        state["source_generation_id"] = record["artifact_ref"]
    if kind == "source-generation" and not parts:
        # Legacy manifest-only acceptance has no historical execution receipt.
        return {**state, "outcome": "accepted", "effect": "indeterminate"}
    if kind == "insight-preview-pages":
        return {**state, "outcome": "running", "effect": "no-effect", "accepted": False}
    if kind not in {"operation", "execution-batch", "source-generation"}:
        refs = [part["artifact_ref"] for part in parts] or [record["artifact_ref"]]
        state["artifact_refs"] = refs
        if kind == "preview-batch":
            state["result"] = audit.machine_preview_summary(binding)
        elif kind == "authorization-batch":
            state["result"] = {"status": "authorized", "authorization_ref": refs[0], "authorization_refs": refs}
        elif kind == "cancelled-preview-batch":
            state["result"] = {"status": "cancelled", "preview_ref": refs[0], "preview_refs": refs}
        return state
    if kind == "operation":
        parts = [{"ordinal": 0, "operation_id": record["artifact_ref"]}]
    sequence = 1 + (1 if record.get("stop_reason") else 0)
    completed = affected = 0
    attempted: list[dict[str, object]] = []
    unattempted: list[int] = []
    outcomes: list[str] = []
    for part in parts:
        ordinal, operation_id = _audit_int(part["ordinal"], field="part ordinal"), part["operation_id"]
        if operation_id is None:
            unattempted.append(ordinal)
            continue
        run = audit.get_operation(str(operation_id))
        events = audit.list_events(str(operation_id))
        sequence += 1 + (_audit_int(events[-1]["sequence"], field="event sequence") if events else 0)
        if run is None:
            outcome = "indeterminate"
        else:
            affected += _audit_int(run["affected_count"], field="affected count")
            if _audit_int(run["unknown_count"], field="unknown count"):
                outcome = "indeterminate"
            elif run["status"] == "completed":
                completed += 1
                outcome = "completed"
            elif run["status"] == "running":
                outcome = "running" if audit.attempt_owner_liveness(str(operation_id)) == "live" else "indeterminate"
            else:
                outcome = "failed"
        outcomes.append(outcome)
        historical = audit.historical_machine_receipt(str(operation_id)) if outcome == "completed" else None
        # Rich-result operations cannot reconstruct their terminal response
        # from live source/index state. Ordinary mutation effects are already
        # closed by the completed run and final audit event above.
        if (
            outcome == "completed"
            and historical is None
            and run is not None
            and _operation_requires_rich_historical_receipt(run["operation_name"])
        ):
            outcome = "indeterminate"
            outcomes[-1] = outcome
        attempted.append(
            {
                "ordinal": ordinal,
                "operation_id": operation_id,
                "outcome": outcome,
                "receipt": None if historical is None else encode_machine_receipt(historical),
            }
        )
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
    if (
        record.get("operation_name") == "maintenance.embeddings.backfill"
        and record.get("stop_reason")
        and outcome in {"completed", "failed"}
    ):
        outcome = "cancelled" if record["stop_reason"] == "cancelled" else "interrupted"
    result: dict[str, object] | None = None
    if kind == "source-generation" and len(attempted) == 1 and attempted[0]["outcome"] == "completed":
        receipt = attempted[0]["receipt"]
        if isinstance(receipt, dict) and receipt.get("kind") == "ingest/v1":
            result = receipt
    if record.get("operation_name") == "maintenance.embeddings.backfill" and attempted:
        run = audit.get_operation(str(attempted[0]["operation_id"])) if attempted[0]["operation_id"] else None
        result = None if run is None else _embedding_terminal_receipt(run.get("error_summary"))
        if outcome in {"completed", "cancelled", "interrupted", "failed"} and result is None:
            # This route's CLI result carries partial counts and cost data that
            # generic operation counters cannot reconstruct. Missing or
            # malformed audit data must not be presented as a complete result.
            outcome = "indeterminate"
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
        **({"result": result} if result is not None else {}),
        "stop_reason": record.get("stop_reason"),
    }
