"""Daemon transport and result adaptation for public archive mutations."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from polylogue.config import Config, active_archive_root
from polylogue.core.errors import PolylogueError


class FacadeDaemonRequiredError(PolylogueError):
    """The archive's resident writer is unavailable for a public mutation."""

    code = "daemon_required"


@dataclass(frozen=True)
class FacadeProductReceipt:
    status: str
    affected_count: int
    domain_receipt: dict[str, Any]


@dataclass(frozen=True)
class FacadeProductPlan:
    context: dict[str, Any]


_EXISTING_OPERATIONS = {
    "add_mark": "user.mark.add",
    "remove_mark": "user.mark.remove",
    "save_annotation": "user.annotation.save",
    "delete_annotation": "user.annotation.delete",
    "delete_view": "user.saved_view.delete",
    "capture_assertion_candidate": "mutation.assertion.candidate.capture",
}


def _wire_request(product: str, fields: dict[str, Any]) -> tuple[str, dict[str, object]]:
    if product in {"add_mark", "remove_mark"}:
        return _EXISTING_OPERATIONS[product], {
            "session_id": fields["owner_session_id"],
            "target_type": fields["target_type"],
            "target_id": fields["target_id"],
            "mark_type": fields["mark_type"],
        }
    if product == "save_annotation":
        return "user.annotation.save", {
            "annotation_id": fields["annotation_id"],
            "session_id": fields["owner_session_id"],
            "note_text": fields["note_text"],
            "target_type": fields["target_type"],
            "target_id": fields["target_id"],
        }
    if product == "delete_annotation":
        return "user.annotation.delete", {"id": fields["annotation_id"]}
    if product == "delete_view":
        return "user.saved_view.delete", {"id": fields["view_id"]}
    if product == "capture_assertion_candidate":
        return "mutation.assertion.candidate.capture", {
            "body_text": fields["body_text"],
            "kind": fields["kind"].value,
            "refs": list(fields["refs"]),
            "scope_refs": list(fields["scope_refs"]),
            "cwd": None if fields["cwd"] is None else str(fields["cwd"]),
            "author_ref": fields["author_ref"],
            "author_kind": fields["author_kind"],
            "idempotency_key": fields["idempotency_key"],
            "ttl_seconds": fields["ttl_seconds"],
        }
    return f"mutation.facade.{product}", dict(fields)


def _adapt_existing(product: str, state: dict[str, Any]) -> tuple[FacadeProductReceipt, FacadeProductPlan]:
    result = state.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"daemon {product} returned no mutation result")
    affected = int(state.get("affected_count") or 0)
    domain: dict[str, Any] = {}
    if product == "save_annotation":
        domain["created"] = result.get("detail") is None
    elif product == "capture_assertion_candidate":
        domain["claim"] = result
    status = "applied" if affected else "no_op"
    return FacadeProductReceipt(status, affected, domain), FacadeProductPlan({})


async def submit_facade_product(
    config: Config, product: str, **fields: Any
) -> tuple[FacadeProductReceipt, FacadeProductPlan]:
    """Submit once and preserve an indeterminate accepted write as an error."""
    operation, payload = _wire_request(product, fields)
    state = await submit_facade_operation(config, operation, payload)
    if product in _EXISTING_OPERATIONS:
        return _adapt_existing(product, state)
    result = state.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"daemon {product} returned no product receipt")
    receipt = FacadeProductReceipt(
        status=str(result["status"]),
        affected_count=int(result["affected_count"]),
        domain_receipt=dict(result.get("domain_receipt") or {}),
    )
    plan = FacadeProductPlan(dict(result.get("plan_context") or {}))
    return receipt, plan


async def submit_facade_operation(config: Config, operation: str, payload: dict[str, object]) -> dict[str, Any]:
    """Submit a declared daemon write and return its validated product result."""
    from polylogue.daemon.api_auth import resolve_api_auth_token
    from polylogue.daemon.socket_path import daemon_socket_path
    from polylogue.daemon_client import DaemonClient
    from polylogue.operations.daemon_protocol import daemon_operation_spec

    spec = daemon_operation_spec(operation)
    if spec is None:
        raise ValueError(f"daemon mutation operation is not declared: {operation}")
    root = active_archive_root(config)
    client = DaemonClient(
        daemon_socket_path(root),
        timeout_s=spec.deadline_s,
        auth_token=lambda: resolve_api_auth_token(
            getattr(config, "api_auth_token", None),
            allow_no_auth=getattr(config, "api_allow_no_auth", False),
        ),
    )
    from polylogue.operations.daemon_errors import DaemonMutationIndeterminateError

    request_id = uuid.uuid4().hex
    pending = asyncio.create_task(
        asyncio.to_thread(
            client.operation_to_completion,
            operation,
            payload,
            archive_root=str(root),
            request_id=request_id,
        )
    )
    try:
        envelope = await asyncio.shield(pending)
    except asyncio.CancelledError as exc:
        # The worker thread may have sent the write before this task was
        # cancelled. Its durable request ID remains available for recovery.
        pending.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        raise DaemonMutationIndeterminateError(method="POST", path="/api/operation", request_id=request_id) from exc
    if envelope is None:
        raise FacadeDaemonRequiredError(f"start `polylogued run` for {root} to apply {operation}")
    if envelope.get("outcome") == "rejected":
        from polylogue.operations.archive_mutation import (
            MutationBlockedError,
            MutationTargetVanishedError,
            SessionNotFoundError,
        )
        from polylogue.operations.daemon_errors import DaemonOperationRejectedError

        error = envelope.get("error")
        error = error if isinstance(error, dict) else {}
        code = str(error.get("code") or "rejected")
        detail = str(error.get("detail") or code)
        if code == "session_not_found":
            raise SessionNotFoundError(str(payload.get("session_id") or detail))
        if code == "mutation_target_vanished":
            raise MutationTargetVanishedError(detail)
        if code == "mutation_blocked":
            raise MutationBlockedError(operation, detail)
        if operation == "mutation.session.tag" and code == "selection_is_stale":
            session_ids = payload.get("session_ids")
            if isinstance(session_ids, list) and len(session_ids) == 1:
                raise SessionNotFoundError(str(session_ids[0]))
        raise DaemonOperationRejectedError(code, detail)
    if envelope.get("outcome") != "completed":
        raise RuntimeError(f"daemon mutation {operation} did not complete: {envelope.get('outcome')}")
    state = envelope.get("result")
    if not isinstance(state, dict):
        raise ValueError(f"daemon {operation} returned no result")
    return state


async def submit_facade_writer(config: Config, name: str, payload: dict[str, object]) -> Any:
    state = await submit_facade_operation(config, f"mutation.facade.{name}", payload)
    result = state.get("result")
    if not isinstance(result, dict) or "value" not in result:
        raise ValueError(f"daemon writer {name} returned no value")
    return result["value"]
