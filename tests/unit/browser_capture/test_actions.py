"""Provider-neutral browser action receiver contracts."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import cast

import pytest

from polylogue.browser_capture import actions as browser_actions
from polylogue.browser_capture.actions import (
    BrowserActionConflictError,
    BrowserActionLeaseError,
    BrowserActionQuotaError,
    claim_action,
    enqueue_action,
    get_action,
    read_action_attachment,
    reconcile_action,
    update_action,
)
from polylogue.browser_capture.models import (
    BrowserActionAttachmentInput,
    BrowserActionPresentation,
    BrowserActionReceipt,
    BrowserActionReconcileRequest,
    BrowserActionRequest,
    BrowserActionTarget,
    BrowserActionUpdateRequest,
)
from polylogue.browser_capture.route_contracts import (
    BROWSER_CAPTURE_ROUTE_CONTRACTS,
    browser_capture_route_contract_for,
)
from polylogue.browser_capture.server import make_server
from tests.infra.frozen_clock import FrozenClock

pytestmark = pytest.mark.frozen_clock_modules("polylogue.browser_capture.actions")
_RECEIVER_ID = "rx-browser-action-test"
_ORIGIN = "chrome-extension://polylogue-browser-action-test"


def _request(
    *,
    action_id: str = "action-1",
    idempotency_key: str = "iteration-1",
    operation: str = "conversation.create",
    conversation_id: str = "new",
    conversation_url: str | None = None,
    project_ref: str | None = None,
    submit_policy: str = "submit_once",
    text: str = "Perform the requested analysis.",
) -> BrowserActionRequest:
    return BrowserActionRequest(
        action_id=action_id,
        idempotency_key=idempotency_key,
        provider="chatgpt",
        operation=operation,  # type: ignore[arg-type]
        target=BrowserActionTarget(
            conversation_id=conversation_id,
            conversation_url=conversation_url,
            project_ref=project_ref,
        ),
        text=text,
        attachments=[
            BrowserActionAttachmentInput(
                name="context.txt",
                mime_type="text/plain",
                content_base64=base64.b64encode(b"exact context").decode(),
            )
        ],
        presentation=BrowserActionPresentation(
            model_slug="gpt-5-6-pro",
            model_label="GPT-5.6 Sol",
            effort_label="Pro",
        ),
        submit_policy=submit_policy,  # type: ignore[arg-type]
    )


def _receipt(
    action_id: str,
    *,
    receiver_id: str = _RECEIVER_ID,
    extension: str = "extension-one",
) -> BrowserActionReceipt:
    return BrowserActionReceipt(
        action_id=action_id,
        receiver_id=receiver_id,
        extension_instance_id=extension,
        provider_conversation_id="conversation-1",
        provider_conversation_url="https://chatgpt.com/c/conversation-1",
        provider_turn_id="turn-user-1",
        observed_surface="Chat",
        observed_model="GPT-5.6 Sol",
        observed_effort="Pro",
        provider_evidence={"current_node": "turn-assistant-1"},
        observed_at="2026-07-16T00:01:00+00:00",
    )


def test_enqueue_hash_pins_inputs_and_is_idempotent(tmp_path: Path) -> None:
    first = enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    repeated = enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)

    assert repeated == first
    assert first.receiver_id == _RECEIVER_ID
    assert first.contract == "polylogue.browser-actions/v1"
    assert first.submit_policy == "submit_once"
    assert len(first.request_sha256) == 64
    attachment, content = read_action_attachment(
        first.action_id,
        first.attachments[0].attachment_id,
        spool_path=tmp_path,
    ) or pytest.fail("missing action attachment")
    assert attachment.sha256 == first.attachments[0].sha256
    assert content == b"exact context"

    with pytest.raises(BrowserActionConflictError, match="different input"):
        enqueue_action(
            _request(text="Different request."),
            receiver_id=_RECEIVER_ID,
            spool_path=tmp_path,
        )


def test_enqueue_rejects_receiver_reserved_action_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        enqueue_action(
            _request(action_id="capabilities"),
            receiver_id=_RECEIVER_ID,
            spool_path=tmp_path,
        )


def test_action_quota_bounds_active_work_without_discarding_terminal_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(browser_actions, "ACTION_MAX_ACTIVE", 1)
    first = enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    with pytest.raises(BrowserActionQuotaError, match="active"):
        enqueue_action(
            _request(action_id="action-2", idempotency_key="iteration-2"),
            receiver_id=_RECEIVER_ID,
            spool_path=tmp_path,
        )

    claimed = claim_action("extension-one", spool_path=tmp_path) or pytest.fail("action was not claimed")
    completed = update_action(
        first.action_id,
        BrowserActionUpdateRequest(
            owner_instance_id=claimed.lease_owner or "",
            outcome="submitted",
            phase="submitted",
            receipt=_receipt(first.action_id),
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing completed action")
    assert completed.receipt and completed.receipt.provider_turn_id == "turn-user-1"

    second = enqueue_action(
        _request(action_id="action-2", idempotency_key="iteration-2"),
        receiver_id=_RECEIVER_ID,
        spool_path=tmp_path,
    )
    assert second.status == "queued"
    assert get_action(first.action_id, spool_path=tmp_path) == completed


def test_capabilities_fail_closed_for_unsupported_presentation_and_target(tmp_path: Path) -> None:
    with pytest.raises(BrowserActionConflictError, match="presentation"):
        enqueue_action(
            _request().model_copy(
                update={
                    "presentation": BrowserActionPresentation(
                        model_slug="default",
                        model_label="Auto",
                        effort_label="Medium",
                    )
                }
            ),
            receiver_id=_RECEIVER_ID,
            spool_path=tmp_path,
        )
    with pytest.raises(BrowserActionConflictError, match="target URL"):
        enqueue_action(
            _request(conversation_url="https://example.com/c/new"),
            receiver_id=_RECEIVER_ID,
            spool_path=tmp_path,
        )


def test_reply_and_project_target_are_explicit(tmp_path: Path) -> None:
    action = enqueue_action(
        _request(
            operation="conversation.reply",
            conversation_id="conversation-1",
            conversation_url="https://chatgpt.com/g/g-p-project/c/conversation-1?tab=chats",
            project_ref="g-p-project",
        ),
        receiver_id=_RECEIVER_ID,
        spool_path=tmp_path,
    )
    assert action.operation == "conversation.reply"
    assert action.target.project_ref == "g-p-project"


def test_pre_submit_lease_is_replaceable_but_submit_intent_is_quarantined(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    first = claim_action("extension-one", spool_path=tmp_path, lease_seconds=30)
    assert first is not None and first.lease_owner == "extension-one"
    frozen_clock.advance(31)
    replacement = claim_action("extension-two", spool_path=tmp_path, lease_seconds=30)
    assert replacement is not None and replacement.lease_owner == "extension-two"

    submitted_intent = update_action(
        replacement.action_id,
        BrowserActionUpdateRequest(
            owner_instance_id="extension-two",
            outcome="progress",
            phase="submit_intent",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing submit intent")
    original_intent_at = submitted_intent.submit_intent_at
    original_expiry = submitted_intent.lease_expires_at
    frozen_clock.advance(120)
    renewed = update_action(
        replacement.action_id,
        BrowserActionUpdateRequest(
            owner_instance_id="extension-two",
            outcome="progress",
            phase="submit_intent",
            detail="lease heartbeat",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing renewed submit intent")
    assert renewed.submit_intent_at == original_intent_at
    assert renewed.lease_expires_at != original_expiry
    frozen_clock.advance(181)
    assert claim_action("extension-three", spool_path=tmp_path) is None
    quarantined = get_action(replacement.action_id, spool_path=tmp_path) or pytest.fail("missing action")
    assert quarantined.status == "outcome_unknown"
    assert quarantined.lease_owner is None


def test_lease_owner_receipt_and_explicit_uncertainty_reconciliation(tmp_path: Path) -> None:
    action = enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    claimed = claim_action("extension-one", spool_path=tmp_path) or pytest.fail("action was not claimed")
    with pytest.raises(BrowserActionLeaseError):
        update_action(
            action.action_id,
            BrowserActionUpdateRequest(owner_instance_id="extension-two", phase="preparing"),
            spool_path=tmp_path,
        )
    unknown = update_action(
        action.action_id,
        BrowserActionUpdateRequest(
            owner_instance_id=claimed.lease_owner or "",
            outcome="outcome_unknown",
            phase="outcome_unknown",
            detail="submit channel ended without a receipt",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing action")
    assert unknown.status == "outcome_unknown"

    reconciled = reconcile_action(
        action.action_id,
        BrowserActionReconcileRequest(
            resolution="submitted",
            detail="provider conversation and user turn inspected",
            receipt=_receipt(action.action_id),
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing action")
    assert reconciled.status == "submitted"
    assert reconciled.receipt and reconciled.receipt.provider_turn_id == "turn-user-1"
    retried = update_action(
        action.action_id,
        BrowserActionUpdateRequest(
            owner_instance_id="extension-one",
            outcome="submitted",
            phase="submitted",
            receipt=_receipt(action.action_id),
        ),
        spool_path=tmp_path,
    )
    assert retried == reconciled


@contextmanager
def _receiver(tmp_path: Path) -> Iterator[tuple[str, int]]:
    server = make_server("127.0.0.1", 0, spool_path=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address[:2])
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _http(
    host: str,
    port: int,
    method: str,
    path: str,
    *,
    body: object | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    connection = HTTPConnection(host, port)
    headers = {"Origin": _ORIGIN}
    payload = json.dumps(body) if body is not None else None
    if payload is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    content = response.read()
    response_headers = {name.lower(): value for name, value in response.getheaders()}
    connection.close()
    return response.status, content, response_headers


def test_http_contract_create_claim_attachment_update_and_read(tmp_path: Path) -> None:
    with _receiver(tmp_path) as (host, port):
        status, content, _ = _http(host, port, "GET", "/v1/browser-actions/capabilities")
        capabilities = json.loads(content)
        assert status == HTTPStatus.OK
        assert capabilities["providers"]["chatgpt"]["submit_policies"] == ["stage_only", "submit_once"]

        status, content, _ = _http(
            host,
            port,
            "POST",
            "/v1/browser-actions",
            body=_request().model_dump(mode="json"),
        )
        created = json.loads(content)["action"]
        assert status == HTTPStatus.ACCEPTED

        status, content, _ = _http(host, port, "GET", "/v1/browser-actions?claim_by=extension-one")
        claimed = json.loads(content)["actions"][0]
        assert status == HTTPStatus.OK
        assert claimed["action_id"] == created["action_id"]

        attachment_id = created["attachments"][0]["attachment_id"]
        status, content, headers = _http(
            host,
            port,
            "GET",
            f"/v1/browser-actions/{created['action_id']}/attachments/{attachment_id}",
        )
        assert status == HTTPStatus.OK
        assert content == b"exact context"
        assert headers["content-type"] == "text/plain"

        status, _, _ = _http(
            host,
            port,
            "POST",
            f"/v1/browser-actions/{created['action_id']}/events",
            body={
                "owner_instance_id": "extension-one",
                "outcome": "submitted",
                "phase": "submitted",
                "receipt": _receipt(
                    created["action_id"],
                    receiver_id=created["receiver_id"],
                ).model_dump(mode="json"),
            },
        )
        assert status == HTTPStatus.OK
        status, content, _ = _http(host, port, "GET", f"/v1/browser-actions/{created['action_id']}")
        assert status == HTTPStatus.OK
        assert json.loads(content)["action"]["receipt"]["provider_turn_id"] == "turn-user-1"


def test_route_contracts_cover_every_browser_action_route() -> None:
    kinds = {contract.kind for contract in BROWSER_CAPTURE_ROUTE_CONTRACTS}
    assert {
        "browser_action_capabilities",
        "browser_action_enqueue",
        "browser_action_list_claim",
        "browser_action_read",
        "browser_action_attachment",
        "browser_action_update",
        "browser_action_reconcile",
    } <= kinds
    assert browser_capture_route_contract_for("GET", "/v1/browser-actions/action-1") is not None
    assert browser_capture_route_contract_for("POST", "/v1/browser-actions/action-1/events") is not None
    assert browser_capture_route_contract_for("POST", "/v1/browser-actions/action-1/reconcile") is not None
