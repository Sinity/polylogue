"""Provider-neutral browser action receiver contracts."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import cast

import pytest
from click.testing import CliRunner

from polylogue.browser_capture import actions as browser_actions
from polylogue.browser_capture.actions import (
    BrowserActionConflictError,
    BrowserActionLeaseError,
    BrowserActionQuotaError,
    claim_action,
    decide_action_approval,
    enqueue_action,
    get_action,
    open_action_attachment,
    reconcile_action,
    store_action_attachment,
    update_action,
)
from polylogue.browser_capture.models import (
    BrowserActionApprovalDecisionRequest,
    BrowserActionAttachmentInput,
    BrowserActionPresentation,
    BrowserActionReceipt,
    BrowserActionReconcileRequest,
    BrowserActionRequest,
    BrowserActionTarget,
    BrowserActionUpdateRequest,
)
from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig
from polylogue.browser_capture.route_contracts import (
    BROWSER_CAPTURE_ROUTE_CONTRACTS,
    browser_capture_route_contract_for,
)
from polylogue.browser_capture.server import make_server
from polylogue.core import durable_fs
from tests.infra.frozen_clock import FrozenClock

pytestmark = pytest.mark.frozen_clock_modules("polylogue.browser_capture.actions")
_RECEIVER_ID = "rx-browser-action-test"
_ORIGIN = "chrome-extension://polylogue-browser-action-test"


@pytest.fixture(autouse=True)
def staged_context_input(tmp_path: Path) -> None:
    store_action_attachment(io.BytesIO(b"exact context").read, 13, spool_path=tmp_path)


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
                attachment_ref=hashlib.sha256(b"exact context").hexdigest(),
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
    with open_action_attachment(first.action_id, first.attachments[0].attachment_id, spool_path=tmp_path) as result:
        assert result is not None
        attachment, stream = result
        assert attachment.sha256 == first.attachments[0].sha256
        assert stream.read() == b"exact context"

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


def test_free_tier_chatgpt_presentation_is_a_second_supported_capability(tmp_path: Path) -> None:
    # Most accounts (including plain-tier ones) never see the paid Pro
    # composer surface at all; the receiver must not require it as the sole
    # supported presentation, or the conduit only works for a minority of
    # real accounts.
    action = enqueue_action(
        _request().model_copy(
            update={
                "presentation": BrowserActionPresentation(
                    surface="chat",
                    model_slug="chatgpt-auto",
                    model_label="ChatGPT",
                    effort_label="Standard",
                )
            }
        ),
        receiver_id=_RECEIVER_ID,
        spool_path=tmp_path,
    )
    assert action.presentation.model_label == "ChatGPT"
    assert action.presentation.effort_label == "Standard"


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


def test_submit_once_reply_is_held_for_explicit_operator_approval(tmp_path: Path) -> None:
    # polylogue-yyvg.7: a submit_once conversation.reply posts one real,
    # provider-visible turn into an EXISTING conversation with no automatic
    # undo, so it must never be automatically claimable straight off enqueue.
    action = enqueue_action(
        _request(operation="conversation.reply", conversation_id="conversation-1"),
        receiver_id=_RECEIVER_ID,
        spool_path=tmp_path,
    )
    assert action.status == "awaiting_approval"
    assert action.phase == "awaiting_approval"
    assert action.requires_operator_approval is True
    assert action.approval_reason == "destructive_submit"
    assert action.approval_requested_at is not None
    assert action.approved_at is None
    assert action.declined_at is None
    assert claim_action("extension-one", spool_path=tmp_path) is None


def test_submit_once_create_is_not_held_for_approval(tmp_path: Path) -> None:
    # A fresh conversation is lower-consequence than replying into an
    # existing one the operator may not be watching -- only conversation.reply
    # is gated (regression guard for the default request fixture, which is
    # conversation.create + submit_once and must remain immediately claimable).
    action = enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    assert action.status == "queued"
    assert action.requires_operator_approval is False
    assert action.approval_reason is None
    claimed = claim_action("extension-one", spool_path=tmp_path)
    assert claimed is not None and claimed.action_id == action.action_id


def test_stage_only_reply_is_not_held_for_approval(tmp_path: Path) -> None:
    # A staged draft never submits to the provider on its own -- only
    # submit_once replies carry the un-undoable consequence that needs an
    # explicit decision.
    action = enqueue_action(
        _request(operation="conversation.reply", conversation_id="conversation-1", submit_policy="stage_only"),
        receiver_id=_RECEIVER_ID,
        spool_path=tmp_path,
    )
    assert action.status == "queued"
    assert action.requires_operator_approval is False


def test_approving_a_held_action_makes_it_claimable(tmp_path: Path) -> None:
    action = enqueue_action(
        _request(operation="conversation.reply", conversation_id="conversation-1"),
        receiver_id=_RECEIVER_ID,
        spool_path=tmp_path,
    )
    decided = decide_action_approval(
        action.action_id,
        BrowserActionApprovalDecisionRequest(
            extension_instance_id="extension-one",
            decision="approve",
            detail="operator reviewed the reply text and target conversation",
        ),
        spool_path=tmp_path,
    )
    assert decided is not None
    assert decided.status == "queued"
    assert decided.approved_at is not None
    assert decided.approved_by == "extension-one"
    # The historical fact that this went through approval is preserved even
    # though the live gate (status) has moved on.
    assert decided.requires_operator_approval is True
    assert decided.approval_reason == "destructive_submit"

    claimed = claim_action("extension-two", spool_path=tmp_path)
    assert claimed is not None and claimed.action_id == action.action_id


def test_declining_a_held_action_is_terminal_and_never_claimable(tmp_path: Path) -> None:
    action = enqueue_action(
        _request(operation="conversation.reply", conversation_id="conversation-1"),
        receiver_id=_RECEIVER_ID,
        spool_path=tmp_path,
    )
    decided = decide_action_approval(
        action.action_id,
        BrowserActionApprovalDecisionRequest(
            extension_instance_id="extension-one",
            decision="decline",
            detail="wrong conversation target",
        ),
        spool_path=tmp_path,
    )
    assert decided is not None
    assert decided.status == "cancelled"
    assert decided.declined_at is not None
    assert claim_action("extension-one", spool_path=tmp_path) is None
    # A second decision against an already-decided action is a conflict, not
    # a silent no-op -- it must land on the exact hold it was made against.
    with pytest.raises(BrowserActionConflictError):
        decide_action_approval(
            action.action_id,
            BrowserActionApprovalDecisionRequest(extension_instance_id="extension-one", decision="approve"),
            spool_path=tmp_path,
        )


def test_deciding_on_an_action_that_was_never_held_is_a_conflict(tmp_path: Path) -> None:
    action = enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    assert action.status == "queued"
    with pytest.raises(BrowserActionConflictError):
        decide_action_approval(
            action.action_id,
            BrowserActionApprovalDecisionRequest(extension_instance_id="extension-one", decision="approve"),
            spool_path=tmp_path,
        )


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
    preserved = update_action(
        replacement.action_id,
        BrowserActionUpdateRequest(
            owner_instance_id="extension-two",
            outcome="progress",
            phase="preparing",
            detail="late pre-submit heartbeat",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing preserved submit intent")
    assert preserved.status == "submit_intent"
    assert preserved.phase == "submit_intent"
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


@pytest.mark.parametrize(
    "outcome",
    ["provider_warning", "rate_limited", "safety_locked", "auth_challenge", "capability_mismatch", "provider_drift"],
)
def test_typed_provider_failures_block_and_release_the_lease(outcome: str, tmp_path: Path) -> None:
    action = enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    claimed = claim_action("extension-one", spool_path=tmp_path) or pytest.fail("action was not claimed")

    blocked = update_action(
        action.action_id,
        BrowserActionUpdateRequest(
            owner_instance_id=claimed.lease_owner or "",
            outcome=outcome,  # type: ignore[arg-type]
            phase="provider_action_failed",
            detail=f"typed {outcome} fixture",
            retry_after_seconds=120 if outcome == "rate_limited" else None,
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing blocked action")

    assert blocked.status == "blocked"
    assert blocked.failure_kind == outcome
    assert blocked.last_error == f"typed {outcome} fixture"
    assert blocked.lease_owner is None
    assert blocked.lease_expires_at is None
    assert blocked.retry_after_seconds == (120 if outcome == "rate_limited" else None)

    # A blocked action is a terminal disposition, not silently revivable by a
    # different owner racing in — this is AC2's "a terminal job cannot revive
    # it" applied to the action ledger, not just the extension's capture path.
    assert claim_action("extension-two", spool_path=tmp_path) is None

    # Idempotent resubmit of the identical outcome/detail is safe (the client
    # may retry its own POST after a network blip) and returns the same
    # terminal record without raising or mutating state.
    replayed = update_action(
        action.action_id,
        BrowserActionUpdateRequest(
            owner_instance_id="extension-two",
            outcome=outcome,  # type: ignore[arg-type]
            phase="provider_action_failed",
            detail=f"typed {outcome} fixture",
        ),
        spool_path=tmp_path,
    )
    assert replayed == blocked

    # A conflicting outcome/detail on the same terminal action is rejected,
    # not silently overwritten.
    with pytest.raises(BrowserActionConflictError):
        update_action(
            action.action_id,
            BrowserActionUpdateRequest(
                owner_instance_id="extension-two",
                outcome=outcome,  # type: ignore[arg-type]
                phase="provider_action_failed",
                detail="a different failure detail",
            ),
            spool_path=tmp_path,
        )


def test_reply_receipt_must_match_the_requested_conversation(tmp_path: Path) -> None:
    action = enqueue_action(
        _request(operation="conversation.reply", conversation_id="conversation-target"),
        receiver_id=_RECEIVER_ID,
        spool_path=tmp_path,
    )
    assert action.status == "awaiting_approval"
    decide_action_approval(
        action.action_id,
        BrowserActionApprovalDecisionRequest(extension_instance_id="extension-one", decision="approve"),
        spool_path=tmp_path,
    )
    claimed = claim_action("extension-one", spool_path=tmp_path) or pytest.fail("action was not claimed")
    wrong_receipt = _receipt(action.action_id).model_copy(
        update={
            "provider_conversation_id": "conversation-other",
            "provider_conversation_url": "https://chatgpt.com/c/conversation-other",
        }
    )
    with pytest.raises(BrowserActionConflictError, match="conversation mismatch"):
        update_action(
            action.action_id,
            BrowserActionUpdateRequest(
                owner_instance_id=claimed.lease_owner or "",
                outcome="submitted",
                phase="submitted",
                receipt=wrong_receipt,
            ),
            spool_path=tmp_path,
        )


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


def test_http_contract_approval_gates_a_destructive_reply_until_the_operator_decides(tmp_path: Path) -> None:
    with _receiver(tmp_path) as (host, port):
        status, content, _ = _http(
            host,
            port,
            "POST",
            "/v1/browser-actions",
            body=_request(operation="conversation.reply", conversation_id="conversation-1").model_dump(mode="json"),
        )
        assert status == HTTPStatus.ACCEPTED
        created = json.loads(content)["action"]
        assert created["status"] == "awaiting_approval"
        assert created["requires_operator_approval"] is True
        assert created["approval_reason"] == "destructive_submit"

        # Not claimable while the decision is pending.
        status, content, _ = _http(host, port, "GET", "/v1/browser-actions?claim_by=extension-one")
        assert status == HTTPStatus.OK
        assert json.loads(content)["actions"] == []

        status, content, _ = _http(
            host,
            port,
            "POST",
            f"/v1/browser-actions/{created['action_id']}/approval",
            body={"extension_instance_id": "extension-one", "decision": "approve"},
        )
        assert status == HTTPStatus.OK
        approved = json.loads(content)["action"]
        assert approved["status"] == "queued"
        assert approved["approved_by"] == "extension-one"

        status, content, _ = _http(host, port, "GET", "/v1/browser-actions?claim_by=extension-one")
        assert status == HTTPStatus.OK
        assert json.loads(content)["actions"][0]["action_id"] == created["action_id"]

        # Deciding again on an already-decided action is a conflict.
        status, content, _ = _http(
            host,
            port,
            "POST",
            f"/v1/browser-actions/{created['action_id']}/approval",
            body={"extension_instance_id": "extension-one", "decision": "decline"},
        )
        assert status == HTTPStatus.CONFLICT
        assert json.loads(content)["error"] == "browser_action_approval_conflict"


def test_http_contract_rejects_noncanonical_action_ids(tmp_path: Path) -> None:
    with _receiver(tmp_path) as (host, port):
        for method, path, body in (
            ("GET", "/v1/browser-actions/foo@bar", None),
            ("GET", "/v1/browser-actions/foo@bar/attachments/attachment-1", None),
            (
                "POST",
                "/v1/browser-actions/foo@bar/events",
                {"owner_instance_id": "extension-one", "outcome": "progress", "phase": "preparing"},
            ),
        ):
            status, content, _ = _http(host, port, method, path, body=body)
            assert status == HTTPStatus.BAD_REQUEST
            assert json.loads(content)["error"] == "invalid_browser_action_id"


def test_attachment_download_headers_are_safe_for_unicode_metadata(tmp_path: Path) -> None:
    request = _request().model_copy(
        update={
            "attachments": [
                BrowserActionAttachmentInput(
                    name="résumé.zip",
                    mime_type="text/plain; charset=utf-8",
                    attachment_ref=store_action_attachment(io.BytesIO(b"safe bytes").read, 10, spool_path=tmp_path),
                )
            ]
        }
    )
    with _receiver(tmp_path) as (host, port):
        status, content, _ = _http(
            host,
            port,
            "POST",
            "/v1/browser-actions",
            body=request.model_dump(mode="json"),
        )
        assert status == HTTPStatus.ACCEPTED
        created = json.loads(content)["action"]
        attachment_id = created["attachments"][0]["attachment_id"]
        status, content, headers = _http(
            host,
            port,
            "GET",
            f"/v1/browser-actions/{created['action_id']}/attachments/{attachment_id}",
        )
        assert status == HTTPStatus.OK
        assert content == b"safe bytes"
        assert headers["content-type"] == "application/octet-stream"
        assert "filename*=UTF-8''r%C3%A9sum%C3%A9.zip" in headers["content-disposition"]


def test_route_contracts_cover_every_browser_action_route() -> None:
    kinds = {contract.kind for contract in BROWSER_CAPTURE_ROUTE_CONTRACTS}
    assert {
        "browser_action_capabilities",
        "browser_action_enqueue",
        "browser_action_list_claim",
        "browser_action_read",
        "browser_action_attachment",
        "browser_action_attachment_upload",
        "browser_action_update",
        "browser_action_reconcile",
        "browser_action_approval",
    } <= kinds
    assert browser_capture_route_contract_for("PUT", "/v1/browser-action-attachments") is not None
    assert browser_capture_route_contract_for("GET", "/v1/browser-actions/action-1") is not None
    assert browser_capture_route_contract_for("POST", "/v1/browser-actions/action-1/events") is not None
    assert browser_capture_route_contract_for("POST", "/v1/browser-actions/action-1/reconcile") is not None
    assert browser_capture_route_contract_for("POST", "/v1/browser-actions/action-1/approval") is not None


def test_attachment_upload_streams_above_old_limit_and_pins_http_action(tmp_path: Path) -> None:
    size = 17 * 1024 * 1024 + 7
    digest = hashlib.sha256()
    chunk = b"n" * (64 * 1024)
    remaining = size
    with _receiver(tmp_path) as (host, port):
        connection = HTTPConnection(host, port)
        connection.putrequest("PUT", "/v1/browser-action-attachments")
        connection.putheader("Origin", _ORIGIN)
        connection.putheader("Content-Length", str(size))
        connection.endheaders()
        while remaining:
            part = chunk[: min(remaining, len(chunk))]
            connection.send(part)
            digest.update(part)
            remaining -= len(part)
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == HTTPStatus.CREATED
        connection.close()
        assert payload == {"attachment_ref": digest.hexdigest(), "size_bytes": size}
        request = _request().model_copy(
            update={
                "attachments": [
                    BrowserActionAttachmentInput(
                        name="neutral.bin",
                        attachment_ref=payload["attachment_ref"],
                    )
                ]
            }
        )
        status, content, _ = _http(host, port, "POST", "/v1/browser-actions", body=request.model_dump(mode="json"))
        assert status == HTTPStatus.ACCEPTED
        action = json.loads(content)["action"]
        item = action["attachments"][0]
        connection = HTTPConnection(host, port)
        connection.request(
            "GET",
            f"/v1/browser-actions/{action['action_id']}/attachments/{item['attachment_id']}",
            headers={"Origin": _ORIGIN},
        )
        response = connection.getresponse()
        assert response.status == HTTPStatus.OK
        assert int(response.getheader("Content-Length") or "-1") == size
        observed = hashlib.sha256()
        count = 0
        while part := response.read(64 * 1024):
            observed.update(part)
            count += len(part)
        connection.close()
        assert (count, observed.hexdigest()) == (size, digest.hexdigest())
    # Cold action/read and the independently retained upload have exact bytes.
    cold = get_action(action["action_id"], spool_path=tmp_path)
    assert cold is not None
    assert cold.attachments[0].size_bytes == size
    assert (tmp_path / "browser-actions" / ".inputs" / digest.hexdigest()).stat().st_size == size


def test_attachment_input_reads_are_bounded_and_partial_input_is_never_acknowledged(tmp_path: Path) -> None:
    requests: list[int] = []
    remaining = 17 * 1024 * 1024

    def read(size: int) -> bytes:
        nonlocal remaining
        requests.append(size)
        count = min(size, remaining)
        remaining -= count
        return b"x" * count

    reference = store_action_attachment(read, remaining, spool_path=tmp_path)
    assert len(reference) == 64
    assert max(requests) <= browser_actions.ACTION_ATTACHMENT_CHUNK_BYTES
    assert remaining == 0
    with pytest.raises(ValueError, match="incomplete"):
        store_action_attachment(io.BytesIO(b"short").read, 6, spool_path=tmp_path)
    inputs = tmp_path / "browser-actions" / ".inputs"
    assert list(inputs.glob(".upload-*")) == []
    assert not (inputs / hashlib.sha256(b"short").hexdigest()).exists()
    empty = store_action_attachment(io.BytesIO().read, 0, spool_path=tmp_path)
    assert empty == hashlib.sha256(b"").hexdigest()


def test_attachment_upload_refuses_inline_predecessor_and_missing_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        BrowserActionAttachmentInput.model_validate({"name": "neutral.txt", "content_base64": "YWJj"})
    request = _request().model_copy(
        update={"attachments": [BrowserActionAttachmentInput(name="neutral.txt", attachment_ref="00" * 32)]}
    )
    with pytest.raises(FileNotFoundError):
        enqueue_action(request, receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    assert get_action("action-1", spool_path=tmp_path) is None


def test_attachment_upload_sync_failure_never_returns_a_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_fsync(_descriptor: int) -> None:
        raise OSError(errno.EIO, "synthetic disk failure")

    with _receiver(tmp_path) as (host, port):
        monkeypatch.setattr(os, "fsync", fail_fsync)
        connection = HTTPConnection(host, port)
        connection.request("PUT", "/v1/browser-action-attachments", body=b"neutral", headers={"Origin": _ORIGIN})
        response = connection.getresponse()
        assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR
        assert json.loads(response.read()) == {
            "ok": False,
            "receiver": "polylogue-browser-capture",
            "schema_version": 1,
            "error": "attachment_storage_unavailable",
        }
        connection.close()
    inputs = tmp_path / "browser-actions" / ".inputs"
    assert list(inputs.glob(".upload-*")) == []
    assert not (inputs / hashlib.sha256(b"neutral").hexdigest()).exists()


def test_attachment_directory_entries_settle_before_upload_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "new" / "spool"
    events: list[tuple[str, Path]] = []
    original_sync = durable_fs.sync_directory
    original_link = os.link

    def sync(path: Path) -> None:
        original_sync(path)
        events.append(("sync", path))

    def link(source: Path, target: Path) -> None:
        original_link(source, target)
        events.append(("link", target))

    monkeypatch.setattr(browser_actions, "sync_directory", sync)
    monkeypatch.setattr(os, "link", link)
    reference = store_action_attachment(io.BytesIO(b"neutral").read, 7, spool_path=root)
    target = root / "browser-actions" / ".inputs" / reference
    published = events.index(("link", target))
    for parent in [tmp_path, root.parent, root, root / "browser-actions"]:
        assert ("sync", parent) in events[:published]
    assert ("sync", target.parent) in events[published + 1 :]
    events.clear()
    assert store_action_attachment(io.BytesIO(b"neutral").read, 7, spool_path=root) == reference
    assert ("sync", target.parent) in events


def test_attachment_download_retains_verified_original_inode(tmp_path: Path) -> None:
    action = enqueue_action(_request(), receiver_id=_RECEIVER_ID, spool_path=tmp_path)
    item = action.attachments[0]
    path = tmp_path / "browser-actions" / action.action_id / "attachments" / item.attachment_id
    with open_action_attachment(action.action_id, item.attachment_id, spool_path=tmp_path) as result:
        assert result is not None
        _, stream = result
        replacement = path.with_suffix(".new")
        replacement.write_bytes(b"different bytes")
        replacement.replace(path)
        assert stream.read() == b"exact context"
    assert stream.closed
    with pytest.raises(BrowserActionConflictError):
        with open_action_attachment(action.action_id, item.attachment_id, spool_path=tmp_path):
            pass


def test_action_cli_streams_the_original_open_file_into_reference_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.daemon import browser_capture as capture_cli

    path = tmp_path / "neutral.bin"
    size = 17 * 1024 * 1024
    with path.open("wb") as stream:
        for _ in range(size // 65536):
            stream.write(b"n" * 65536)

    def refuse_whole_read(_path: Path) -> bytes:
        raise AssertionError("whole-file input read")

    monkeypatch.setattr(Path, "read_bytes", refuse_whole_read)
    monkeypatch.setattr(browser_actions, "browser_capture_spool_root", lambda: tmp_path)
    monkeypatch.setattr(BrowserCaptureReceiverConfig, "default", lambda: SimpleNamespace(spool_path=tmp_path))
    monkeypatch.setattr(capture_cli, "receiver_identity", lambda _config: _RECEIVER_ID)
    result = CliRunner().invoke(
        capture_cli.action_command,
        [
            "--provider",
            "chatgpt",
            "--text",
            "Neutral request",
            "--attachment",
            str(path),
            "--model-slug",
            "gpt-5-6-pro",
            "--model-label",
            "GPT-5.6 Sol",
            "--effort-label",
            "Pro",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    action = json.loads(result.output)
    assert action["attachments"][0]["size_bytes"] == size
    assert path.stat().st_size == size
    assert (tmp_path / "browser-actions" / ".inputs" / action["attachments"][0]["sha256"]).stat().st_size == size


def test_attachment_directory_retry_rechecks_preexisting_unsynced_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "retry" / "spool"
    original = durable_fs.sync_directory
    failed = False
    observed: list[Path] = []

    def sync(path: Path) -> None:
        nonlocal failed
        if path == tmp_path and not failed:
            failed = True
            raise OSError(errno.EIO, "synthetic ancestor fault")
        original(path)
        observed.append(path)

    monkeypatch.setattr(browser_actions, "sync_directory", sync)
    with pytest.raises(OSError):
        store_action_attachment(io.BytesIO(b"neutral").read, 7, spool_path=root)
    assert (root / "browser-actions" / ".inputs").exists()
    observed.clear()
    reference = store_action_attachment(io.BytesIO(b"neutral").read, 7, spool_path=root)
    assert tmp_path in observed
    assert (root / "browser-actions" / ".inputs" / reference).exists()


def test_local_action_enqueue_does_not_publish_receiver_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from polylogue.browser_capture.receiver import load_or_mint_receiver_token
    from polylogue.daemon import browser_capture as capture_cli

    current = load_or_mint_receiver_token()

    def refuse_startup(*args: object, **kwargs: object) -> None:
        raise AssertionError("local enqueue must not publish receiver credentials")

    monkeypatch.setattr(capture_cli, "resolve_receiver_auth_token", refuse_startup)
    arguments = [
        "--provider",
        "chatgpt",
        "--text",
        "Neutral action",
        "--model-slug",
        "gpt-5-6-pro",
        "--model-label",
        "GPT-5.6 Sol",
        "--effort-label",
        "Pro",
        "--format",
        "json",
    ]
    result = CliRunner().invoke(capture_cli.action_command, arguments)
    assert result.exit_code == 0, result.output
    assert load_or_mint_receiver_token() == current
    refused = CliRunner().invoke(capture_cli.action_command, [*arguments, "--auth-token", "neutral-other-token"])
    assert refused.exit_code == 2
    assert load_or_mint_receiver_token() == current
