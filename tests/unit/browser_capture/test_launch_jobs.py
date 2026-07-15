from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Literal, cast

import pytest
from click.testing import CliRunner

from polylogue.browser_capture.launch_jobs import (
    BrowserLaunchConflictError,
    BrowserLaunchHandoffError,
    BrowserLaunchLeaseError,
    accept_launch_handoff,
    claim_due_launch_job,
    control_launch_job,
    enqueue_launch_job,
    list_launch_jobs,
    read_launch_attachment,
    update_launch_job,
)
from polylogue.browser_capture.models import (
    BrowserLaunchAttachmentInput,
    BrowserLaunchHandoffRequest,
    BrowserLaunchJobControlRequest,
    BrowserLaunchJobRequest,
    BrowserLaunchJobUpdateRequest,
)
from polylogue.browser_capture.server import make_server
from polylogue.browser_capture.sol_pro_prompt import SOL_PRO_PROMPT_PROFILE, sol_pro_prompt_sha256
from polylogue.daemon.cli import main as daemon_cli
from tests.infra.frozen_clock import FrozenClock

_EXTENSION_ORIGIN = "chrome-extension://polylogue-launch-test"
pytestmark = pytest.mark.frozen_clock_modules("polylogue.browser_capture.launch_jobs")


@contextmanager
def _running_receiver(tmp_path: Path) -> Iterator[tuple[str, int]]:
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


def _http(host: str, port: int, method: str, path: str, body: object | None = None) -> HTTPResponse:
    conn = HTTPConnection(host, port)
    payload = json.dumps(body) if body is not None else None
    headers = {"Origin": _EXTENSION_ORIGIN}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    return conn.getresponse()


def _request(
    *,
    not_before: str | None = None,
    job_id: str | None = None,
    cadence_minutes: Literal[1, 5, 15, 30, 60] = 1,
) -> BrowserLaunchJobRequest:
    return BrowserLaunchJobRequest(
        job_title="Implement the durable Sol Pro launch queue",
        scope_prompt="Implement the receiver queue described by the relevant Bead.",
        cadence_minutes=cadence_minutes,
        not_before=not_before,
        job_id=job_id,
        attachments=[
            BrowserLaunchAttachmentInput(
                name="context.tar.gz",
                mime_type="application/gzip",
                content_base64=base64.b64encode(b"project context").decode(),
            )
        ],
    )


def _handoff_zip(
    *,
    corrupt_checksum: bool = False,
    prompt_profile: str | None = SOL_PRO_PROMPT_PROFILE,
) -> bytes:
    files = {
        "README.md": b"read me\n",
        "SUMMARY.md": b"summary\n",
        "VERIFICATION-LIMITS.md": b"not run in Polylogue\n",
        "PATCHES/0001.patch": b"diff --git a/x b/x\n",
        "DESIGN/architecture.md": b"design\n",
        "TESTS/plan.md": b"test plan\n",
    }
    records = [
        {
            "path": name,
            "sha256": "0" * 64 if corrupt_checksum and name == "README.md" else hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "purpose": "deliverable",
            "apply_order": index,
        }
        for index, (name, content) in enumerate(files.items(), start=1)
    ]
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest: dict[str, object] = {"files": records}
        if prompt_profile is not None:
            manifest["prompt_profile"] = prompt_profile
        archive.writestr("MANIFEST.json", json.dumps(manifest))
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_enqueue_copies_and_hash_verifies_attachment(tmp_path: Path) -> None:
    job = enqueue_launch_job(_request(), spool_path=tmp_path)

    assert job.mode == "chat"
    assert job.model_slug == "gpt-5-6-pro"
    assert job.model_label == "GPT-5.6 Sol"
    assert job.effort_label == "Pro"
    assert job.thinking_effort == "standard"
    assert job.prompt_profile == SOL_PRO_PROMPT_PROFILE
    assert job.prompt_prefix_sha256 == sol_pro_prompt_sha256()
    assert "Work deeply" in job.prompt
    assert "Implement the receiver queue" in job.prompt
    assert job.status == "queued"
    assert job.attachments[0].size_bytes == len(b"project context")
    attachment, content = read_launch_attachment(
        job.job_id,
        job.attachments[0].attachment_id,
        spool_path=tmp_path,
    ) or pytest.fail("missing attachment")
    assert attachment.sha256 == job.attachments[0].sha256
    assert content == b"project context"
    assert list_launch_jobs(spool_path=tmp_path)[0].prompt == job.prompt


def test_enqueue_rejects_unsafe_metadata_and_invalid_schedule(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="basename"):
        BrowserLaunchAttachmentInput(name="bad\r\nheader.zip", content_base64="")
    with pytest.raises(ValueError, match="ISO-8601"):
        enqueue_launch_job(_request(not_before="sometime later"), spool_path=tmp_path)


def test_two_instances_cannot_overlap_the_submission_critical_section(tmp_path: Path) -> None:
    first = enqueue_launch_job(_request(), spool_path=tmp_path)
    enqueue_launch_job(_request(), spool_path=tmp_path)

    claimed = claim_due_launch_job("live-browser", spool_path=tmp_path)
    assert claimed is not None and claimed.job_id == first.job_id
    assert claimed.lease_owner == "live-browser"
    assert claim_due_launch_job("live-browser", spool_path=tmp_path) is None
    assert claim_due_launch_job("private-browser", spool_path=tmp_path) is None

    submitting = update_launch_job(
        first.job_id,
        BrowserLaunchJobUpdateRequest(
            owner_instance_id="live-browser",
            outcome="progress",
            phase="submit_intent",
            tab_id=42,
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing submitting job")
    assert submitting.status == "submitting"
    recovered = claim_due_launch_job("live-browser", spool_path=tmp_path)
    assert recovered is not None and recovered.job_id == first.job_id


def test_submitted_chats_continue_in_parallel_at_configured_cadence(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    first = enqueue_launch_job(_request(), spool_path=tmp_path)
    second = enqueue_launch_job(_request(), spool_path=tmp_path)
    claim_due_launch_job("live-browser", spool_path=tmp_path)
    submitted = update_launch_job(
        first.job_id,
        BrowserLaunchJobUpdateRequest(owner_instance_id="live-browser", outcome="submitted", phase="submitted"),
        spool_path=tmp_path,
    ) or pytest.fail("missing first job")

    assert submitted.lease_owner == "live-browser"
    assert claim_due_launch_job("private-browser", spool_path=tmp_path) is None
    frozen_clock.advance(61)
    next_claim = claim_due_launch_job("private-browser", spool_path=tmp_path) or pytest.fail("not claimed")
    assert next_claim.job_id == second.job_id
    assert next_claim.lease_owner == "private-browser"
    assert list_launch_jobs(spool_path=tmp_path)[-1].status == "submitted"


def test_launch_now_prioritizes_out_of_order_without_bypassing_provider_circuit(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    first = enqueue_launch_job(_request(), spool_path=tmp_path)
    second = enqueue_launch_job(_request(), spool_path=tmp_path)
    prioritized = control_launch_job(
        second.job_id,
        BrowserLaunchJobControlRequest(action="launch_now"),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")
    assert prioritized.manual_priority is True
    claimed = claim_due_launch_job("live-browser", spool_path=tmp_path) or pytest.fail("not claimed")
    assert claimed.job_id == second.job_id
    assert claimed.manual_priority is False
    assert list_launch_jobs(spool_path=tmp_path)[-1].job_id == first.job_id

    submitted = update_launch_job(
        second.job_id,
        BrowserLaunchJobUpdateRequest(owner_instance_id="live-browser", outcome="submitted", phase="submitted"),
        spool_path=tmp_path,
    ) or pytest.fail("missing prioritized job")
    assert submitted.status == "submitted"
    control_launch_job(
        first.job_id,
        BrowserLaunchJobControlRequest(action="launch_now"),
        spool_path=tmp_path,
    )
    same_owner = claim_due_launch_job("live-browser", spool_path=tmp_path)
    assert same_owner is not None and same_owner.job_id == first.job_id

    intent = update_launch_job(
        first.job_id,
        BrowserLaunchJobUpdateRequest(
            owner_instance_id="live-browser",
            outcome="progress",
            phase="submit_intent",
            tab_id=42,
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")
    assert intent.status == "submitting"
    assert datetime.fromisoformat(intent.lease_expires_at or "") > frozen_clock.now() + timedelta(hours=5)
    assert claim_due_launch_job("private-browser", spool_path=tmp_path) is None


def test_rate_limit_cooldown_and_protocol_pause_are_receiver_authoritative(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    job = enqueue_launch_job(_request(), spool_path=tmp_path)
    enqueue_launch_job(_request(), spool_path=tmp_path)
    claimed = claim_due_launch_job("one", spool_path=tmp_path) or pytest.fail("not claimed")
    cooled = update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(
            owner_instance_id="one",
            outcome="rate_limited",
            phase="provider_backoff",
            retry_after_seconds=900,
            detail="http_429",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")
    assert claimed.job_id == cooled.job_id
    assert cooled.status == "cooldown"
    assert cooled.cooldown_reason == "rate_limited"
    assert datetime.fromisoformat(cooled.next_attempt_at) > frozen_clock.now() + timedelta(minutes=14)
    assert claim_due_launch_job("two", spool_path=tmp_path) is None
    with pytest.raises(BrowserLaunchConflictError, match="cannot bypass"):
        control_launch_job(
            job.job_id,
            BrowserLaunchJobControlRequest(action="launch_now"),
            spool_path=tmp_path,
        )
    with pytest.raises(BrowserLaunchConflictError, match="cannot bypass"):
        control_launch_job(
            job.job_id,
            BrowserLaunchJobControlRequest(action="retry"),
            spool_path=tmp_path,
        )

    frozen_clock.advance(901)
    resumed = control_launch_job(
        job.job_id,
        BrowserLaunchJobControlRequest(action="retry"),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")
    assert resumed.status == "queued"
    assert claim_due_launch_job("two", spool_path=tmp_path) is not None
    paused = update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(
            owner_instance_id="two",
            outcome="protocol_mismatch",
            phase="preflight",
            detail="Work selected",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")
    assert paused.status == "paused"
    assert paused.cooldown_reason == "protocol_mismatch"


def test_post_submit_provider_circuit_never_auto_launches_duplicate_conversation(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    job = enqueue_launch_job(_request(), spool_path=tmp_path)
    enqueue_launch_job(_request(), spool_path=tmp_path)
    claim_due_launch_job("owner", spool_path=tmp_path)
    update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(owner_instance_id="owner", outcome="submitted", phase="submitted"),
        spool_path=tmp_path,
    )
    blocked = update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(
            owner_instance_id="owner",
            outcome="safety_locked",
            phase="provider_safety_lock",
            retry_after_seconds=3600,
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")

    assert blocked.status == "paused"
    assert blocked.conversation_url is None
    assert claim_due_launch_job("other", spool_path=tmp_path) is None
    frozen_clock.advance(3601)
    next_job = claim_due_launch_job("other", spool_path=tmp_path) or pytest.fail("next job not claimed")
    assert next_job.job_id != blocked.job_id
    assert list_launch_jobs(spool_path=tmp_path)[-1].status == "paused"


def test_post_submit_rate_limit_uses_receiver_exponential_backoff(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    job = enqueue_launch_job(_request(), spool_path=tmp_path)
    claim_due_launch_job("owner", spool_path=tmp_path)
    update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(owner_instance_id="owner", outcome="submitted", phase="submitted"),
        spool_path=tmp_path,
    )

    blocked = update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(
            owner_instance_id="owner",
            outcome="rate_limited",
            phase="provider_rate_limit",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")

    retry_at = datetime.fromisoformat(blocked.next_attempt_at)
    assert blocked.status == "paused"
    assert blocked.retry_after_seconds is None
    assert frozen_clock.now() + timedelta(minutes=30) <= retry_at
    assert retry_at < frozen_clock.now() + timedelta(minutes=33)


def test_unknown_submit_is_quarantined_until_operator_confirms_absence(tmp_path: Path) -> None:
    job = enqueue_launch_job(_request(), spool_path=tmp_path)
    enqueue_launch_job(_request(), spool_path=tmp_path)
    claim_due_launch_job("owner", spool_path=tmp_path)
    unknown = update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(
            owner_instance_id="owner",
            outcome="submission_unknown",
            phase="unknown_submit_outcome",
            detail="execution channel ended after the submit boundary",
            tab_id=42,
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")

    assert unknown.status == "submission_unknown"
    assert unknown.lease_owner is None
    assert claim_due_launch_job("other", spool_path=tmp_path) is not None
    with pytest.raises(BrowserLaunchConflictError):
        control_launch_job(
            job.job_id,
            BrowserLaunchJobControlRequest(action="retry"),
            spool_path=tmp_path,
        )
    with pytest.raises(ValueError, match="inspection receipt"):
        BrowserLaunchJobControlRequest(action="confirm_no_conversation")

    requeued = control_launch_job(
        job.job_id,
        BrowserLaunchJobControlRequest(
            action="confirm_no_conversation",
            inspection_receipt="operator inspected ChatGPT and found no matching conversation",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")
    assert requeued.status == "queued"
    assert requeued.manual_priority is True
    assert requeued.events[-1].kind == "operator_confirm_no_conversation"
    assert requeued.events[-1].detail == "operator inspected ChatGPT and found no matching conversation"


def test_unknown_submit_can_be_reconciled_to_an_existing_conversation(tmp_path: Path) -> None:
    job = enqueue_launch_job(_request(), spool_path=tmp_path)
    next_job = enqueue_launch_job(
        _request(job_id="launch-after-reconciliation", cadence_minutes=15),
        spool_path=tmp_path,
    )
    claim_due_launch_job("owner", spool_path=tmp_path)
    update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(
            owner_instance_id="owner",
            outcome="submission_unknown",
            phase="unknown_submit_outcome",
            detail="provider accepted submit before the execution channel ended",
            tab_id=42,
        ),
        spool_path=tmp_path,
    )

    with pytest.raises(ValueError, match="conversation id and URL"):
        BrowserLaunchJobControlRequest(
            action="confirm_existing_conversation",
            inspection_receipt="inspected retained tab",
        )
    with pytest.raises(ValueError, match="matching its id"):
        BrowserLaunchJobControlRequest(
            action="confirm_existing_conversation",
            inspection_receipt="inspected retained tab",
            conversation_id="conversation-1",
            conversation_url="https://chatgpt.com/c/different-conversation",
        )

    reconciled = control_launch_job(
        job.job_id,
        BrowserLaunchJobControlRequest(
            action="confirm_existing_conversation",
            inspection_receipt="operator inspected retained tab and found the matching conversation",
            conversation_id="conversation-1",
            conversation_url="https://chatgpt.com/c/conversation-1",
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")

    assert reconciled.status == "submitted"
    assert reconciled.phase == "operator_confirmed_existing_conversation"
    assert reconciled.conversation_id == "conversation-1"
    assert reconciled.conversation_url == "https://chatgpt.com/c/conversation-1"
    assert reconciled.attempts == 1
    assert reconciled.last_error is None
    assert reconciled.lease_owner is None
    assert reconciled.events[-1].kind == "operator_confirm_existing_conversation"
    adopted = claim_due_launch_job("monitor", spool_path=tmp_path) or pytest.fail("monitor did not adopt job")
    assert adopted.job_id == job.job_id
    assert adopted.status == "submitted"
    assert adopted.lease_owner == "monitor"
    assert claim_due_launch_job("other", spool_path=tmp_path) is None
    delayed = next(item for item in list_launch_jobs(spool_path=tmp_path) if item.job_id == next_job.job_id)
    assert delayed.status == "cooldown"
    assert delayed.cooldown_reason == "cadence"


def test_non_owner_update_and_invalid_terminal_control_fail_closed(tmp_path: Path) -> None:
    job = enqueue_launch_job(_request(), spool_path=tmp_path)
    claim_due_launch_job("owner", spool_path=tmp_path)
    with pytest.raises(BrowserLaunchLeaseError):
        update_launch_job(
            job.job_id,
            BrowserLaunchJobUpdateRequest(owner_instance_id="other", phase="uploading"),
            spool_path=tmp_path,
        )
    with pytest.raises(BrowserLaunchConflictError, match="cannot pause an active launch"):
        control_launch_job(
            job.job_id,
            BrowserLaunchJobControlRequest(action="pause"),
            spool_path=tmp_path,
        )
    with pytest.raises(ValueError):
        BrowserLaunchJobUpdateRequest.model_validate(
            {"owner_instance_id": "owner", "outcome": "completed", "phase": "handoff_validated"}
        )
    update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(owner_instance_id="owner", outcome="submitted", phase="submitted"),
        spool_path=tmp_path,
    )
    completed = accept_launch_handoff(
        job.job_id,
        BrowserLaunchHandoffRequest(
            owner_instance_id="owner",
            name="polylogue-sol-pro-launch-handoff.zip",
            content_base64=base64.b64encode(_handoff_zip()).decode(),
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")
    assert completed.status == "completed"
    with pytest.raises(BrowserLaunchConflictError):
        control_launch_job(
            job.job_id,
            BrowserLaunchJobControlRequest(action="cancel"),
            spool_path=tmp_path,
        )


def test_handoff_completion_requires_locally_validated_cohesive_zip(tmp_path: Path) -> None:
    job = enqueue_launch_job(_request(), spool_path=tmp_path)
    claim_due_launch_job("owner", spool_path=tmp_path)
    update_launch_job(
        job.job_id,
        BrowserLaunchJobUpdateRequest(owner_instance_id="owner", outcome="submitted", phase="submitted"),
        spool_path=tmp_path,
    )

    with pytest.raises(BrowserLaunchHandoffError, match="checksum mismatch"):
        accept_launch_handoff(
            job.job_id,
            BrowserLaunchHandoffRequest(
                owner_instance_id="owner",
                name="polylogue-sol-pro-launch-handoff.zip",
                content_base64=base64.b64encode(_handoff_zip(corrupt_checksum=True)).decode(),
            ),
            spool_path=tmp_path,
        )

    for profile in (None, "polylogue-sol-pro-worker-wrong"):
        with pytest.raises(BrowserLaunchHandoffError, match="prompt profile mismatch"):
            accept_launch_handoff(
                job.job_id,
                BrowserLaunchHandoffRequest(
                    owner_instance_id="owner",
                    name="polylogue-sol-pro-launch-handoff.zip",
                    content_base64=base64.b64encode(_handoff_zip(prompt_profile=profile)).decode(),
                ),
                spool_path=tmp_path,
            )

    content = _handoff_zip()
    completed = accept_launch_handoff(
        job.job_id,
        BrowserLaunchHandoffRequest(
            owner_instance_id="owner",
            name="polylogue-sol-pro-launch-handoff.zip",
            content_base64=base64.b64encode(content).decode(),
        ),
        spool_path=tmp_path,
    ) or pytest.fail("missing job")
    assert completed.status == "completed"
    assert completed.phase == "handoff_validated"
    assert completed.handoff_sha256 == hashlib.sha256(content).hexdigest()
    assert completed.handoff_file_count == 7
    assert completed.handoff_artifact_ref is not None
    assert (tmp_path / completed.handoff_artifact_ref).read_bytes() == content


def test_future_job_is_not_claimed_early(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    future = (frozen_clock.now() + timedelta(minutes=10)).isoformat()
    job = enqueue_launch_job(_request(not_before=future), spool_path=tmp_path)
    assert claim_due_launch_job("one", spool_path=tmp_path) is None
    stored = list_launch_jobs(spool_path=tmp_path)[0]
    assert stored.job_id == job.job_id
    assert stored.status == "cooldown"
    assert stored.cooldown_reason == "cadence"
    control_launch_job(
        job.job_id,
        BrowserLaunchJobControlRequest(action="launch_now"),
        spool_path=tmp_path,
    )
    assert claim_due_launch_job("one", spool_path=tmp_path) is None


def test_launch_cli_copies_targeted_files_and_hardcodes_chat_sol_pro(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Build the package and return exactly one ZIP.\n", encoding="utf-8")
    attachment = tmp_path / "context.tar.gz"
    attachment.write_bytes(b"targeted context")
    spool = tmp_path / "spool"

    result = CliRunner().invoke(
        daemon_cli,
        [
            "browser-capture",
            "launch",
            "--prompt-file",
            str(prompt),
            "--title",
            "Build a durable launch handoff",
            "--attachment",
            str(attachment),
            "--cadence",
            "60",
            "--job-id",
            "launch-cli-test",
            "--spool",
            str(spool),
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    job = list_launch_jobs(spool_path=spool)[0]
    assert job.job_id == "launch-cli-test"
    assert job.job_title == "Build a durable launch handoff"
    assert job.prompt.startswith("# Mission: Build a durable launch handoff\n")
    assert (job.mode, job.model_label, job.effort_label) == ("chat", "GPT-5.6 Sol", "Pro")
    assert job.cadence_minutes == 60
    stored_attachment = read_launch_attachment(job.job_id, job.attachments[0].attachment_id, spool_path=spool)
    assert stored_attachment is not None and stored_attachment[1] == b"targeted context"


def test_receiver_routes_enqueue_claim_and_serve_hash_pinned_inputs(tmp_path: Path) -> None:
    with _running_receiver(tmp_path) as (host, port):
        created = _http(host, port, "POST", "/v1/launch-jobs", _request().model_dump(mode="json"))
        created_body = json.loads(created.read())
        job_id = created_body["job"]["job_id"]
        attachment = created_body["job"]["attachments"][0]

        claimed = _http(host, port, "GET", "/v1/launch-jobs?claim_by=live-browser")
        claimed_body = json.loads(claimed.read())
        attachment_response = _http(
            host,
            port,
            "GET",
            f"/v1/launch-jobs/{job_id}/attachments/{attachment['attachment_id']}",
        )
        attachment_bytes = attachment_response.read()
        blocked = _http(host, port, "GET", "/v1/launch-jobs?claim_by=private-browser")
        blocked_body = json.loads(blocked.read())
        submitted = _http(
            host,
            port,
            "POST",
            f"/v1/launch-jobs/{job_id}/events",
            {"owner_instance_id": "live-browser", "outcome": "submitted", "phase": "submitted"},
        )
        submitted.read()
        handoff_content = _handoff_zip()
        handoff = _http(
            host,
            port,
            "POST",
            f"/v1/launch-jobs/{job_id}/handoff",
            {
                "owner_instance_id": "live-browser",
                "name": "polylogue-sol-pro-launch-handoff.zip",
                "content_base64": base64.b64encode(handoff_content).decode(),
            },
        )
        handoff_body = json.loads(handoff.read())

    assert created.status == HTTPStatus.ACCEPTED
    assert claimed.status == HTTPStatus.OK
    assert claimed_body["jobs"][0]["lease_owner"] == "live-browser"
    assert blocked_body["jobs"] == []
    assert attachment_response.status == HTTPStatus.OK
    assert attachment_response.getheader("Content-Disposition") == 'attachment; filename="context.tar.gz"'
    assert attachment_bytes == b"project context"
    assert handoff.status == HTTPStatus.OK
    assert handoff_body["job"]["status"] == "completed"
    assert handoff_body["job"]["handoff_sha256"] == hashlib.sha256(handoff_content).hexdigest()


def test_receiver_reports_corrupt_launch_state_instead_of_hiding_it(tmp_path: Path) -> None:
    corrupt = tmp_path / "launch-jobs" / "corrupt" / "job.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{not-json", encoding="utf-8")

    with _running_receiver(tmp_path) as (host, port):
        response = _http(host, port, "GET", "/v1/launch-jobs")
        body = json.loads(response.read())

    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert body["error"] == "write_failed"
