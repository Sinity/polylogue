from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from devtools import live_provider_proof_service


@pytest.mark.parametrize(
    "branch",
    [
        "empty",
        "invalid",
        "missing_ok",
        "native_capture_unavailable",
        "provider_throttle_authority_unavailable",
        "rate_limited",
        "capture_cancelled",
        "capture_rejected",
        "capture_timed_out",
        "private_error",
        "cancelled_outcome",
        "deferred",
        "bridge_code:conversation_api_url_not_found",
        "bridge_code:asset_body_stream_unavailable",
        "bridge_code:receiver_unpaired",
        "bridge_code:capture_staging_unavailable",
        "bridge_code:capture_staging_sender_invalid",
        "bridge_code:capture_staging_request_invalid",
        "bridge_code:capture_staging_owner_mismatch",
        "bridge_code:capture_staging_invalid_ref",
        "bridge_code:capture_staging_sequence_mismatch",
        "bridge_code:capture_staging_records_unavailable",
        "bridge_code:native_acquisition_sequence_invalid",
        "bridge_code:capture_response_metadata_invalid",
        "bridge_code:capture_response_metadata_conflict",
        "bridge_code:capture_staging_incomplete",
        "bridge_code:capture_staging_interrupted",
        "bridge_code:capture_staging_missing_bytes",
        "bridge_code:conversation_api_url_not_found:private-token",
        "bridge_stage:admission",
        "bridge_stage:provider_fetch",
        "bridge_stage:staging",
        "bridge_stage:private",
        "bridge_stage:malformed",
        "bridge_stage:accepted",
        "bridge_refused",
        "bridge_private",
        "bridge_ambiguous",
        "response_ok",
        "identity_matches",
        "native_digest_valid",
        "native_full",
        "turn_count_valid",
        "attachment_count_valid",
        "artifact_present",
        "receiver_request_present",
    ],
)
@pytest.mark.parametrize("provider_name", ["chatgpt", "claude-ai"])
def test_returned_capture_refusal_survives_original_evaluation_summary_cleanup_and_python_boundary(
    branch: str,
    provider_name: str,
) -> None:
    script = (
        "const branch = "
        + json.dumps(branch)
        + "; const providerName = "
        + json.dumps(provider_name)
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { captureProvider, providerSummary, inProofPhase, settleProofCleanup, currentProofFailure } from './scripts/live_provider_proof.mjs';
const secret = 'https://private.invalid/conversation?token=private-transcript';
const provider = {provider:providerName, nativeId:'synthetic', url:'https://example.invalid/synthetic'};
let result = {ok:true,envelope:{session:{provider:providerName,provider_session_id:'synthetic'},receiver_native:{sha256:'a'.repeat(64)},capture_summary:{captureMode:'native_full',turnCount:1,attachmentCount:0}},captureResult:{artifact_ref:'synthetic',receiver_request_id:'synthetic'}};
if (branch === 'empty') result = null;
else if (branch === 'invalid') result = secret;
else if (branch === 'missing_ok') result = {};
else if (branch === 'private_error') result = {ok:false,error:secret};
else if (branch === 'cancelled_outcome') result = {ok:false,outcome:'cancelled'};
else if (branch === 'deferred') result = {ok:true,deferred:true,envelope:result.envelope};
else if (branch.startsWith('bridge_')) {
  result = {ok:false,error:'native_capture_unavailable',native_attempts:[{stage:'page_bridge_fetch',ok:false,status:403,accepted:false,error:branch === 'bridge_private' ? secret : branch.startsWith('bridge_code:') ? branch.slice('bridge_code:'.length) : 'capture_rejected',url:secret,body:secret}]};
  if(branch.startsWith('bridge_stage:')) result.native_attempts[0].failure_stage = branch === 'bridge_stage:private' ? secret : branch === 'bridge_stage:malformed' ? {stage:'admission',private:secret} : branch === 'bridge_stage:accepted' ? 'admission' : branch.slice('bridge_stage:'.length);
  if(branch === 'bridge_stage:accepted') result.native_attempts[0].accepted = true;
  if(branch === 'bridge_ambiguous') result.native_attempts.push({...result.native_attempts[0],accepted:true});
} else if(branch === 'response_ok') result.ok = false;
else if(branch === 'identity_matches') result.envelope.session.provider_session_id = secret;
else if(branch === 'native_digest_valid') result.envelope.receiver_native.sha256 = secret;
else if(branch === 'native_full') result.envelope.capture_summary.captureMode = secret;
else if(branch === 'turn_count_valid') result.envelope.capture_summary.turnCount = 0;
else if(branch === 'attachment_count_valid') result.envelope.capture_summary.attachmentCount = -1;
else if(branch === 'artifact_present') delete result.captureResult.artifact_ref;
else if(branch === 'receiver_request_present') delete result.captureResult.receiver_request_id;
else result = {ok:false,error:branch};
const __polylogueOwnedProviderProof = {consumeCapture:async(id,nativeId)=>{assert.equal(id,1);assert.equal(nativeId,'synthetic');return result;}};
const popup = {call:async(_method,params)=>({result:{value:await vm.runInNewContext(params.expression,{__polylogueOwnedProviderProof,Date,URL,setTimeout})}})};
const captured = await inProofPhase('capture',()=>captureProvider(popup,provider,1));
let primary;
try { await inProofPhase('summary',()=>{assert.equal(providerSummary(provider,captured).ok,false);throw new Error('proof_capture_incomplete');}); }
catch(error){primary=error;}
await settleProofCleanup(null,primary);
const report = currentProofFailure(primary);
assert.deepEqual(report.error,{phase:'summary',category:'capture_incomplete'});
assert.equal(JSON.stringify(report).includes(secret),false);
console.log(JSON.stringify(report));
"""
    )
    process = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path("browser-extension"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    error = live_provider_proof_service.failed_child(process.stdout, [])
    assert error.report == report
    assert report["error"] == {"phase": "summary", "category": "capture_incomplete"}
    assert set(report["cleanup"].values()) <= {"settled", "not_required"}
    evidence = report["capture_evidence"][0]
    assert evidence["provider"] == provider_name
    expected = {
        "empty": "response_empty",
        "invalid": "response_invalid",
        "missing_ok": "response_invalid",
        "private_error": "unknown",
        "cancelled_outcome": "capture_cancelled",
        "deferred": "deferred",
        "bridge_refused": "native_capture_unavailable",
        "bridge_private": "native_capture_unavailable",
        "bridge_ambiguous": "native_capture_unavailable",
        "response_ok": "unknown",
    }.get(branch, "accepted" if branch in live_provider_proof_service._SUMMARY_CHECKS else branch)
    if branch.startswith(("bridge_code:", "bridge_stage:")):
        expected = "native_capture_unavailable"
    assert evidence["category"] == expected
    if branch in live_provider_proof_service._SUMMARY_CHECKS:
        assert evidence["summary"][branch] is False
        assert sum(evidence["summary"].values()) == 7
    if branch.startswith("bridge_"):
        assert evidence["bridge"] == (
            {"observed": False, "accepted": None, "status": None, "category": "none", "failure_stage": "unknown"}
            if branch == "bridge_ambiguous"
            else {
                "observed": True,
                "accepted": branch == "bridge_stage:accepted",
                "status": 403,
                "failure_stage": branch.removeprefix("bridge_stage:")
                if branch in {"bridge_stage:admission", "bridge_stage:provider_fetch", "bridge_stage:staging"}
                else "unknown",
                "category": (
                    branch.removeprefix("bridge_code:")
                    if branch.startswith("bridge_code:")
                    and branch.removeprefix("bridge_code:") in live_provider_proof_service._BRIDGE_FAILURE_CATEGORIES
                    else "unknown"
                    if branch == "bridge_private" or branch.startswith("bridge_code:")
                    else "capture_rejected"
                ),
            }
        )


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "category",
        "provider",
        "summary",
        "bridge",
        "status_bool",
        "status_range",
        "failure_stage",
        "stage_shape",
        "stage_accepted",
        "stage_unobserved",
        "stage_no_error",
        "duplicate",
    ],
)
def test_python_rejects_private_or_malformed_capture_evidence(fault: str) -> None:
    evidence: dict[str, Any] = {
        "provider": "chatgpt",
        "category": "native_capture_unavailable",
        "bridge": {
            "observed": True,
            "accepted": False,
            "status": 403,
            "category": "unknown",
            "failure_stage": "unknown",
        },
        "summary": dict.fromkeys(live_provider_proof_service._SUMMARY_CHECKS, False),
    }
    private = "https://private.invalid/?token=private-transcript"
    if fault == "extra":
        evidence["body"] = private
    elif fault in {"category", "provider"}:
        evidence[fault] = private
    elif fault == "summary":
        evidence["summary"]["response_ok"] = private
    elif fault == "bridge":
        evidence["bridge"]["category"] = private
    elif fault == "failure_stage":
        evidence["bridge"]["failure_stage"] = private
    elif fault == "stage_shape":
        evidence["bridge"]["failure_stage"] = {"stage": "admission", "private": private}
    elif fault == "stage_accepted":
        evidence["bridge"]["failure_stage"] = "admission"
        evidence["bridge"]["accepted"] = True
    elif fault == "stage_unobserved":
        evidence["bridge"]["failure_stage"] = "admission"
        evidence["bridge"]["observed"] = False
    elif fault == "stage_no_error":
        evidence["bridge"]["failure_stage"] = "admission"
        evidence["bridge"]["category"] = "none"
    elif fault == "status_bool":
        evidence["bridge"]["status"] = True
    elif fault == "status_range":
        evidence["bridge"]["status"] = 600
    report = {
        "ok": False,
        "error": {"phase": "summary", "category": "capture_incomplete"},
        "native_progress": [],
        "capture_evidence": [evidence] * (2 if fault == "duplicate" else 1),
        "cleanup": dict.fromkeys(["receiver", "permission", "mutations", "targets"], "settled"),
    }
    error = live_provider_proof_service.failed_child(json.dumps(report), [])
    assert error.report["error"] == {"phase": "unknown", "category": "operation_failed"}
    assert error.report["capture_evidence"] == []
    assert private not in json.dumps(error.report)


@pytest.mark.parametrize(
    "urls",
    [
        [],
        ["https://chatgpt.com/"],
        ["https://claude.ai/"],
        ["https://chatgpt.com/c/one?view=two"],
        ["https://chatgpt.com/c/one#two"],
        ["https://elsewhere.invalid/c/one"],
        ["http://chatgpt.com/c/one"],
        ["https://chatgpt.com/c/one/extra"],
        ["https://chatgpt.com/c/one", "https://chatgpt.com/c/two"],
        [None],
    ],
)
def test_conversation_selection_rejects_non_exact_routes_before_browser_mutation(tmp_path: Path, urls: object) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(urls))
    with pytest.raises(ValueError):
        live_provider_proof_service.conversation_targets(selection)


def test_conversation_selection_preserves_exact_identity(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(["https://chatgpt.com/c/synthetic-one", "https://claude.ai/chat/synthetic-two"]))
    assert live_provider_proof_service.conversation_targets(selection) == [
        {"name": "chatgpt", "url": "https://chatgpt.com/c/synthetic-one", "nativeId": "synthetic-one"},
        {"name": "claude", "url": "https://claude.ai/chat/synthetic-two", "nativeId": "synthetic-two"},
    ]


def _artifact_receipt(tmp_path: Path, envelope: dict[str, Any], turns: int, attachments: int) -> dict[str, object]:
    import hashlib

    session = envelope["session"]
    assert isinstance(session, dict)
    literal = json.dumps(envelope).encode()
    (tmp_path / "artifact.json").write_bytes(literal)
    return {
        "artifact_ref": "artifact.json",
        "artifact_sha256": hashlib.sha256(literal).hexdigest(),
        "provider": session["provider"],
        "provider_session_id_sha256": hashlib.sha256(session["provider_session_id"].encode()).hexdigest(),
        "turn_count": turns,
        "attachment_count": attachments,
    }


@pytest.mark.parametrize("fixture", ["native-rich-blocks-v1.json", "native-duplicate-attachment-occurrences-v1.json"])
def test_live_proof_validates_actual_canonical_lowering_and_asset_occurrences(tmp_path: Path, fixture: str) -> None:
    from tests.infra.live_provider_proof import native_proof_artifact

    envelope, turns, attachments = native_proof_artifact(tmp_path, fixture)
    receipt = _artifact_receipt(tmp_path, envelope, turns, attachments)
    result = live_provider_proof_service.verify_captured_artifact(tmp_path, receipt)
    assert result["message_count"] == turns and turns > 0
    assert isinstance(result["block_count"], int)
    assert result["block_count"] > 0
    assert result["attachment_count"] == attachments and attachments > 0
    assert isinstance(result["parent_message_count"], int)
    assert result["parent_message_count"] > 0
    assert isinstance(result["content_hash"], str)
    assert len(result["content_hash"]) == 64
    assert isinstance(result["attachment_directions"], list)
    assert "model_output" in result["attachment_directions"]
    if fixture == "native-duplicate-attachment-occurrences-v1.json":
        assert result["attachment_directions"] == ["model_output", "user_input"]
    if fixture == "native-rich-blocks-v1.json":
        assert isinstance(result["tool_use_count"], int)
        assert result["tool_use_count"] > 0


@pytest.mark.parametrize(
    "mutation",
    [
        "omitted",
        "text",
        "parent",
        "blocks",
        "digest",
        "count",
        "outside",
        "asset_omitted",
        "asset_digest",
        "asset_receipt",
        "asset_owner",
    ],
)
def test_live_proof_refuses_incomplete_or_forged_native_artifact(tmp_path: Path, mutation: str) -> None:
    from tests.infra.live_provider_proof import native_proof_artifact

    envelope, turns, attachments = native_proof_artifact(tmp_path, "native-rich-blocks-v1.json")
    if mutation == "omitted":
        envelope["session"]["turns"] = []
    elif mutation == "text":
        envelope["session"]["turns"][0]["text"] = "changed canonical content"
    elif mutation == "parent":
        envelope["session"]["turns"][1]["parent_turn_id"] = None
    elif mutation == "blocks":
        envelope["session"]["turns"][1]["blocks"] = []
    if mutation == "asset_omitted":
        envelope["session"]["attachments"] = []
    elif mutation == "asset_digest":
        envelope["session"]["attachments"][0]["provider_meta"]["content_sha256"] = "0" * 64
    elif mutation == "asset_receipt":
        envelope["session"]["attachments"][0]["provider_meta"].pop("asset_acquisition")
    elif mutation == "asset_owner":
        envelope["session"]["attachments"][0]["provider_meta"]["native_turn_ordinal"] = 0
    receipt = _artifact_receipt(tmp_path, envelope, turns, attachments)
    if mutation == "digest":
        receipt["artifact_sha256"] = "0" * 64
    elif mutation == "count":
        receipt["attachment_count"] = 0
    elif mutation == "outside":
        receipt["artifact_ref"] = "../outside.json"
    with pytest.raises(ValueError):
        live_provider_proof_service.verify_captured_artifact(tmp_path, receipt)


def test_live_proof_sanitizes_private_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(**_kwargs: object) -> None:
        raise ValueError("private synthetic transcript must not enter job output")

    monkeypatch.setattr(live_provider_proof_service, "run_proof", fail)
    assert (
        live_provider_proof_service.main(
            [
                "--json",
                "--conversations-file",
                str(tmp_path / "selection.json"),
                "--chrome-user-data-dir",
                str(tmp_path),
                "--evidence-root",
                str(tmp_path / "evidence"),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "private synthetic transcript" not in output
    assert json.loads(output)["error"]["type"] == "ValueError"


def test_live_proof_preserves_claude_tool_errors_and_signatures(tmp_path: Path) -> None:
    from polylogue.core.enums import Provider
    from tests.infra.live_provider_proof import native_proof_artifact

    envelope, turns, attachments = native_proof_artifact(tmp_path, "native-rich-blocks-v1.json", Provider.CLAUDE_AI)
    receipt = _artifact_receipt(tmp_path, envelope, turns, attachments)
    result = live_provider_proof_service.verify_captured_artifact(tmp_path, receipt)
    assert result["message_count"] == 4
    assert isinstance(result["tool_use_count"], int)
    assert result["tool_use_count"] > 0
    assert isinstance(result["tool_result_count"], int)
    assert result["tool_result_count"] > 0
    assert isinstance(result["tool_error_count"], int)
    assert result["tool_error_count"] > 0
    # Full block equality above includes the real canonical signature field.
    assert any(
        block.get("signature") == "synthetic-signature"
        for turn in envelope["session"]["turns"]
        for block in turn["blocks"]
    )


@pytest.mark.parametrize("page", ["popup", "admin", "provider"])
@pytest.mark.parametrize("response", ["verified", "placement_refused", "unknown"])
def test_shutdown_awaits_original_owned_window_response_before_target_closure(page: str, response: str) -> None:
    script = (
        "const options = "
        + json.dumps({"page": page, "response": response})
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import { openProofWindow, ownProofBrowser, requireProofRunning, closeOwnedProofWindows, installShutdownCleanup } from './scripts/live_provider_proof.mjs';
const urls = { popup: 'chrome-extension://synthetic/src/popup.html', admin: 'chrome://extensions/', provider: 'https://chatgpt.com/c/synthetic' };
const events = [];
let controlCalls = 0;
let reply;
ownProofBrowser({ call: async (method, params) => {
  assert.equal(method, 'Target.closeTarget');
  assert.equal(params.targetId, 'A'.repeat(32));
  assert(events.includes('response_settled'));
  events.push('target_closed');
  return { success: true };
} });
process.once('exit', () => process.stdout.write(JSON.stringify({ events, controlCalls }) + '\n'));
installShutdownCleanup();
const creation = openProofWindow(urls[options.page], 1000, () => {
  controlCalls += 1;
  events.push('creation_started');
  return new Promise(resolve => { reply = resolve; });
});
process.emit('SIGTERM');
// Let the actual signal handler reach target cleanup while the original
// remote transaction is still suspended. A premature exit loses this reply.
await new Promise(setImmediate);
assert.throws(() => requireProofRunning());
assert.throws(() => openProofWindow('https://claude.ai/chat/next', 1000, () => { controlCalls += 1; }));
assert(!events.includes('target_closed'));
events.push('response_settled');
reply(options.response === 'unknown' ? {} : {
  id: 'A'.repeat(32), url: urls[options.page], parked: options.response === 'verified', workspace: 'agentbrowser', show_with: 'F7',
});
await creation.catch(() => undefined);
const settlement = closeOwnedProofWindows();
assert.equal(settlement, closeOwnedProofWindows());
await settlement.catch(() => undefined);
"""
    )
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path("browser-extension"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 143, result.stderr
    reports = [json.loads(line) for line in result.stdout.splitlines()]
    evidence = next(report for report in reports if "events" in report)
    terminal = next(report for report in reports if report.get("ok") is False)
    assert terminal["error"]["category"] == "shutdown"
    assert terminal["cleanup"]["targets"] == ("settled" if response == "verified" else "failed")
    assert evidence["controlCalls"] == 1
    if response == "unknown":
        assert evidence["events"] == ["creation_started", "response_settled"]
    else:
        assert evidence["events"] == ["creation_started", "response_settled", "target_closed"]
    assert ("proof_signal_owned_cleanup_failed" in result.stderr) is (response != "verified")


@pytest.mark.parametrize(
    "child_output",
    [
        "known",
        "malformed",
        "private_phase",
        "private_category",
        "private_cleanup",
        "extra_private",
        "wrong_ok",
        "wrong_shape",
        "null_category",
        "multiple_reports",
    ],
)
@pytest.mark.parametrize("returncode", [0, 1])
def test_failed_child_report_crosses_actual_service_boundary_without_private_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    child_output: str,
    returncode: int,
) -> None:
    report: dict[str, Any] = {
        "ok": False,
        "error": {"phase": "permission_grant", "category": "operation_failed"},
        "native_progress": [],
        "capture_evidence": [],
        "cleanup": {"receiver": "settled", "permission": "unknown", "mutations": "failed", "targets": "settled"},
    }
    secret = "https://private.invalid/conversation?token=private-transcript"
    if child_output == "private_phase":
        report["error"]["phase"] = secret
    elif child_output == "private_category":
        report["error"]["category"] = secret
    elif child_output == "private_cleanup":
        report["cleanup"]["receiver"] = secret
    elif child_output == "extra_private":
        report["stack"] = secret
    elif child_output == "wrong_ok":
        report["ok"] = True
    elif child_output == "wrong_shape":
        report["cleanup"] = [secret]
    elif child_output == "null_category":
        report["error"]["category"] = None
    stdout = secret if child_output == "malformed" else json.dumps(report)
    if child_output == "multiple_reports":
        stdout += "\n" + stdout

    def failed_run(**_kwargs: object) -> None:
        raise live_provider_proof_service.failed_child(stdout, [])

    monkeypatch.setattr(live_provider_proof_service, "run_proof", failed_run)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(["https://chatgpt.com/c/synthetic"]))
    assert (
        live_provider_proof_service.main(
            [
                "--json",
                "--conversations-file",
                str(selection),
                "--chrome-user-data-dir",
                str(tmp_path),
                "--evidence-root",
                str(tmp_path / "evidence"),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert secret not in output
    if child_output == "known":
        assert json.loads(output) == {**report, "receiver_requests": []}
    else:
        assert json.loads(output)["error"] == {"phase": "unknown", "category": "operation_failed"}
        assert set(json.loads(output)["cleanup"].values()) == {"unknown"}
        assert json.loads(output)["receiver_requests"] == []


def test_node_phase_and_terminal_report_use_fixed_categories_and_publish_once() -> None:
    script = r"""
import assert from 'node:assert/strict';
import { inProofPhase, proofFailureReport, currentProofFailure, publishProofFailure } from './scripts/live_provider_proof.mjs';
const secret = 'https://private.invalid/conversation?token=private-transcript';
for (const phase of ['pause', 'revision', 'permission_grant', 'receiver_pairing', 'capture', 'summary']) {
  const error = new Error(secret);
  await assert.rejects(inProofPhase(phase, () => { throw error; }), observed => observed === error);
  const report = proofFailureReport(phase, error, { receiver: 'settled', permission: 'failed', mutations: 'failed', targets: 'settled' });
  assert.equal(report.error.phase, phase);
  assert.equal(currentProofFailure(error).error.phase, phase);
  assert.equal(report.error.category, 'operation_failed');
  assert(!JSON.stringify(report).includes(secret));
}
assert.deepEqual(proofFailureReport(secret, new Error(secret), { receiver: secret }).error, { phase: 'unknown', category: 'operation_failed' });
assert.equal(proofFailureReport('revision', new Error('proof_installed_revision_mismatch')).error.category, 'revision_mismatch');
assert.equal(proofFailureReport('capture', new AggregateError([new Error(secret)])).error.category, 'cleanup_failed');
await inProofPhase('permission_grant', () => undefined);
publishProofFailure(new Error(secret));
await inProofPhase('capture', () => undefined);
publishProofFailure(new Error(secret));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path("browser-extension"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert len(result.stdout.splitlines()) == 1
    report = json.loads(result.stdout)
    assert report["error"] == {"phase": "permission_grant", "category": "operation_failed"}
    assert "private-transcript" not in result.stdout


def test_original_cdp_evaluation_preserves_only_whitelisted_page_exception_categories() -> None:
    script = r"""
import assert from 'node:assert/strict';
import { evaluateJson, proofFailureReport } from './scripts/live_provider_proof.mjs';
const secret = 'https://private.invalid/synthetic?token=private-transcript';
const known = [
  ['proof_receiver_configuration_failed', 'receiver_configuration_failed'],
  ['proof_receiver_permission_refused', 'receiver_permission_refused'],
  ['proof_receiver_handshake_failed', 'receiver_handshake_failed'],
  ['proof_receiver_configuration_changed', 'configuration_changed'],
  ['proof_pause_restore_failed', 'pause_failed'],
  ['proof_host_permission_failed', 'control_failed'],
  ['proof_installed_resource_missing', 'revision_missing'],
];
for (const [code, category] of known) {
  const client = {call: async (method, params) => {
    assert.equal(method, 'Runtime.evaluate');
    assert.equal(params.awaitPromise, true);
    assert.equal(params.returnByValue, true);
    return {exceptionDetails: {text: 'Uncaught (in promise)', exception: {description: `Error: ${code}\n at ${secret}`}}};
  }};
  let primary;
  try { await evaluateJson(client, 'synthetic expression'); } catch (error) { primary = error; }
  assert.equal(primary.message, code);
  const terminal = new AggregateError([primary], 'synthetic cleanup', {cause: primary});
  const report = proofFailureReport('receiver_pairing', terminal, {receiver: 'settled', permission: 'settled', targets: 'settled', mutations: 'failed'});
  assert.equal(report.error.category, category);
  assert.equal(report.cleanup.mutations, 'failed');
  assert(!JSON.stringify(report).includes(secret));
  assert(!primary.stack.includes(secret));
}
for (const description of [secret, `Error: proof_receiver_handshake_failed ${secret}`, `TypeError: proof_receiver_handshake_failed`, null, {private: secret}]) {
  const client = {call: async () => ({exceptionDetails: {text: secret, exception: {description}}})};
  let error;
  try { await evaluateJson(client, 'synthetic expression'); } catch (caught) { error = caught; }
  const report = proofFailureReport('receiver_pairing', error);
  assert.equal(error.message, 'proof_evaluation_failed');
  assert.equal(report.error.category, 'operation_failed');
  assert(!JSON.stringify(report).includes(secret));
  assert(!error.stack.includes(secret));
}
assert.deepEqual(await evaluateJson({call: async () => ({result: {value: {ok: true}}})}, 'synthetic expression'), {ok: true});
console.log(JSON.stringify({ok: true}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path("browser-extension"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"ok": True}


@pytest.mark.parametrize("capture_result", ["returned", "rejected"])
@pytest.mark.parametrize("close_result", ["settled", "failed"])
def test_signal_preserves_interrupted_capture_phase_while_original_target_cleanup_settles(
    capture_result: str, close_result: str
) -> None:
    script = (
        "const options = "
        + json.dumps({"capture": capture_result, "close": close_result})
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import { inProofPhase, currentProofFailure, installShutdownCleanup, ownProofBrowser, openProofWindow } from './scripts/live_provider_proof.mjs';
const events = [];
let finishCapture, finishClose;
const target = 'A'.repeat(32);
ownProofBrowser({ call: async (method, params) => {
  assert.equal(method, 'Target.closeTarget');
  assert.equal(params.targetId, target);
  events.push('target_cleanup_started');
  finishCapture();
  return new Promise(resolve => { finishClose = () => {
    events.push('target_cleanup_settled');
    resolve({ success: options.close === 'settled' });
  }; });
} });
await openProofWindow('https://chatgpt.com/c/synthetic', 1000, async () => ({
  id: target, url: 'https://chatgpt.com/c/synthetic', parked: true, workspace: 'agentbrowser', show_with: 'F7',
}));
installShutdownCleanup();
const main = (async () => {
  try {
    await inProofPhase('capture', () => new Promise((resolve, reject) => {
      finishCapture = () => {
        events.push('capture_cancelled');
        if (options.capture === 'returned') resolve({ ok: false });
        else reject(new Error('private synthetic provider fault'));
      };
    }));
    await inProofPhase('summary', () => { events.push('summary_started'); });
  } catch (error) {
    const report = currentProofFailure(error);
    assert.equal(report.error.phase, 'capture');
    if (options.capture === 'returned') assert.equal(report.error.category, 'shutdown');
    events.push('main_stopped');
  }
})();
process.emit('SIGTERM');
await new Promise(setImmediate);
await main;
assert(events.includes('target_cleanup_started'));
assert(!events.includes('summary_started'));
assert(!events.includes('target_cleanup_settled'));
assert.equal(currentProofFailure(new Error('proof_shutdown_requested')).error.phase, 'capture');
process.once('exit', () => process.stdout.write(JSON.stringify({ events }) + '\n'));
finishClose();
"""
    )
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path("browser-extension"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 143, result.stderr
    reports = [json.loads(line) for line in result.stdout.splitlines()]
    terminal = next(report for report in reports if report.get("ok") is False)
    evidence = next(report for report in reports if "events" in report)
    assert terminal["error"] == {"phase": "capture", "category": "shutdown"}
    assert terminal["cleanup"]["targets"] == close_result
    assert evidence["events"] == [
        "target_cleanup_started",
        "capture_cancelled",
        "main_stopped",
        "target_cleanup_settled",
    ]
    assert "private synthetic provider fault" not in result.stdout


@pytest.mark.parametrize(
    "progress",
    [
        [{"stage": "body", "state": "BEGIN"}],
        [{"stage": "native_prepare", "state": "BEGIN", "source": "background_debug_log"}],
        [{"stage": "native_prepare", "state": "BEGIN"}],
        [{"stage": "body", "state": "BEGIN", "source": "background_debug_log"}],
        [{"stage": [], "state": "BEGIN"}],
        [{"stage": "body", "state": "BEGIN", "token": "private-token"}],
        [{"stage": "https://private.invalid", "state": "END"}],
        [{"stage": "body", "state": "private-token"}],
        [{"stage": "body", "state": "BEGIN"}] * 7,
    ],
)
def test_original_node_and_python_progress_validators_retain_only_fixed_bounded_markers(progress: object) -> None:
    report = {
        "ok": False,
        "error": {"phase": "capture", "category": "shutdown"},
        "cleanup": dict.fromkeys(["receiver", "permission", "mutations", "targets"], "settled"),
        "native_progress": progress,
        "capture_evidence": [],
    }
    valid = progress in (
        [{"stage": "body", "state": "BEGIN"}],
        [{"stage": "native_prepare", "state": "BEGIN", "source": "background_debug_log"}],
    )
    assert bool(live_provider_proof_service.child_failure_report(json.dumps(report))) is valid
    script = (
        "const progress = "
        + json.dumps(progress)
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import { retainNativeProgress, currentProofFailure } from './scripts/live_provider_proof.mjs';
let accepted = false;
try { retainNativeProgress(progress); accepted = true; } catch(error) { assert.equal(error.message, 'proof_capture_incomplete'); }
const report = currentProofFailure(new Error('proof_shutdown_requested'));
console.log(JSON.stringify({accepted, progress: report.native_progress}));
"""
    )
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path("browser-extension"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value == {"accepted": valid, "progress": progress if valid else []}
    assert "private" not in result.stdout


def test_original_capture_response_retains_progress_before_signal_cleanup_publication() -> None:
    script = r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { captureProvider, inProofPhase, currentProofFailure, installShutdownCleanup, ownProofBrowser, openProofWindow } from './scripts/live_provider_proof.mjs';
let reply, close;
const progress = [{stage:'provider_response',state:'END'}, {stage:'body',state:'BEGIN'}];
const __polylogueOwnedProviderProof = {consumeCapture:()=>new Promise(resolve=>{reply=resolve;})};
const popup = {call:async (_method, params)=>({result:{value:await vm.runInNewContext(params.expression,{__polylogueOwnedProviderProof,Date,URL,setTimeout})}})};
ownProofBrowser({call:async()=>{reply({ok:false,outcome:'cancelled',native_progress:progress});return new Promise(resolve=>{close=()=>resolve({success:true});});}});
await openProofWindow('https://chatgpt.com/c/synthetic',1000,async()=>({id:'A'.repeat(32),url:'https://chatgpt.com/c/synthetic',parked:true,workspace:'agentbrowser',show_with:'F7'}));
installShutdownCleanup();
const main = inProofPhase('capture',()=>captureProvider(popup,{url:'https://chatgpt.com/c/synthetic',nativeId:'synthetic',provider:'chatgpt'},1));
while (!reply) await Promise.resolve();
process.emit('SIGTERM');
const response = await main;
assert.equal(response.result.ok,false);
assert.deepEqual(currentProofFailure(new Error('proof_shutdown_requested')).native_progress,progress);
while (!close) await Promise.resolve();
close();
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path("browser-extension"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 143, result.stderr
    report = json.loads(result.stdout)
    assert report["error"] == {"phase": "capture", "category": "shutdown"}
    assert report["native_progress"] == [
        {"stage": "provider_response", "state": "END"},
        {"stage": "body", "state": "BEGIN"},
    ]
    assert report["cleanup"]["targets"] == "settled"


def test_original_control_child_maps_only_fixed_placement_refusal_without_private_stderr() -> None:
    script = r"""
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { runChromeControl, proofFailureReport } from './scripts/live_provider_proof.mjs';
const secret = 'https://private.invalid/conversation?token=private-content';
const known = `window ${'A'.repeat(32)} opened but was not verified on agentbrowser; last compositor state: {"title":"${secret}"}; visible=true; stable_checks=0; focus_before=0x123; hyprctl=/private/host; instances=1; signature=set\n`;
for (const [bytes, expected] of [
  [known, 'window_visibility_refused'],
  [known.replace('visible=true', 'visible=false'), 'window_refused'],
  [known.replace('visible=true', 'visible=unknown'), 'window_refused'],
  [known.replace(/\{.*\}/, 'unavailable').replace('visible=true', 'visible=unknown'), 'window_refused'],
  [known.replace('visible=true', 'visible=private'), 'control_failed'],
  [known.replace('opened but was not verified', 'arbitrary private text'), 'control_failed'],
  [known.replace(secret, '; visible=true; stable_checks=9;').replace('}; visible=true', '}; visible=false'), 'window_refused'],
  [`arbitrary ${secret}; visible=true; stable_checks=0;\n`, 'control_failed'],
  [`${secret}\n${known}`, 'window_visibility_refused'],
  [`compositor state changed after navigating agent window: before=${secret} after=${secret}\n`, 'window_compositor_changed'],
  [`focused compositor client disappeared while identifying target: address=${secret}\n`, 'window_compositor_changed'],
  [`failed to create agent browser target (status 1) ${secret}\n`, 'window_creation_failed'],
  [`failed to navigate parked agent target ${secret} (status 1)\n`, 'window_navigation_failed'],
  [`failed to install temporary agent-window compositor rules\n`, 'window_rules_failed'],
  [`agent-window requires a live Hyprland compositor with a focused operator client; focus=${secret}\n`, 'window_precondition_failed'],
  [`private prefix compositor state changed after navigating agent window: ${secret}\n`, 'control_failed'],
  [`compositor state changed unknown phase: ${secret}\n`, 'control_failed'],

]) {
  let child;
  const pending = runChromeControl(['agent-window', '--url', secret], 1000, (command, args, options) => {
    assert.equal(command, '/home/sinity/.local/bin/sinnix-chrome-control');
    assert.deepEqual(args, ['agent-window', '--url', secret]);
    assert.deepEqual(options.stdio, ['ignore', 'pipe', 'pipe']);
    child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
    return child;
  });
  // Split every byte, including the fixed prefix and visibility token.
  for (const byte of Buffer.from(bytes)) child.stderr.emit('data', Buffer.from([byte]));
  child.emit('close', 1);
  let primary;
  try { await pending; } catch (error) { primary = error; }
  const report = proofFailureReport('provider_open', primary);
  assert.equal(report.error.category, expected);
  assert.equal(report.error.phase, 'provider_open');
  assert.ok(!JSON.stringify(report).includes(secret));
  assert.ok(!primary.message.includes(secret));
  console.log(JSON.stringify(report));
}
let success;
const successful = runChromeControl(['status'], 1000, () => {
  success = new EventEmitter(); success.stdout = new EventEmitter(); success.stderr = new EventEmitter(); return success;
});
success.stderr.emit('data', known);
success.stdout.emit('data', JSON.stringify({ok: true}) + '\n'); success.emit('close', 0);
assert.deepEqual(await successful, {ok: true});
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path(__file__).resolve().parents[3] / "browser-extension",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    reports = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(reports) == 17
    for report in reports:
        retained = live_provider_proof_service.failed_child(json.dumps(report), []).report
        assert retained["error"] == report["error"]
