from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
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
const chrome = {tabs:{query:async()=>[{id:1,url:provider.url,pinned:false}],sendMessage:async(_id,message)=>{
  assert.deepEqual(JSON.parse(JSON.stringify(message)),{type:'polylogue.capturePage',providerSessionId:'synthetic'});return result;
}}};
const popup = {call:async(_method,params)=>({result:{value:await vm.runInNewContext(params.expression,{chrome,Date,URL,setTimeout})}})};
const captured = await inProofPhase('capture',()=>captureProvider(popup,provider,1,1000));
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


class _FakeServer:
    server_address = ("127.0.0.1", 49120)

    def serve_forever(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def server_close(self) -> None:
        return None


class _FakeThread:
    def __init__(self, **_kwargs: object) -> None:
        return None

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        del timeout


def test_live_provider_timeout_terminates_group_and_becomes_typed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(live_provider_proof_service, "require_declared_operation_context", lambda _operation: "unit")
    bound: list[object] = []

    def fake_make_server(_host: str, port: int, **_kwargs: object) -> _FakeServer:
        bound.append(port)
        return _FakeServer()

    monkeypatch.setattr(live_provider_proof_service, "make_server", fake_make_server)
    monkeypatch.setattr(live_provider_proof_service, "Thread", _FakeThread)
    process = SimpleNamespace(
        communicate=lambda **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(["node"], 120)),
        returncode=None,
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    terminated: list[object] = []
    monkeypatch.setattr(live_provider_proof_service, "terminate_process_group", terminated.append)

    selection = tmp_path / "conversations.json"
    selection.write_text(json.dumps(["https://chatgpt.com/c/synthetic-proof"]))
    assert live_provider_proof_service.main(["--json", "--conversations-file", str(selection)]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == {"phase": "unknown", "category": "control_timeout"}
    assert set(payload["cleanup"].values()) == {"unknown"}
    assert payload["receiver_requests"] == []
    assert terminated == [process, process]
    assert bound == [0]


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
    assert live_provider_proof_service.main(["--json", "--conversations-file", str(tmp_path / "selection.json")]) == 1
    output = capsys.readouterr().out
    assert "private synthetic transcript" not in output
    assert json.loads(output)["error"]["type"] == "ValueError"


def test_native_browser_summary_and_actual_pairing_restore_contract() -> None:
    """The same helpers used by the owned extension page run without live Chrome."""
    script = r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { configureReceiver, restoreReceiverConfiguration, providerSummary } from './scripts/live_provider_proof.mjs';
import { receiverConfigurationOwner } from './tests/infra/receiver_configuration.js';
const provider = { host: 'chatgpt.com', provider: 'chatgpt', nativeId: 'synthetic-one' };
const envelope = { session: { provider: 'chatgpt', provider_session_id: 'synthetic-one', turns: [] }, receiver_native: { sha256: 'a'.repeat(64) }, capture_summary: { captureMode: 'native_full', turnCount: 2, attachmentCount: 1 } };
const payload = { result: { ok: true, envelope, captureResult: { artifact_ref: 'synthetic.json', receiver_request_id: 'request' } } };
assert.equal(providerSummary(provider, payload).ok, true);
for (const changed of [ { ...envelope, capture_summary: {} }, { ...envelope, receiver_native: {} }, { ...envelope, session: { ...envelope.session, provider_session_id: 'other' } } ]) assert.equal(providerSummary(provider, { result: { ...payload.result, envelope: changed } }).ok, false);
assert.equal(providerSummary(provider, { result: { ...payload.result, captureResult: {} } }).ok, false);
let values = { receiverBaseUrl: 'http://127.0.0.1:8765', polylogueReceiverPairing: { receiver_id: 'old' }, queue: ['retained'] };
const previous = structuredClone(values);
const messages = [];
const chrome = { permissions: { contains: async () => true }, storage: { local: {
  get: async keys => Object.fromEntries((Array.isArray(keys) ? keys : Object.keys(keys)).filter(key => Object.hasOwn(values, key) || !Array.isArray(keys)).map(key => [key, Object.hasOwn(values, key) ? values[key] : keys[key]])),
  set: async rows => Object.assign(values, rows), remove: async keys => (Array.isArray(keys) ? keys : [keys]).forEach(key => delete values[key]),
} }, runtime: { sendMessage: async message => {
  messages.push(message);
  if (message.type === 'polylogue.configureReceiver' || message.type === 'polylogue.receiverPairing.reset') return receiverOwner.send(message);
  return { ok: true };
} } };
const receiverOwner = receiverConfigurationOwner(chrome, async () => ({ body: { ok: true, receiver_id: 'proof', api_schema: 'polylogue-browser-capture/v1' }, response: { ok: true, status: 200 } }));
const client = { call: async (_method, params) => {
  try { return { result: { value: await vm.runInNewContext(params.expression, { chrome }) } }; }
  catch (error) { return { exceptionDetails: { text: 'Uncaught (in promise)', exception: { description: `Error: ${error.message}\n at private synthetic stack` } } }; }
} };
const admitted = await configureReceiver(client, 'http://127.0.0.1:49001');
assert.equal(admitted.receiver_id, 'proof');
assert.deepEqual(messages.map(message => message.type), ['polylogue.ambient.configure', 'polylogue.configureReceiver', 'polylogue.receiverPairing.reset']);
await restoreReceiverConfiguration(client, previous, { baseUrl: 'http://127.0.0.1:49001', receiverId: 'proof', revision: admitted.revision });
assert.deepEqual(JSON.parse(JSON.stringify(values)), previous);
assert.equal(messages.at(-1).automatic_capture_enabled, false);
assert(!messages.some(message => Object.hasOwn(message, 'receiverAuthToken')));
// A setup fault before any config mutation restores only the unchanged snapshot.
await restoreReceiverConfiguration(client, previous, { baseUrl: 'http://127.0.0.1:49001', receiverId: null, revision: null });
values.receiverBaseUrl = 'http://concurrent';
await assert.rejects(restoreReceiverConfiguration(client, previous, { baseUrl: 'http://127.0.0.1:49001', receiverId: 'proof', revision: admitted.revision }));
assert.equal(values.receiverBaseUrl, 'http://concurrent');
values = { receiverBaseUrl: 'http://127.0.0.1:49001', polylogueReceiverPairing: { receiver_id: 'proof' }, queue: ['retained'] };
const readmitted = await configureReceiver(client, 'http://127.0.0.1:49001');
await restoreReceiverConfiguration(client, {}, { baseUrl: 'http://127.0.0.1:49001', receiverId: 'proof', revision: readmitted.revision });
assert.deepEqual(values, { queue: ['retained'] });
chrome.runtime.sendMessage = async () => ({ ok: true, health: { status: 'offline' }, pairing: null });
await assert.rejects(configureReceiver(client, 'http://127.0.0.1:49001'));
console.log(JSON.stringify({ ok: true }));
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


@pytest.mark.parametrize("settled_before_signal", [False, True])
@pytest.mark.parametrize("concurrent_configuration", [False, True])
def test_signal_cleanup_settles_owned_grant_and_removes_permission_after_restore_refusal(
    settled_before_signal: bool, concurrent_configuration: bool
) -> None:
    script = (
        "const options = "
        + json.dumps({"settled": settled_before_signal, "concurrent": concurrent_configuration})
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { proofReceiverCustody, requestProofHostPermission, cleanupProofReceiver, installShutdownCleanup } from './scripts/live_provider_proof.mjs';
const events = [];
const previous = { receiverBaseUrl: 'http://127.0.0.1:8765' };
const values = { ...previous, queue: ['retained'] };
let finishGrant;
let granted = false;
const chrome = { storage: { local: {
  get: async keys => Object.fromEntries((Array.isArray(keys) ? keys : Object.keys(keys)).filter(key => Object.hasOwn(values, key) || !Array.isArray(keys)).map(key => [key, Object.hasOwn(values, key) ? values[key] : keys[key]])),
  set: async rows => { events.push('restore'); Object.assign(values, rows); },
  remove: async keys => (Array.isArray(keys) ? keys : [keys]).forEach(key => delete values[key]),
} }, runtime: { sendMessage: async message => {
  // The extension restores only an unchanged receiver (this proof never
  // applied its owned configuration); a foreign change is refused.
  if (message?.type === 'polylogue.configureReceiver' && message.restore) {
    const unchanged = ['receiverBaseUrl', 'polylogueReceiverPairing'].every(key => values[key] === message.restore.previous[key]);
    return unchanged ? { ok: true } : { ok: false, error: 'proof_receiver_configuration_changed' };
  }
  return { ok: true };
} }, permissions: {
  contains: async () => granted,
  request: () => new Promise(resolve => {
    events.push('grant_started');
    finishGrant = () => { events.push('grant_settled'); granted = true; resolve(true); };
  }),
  remove: async () => {
    assert.equal(granted, true);
    assert(events.indexOf('grant_settled') > events.indexOf('grant_started'));
    assert(events.indexOf('prompt_restore_settled') > events.indexOf('prompt_restore_started'));
    events.push('permission_removed');
    granted = false;
    assert.deepEqual(values.queue, ['retained']);
    if (options.concurrent) assert.equal(values.receiverBaseUrl, 'http://concurrent');
    else assert.equal(values.receiverBaseUrl, previous.receiverBaseUrl);
    process.stdout.write(JSON.stringify({ events, configuration: values.receiverBaseUrl }) + '\n');
    return true;
  },
} };
const client = { call: async (_method, params) => {
  try { return { result: { value: await vm.runInNewContext(params.expression, { chrome }) } }; }
  catch (error) { return { exceptionDetails: { text: 'Uncaught (in promise)', exception: { description: `Error: ${error.message}\n at private synthetic stack` } } }; }
} };
const owner = proofReceiverCustody(client, previous, { baseUrl: 'http://proof', receiverId: null }, 'http://proof/*');
installShutdownCleanup();
const grant = requestProofHostPermission(owner, async () => {
  events.push('prompt_restore_started');
  await new Promise(resolve => setImmediate(resolve));
  events.push('prompt_restore_settled');
});
while (!finishGrant) await Promise.resolve();
if (options.settled) { finishGrant(); await grant; }
if (options.concurrent) values.receiverBaseUrl = 'http://concurrent';
process.emit('SIGTERM');
// An in-flight grant must remain owned until its callback settles.
if (!options.settled) { assert.equal(granted, false); finishGrant(); }
await grant;
assert.equal(owner.cleaning, true);
assert(owner.settlement);
const settlement = owner.settlement;
assert.equal(settlement, cleanupProofReceiver(owner));
await settlement.catch(() => undefined);
assert.throws(() => requestProofHostPermission(owner));
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
    assert terminal["cleanup"]["permission"] == "settled"
    assert terminal["cleanup"]["receiver"] == ("failed" if concurrent_configuration else "settled")
    assert evidence["events"].count("permission_removed") == 1
    assert evidence["events"].index("permission_removed") > evidence["events"].index("grant_settled")
    assert evidence["configuration"] == ("http://concurrent" if concurrent_configuration else "http://127.0.0.1:8765")
    assert ("proof_signal_receiver_cleanup_failed" in result.stderr) is concurrent_configuration


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
    assert ("proof_signal_target_cleanup_failed" in result.stderr) is (response != "verified")


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
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(live_provider_proof_service, "require_declared_operation_context", lambda _operation: "unit")
    calls: list[str] = []

    class Server(_FakeServer):
        def shutdown(self) -> None:
            calls.append("shutdown")

        def server_close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(live_provider_proof_service, "make_server", lambda *_args, **_kwargs: Server())
    monkeypatch.setattr(live_provider_proof_service, "Thread", _FakeThread)
    process = SimpleNamespace(communicate=lambda **_kwargs: (stdout, secret), returncode=returncode)
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        live_provider_proof_service, "terminate_process_group", lambda _process: calls.append("terminate")
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(["https://chatgpt.com/c/synthetic"]))
    assert live_provider_proof_service.main(["--json", "--conversations-file", str(selection)]) == 1
    output = capsys.readouterr().out
    assert secret not in output
    if child_output == "known":
        assert json.loads(output) == {**report, "receiver_requests": []}
    else:
        assert json.loads(output)["error"] == {"phase": "unknown", "category": "operation_failed"}
        assert set(json.loads(output)["cleanup"].values()) == {"unknown"}
        assert json.loads(output)["receiver_requests"] == []
    assert calls == ["terminate", "shutdown", "close"]
    assert list(tmp_path.iterdir()) == [selection]


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


@pytest.mark.parametrize("fault", ["none", "grant", "configure", "restore", "remove"])
def test_actual_receiver_cleanup_reports_each_owned_outcome_independently(fault: str) -> None:
    script = (
        "const fault = "
        + json.dumps(fault)
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { proofReceiverCustody, requestProofHostPermission, configureProofReceiver, cleanupProofReceiver, currentProofFailure } from './scripts/live_provider_proof.mjs';
import { receiverConfigurationOwner } from './tests/infra/receiver_configuration.js';
const secret = 'private synthetic transcript';
const values = {};
const events = [];
let restoring = false;
const chrome = { storage: { local: {
  get: async keys => Object.fromEntries((Array.isArray(keys) ? keys : Object.keys(keys)).filter(k => Object.hasOwn(values, k) || !Array.isArray(keys)).map(k => [k, Object.hasOwn(values, k) ? values[k] : keys[k]])),
  set: async rows => { if (fault === 'restore' && restoring) throw new Error(secret); Object.assign(values, rows); },
  remove: async keys => { if (fault === 'restore' && restoring) throw new Error(secret); (Array.isArray(keys) ? keys : [keys]).forEach(k => delete values[k]); },
} }, runtime: { sendMessage: async message => {
  if (message.type === 'polylogue.configureReceiver') {
    if (message.restore) { restoring = true; return receiverOwner.send(message); }
    const admitted = await receiverOwner.send(message);
    if (fault === 'configure') throw new Error(secret);
    return admitted;
  }
  if (message.type === 'polylogue.receiverPairing.reset') return receiverOwner.send(message);
  return { ok: true };
} }, permissions: {
  contains: async () => events.includes('grant'),
  request: async () => { events.push('grant'); if (fault === 'grant') throw new Error(secret); return true; },
  remove: async () => { events.push('remove'); if (fault === 'remove') throw new Error(secret); return true; },
} };
const receiverOwner = receiverConfigurationOwner(chrome, async () => ({ body: { ok: true, receiver_id: 'proof', api_schema: 'polylogue-browser-capture/v1' }, response: { ok: true, status: 200 } }));
const client = { call: async (_method, params) => {
  try { return { result: { value: await vm.runInNewContext(params.expression, { chrome }) } }; }
  catch (error) { return { exceptionDetails: { text: 'Uncaught (in promise)', exception: { description: `Error: ${error.message}\n at private synthetic stack` } } }; }
} };
const owner = proofReceiverCustody(client, {}, { baseUrl: 'http://127.0.0.1:49001', receiverId: null }, 'http://127.0.0.1:49001/*');
await requestProofHostPermission(owner).catch(() => undefined);
await configureProofReceiver(owner).catch(() => undefined);
let failure;
try { await cleanupProofReceiver(owner); } catch (error) { failure = error; }
const report = currentProofFailure(failure);
assert(!JSON.stringify(report).includes(secret));
assert.equal(report.cleanup.receiver, ['restore', 'configure'].includes(fault) ? 'failed' : 'settled');
assert.equal(report.cleanup.permission, fault === 'grant' ? 'unknown' : fault === 'remove' ? 'failed' : 'settled');
assert.equal(report.cleanup.mutations, ['grant', 'configure'].includes(fault) ? 'failed' : 'settled');
assert.equal(events.includes('remove'), fault !== 'grant');
assert.equal(Boolean(failure), fault !== 'none');
if (fault === 'configure') assert.equal(values.receiverBaseUrl, 'http://127.0.0.1:49001');
console.log(JSON.stringify(report));
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
    assert "private synthetic transcript" not in result.stdout
    report = json.loads(result.stdout)
    assert report["cleanup"]["targets"] == "not_required"


@pytest.mark.parametrize("phase", ["configuration", "handshake", "permission"])
def test_normal_cleanup_preserves_actual_primary_receiver_refusal_category(phase: str) -> None:
    script = (
        "const fault = "
        + json.dumps(phase)
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { proofReceiverCustody, configureProofReceiver, settleProofCleanup, inProofPhase, currentProofFailure } from './scripts/live_provider_proof.mjs';
import { receiverConfigurationOwner } from './tests/infra/receiver_configuration.js';
const values = {};
const messages = [];
const chrome = { permissions: { contains: async () => true }, storage: { local: {
  get: async keys => Object.fromEntries((Array.isArray(keys) ? keys : Object.keys(keys)).filter(k => Object.hasOwn(values, k) || !Array.isArray(keys)).map(k => [k, Object.hasOwn(values, k) ? values[k] : keys[k]])),
  set: async rows => Object.assign(values, rows), remove: async keys => (Array.isArray(keys) ? keys : [keys]).forEach(k => delete values[k]),
} }, runtime: { sendMessage: async message => {
  messages.push(message);
  if (message.type === 'polylogue.configureReceiver') {
    if (message.restore) return receiverOwner.send(message);
    if (fault === 'configuration') return { ok: false, error: 'private synthetic receiver refusal' };
    if (fault === 'permission') return { ok: false, error: 'receiver_origin_not_permitted' };
    return receiverOwner.send(message);
  }
  if (message.type === 'polylogue.receiverPairing.reset') return receiverOwner.send(message);
  return { ok: true };
} } };
const receiverOwner = receiverConfigurationOwner(chrome, async () => { throw new Error('neutral-unreachable'); });
const client = { call: async (_method, params) => {
  try { return { result: { value: await vm.runInNewContext(params.expression, { chrome }) } }; }
  catch (error) { return { exceptionDetails: { text: 'Uncaught (in promise)', exception: { description: `Error: ${error.message}\n at private synthetic stack` } } }; }
} };
const owner = proofReceiverCustody(client, {}, { baseUrl: 'http://127.0.0.1:49001', receiverId: null }, 'http://127.0.0.1:49001/*');
let primary;
try { await inProofPhase('receiver_pairing', () => configureProofReceiver(owner)); } catch (error) { primary = error; }
assert(primary);
let terminal;
try { await settleProofCleanup(owner, primary); } catch (error) { terminal = error; }
assert.equal(terminal.cause, primary);
const report = currentProofFailure(terminal);
assert.equal(report.error.phase, 'receiver_pairing');
assert.equal(report.error.category, {configuration: 'receiver_configuration_failed', handshake: 'receiver_handshake_failed', permission: 'receiver_permission_refused'}[fault]);
assert.deepEqual(report.cleanup, { receiver: 'settled', permission: 'not_required', mutations: 'failed', targets: 'not_required' });
assert.deepEqual(values, {});
assert(!JSON.stringify(report).includes('private synthetic'));
console.log(JSON.stringify(report));
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
    assert json.loads(result.stdout)["cleanup"]["receiver"] == "settled"


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


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("granted", [False, True, None])
@pytest.mark.parametrize("active", [False, True, None])
def test_optional_permission_request_preserves_preexisting_access_and_exact_custody(
    existing: bool, granted: bool | None, active: bool | None
) -> None:
    script = (
        "const options = "
        + json.dumps({"existing": existing, "granted": granted, "active": active})
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { proofReceiverCustody, requestProofHostPermission, settleProofCleanup, inProofPhase, currentProofFailure } from './scripts/live_provider_proof.mjs';
const events = [];
const origin = 'http://127.0.0.1:49152/*';
const chrome = {permissions: {
  contains: async request => {assert.deepEqual(Array.from(request.origins), [origin]);events.push('contains');return events.includes('request') ? options.active : options.existing;},
  request: async request => {assert.deepEqual(Array.from(request.origins), [origin]);events.push('request');return options.granted;},
  remove: async request => {assert.deepEqual(Array.from(request.origins), [origin]);events.push('remove');return true;},
}, storage: {local: {get: async () => ({}), set: async () => {}, remove: async () => {}}}, runtime: {sendMessage: async () => ({ok: true})}};
const client = {call: async (method, params) => {
  assert.equal(method, 'Runtime.evaluate');
  assert.equal(params.userGesture, params.expression.includes('permissions.request('));
  assert(!params.expression.includes('developerPrivate'));
  return {result: {value: await vm.runInNewContext(params.expression, {chrome})}};
}};
const owner = proofReceiverCustody(client, {}, {baseUrl: 'http://proof', receiverId: null}, origin);
let primary;
try {await inProofPhase('permission_grant', () => requestProofHostPermission(owner));} catch(error) {primary=error;}
const newlyGranted = !options.existing && options.granted === true;
assert.equal(owner.permissionAdded, newlyGranted);
const success = options.existing || (options.granted === true && options.active === true);
assert.equal(Boolean(primary), !success);
let terminal;
try {await settleProofCleanup(owner, primary);} catch(error) {terminal=error;}
assert.equal(owner.permissionAdded, false);
assert.equal(events.includes('request'), !options.existing);
assert.equal(events.includes('remove'), newlyGranted);
assert.equal(owner.cleanup.permission, newlyGranted ? 'settled' : options.existing || options.granted === false ? 'not_required' : 'unknown');
assert.equal(owner.cleanup.receiver, 'settled');
assert.equal(owner.cleanup.mutations, success ? 'settled' : 'failed');
if(primary) {
  assert.equal(terminal.cause, primary);
  const report=currentProofFailure(terminal);
  assert.equal(report.error.phase, 'permission_grant');
  const refused = options.granted === false || (options.granted === true && options.active === false);
  assert.equal(report.error.category, refused ? 'receiver_permission_refused' : 'operation_failed');
}
console.log(JSON.stringify({ok: true}));
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
    assert json.loads(result.stdout) == {"ok": True}


@pytest.mark.parametrize("removed", [False, None])
def test_optional_permission_removal_refusal_retains_failed_custody(removed: bool | None) -> None:
    script = (
        "const removed = "
        + json.dumps(removed)
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { proofReceiverCustody, requestProofHostPermission, cleanupProofReceiver, currentProofFailure } from './scripts/live_provider_proof.mjs';
let active = false;
const chrome = { permissions: {
 contains: async () => active,
 request: async () => { active = true; return true; },
 remove: async () => removed,
}, storage: {local: {get: async () => ({}), set: async () => {}, remove: async () => {}}}, runtime: {sendMessage: async () => ({ok: true})}};
const client = {call: async (_method, params) => ({result: {value: await vm.runInNewContext(params.expression, {chrome})}})};
const owner = proofReceiverCustody(client, {}, {baseUrl: 'http://proof', receiverId: null}, 'http://proof/*');
await requestProofHostPermission(owner);
let failure;
try {await cleanupProofReceiver(owner);} catch(error) {failure=error;}
assert(failure);
assert.equal(owner.permissionAdded, true);
assert.equal(owner.cleanup.permission, 'failed');
assert.equal(owner.cleanup.receiver, 'settled');
assert.equal(currentProofFailure(failure).error.category, 'cleanup_failed');
console.log(JSON.stringify({ok: true}));
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
    assert json.loads(result.stdout) == {"ok": True}


def test_shutdown_during_existing_permission_check_refuses_later_optional_request() -> None:
    script = r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { proofReceiverCustody, requestProofHostPermission, cleanupProofReceiver } from './scripts/live_provider_proof.mjs';
let releaseCheck;
let requests = 0;
const chrome = {permissions: {
 contains: () => new Promise(resolve => {releaseCheck = () => resolve(false);}),
 request: async () => {requests++;return true;},
}, storage: {local: {get: async () => ({}), set: async () => {}, remove: async () => {}}}, runtime: {sendMessage: async () => ({ok: true})}};
const client = {call: async (_method, params) => ({result: {value: await vm.runInNewContext(params.expression, {chrome})}})};
const owner = proofReceiverCustody(client, {}, {baseUrl: 'http://proof', receiverId: null}, 'http://proof/*');
const request = requestProofHostPermission(owner);
const cleanup = cleanupProofReceiver(owner);
releaseCheck();
await assert.rejects(request, {message: 'proof_shutdown_requested'});
await assert.rejects(cleanup);
assert.equal(requests, 0);
assert.equal(owner.permissionAdded, false);
assert.equal(owner.cleanup.mutations, 'failed');
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


@pytest.mark.parametrize(
    "recovered", ["known", "private", "invalid_utf8", "decode", "partial_known", "partial_private"]
)
def test_timeout_consumes_final_or_partial_strict_child_report_without_private_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], recovered: str
) -> None:
    report = {
        "ok": False,
        "error": {"phase": "capture", "category": "shutdown"},
        "native_progress": [],
        "capture_evidence": [],
        "cleanup": {"receiver": "settled", "permission": "settled", "mutations": "failed", "targets": "settled"},
    }
    secret = "https://private.invalid/conversation?token=private-transcript"
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(live_provider_proof_service, "require_declared_operation_context", lambda _operation: "unit")
    monkeypatch.setattr(live_provider_proof_service, "make_server", lambda *_args, **_kwargs: _FakeServer())
    monkeypatch.setattr(live_provider_proof_service, "Thread", _FakeThread)
    events: list[str] = []

    class Process:
        returncode = 143

        def communicate(self, **_kwargs: object) -> tuple[str, str]:
            events.append("communicate")
            if events.count("communicate") == 1:
                raise subprocess.TimeoutExpired(["node"], 120, output=secret.encode(), stderr=secret.encode())
            assert "terminate" in events
            if recovered.startswith("partial_"):
                output = json.dumps(report).encode() if recovered == "partial_known" else secret.encode()
                raise subprocess.TimeoutExpired(["node"], 2, output=output, stderr=secret.encode())
            if recovered == "decode":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, secret)
            if recovered == "invalid_utf8":
                raise subprocess.TimeoutExpired(["node"], 2, output=b"\xff", stderr=secret.encode())
            return json.dumps(report) if recovered == "known" else secret, secret

    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        live_provider_proof_service, "terminate_process_group", lambda _process: events.append("terminate")
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(["https://chatgpt.com/c/synthetic"]))
    assert live_provider_proof_service.main(["--json", "--conversations-file", str(selection)]) == 1
    output = capsys.readouterr().out
    assert secret not in output
    payload = json.loads(output)
    if recovered in {"known", "partial_known"}:
        assert payload == {**report, "receiver_requests": []}
    else:
        assert payload["error"] == {"phase": "unknown", "category": "control_timeout"}
        assert set(payload["cleanup"].values()) == {"unknown"}
    assert events == ["communicate", "terminate", "communicate", "terminate"]


@pytest.mark.parametrize("fault", ["decode", "artifact", "extension"])
def test_child_decode_and_receipt_faults_use_fixed_failure_without_private_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fault: str
) -> None:
    secret = "https://private.invalid/conversation?token=private-transcript"
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(live_provider_proof_service, "require_declared_operation_context", lambda _operation: "unit")
    monkeypatch.setattr(live_provider_proof_service, "make_server", lambda *_args, **_kwargs: _FakeServer())
    monkeypatch.setattr(live_provider_proof_service, "Thread", _FakeThread)

    class Process:
        returncode = 0

        def communicate(self, **_kwargs: object) -> tuple[str, str]:
            if fault == "decode":
                raise UnicodeDecodeError("utf-8", secret.encode() + b"\xff", len(secret), len(secret) + 1, secret)
            return json.dumps({"ok": True, "providers": {"chatgpt.com": {"artifact_ref": "synthetic.json"}}}), secret

    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(live_provider_proof_service, "terminate_process_group", lambda _process: None)

    def verify(*_args: object) -> dict[str, object]:
        if fault == "artifact":
            raise ValueError(secret)
        return {}

    monkeypatch.setattr(live_provider_proof_service, "verify_captured_artifact", verify)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(["https://chatgpt.com/c/synthetic"]))
    assert live_provider_proof_service.main(["--json", "--conversations-file", str(selection)]) == 1
    output = capsys.readouterr().out
    assert secret not in output
    payload = json.loads(output)
    assert payload["error"] == (
        {"phase": "unknown", "category": "operation_failed"}
        if fault == "decode"
        else {"phase": "summary", "category": "capture_incomplete"}
    )
    assert set(payload["cleanup"].values()) == {"unknown"}
    assert payload["receiver_requests"] == []
    assert list(tmp_path.iterdir()) == [selection]


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
const chrome = {tabs:{query:async()=>[{id:1,url:'https://chatgpt.com/c/synthetic',pinned:false}],sendMessage:()=>new Promise(resolve=>{reply=resolve;})}};
const popup = {call:async (_method, params)=>({result:{value:await vm.runInNewContext(params.expression,{chrome,Date,URL,setTimeout})}})};
ownProofBrowser({call:async()=>{reply({ok:false,outcome:'cancelled',native_progress:progress});return new Promise(resolve=>{close=()=>resolve({success:true});});}});
await openProofWindow('https://chatgpt.com/c/synthetic',1000,async()=>({id:'A'.repeat(32),url:'https://chatgpt.com/c/synthetic',parked:true,workspace:'agentbrowser',show_with:'F7'}));
installShutdownCleanup();
const main = inProofPhase('capture',()=>captureProvider(popup,{url:'https://chatgpt.com/c/synthetic',nativeId:'synthetic',provider:'chatgpt'},1,1000));
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


@pytest.mark.parametrize("permission", ["granted", "refused", "preexisting", "cancelled"])
def test_permission_owner_waits_for_prompt_workspace_restore_before_cleanup(permission: str) -> None:
    script = (
        "const option = "
        + json.dumps(permission)
        + ";\n"
        + r"""
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { proofReceiverCustody, requestProofHostPermission, cleanupProofReceiver, proofFailureReport } from './scripts/live_provider_proof.mjs';
const events=[]; const values={}; let active=option === 'preexisting'; let settle;
const chrome={storage:{local:{get:async()=>values,set:async()=>{},remove:async()=>{}}},runtime:{sendMessage:async()=>({ok:true})},permissions:{
 contains:async()=>active,
 request:()=>new Promise((resolve,reject)=>{ settle=()=>{ events.push('request_settled'); if(option === 'cancelled') reject(new Error('proof_shutdown_requested')); else {active=option === 'granted'; resolve(active);} }; }),
 remove:async()=>{events.push('remove');active=false;return true;},
}};
const client={call:async(_method,{expression})=>({result:{value:await vm.runInNewContext(expression,{chrome})}})};
const owner=proofReceiverCustody(client,{}, {baseUrl:'http://127.0.0.1:49000',receiverId:null}, 'http://127.0.0.1:49000/*');
let restored; let entered;
const enteredPromise=new Promise(resolve=>entered=resolve);
const grant=requestProofHostPermission(owner,()=>{events.push('restore_started');entered();return new Promise(resolve=>restored=()=>{events.push('restore_settled');resolve();});});
let primary;
const caught=grant.catch(error=>{primary=error;});
while (option !== 'preexisting' && !settle) await new Promise(resolve=>setImmediate(resolve));
const cleanup=cleanupProofReceiver(owner).catch(()=>{});
if(settle) settle();
await enteredPromise;
assert.ok(!events.includes('remove'));
assert.equal(events.at(-1),'restore_started');
restored(); await caught; await cleanup;
if(option === 'granted') assert.deepEqual(events,['request_settled','restore_started','restore_settled','remove']);
else assert.ok(!events.includes('remove'));
if(option === 'refused') assert.equal(proofFailureReport('permission_grant',primary).error.category,'receiver_permission_refused');
if(option === 'cancelled') assert.equal(proofFailureReport('permission_grant',primary).error.category,'shutdown');
if(option === 'preexisting') assert.equal(active,true);
"""
    )
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path(__file__).resolve().parents[3] / "browser-extension",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_prompt_restore_failure_preserves_original_permission_refusal() -> None:
    script = r"""
import assert from 'node:assert/strict';
import { proofReceiverCustody, requestProofHostPermission, proofFailureReport } from './scripts/live_provider_proof.mjs';
const client={call:async(_method,{expression})=>({result:{value:false}})};
const owner=proofReceiverCustody(client,{}, {baseUrl:'http://127.0.0.1:49000',receiverId:null}, 'http://127.0.0.1:49000/*');
let primary;
try{await requestProofHostPermission(owner,async()=>{throw new Error('proof_desktop_unavailable');});}catch(error){primary=error;}
assert.ok(primary instanceof AggregateError);
assert.equal(proofFailureReport('permission_grant',primary).error.category,'receiver_permission_refused');
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path(__file__).resolve().parents[3] / "browser-extension",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_popup_binding_operator_restoration_and_hidden_preflight_fail_closed() -> None:
    script = r"""
import assert from 'node:assert/strict';
import { captureProofOperator, bindProofPopup, restoreProofOperator, requireHiddenProofWorkspace, proofFailureReport } from './scripts/live_provider_proof.mjs';
const original={address:'0x123',stable_id:0x1a2b,workspace_id:2,monitor_id:0,monitor_workspaces:[{monitor_id:0,workspace_id:2},{monitor_id:1,workspace_id:3}]};
const popup={address:'0x456',stable_id:0x1234,workspace_id:-1337,monitor_id:1};
const originalRaw={address:'0x123',stableId:'1a2b',workspace:{id:2,name:'2'},monitor:0};
const popupRaw={address:'0x456',stableId:'1234',workspace:{id:-1337,name:'agentbrowser'},monitor:1,class:'google-chrome',title:`Polylogue proof ${'A'.repeat(32)} - Google Chrome`};
const target='A'.repeat(32); const calls=[];
assert.deepEqual(await captureProofOperator(async command=>{calls.push(command);return command[0]==='monitors'?[{id:0,activeWorkspace:{id:2,name:'2'}},{id:1,activeWorkspace:{id:3,name:'3'}}]:originalRaw;}),original);
let title;
const client={call:async(_method,{expression})=>{title=expression;return {result:{value:true}};}};
assert.deepEqual(await bindProofPopup(client,target,()=>1000,async command=>{calls.push(command);return [popupRaw];}),popup);
assert.ok(title.includes(target));
let lookups=0; let waits=0;
assert.deepEqual(await bindProofPopup(client,target,()=>1000,async()=>++lookups===1?[]:[popupRaw],async()=>{waits+=1;}),popup);
assert.equal(lookups,2);assert.equal(waits,1);
for(const [raw,restored] of [[originalRaw,true],[{...originalRaw,address:'0x789',stableId:'7'},false]]) assert.equal(await restoreProofOperator(original,popup,async command=>{calls.push(command);return command[0]==='eval'?{ok:true}:raw;}),restored);
await requireHiddenProofWorkspace(async command=>{calls.push(command);return [{id:0,activeWorkspace:{id:2,name:'2'}}];});
let primary;
try{await requireHiddenProofWorkspace(async()=>[{id:1,activeWorkspace:{id:-1337,name:'agentbrowser'}}]);}catch(error){primary=error;}
assert.equal(proofFailureReport('provider_preflight',primary).error.category,'window_visibility_refused');
for(const response of [null,{}, [], [{}]]) await assert.rejects(requireHiddenProofWorkspace(async()=>response),{message:'proof_desktop_unavailable'});
await assert.rejects(captureProofOperator(async()=>({...originalRaw,address:'private-invalid'})),{message:'proof_desktop_unavailable'});
await assert.rejects(bindProofPopup(client,target,()=>1000,async()=>[popupRaw,popupRaw]),{message:'proof_popup_binding_failed'});
await assert.rejects(restoreProofOperator(original,popup,async()=>({ok:false})),{message:'proof_desktop_unavailable'});
for(const [wire,identity] of [['1A2B',0x1a2b],['1234',0x1234],['0',0],['1fffffffffffff',Number.MAX_SAFE_INTEGER],[42,42]]) {
 const raw={...originalRaw,stableId:wire};
 const captured=await captureProofOperator(async command=>command[0]==='monitors'?[{id:0,activeWorkspace:{id:2,name:'2'}}]:raw);
 assert.equal(captured.stable_id,identity);
 const bound=await bindProofPopup(client,target,()=>1000,async()=>[{...popupRaw,stableId:wire}]);
 assert.equal(bound.stable_id,identity);
 assert.equal(await restoreProofOperator({...original,stable_id:identity},popup,async command=>{if(command[0]==='eval'){assert.ok(command[1].includes(`original.stable_id == ${identity}`));return {ok:true};}return raw;}),true);
}
for(const wire of ['', '0x1234', '-1', '1g', ' 1', '1 ', '20000000000000', null, true, -1, 1.5, Infinity, NaN, Number.MAX_SAFE_INTEGER+1]) {
 const raw={...originalRaw,stableId:wire};
 await assert.rejects(captureProofOperator(async()=>raw),{message:'proof_desktop_unavailable'});
 await assert.rejects(bindProofPopup(client,target,()=>1000,async()=>[{...popupRaw,stableId:wire}]),{message:'proof_popup_binding_failed'});
 await assert.rejects(restoreProofOperator(original,popup,async command=>command[0]==='eval'?{ok:true}:raw),{message:'proof_desktop_unavailable'});
}
assert.ok(calls.length>0);
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path(__file__).resolve().parents[3] / "browser-extension",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = {
        "ok": False,
        "error": {"phase": "provider_preflight", "category": "window_visibility_refused"},
        "cleanup": dict.fromkeys(["mutations", "permission", "receiver", "targets"], "settled"),
        "native_progress": [],
        "capture_evidence": [],
    }
    assert live_provider_proof_service.child_failure_report(json.dumps(report)) == report


def test_native_desktop_command_decodes_whole_json_before_identity_and_focus_checks() -> None:
    script = r"""
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { runProofDesktop, runChromeControl, captureProofOperator, bindProofPopup, restoreProofOperator, requireHiddenProofWorkspace } from './scripts/live_provider_proof.mjs';
function emitted(bytes, desktop=true) {
 return (_command,_args,options)=>{
  if(desktop) assert.equal(Object.hasOwn(options.env,'LD_LIBRARY_PATH'),false);
  const child=new EventEmitter();child.stdout=new EventEmitter();child.stderr=new EventEmitter();
  queueMicrotask(()=>{for(const chunk of [bytes.slice(0,3),bytes.slice(3)]) child.stdout.emit('data',Buffer.from(chunk));child.emit('close',0);});
  return child;
 };
}
const monitors=[{id:0,activeWorkspace:{id:2,name:'2'}}];
const raw={address:'0x123',stableId:'1a2b',workspace:{id:2,name:'2'},monitor:0};
const popupRaw={address:'0x456',stableId:'1234',workspace:{id:-1337,name:'agentbrowser'},monitor:0,class:'google-chrome',title:`Polylogue proof ${'A'.repeat(32)} - Google Chrome`};
for(const wire of ['1a2b','1234',42]) {
 const value={...raw,stableId:wire};
 const control=command=>runProofDesktop(command,emitted(command[0]==='eval'?'ok\n':JSON.stringify(command[0]==='monitors'?monitors:value,null,2)+'\n'));
 const original=await captureProofOperator(control);
 assert.equal(original.stable_id,typeof wire==='string'?parseInt(wire,16):wire);
 const popup=await bindProofPopup({call:async()=>({result:{value:true}})},'A'.repeat(32),()=>1000,command=>runProofDesktop(command,emitted(JSON.stringify([popupRaw],null,2))));
 assert.equal(popup.stable_id,0x1234);
 assert.equal(await restoreProofOperator(original,popup,control),true);
 await requireHiddenProofWorkspace(control);
}
for(const bytes of ['{','diagnostic\n'+JSON.stringify(raw),JSON.stringify(raw)+'\n{}',JSON.stringify(monitors)+'\n[]']) {
 await assert.rejects(runProofDesktop(['activewindow','-j'],emitted(bytes)),{message:'proof_desktop_unavailable'});
 await assert.rejects(runProofDesktop(['monitors','-j'],emitted(bytes)),{message:'proof_desktop_unavailable'});
 await assert.rejects(runProofDesktop(['clients','-j'],emitted(bytes)),{message:'proof_desktop_unavailable'});
}
assert.deepEqual(await runChromeControl(['status'],1000,emitted('diagnostic\n'+JSON.stringify({ok:true})+'\n',false)),{ok:true});
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=Path(__file__).resolve().parents[3] / "browser-extension",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_live_provider_proof_refuses_before_receiver_or_chrome_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_provider_proof_service, "require_declared_operation_context", lambda _operation: "unit")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "operator-root"))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unisolated provider proof attempted a child process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(live_provider_proof_service.ChildProofError) as failure:
        live_provider_proof_service._run_proof_locked(targets=[])
    assert failure.value.report["error"] == {
        "phase": "extension_load",
        "category": "provider_target_isolation_unavailable",
    }
    assert set(failure.value.report["cleanup"].values()) == {"not_required"}
    assert not list(tmp_path.iterdir())
    assert os.environ["POLYLOGUE_ARCHIVE_ROOT"] == str(tmp_path / "operator-root")
