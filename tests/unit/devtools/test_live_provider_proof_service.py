from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools import live_provider_proof_service


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

    assert live_provider_proof_service.main(["--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "TimeoutExpired"
    assert "120" in payload["error"]["message"]
    assert terminated == [process, process]
    assert bound == [0]


@pytest.mark.parametrize("cleanup_route", ["finally", "signal", "foreign"])
def test_primary_proof_restoration_uses_receiver_mutation_owner(cleanup_route: str) -> None:
    script = r"""
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Script, createContext } from "node:vm";
import { receiverConfigurationOwner } from "./browser-extension/tests/infra/receiver_configuration.js";
import { configureReceiver, receiverConfiguration, restoreProofReceiverAfterConfiguration } from "./browser-extension/scripts/live_provider_proof.mjs";
const original = { receiverBaseUrl: "http://127.0.0.1:8765", receiverAuthToken: "neutral-original", polylogueReceiverPairing: { receiver_id: "neutral-original", state: "online" } };
const storage = structuredClone(original);
let release;
const suspended = new Promise(resolve => { release = resolve; });
let entered;
const started = new Promise(resolve => { entered = resolve; });
const chrome = { permissions: { contains: async () => true }, storage: { local: {
  async get(defaults) { const keys = Array.isArray(defaults) ? defaults : Object.keys(defaults); return Object.fromEntries(keys.filter(key => Object.hasOwn(storage,key) || !Array.isArray(defaults)).map(key => [key, Object.hasOwn(storage,key) ? storage[key] : defaults[key]])); },
  async set(values) { if (values.receiverAuthToken === "neutral-proof") { entered(); await suspended; } Object.assign(storage, values); },
  async remove(keys) { for (const key of Array.isArray(keys) ? keys : [keys]) delete storage[key]; }
} } };
const owner = receiverConfigurationOwner(chrome);
chrome.runtime = { sendMessage: message => owner.send(message) };
const worker = { async call(method, params) { assert.equal(method,"Runtime.evaluate"); return { result: { value: await new Script(params.expression).runInContext(createContext({chrome})) } }; }, close() {} };
const saved = await receiverConfiguration(worker);
assert.deepEqual(JSON.parse(JSON.stringify(saved)), original);
const proofSource = readFileSync("./browser-extension/scripts/live_provider_proof.mjs", "utf8");
let closed = false;
const browser = { close() {} };
const handlers = new Map();
const context = createContext({
  _CONTROL_TIMEOUT_MS:10000,_CDP_PORT:9222,
  Date, Promise, Object, JSON, Error, AggregateError, Math, configureReceiver, receiverConfiguration, restoreProofReceiverAfterConfiguration,
  requireExpectedServiceContext() {},
  fixedInputs: () => ({extensionRoot:"neutral",receiverBaseUrl:"http://127.0.0.1:41234",receiverToken:"neutral-proof",providers:process.env.CLEANUP_ROUTE === "foreign" ? ["neutral"] : [],timeoutMs:90000,startupTimeoutMs:30000,interactiveWaitMs:0}),
  PROVIDERS:{neutral:{url:"https://example.invalid"}},
  openAgentWindow:async()=>{storage.receiverAuthToken="neutral-independent";throw new Error("neutral_primary_failure");},
  path:{join:()=>"neutral"}, readFileSync:()=>"{}", runChromeControl:async()=>({}), waitJson:async()=>({webSocketDebuggerUrl:"neutral"}), connectCdp:async()=>browser,
  waitForExtensionWorker:async()=>worker,unpackedExtensionId:()=>"neutral",closeProofTargets:async()=>{closed=true;},
  process:{once:(signal,callback)=>handlers.set(signal,callback),exit:()=>{closed=true;}},
  activeBrowserClient:null,createdTargetIds:[],shutdownRequested:false,pendingReceiverRestore:null
});
const shutdown = proofSource.slice(proofSource.indexOf("function installShutdownCleanup("), proofSource.indexOf("async function runLiveProviderProof("));
const run = proofSource.slice(proofSource.indexOf("async function runLiveProviderProof("), proofSource.indexOf("if (process.argv[1]"));
new Script(shutdown + run + "\nglobalThis.run = runLiveProviderProof;").runInContext(context);
const running = context.run();
await started;
if (process.env.CLEANUP_ROUTE === "signal") handlers.get("SIGTERM")();
await Promise.resolve();
assert.equal(storage.receiverAuthToken,"neutral-original");
release();
if(process.env.CLEANUP_ROUTE === "foreign") {
  await assert.rejects(running, error => error.message === "proof_receiver_cleanup_failed" && error.errors[0].message === "neutral_primary_failure" && error.errors[1].message === "proof_receiver_configuration_changed");
} else await running;
for(let i=0;i<30&&!closed;i++) await new Promise(resolve=>setImmediate(resolve));
assert.equal(closed,true);
if(process.env.CLEANUP_ROUTE === "foreign") assert.equal(storage.receiverAuthToken,"neutral-independent");
else assert.deepEqual(JSON.parse(JSON.stringify(storage)), original);
process.stdout.write("owned restoration settled\n");
"""
    environment = dict(os.environ, CLEANUP_ROUTE=cleanup_route)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "owned restoration settled\n"
