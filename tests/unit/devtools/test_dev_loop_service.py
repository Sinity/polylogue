from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
import tomllib

from devtools import dev_loop_service
from devtools.command_catalog import COMMAND_SPECS


def test_declared_operation_has_a_json_contract_and_no_retired_keys() -> None:
    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["dev_loop_proof"]

    assert operation["exec"] == ["python", "-m", "devtools.dev_loop_service", "--json"]
    assert operation["result"] == "json"
    assert operation["cache"] == "none"
    assert operation["timeout_seconds"] == 900
    # sinnixd has no port-lease/service contract: the proof binds its own ports.
    retired = {"service", "estimate_memory_bytes", "exclusive_keys", "scratch", "supersede"}
    assert all(not retired & set(declared) for declared in descriptor["operations"].values())
    verify_all = descriptor["operations"]["verify_all"]
    assert verify_all["timeout_seconds"] == 14400
    assert verify_all["exec"] == ["env", "POLYLOGUE_PYTEST_WORKERS=2", "devtools", "verify", "--all"]
    assert {"WAYLAND_DISPLAY", "DISPLAY"} <= set(descriptor["environment"]["inherit"])
    assert all(spec.module != "devtools.dev_loop_service" for spec in COMMAND_SPECS)
    assert all(spec.module != "devtools.deployment_browser_smoke_service" for spec in COMMAND_SPECS)


def _fixed_service_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_loop_service, "require_declared_operation_context", lambda operation: f"unit-{operation}")
    monkeypatch.setenv("AGENTCTL_PROJECT_ID", "polylogue")
    monkeypatch.setenv("AGENTCTL_OPERATION", "dev_loop_proof")
    monkeypatch.setenv("AGENTCTL_JOB_ID", "123e4567-e89b-42d3-a456-426614174000")


def test_run_proof_uses_self_bound_free_ports_and_product_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_service_context(monkeypatch)
    monkeypatch.setattr(dev_loop_service, "_free_loopback_ports", lambda count: [48801, 48865][:count])
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    initialized: list[Path] = []
    monkeypatch.setattr(dev_loop_service, "initialize_active_archive_root", initialized.append)
    monkeypatch.setattr(dev_loop_service, "run_receiver_smoke", lambda **_kwargs: {"ok": True})
    started: dict[str, object] = {}
    monkeypatch.setattr(dev_loop_service, "_start_daemon", lambda **kwargs: started.update(kwargs))
    monkeypatch.setattr(dev_loop_service, "terminate_process_group", lambda _process: None)
    monkeypatch.setattr(dev_loop_service, "_await_api", lambda **_kwargs: None)
    monkeypatch.setattr(dev_loop_service, "_run_shared_chrome_control", lambda **_kwargs: None)
    monkeypatch.setattr(
        dev_loop_service,
        "_submit_deterministic_captures",
        lambda **_kwargs: {
            "chatgpt": {"provider": "chatgpt", "provider_session_id": "agentctl-proof-chatgpt"},
            "claude-ai": {"provider": "claude-ai", "provider_session_id": "agentctl-proof-claude-ai"},
        },
    )
    monkeypatch.setattr(
        dev_loop_service,
        "_poll_archive_state",
        lambda **kwargs: {"indexed_session_id": f"indexed:{kwargs['provider_session_id']}"},
    )
    monkeypatch.setattr(dev_loop_service, "_fetch_api_messages", lambda **_kwargs: True)

    payload = dev_loop_service.run_proof()

    assert payload == {
        "ok": True,
        "ports": {"api": 48801, "browser_capture": 48865},
        "receiver_auth": {"ok": True},
        "shared_chrome": {"ok": True},
        "provider_capture": {
            "providers": ["chatgpt", "claude-ai"],
            "archive_converged": True,
            "api_converged": True,
        },
    }
    assert started["api_port"] == 48801
    assert started["capture_port"] == 48865
    assert initialized == [tmp_path / "scratch" / "polylogue-dev-loop-proof" / "archive"]
    environment = started["environment"]
    assert isinstance(environment, dict)
    assert environment["POLYLOGUE_API_PORT"] == "48801"
    assert environment["POLYLOGUE_BROWSER_CAPTURE_PORT"] == "48865"


def test_started_daemon_uses_fixed_proof_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    launched: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> object:
        launched.update(command=command, kwargs=kwargs)
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    dev_loop_service._start_daemon(
        repo_root=tmp_path,
        environment={},
        artifact_root=artifact_root,
        api_port=48801,
        capture_port=48865,
    )

    command = launched["command"]
    assert isinstance(command, list)
    token_index = command.index("--browser-capture-auth-token")
    assert command[token_index + 1] == dev_loop_service._RECEIVER_TOKEN
    api_token_index = command.index("--api-auth-token")
    assert command[api_token_index + 1] == dev_loop_service._API_TOKEN


def test_convergence_reads_use_the_matching_service_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[dict[str, object]] = []

    def fake_get(url: str, **kwargs: object) -> tuple[int, dict[str, object]]:
        observed.append({"url": url, **kwargs})
        if "archive-state" in url:
            return 200, {"raw_row_exists": True, "indexed_session_exists": True, "indexed_session_id": "indexed"}
        return 200, {
            "session_id": "indexed",
            "messages": [
                {
                    "id": "message-1",
                    "role": "user",
                    "text": "proof",
                    "target_ref": {
                        "target_type": "message",
                        "target_id": "message-1",
                        "session_id": "indexed",
                        "message_id": "message-1",
                        "identity_key": "message:indexed:message-1",
                    },
                }
            ],
        }

    monkeypatch.setattr(dev_loop_service, "_http_get_json", fake_get)

    assert (
        dev_loop_service._poll_archive_state(
            receiver_url="http://receiver",
            provider="chatgpt",
            provider_session_id="proof",
            timeout_s=0.1,
        )
        is not None
    )
    assert dev_loop_service._fetch_api_messages(api_url="http://api", session_id="indexed") is True
    assert observed == [
        {
            "url": "http://receiver/v1/archive-state?provider=chatgpt&provider_session_id=proof",
            "bearer_token": dev_loop_service._RECEIVER_TOKEN,
        },
        {
            "url": "http://api/api/sessions/indexed/messages?limit=5",
            "bearer_token": dev_loop_service._API_TOKEN,
        },
    ]


def test_run_proof_rejects_one_malformed_expected_provider_before_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixed_service_context(monkeypatch)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(dev_loop_service, "initialize_active_archive_root", lambda _root: None)
    monkeypatch.setattr(dev_loop_service, "run_receiver_smoke", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(dev_loop_service, "_start_daemon", lambda **_kwargs: object())
    monkeypatch.setattr(dev_loop_service, "terminate_process_group", lambda _process: None)
    monkeypatch.setattr(dev_loop_service, "_await_api", lambda **_kwargs: None)
    monkeypatch.setattr(dev_loop_service, "_run_shared_chrome_control", lambda **_kwargs: None)
    monkeypatch.setattr(
        dev_loop_service,
        "_submit_deterministic_captures",
        lambda **_kwargs: {
            "chatgpt": {"provider": "chatgpt", "provider_session_id": "capture-id"},
            "claude-ai": {"provider": "claude-ai"},
        },
    )
    monkeypatch.setattr(
        dev_loop_service,
        "_poll_archive_state",
        lambda **_kwargs: pytest.fail("malformed captures must fail before convergence polling"),
    )

    with pytest.raises(RuntimeError, match="entries were malformed: claude-ai"):
        dev_loop_service.run_proof()


def test_free_loopback_ports_are_distinct_and_bindable() -> None:
    ports = dev_loop_service._free_loopback_ports(2)

    assert len(set(ports)) == 2
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", port))


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "wrong", "messages": [{"id": "message-1", "role": "user", "target_ref": {}}]},
        {"session_id": "indexed", "messages": [{}]},
        {"session_id": "indexed", "messages": []},
    ],
)
def test_api_message_convergence_rejects_wrong_or_malformed_response(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> None:
    monkeypatch.setattr(dev_loop_service, "_http_get_json", lambda *_args, **_kwargs: (200, payload))

    assert dev_loop_service._fetch_api_messages(api_url="http://api", session_id="indexed") is False


@pytest.mark.parametrize(
    ("environment", "value", "message"),
    [
        ("AGENTCTL_OPERATION", "other", "rejects execution outside its fixed service context"),
        ("AGENTCTL_PROJECT_ID", "other", "rejects execution outside its fixed service context"),
    ],
)
def test_operation_context_guard_rejects_missing_or_wrong_shell_context(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    value: str | None,
    message: str,
) -> None:
    _fixed_service_context(monkeypatch)
    monkeypatch.setattr(
        dev_loop_service,
        "require_declared_operation_context",
        lambda operation: (
            (_ for _ in ()).throw(ValueError("rejects execution outside its fixed service context"))
            if os.environ.get("AGENTCTL_PROJECT_ID") != "polylogue" or os.environ.get("AGENTCTL_OPERATION") != operation
            else f"unit-{operation}"
        ),
    )
    if value is None:
        monkeypatch.delenv(environment, raising=False)
    else:
        monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match=message):
        dev_loop_service._require_agentctl_operation_context()


def test_receiver_smoke_proves_auth_rejection_and_accepted_capture(tmp_path: Path) -> None:
    payload = dev_loop_service.run_receiver_smoke(spool_path=tmp_path / "spool")

    assert payload["ok"] is True
    assert payload["rejected_status"] == 401
    assert payload["accepted_status"] == 202
    assert isinstance(payload["artifact_ref"], str)


def test_shared_chrome_control_is_the_only_dev_loop_browser_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = type(
        "Process",
        (),
        {
            "args": ["node", "scripts/dev_loop_shared_chrome_proof.mjs"],
            "returncode": 0,
            "communicate": lambda self, **_kwargs: ('{"ok":true}\n', ""),
        },
    )()
    launched: dict[str, object] = {}
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: (launched.update(command=command, kwargs=kwargs), process)[1],
    )
    monkeypatch.setattr(dev_loop_service, "terminate_process_group", lambda _process: None)

    dev_loop_service._run_shared_chrome_control(repo_root=tmp_path)

    assert launched["command"] == ["node", "scripts/dev_loop_shared_chrome_proof.mjs"]
    kwargs = cast(dict[str, Any], launched["kwargs"])
    environment = cast(dict[str, str], kwargs["env"])
    assert environment["POLYLOGUE_DEV_LOOP_EXTENSION_ROOT"] == str(tmp_path / "browser-extension")
    assert not {name for name in environment if name.startswith("POLYLOGUE_") and ("CDP" in name or "PROFILE" in name)}


def test_shared_chrome_control_preserves_bounded_child_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = type(
        "Process",
        (),
        {
            "args": ["node", "scripts/dev_loop_shared_chrome_proof.mjs"],
            "returncode": 1,
            "communicate": lambda self, **_kwargs: ("", "first line\ncontrol boundary rejected the window\n"),
        },
    )()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(dev_loop_service, "terminate_process_group", lambda _process: None)

    with pytest.raises(RuntimeError, match="control boundary rejected the window") as failure:
        dev_loop_service._run_shared_chrome_control(repo_root=tmp_path)

    assert "\n" not in str(failure.value)


def test_deterministic_captures_still_exercise_receiver_provider_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[dict[str, object]] = []

    def accepted(**kwargs: object) -> tuple[int, dict[str, object]]:
        observed.append(kwargs)
        return 202, {"ok": True}

    monkeypatch.setattr(dev_loop_service, "_receiver_post", accepted)

    captures = dev_loop_service._submit_deterministic_captures(capture_port=48865, session_id="proof")

    assert captures == {
        "chatgpt": {"provider": "chatgpt", "provider_session_id": "proof-chatgpt"},
        "claude-ai": {"provider": "claude-ai", "provider_session_id": "proof-claude-ai"},
    }
    payloads = [cast(dict[str, dict[str, str]], entry["body"]) for entry in observed]
    assert [payload["session"]["provider"] for payload in payloads] == ["chatgpt", "claude-ai"]
    assert all(entry["token"] == dev_loop_service._RECEIVER_TOKEN for entry in observed)


def test_shared_chrome_node_boundary_launches_only_sinnix_control() -> None:
    program = """
import { EventEmitter } from 'node:events';
import { runChromeControl } from './scripts/dev_loop_shared_chrome_proof.mjs';

const calls = [];
const child = new EventEmitter();
child.stdout = new EventEmitter();
child.stderr = new EventEmitter();
child.kill = () => undefined;
const result = await runChromeControl(['status'], 1000, (command, args, options) => {
  calls.push({ command, args, options });
  queueMicrotask(() => child.emit('close', 0));
  return child;
});
console.log(JSON.stringify({ calls, result }));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", program],
        cwd="browser-extension",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "calls": [
            {
                "command": "/home/sinity/.local/bin/sinnix-chrome-control",
                "args": ["status"],
                "options": {"stdio": ["ignore", "pipe", "pipe"]},
            }
        ],
        "result": {},
    }


def test_shared_chrome_node_workflow_closes_only_its_returned_target() -> None:
    program = """
import { runSharedChromeControlWorkflow } from './scripts/dev_loop_shared_chrome_proof.mjs';

const calls = [];
const control = async (args) => {
  calls.push(args);
  if (args[0] === 'agent-window') return { id: 'A'.repeat(32), url: 'about:blank', parked: true, workspace: 'agentbrowser', show_with: 'F7' };
  if (args[0] === 'close' && args[1] !== 'A'.repeat(32)) throw new Error('attempted to close an unowned target');
  return {};
};
const result = await runSharedChromeControlWorkflow({ extensionRoot: '.', control });
console.log(JSON.stringify({ calls, result }));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", program],
        cwd="browser-extension",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "calls": [
            ["status"],
            ["load-extension", "--path", "."],
            ["agent-window", "--url", "about:blank"],
            ["close", "A" * 32],
        ],
        "result": {"ok": True, "shared_chrome": {"extension_loaded": True, "target_closed": True}},
    }


def test_shared_chrome_node_workflow_rejects_special_workspace_and_reclaims_target() -> None:
    program = """
import { runSharedChromeControlWorkflow } from './scripts/dev_loop_shared_chrome_proof.mjs';

const calls = [];
const control = async (args) => {
  calls.push(args);
  if (args[0] === 'agent-window') return { id: 'B'.repeat(32), url: 'about:blank', parked: true, workspace: ['special', 'agentbrowser'].join(':'), show_with: 'F7' };
  if (args[0] === 'close' && args[1] !== 'B'.repeat(32)) throw new Error('attempted to close an unowned target');
  return {};
};
try {
  await runSharedChromeControlWorkflow({ extensionRoot: '.', control });
  process.exitCode = 2;
} catch (error) {
  console.log(JSON.stringify({ message: error.message, calls }));
}
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", program],
        cwd="browser-extension",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert "agentbrowser" in payload["message"]
    assert payload["calls"][-1] == ["close", "B" * 32]


def test_api_readiness_uses_the_unauthenticated_liveness_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def live(url: str, **_kwargs: object) -> tuple[int, dict[str, str]]:
        observed.append(url)
        return 200, {"status": "alive"}

    monkeypatch.setattr(
        dev_loop_service,
        "_http_get_json",
        live,
    )

    dev_loop_service._await_api(base_url="http://127.0.0.1:48801", timeout_s=0.1)

    assert observed == ["http://127.0.0.1:48801/healthz/live"]


def test_main_emits_one_bounded_json_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(dev_loop_service, "run_proof", lambda: (_ for _ in ()).throw(ValueError("x" * 600)))

    assert dev_loop_service.main(["--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert len(payload["error"]["message"]) == 512
