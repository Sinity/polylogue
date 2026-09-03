from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

from devtools import deployment_browser_smoke_service
from devtools.command_catalog import COMMAND_SPECS


def test_declared_browser_smoke_has_no_private_browser_service_lease() -> None:
    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["deployment_browser_smoke"]

    assert operation["exec"] == ["python", "-m", "devtools.deployment_browser_smoke_service", "--json"]
    assert operation["result"] == "json"
    assert operation["cache"] == "none"
    assert operation["timeout_seconds"] == 180
    assert "service" not in operation
    assert all(spec.module != "devtools.deployment_browser_smoke_service" for spec in COMMAND_SPECS)


def test_declared_live_provider_proof_declares_no_port_lease() -> None:
    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["live_provider_proof"]

    assert operation["exec"] == ["python", "-m", "devtools.live_provider_proof_service", "--json"]
    assert operation["cache"] == "none"
    assert operation["timeout_seconds"] == 180
    assert "service" not in operation
    assert "parameters" not in operation
    assert all(spec.module != "devtools.live_provider_proof_service" for spec in COMMAND_SPECS)


def test_sinnixd_parser_accepts_the_unleased_shared_chrome_operation() -> None:
    """Cross-contract proof against the production Sinnixd descriptor parser."""
    repository_root = Path(__file__).resolve().parents[3]
    sinnix_root = Path("/realm/project/sinnix")
    package_roots = (
        sinnix_root / "pkgs" / "sinnixd",
        sinnix_root / "pkgs" / "sinnix-mcp",
        sinnix_root / "pkgs" / "sinnix-lib",
    )
    parser_program = """
import json
import sys
from pathlib import Path

from sinnixd.projects import load_project_adapter

adapter = load_project_adapter(Path(sys.argv[1]))
proof = adapter.operation("deployment_browser_smoke")
print(json.dumps({
    "project_id": adapter.project_id,
    "operation_count": len(adapter.operations),
    "proof": {
        "command": proof.command,
        "parameters": proof.parameters,
    },
    "service_operations": sorted(
        operation.name for operation in adapter.operations if getattr(operation, "service", None) is not None
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", parser_program, str(repository_root)],
        cwd=sinnix_root,
        env=os.environ | {"PYTHONPATH": os.pathsep.join(map(str, package_roots))},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    parsed = json.loads(completed.stdout)
    assert parsed["project_id"] == "polylogue"
    assert parsed["operation_count"] >= 6
    assert parsed["proof"] == {
        "command": ["python", "-m", "devtools.deployment_browser_smoke_service", "--json"],
        "parameters": [],
    }
    # Sinnixd allocates no ports; every proof binds its own.
    assert parsed["service_operations"] == []
    assert all(spec.module != "devtools.deployment_browser_smoke_service" for spec in COMMAND_SPECS)


def test_private_live_provider_module_imports_without_launching_chrome() -> None:
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", "import('./scripts/live_provider_proof.mjs')"],
        cwd="browser-extension",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_live_provider_rejects_an_agent_window_without_hidden_parking() -> None:
    completed = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            "import('./scripts/live_provider_proof.mjs').then(m => { try { m.assertAgentWindow({ id: 'A'.repeat(32), url: 'https://chatgpt.com/', parked: false, workspace: 'agentbrowser', show_with: 'F7' }, 'https://chatgpt.com/'); process.exitCode = 2; } catch (error) { console.log(error.message); } })",
        ],
        cwd="browser-extension",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "verified hidden on agentbrowser" in completed.stdout


def test_live_provider_accepts_a_parked_control_plane_window() -> None:
    completed = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            "import('./scripts/live_provider_proof.mjs').then(m => console.log(m.assertAgentWindow({ id: 'A'.repeat(32), url: 'https://chatgpt.com/', parked: true, workspace: 'agentbrowser', show_with: 'F7' }, 'https://chatgpt.com/')))",
        ],
        cwd="browser-extension",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "A" * 32


def test_live_provider_module_rejects_forged_environment_before_node_can_launch(tmp_path: Path) -> None:
    environment = os.environ | {
        "SINNIXD_JOB_ID": "123e4567-e89b-42d3-a456-426614174000",
        "SINNIXD_PROJECT_ID": "polylogue",
        "SINNIXD_OPERATION": "live_provider_proof",
        "POLYLOGUE_LIVE_PROVIDER_RECEIVER_PORT": "49120",
        "POLYLOGUE_LIVE_PROVIDER_RECEIVER_TOKEN": "forged",
        "TMPDIR": str(tmp_path),
    }
    completed = subprocess.run(
        ["node", "scripts/live_provider_proof.mjs"],
        cwd="browser-extension",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "matching Sinnixd transient unit" in completed.stderr


def test_shared_browser_service_launches_only_the_fixed_control_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        deployment_browser_smoke_service, "require_declared_operation_context", lambda operation: f"unit-{operation}"
    )
    launched: dict[str, object] = {}
    process = type(
        "Process",
        (),
        {
            "returncode": 0,
            "communicate": lambda self, **_kwargs: (
                '{"ok":true,"render":{"url":"http://127.0.0.1:8766/","dom_bytes":12,"screenshot_bytes":3,"target_closed":true}}',
                "",
            ),
        },
    )()
    monkeypatch.setenv("POLYLOGUE_TEST_CDP_OPTION", "48992")
    monkeypatch.setenv("POLYLOGUE_TEST_PROFILE_OPTION", "forged")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: (launched.update(command=command, kwargs=kwargs), process)[1],
    )
    monkeypatch.setattr(deployment_browser_smoke_service, "terminate_process_group", lambda _process: None)

    result = deployment_browser_smoke_service.run_smoke(repo_root=tmp_path, timeout_s=7)

    assert result == {
        "ok": True,
        "render": {
            "url": "http://127.0.0.1:8766/",
            "dom_bytes": 12,
            "screenshot_bytes": 3,
            "target_closed": True,
        },
    }
    assert launched["command"] == ["node", "scripts/deployment_shared_chrome_smoke.mjs"]
    kwargs = launched["kwargs"]
    assert isinstance(kwargs, dict)
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert not {name for name in environment if "CDP" in name or "PROFILE" in name or "BROWSER_EXECUTABLE" in name}


def test_shared_browser_service_context_guard_rejects_invalid_shell_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deployment_browser_smoke_service,
        "require_declared_operation_context",
        lambda _operation: (_ for _ in ()).throw(ValueError("fixed operation context")),
    )

    with pytest.raises(ValueError, match="fixed operation context"):
        deployment_browser_smoke_service.run_smoke()


def test_shared_chrome_deployment_workflow_records_render_evidence_and_closes_only_its_target() -> None:
    program = """
import { runDeploymentSharedChromeSmoke } from './scripts/deployment_shared_chrome_smoke.mjs';

const calls = [];
const targetId = 'A'.repeat(32);
const control = async (args) => {
  calls.push(args);
  if (args[0] === 'agent-window') return Buffer.from(JSON.stringify({ id: targetId, url: 'http://127.0.0.1:8766/', parked: true, workspace: 'agentbrowser', show_with: 'F7' }));
  if (args[0] === 'get-html') return Buffer.from('<html><head><title>Polylogue</title></head><body><div id="conv-header"></div></body></html>');
  if (args[0] === 'screenshot') return Buffer.from('png');
  if (args[0] === 'close' && args[1] !== targetId) throw new Error('attempted to close an unowned target');
  return Buffer.alloc(0);
};
const result = await runDeploymentSharedChromeSmoke({ control });
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
    assert payload["calls"] == [
        ["status"],
        ["agent-window", "--url", "http://127.0.0.1:8766/"],
        [
            "await",
            "A" * 32,
            "--js",
            "document.readyState === 'complete'",
            "--timeout-sec",
            "30",
        ],
        ["get-html", "A" * 32],
        ["screenshot", "A" * 32, "--format", "png"],
        ["close", "A" * 32],
    ]
    assert payload["result"] == {
        "ok": True,
        "render": {
            "url": "http://127.0.0.1:8766/",
            "dom_bytes": len(
                '<html><head><title>Polylogue</title></head><body><div id="conv-header"></div></body></html>'
            ),
            "screenshot_bytes": 3,
            "target_closed": True,
        },
    }


def test_shared_chrome_deployment_workflow_reclaims_only_created_target_after_readiness_failure() -> None:
    program = """
import { runDeploymentSharedChromeSmoke } from './scripts/deployment_shared_chrome_smoke.mjs';

const calls = [];
const targetId = 'B'.repeat(32);
const control = async (args) => {
  calls.push(args);
  if (args[0] === 'agent-window') return Buffer.from(JSON.stringify({ id: targetId, url: 'http://127.0.0.1:8766/', parked: true, workspace: 'agentbrowser', show_with: 'F7' }));
  if (args[0] === 'await') throw new Error('page never became ready');
  if (args[0] === 'close' && args[1] !== targetId) throw new Error('attempted to close an unowned target');
  return Buffer.alloc(0);
};
try {
  await runDeploymentSharedChromeSmoke({ control });
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
    assert payload["message"] == "page never became ready"
    assert payload["calls"][-1] == ["close", "B" * 32]


def test_shared_chrome_deployment_workflow_rejects_an_error_document_and_closes_its_target() -> None:
    program = """
import { runDeploymentSharedChromeSmoke } from './scripts/deployment_shared_chrome_smoke.mjs';

const calls = [];
const targetId = 'C'.repeat(32);
const control = async (args) => {
  calls.push(args);
  if (args[0] === 'agent-window') return Buffer.from(JSON.stringify({ id: targetId, url: 'http://127.0.0.1:8766/', parked: true, workspace: 'agentbrowser', show_with: 'F7' }));
  if (args[0] === 'get-html') return Buffer.from('<html><head><title>127.0.0.1</title></head><body>ERR_CONNECTION_REFUSED</body></html>');
  if (args[0] === 'screenshot') return Buffer.from('png');
  if (args[0] === 'close' && args[1] !== targetId) throw new Error('attempted to close an unowned target');
  return Buffer.alloc(0);
};
try {
  await runDeploymentSharedChromeSmoke({ control });
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
    assert payload["message"] == "shared Chrome deployment render did not contain the Polylogue root marker"
    assert payload["calls"][-1] == ["close", "C" * 32]


def test_shared_chrome_signal_cleanup_closes_only_the_owned_target_once() -> None:
    program = r"""
import { createOwnedTargetCleanup } from './scripts/shared_chrome_proof_cleanup.mjs';

const targetId = 'D'.repeat(32);
const unrelatedTargetId = 'E'.repeat(32);
const cleanup = createOwnedTargetCleanup({
  targetId,
  control: async (args) => {
    if (args[0] !== 'close' || args[1] !== targetId) throw new Error('attempted to close an unowned target');
    process.stdout.write(JSON.stringify({ closed: args[1] }) + '\n');
  },
});
process.stdout.write(JSON.stringify({ ready: true, unrelatedTargetId }) + '\n');
setInterval(() => {}, 1000);
setTimeout(() => process.kill(process.pid, 'SIGTERM'), 10);
await new Promise(() => {});
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", program],
        cwd="browser-extension",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == -15, completed.stderr
    events = [json.loads(line) for line in completed.stdout.splitlines() if line]
    assert events == [{"ready": True, "unrelatedTargetId": "E" * 32}, {"closed": "D" * 32}]
