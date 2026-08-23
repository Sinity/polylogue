from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import tomllib

from devtools import deployment_browser_smoke_service, deployment_smoke
from devtools.command_catalog import COMMAND_SPECS


def test_declared_browser_smoke_is_private_and_has_one_leased_cdp_port() -> None:
    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["deployment_browser_smoke"]

    assert operation["exec"] == ["python", "-m", "devtools.deployment_browser_smoke_service", "--json"]
    assert operation["result"] == "json"
    assert operation["cache"] == "none"
    assert operation["timeout_seconds"] == 180
    assert operation["service"] == {
        "readiness": "project-command",
        "lifetime": "job",
        "ports": {
            "browser_cdp": {
                "environment": "POLYLOGUE_DEPLOYMENT_BROWSER_CDP_PORT",
                "range": [48992, 49055],
            }
        },
    }
    assert all(spec.module != "devtools.deployment_browser_smoke_service" for spec in COMMAND_SPECS)


def test_declared_live_provider_proof_has_fixed_service_owned_inputs() -> None:
    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["live_provider_proof"]

    assert operation["exec"] == ["python", "-m", "devtools.live_provider_proof_service", "--json"]
    assert operation["cache"] == "none"
    assert operation["timeout_seconds"] == 180
    assert operation["service"] == {
        "readiness": "project-command",
        "lifetime": "job",
        "ports": {
            "browser_cdp": {"environment": "POLYLOGUE_LIVE_PROVIDER_CDP_PORT", "range": [49056, 49119]},
            "browser_capture": {"environment": "POLYLOGUE_LIVE_PROVIDER_RECEIVER_PORT", "range": [49120, 49183]},
        },
    }
    assert "parameters" not in operation
    assert all(spec.module != "devtools.live_provider_proof_service" for spec in COMMAND_SPECS)


def test_declared_profile_provisioning_has_no_browser_or_caller_arguments() -> None:
    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["live_provider_profile_provision"]

    assert operation == {
        "description": "Provision the fixed copied Chrome profile for the live-provider proof",
        "exec": ["python", "-m", "devtools.live_provider_profile_provision_service", "--json"],
        "pool": "interactive",
        "result": "json",
        "cache": "none",
        "timeout_seconds": 120,
        "exclusive_keys": ["polylogue:live-provider-proof"],
        "scratch": "nvme",
        "service": {"readiness": "project-command", "lifetime": "job"},
    }
    assert all(spec.module != "devtools.live_provider_profile_provision_service" for spec in COMMAND_SPECS)


def test_private_live_provider_module_imports_without_launching_chrome() -> None:
    completed = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            "import('./scripts/live_provider_proof.mjs')",
        ],
        cwd="browser-extension",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_live_provider_module_rejects_forged_environment_before_node_can_launch(tmp_path: Path) -> None:
    environment = os.environ | {
        "SINNIXD_JOB_ID": "123e4567-e89b-42d3-a456-426614174000",
        "SINNIXD_PROJECT_ID": "polylogue",
        "SINNIXD_OPERATION": "live_provider_proof",
        "POLYLOGUE_LIVE_PROVIDER_CDP_PORT": "49056",
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


def test_private_browser_service_uses_only_fixed_target_and_leased_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deployment_browser_smoke_service, "require_declared_service_context", lambda operation: f"unit-{operation}"
    )
    monkeypatch.setenv("SINNIXD_PROJECT_ID", "polylogue")
    monkeypatch.setenv("SINNIXD_OPERATION", "deployment_browser_smoke")
    monkeypatch.setenv("POLYLOGUE_DEPLOYMENT_BROWSER_CDP_PORT", "48992")
    called: dict[str, object] = {}

    def fake_probe(url: str, **kwargs: object) -> deployment_smoke.BrowserRenderProbe:
        called["url"] = url
        called.update(kwargs)
        return deployment_smoke.BrowserRenderProbe(url=url, executable="/nix/store/chromium", exit_code=0, ok=True)

    monkeypatch.setattr(deployment_smoke, "_probe_browser_render", fake_probe)

    result = deployment_browser_smoke_service.run_smoke(timeout_s=7)

    assert result.ok is True
    assert called == {
        "url": "http://127.0.0.1:8766/",
        "path": deployment_smoke.SYSTEMWIDE_PATH,
        "timeout_s": 7,
        "executable": None,
        "debugging_port": 48992,
    }


@pytest.mark.parametrize(
    ("project", "operation", "port"),
    [
        ("other", "deployment_browser_smoke", "48992"),
        ("polylogue", "other", "48992"),
        ("polylogue", "deployment_browser_smoke", "49056"),
    ],
)
def test_private_browser_service_context_guard_rejects_invalid_shell_context(
    monkeypatch: pytest.MonkeyPatch, project: str, operation: str, port: str
) -> None:
    monkeypatch.setattr(
        deployment_browser_smoke_service,
        "require_declared_service_context",
        lambda operation: (
            (_ for _ in ()).throw(ValueError("fixed service context"))
            if os.environ.get("SINNIXD_PROJECT_ID") != "polylogue" or os.environ.get("SINNIXD_OPERATION") != operation
            else f"unit-{operation}"
        ),
    )
    monkeypatch.setenv("SINNIXD_PROJECT_ID", project)
    monkeypatch.setenv("SINNIXD_OPERATION", operation)
    monkeypatch.setenv("POLYLOGUE_DEPLOYMENT_BROWSER_CDP_PORT", port)

    with pytest.raises(ValueError, match="fixed service context|fixed deployment-browser port range"):
        deployment_browser_smoke_service._service_context_port()
