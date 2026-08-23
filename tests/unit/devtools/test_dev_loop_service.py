from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomllib

from devtools import dev_loop_service
from devtools.command_catalog import COMMAND_SPECS


def test_declared_operation_has_fixed_json_service_contract() -> None:
    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["dev_loop_proof"]

    assert operation["exec"] == ["python", "-m", "devtools.dev_loop_service", "--json"]
    assert operation["result"] == "json"
    assert operation["cache"] == "none"
    assert operation["timeout_seconds"] == 900
    assert operation["service"] == {
        "readiness": "project-command",
        "lifetime": "job",
        "ports": {
            "api": {"environment": "POLYLOGUE_API_PORT", "range": [48800, 48863]},
            "browser_capture": {"environment": "POLYLOGUE_BROWSER_CAPTURE_PORT", "range": [48864, 48927]},
            "browser_cdp": {"environment": "POLYLOGUE_BROWSER_CDP_PORT", "range": [48928, 48991]},
        },
    }
    assert descriptor["operations"]["verify_all"]["timeout_seconds"] == 14400
    assert all(spec.module != "devtools.dev_loop_service" for spec in COMMAND_SPECS)
    assert all(spec.module != "devtools.deployment_browser_smoke_service" for spec in COMMAND_SPECS)


def _fixed_service_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SINNIXD_PROJECT_ID", "polylogue")
    monkeypatch.setenv("SINNIXD_OPERATION", "dev_loop_proof")
    monkeypatch.setenv("POLYLOGUE_API_PORT", "48801")
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_PORT", "48865")
    monkeypatch.setenv("POLYLOGUE_BROWSER_CDP_PORT", "48929")


def test_run_proof_uses_only_agentctl_injected_ports_and_product_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_service_context(monkeypatch)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(dev_loop_service, "initialize_archive_tier_files", lambda **_kwargs: object())
    monkeypatch.setattr(dev_loop_service, "run_receiver_smoke", lambda **_kwargs: {"ok": True})
    started: dict[str, object] = {}
    monkeypatch.setattr(dev_loop_service, "_start_daemon", lambda **kwargs: started.update(kwargs))
    monkeypatch.setattr(dev_loop_service, "_await_api", lambda **_kwargs: None)
    monkeypatch.setattr(
        dev_loop_service,
        "_run_provider_capture",
        lambda **_kwargs: {
            "chatgpt": {"provider": "chatgpt", "provider_session_id": "agentctl-proof"},
            "claude": {"provider": "claude-ai", "provider_session_id": "agentctl-proof"},
        },
    )
    monkeypatch.setattr(dev_loop_service, "_poll_archive_state", lambda **_kwargs: True)
    monkeypatch.setattr(dev_loop_service, "_fetch_api_messages", lambda **_kwargs: True)

    payload = dev_loop_service.run_proof()

    assert payload == {
        "ok": True,
        "ports": {"api": 48801, "browser_capture": 48865, "browser_cdp": 48929},
        "receiver_auth": {"ok": True},
        "provider_capture": {
            "providers": ["chatgpt", "claude"],
            "archive_converged": True,
            "api_converged": True,
        },
    }
    assert started["api_port"] == 48801
    assert started["capture_port"] == 48865
    environment = started["environment"]
    assert isinstance(environment, dict)
    assert environment["POLYLOGUE_API_PORT"] == "48801"
    assert environment["POLYLOGUE_BROWSER_CAPTURE_PORT"] == "48865"


@pytest.mark.parametrize(
    ("environment", "value", "message"),
    [
        ("POLYLOGUE_API_PORT", None, "POLYLOGUE_API_PORT must be present"),
        ("POLYLOGUE_BROWSER_CAPTURE_PORT", "48801", "outside the fixed dev-loop service port range"),
        ("POLYLOGUE_BROWSER_CDP_PORT", "49000", "outside the fixed dev-loop service port range"),
        ("SINNIXD_OPERATION", "other", "rejects execution outside its fixed service context"),
    ],
)
def test_service_context_guard_rejects_missing_or_wrong_shell_context(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    value: str | None,
    message: str,
) -> None:
    _fixed_service_context(monkeypatch)
    if value is None:
        monkeypatch.delenv(environment, raising=False)
    else:
        monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match=message):
        dev_loop_service._require_agentctl_service_context()


def test_receiver_smoke_proves_auth_rejection_and_accepted_capture(tmp_path: Path) -> None:
    payload = dev_loop_service.run_receiver_smoke(spool_path=tmp_path / "spool")

    assert payload["ok"] is True
    assert payload["rejected_status"] == 401
    assert payload["accepted_status"] == 202
    assert isinstance(payload["artifact_ref"], str)


def test_main_emits_one_bounded_json_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(dev_loop_service, "run_proof", lambda: (_ for _ in ()).throw(ValueError("x" * 600)))

    assert dev_loop_service.main(["--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert len(payload["error"]["message"]) == 512
