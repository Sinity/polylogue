from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomllib

from devtools import dev_loop_service


def test_declared_operation_has_fixed_json_service_contract() -> None:
    descriptor = tomllib.loads(Path(".agentctl/project.toml").read_text(encoding="utf-8"))
    operation = descriptor["operations"]["dev_loop_proof"]

    assert operation["exec"] == ["devtools", "workspace", "dev-loop-service", "--json"]
    assert operation["result"] == "json"
    assert operation["cache"] == "none"
    assert operation["timeout_seconds"] == 900
    assert operation["service"] == {
        "readiness": "project-command",
        "lifetime": "job",
        "ports": {
            "api": {"environment": "POLYLOGUE_API_PORT", "range": [48800, 48863]},
            "browser_capture": {"environment": "POLYLOGUE_BROWSER_CAPTURE_PORT", "range": [48864, 48927]},
        },
    }


def test_run_proof_uses_only_agentctl_injected_ports_and_product_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setenv("POLYLOGUE_API_PORT", "48801")
    monkeypatch.setenv("POLYLOGUE_BROWSER_CAPTURE_PORT", "48865")
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

    payload = dev_loop_service.run_proof(repo_root=tmp_path / "checkout")

    assert payload == {
        "ok": True,
        "ports": {"api": 48801, "browser_capture": 48865},
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
    ("api_port", "capture_port", "message"),
    [
        (None, "48865", "POLYLOGUE_API_PORT must be injected"),
        ("48801", None, "POLYLOGUE_BROWSER_CAPTURE_PORT must be injected"),
        ("48801", "48801", "duplicate API and browser-capture ports"),
    ],
)
def test_run_proof_rejects_missing_or_duplicated_lease_ports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_port: str | None,
    capture_port: str | None,
    message: str,
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    for name, value in {
        "POLYLOGUE_API_PORT": api_port,
        "POLYLOGUE_BROWSER_CAPTURE_PORT": capture_port,
    }.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        dev_loop_service.run_proof(repo_root=tmp_path / "checkout")


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
