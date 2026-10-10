from __future__ import annotations

import hashlib
import http.client
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from devtools import live_provider_proof_service as service
from devtools.native_transport_proof import NativeProofCustodyError
from tests.infra.live_provider_proof import native_proof_artifact


@pytest.mark.parametrize("fault", ["none", "child", "identity", "isolation", "constructor", "start", "custody"])
def test_owned_provider_runtime_uses_private_native_authority_and_settles_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    profile = tmp_path / "chrome"
    hosts = profile / "NativeMessagingHosts"
    hosts.mkdir(parents=True)
    operator_manifest = hosts / "com.polylogue.browser_capture.json"
    operator_manifest.write_text("operator untouched")
    wrapper = tmp_path / "native-wrapper"
    wrapper.write_text("#!/bin/sh\nexit 0\n")
    wrapper.chmod(0o700)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("devtools.live_provider_proof_service.tempfile.tempdir", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_RECEIVER_AUTH_TOKEN", "operator-secret")
    monkeypatch.setattr(service, "require_declared_operation_context", lambda _operation: "neutral")
    monkeypatch.setattr("devtools.native_transport_proof.shutil.which", lambda _name: str(wrapper))
    before = dict(os.environ)
    constructor: dict[str, Any] = {}
    processes: list[object] = []

    if fault == "start":

        class FailedThread:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("neutral startup failure")

        monkeypatch.setattr(service, "Thread", FailedThread)

    def build(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert args[1].endswith("owned_provider_extension.mjs")
        scope = json.loads(Path(args[args.index("--scope") + 1]).read_text())
        constructor.update(scope)
        env = kwargs["env"]
        assert "POLYLOGUE_RECEIVER_AUTH_TOKEN" not in env
        assert "token" not in json.dumps(scope).lower()
        if fault == "constructor":
            raise subprocess.CalledProcessError(1, args)
        destination = Path(args[args.index("--destination") + 1])
        destination.mkdir()
        host = args[args.index("--host") + 1]
        constructor["host"] = host
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                {
                    "kind": "owned-provider-runtime",
                    "owned_targets_bound": False,
                    "host_name": host,
                    "extension_id": "a" * 32,
                }
            ),
        )

    class Process:
        returncode = 1 if fault == "child" else 0

        def __init__(self, args: list[str], **kwargs: Any) -> None:
            assert args == ["node", "scripts/owned_provider_proof.mjs"]
            env = kwargs["env"]
            assert "POLYLOGUE_RECEIVER_AUTH_TOKEN" not in env
            assert not any("TOKEN" in key for key in env if key.startswith("POLYLOGUE_"))
            manifest = json.loads((hosts / (constructor["host"] + ".json")).read_text())
            assert manifest["allowed_origins"] == ["chrome-extension://" + "a" * 32 + "/"]
            self.root = Path(env["TMPDIR"])
            self.extension = env["POLYLOGUE_LIVE_PROVIDER_EXTENSION_ROOT"]
            if fault == "custody":
                (hosts / (constructor["host"] + ".json")).write_text("changed neutral manifest")
            processes.append(self)

        def communicate(self) -> tuple[str, str]:
            # Exercise the actual private listener, with no operator credential.
            endpoint = urlsplit(constructor["receiverUrl"])
            connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port)
            connection.request("GET", "/v1/status")
            response = connection.getresponse()
            assert response.status == 401
            response.read()
            connection.close()
            spool = self.root / "browser-capture"
            envelope, turns, attachments = native_proof_artifact(self.root, "native-rich-blocks-v1.json")
            native_id = envelope["session"]["provider_session_id"]
            assert constructor["targets"][0]["nativeId"] == native_id
            literal = json.dumps(envelope).encode()
            (spool / "artifact.json").write_bytes(literal)
            receipt = {
                "artifact_ref": "artifact.json",
                "artifact_sha256": hashlib.sha256(literal).hexdigest(),
                "provider": envelope["session"]["provider"],
                "provider_session_id_sha256": hashlib.sha256(native_id.encode()).hexdigest(),
                "turn_count": turns,
                "attachment_count": attachments,
            }
            if fault == "identity":
                receipt["provider_session_id_sha256"] = "0" * 64
            payload = {
                "ok": True,
                "providers": {"chatgpt.com": receipt},
                "automatic_capture_enabled": True,
                "extension": {"id": "a" * 32},
                "proof_binding": {"extension_id": "a" * 32, "host_name": constructor["host"]},
                "isolation": {
                    "declared_window_count": 1,
                    "admitted_tab_count": 1,
                    "static_content_scripts": fault == "isolation",
                    "document_bound_effects": True,
                    "current_window": "first_declared_owned_window",
                },
            }
            return json.dumps(payload), "neutral private stderr"

    monkeypatch.setattr("devtools.live_provider_proof_service.subprocess.run", build)
    monkeypatch.setattr("devtools.live_provider_proof_service.subprocess.Popen", Process)
    terminated: list[object] = []
    monkeypatch.setattr(service, "terminate_process_group", terminated.append)
    envelope, _, _ = native_proof_artifact(tmp_path, "native-rich-blocks-v1.json")
    native_id = envelope["session"]["provider_session_id"]
    targets = [{"name": "chatgpt", "url": "https://chatgpt.com/c/" + native_id, "nativeId": native_id}]
    if fault == "none":
        result = service._run_proof_locked(targets=targets, chrome_user_data_dir=profile)
        assert result["ok"] is True
        assert result["archive_convergence"] == "not_exercised"
        assert result["automatic_capture_enabled"] is True
        assert result["receiver_requests"] == [{"method": "GET", "path": "/v1/status", "status": 401}]
        providers = result["providers"]
        assert isinstance(providers, dict)
        assert providers["chatgpt.com"]["message_count"] > 0
    else:
        expected = (
            subprocess.CalledProcessError
            if fault == "constructor"
            else RuntimeError
            if fault == "start"
            else NativeProofCustodyError
            if fault == "custody"
            else service.ChildProofError
        )
        with pytest.raises(expected):
            service._run_proof_locked(targets=targets, chrome_user_data_dir=profile)
    assert os.environ == before
    assert operator_manifest.read_text() == "operator untouched"
    if fault == "custody":
        manifest = hosts / (constructor["host"] + ".json")
        assert manifest.read_text() == "changed neutral manifest"
        retained = list(tmp_path.glob("polylogue-owned-provider-proof-*"))
        assert len(retained) == 1
        assert (retained[0] / "native-transport-proof/native-host").is_file()
    else:
        assert list(hosts.iterdir()) == [operator_manifest]
        assert not list(tmp_path.glob("polylogue-owned-provider-proof-*"))
    assert terminated == processes


def test_owned_provider_empty_scope_refuses_before_receiver_or_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "require_declared_operation_context", lambda _operation: "neutral")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("empty scope attempted receiver or browser work")

    monkeypatch.setattr(service, "make_server", forbidden)
    monkeypatch.setattr("devtools.live_provider_proof_service.subprocess.Popen", forbidden)
    with pytest.raises(ValueError, match="explicitly owned"):
        service._run_proof_locked(targets=[], chrome_user_data_dir=tmp_path)
