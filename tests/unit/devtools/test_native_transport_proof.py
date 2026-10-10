"""Exact scoped manifest custody without touching the operator's fixed host."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from devtools import native_transport_proof


@pytest.mark.parametrize("fail", [False, True])
def test_scoped_manifest_binds_neutral_paths_and_removes_only_its_owned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    profile = tmp_path / "actual-custom-chrome"
    manifests = profile / "NativeMessagingHosts"
    manifests.mkdir(parents=True)
    fixed = manifests / "com.polylogue.browser_capture.json"
    fixed.write_bytes(b"operator-owned")
    wrapper = tmp_path / "native-wrapper"
    wrapper.write_text("#!/bin/sh\n")
    archive = tmp_path / "neutral-archive"
    archive.mkdir()
    (archive / "browser-capture-receiver-id").write_text("neutral-id")
    secret = archive / "browser-capture-receiver-token"
    secret.write_text("neutral-secret-not-public")
    secret.chmod(0o600)
    monkeypatch.setattr(shutil, "which", lambda _name: str(wrapper))

    def build(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(command[command.index("--destination") + 1])
        destination.mkdir()
        host = command[command.index("--host") + 1]
        return subprocess.CompletedProcess(command, 0, json.dumps({"host_name": host, "extension_id": "a" * 32}), "")

    monkeypatch.setattr(subprocess, "run", build)

    def attachment(_endpoint: str, token: str) -> tuple[str, str]:
        assert token == secret.read_text()
        if fail:
            raise RuntimeError("neutral acquisition failure")
        return "http://127.0.0.1:12345/v1/browser-actions/neutral/attachments/a", "neutral-sha"

    monkeypatch.setattr(native_transport_proof, "_attachment_fixture", attachment)
    environment = {
        "POLYLOGUE_ARCHIVE_ROOT": str(archive),
        "HOME": str(tmp_path / "neutral-home"),
        "TMPDIR": str(tmp_path),
        "UNRELATED_SECRET": "must-not-enter-launcher",
    }
    try:
        with native_transport_proof.scoped_native_transport_proof(
            repo_root=tmp_path,
            scratch=tmp_path,
            environment=environment,
            endpoint="http://127.0.0.1:12345",
            chrome_user_data_dir=profile,
        ) as binding:
            host = binding["POLYLOGUE_DEV_LOOP_NATIVE_HOST"]
            manifest = manifests / f"{host}.json"
            record = json.loads(manifest.read_bytes())
            assert record["allowed_origins"] == [f"chrome-extension://{'a' * 32}/"]
            launcher = Path(record["path"]).read_text()
            assert str(archive) in launcher
            assert str(wrapper) in launcher
            assert "neutral-secret-not-public" not in launcher
            assert "must-not-enter-launcher" not in launcher
            assert fixed.read_bytes() == b"operator-owned"
    except RuntimeError as exc:
        assert fail and str(exc) == "neutral acquisition failure"
    else:
        assert not fail
    assert list(manifests.iterdir()) == [fixed]
    assert fixed.read_bytes() == b"operator-owned"
    assert not (tmp_path / "native-transport-proof").exists()
