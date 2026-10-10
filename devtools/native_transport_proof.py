"""Scoped Chrome native manifest for the declared neutral transport proof."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlsplit

from polylogue.browser_capture.receiver import read_receiver_credential


def _attachment_fixture(endpoint: str, token: str) -> tuple[str, str]:
    """Associate immutable neutral bytes through the ordinary receiver API."""
    data = bytes((index * 31 + 7) % 256 for index in range(1024 * 1024 + 17))
    digest = hashlib.sha256(data).hexdigest()
    target = urlsplit(endpoint)
    with closing(HTTPConnection(target.hostname or "", target.port)) as connection:
        headers = {"Authorization": f"Bearer {token}"}
        connection.request("PUT", "/v1/browser-action-attachments", body=data, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read())
        if response.status != 201 or payload.get("attachment_ref") != digest:
            raise RuntimeError("neutral attachment upload failed")
        request = {
            "provider": "chatgpt",
            "operation": "conversation.create",
            "text": "neutral transport proof",
            "attachments": [{"name": "neutral.bin", "attachment_ref": digest}],
            "presentation": {"model_slug": "gpt-5-6-pro", "model_label": "GPT-5.6 Sol", "effort_label": "Pro"},
            "submit_policy": "stage_only",
        }
        connection.request(
            "POST",
            "/v1/browser-actions",
            body=json.dumps(request),
            headers={**headers, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        if response.status != 202:
            raise RuntimeError("neutral attachment association failed")
    action = payload["action"]
    return (
        f"{endpoint}/v1/browser-actions/{action['action_id']}/attachments/{action['attachments'][0]['attachment_id']}",
        digest,
    )


class NativeProofCustodyError(RuntimeError):
    """The scoped manifest changed; its artifact must remain available."""


@contextmanager
def scoped_native_host(
    *,
    repo_root: Path,
    scratch: Path,
    environment: Mapping[str, str],
    chrome_user_data_dir: Path,
    build_extension: Callable[[Path, str], dict[str, object]],
) -> Iterator[dict[str, str]]:
    """Own one independent native manifest and its isolated extension artifact.

    The caller constructs either the page-only artifact or the owned-window
    runtime artifact. Its child must settle all native ports and unload the
    extension before returning. A changed manifest remains in custody.
    """
    profile = chrome_user_data_dir.absolute()
    if not profile.is_dir():
        raise RuntimeError("owned proof requires the actual Chrome user-data directory")
    proof_root = scratch / "native-transport-proof"
    proof_root.mkdir()
    manifest: Path | None = None
    raw: bytes | None = None
    owned: os.stat_result | None = None
    try:
        host_name = f"com.polylogue.browser_capture.proof_{uuid.uuid4().hex}"
        extension = proof_root / "extension"
        binding = build_extension(extension, host_name)
        extension_id = binding.get("extension_id")
        if binding.get("host_name") != host_name or not isinstance(extension_id, str):
            raise RuntimeError("neutral extension binding invalid")
        wrapper = shutil.which("polylogue-browser-capture-native-host")
        if wrapper is None:
            raise RuntimeError("candidate native host wrapper missing")
        wrapper_path = Path(wrapper).absolute()
        overrides = {
            key: environment[key]
            for key in (
                "HOME",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "XDG_STATE_HOME",
                "XDG_CACHE_HOME",
                "XDG_RUNTIME_DIR",
                "TMPDIR",
                "POLYLOGUE_ARCHIVE_ROOT",
                "POLYLOGUE_CONFIG",
                "POLYLOGUE_SITE_CONFIG",
                "POLYLOGUE_EMBEDDINGS_ENABLED",
            )
            if key in environment
        }
        launcher = proof_root / "native-host"
        launcher.write_text(
            f"#!{sys.executable}\nimport os, sys\n"
            "for key in tuple(os.environ):\n"
            "    if key.startswith('POLYLOGUE_') or key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PYTHONUSERBASE', 'HERMES_HOME'):\n"
            "        os.environ.pop(key, None)\n"
            f"os.environ.update({overrides!r})\n"
            f"os.execv({str(wrapper_path)!r}, [{str(wrapper_path)!r}, *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        launcher.chmod(0o700)
        manifest = profile / "NativeMessagingHosts" / f"{host_name}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "name": host_name,
            "description": "Owned neutral Polylogue transport proof",
            "path": str(launcher),
            "type": "stdio",
            "allowed_origins": [f"chrome-extension://{extension_id}/"],
        }
        raw = json.dumps(record).encode()
        with manifest.open("xb") as output:
            output.write(raw)
        owned = manifest.stat(follow_symlinks=False)
        yield {"extension_root": str(extension), "host_name": host_name, "extension_id": extension_id}
    finally:
        if owned is not None:
            assert manifest is not None and raw is not None
            unchanged = False
            try:
                current = manifest.stat(follow_symlinks=False)
                unchanged = (current.st_dev, current.st_ino) == (
                    owned.st_dev,
                    owned.st_ino,
                ) and manifest.read_bytes() == raw
            except OSError as error:
                raise NativeProofCustodyError("owned proof manifest unreadable; retaining proof custody") from error
            if not unchanged:
                raise NativeProofCustodyError("owned proof manifest changed; retaining proof custody")
            manifest.unlink()
        shutil.rmtree(proof_root)


@contextmanager
def scoped_native_transport_proof(
    *,
    repo_root: Path,
    scratch: Path,
    environment: Mapping[str, str],
    endpoint: str,
    chrome_user_data_dir: Path,
) -> Iterator[dict[str, str]]:
    """Build the page-only native transport fixture under the shared host owner."""

    def build(extension: Path, host_name: str) -> dict[str, object]:
        built = subprocess.run(
            [
                "node",
                str(repo_root / "browser-extension/scripts/proof_extension.mjs"),
                "--destination",
                str(extension),
                "--host",
                host_name,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        binding = json.loads(built.stdout)
        if not isinstance(binding, dict):
            raise ValueError("proof extension constructor binding invalid")
        return binding

    with scoped_native_host(
        repo_root=repo_root,
        scratch=scratch,
        environment=environment,
        chrome_user_data_dir=chrome_user_data_dir,
        build_extension=build,
    ) as host:
        archive = Path(environment["POLYLOGUE_ARCHIVE_ROOT"])
        identity = read_receiver_credential(archive / "browser-capture-receiver-id", secret=False)
        token = read_receiver_credential(archive / "browser-capture-receiver-token", secret=True)
        attachment_url, digest = _attachment_fixture(endpoint, token)
        yield {
            "POLYLOGUE_DEV_LOOP_EXTENSION_ROOT": host["extension_root"],
            "POLYLOGUE_DEV_LOOP_NATIVE_HOST": host["host_name"],
            "POLYLOGUE_DEV_LOOP_RECEIVER_URL": endpoint,
            "POLYLOGUE_DEV_LOOP_RECEIVER_ID": identity,
            "POLYLOGUE_DEV_LOOP_ATTACHMENT_URL": attachment_url,
            "POLYLOGUE_DEV_LOOP_ATTACHMENT_SHA256": digest,
        }
