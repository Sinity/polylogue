from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from polylogue.agent_integration import agentctl_adapter as adapter
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _request() -> dict[str, object]:
    return {
        "schema": 1,
        "request_id": str(uuid4()),
        "correlation_id": str(uuid4()),
        "operation": adapter.OPERATION,
        "owner": adapter.OWNER,
        "principal": "test",
        "arguments": {},
    }


@pytest.fixture
def disposable_archive(tmp_path: Path) -> Generator[Path]:
    root = tmp_path / "archive"
    archive = ArchiveStore(root)
    archive.close()
    yield root


@pytest.fixture
def adapter_command() -> tuple[str, ...]:
    installed = Path(sys.executable).with_name("polylogue-agentctl-adapter")
    interpreter = (
        installed.read_text(encoding="utf-8").splitlines()[0].removeprefix("#!")
        if installed.is_file() and installed.stat().st_size
        else None
    )
    if installed.is_file() and os.access(installed, os.X_OK) and (interpreter is None or Path(interpreter).exists()):
        return (str(installed),)
    return (sys.executable, "-m", "polylogue.agent_integration.agentctl_adapter")


def _invoke_adapter(command: tuple[str, ...], root: Path, request: dict[str, object]) -> dict[str, object]:
    environment = os.environ | {"POLYLOGUE_ARCHIVE_ROOT": str(root)}
    result = subprocess.run(
        command,
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )
    assert result.stderr == ""
    response = json.loads(result.stdout)
    assert isinstance(response, dict)
    return cast(dict[str, object], response)


def test_status_rejects_nonempty_arguments_before_archive_access(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    request["arguments"] = {"scope": "archive"}
    monkeypatch.setattr("polylogue.paths.archive_root", pytest.fail)

    with pytest.raises(adapter.AdapterError, match="does not accept arguments") as caught:
        adapter._read_status(request)

    assert caught.value.code == "INVALID_ARGUMENT"


def test_status_pins_payload_and_binding_to_one_archive_generation(disposable_archive: Path) -> None:
    request = _request()
    previous = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(disposable_archive)
    try:
        response = adapter._read_status(request)
    finally:
        if previous is None:
            os.environ.pop("POLYLOGUE_ARCHIVE_ROOT", None)
        else:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = previous

    identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(disposable_archive))
    assert response["ok"] is True
    assert response["request_id"] == request["request_id"]
    assert response["correlation_id"] == request["correlation_id"]
    assert response["source_bindings"] == [
        {
            "source_ref": adapter.SOURCE_REF,
            "generation": identity.active_generation,
            "root_digest": f"sha256:{identity.authority_identity_digest}",
        }
    ]
    assert response["payload"] == {
        "kind": "inline",
        "value": {
            "operation": adapter.OPERATION,
            "archive": {"total_sessions": 0, "total_messages": 0, "origins": {}},
        },
    }


def test_production_entrypoint_rejects_a_stale_archive_generation(
    adapter_command: tuple[str, ...], disposable_archive: Path
) -> None:
    request = _request()
    request["expected_source_binding"] = {
        "source_ref": adapter.SOURCE_REF,
        "generation": "stale-generation",
        "root_digest": "sha256:" + "0" * 64,
    }

    response = _invoke_adapter(adapter_command, disposable_archive, request)

    assert response["ok"] is False
    assert response["source_bindings"] == []
    assert response["error"] == {
        "schema": adapter.SCHEMA,
        "code": "AUTHORITY_MISMATCH",
        "message": "expected_source_binding does not match the active archive generation",
        "details": {"kind": "inline", "value": {}},
    }


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("owner", "wrong-owner", "AUTHORITY_MISMATCH"),
        ("operation", "polylogue.archive.write", "AUTHORITY_MISMATCH"),
        ("arguments", {"scope": "archive"}, "INVALID_ARGUMENT"),
        (
            "expected_source_binding",
            {
                "source_ref": "sinnix://polylogue/other",
                "generation": "fixture-generation",
                "root_digest": "sha256:" + "0" * 64,
            },
            "AUTHORITY_MISMATCH",
        ),
    ],
)
def test_production_entrypoint_rejects_owner_contract_boundaries(
    adapter_command: tuple[str, ...],
    disposable_archive: Path,
    field: str,
    value: object,
    code: str,
) -> None:
    request = _request()
    request[field] = value

    response = _invoke_adapter(adapter_command, disposable_archive, request)

    assert response["ok"] is False
    assert response["request_id"] == request["request_id"]
    assert response["correlation_id"] == request["correlation_id"]
    assert response["owner"] == adapter.OWNER
    assert response["source_bindings"] == []
    assert response["error"] == {
        "schema": adapter.SCHEMA,
        "code": code,
        "message": {
            "AUTHORITY_MISMATCH": (
                "request owner does not match this adapter"
                if field == "owner"
                else "request operation does not match this adapter"
                if field == "operation"
                else "expected_source_binding names a different source"
            ),
            "INVALID_ARGUMENT": "polylogue.archive.status does not accept arguments",
        }[code],
        "details": {"kind": "inline", "value": {}},
    }
