"""Tests for bounded pytest contract evidence artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from tests.infra.contract_evidence import ContractEvidenceRecorder


def _load_json(path: Path) -> JSONDocument:
    return require_json_document(json.loads(path.read_text(encoding="utf-8")), context="contract evidence artifact")


def _json_str(payload: JSONDocument, key: str) -> str:
    value = payload[key]
    assert isinstance(value, str)
    return value


@pytest.mark.contract
def test_record_contract_evidence_writes_bounded_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    monkeypatch.setenv("POLYLOGUE_CONTRACT_EVIDENCE_DIR", str(tmp_path))

    path = record_contract_evidence.record(
        "cli.json_envelope",
        surface="cli",
        command=("polylogue", "tags", "--format", "json"),
        exit_code=0,
        stdout='{"status":"ok","result":{"tags":[]}}',
        stderr="",
        facts={"parsed_json_status": "ok"},
    )

    payload = _load_json(path)
    assert payload["contract"] == "cli.json_envelope"
    assert payload["surface"] == "cli"
    assert payload["command"] == ["polylogue", "tags", "--format", "json"]
    assert payload["exit_code"] == 0
    assert payload["stdout_sample"] == '{"status":"ok","result":{"tags":[]}}'
    assert payload["facts"] == {"parsed_json_status": "ok"}
    assert _json_str(payload, "test_nodeid").endswith("test_record_contract_evidence_writes_bounded_json")


@pytest.mark.contract
def test_record_contract_evidence_redacts_paths_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    monkeypatch.setenv("POLYLOGUE_CONTRACT_EVIDENCE_DIR", str(tmp_path))
    secret_output = f"path={Path.cwd()} token=super-secret api_key=abc123"

    path = record_contract_evidence.record(
        "cli.error_privacy",
        surface="cli",
        stdout=secret_output,
        stderr=f"secret:abc123 home={Path.home()}",
    )

    payload = _load_json(path)
    combined = f"{payload['stdout_sample']} {payload['stderr_sample']}"
    assert str(Path.cwd()) not in combined
    assert str(Path.home()) not in combined
    assert "super-secret" not in combined
    assert "abc123" not in combined
    assert "<repo>" in combined
    assert "<home>" in combined
    assert "token=<redacted>" in combined
    assert "secret:<redacted>" in combined
    assert "api_key=<redacted>" in combined


@pytest.mark.contract
def test_record_contract_evidence_rejects_non_json_fact_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_contract_evidence: ContractEvidenceRecorder,
) -> None:
    monkeypatch.setenv("POLYLOGUE_CONTRACT_EVIDENCE_DIR", str(tmp_path))
    bad_facts = cast(Mapping[str, JSONValue], {"path": tmp_path})

    with pytest.raises(TypeError, match="contract evidence facts"):
        record_contract_evidence.record(
            "bad.fact",
            surface="cli",
            facts=bad_facts,
        )


def test_contract_evidence_recorder_records_pytest_properties(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POLYLOGUE_CONTRACT_EVIDENCE_DIR", str(tmp_path))
    properties: dict[str, object] = {}

    def record_property(name: str, value: object) -> None:
        properties[name] = value

    recorder = ContractEvidenceRecorder(
        nodeid="tests/unit/example.py::test_example",
        repo_root=Path.cwd(),
        record_property=record_property,
    )
    path = recorder.record("api.query_parity", surface="api", facts={"ids": ["conv-1"]})

    assert path.exists()
    assert properties["polylogue_contract_id"] == "api.query_parity"
    assert properties["polylogue_contract_surface"] == "api"
    assert isinstance(properties["polylogue_contract_evidence"], str)
