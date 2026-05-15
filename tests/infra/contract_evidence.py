"""Pytest fixture for bounded contract evidence artifacts.

Contract tests remain ordinary pytest assertions.  This fixture only records a
small, privacy-bounded artifact after the assertion boundary so reports can
summarize what was exercised without re-running a parallel proof layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument, JSONValue, require_json_document

_DEFAULT_EVIDENCE_DIR = Path(".cache/verification/evidence")
_MAX_SAMPLE_CHARS = 2000
_MAX_JSON_DOCUMENT_BYTES = 8192
_SECRET_ASSIGNMENT_RE = re.compile(r"(?i)(token|secret|api[_-]?key|password)(=|:)\S+")
_SECRET_FLAG_RE = re.compile(r"(?i)(--?(?:token|secret|api[_-]?key|password))(?:=|\s+)\S+")
_UNSAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True, slots=True)
class ContractEvidenceRecorder:
    """Record bounded evidence for one pytest contract node."""

    nodeid: str
    repo_root: Path
    record_property: Callable[[str, object], None]

    def record(
        self,
        contract_id: str,
        *,
        surface: str,
        command: Sequence[str] | None = None,
        request: Mapping[str, JSONValue] | None = None,
        result: Mapping[str, JSONValue] | None = None,
        facts: Mapping[str, JSONValue] | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        exit_code: int | None = None,
    ) -> Path:
        """Write one evidence artifact and expose its path to pytest reports."""
        artifact_dir = _evidence_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / _artifact_filename(self.nodeid, contract_id)
        payload = self._payload(
            contract_id=contract_id,
            surface=surface,
            command=command,
            request=request,
            result=result,
            facts=facts,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        self.record_property("polylogue_contract_id", contract_id)
        self.record_property("polylogue_contract_surface", surface)
        self.record_property("polylogue_contract_evidence", str(artifact_path))
        return artifact_path

    def _payload(
        self,
        *,
        contract_id: str,
        surface: str,
        command: Sequence[str] | None,
        request: Mapping[str, JSONValue] | None,
        result: Mapping[str, JSONValue] | None,
        facts: Mapping[str, JSONValue] | None,
        stdout: str | None,
        stderr: str | None,
        exit_code: int | None,
    ) -> JSONDocument:
        payload: JSONDocument = {
            "schema_version": 1,
            "contract": contract_id,
            "surface": surface,
            "test_nodeid": self.nodeid,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "git_sha": _git_sha(self.repo_root),
            "dirty": _git_dirty(self.repo_root),
        }
        if command is not None:
            payload["command"] = _bounded_text(" ".join(str(part) for part in command), self.repo_root)
        if exit_code is not None:
            payload["exit_code"] = exit_code
        if stdout is not None:
            payload["stdout_sample"] = _bounded_text(stdout, self.repo_root)
        if stderr is not None:
            payload["stderr_sample"] = _bounded_text(stderr, self.repo_root)
        if request is not None:
            payload["request"] = _bounded_json_document(
                request,
                context="contract evidence request",
                repo_root=self.repo_root,
            )
        if result is not None:
            payload["result"] = _bounded_json_document(
                result,
                context="contract evidence result",
                repo_root=self.repo_root,
            )
        if facts is not None:
            payload["facts"] = _bounded_json_document(
                facts,
                context="contract evidence facts",
                repo_root=self.repo_root,
            )
        return payload


@pytest.fixture
def record_contract_evidence(
    request: pytest.FixtureRequest,
    record_property: Callable[[str, object], None],
) -> ContractEvidenceRecorder:
    """Return a recorder for explicit contract-test artifacts."""
    return ContractEvidenceRecorder(
        nodeid=request.node.nodeid,
        repo_root=Path(str(request.config.rootpath)),
        record_property=record_property,
    )


def _evidence_dir() -> Path:
    configured = os.environ.get("POLYLOGUE_CONTRACT_EVIDENCE_DIR")
    return Path(configured) if configured else _DEFAULT_EVIDENCE_DIR


def _artifact_filename(nodeid: str, contract_id: str) -> str:
    slug = _UNSAFE_FILENAME_RE.sub("-", contract_id).strip("-") or "contract"
    digest = hashlib.sha256(f"{nodeid}\0{contract_id}".encode()).hexdigest()[:16]
    return f"{slug}-{digest}.json"


def _json_document(value: Mapping[str, JSONValue], *, context: str) -> JSONDocument:
    return require_json_document(dict(value), context=context)


def _bounded_json_document(
    value: Mapping[str, JSONValue],
    *,
    context: str,
    repo_root: Path,
) -> JSONDocument:
    document = _redacted_json_document(_json_document(value, context=context), repo_root)
    encoded = json.dumps(document, sort_keys=True, ensure_ascii=False)
    encoded_bytes = encoded.encode("utf-8")
    if len(encoded_bytes) <= _MAX_JSON_DOCUMENT_BYTES:
        return document
    return {
        "truncated": True,
        "original_bytes": len(encoded_bytes),
        "sample_json": _bounded_text(encoded, repo_root),
    }


def _redacted_json_document(document: JSONDocument, repo_root: Path) -> JSONDocument:
    redacted = _redacted_json_value(document, repo_root)
    assert isinstance(redacted, dict)
    return redacted


def _redacted_json_value(value: JSONValue, repo_root: Path) -> JSONValue:
    if isinstance(value, str):
        return _redact_text(value, repo_root)
    if isinstance(value, list):
        return [_redacted_json_value(item, repo_root) for item in value]
    if isinstance(value, dict):
        return {key: _redacted_json_value(item, repo_root) for key, item in value.items()}
    return value


def _bounded_text(text: str, repo_root: Path) -> str:
    redacted = _redact_text(text, repo_root)
    if len(redacted) <= _MAX_SAMPLE_CHARS:
        return redacted
    return f"{redacted[:_MAX_SAMPLE_CHARS]}...<truncated {len(redacted) - _MAX_SAMPLE_CHARS} chars>"


def _redact_text(text: str, repo_root: Path) -> str:
    redacted = text
    replacements = {
        str(repo_root): "<repo>",
        str(Path.home()): "<home>",
    }
    for needle, replacement in replacements.items():
        if needle and needle in redacted:
            redacted = redacted.replace(needle, replacement)
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", redacted)
    return _SECRET_FLAG_RE.sub(lambda match: f"{match.group(1)} <redacted>", redacted)


def _git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _git_dirty(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


__all__ = ["ContractEvidenceRecorder", "record_contract_evidence"]
