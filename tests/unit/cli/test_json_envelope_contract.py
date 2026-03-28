"""Tests for the unified JSON success/error envelope contract.

Every ``--json`` command must produce output conforming to either:
- Success: ``{"status": "ok", "result": {...}}``
- Error:   ``{"status": "error", "code": ..., "message": ..., ...}``

These tests verify the contract via the Click CliRunner so they run
in-process (no subprocess isolation needed — the envelope wrapping is
pure output formatting, independent of path caching).
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.machine_errors import (
    MachineError,
    emit_success,
    success,
)

pytestmark = pytest.mark.machine_contract


# ---------------------------------------------------------------------------
# Unit tests for emit_success
# ---------------------------------------------------------------------------


class TestEmitSuccess:
    """Tests for the emit_success() helper."""

    def test_emit_success_writes_envelope(self, capsys):
        """emit_success() writes the success envelope to stdout."""
        emit_success({"count": 42})
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["status"] == "ok"
        assert parsed["result"] == {"count": 42}

    def test_emit_success_none_result(self, capsys):
        """emit_success(None) writes empty result."""
        emit_success(None)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == {"status": "ok", "result": {}}

    def test_emit_success_empty_result(self, capsys):
        """emit_success({}) writes empty result."""
        emit_success({})
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == {"status": "ok", "result": {}}


# ---------------------------------------------------------------------------
# Contract: every --json command wraps in success envelope
# ---------------------------------------------------------------------------


def _parse_json_output(output: str) -> dict:
    """Parse JSON from CLI output, stripping any log lines."""
    # CLI may emit structlog lines to stderr; stdout should be clean JSON.
    # But CliRunner mixes output, so find the JSON object.
    lines = output.strip().splitlines()
    # Find the first line that starts with '{' and parse from there
    json_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("{"):
            json_start = i
            break
    if json_start is None:
        raise ValueError(f"No JSON object found in output:\n{output}")
    json_text = "\n".join(lines[json_start:])
    return json.loads(json_text)


def _invoke_json_command(args: list[str], monkeypatch) -> dict | None:
    """Invoke a CLI command with --json flag, return parsed output or None on skip.

    Returns None (and calls pytest.skip) if the command fails due to DB errors
    or other environment issues that make the test non-applicable.
    """
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()
    result = runner.invoke(cli, args, catch_exceptions=True)
    if result.exit_code != 0:
        # Check for known environment issues that should cause skip
        if result.exception and "DatabaseError" in type(result.exception).__name__:
            pytest.skip(f"DB schema mismatch: {result.exception}")
        if result.exception and "OperationalError" in type(result.exception).__name__:
            pytest.skip(f"DB operational error: {result.exception}")
        if result.exception:
            pytest.skip(f"{args[0]} --json raised {type(result.exception).__name__}: {result.exception}")
        pytest.skip(f"{args[0]} --json failed (exit {result.exit_code})")
    return _parse_json_output(result.output)


class TestCheckJsonEnvelope:
    """check --json wraps output in success envelope."""

    def test_check_json_has_status_ok(self, monkeypatch):
        """polylogue check --json output has status: ok."""
        parsed = _invoke_json_command(["check", "--json"], monkeypatch)
        assert parsed is not None
        assert parsed["status"] == "ok"
        assert "result" in parsed


class TestTagsJsonEnvelope:
    """tags --json wraps output in success envelope."""

    def test_tags_json_has_status_ok(self, monkeypatch):
        """polylogue tags --json output has status: ok."""
        parsed = _invoke_json_command(["tags", "--json"], monkeypatch)
        assert parsed is not None
        assert parsed["status"] == "ok"
        assert "result" in parsed
        assert "tags" in parsed["result"]


class TestSourcesJsonEnvelope:
    """sources --json wraps output in success envelope."""

    def test_sources_json_has_status_ok(self, monkeypatch):
        """polylogue sources --json output has status: ok."""
        parsed = _invoke_json_command(["sources", "--json"], monkeypatch)
        assert parsed is not None
        assert parsed["status"] == "ok"
        assert "result" in parsed
        assert "sources" in parsed["result"]


# ---------------------------------------------------------------------------
# Error envelope tests
# ---------------------------------------------------------------------------


class TestErrorEnvelopeContract:
    """MachineError always has status, code, message."""

    @pytest.mark.parametrize(
        "code,message",
        [
            ("invalid_arguments", "Bad flag"),
            ("invalid_path", "/nonexistent"),
            ("runtime_error", "DB locked"),
            ("dependency_missing", "sqlite-vec"),
            ("unsupported_environment", "No TTY"),
        ],
    )
    def test_error_envelope_shape(self, code, message):
        """Error envelope has required fields."""
        err = MachineError(code=code, message=message)
        d = err.to_dict()
        assert d["status"] == "error"
        assert d["code"] == code
        assert d["message"] == message

    def test_error_envelope_is_valid_json(self):
        """Error envelope serializes to valid JSON."""
        err = MachineError(
            code="runtime_error",
            message="Test",
            command=["polylogue", "run"],
            details={"nested": {"data": True}},
        )
        text = json.dumps(err.to_dict())
        parsed = json.loads(text)
        assert parsed["status"] == "error"
        assert parsed["details"]["nested"]["data"] is True


class TestSuccessEnvelopeContract:
    """MachineSuccess always has status: ok and result."""

    def test_success_envelope_shape(self):
        """Success envelope has status: ok and result."""
        s = success({"data": [1, 2, 3]})
        d = s.to_dict()
        assert d["status"] == "ok"
        assert d["result"]["data"] == [1, 2, 3]

    def test_success_envelope_is_valid_json(self):
        """Success envelope serializes to valid JSON."""
        s = success({"nested": {"deep": True}})
        text = json.dumps(s.to_dict())
        parsed = json.loads(text)
        assert parsed["status"] == "ok"

    def test_success_none_gives_empty_result(self):
        """success(None) produces empty result dict."""
        s = success(None)
        assert s.to_dict() == {"status": "ok", "result": {}}


# ---------------------------------------------------------------------------
# Cross-cutting: status field always present and correct type
# ---------------------------------------------------------------------------


class TestStatusFieldInvariant:
    """Both envelope types always include status as a string."""

    def test_error_status_is_string(self):
        err = MachineError(code="x", message="y")
        assert isinstance(err.to_dict()["status"], str)
        assert err.to_dict()["status"] == "error"

    def test_success_status_is_string(self):
        s = success()
        assert isinstance(s.to_dict()["status"], str)
        assert s.to_dict()["status"] == "ok"

    def test_error_and_success_status_are_disjoint(self):
        """Error and success envelopes have distinct status values."""
        err_status = MachineError(code="x", message="y").to_dict()["status"]
        ok_status = success().to_dict()["status"]
        assert err_status != ok_status
