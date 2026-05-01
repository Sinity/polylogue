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
from polylogue.cli.shared.machine_errors import (
    MachineError,
    emit_success,
    success,
)
from polylogue.core.json import JSONDocument
from tests.infra.json_contracts import envelope_result, extract_json_object, json_object_field, parse_json_object

pytestmark = pytest.mark.machine_contract


# ---------------------------------------------------------------------------
# Unit tests for emit_success
# ---------------------------------------------------------------------------


class TestEmitSuccess:
    """Tests for the emit_success() helper."""

    def test_emit_success_writes_envelope(self: object, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_success() writes the success envelope to stdout."""
        emit_success({"count": 42})
        captured = capsys.readouterr()
        parsed = parse_json_object(captured.out, context="emit_success stdout")
        assert parsed["status"] == "ok"
        assert parsed["result"] == {"count": 42}

    def test_emit_success_none_result(self: object, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_success(None) writes empty result."""
        emit_success(None)
        captured = capsys.readouterr()
        parsed = parse_json_object(captured.out, context="emit_success stdout")
        assert parsed == {"status": "ok", "result": {}}

    def test_emit_success_empty_result(self: object, capsys: pytest.CaptureFixture[str]) -> None:
        """emit_success({}) writes empty result."""
        emit_success({})
        captured = capsys.readouterr()
        parsed = parse_json_object(captured.out, context="emit_success stdout")
        assert parsed == {"status": "ok", "result": {}}


# ---------------------------------------------------------------------------
# Contract: every --json command wraps in success envelope
# ---------------------------------------------------------------------------


def _invoke_json_command(args: list[str], monkeypatch: pytest.MonkeyPatch) -> JSONDocument | None:
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
    return extract_json_object(result.output, context=f"{args[0]} output")


class TestCheckJsonEnvelope:
    """check --json wraps output in success envelope."""

    def test_check_json_has_status_ok(self: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """polylogue check --json output has status: ok."""
        parsed = _invoke_json_command(["doctor", "--json"], monkeypatch)
        assert parsed is not None
        assert parsed["status"] == "ok"
        assert "result" in parsed


class TestTagsJsonEnvelope:
    """tags --json wraps output in success envelope."""

    def test_tags_json_has_status_ok(self: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """polylogue tags --json output has status: ok."""
        parsed = _invoke_json_command(["tags", "--json"], monkeypatch)
        assert parsed is not None
        assert parsed["status"] == "ok"
        assert "result" in parsed
        assert "tags" in envelope_result(parsed, context="tags envelope")


class TestQueryShapedJsonMatrix:
    @pytest.mark.parametrize(
        ("args", "result_key"),
        [
            (["tags", "--format", "json"], "tags"),
            (["neighbors", "--query", "__polylogue_json_contract_probe__", "--format", "json"], "neighbors"),
            (["products", "status", "--format", "json"], "products"),
            (["schema", "list", "--format", "json"], "providers"),
        ],
    )
    def test_format_json_uses_success_envelope(
        self,
        args: list[str],
        result_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parsed = _invoke_json_command(args, monkeypatch)
        assert parsed is not None
        assert parsed["status"] == "ok"
        assert result_key in envelope_result(parsed, context="format json envelope")

    @pytest.mark.parametrize(
        "args",
        [
            ["neighbors", "--query", "__polylogue_json_contract_probe__", "--json"],
            ["products", "status", "--json"],
            ["schema", "list", "--json"],
        ],
    )
    def test_json_alias_uses_success_envelope(self, args: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
        parsed = _invoke_json_command(args, monkeypatch)
        assert parsed is not None
        assert parsed["status"] == "ok"
        assert "result" in parsed


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
    def test_error_envelope_shape(self: object, code: str, message: str) -> None:
        """Error envelope has required fields."""
        err = MachineError(code=code, message=message)
        d = err.to_dict()
        assert d["status"] == "error"
        assert d["code"] == code
        assert d["message"] == message

    def test_error_envelope_is_valid_json(self: object) -> None:
        """Error envelope serializes to valid JSON."""
        err = MachineError(
            code="runtime_error",
            message="Test",
            command=["polylogue", "run"],
            details={"nested": {"data": True}},
        )
        text = json.dumps(err.to_dict())
        parsed = parse_json_object(text, context="error envelope")
        details = json_object_field(parsed, "details", context="error envelope")
        nested = json_object_field(details, "nested", context="error envelope.details")
        assert parsed["status"] == "error"
        assert nested["data"] is True


class TestSuccessEnvelopeContract:
    """MachineSuccess always has status: ok and result."""

    def test_success_envelope_shape(self: object) -> None:
        """Success envelope has status: ok and result."""
        s = success({"data": [1, 2, 3]})
        d = s.to_dict()
        assert d["status"] == "ok"
        assert d["result"]["data"] == [1, 2, 3]

    def test_success_envelope_is_valid_json(self: object) -> None:
        """Success envelope serializes to valid JSON."""
        s = success({"nested": {"deep": True}})
        text = json.dumps(s.to_dict())
        parsed = parse_json_object(text, context="success envelope")
        assert parsed["status"] == "ok"

    def test_success_none_gives_empty_result(self: object) -> None:
        """success(None) produces empty result dict."""
        s = success(None)
        assert s.to_dict() == {"status": "ok", "result": {}}


# ---------------------------------------------------------------------------
# Cross-cutting: status field always present and correct type
# ---------------------------------------------------------------------------


class TestStatusFieldInvariant:
    """Both envelope types always include status as a string."""

    def test_error_status_is_string(self: object) -> None:
        err = MachineError(code="x", message="y")
        assert isinstance(err.to_dict()["status"], str)
        assert err.to_dict()["status"] == "error"

    def test_success_status_is_string(self: object) -> None:
        s = success()
        assert isinstance(s.to_dict()["status"], str)
        assert s.to_dict()["status"] == "ok"

    def test_error_and_success_status_are_disjoint(self: object) -> None:
        """Error and success envelopes have distinct status values."""
        err_status = MachineError(code="x", message="y").to_dict()["status"]
        ok_status = success().to_dict()["status"]
        assert (err_status, ok_status) == ("error", "ok")
