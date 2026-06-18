"""Tests for the unified JSON success/error envelope contract.

Every ``--format json`` command must produce output conforming to either:
- Success: ``{"status": "ok", "result": {...}}``
- Error:   ``{"status": "error", "code": ..., "message": ..., ...}``

These tests verify the contract via the Click CliRunner so they run
in-process (no subprocess isolation needed — the envelope wrapping is
pure output formatting, independent of path caching).
"""

from __future__ import annotations

import json
from pathlib import Path

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
# Contract: every --format json command wraps in success envelope
# ---------------------------------------------------------------------------


def _invoke_json_command(args: list[str], monkeypatch: pytest.MonkeyPatch) -> JSONDocument:
    """Invoke a CLI command with --format json flag, return parsed output.

    Requires workspace_env to already be active (env vars set via monkeypatch)
    so the command runs against a fresh, deterministic empty archive.
    Raises AssertionError (pytest.fail) on non-zero exit instead of skipping.
    """
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()
    result = runner.invoke(cli, args, catch_exceptions=True)
    if result.exit_code != 0:
        exc_info = f" ({type(result.exception).__name__}: {result.exception})" if result.exception else ""
        pytest.fail(f"{' '.join(args)} exited {result.exit_code}{exc_info}\nOutput: {result.output!r}")
    doc = extract_json_object(result.output, context=f"{args[0]} output")
    assert doc is not None, f"No JSON object found in {args[0]} output: {result.output!r}"
    return doc


class TestCheckJsonEnvelope:
    """check --format json wraps output in success envelope."""

    def test_check_json_has_status_ok(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],  # deterministic empty archive, no skip
    ) -> None:
        """polylogue ops doctor --format json output has status: ok."""
        parsed = _invoke_json_command(["ops", "doctor", "--format", "json"], monkeypatch)
        assert parsed["status"] == "ok"
        assert "result" in parsed


class TestTagsJsonEnvelope:
    """tags --format json wraps output in success envelope."""

    def test_tags_json_has_status_ok(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],  # deterministic empty archive, no skip
    ) -> None:
        """polylogue tags --format json output has status: ok."""
        parsed = _invoke_json_command(["tags", "--format", "json"], monkeypatch)
        assert parsed["status"] == "ok"
        assert "result" in parsed
        assert "tags" in envelope_result(parsed, context="tags envelope")


class TestQueryShapedJsonMatrix:
    @pytest.mark.parametrize(
        ("args", "result_key"),
        [
            (["tags", "--format", "json"], "tags"),
            (["insights", "status", "--format", "json"], "insights"),
            (["ops", "schema", "list", "--format", "json"], "providers"),
        ],
    )
    @pytest.mark.contract
    def test_format_json_uses_success_envelope(
        self,
        args: list[str],
        result_key: str,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],  # deterministic empty archive, no skip
    ) -> None:
        parsed = _invoke_json_command(args, monkeypatch)
        assert parsed["status"] == "ok"
        assert result_key in envelope_result(parsed, context="format json envelope")

    @pytest.mark.parametrize(
        "args",
        [
            ["insights", "status", "--format", "json"],
            ["ops", "schema", "list", "--format", "json"],
        ],
    )
    def test_json_alias_uses_success_envelope(
        self,
        args: list[str],
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],  # deterministic empty archive, no skip
    ) -> None:
        parsed = _invoke_json_command(args, monkeypatch)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRACEBACK_SENTINEL = "Traceback (most recent call last)"


def _invoke_raw_json_command(args: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    """Invoke a CLI command and return (exit_code, raw_stdout).

    Unlike _invoke_json_command, this does NOT assert success — callers
    decide what to assert.  Used for both success and error path tests.

    Re-raises non-SystemExit exceptions so that command crashes are not
    silently swallowed by catch_exceptions=True and turned into false-passing
    negative-path tests.
    """
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()
    result = runner.invoke(cli, args, catch_exceptions=True)
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.output


def _init_empty_archive(workspace_env: dict[str, Path]) -> None:
    """Bootstrap an empty archive `index.db` under the workspace archive root.

    The query verbs read the archive ``index.db`` directly; without an
    initialized archive they exit 1 with "index database not found".
    Tests that exercise the empty-but-existing-archive contract initialize the
    archive store first so browse mode can report zero matches.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(workspace_env["archive_root"]):
        pass


# ---------------------------------------------------------------------------
# Query verb: list --format json
# ---------------------------------------------------------------------------


class TestReadAllJsonContract:
    """read --all --format json emits valid JSON."""

    @pytest.mark.contract
    def test_read_all_json_empty_archive_returns_empty_envelope(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """polylogue read --all --format json returns an empty success envelope on empty archive.

        Archive browse mode ("show me everything, there is nothing") is a valid
        success: read --all exits 0 with a parseable archive envelope reporting
        zero rows. Only search mode (a query that matched nothing) exits 2.
        """
        _init_empty_archive(workspace_env)
        exit_code, output = _invoke_raw_json_command(["read", "--all", "--format", "json"], monkeypatch)
        assert exit_code == 0, (
            f"read --all --format json on empty archive: expected exit 0, got {exit_code}: {output!r}"
        )
        assert TRACEBACK_SENTINEL not in output
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert parsed.get("mode") == "list"
        assert parsed.get("total") == 0

    def test_read_all_json_invalid_format_choice_no_traceback(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """read --all --format with unknown value exits non-zero without traceback."""
        exit_code, output = _invoke_raw_json_command(["read", "--all", "--format", "xml"], monkeypatch)
        assert exit_code != 0
        assert TRACEBACK_SENTINEL not in output


# ---------------------------------------------------------------------------
# Query verb: stats --format json
# ---------------------------------------------------------------------------


class TestAnalyzeJsonContract:
    """analyze --format json emits valid JSON (error envelope on empty archive).

    When the archive has no sessions, stats exits code 2 with a machine-error
    envelope. When sessions exist, it emits a raw JSON object with dimension/rows/summary.
    Both shapes are valid JSON — the contract tests verify the envelope in both cases.
    """

    @pytest.mark.contract
    def test_stats_json_empty_archive_returns_empty_envelope(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """polylogue analyze --format json returns an empty success envelope on empty archive."""
        _init_empty_archive(workspace_env)
        exit_code, output = _invoke_raw_json_command(["analyze", "--format", "json"], monkeypatch)
        assert exit_code == 0, f"analyze --format json on empty archive: expected exit 0, got {exit_code}: {output!r}"
        assert TRACEBACK_SENTINEL not in output
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert parsed.get("total_sessions", 0) == 0

    @pytest.mark.contract
    def test_stats_by_origin_json_empty_archive_returns_empty_envelope(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """polylogue analyze --by origin --format json returns empty envelope on empty archive."""
        _init_empty_archive(workspace_env)
        exit_code, output = _invoke_raw_json_command(["analyze", "--by", "origin", "--format", "json"], monkeypatch)
        assert exit_code == 0, (
            f"analyze --by origin --format json on empty archive: expected exit 0, got {exit_code}: {output!r}"
        )
        assert TRACEBACK_SENTINEL not in output
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert parsed.get("mode") == "stats_by"
        assert parsed.get("items") == []

    def test_stats_json_invalid_by_value_no_traceback(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """analyze --by with unknown value exits non-zero without traceback."""
        exit_code, output = _invoke_raw_json_command(
            ["analyze", "--by", "nonexistent_dimension", "--format", "json"], monkeypatch
        )
        assert exit_code != 0
        assert TRACEBACK_SENTINEL not in output


# ---------------------------------------------------------------------------
# ops status --format json
# ---------------------------------------------------------------------------


class TestStatusJsonContract:
    """ops status --format json emits raw JSON (no envelope — direct archive query)."""

    @pytest.mark.contract
    def test_status_json_daemon_not_running(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """polylogue ops status --format json returns JSON when daemon is not running.

        Uses an unreachable daemon URL so the command falls back to the
        direct archive read path, which always returns structured JSON.
        """
        exit_code, output = _invoke_raw_json_command(
            ["ops", "status", "--daemon-url", "http://127.0.0.1:19999", "--format", "json"],
            monkeypatch,
        )
        assert exit_code == 0, f"ops status --format json exited {exit_code}: {output!r}"
        assert TRACEBACK_SENTINEL not in output
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert "daemon_liveness" in parsed, f"Missing 'daemon_liveness' key: {parsed.keys()}"
        assert parsed["daemon_liveness"] is False

    def test_status_json_no_traceback_on_bad_url(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """ops status with malformed daemon URL exits cleanly without traceback."""
        exit_code, output = _invoke_raw_json_command(
            ["ops", "status", "--daemon-url", "not-a-url", "--format", "json"],
            monkeypatch,
        )
        # May exit 0 (fallback) or non-zero; must not produce traceback
        assert TRACEBACK_SENTINEL not in output


# ---------------------------------------------------------------------------
# config --format json
# ---------------------------------------------------------------------------


class TestConfigJsonContract:
    """config --format json emits raw JSON config object (no envelope)."""

    @pytest.mark.contract
    def test_config_json_is_valid_json_object(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """polylogue config --format json outputs a valid JSON object."""
        exit_code, output = _invoke_raw_json_command(["config", "--format", "json"], monkeypatch)
        assert exit_code == 0, f"config --format json exited {exit_code}: {output!r}"
        assert TRACEBACK_SENTINEL not in output
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_config_json_invalid_format_no_traceback(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """config --format with unsupported value exits non-zero without traceback."""
        exit_code, output = _invoke_raw_json_command(["config", "--format", "xml"], monkeypatch)
        assert exit_code != 0
        assert TRACEBACK_SENTINEL not in output


# ---------------------------------------------------------------------------
# schema explain --format json
# ---------------------------------------------------------------------------


class TestSchemaExplainJsonContract:
    """schema explain --provider <p> --format json uses the emit_success envelope."""

    @pytest.mark.contract
    def test_schema_explain_json_claude_ai(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """schema explain --provider claude-ai --format json uses success envelope."""
        parsed = _invoke_json_command(
            ["ops", "schema", "explain", "--provider", "claude-ai", "--format", "json"],
            monkeypatch,
        )
        assert parsed["status"] == "ok"
        result = envelope_result(parsed, context="schema explain envelope")
        assert "provider" in result or "schema" in result or len(result) > 0

    def test_schema_explain_json_unknown_provider_no_traceback(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """schema explain with unknown provider exits non-zero without traceback."""
        exit_code, output = _invoke_raw_json_command(
            ["ops", "schema", "explain", "--provider", "nonexistent_provider_xyz", "--format", "json"],
            monkeypatch,
        )
        assert exit_code != 0
        assert TRACEBACK_SENTINEL not in output

    def test_schema_explain_json_missing_provider_no_traceback(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """schema explain without --provider exits non-zero without traceback."""
        exit_code, output = _invoke_raw_json_command(
            ["ops", "schema", "explain", "--format", "json"],
            monkeypatch,
        )
        assert exit_code != 0
        assert TRACEBACK_SENTINEL not in output


# ---------------------------------------------------------------------------
# Cross-command: all --format json commands produce valid JSON
# ---------------------------------------------------------------------------


class TestAllJsonCommandsProduceValidJson:
    """Matrix: every command with --format json must produce parseable JSON.

    Commands that query the archive return exit 0 on success or exit 2 with a
    JSON error envelope when no results match (no_results). Both paths produce
    valid JSON — the contract is that output is always machine-parseable.
    """

    @pytest.mark.parametrize(
        ("args", "contract_id", "allow_no_results"),
        [
            (["ops", "doctor", "--format", "json"], "cli.doctor_json_matrix", False),
            (["tags", "--format", "json"], "cli.tags_json_matrix", False),
            (["ops", "schema", "list", "--format", "json"], "cli.schema_list_json_matrix", False),
            (["config", "--format", "json"], "cli.config_json_matrix", False),
            # read --all browse on empty archive → exit 0 + empty archive envelope (valid JSON)
            (["read", "--all", "--format", "json"], "cli.read_all_json_matrix", False),
            (
                ["ops", "status", "--daemon-url", "http://127.0.0.1:19999", "--format", "json"],
                "cli.status_json_matrix",
                False,
            ),
        ],
    )
    @pytest.mark.contract
    def test_command_json_output_is_parseable(
        self: object,
        args: list[str],
        contract_id: str,
        allow_no_results: bool,
        monkeypatch: pytest.MonkeyPatch,
        workspace_env: dict[str, Path],
    ) -> None:
        """Every --format json command produces parseable JSON on any exit code."""
        _init_empty_archive(workspace_env)
        exit_code, output = _invoke_raw_json_command(args, monkeypatch)
        allowed_codes = {0, 2} if allow_no_results else {0}
        assert exit_code in allowed_codes, (
            f"{args[0]} --format json exited {exit_code} (allowed: {allowed_codes}): {output!r}"
        )
        assert TRACEBACK_SENTINEL not in output
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{' '.join(args)} produced invalid JSON: {exc}\nOutput: {output!r}")
        assert parsed is not None
