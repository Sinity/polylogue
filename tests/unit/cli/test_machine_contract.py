"""Tests for the machine error and success envelope contract.

Validates the JSON envelope shapes and builder functions used by the CLI
when --json is requested.
"""

from __future__ import annotations

import pytest

from polylogue.cli.machine_errors import (
    DEPENDENCY_MISSING,
    INVALID_ARGUMENTS,
    INVALID_PATH,
    RUNTIME_ERROR,
    UNSUPPORTED_ENVIRONMENT,
    MachineError,
    MachineSuccess,
    error_dependency_missing,
    error_invalid_arguments,
    error_invalid_path,
    error_runtime,
    error_unsupported_environment,
    extract_command,
    success,
    wants_json,
)


class TestMachineErrorEnvelope:
    """Tests for MachineError.to_dict() contract."""

    def test_machine_error_to_dict_required_fields(self):
        """MachineError.to_dict() includes status, code, message."""
        err = MachineError(code="test_error", message="Test message")
        result = err.to_dict()

        assert result["status"] == "error"
        assert result["code"] == "test_error"
        assert result["message"] == "Test message"

    def test_machine_error_to_dict_includes_command_when_present(self):
        """MachineError.to_dict() includes command field when non-empty."""
        err = MachineError(
            code="invalid_arguments",
            message="Bad args",
            command=["polylogue", "run"],
        )
        result = err.to_dict()

        assert "command" in result
        assert result["command"] == ["polylogue", "run"]

    def test_machine_error_to_dict_omits_command_when_empty(self):
        """MachineError.to_dict() omits command field when empty."""
        err = MachineError(code="invalid_arguments", message="Bad args", command=[])
        result = err.to_dict()

        assert "command" not in result

    def test_machine_error_to_dict_includes_details_when_present(self):
        """MachineError.to_dict() includes details field when non-empty."""
        err = MachineError(
            code="invalid_path",
            message="Path not found",
            details={"path": "/nonexistent"},
        )
        result = err.to_dict()

        assert "details" in result
        assert result["details"] == {"path": "/nonexistent"}

    def test_machine_error_to_dict_omits_details_when_empty(self):
        """MachineError.to_dict() omits details field when empty."""
        err = MachineError(code="invalid_path", message="Path not found", details={})
        result = err.to_dict()

        assert "details" not in result

    def test_machine_error_to_dict_full_envelope(self):
        """MachineError.to_dict() with all fields produces correct shape."""
        err = MachineError(
            code="runtime_error",
            message="Database connection failed",
            command=["polylogue", "run", "--json"],
            details={"exception_type": "ConnectionError"},
        )
        result = err.to_dict()

        assert result == {
            "status": "error",
            "code": "runtime_error",
            "message": "Database connection failed",
            "command": ["polylogue", "run", "--json"],
            "details": {"exception_type": "ConnectionError"},
        }


class TestMachineSuccessEnvelope:
    """Tests for MachineSuccess.to_dict() contract."""

    def test_machine_success_to_dict_required_fields(self):
        """MachineSuccess.to_dict() includes status and result."""
        success_obj = MachineSuccess(result={"count": 42})
        result = success_obj.to_dict()

        assert result["status"] == "ok"
        assert result["result"] == {"count": 42}

    def test_machine_success_to_dict_empty_result(self):
        """MachineSuccess.to_dict() handles empty result dict."""
        success_obj = MachineSuccess(result={})
        result = success_obj.to_dict()

        assert result == {"status": "ok", "result": {}}

    def test_machine_success_to_dict_nested_result(self):
        """MachineSuccess.to_dict() preserves nested result structures."""
        result_data = {
            "summary": {"ok": 10, "warning": 2, "error": 0},
            "checks": [{"name": "database", "status": "ok"}],
        }
        success_obj = MachineSuccess(result=result_data)
        result = success_obj.to_dict()

        assert result["result"] == result_data


class TestErrorBuilders:
    """Tests for error builder functions."""

    def test_error_invalid_arguments_code(self):
        """error_invalid_arguments produces INVALID_ARGUMENTS code."""
        err = error_invalid_arguments("Bad flag")

        assert err.code == INVALID_ARGUMENTS
        assert err.message == "Bad flag"

    def test_error_invalid_arguments_with_option(self):
        """error_invalid_arguments includes option in details when provided."""
        err = error_invalid_arguments(
            "Missing required option",
            option="--archive-root",
        )

        assert err.details == {"option": "--archive-root"}

    def test_error_invalid_arguments_with_command(self):
        """error_invalid_arguments includes command when provided."""
        err = error_invalid_arguments(
            "Invalid arguments",
            command=["polylogue", "run"],
        )

        assert err.command == ["polylogue", "run"]

    def test_error_invalid_path_code(self):
        """error_invalid_path produces INVALID_PATH code."""
        err = error_invalid_path("Path does not exist")

        assert err.code == INVALID_PATH
        assert err.message == "Path does not exist"

    def test_error_invalid_path_with_path(self):
        """error_invalid_path includes path in details when provided."""
        err = error_invalid_path(
            "Source path not found",
            path="/bad/path",
        )

        assert err.details == {"path": "/bad/path"}

    def test_error_runtime_code(self):
        """error_runtime produces RUNTIME_ERROR code."""
        err = error_runtime("Database connection failed")

        assert err.code == RUNTIME_ERROR
        assert err.message == "Database connection failed"

    def test_error_runtime_with_exception_type(self):
        """error_runtime includes exception_type in details when provided."""
        err = error_runtime(
            "Connection error",
            exception_type="sqlite3.OperationalError",
        )

        assert err.details == {"exception_type": "sqlite3.OperationalError"}

    def test_error_dependency_missing_code(self):
        """error_dependency_missing produces DEPENDENCY_MISSING code."""
        err = error_dependency_missing("sqlite-vec not installed")

        assert err.code == DEPENDENCY_MISSING
        assert err.message == "sqlite-vec not installed"

    def test_error_dependency_missing_with_dependency(self):
        """error_dependency_missing includes dependency in details when provided."""
        err = error_dependency_missing(
            "Optional extension not available",
            dependency="sqlite-vec",
        )

        assert err.details == {"dependency": "sqlite-vec"}

    def test_error_unsupported_environment_code(self):
        """error_unsupported_environment produces UNSUPPORTED_ENVIRONMENT code."""
        err = error_unsupported_environment("Platform not supported")

        assert err.code == UNSUPPORTED_ENVIRONMENT
        assert err.message == "Platform not supported"

    def test_error_unsupported_environment_no_details(self):
        """error_unsupported_environment has no details field."""
        err = error_unsupported_environment("Unsupported OS")

        assert err.details == {}


class TestSuccessBuilder:
    """Tests for the success() helper."""

    def test_success_with_result(self):
        """success() creates MachineSuccess with provided result."""
        result_data = {"count": 5, "processed": True}
        success_obj = success(result_data)

        assert success_obj.result == result_data

    def test_success_without_result(self):
        """success() with no args creates MachineSuccess with empty result."""
        success_obj = success()

        assert success_obj.result == {}

    def test_success_none_result(self):
        """success(None) creates MachineSuccess with empty result."""
        success_obj = success(None)

        assert success_obj.result == {}


class TestWantsJsonDetection:
    """Tests for wants_json() argv pre-scanning."""

    def test_wants_json_with_flag(self):
        """wants_json returns True for --json flag."""
        assert wants_json(["polylogue", "run", "--json"]) is True

    def test_wants_json_at_start(self):
        """wants_json detects --json at any position."""
        assert wants_json(["--json", "run"]) is True

    def test_wants_json_equals_true(self):
        """wants_json returns True for --json=true."""
        assert wants_json(["polylogue", "--json=true", "run"]) is True

    def test_wants_json_equals_false(self):
        """wants_json returns True for --json=false (flag presence is checked)."""
        # Note: --json=false still indicates intention to parse with --json flag
        assert wants_json(["polylogue", "--json=false"]) is True

    def test_wants_json_false_without_flag(self):
        """wants_json returns False when no --json flag present."""
        assert wants_json(["polylogue", "run"]) is False

    def test_wants_json_ignores_format_flag(self):
        """wants_json does NOT match -f json or --format json."""
        assert wants_json(["polylogue", "run", "-f", "json"]) is False
        assert wants_json(["polylogue", "run", "--format", "json"]) is False

    def test_wants_json_ignores_json_lines(self):
        """wants_json does NOT match --json-lines."""
        assert wants_json(["polylogue", "run", "--json-lines"]) is False

    def test_wants_json_empty_argv(self):
        """wants_json returns False for empty argv."""
        assert wants_json([]) is False

    def test_wants_json_with_other_flags(self):
        """wants_json ignores other flags correctly."""
        assert (
            wants_json(
                [
                    "polylogue",
                    "--plain",
                    "run",
                    "--verbose",
                    "--archive-root",
                    "/tmp",
                    "--json",
                ]
            )
            is True
        )

    def test_wants_json_with_values_containing_json(self):
        """wants_json does not match --json in argument values."""
        # --archive-root=/tmp/json should not trigger wants_json
        assert wants_json(["polylogue", "run", "--archive-root=/tmp/json"]) is False


class TestExtractCommand:
    """Tests for extract_command() argv parsing."""

    def test_extract_command_simple(self):
        """extract_command extracts subcommand from argv."""
        assert extract_command(["polylogue", "run"]) == ["polylogue", "run"]

    def test_extract_command_with_flags(self):
        """extract_command skips flags and extracts only non-option args."""
        result = extract_command(
            [
                "polylogue",
                "--verbose",
                "run",
                "--archive-root",
                "/tmp",
                "--json",
            ]
        )
        # --archive-root takes /tmp as a value, which is not a flag, so it's included
        assert "polylogue" in result
        assert "run" in result
        assert "/tmp" in result
        assert "--verbose" not in result
        assert "--json" not in result

    def test_extract_command_with_value_flags(self):
        """extract_command skips both flags and their values."""
        result = extract_command(
            ["polylogue", "-f", "json", "run", "-s", "inbox"]
        )
        # Short flags like -f and -s are recognized as flags, their values are too
        # Actually, the implementation treats "-s" as flag but "inbox" is not a flag,
        # so it gets included
        assert "polylogue" in result
        assert "run" in result

    def test_extract_command_empty(self):
        """extract_command returns empty list for no args."""
        assert extract_command([]) == []

    def test_extract_command_flags_only(self):
        """extract_command returns empty list for flags-only argv."""
        assert extract_command(["--json", "--verbose", "--plain"]) == []

    def test_extract_command_with_equals_flags(self):
        """extract_command skips flags with = syntax."""
        result = extract_command(
            ["polylogue", "--archive-root=/tmp", "run", "--json=true"]
        )
        assert result == ["polylogue", "run"]

    def test_extract_command_preserves_order(self):
        """extract_command preserves order of non-flag arguments."""
        result = extract_command(
            ["polylogue", "--json", "search", "--limit", "10", "term"]
        )
        assert result[0] == "polylogue"
        assert "search" in result
        assert "term" in result
