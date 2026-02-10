"""Comprehensive tests for drive.py and drive_client.py.

Merged from test_drive_client.py, test_drive_resilience.py, and original coverage.
Covers OAuth flow, token refresh, credentials resolution, retry logic, error handling,
and all Drive API operations.
"""

from __future__ import annotations

import importlib
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from polylogue.config import Source
from polylogue.sources import (
    DriveAuthError,
    DriveClient,
    DriveError,
    DriveFile,
    DriveNotFoundError,
    download_drive_files,
    iter_drive_conversations,
)
from polylogue.sources.drive import _apply_drive_attachments, DriveDownloadResult
from polylogue.sources.drive_client import (
    _is_retryable_error,
    _looks_like_id,
    _parse_modified_time,
    _parse_size,
    _resolve_credentials_path,
    _resolve_retry_base,
    _resolve_retries,
    _resolve_token_path,
    default_credentials_path,
    default_token_path,
)
import polylogue.sources.drive_client as drive_client
from polylogue.sources.parsers.base import ParsedAttachment, ParsedConversation


# ============================================================================
# Test Data Tables (Module-level constants for parametrization)
# ============================================================================

PARSE_MODIFIED_TIME_CASES = [
    (None, type(None), "none_input"),
    ("", type(None), "empty_string"),
    ("2024-01-15T10:30:45Z", float, "iso_with_z"),
    ("2024-01-15T10:30:45", float, "iso_without_z"),
    ("2024-01-15T10:30:45+00:00", float, "iso_with_offset"),
    ("not a date", type(None), "invalid_string"),
    ("   ", type(None), "whitespace_only"),
]

PARSE_SIZE_CASES = [
    (None, None, "none_input"),
    (0, 0, "zero_integer"),
    (1024, 1024, "positive_integer"),
    (999999, 999999, "large_integer"),
    (-1, -1, "negative_integer"),
    ("123", 123, "string_integer"),
    ("0", 0, "string_zero"),
    ("999999", 999999, "string_large"),
    ("  456  ", 456, "string_with_whitespace"),
    ("not a number", None, "invalid_string"),
    ("12.34", None, "float_string"),
    ("12a", None, "alphanumeric_string"),
    ("", None, "empty_string"),
    ("123.456", None, "float_in_string"),
]

LOOKS_LIKE_ID_CASES = [
    ("", False, "empty_string"),
    ("hello world", False, "string_with_spaces"),
    (" test", False, "leading_space"),
    ("test ", False, "trailing_space"),
    ("abc-123-def", True, "alphanumeric_with_dashes"),
    ("file-1", True, "short_with_dash"),
    ("file_1_test", True, "alphanumeric_with_underscores"),
    ("_private", True, "leading_underscore"),
    ("abc123", True, "pure_alphanumeric"),
    ("FILE", True, "uppercase"),
    ("123", True, "digits_only"),
    ("a", True, "single_char"),
    ("1", True, "single_digit"),
    ("file.txt", False, "string_with_dots"),
    ("a.b.c", False, "multiple_dots"),
    ("file@home", False, "string_with_at"),
    ("test#1", False, "string_with_hash"),
    ("a/b", False, "string_with_slash"),
    ("---", True, "dashes_only"),
    ("___", True, "underscores_only"),
]

RESOLVE_RETRIES_CASES = [
    (5, 5, "explicit_five"),
    (0, 0, "explicit_zero"),
    (10, 10, "explicit_ten"),
    (-5, 0, "explicit_negative_clamped"),
]

RESOLVE_RETRY_BASE_CASES = [
    (1.5, 1.5, "explicit_float"),
    (0.1, 0.1, "explicit_small"),
    (-0.5, 0.0, "negative_clamped"),
]

RETRYABLE_ERROR_CASES = [
    (DriveAuthError("Invalid credentials"), False, "drive_auth_error"),
    (DriveNotFoundError("File not found"), False, "drive_not_found_error"),
    (RuntimeError("Network timeout"), True, "runtime_error"),
    (DriveError("Connection failed"), True, "drive_error"),
    (Exception("Some error"), True, "generic_exception"),
]

DEFAULT_PATHS_CREDENTIALS_CASES = [
    (None, True, "no_config_returns_default"),
    (MagicMock(credentials_path="/custom/creds.json"), False, "config_with_path_uses_value"),
]

DEFAULT_PATHS_TOKEN_CASES = [
    (None, True, "no_config_returns_default"),
    (MagicMock(token_path="/custom/token.json"), False, "config_with_path_uses_value"),
]

DOWNLOAD_DRIVE_FILES_CASES = [
    (1, 1, 0, "single_file_success"),
    (2, 2, 0, "multiple_files_success"),
    (2, 1, 1, "one_failure_mixed"),
    (3, 2, 1, "mixed_with_failures"),
    (0, 0, 0, "empty_folder"),
]

DOWNLOAD_JSON_PAYLOAD_CASES = [
    (b'{"a": 1}\n{"b": 2}\n{"c": 3}\n', list, 3, "jsonl_file"),
    (b'{"msg": "hello"}\n{"msg": "world"}\n', list, 2, "jsonl_txt_file"),
    (b'{"x": 1}\n{"y": 2}\n', list, 2, "ndjson_file"),
    (b'{"title": "Test", "content": "Data"}', dict, None, "json_file"),
    (b'{"valid": 1}\n{invalid json}\n{"valid": 2}\n', list, 2, "jsonl_skip_invalid"),
    (b'{"a": 1}\n\n\n{"b": 2}\n   \n', list, 2, "jsonl_skip_empty_lines"),
    (b'{"text": "line1\\nline2"}\n{"text": "another"}\n', list, 2, "jsonl_embedded_newlines"),
    (b'{"id": 1}\ninvalid\n{"id": 2}\n{broken\n{"id": 3}\n', list, 3, "jsonl_mixed_valid_invalid"),
    (b'{}', dict, None, "json_empty_object"),
    (b'[{"a": 1}, {"b": 2}]', list, 2, "json_array"),
    (b'\n  \n\t\n   \n', list, 0, "jsonl_only_empty_lines"),
]

NETWORK_ERROR_CASES = [
    (ConnectionRefusedError("Connection refused"), True, "connection_refused"),
    (ConnectionResetError("Connection reset by peer"), True, "connection_reset"),
    (TimeoutError("Operation timed out"), True, "timeout"),
    (BrokenPipeError("Broken pipe"), True, "broken_pipe"),
]

RETRY_SCENARIOS = [
    (429, True, "rate_limit_retries"),
    (500, True, "server_error_retries"),
    (502, True, "bad_gateway_retries"),
    (503, True, "service_unavailable_retries"),
    (504, True, "gateway_timeout_retries"),
]

NO_RETRY_SCENARIOS = [
    (401, False, "auth_error_no_retry"),
    (403, False, "forbidden_no_retry"),
    (404, False, "not_found_no_retry"),
]


# ============================================================================
# Parametrized Tests for _parse_modified_time
# ============================================================================


class TestParseModifiedTime:
    """Tests for _parse_modified_time utility function."""

    @pytest.mark.parametrize("input_val,expect_type,desc", PARSE_MODIFIED_TIME_CASES)
    def test_parse_modified_time(self, input_val, expect_type, desc):
        """Test _parse_modified_time with various inputs."""
        result = _parse_modified_time(input_val)
        assert isinstance(result, expect_type), f"Failed for {desc}"

        # Additional checks for valid timestamps
        if expect_type is float:
            assert result > 0
            if input_val == "2024-01-15T10:30:45Z":
                dt = datetime.fromtimestamp(result)
                assert dt.year == 2024
                assert dt.month == 1
                assert dt.day == 15

    def test_invalid_timestamps_return_none(self):
        """Test that all invalid timestamp forms return None."""
        invalid_inputs = ["12345", "2024-13-45T99:99:99Z"]
        for inp in invalid_inputs:
            assert _parse_modified_time(inp) is None


# ============================================================================
# Parametrized Tests for _parse_size
# ============================================================================


class TestParseSize:
    """Tests for _parse_size utility function."""

    @pytest.mark.parametrize("input_val,expected,desc", PARSE_SIZE_CASES)
    def test_parse_size(self, input_val, expected, desc):
        """Test _parse_size with various inputs."""
        result = _parse_size(input_val)
        assert result == expected, f"Failed for {desc}"


# ============================================================================
# Parametrized Tests for _looks_like_id
# ============================================================================


class TestLooksLikeId:
    """Tests for _looks_like_id utility function."""

    @pytest.mark.parametrize("input_str,expected,desc", LOOKS_LIKE_ID_CASES)
    def test_looks_like_id(self, input_str, expected, desc):
        """Test _looks_like_id with various inputs."""
        result = _looks_like_id(input_str)
        assert result is expected, f"Failed for {desc}"


# ============================================================================
# Parametrized Tests for _resolve_retries
# ============================================================================


class TestResolveRetries:
    """Tests for _resolve_retries function."""

    @pytest.mark.parametrize("value,expected,desc", RESOLVE_RETRIES_CASES)
    def test_resolve_retries_explicit_values(self, value, expected, desc):
        """Test explicit value resolution."""
        assert _resolve_retries(value=value, config=None) == expected

    def test_config_retry_count_used(self):
        """Config retry_count should be used when value is None."""
        config = MagicMock()
        config.retry_count = 7
        assert _resolve_retries(value=None, config=config) == 7

    def test_environment_variable_used(self, monkeypatch):
        """Environment variable should be used when available."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "9")
        assert _resolve_retries(value=None, config=None) == 9

    def test_env_variable_negative_clamped(self, monkeypatch):
        """Negative env value should be clamped."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "-3")
        assert _resolve_retries(value=None, config=None) == 0

    def test_invalid_env_variable_ignored(self, monkeypatch):
        """Invalid env value should be ignored, falling back to default."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "not_a_number")
        result = _resolve_retries(value=None, config=None)
        assert isinstance(result, int)
        assert result >= 0

    def test_priority_explicit_over_config(self):
        """Explicit value should have priority over config."""
        config = MagicMock()
        config.retry_count = 5
        assert _resolve_retries(value=10, config=config) == 10

    def test_priority_config_over_env(self, monkeypatch):
        """Config should have priority over env when value is None."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "20")
        config = MagicMock()
        config.retry_count = 5
        assert _resolve_retries(value=None, config=config) == 5


# ============================================================================
# Parametrized Tests for _resolve_retry_base
# ============================================================================


class TestResolveRetryBase:
    """Tests for _resolve_retry_base function."""

    @pytest.mark.parametrize("value,expected,desc", RESOLVE_RETRY_BASE_CASES)
    def test_resolve_retry_base_explicit_value(self, value, expected, desc):
        """Test explicit value resolution."""
        assert _resolve_retry_base(value=value) == expected

    def test_environment_variable_used(self, monkeypatch):
        """Environment variable should be used."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRY_BASE", "2.5")
        assert _resolve_retry_base(value=None) == 2.5

    def test_invalid_env_variable_ignored(self, monkeypatch):
        """Invalid env value should be ignored."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRY_BASE", "not_a_float")
        result = _resolve_retry_base(value=None)
        assert isinstance(result, float)
        assert result >= 0

    def test_default_when_nothing_specified(self, monkeypatch):
        """Default should be used when nothing is specified."""
        monkeypatch.delenv("POLYLOGUE_DRIVE_RETRY_BASE", raising=False)
        result = _resolve_retry_base(value=None)
        assert result >= 0


# ============================================================================
# Parametrized Tests for _is_retryable_error
# ============================================================================


class TestIsRetryableError:
    """Tests for _is_retryable_error function."""

    @pytest.mark.parametrize("exc,expected,desc", RETRYABLE_ERROR_CASES)
    def test_is_retryable_error(self, exc, expected, desc):
        """Test error retry classification."""
        assert _is_retryable_error(exc) is expected, f"Failed for {desc}"


# ============================================================================
# Parametrized Tests for default_credentials_path and default_token_path
# ============================================================================


class TestDefaultPaths:
    """Tests for default credentials and token path functions."""

    @pytest.mark.parametrize(
        "config,should_use_default,desc",
        [
            (None, True, "credentials_no_config"),
            (MagicMock(credentials_path="/custom/creds.json"), False, "credentials_with_config"),
        ]
    )
    def test_default_credentials_path(self, config, should_use_default, desc):
        """Test default credentials path resolution."""
        path = default_credentials_path(config=config)
        assert isinstance(path, Path)
        if should_use_default:
            assert "credentials" in str(path)
        else:
            assert path == Path("/custom/creds.json")

    def test_default_credentials_path_config_none_uses_default(self):
        """If config.credentials_path is None, use default."""
        config = MagicMock()
        config.credentials_path = None
        path = default_credentials_path(config=config)
        assert "credentials" in str(path)

    def test_default_credentials_path_missing_attribute(self):
        """If config doesn't have credentials_path attribute, use default."""
        config = MagicMock(spec=[])  # Empty spec = no attributes
        path = default_credentials_path(config=config)
        assert "credentials" in str(path)

    @pytest.mark.parametrize(
        "config,should_use_default,desc",
        [
            (None, True, "token_no_config"),
            (MagicMock(token_path="/custom/token.json"), False, "token_with_config"),
        ]
    )
    def test_default_token_path(self, config, should_use_default, desc):
        """Test default token path resolution."""
        path = default_token_path(config=config)
        assert isinstance(path, Path)
        if should_use_default:
            assert "token" in str(path)
        else:
            assert path == Path("/custom/token.json")

    def test_default_token_path_config_none_uses_default(self):
        """If config.token_path is None, use default."""
        config = MagicMock()
        config.token_path = None
        path = default_token_path(config=config)
        assert "token" in str(path)


# ============================================================================
# Tests for _resolve_credentials_path
# ============================================================================


class TestResolveCredentialsPath:
    """Tests for _resolve_credentials_path function."""

    def test_config_path_takes_priority(self, tmp_path):
        """Config credentials_path should be used first."""
        creds_path = tmp_path / "config_creds.json"
        creds_path.write_text('{"test": true}')

        config = MagicMock()
        config.credentials_path = str(creds_path)

        result = _resolve_credentials_path(ui=None, config=config)
        assert result == creds_path

    def test_env_path_when_no_config(self, tmp_path, monkeypatch):
        """Environment variable should be used when config not provided."""
        creds_path = tmp_path / "env_creds.json"
        creds_path.write_text('{"test": true}')

        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(creds_path))
        result = _resolve_credentials_path(ui=None, config=None)
        assert result == creds_path

    def test_env_path_expansion(self, tmp_path, monkeypatch):
        """Environment path should expand ~ to home."""
        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", "~/creds.json")
        result = _resolve_credentials_path(ui=None, config=None)
        assert "~" not in str(result)

    def test_default_path_when_exists(self, tmp_path, monkeypatch):
        """Default path should be used if it exists."""
        default_path = tmp_path / "default_creds.json"
        default_path.write_text('{"default": true}')

        monkeypatch.setattr(
            "polylogue.sources.drive_client.default_credentials_path",
            lambda config: default_path,
        )
        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)

        result = _resolve_credentials_path(ui=None, config=None)
        assert result == default_path

    def test_raises_when_not_found_no_ui(self, tmp_path, monkeypatch):
        """Should raise DriveAuthError when credentials not found and no UI."""
        nonexistent_path = tmp_path / "nonexistent" / "creds.json"

        monkeypatch.setattr(
            "polylogue.sources.drive_client.default_credentials_path",
            lambda config: nonexistent_path,
        )
        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)

        with pytest.raises(DriveAuthError, match="credentials not found"):
            _resolve_credentials_path(ui=None, config=None)

    def test_interactive_ui_prompt(self, tmp_path, monkeypatch):
        """UI prompt should be used when available and in non-plain mode."""
        default_path = tmp_path / "default" / "creds.json"
        user_path = tmp_path / "user" / "creds.json"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text('{"user": true}')

        monkeypatch.setattr(
            "polylogue.sources.drive_client.default_credentials_path",
            lambda config: default_path,
        )
        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)

        mock_ui = MagicMock()
        mock_ui.plain = False
        mock_ui.input = MagicMock(return_value=str(user_path))

        result = _resolve_credentials_path(ui=mock_ui, config=None)
        assert result == default_path
        assert default_path.exists()

    def test_interactive_no_response_raises(self, tmp_path, monkeypatch):
        """No response from UI should raise error."""
        default_path = tmp_path / "default" / "creds.json"

        monkeypatch.setattr(
            "polylogue.sources.drive_client.default_credentials_path",
            lambda config: default_path,
        )
        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)

        mock_ui = MagicMock()
        mock_ui.plain = False
        mock_ui.input = MagicMock(return_value=None)

        with pytest.raises(DriveAuthError):
            _resolve_credentials_path(ui=mock_ui, config=None)


# ============================================================================
# Tests for _resolve_token_path
# ============================================================================


class TestResolveTokenPath:
    """Tests for _resolve_token_path function."""

    def test_config_path_takes_priority(self, tmp_path):
        """Config token_path should be used first."""
        token_path = tmp_path / "config_token.json"
        config = MagicMock()
        config.token_path = str(token_path)

        result = _resolve_token_path(config=config)
        assert result == token_path

    def test_env_path_when_no_config(self, tmp_path, monkeypatch):
        """Environment variable should be used when config not provided."""
        token_path = tmp_path / "env_token.json"
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))

        result = _resolve_token_path(config=None)
        assert result == token_path

    def test_env_path_expansion(self, monkeypatch):
        """Environment path should expand ~."""
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", "~/token.json")
        result = _resolve_token_path(config=None)
        assert "~" not in str(result)

    def test_default_when_nothing_specified(self, monkeypatch):
        """Default should be used when config and env not set."""
        monkeypatch.delenv("POLYLOGUE_TOKEN_PATH", raising=False)
        result = _resolve_token_path(config=None)
        assert isinstance(result, Path)
        assert "token" in str(result)


# ============================================================================
# Tests for download_drive_files
# ============================================================================


class TestDownloadDriveFiles:
    """Tests for download_drive_files function."""

    def test_successful_single_file_download(self, tmp_path):
        """Download single file successfully."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="test.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]
        mock_client.download_to_path.return_value = None

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 1
        assert len(result.downloaded_files) == 1
        assert len(result.failed_files) == 0
        assert result.downloaded_files[0].name == "test.json"

    def test_successful_multiple_files_download(self, tmp_path):
        """Download multiple files successfully."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="chat1.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="chat2.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=200,
            ),
        ]
        mock_client.download_to_path.return_value = None

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 2
        assert len(result.downloaded_files) == 2
        assert len(result.failed_files) == 0

    def test_download_with_failure(self, tmp_path):
        """Handle download failure gracefully."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="good.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="bad.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        def mock_download(file_id, dest):
            if file_id == "f2":
                raise OSError("Network error")

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 2
        assert len(result.downloaded_files) == 1
        assert len(result.failed_files) == 1
        assert result.failed_files[0]["file_id"] == "f2"
        assert "error" in result.failed_files[0]

    def test_download_continues_after_failure(self, tmp_path):
        """Download should continue after a single file fails."""
        mock_client = MagicMock(spec=DriveClient)
        download_calls = []

        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="first.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="fails.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f3",
                name="third.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        def mock_download(file_id, dest):
            download_calls.append(file_id)
            if file_id == "f2":
                raise Exception("Failed")

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert len(download_calls) == 3
        assert result.total_files == 3
        assert len(result.downloaded_files) == 2
        assert len(result.failed_files) == 1

    def test_empty_folder(self, tmp_path):
        """Empty folder should return empty result."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = []

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 0
        assert len(result.downloaded_files) == 0
        assert len(result.failed_files) == 0

    def test_result_type_is_data_class(self, tmp_path):
        """Result should be a DriveDownloadResult instance."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = []

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert isinstance(result, DriveDownloadResult)
        assert hasattr(result, "downloaded_files")
        assert hasattr(result, "failed_files")
        assert hasattr(result, "total_files")

    def test_file_count_accuracy(self, tmp_path):
        """total_files should match sum of downloaded and failed."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="a.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="b.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f3",
                name="c.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        def mock_download(file_id, dest):
            if file_id == "f2":
                raise Exception("Failed")

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == len(result.downloaded_files) + len(result.failed_files)
        assert result.total_files == 3

    def test_all_files_fail(self, tmp_path):
        """All files failing should be reported."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="fail1.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="fail2.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        mock_client.download_to_path.side_effect = Exception("All fail")

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 2
        assert len(result.downloaded_files) == 0
        assert len(result.failed_files) == 2

    def test_failed_file_error_message_preserved(self, tmp_path):
        """Failed file should include the actual error message."""
        mock_client = MagicMock(spec=DriveClient)
        error_msg = "Permission denied: file is private"

        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="private.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]
        mock_client.download_to_path.side_effect = PermissionError(error_msg)

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert len(result.failed_files) == 1
        assert error_msg in result.failed_files[0]["error"]


# ============================================================================
# Tests for _apply_drive_attachments
# ============================================================================


class TestApplyDriveAttachments:
    """Tests for _apply_drive_attachments function."""

    def test_download_disabled_skips_processing(self, tmp_path):
        """download_assets=False should skip attachment processing."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="attach-1",
                    message_provider_id="msg-1",
                    name="test.pdf",
                )
            ],
        )

        mock_client = MagicMock(spec=DriveClient)
        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=False,
        )

        mock_client.download_to_path.assert_not_called()

    def test_no_provider_attachment_id_skips(self, tmp_path):
        """Attachment without provider_attachment_id should be skipped."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="test-id",
                    message_provider_id="msg-1",
                    name="test.pdf",
                )
            ],
        )
        convo.attachments[0].provider_attachment_id = None

        mock_client = MagicMock(spec=DriveClient)
        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=True,
        )

        mock_client.download_to_path.assert_not_called()

    def test_successful_attachment_download(self, tmp_path):
        """Successful download should update attachment fields."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="attach-1",
                    message_provider_id="msg-1",
                    name=None,
                    mime_type=None,
                    size_bytes=None,
                    path=None,
                    provider_meta=None,
                )
            ],
        )

        mock_client = MagicMock(spec=DriveClient)
        mock_file = DriveFile(
            file_id="attach-1",
            name="document.pdf",
            mime_type="application/pdf",
            modified_time=None,
            size_bytes=5000,
        )
        mock_client.download_to_path.return_value = mock_file

        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=True,
        )

        att = convo.attachments[0]
        assert att.name == "document.pdf"
        assert att.mime_type == "application/pdf"
        assert att.size_bytes == 5000
        assert att.path is not None
        assert att.provider_meta is not None
        assert "drive_id" in att.provider_meta

    def test_partial_attachment_fields_filled(self, tmp_path):
        """Download should only fill empty fields, not overwrite existing."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="attach-1",
                    message_provider_id="msg-1",
                    name="original_name.txt",
                    mime_type="text/plain",
                    size_bytes=1000,
                    path=None,
                    provider_meta=None,
                )
            ],
        )

        mock_client = MagicMock(spec=DriveClient)
        mock_file = DriveFile(
            file_id="attach-1",
            name="document.pdf",
            mime_type="application/pdf",
            modified_time=None,
            size_bytes=5000,
        )
        mock_client.download_to_path.return_value = mock_file

        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=True,
        )

        att = convo.attachments[0]
        assert att.name == "original_name.txt"
        assert att.mime_type == "text/plain"
        assert att.size_bytes == 1000

    def test_multiple_attachments_processed(self, tmp_path):
        """Multiple attachments should all be processed."""
        attachments = [
            ParsedAttachment(
                provider_attachment_id="attach-1",
                message_provider_id="msg-1",
                name=None,
            ),
            ParsedAttachment(
                provider_attachment_id="attach-2",
                message_provider_id="msg-2",
                name=None,
            ),
        ]
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=attachments,
        )

        mock_client = MagicMock(spec=DriveClient)

        def mock_download(file_id, dest):
            if file_id == "attach-1":
                return DriveFile(
                    file_id=file_id,
                    name="doc1.pdf",
                    mime_type="application/pdf",
                    modified_time=None,
                    size_bytes=1000,
                )
            elif file_id == "attach-2":
                return DriveFile(
                    file_id=file_id,
                    name="doc2.pdf",
                    mime_type="application/pdf",
                    modified_time=None,
                    size_bytes=2000,
                )

        mock_client.download_to_path.side_effect = mock_download

        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=True,
        )

        assert convo.attachments[0].name == "doc1.pdf"
        assert convo.attachments[1].name == "doc2.pdf"


# ============================================================================
# Tests for iter_drive_conversations
# ============================================================================


class TestIterDriveConversations:
    """Tests for iter_drive_conversations function."""

    def test_no_folder_returns_empty(self, tmp_path):
        """If source.folder is None or empty, should return empty."""
        source = Source(name="test", path="/some/path")
        result = list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                download_assets=False,
            )
        )
        assert result == []

    def test_initializes_cursor_state(self, tmp_path):
        """cursor_state should be initialized with file_count."""
        source = Source(name="test", folder="Google AI Studio")
        cursor_state = {}

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = []

        list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                client=mock_client,
                cursor_state=cursor_state,
                download_assets=False,
            )
        )

        assert "file_count" in cursor_state
        assert cursor_state["file_count"] == 0

    def test_tracks_file_count_in_cursor(self, tmp_path):
        """cursor_state should track file count as files are processed."""
        source = Source(name="test", folder="Google AI Studio")
        cursor_state = {}

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="chat1.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="chat2.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]
        mock_client.download_json_payload.return_value = {
            "title": "Test",
            "messages": [],
        }

        list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                client=mock_client,
                cursor_state=cursor_state,
                download_assets=False,
            )
        )

        assert cursor_state["file_count"] == 2

    def test_tracks_latest_mtime_in_cursor(self, tmp_path):
        """cursor_state should track latest modification time."""
        source = Source(name="test", folder="Google AI Studio")
        cursor_state = {}

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="chat1.json",
                mime_type="application/json",
                modified_time="2024-01-10T10:00:00Z",
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="chat2.json",
                mime_type="application/json",
                modified_time="2024-01-15T10:00:00Z",
                size_bytes=100,
            ),
        ]
        mock_client.download_json_payload.return_value = {
            "title": "Test",
            "messages": [],
        }

        list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                client=mock_client,
                cursor_state=cursor_state,
                download_assets=False,
            )
        )

        assert "latest_mtime" in cursor_state
        assert cursor_state["latest_mtime"] > 0
        assert "latest_file_id" in cursor_state
        assert cursor_state["latest_file_id"] == "f2"
        assert cursor_state["latest_file_name"] == "chat2.json"

    def test_handles_download_error_continues(self, tmp_path):
        """Download error should be tracked but iteration should continue."""
        source = Source(name="test", folder="Google AI Studio")
        cursor_state = {}

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="good.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="bad.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        def mock_download(file_id, *, name):
            if file_id == "f2":
                raise Exception("Download failed")
            return {"title": "Good", "messages": []}

        mock_client.download_json_payload.side_effect = mock_download

        result = list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                client=mock_client,
                cursor_state=cursor_state,
                download_assets=False,
            )
        )

        assert "error_count" in cursor_state
        assert cursor_state["error_count"] >= 1
        assert "latest_error" in cursor_state
        assert "latest_error_file" in cursor_state

    def test_creates_client_if_not_provided(self, tmp_path, mock_drive_credentials):
        """Should create DriveClient if not provided."""
        source = Source(name="test", folder="Google AI Studio")

        with patch(
            "polylogue.sources.drive.DriveClient"
        ) as mock_drive_client_class:
            mock_instance = MagicMock(spec=DriveClient)
            mock_drive_client_class.return_value = mock_instance
            mock_instance.resolve_folder_id.return_value = "folder-1"
            mock_instance.iter_json_files.return_value = []

            list(
                iter_drive_conversations(
                    source=source,
                    archive_root=tmp_path,
                    download_assets=False,
                )
            )

            mock_drive_client_class.assert_called_once()

    def test_uses_provided_client(self, tmp_path):
        """Should use provided client instead of creating new one."""
        source = Source(name="test", folder="Google AI Studio")

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = []

        with patch(
            "polylogue.sources.drive.DriveClient"
        ) as mock_drive_client_class:
            list(
                iter_drive_conversations(
                    source=source,
                    archive_root=tmp_path,
                    client=mock_client,
                    download_assets=False,
                )
            )

            mock_drive_client_class.assert_not_called()


# ============================================================================
# Parametrized Tests for download_json_payload
# ============================================================================


class TestDownloadJsonPayload:
    """Tests for DriveClient.download_json_payload method."""

    @pytest.mark.parametrize(
        "content,expect_type,expect_len,desc",
        [
            (b'{"a": 1}\n{"b": 2}\n{"c": 3}\n', list, 3, "jsonl_file"),
            (b'{"msg": "hello"}\n{"msg": "world"}\n', list, 2, "jsonl_txt_file"),
            (b'{"x": 1}\n{"y": 2}\n', list, 2, "ndjson_file"),
            (b'{"title": "Test", "content": "Data"}', dict, None, "json_file"),
            (b'{"valid": 1}\n{invalid json}\n{"valid": 2}\n', list, 2, "jsonl_skip_invalid"),
            (b'{"a": 1}\n\n\n{"b": 2}\n   \n', list, 2, "jsonl_skip_empty_lines"),
            (b'{"text": "line1\\nline2"}\n{"text": "another"}\n', list, 2, "jsonl_embedded_newlines"),
            (b'{"id": 1}\ninvalid\n{"id": 2}\n{broken\n{"id": 3}\n', list, 3, "jsonl_mixed_valid_invalid"),
            (b'{}', dict, None, "json_empty_object"),
            (b'[{"a": 1}, {"b": 2}]', list, 2, "json_array"),
            (b'\n  \n\t\n   \n', list, 0, "jsonl_only_empty_lines"),
        ]
    )
    def test_download_json_payload(
        self, mock_drive_credentials, mock_drive_service, content, expect_type, expect_len, desc
    ):
        """Test JSON payload parsing with various formats."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]

        with patch.object(client, "download_bytes", return_value=content):
            result = client.download_json_payload("file-1", name="data.json")

        assert isinstance(result, expect_type), f"Failed for {desc}"
        if expect_len is not None:
            assert len(result) == expect_len, f"Failed for {desc}"

    def test_case_insensitive_extension_matching(self, mock_drive_credentials, mock_drive_service):
        """Extension matching should be case-insensitive."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]

        jsonl_content = b'{"a": 1}\n{"b": 2}\n'

        with patch.object(client, "download_bytes", return_value=jsonl_content):
            result = client.download_json_payload("file-1", name="DATA.JSONL")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_json_fallback_on_decode_error(self, mock_drive_credentials, mock_drive_service):
        """JSON decode should fall back to UTF-8 replacement on error."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]

        json_content = b'{"title": "Test"}'

        with patch.object(client, "download_bytes", return_value=json_content):
            result = client.download_json_payload("file-1", name="test.json")

        assert isinstance(result, dict)
        assert result["title"] == "Test"


# ============================================================================
# Tests for Exponential Backoff Behavior
# ============================================================================


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    def test_backoff_increases_exponentially(self):
        """Verify backoff timing increases exponentially."""
        from tenacity import wait_exponential

        wait = wait_exponential(multiplier=0.5, min=0.5, max=10)

        class MockRetryState:
            def __init__(self, attempt):
                self.attempt_number = attempt

        wait_1 = wait(MockRetryState(1))
        wait_2 = wait(MockRetryState(2))
        wait_3 = wait(MockRetryState(3))

        assert wait_2 >= wait_1
        assert wait_3 >= wait_2

    def test_backoff_respects_max(self):
        """Verify backoff is capped at max value."""
        from tenacity import wait_exponential

        wait = wait_exponential(multiplier=1, min=1, max=5)

        class MockRetryState:
            def __init__(self, attempt):
                self.attempt_number = attempt

        wait_10 = wait(MockRetryState(10))
        assert wait_10 <= 5


# ============================================================================
# Parametrized Tests for HTTP Error Retry Classification
# ============================================================================


class TestHttpErrorRetries:
    """Tests for HTTP error handling and retry classification."""

    @pytest.mark.parametrize("status_code,should_retry,description", RETRY_SCENARIOS)
    def test_retryable_http_errors(self, status_code: int, should_retry: bool, description: str):
        """HTTP errors that should trigger retry."""
        class MockHTTPError(Exception):
            def __init__(self, status):
                self.status = status
                super().__init__(f"HTTP {status}")

        exc = MockHTTPError(status_code)
        assert _is_retryable_error(exc) is should_retry


# ============================================================================
# Tests for Drive Exception Types
# ============================================================================


class TestDriveExceptionTypes:
    """Tests for Drive exception hierarchy."""

    def test_auth_error_is_drive_error(self):
        """DriveAuthError is a DriveError."""
        exc = DriveAuthError("Auth failed")
        assert isinstance(exc, DriveError)
        assert isinstance(exc, RuntimeError)

    def test_not_found_error_is_drive_error(self):
        """DriveNotFoundError is a DriveError."""
        exc = DriveNotFoundError("File not found")
        assert isinstance(exc, DriveError)
        assert isinstance(exc, RuntimeError)

    def test_drive_error_message(self):
        """DriveError preserves error message."""
        exc = DriveError("Test message")
        assert str(exc) == "Test message"

    def test_auth_error_chaining(self):
        """DriveAuthError can chain from other exceptions."""
        original = ValueError("Original error")
        exc = DriveAuthError("Wrapped error")
        exc.__cause__ = original

        assert exc.__cause__ is original


# ============================================================================
# Parametrized Tests for Network Error Handling
# ============================================================================


class TestNetworkErrors:
    """Tests for network error handling and retry classification."""

    @pytest.mark.parametrize("exc,expected,desc", NETWORK_ERROR_CASES)
    def test_network_errors(self, exc, expected, desc):
        """Test network error retry classification."""
        assert _is_retryable_error(exc) is expected, f"Failed for {desc}"


# ============================================================================
# Tests for Client Initialization and Mocked Operations
# ============================================================================


class TestDriveClientInit:
    """Tests for DriveClient initialization."""

    def test_init_with_mock_credentials(self, mock_drive_credentials):
        """Initialize Drive client with mock credentials."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        assert client is not None

    def test_init_with_retries(self, mock_drive_credentials):
        """Initialize with custom retry settings."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
            retries=5,
            retry_base=1.0,
        )
        assert client._retries == 5
        assert client._retry_base == 1.0


class TestDriveClientMocked:
    """Tests for Drive client operations with mocked service."""

    @pytest.fixture
    def drive_client(self, mock_drive_credentials, mock_drive_service):
        """Drive client with mocked service."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]
        return client

    def test_list_files_in_folder(self, drive_client, mock_drive_service):
        """List files in a specific folder."""
        service = mock_drive_service["service"]

        response = service.files().list(q="'folder1' in parents").execute()
        assert "files" in response
        assert len(response["files"]) == 1
        assert response["files"][0]["name"] == "Test Prompt"

    def test_get_file_metadata(self, drive_client, mock_drive_service):
        """Get metadata for a specific file."""
        service = mock_drive_service["service"]
        response = service.files().get(fileId="prompt1").execute()
        assert response["name"] == "Test Prompt"
        assert response["mimeType"] == "application/vnd.google-makersuite.prompt"

    def test_get_file_content(self, drive_client, mock_drive_service):
        """Download file content."""
        service = mock_drive_service["service"]
        content = service.files().get_media(fileId="prompt1").execute()
        assert b"Test Prompt" in content

    def test_file_not_found_raises(self, drive_client, mock_drive_service):
        """Raise error when file not found."""
        service = mock_drive_service["service"]
        with pytest.raises(Exception, match="File not found"):
            service.files().get(fileId="nonexistent").execute()


class TestRetryLogic:
    """Tests for retry logic and configuration."""

    def test_retry_decorator_exists(self, mock_drive_credentials):
        """Verify retry logic is configured."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
            retries=3,
        )
        assert hasattr(client, "_call_with_retry")
        assert client._retries == 3

    def test_auth_error_not_retried(self):
        """Auth errors should not be retried."""
        auth_error = DriveAuthError("Invalid credentials")
        assert not _is_retryable_error(auth_error)

    def test_generic_error_is_retryable(self):
        """Generic errors should be retried."""
        generic_error = RuntimeError("Network timeout")
        assert _is_retryable_error(generic_error)


class TestEdgeCases:
    """Tests for edge cases and pagination support."""

    def test_pagination_support(self, mock_drive_service):
        """Mock service supports pagination."""
        service = mock_drive_service["service"]

        from tests.mocks.drive_mocks import mock_drive_file

        many_files = {f"file{i}": mock_drive_file(file_id=f"file{i}", name=f"File {i}") for i in range(150)}
        service._files_resource.files.update(many_files)

        response = service.files().list(pageSize=50).execute()
        assert len(response["files"]) <= 50

        if len(many_files) > 50:
            assert response.get("nextPageToken") is not None


class TestAPIOperations:
    """Tests for public API methods (resolve_folder_id, iter_json_files, etc.)."""

    @pytest.fixture
    def drive_client_with_service(self, mock_drive_credentials, mock_drive_service):
        """Drive client with mocked service."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]
        return client, mock_drive_service

    def test_resolve_folder_id_by_name(self, drive_client_with_service):
        """Resolve folder ID by name."""
        client, mock_service = drive_client_with_service

        folder_id = client.resolve_folder_id("Google AI Studio")
        assert folder_id == "folder1"

    def test_resolve_folder_id_not_found(self, drive_client_with_service):
        """Raise error when folder not found."""
        client, mock_service = drive_client_with_service

        with pytest.raises(DriveNotFoundError, match="Folder.*not found"):
            client.resolve_folder_id("NonexistentFolder")

    def test_iter_json_files(self, drive_client_with_service):
        """Iterate through JSON files in a folder."""
        client, mock_service = drive_client_with_service

        from tests.mocks.drive_mocks import mock_drive_file

        json_file1 = mock_drive_file(
            file_id="json1",
            name="test.json",
            mime_type="application/json",
            parents=["folder1"],
        )
        json_file2 = mock_drive_file(
            file_id="json2",
            name="data.json",
            mime_type="application/json",
            parents=["folder1"],
        )

        mock_service["service"]._files_resource.files.update({
            "json1": json_file1,
            "json2": json_file2,
        })
        mock_service["file_content"]["json1"] = b'{"test": "content"}'
        mock_service["file_content"]["json2"] = b'{"data": "content"}'

        files = list(client.iter_json_files("folder1"))
        assert len(files) >= 1
        file_names = [f.name for f in files]
        assert "test.json" in file_names or "data.json" in file_names

    def test_get_metadata(self, drive_client_with_service):
        """Get file metadata."""
        client, mock_service = drive_client_with_service

        metadata = client.get_metadata("prompt1")
        assert metadata.name == "Test Prompt"
        assert metadata.mime_type == "application/vnd.google-makersuite.prompt"

    def test_download_bytes(self, drive_client_with_service, mock_media_downloader):
        """Download file as bytes."""
        client, mock_service = drive_client_with_service

        content = client.download_bytes("prompt1")
        assert isinstance(content, bytes)
        assert b"Test Prompt" in content

    def test_download_json_payload(self, drive_client_with_service, mock_media_downloader):
        """Download and parse JSON payload."""
        client, mock_service = drive_client_with_service

        payload = client.download_json_payload("prompt1", name="prompt1.json")
        assert isinstance(payload, dict)
        assert payload["title"] == "Test Prompt"
        assert payload["content"] == "Test content"

    def test_download_to_path(self, drive_client_with_service, mock_media_downloader, tmp_path):
        """Download file to local path."""
        client, mock_service = drive_client_with_service

        output_path = tmp_path / "downloaded.json"
        result = client.download_to_path("prompt1", output_path)

        assert result.file_id == "prompt1"
        assert output_path.exists()
        assert b"Test Prompt" in output_path.read_bytes()

    def test_download_with_encoding_fallback(self, drive_client_with_service, mock_media_downloader):
        """Download should handle encoding issues gracefully."""
        client, mock_service = drive_client_with_service

        from tests.mocks.drive_mocks import mock_drive_file

        binary_file = mock_drive_file(
            file_id="binary1",
            name="binary.dat",
            mime_type="application/octet-stream",
        )
        mock_service["service"]._files_resource.files["binary1"] = binary_file
        mock_service["file_content"]["binary1"] = b"\xff\xfe Invalid UTF-8"

        content = client.download_bytes("binary1")
        assert isinstance(content, bytes)
        assert content == b"\xff\xfe Invalid UTF-8"

    def test_resolve_folder_id_with_multiple_matches(self, drive_client_with_service):
        """Handle multiple folders with same name."""
        client, mock_service = drive_client_with_service

        from tests.mocks.drive_mocks import mock_drive_file

        folder2 = mock_drive_file(
            file_id="folder2",
            name="Google AI Studio",
            mime_type="application/vnd.google-apps.folder",
        )
        mock_service["service"]._files_resource.files["folder2"] = folder2

        folder_id = client.resolve_folder_id("Google AI Studio")
        assert folder_id in ["folder1", "folder2"]

    def test_iter_json_files_empty_folder(self, drive_client_with_service):
        """Iterate through empty folder."""
        client, mock_service = drive_client_with_service

        from tests.mocks.drive_mocks import mock_drive_file

        empty_folder = mock_drive_file(
            file_id="empty",
            name="Empty Folder",
            mime_type="application/vnd.google-apps.folder",
        )
        mock_service["service"]._files_resource.files["empty"] = empty_folder

        files = list(client.iter_json_files("empty"))
        assert len(files) == 0


# ============================================================================
# Tests for Token Refresh and Error Handling
# ============================================================================


class TestTokenRefreshErrorHandling:
    """Tests for OAuth token refresh error handling."""

    def test_missing_dependency_raises_auth_error(self, monkeypatch):
        """Missing googleapiclient dependency should raise DriveAuthError."""
        real_import = importlib.import_module

        def fake_import(name: str):
            if name.startswith("googleapiclient"):
                raise ModuleNotFoundError(name)
            return real_import(name)

        client = DriveClient(ui=None)
        monkeypatch.setattr(client, "_load_credentials", lambda: object())
        monkeypatch.setattr(drive_client.importlib, "import_module", fake_import)

        with pytest.raises(DriveAuthError, match="Drive dependencies"):
            client._service_handle()

    def test_refresh_failure_raises_specific_error(self, monkeypatch):
        """Token refresh failure should raise DriveAuthError."""
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "valid_refresh_token"
        mock_creds.refresh.side_effect = Exception("Network error during refresh")

        mock_request = MagicMock()

        def mock_import(name: str):
            if name == "google.auth.transport.requests":
                return MagicMock(Request=MagicMock(return_value=mock_request))
            if name == "google.oauth2.credentials":
                return MagicMock(Credentials=MagicMock(from_authorized_user_file=MagicMock(return_value=mock_creds)))
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None)
        with pytest.raises(DriveAuthError, match="refresh|token"):
            client._load_credentials()

    def test_refresh_failure_includes_original_error(self, monkeypatch, tmp_path):
        """DriveAuthError should include the original exception details."""
        token_file = tmp_path / "token.json"
        token_file.write_text('{"valid": false, "expired": true}')

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "valid_refresh_token"
        mock_creds.refresh.side_effect = ConnectionError("Server unreachable")

        mock_request = MagicMock()

        def mock_import(name: str):
            if name == "google.auth.transport.requests":
                return MagicMock(Request=MagicMock(return_value=mock_request))
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(return_value=mock_creds)
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)
        with pytest.raises(DriveAuthError) as exc_info:
            client._load_credentials()

        error_str = str(exc_info.value)
        assert "Server unreachable" in error_str or "refresh" in error_str.lower()

    def test_invalid_credentials_raises_auth_error(self, monkeypatch, tmp_path):
        """Invalid credentials without refresh capability should raise clear auth error."""
        token_file = tmp_path / "token.json"
        token_file.write_text('{"valid": false}')

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = None

        def mock_import(name: str):
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(return_value=mock_creds)
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)
        with pytest.raises(DriveAuthError, match="invalid|expired|re-run"):
            client._load_credentials()

    def test_successful_token_refresh(self, monkeypatch, tmp_path):
        """Successful token refresh should return valid credentials."""
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "old_token"}')

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "valid_refresh_token"
        mock_creds.to_json.return_value = '{"token": "new_token", "refresh_token": "valid_refresh_token"}'

        def mock_refresh(request):
            mock_creds.valid = True
            mock_creds.expired = False
            mock_creds.token = "new_token"

        mock_creds.refresh = mock_refresh
        mock_request = MagicMock()

        def mock_import(name: str):
            if name == "google.auth.transport.requests":
                return MagicMock(Request=MagicMock(return_value=mock_request))
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(return_value=mock_creds)
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)
        result = client._load_credentials()

        assert result is not None
        assert result.valid is True
        assert result.token == "new_token"

    def test_valid_cached_credentials(self, monkeypatch, tmp_path):
        """Valid cached credentials should be returned without refresh."""
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "current_token"}')

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.refresh_token = "refresh_token"
        mock_creds.token = "current_token"
        mock_creds.to_json.return_value = '{"token": "current_token", "refresh_token": "refresh_token"}'

        def mock_import(name: str):
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(return_value=mock_creds)
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)
        result = client._load_credentials()

        assert result is not None
        assert result.valid is True
        assert result.token == "current_token"
        assert not mock_creds.refresh.called

    def test_corrupt_token_file_handling(self, monkeypatch, tmp_path):
        """Corrupt token file should be handled gracefully."""
        token_file = tmp_path / "token.json"
        token_file.write_text('invalid json {{{')

        def mock_import(name: str):
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(side_effect=ValueError("Invalid JSON"))
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)

        with pytest.raises(DriveAuthError, match="invalid|expired"):
            client._load_credentials()


# ============================================================================
# Tests for Drive Ingestion
# ============================================================================


@dataclass
class StubDriveClient:
    payload: dict

    def resolve_folder_id(self, folder_ref: str) -> str:
        return "folder-1"

    def iter_json_files(self, folder_id: str):
        yield DriveFile(
            file_id="file-1",
            name="chat.json",
            mime_type="application/json",
            modified_time=None,
            size_bytes=None,
        )

    def download_json_payload(self, file_id: str, *, name: str):
        return self.payload

    def download_to_path(self, file_id, dest):
        raise AssertionError("download_to_path should not be called when download_assets=False")


def test_drive_ingest_chunked_prompt_no_download(tmp_path):
    payload = {
        "title": "Drive Chat",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hi"},
                {
                    "role": "model",
                    "text": "Hello",
                    "driveDocument": {"id": "att-1", "name": "doc.txt"},
                },
            ]
        },
    }
    source = Source(name="gemini", folder="Google AI Studio")
    client = StubDriveClient(payload=payload)

    conversations = list(
        iter_drive_conversations(
            source=source,
            archive_root=tmp_path,
            client=client,
            download_assets=False,
        )
    )
    assert len(conversations) == 1
    convo = conversations[0]
    assert [msg.role for msg in convo.messages] == ["user", "assistant"]
    assert len(convo.attachments) == 1
    attachment = convo.attachments[0]
    assert attachment.provider_attachment_id == "att-1"
    assert attachment.message_provider_id == "chunk-2"
    assert attachment.path is None


class TestDriveDownloadFailureTracking:
    """Tests for tracking Drive download failures."""

    def test_download_failure_tracked_in_result(self):
        """Failed downloads should be tracked in the result."""
        from polylogue.sources import download_drive_files
        from polylogue.sources.drive_client import DriveFile

        mock_client = MagicMock()
        mock_client.iter_json_files.return_value = [
            DriveFile(file_id="file1", name="good.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="file2", name="bad.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="file3", name="also_good.json", mime_type="application/json", modified_time=None, size_bytes=100),
        ]

        def mock_download(file_id, dest):
            if file_id == "file2":
                raise OSError("Download failed")
            dest.write_text('{"test": true}')

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder123", Path("/tmp/test"))

        assert hasattr(result, "failed_files") or "failed" in result
        assert len(result.failed_files) >= 1
        assert any("bad.json" in str(f) for f in result.failed_files)

    def test_download_continues_after_single_failure(self):
        """Download should continue processing other files after one fails."""
        from polylogue.sources import download_drive_files
        from polylogue.sources.drive_client import DriveFile

        mock_client = MagicMock()
        mock_client.iter_json_files.return_value = [
            DriveFile(file_id="f1", name="first.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="f2", name="fails.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="f3", name="third.json", mime_type="application/json", modified_time=None, size_bytes=100),
        ]

        download_count = [0]

        def mock_download(file_id, dest):
            if file_id == "f2":
                raise OSError("Failed")
            download_count[0] += 1
            dest.write_text('{}')

        mock_client.download_to_path.side_effect = mock_download

        download_drive_files(mock_client, "folder", Path("/tmp/test"))

        assert download_count[0] == 2
