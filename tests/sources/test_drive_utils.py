"""Drive utility function tests — parsing, ID detection, retry config, credential resolution."""

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
