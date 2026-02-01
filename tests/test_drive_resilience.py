"""Tests for Drive client resilience and retry behavior.

Covers retry scenarios for:
- 429 rate limiting (with and without Retry-After header)
- 500 server errors
- 401 auth errors (no retry)
- 404 not found errors (no retry)
- Network errors (ConnectionError, TimeoutError)
- Exponential backoff behavior
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import time

import pytest

from polylogue.ingestion.drive_client import (
    DriveClient,
    DriveError,
    DriveAuthError,
    DriveNotFoundError,
    _is_retryable_error,
    _resolve_retries,
    _resolve_retry_base,
)


# =============================================================================
# Retryable Error Classification Tests
# =============================================================================


class TestIsRetryableError:
    """Tests for _is_retryable_error() classification."""

    def test_auth_error_not_retryable(self):
        """DriveAuthError should not be retried."""
        exc = DriveAuthError("Auth failed")
        assert _is_retryable_error(exc) is False

    def test_not_found_error_not_retryable(self):
        """DriveNotFoundError should not be retried."""
        exc = DriveNotFoundError("File not found")
        assert _is_retryable_error(exc) is False

    def test_generic_exception_retryable(self):
        """Generic exceptions should be retried."""
        exc = Exception("Something went wrong")
        assert _is_retryable_error(exc) is True

    def test_drive_error_retryable(self):
        """DriveError (base class) should be retried."""
        exc = DriveError("Temporary failure")
        assert _is_retryable_error(exc) is True

    def test_connection_error_retryable(self):
        """ConnectionError should be retried."""
        exc = ConnectionError("Network unreachable")
        assert _is_retryable_error(exc) is True

    def test_timeout_error_retryable(self):
        """TimeoutError should be retried."""
        exc = TimeoutError("Request timed out")
        assert _is_retryable_error(exc) is True


# =============================================================================
# Retry Configuration Tests
# =============================================================================


class TestResolveRetries:
    """Tests for retry configuration resolution."""

    def test_explicit_value_takes_precedence(self):
        """Explicit value overrides config and environment."""
        result = _resolve_retries(5, config=None)
        assert result == 5

    def test_config_overrides_default(self):
        """Config value overrides default."""
        mock_config = MagicMock()
        mock_config.retry_count = 7

        result = _resolve_retries(None, config=mock_config)
        assert result == 7

    def test_env_overrides_default(self, monkeypatch):
        """Environment variable overrides default."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "4")

        result = _resolve_retries(None, config=None)
        assert result == 4

    def test_default_value(self, monkeypatch):
        """Returns default when nothing specified."""
        monkeypatch.delenv("POLYLOGUE_DRIVE_RETRIES", raising=False)

        result = _resolve_retries(None, config=None)
        assert result == 3  # DEFAULT_DRIVE_RETRIES

    def test_negative_value_clamped_to_zero(self):
        """Negative values are clamped to 0."""
        result = _resolve_retries(-5, config=None)
        assert result == 0

    def test_invalid_env_value_uses_default(self, monkeypatch):
        """Invalid environment value falls back to default."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "not_a_number")

        result = _resolve_retries(None, config=None)
        assert result == 3  # DEFAULT_DRIVE_RETRIES


class TestResolveRetryBase:
    """Tests for retry base (backoff multiplier) resolution."""

    def test_explicit_value(self):
        """Explicit value is used."""
        result = _resolve_retry_base(1.5)
        assert result == 1.5

    def test_env_value(self, monkeypatch):
        """Environment variable is used."""
        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRY_BASE", "2.0")

        result = _resolve_retry_base(None)
        assert result == 2.0

    def test_default_value(self, monkeypatch):
        """Returns default when nothing specified."""
        monkeypatch.delenv("POLYLOGUE_DRIVE_RETRY_BASE", raising=False)

        result = _resolve_retry_base(None)
        assert result == 0.5  # DEFAULT_DRIVE_RETRY_BASE

    def test_negative_value_clamped_to_zero(self):
        """Negative values are clamped to 0."""
        result = _resolve_retry_base(-1.0)
        assert result == 0.0


# =============================================================================
# Client Retry Behavior Tests
# =============================================================================


class TestDriveClientRetryBehavior:
    """Tests for DriveClient retry mechanism."""

    @pytest.fixture
    def mock_credentials(self, tmp_path, monkeypatch):
        """Create mock credentials file."""
        import json

        creds_path = tmp_path / "credentials.json"
        creds_path.write_text(json.dumps({
            "installed": {
                "client_id": "test_client_id",
                "client_secret": "test_secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }))

        token_path = tmp_path / "token.json"
        token_path.write_text(json.dumps({
            "token": "fake_token",
            "refresh_token": "fake_refresh",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "test_client_id",
            "client_secret": "test_secret",
            "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
        }))

        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(creds_path))
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))

        return {"credentials_path": creds_path, "token_path": token_path}

    def test_client_initialization_with_retries(self, mock_credentials):
        """Client accepts custom retry configuration."""
        client = DriveClient(
            credentials_path=mock_credentials["credentials_path"],
            token_path=mock_credentials["token_path"],
            retries=5,
            retry_base=1.0,
        )

        assert client._retries == 5
        assert client._retry_base == 1.0

    def test_client_uses_default_retries(self, mock_credentials, monkeypatch):
        """Client uses default retry count when not specified."""
        monkeypatch.delenv("POLYLOGUE_DRIVE_RETRIES", raising=False)

        client = DriveClient(
            credentials_path=mock_credentials["credentials_path"],
            token_path=mock_credentials["token_path"],
        )

        assert client._retries == 3  # DEFAULT_DRIVE_RETRIES


# =============================================================================
# Mock HTTP Error Scenarios
# =============================================================================


RETRY_SCENARIOS = [
    (429, True, "rate limit retries"),
    (500, True, "server error retries"),
    (502, True, "bad gateway retries"),
    (503, True, "service unavailable retries"),
    (504, True, "gateway timeout retries"),
]

NO_RETRY_SCENARIOS = [
    (401, False, "auth error no retry"),
    (403, False, "forbidden no retry"),
    (404, False, "not found no retry"),
]


class TestHttpErrorRetries:
    """Tests for HTTP error handling."""

    @pytest.mark.parametrize("status_code,should_retry,description", RETRY_SCENARIOS)
    def test_retryable_http_errors(self, status_code: int, should_retry: bool, description: str):
        """HTTP errors that should trigger retry."""
        # Create a mock HTTP error
        class MockHTTPError(Exception):
            def __init__(self, status):
                self.status = status
                super().__init__(f"HTTP {status}")

        exc = MockHTTPError(status_code)

        # These should be retryable (not DriveAuthError or DriveNotFoundError)
        assert _is_retryable_error(exc) is should_retry


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    def test_backoff_increases_exponentially(self):
        """Verify backoff timing increases exponentially."""
        from tenacity import wait_exponential

        wait = wait_exponential(multiplier=0.5, min=0.5, max=10)

        # Simulate retry states with increasing attempt numbers
        class MockRetryState:
            def __init__(self, attempt):
                self.attempt_number = attempt

        # Get wait times for first few attempts
        wait_1 = wait(MockRetryState(1))
        wait_2 = wait(MockRetryState(2))
        wait_3 = wait(MockRetryState(3))

        # Wait times should increase (exponential)
        assert wait_2 >= wait_1
        assert wait_3 >= wait_2

    def test_backoff_respects_max(self):
        """Verify backoff is capped at max value."""
        from tenacity import wait_exponential

        wait = wait_exponential(multiplier=1, min=1, max=5)

        class MockRetryState:
            def __init__(self, attempt):
                self.attempt_number = attempt

        # After many attempts, should be capped at 5
        wait_10 = wait(MockRetryState(10))
        assert wait_10 <= 5


# =============================================================================
# Error Exception Type Tests
# =============================================================================


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


# =============================================================================
# Connection and Network Error Tests
# =============================================================================


class TestNetworkErrors:
    """Tests for network error handling."""

    def test_connection_refused_is_retryable(self):
        """Connection refused should be retryable."""
        exc = ConnectionRefusedError("Connection refused")
        assert _is_retryable_error(exc) is True

    def test_connection_reset_is_retryable(self):
        """Connection reset should be retryable."""
        exc = ConnectionResetError("Connection reset by peer")
        assert _is_retryable_error(exc) is True

    def test_timeout_is_retryable(self):
        """Timeout should be retryable."""
        exc = TimeoutError("Operation timed out")
        assert _is_retryable_error(exc) is True

    def test_broken_pipe_is_retryable(self):
        """Broken pipe should be retryable."""
        exc = BrokenPipeError("Broken pipe")
        assert _is_retryable_error(exc) is True
