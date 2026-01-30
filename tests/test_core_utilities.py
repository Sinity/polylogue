"""Tests for core utility modules: env, dates, version, export.

These modules had 0% coverage before this file was created.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Tests for polylogue/core/env.py
# =============================================================================


class TestGetEnv:
    """Tests for get_env() function with POLYLOGUE_* precedence."""

    def test_get_env_returns_prefixed_first(self, monkeypatch):
        """POLYLOGUE_* variable takes precedence over unprefixed."""
        from polylogue.core.env import get_env

        monkeypatch.setenv("QDRANT_URL", "http://global:6333")
        monkeypatch.setenv("POLYLOGUE_QDRANT_URL", "http://local:6333")

        result = get_env("QDRANT_URL")
        assert result == "http://local:6333"

    def test_get_env_falls_back_to_unprefixed(self, monkeypatch):
        """Falls back to unprefixed when prefixed not set."""
        from polylogue.core.env import get_env

        monkeypatch.setenv("QDRANT_URL", "http://global:6333")
        monkeypatch.delenv("POLYLOGUE_QDRANT_URL", raising=False)

        result = get_env("QDRANT_URL")
        assert result == "http://global:6333"

    def test_get_env_returns_default(self, monkeypatch):
        """Returns default when neither variable is set."""
        from polylogue.core.env import get_env

        monkeypatch.delenv("MISSING_VAR", raising=False)
        monkeypatch.delenv("POLYLOGUE_MISSING_VAR", raising=False)

        result = get_env("MISSING_VAR", "fallback")
        assert result == "fallback"

    def test_get_env_returns_none_without_default(self, monkeypatch):
        """Returns None when neither variable is set and no default given."""
        from polylogue.core.env import get_env

        monkeypatch.delenv("TOTALLY_MISSING", raising=False)
        monkeypatch.delenv("POLYLOGUE_TOTALLY_MISSING", raising=False)

        result = get_env("TOTALLY_MISSING")
        assert result is None

    def test_get_env_empty_string_is_falsy(self, monkeypatch):
        """Empty string values are treated as falsy (falls through)."""
        from polylogue.core.env import get_env

        monkeypatch.setenv("POLYLOGUE_EMPTY", "")
        monkeypatch.setenv("EMPTY", "real_value")

        result = get_env("EMPTY")
        assert result == "real_value"


class TestGetEnvMulti:
    """Tests for get_env_multi() with multiple fallback names."""

    def test_get_env_multi_prefixed_first(self, monkeypatch):
        """Prefixed variable takes precedence over all."""
        from polylogue.core.env import get_env_multi

        monkeypatch.setenv("POLYLOGUE_GOOGLE_API_KEY", "polylogue_key")
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini_key")

        result = get_env_multi("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert result == "polylogue_key"

    def test_get_env_multi_unprefixed_primary(self, monkeypatch):
        """Unprefixed primary takes precedence over alternatives."""
        from polylogue.core.env import get_env_multi

        monkeypatch.delenv("POLYLOGUE_GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini_key")

        result = get_env_multi("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert result == "google_key"

    def test_get_env_multi_falls_to_alternative(self, monkeypatch):
        """Falls back to alternative when primary not set."""
        from polylogue.core.env import get_env_multi

        monkeypatch.delenv("POLYLOGUE_GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini_key")

        result = get_env_multi("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert result == "gemini_key"

    def test_get_env_multi_returns_default(self, monkeypatch):
        """Returns default when no variables are set."""
        from polylogue.core.env import get_env_multi

        monkeypatch.delenv("POLYLOGUE_MISSING", raising=False)
        monkeypatch.delenv("MISSING", raising=False)
        monkeypatch.delenv("ALSO_MISSING", raising=False)

        result = get_env_multi("MISSING", "ALSO_MISSING", default="default_val")
        assert result == "default_val"

    def test_get_env_multi_no_alternatives(self, monkeypatch):
        """Works with no alternative names provided."""
        from polylogue.core.env import get_env_multi

        monkeypatch.setenv("SINGLE_KEY", "single_value")
        monkeypatch.delenv("POLYLOGUE_SINGLE_KEY", raising=False)

        result = get_env_multi("SINGLE_KEY")
        assert result == "single_value"


# =============================================================================
# Tests for polylogue/core/dates.py
# =============================================================================


class TestParseDate:
    """Tests for parse_date() with natural language support."""

    def test_parse_date_iso_format(self):
        """Parses ISO format dates correctly."""
        from polylogue.core.dates import parse_date

        result = parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo is not None  # Should be timezone-aware

    def test_parse_date_iso_with_time(self):
        """Parses ISO format with time correctly."""
        from polylogue.core.dates import parse_date

        result = parse_date("2024-01-15T10:30:00")
        assert result is not None
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_date_natural_yesterday(self):
        """Parses 'yesterday' naturally."""
        from polylogue.core.dates import parse_date

        result = parse_date("yesterday")
        assert result is not None
        # Should be before now
        assert result < datetime.now(timezone.utc)

    def test_parse_date_natural_relative(self):
        """Parses relative dates like '2 days ago'."""
        from polylogue.core.dates import parse_date

        result = parse_date("2 days ago")
        assert result is not None
        # Should be in the past
        now = datetime.now(timezone.utc)
        assert result < now

    def test_parse_date_invalid_returns_none(self):
        """Returns None for unparseable input."""
        from polylogue.core.dates import parse_date

        result = parse_date("not a date at all xyz123")
        assert result is None

    def test_parse_date_returns_utc_aware(self):
        """All returned dates are UTC-aware."""
        from polylogue.core.dates import parse_date

        result = parse_date("2024-06-15")
        assert result is not None
        assert result.tzinfo is not None


class TestFormatDateIso:
    """Tests for format_date_iso() function."""

    def test_format_date_iso_basic(self):
        """Formats datetime as ISO string."""
        from polylogue.core.dates import format_date_iso

        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = format_date_iso(dt)
        assert result == "2024-01-15 10:30:45"

    def test_format_date_iso_midnight(self):
        """Formats midnight correctly."""
        from polylogue.core.dates import format_date_iso

        dt = datetime(2024, 12, 25, 0, 0, 0)
        result = format_date_iso(dt)
        assert result == "2024-12-25 00:00:00"

    def test_format_date_iso_with_timezone(self):
        """Works with timezone-aware datetime."""
        from polylogue.core.dates import format_date_iso

        dt = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = format_date_iso(dt)
        assert result == "2024-06-15 14:30:00"


# =============================================================================
# Tests for polylogue/version.py
# =============================================================================


class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_version_info_str_without_commit(self):
        """String representation without commit hash."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.2.3")
        assert str(info) == "1.2.3"
        assert info.full == "1.2.3"
        assert info.short == "1.2.3"

    def test_version_info_str_with_commit(self):
        """String representation with commit hash."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.2.3", commit="abc123def456")
        assert str(info) == "1.2.3+abc123de"  # First 8 chars
        assert info.full == "1.2.3+abc123de"
        assert info.short == "1.2.3"

    def test_version_info_str_with_dirty(self):
        """String representation with dirty flag."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="1.2.3", commit="abc123def456", dirty=True)
        assert str(info) == "1.2.3+abc123de-dirty"
        assert info.full == "1.2.3+abc123de-dirty"
        assert info.short == "1.2.3"

    def test_version_info_short_property(self):
        """Short property always returns just version."""
        from polylogue.version import VersionInfo

        info = VersionInfo(version="0.9.0", commit="deadbeef", dirty=True)
        assert info.short == "0.9.0"


class TestVersionResolution:
    """Tests for version resolution mechanism."""

    def test_version_info_available(self):
        """VERSION_INFO is available at module level."""
        from polylogue.version import VERSION_INFO

        assert VERSION_INFO is not None
        assert hasattr(VERSION_INFO, "version")
        assert hasattr(VERSION_INFO, "full")
        assert hasattr(VERSION_INFO, "short")

    def test_polylogue_version_available(self):
        """POLYLOGUE_VERSION constant is available."""
        from polylogue.version import POLYLOGUE_VERSION

        assert POLYLOGUE_VERSION is not None
        assert isinstance(POLYLOGUE_VERSION, str)
        assert len(POLYLOGUE_VERSION) > 0


# =============================================================================
# Tests for polylogue/export.py
# =============================================================================


class TestExportJsonl:
    """Tests for export_jsonl() function."""

    def test_export_jsonl_creates_output_file(self, workspace_env):
        """Creates output file in correct location."""
        from polylogue.export import export_jsonl

        archive_root = workspace_env["archive_root"]

        result_path = export_jsonl(archive_root=archive_root)

        assert result_path.exists()
        assert result_path.suffix == ".jsonl"

    def test_export_jsonl_custom_output_path(self, workspace_env):
        """Uses custom output path when provided."""
        from polylogue.export import export_jsonl

        archive_root = workspace_env["archive_root"]
        custom_output = workspace_env["data_root"] / "custom_export.jsonl"

        result_path = export_jsonl(archive_root=archive_root, output_path=custom_output)

        assert result_path == custom_output
        assert custom_output.exists()

    def test_export_jsonl_empty_database(self, workspace_env):
        """Handles empty database gracefully."""
        from polylogue.export import export_jsonl

        archive_root = workspace_env["archive_root"]

        result_path = export_jsonl(archive_root=archive_root)

        # File should exist but be empty (no conversations)
        assert result_path.exists()
        content = result_path.read_text()
        assert content == ""  # No lines for empty database

    def test_export_jsonl_with_conversation(self, workspace_env, db_path):
        """Exports conversation with messages and attachments."""
        from tests.helpers import ConversationBuilder
        from polylogue.export import export_jsonl

        # Create test conversation
        (ConversationBuilder(db_path, "test-conv")
         .title("Export Test")
         .provider("chatgpt")
         .add_message("m1", role="user", text="Hello")
         .add_message("m2", role="assistant", text="Hi there!")
         .save())

        archive_root = workspace_env["archive_root"]
        result_path = export_jsonl(archive_root=archive_root)

        assert result_path.exists()
        content = result_path.read_text().strip()
        lines = content.split("\n")
        assert len(lines) == 1  # One conversation

        # Parse the JSONL line
        export_data = json.loads(lines[0])
        assert "conversation" in export_data
        assert "messages" in export_data
        assert export_data["conversation"]["title"] == "Export Test"
        assert len(export_data["messages"]) == 2

    def test_export_jsonl_creates_parent_dirs(self, workspace_env):
        """Creates parent directories for output path."""
        from polylogue.export import export_jsonl

        archive_root = workspace_env["archive_root"]
        nested_output = workspace_env["data_root"] / "deeply" / "nested" / "export.jsonl"

        result_path = export_jsonl(archive_root=archive_root, output_path=nested_output)

        assert result_path.exists()
        assert result_path.parent.exists()
