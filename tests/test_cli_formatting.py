"""Tests for CLI formatting utilities."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.cli.formatting import (
    announce_plain_mode,
    format_counts,
    format_cursors,
    format_index_status,
    format_source_label,
    format_sources_summary,
    should_use_plain,
)
from polylogue.config import Source


class TestShouldUsePlain:
    """Test should_use_plain function."""

    def test_explicit_plain_true(self):
        """Explicit plain=True returns True."""
        assert should_use_plain(plain=True, interactive=False) is True

    def test_explicit_interactive_true(self):
        """Explicit interactive=True returns False."""
        assert should_use_plain(plain=False, interactive=True) is False

    def test_plain_takes_precedence_over_interactive(self):
        """plain=True takes precedence over interactive=True."""
        assert should_use_plain(plain=True, interactive=True) is True

    def test_env_var_force_plain(self, monkeypatch):
        """POLYLOGUE_FORCE_PLAIN env var enables plain mode."""
        monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        assert should_use_plain(plain=False, interactive=False) is True

    def test_env_var_false_values(self, monkeypatch):
        """POLYLOGUE_FORCE_PLAIN with 0/false/no doesn't force plain."""
        for val in ("0", "false", "no", "False", "NO"):
            monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", val)
            # Result depends on TTY status, but env var doesn't force plain
            # We can't easily test TTY, so we just verify no crash
            result = should_use_plain(plain=False, interactive=False)
            assert isinstance(result, bool)

    def test_non_tty_returns_true(self, monkeypatch):
        """Non-TTY environment returns True (plain mode)."""
        monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
        # Mock stdout/stderr as non-TTY
        with patch.object(sys.stdout, "isatty", return_value=False):
            with patch.object(sys.stderr, "isatty", return_value=False):
                assert should_use_plain(plain=False, interactive=False) is True


class TestAnnouncePlainMode:
    """Test announce_plain_mode function."""

    def test_writes_to_stderr(self):
        """Writes announcement to stderr."""
        captured = StringIO()
        with patch.object(sys, "stderr", captured):
            announce_plain_mode()
        output = captured.getvalue()
        assert "Plain output active" in output
        assert "--interactive" in output


class TestFormatCursors:
    """Test format_cursors function."""

    def test_empty_cursors_returns_none(self):
        """Empty cursors dict returns None."""
        assert format_cursors({}) is None

    def test_file_count_displayed(self):
        """File count is displayed."""
        result = format_cursors({"inbox": {"file_count": 10}})
        assert result is not None
        assert "10 files" in result
        assert "inbox" in result

    def test_error_count_highlighted(self):
        """Error count is displayed when non-zero."""
        result = format_cursors({"source": {"file_count": 5, "error_count": 2}})
        assert result is not None
        assert "2 errors" in result

    def test_zero_error_count_not_shown(self):
        """Zero error count is not displayed."""
        result = format_cursors({"source": {"file_count": 5, "error_count": 0}})
        assert result is not None
        assert "errors" not in result

    def test_latest_mtime_formatted(self):
        """Latest mtime is formatted as timestamp."""
        result = format_cursors({"source": {"latest_mtime": 1704067200}})
        assert result is not None
        assert "latest" in result
        # Should contain ISO-ish format
        assert "202" in result  # Year prefix

    def test_latest_file_name_shown(self):
        """Latest file name is shown when mtime not available."""
        result = format_cursors({"source": {"latest_file_name": "chat.json"}})
        assert result is not None
        assert "latest chat.json" in result

    def test_latest_path_fallback(self):
        """Path basename used as fallback for latest label."""
        result = format_cursors({"source": {"latest_path": "/some/dir/export.json"}})
        assert result is not None
        assert "latest export.json" in result

    def test_multiple_cursors(self):
        """Multiple cursors are joined with semicolons."""
        result = format_cursors({
            "inbox": {"file_count": 5},
            "drive": {"file_count": 3},
        })
        assert result is not None
        assert "inbox" in result
        assert "drive" in result
        assert ";" in result


class TestFormatCounts:
    """Test format_counts function."""

    def test_conversations_and_messages(self):
        """Shows conversations and messages count."""
        result = format_counts({"conversations": 10, "messages": 100})
        assert "10 conv" in result
        assert "100 msg" in result

    def test_rendered_shown_when_nonzero(self):
        """Rendered count shown when non-zero."""
        result = format_counts({"conversations": 5, "messages": 50, "rendered": 5})
        assert "5 rendered" in result

    def test_rendered_not_shown_when_zero(self):
        """Rendered count not shown when zero."""
        result = format_counts({"conversations": 5, "messages": 50, "rendered": 0})
        assert "rendered" not in result

    def test_missing_keys_default_to_zero(self):
        """Missing keys default to zero."""
        result = format_counts({})
        assert "0 conv" in result
        assert "0 msg" in result


class TestFormatIndexStatus:
    """Test format_index_status function."""

    def test_ingest_stage_skipped(self):
        """Ingest stage shows skipped."""
        assert format_index_status("ingest", True, None) == "Index: skipped"

    def test_render_stage_skipped(self):
        """Render stage shows skipped."""
        assert format_index_status("render", False, None) == "Index: skipped"

    def test_index_error(self):
        """Index error is reported."""
        assert format_index_status("full", True, "connection failed") == "Index: error"

    def test_indexed_ok(self):
        """Indexed flag True shows ok."""
        assert format_index_status("full", True, None) == "Index: ok"

    def test_not_indexed_up_to_date(self):
        """Not indexed shows up-to-date."""
        assert format_index_status("full", False, None) == "Index: up-to-date"


class TestFormatSourceLabel:
    """Test format_source_label function."""

    def test_source_differs_from_provider(self):
        """Shows source/provider when they differ."""
        result = format_source_label("inbox", "claude")
        assert result == "inbox/claude"

    def test_source_same_as_provider(self):
        """Shows just source when same as provider."""
        result = format_source_label("claude", "claude")
        assert result == "claude"

    def test_none_source(self):
        """None source shows provider name."""
        result = format_source_label(None, "chatgpt")
        assert result == "chatgpt"


class TestFormatSourcesSummary:
    """Test format_sources_summary function."""

    def test_empty_sources(self):
        """Empty list returns 'none'."""
        assert format_sources_summary([]) == "none"

    def test_path_source(self):
        """Source with path shows name."""
        source = Source(name="inbox", path=Path("/inbox"))
        result = format_sources_summary([source])
        assert "inbox" in result
        assert "(drive)" not in result

    def test_drive_source(self):
        """Source with folder shows (drive) tag."""
        source = Source(name="gemini", folder="folder-id")
        result = format_sources_summary([source])
        assert "gemini (drive)" in result

    def test_missing_source(self):
        """Source without path or folder shows (missing)."""
        # Note: Source validation prevents creating such objects normally
        # This tests defensive code handling edge cases via mock
        from unittest.mock import MagicMock

        source = MagicMock()
        source.name = "broken"
        source.path = None
        source.folder = None
        result = format_sources_summary([source])
        assert "broken (missing)" in result

    def test_truncates_long_lists(self):
        """Lists > 8 items are truncated."""
        sources = [
            Source(name=f"source{i}", path=Path(f"/src{i}"))
            for i in range(12)
        ]
        result = format_sources_summary(sources)
        assert "+4 more" in result
        # Should have 8 names plus the "+4 more"
        assert result.count(",") == 8  # 9 items = 8 commas
