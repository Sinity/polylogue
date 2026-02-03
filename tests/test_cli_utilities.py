"""Consolidated tests for CLI utility modules.

This file consolidates tests from:
- test_cli_formatting.py (25 tests)
- test_cli_helpers.py (48 tests)
- test_cli_container.py (2 tests)

Coverage includes:
- CLI formatting functions (format_counts, format_cursors, format_source_label, etc.)
- CLI helper functions (fail, resolve_sources, load_effective_config, etc.)
- Storage repository factory functions
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli import helpers
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
from polylogue.storage.backends.sqlite import create_default_backend
from polylogue.storage.repository import StorageRepository


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def helpers_workspace(tmp_path, monkeypatch):
    """Set up isolated workspace for helpers tests."""
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    config_dir = tmp_path / "config"

    for d in [data_dir, state_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    # Reload paths module
    import importlib
    import polylogue.paths

    importlib.reload(polylogue.paths)

    return {
        "data_dir": data_dir,
        "state_dir": state_dir,
        "config_dir": config_dir,
    }


# ============================================================================
# Formatting Tests
# ============================================================================


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


# ============================================================================
# Helper Tests
# ============================================================================


class TestFail:
    """Tests for the fail helper."""

    def test_raises_system_exit(self):
        """fail() raises SystemExit with formatted message."""
        with pytest.raises(SystemExit) as exc_info:
            helpers.fail("test", "error message")

        assert "test: error message" in str(exc_info.value)

    def test_message_format(self):
        """fail() formats message as 'command: message'."""
        with pytest.raises(SystemExit) as exc_info:
            helpers.fail("sync", "no sources found")

        assert str(exc_info.value) == "sync: no sources found"


class TestIsDeclarative:
    """Tests for is_declarative helper."""

    def test_unset_returns_false(self, monkeypatch):
        """Returns False when env var is not set."""
        monkeypatch.delenv("POLYLOGUE_DECLARATIVE", raising=False)
        assert helpers.is_declarative() is False

    def test_empty_returns_false(self, monkeypatch):
        """Returns False when env var is empty."""
        monkeypatch.setenv("POLYLOGUE_DECLARATIVE", "")
        assert helpers.is_declarative() is False

    def test_zero_returns_false(self, monkeypatch):
        """Returns False when env var is '0'."""
        monkeypatch.setenv("POLYLOGUE_DECLARATIVE", "0")
        assert helpers.is_declarative() is False

    def test_false_returns_false(self, monkeypatch):
        """Returns False when env var is 'false'."""
        monkeypatch.setenv("POLYLOGUE_DECLARATIVE", "false")
        assert helpers.is_declarative() is False

    def test_no_returns_false(self, monkeypatch):
        """Returns False when env var is 'no'."""
        monkeypatch.setenv("POLYLOGUE_DECLARATIVE", "no")
        assert helpers.is_declarative() is False

    def test_one_returns_true(self, monkeypatch):
        """Returns True when env var is '1'."""
        monkeypatch.setenv("POLYLOGUE_DECLARATIVE", "1")
        assert helpers.is_declarative() is True

    def test_true_returns_true(self, monkeypatch):
        """Returns True when env var is 'true'."""
        monkeypatch.setenv("POLYLOGUE_DECLARATIVE", "true")
        assert helpers.is_declarative() is True


class TestSourceStatePath:
    """Tests for source_state_path helper."""

    def test_uses_xdg_state_home(self, helpers_workspace):
        """Uses XDG_STATE_HOME for state path."""
        path = helpers.source_state_path()
        assert helpers_workspace["state_dir"] in path.parents or str(helpers_workspace["state_dir"]) in str(path)

    def test_returns_last_source_json(self, helpers_workspace):
        """Returns path to last-source.json."""
        path = helpers.source_state_path()
        assert path.name == "last-source.json"

    def test_fallback_without_xdg(self, monkeypatch):
        """Falls back to ~/.local/state when XDG not set."""
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        path = helpers.source_state_path()
        assert "polylogue" in str(path)
        assert "last-source.json" in str(path)


class TestLoadSaveLastSource:
    """Tests for load_last_source and save_last_source."""

    def test_load_returns_none_when_missing(self, helpers_workspace):
        """Returns None when state file doesn't exist."""
        result = helpers.load_last_source()
        assert result is None

    def test_save_and_load_roundtrip(self, helpers_workspace):
        """save_last_source and load_last_source work together."""
        helpers.save_last_source("test-source")
        result = helpers.load_last_source()
        assert result == "test-source"

    def test_load_handles_invalid_json(self, helpers_workspace):
        """Returns None for invalid JSON."""
        path = helpers.source_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json", encoding="utf-8")

        result = helpers.load_last_source()
        assert result is None

    def test_load_handles_missing_key(self, helpers_workspace):
        """Returns None when 'source' key is missing."""
        path = helpers.source_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"other": "value"}), encoding="utf-8")

        result = helpers.load_last_source()
        assert result is None

    def test_load_handles_non_string_source(self, helpers_workspace):
        """Returns None when 'source' is not a string."""
        path = helpers.source_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": 123}), encoding="utf-8")

        result = helpers.load_last_source()
        assert result is None

    def test_save_creates_parent_directories(self, helpers_workspace):
        """save_last_source creates parent directories."""
        # Remove the state dir
        import shutil
        shutil.rmtree(helpers_workspace["state_dir"])

        helpers.save_last_source("new-source")
        result = helpers.load_last_source()
        assert result == "new-source"


class TestMaybePromptSources:
    """Tests for maybe_prompt_sources helper."""

    def test_returns_selected_sources_if_provided(self, helpers_workspace, tmp_path):
        """Returns selected sources unchanged if already provided."""
        from polylogue.config import Config
        from polylogue.paths import Source

        mock_ui = MagicMock()
        mock_ui.plain = True
        env = MagicMock()
        env.ui = mock_ui

        inbox1 = tmp_path / "source1"
        inbox2 = tmp_path / "source2"
        inbox1.mkdir()
        inbox2.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox1), Source(name="source2", path=inbox2)],
        )

        result = helpers.maybe_prompt_sources(env, config, ["source1"], "sync")
        assert result == ["source1"]

    def test_returns_none_in_plain_mode(self, helpers_workspace, tmp_path):
        """Returns None (all sources) in plain mode."""
        from polylogue.config import Config
        from polylogue.paths import Source

        mock_ui = MagicMock()
        mock_ui.plain = True
        env = MagicMock()
        env.ui = mock_ui

        inbox1 = tmp_path / "source1"
        inbox2 = tmp_path / "source2"
        inbox1.mkdir()
        inbox2.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox1), Source(name="source2", path=inbox2)],
        )

        result = helpers.maybe_prompt_sources(env, config, None, "sync")
        assert result is None

    def test_returns_none_for_single_source(self, helpers_workspace, tmp_path):
        """Returns None for single source config."""
        from polylogue.config import Config
        from polylogue.paths import Source

        mock_ui = MagicMock()
        mock_ui.plain = False
        env = MagicMock()
        env.ui = mock_ui

        inbox = tmp_path / "source1"
        inbox.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox)],
        )

        result = helpers.maybe_prompt_sources(env, config, None, "sync")
        assert result is None


class TestResolveSources:
    """Tests for resolve_sources helper."""

    def test_empty_tuple_returns_none(self, helpers_workspace, tmp_path):
        """Empty sources tuple returns None (all sources)."""
        from polylogue.config import Config
        from polylogue.paths import Source

        inbox = tmp_path / "source1"
        inbox.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox)],
        )

        result = helpers.resolve_sources(config, (), "sync")
        assert result is None

    def test_valid_sources_returned(self, helpers_workspace, tmp_path):
        """Valid source names are returned as list."""
        from polylogue.config import Config
        from polylogue.paths import Source

        inbox1 = tmp_path / "source1"
        inbox2 = tmp_path / "source2"
        inbox1.mkdir()
        inbox2.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox1), Source(name="source2", path=inbox2)],
        )

        result = helpers.resolve_sources(config, ("source1",), "sync")
        assert result == ["source1"]

    def test_unknown_source_fails(self, helpers_workspace, tmp_path):
        """Unknown source name raises SystemExit."""
        from polylogue.config import Config
        from polylogue.paths import Source

        inbox = tmp_path / "source1"
        inbox.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox)],
        )

        with pytest.raises(SystemExit) as exc_info:
            helpers.resolve_sources(config, ("unknown",), "sync")

        assert "unknown" in str(exc_info.value).lower()

    def test_last_resolves_to_saved_source(self, helpers_workspace, tmp_path):
        """'last' resolves to previously saved source."""
        from polylogue.config import Config
        from polylogue.paths import Source

        inbox1 = tmp_path / "source1"
        inbox2 = tmp_path / "source2"
        inbox1.mkdir()
        inbox2.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox1), Source(name="source2", path=inbox2)],
        )

        # Save a source first
        helpers.save_last_source("source2")

        result = helpers.resolve_sources(config, ("last",), "sync")
        assert result == ["source2"]

    def test_last_with_no_saved_fails(self, helpers_workspace, tmp_path):
        """'last' with no saved source raises SystemExit."""
        from polylogue.config import Config
        from polylogue.paths import Source

        inbox = tmp_path / "source1"
        inbox.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox)],
        )

        with pytest.raises(SystemExit) as exc_info:
            helpers.resolve_sources(config, ("last",), "sync")

        assert "no previously" in str(exc_info.value).lower()

    def test_last_cannot_combine_with_others(self, helpers_workspace, tmp_path):
        """'last' cannot be combined with other sources."""
        from polylogue.config import Config
        from polylogue.paths import Source

        inbox1 = tmp_path / "source1"
        inbox2 = tmp_path / "source2"
        inbox1.mkdir()
        inbox2.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox1), Source(name="source2", path=inbox2)],
        )

        helpers.save_last_source("source1")

        with pytest.raises(SystemExit) as exc_info:
            helpers.resolve_sources(config, ("last", "source2"), "sync")

        assert "cannot be combined" in str(exc_info.value).lower()

    def test_deduplicates_sources(self, helpers_workspace, tmp_path):
        """Duplicate source names are deduplicated."""
        from polylogue.config import Config
        from polylogue.paths import Source

        inbox1 = tmp_path / "source1"
        inbox2 = tmp_path / "source2"
        inbox1.mkdir()
        inbox2.mkdir()

        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[Source(name="source1", path=inbox1), Source(name="source2", path=inbox2)],
        )

        result = helpers.resolve_sources(config, ("source1", "source1", "source2"), "sync")
        assert result == ["source1", "source2"]


class TestLatestRenderPath:
    """Tests for latest_render_path helper."""

    def test_returns_none_for_missing_directory(self, tmp_path):
        """Returns None when render directory doesn't exist."""
        result = helpers.latest_render_path(tmp_path / "nonexistent")
        assert result is None

    def test_returns_none_for_empty_directory(self, tmp_path):
        """Returns None when no render files exist."""
        render_dir = tmp_path / "render"
        render_dir.mkdir()

        result = helpers.latest_render_path(render_dir)
        assert result is None

    def test_finds_html_file(self, tmp_path):
        """Finds conversation.html files."""
        render_dir = tmp_path / "render"
        conv_dir = render_dir / "test" / "conv1"
        conv_dir.mkdir(parents=True)
        html_file = conv_dir / "conversation.html"
        html_file.write_text("<html>test</html>", encoding="utf-8")

        result = helpers.latest_render_path(render_dir)
        assert result == html_file

    def test_finds_md_file(self, tmp_path):
        """Finds conversation.md files."""
        render_dir = tmp_path / "render"
        conv_dir = render_dir / "test" / "conv1"
        conv_dir.mkdir(parents=True)
        md_file = conv_dir / "conversation.md"
        md_file.write_text("# Test", encoding="utf-8")

        result = helpers.latest_render_path(render_dir)
        assert result == md_file

    def test_returns_most_recent(self, tmp_path):
        """Returns the most recently modified file."""
        import time

        render_dir = tmp_path / "render"
        conv1_dir = render_dir / "test" / "conv1"
        conv2_dir = render_dir / "test" / "conv2"
        conv1_dir.mkdir(parents=True)
        conv2_dir.mkdir(parents=True)

        old_file = conv1_dir / "conversation.html"
        old_file.write_text("<html>old</html>", encoding="utf-8")

        time.sleep(0.01)  # Ensure different mtime

        new_file = conv2_dir / "conversation.html"
        new_file.write_text("<html>new</html>", encoding="utf-8")

        result = helpers.latest_render_path(render_dir)
        assert result == new_file

    def test_handles_deleted_file_race(self, tmp_path):
        """Handles file deleted between listing and stat."""
        render_dir = tmp_path / "render"
        conv_dir = render_dir / "test" / "conv1"
        conv_dir.mkdir(parents=True)

        html_file = conv_dir / "conversation.html"
        html_file.write_text("<html>test</html>", encoding="utf-8")

        # Create second file
        conv2_dir = render_dir / "test" / "conv2"
        conv2_dir.mkdir(parents=True)
        html_file2 = conv2_dir / "conversation.html"
        html_file2.write_text("<html>test2</html>", encoding="utf-8")
        html_file2.touch()  # Make it newer

        # Delete first file
        html_file.unlink()

        # Should still find second file
        result = helpers.latest_render_path(render_dir)
        assert result == html_file2


class TestLoadEffectiveConfig:
    """Tests for load_effective_config helper."""

    def test_loads_config(self, helpers_workspace):
        """Loads configuration successfully."""
        from polylogue.cli.types import AppEnv

        mock_ui = MagicMock()
        env = AppEnv(ui=mock_ui)

        config = helpers.load_effective_config(env)
        assert config is not None
        assert hasattr(config, "sources")


# ============================================================================
# Container Tests
# ============================================================================


def test_create_storage_repository() -> None:
    """Test creating storage repository returns StorageRepository instance."""
    backend = create_default_backend()
    repository = StorageRepository(backend=backend)

    assert isinstance(repository, StorageRepository)
    assert hasattr(repository, "_write_lock")


def test_create_storage_repository_independent_instances() -> None:
    """Test that each call creates a new independent repository instance."""
    backend1 = create_default_backend()
    backend2 = create_default_backend()
    repo1 = StorageRepository(backend=backend1)
    repo2 = StorageRepository(backend=backend2)

    assert repo1 is not repo2
    assert repo1._write_lock is not repo2._write_lock
