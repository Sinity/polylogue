"""Tests for polylogue.cli.helpers module.

Coverage targets:
- fail: raises SystemExit with message
- is_declarative: checks POLYLOGUE_DECLARATIVE env var
- source_state_path: returns correct XDG path
- load_last_source/save_last_source: persistence of source selection
- maybe_prompt_sources: interactive source selection
- load_effective_config: config loading
- resolve_sources: source name validation
- print_summary: output formatting
- latest_render_path: finds most recent render file
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli import helpers


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
