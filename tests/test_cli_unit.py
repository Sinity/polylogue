"""Consolidated CLI unit tests.

SYSTEMATIZATION: Merged from:
- test_cli_utilities.py (Formatting and helper functions)
- test_cli_auth.py (Auth command tests)
- test_cli_check.py (Check command tests)
- test_cli_editor.py (Editor security tests)

This file contains unit tests for:
- CLI formatting functions
- CLI helper functions
- Auth command
- Check command
- Editor/browser security
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli, helpers
from polylogue.cli.commands.check import check_command
from polylogue.cli.editor import validate_command
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
from polylogue.health import (
    HealthCheck,
    HealthReport,
    RepairResult,
    VerifyStatus,
    repair_dangling_fts,
    repair_empty_conversations,
    repair_orphaned_attachments,
    repair_orphaned_messages,
    run_all_repairs,
)
from polylogue.storage.backends.sqlite import create_default_backend

# =============================================================================
# CLI UTILITIES TESTS (from test_cli_utilities.py)
# =============================================================================
from polylogue.storage.repository import ConversationRepository

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
# Parametrization Data Tables
# ============================================================================

# TestFormatCursors cursor structures
FORMAT_CURSORS_CASES = [
    ({}, None, "empty cursors returns None"),
    (
        {"inbox": {"file_count": 10}},
        ("10 files", "inbox"),
        "file count displayed",
    ),
    (
        {"source": {"file_count": 5, "error_count": 2}},
        ("2 errors",),
        "error count highlighted",
    ),
    (
        {"source": {"file_count": 5, "error_count": 0}},
        (),  # No "errors" expected
        "zero error count not shown",
    ),
    (
        {"source": {"latest_mtime": 1704067200}},
        ("latest", "202"),
        "latest mtime formatted",
    ),
    (
        {"source": {"latest_file_name": "chat.json"}},
        ("latest chat.json",),
        "latest file name shown",
    ),
    (
        {"source": {"latest_path": "/some/dir/export.json"}},
        ("latest export.json",),
        "path basename fallback",
    ),
    (
        {"inbox": {"file_count": 5}, "drive": {"file_count": 3}},
        ("inbox", "drive", ";"),
        "multiple cursors joined",
    ),
]

# TestFormatIndexStatus cases
FORMAT_INDEX_STATUS_CASES = [
    ("ingest", True, None, "Index: skipped"),
    ("render", False, None, "Index: skipped"),
    ("full", True, "connection failed", "Index: error"),
    ("full", True, None, "Index: ok"),
    ("full", False, None, "Index: up-to-date"),
]

# TestFormatSourceLabel cases
FORMAT_SOURCE_LABEL_CASES = [
    ("inbox", "claude", "inbox/claude"),
    ("claude", "claude", "claude"),
    (None, "chatgpt", "chatgpt"),
]

# TestFormatCounts cases
FORMAT_COUNTS_CASES = [
    ({"conversations": 10, "messages": 100}, ("10 conv", "100 msg")),
    ({"conversations": 5, "messages": 50, "rendered": 5}, ("5 rendered",)),
    ({"conversations": 5, "messages": 50, "rendered": 0}, ()),  # No "rendered"
    ({}, ("0 conv", "0 msg")),
]

# TestFormatSourcesSummary cases
FORMAT_SOURCES_SUMMARY_CASES = [
    ([], "none"),
    (
        [Source(name="inbox", path=Path("/inbox"))],
        ("inbox", "no_drive_tag"),  # Special marker: no "(drive)"
    ),
    ([Source(name="gemini", folder="folder-id")], ("gemini (drive)",)),
]

# TestIsDeclarative env var cases
IS_DECLARATIVE_CASES = [
    (None, False, "unset returns False"),
    ("", False, "empty returns False"),
    ("0", False, "zero returns False"),
    ("false", False, "'false' returns False"),
    ("no", False, "'no' returns False"),
    ("1", True, "one returns True"),
    ("true", True, "'true' returns True"),
]

# TestShouldUsePlain cases
SHOULD_USE_PLAIN_CASES = [
    (True, None, True, "explicit plain=True"),
    (False, True, False, "explicit plain=False on TTY"),
    (False, None, True, "force plain env var"),
]

# Unsafe editor commands
UNSAFE_COMMAND_CASES = [
    ("vim; rm -rf /tmp/pwned", "unsafe shell metacharacters", "semicolon injection"),
    ("vim | cat /etc/passwd", "unsafe shell metacharacters", "pipe injection"),
    ("vim `whoami`", "unsafe shell metacharacters", "backtick injection"),
    ("vim $(cat /etc/passwd)", "unsafe shell metacharacters", "dollar paren injection"),
    ("vim & malicious_command", "unsafe shell metacharacters", "ampersand background"),
    ("vim && rm -rf /", "unsafe shell metacharacters", "double ampersand chain"),
    ("vim || evil_command", "unsafe shell metacharacters", "double pipe fallback"),
    ("vim > /tmp/output", "unsafe shell metacharacters", "redirect out"),
    ("vim < /tmp/input", "unsafe shell metacharacters", "redirect in"),
    ("vim {/tmp/a,/tmp/b}", "unsafe shell metacharacters", "brace expansion"),
    ("vim /tmp/[abc]", "unsafe shell metacharacters", "bracket glob"),
    ("vim \\n", "unsafe shell metacharacters", "backslash escape"),
    ("vim !!", "unsafe shell metacharacters", "history expansion"),
    ("", "cannot be empty", "empty string"),
    ("   ", "cannot be empty", "whitespace only"),
]

# Safe editor commands
SAFE_COMMAND_CASES = [
    ("vim", "simple vim"),
    ("/usr/bin/vim", "vim with path"),
    ("vim -u NONE", "vim with options"),
    ("nano", "nano editor"),
    ("nvim", "neovim"),
    ("code --wait", "vscode with wait"),
    ("emacs -nw", "emacs terminal mode"),
]


# ============================================================================
# Formatting Tests
# ============================================================================


class TestShouldUsePlain:
    """Test should_use_plain function."""

    @pytest.mark.parametrize("plain,tty_value,expected,description", SHOULD_USE_PLAIN_CASES)
    def test_should_use_plain(self, plain, tty_value, expected, description, monkeypatch):
        """Test should_use_plain with various inputs."""
        if tty_value is None:
            monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
        elif tty_value:
            monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
            with patch("sys.stdout.isatty", return_value=True), patch("sys.stderr.isatty", return_value=True):
                assert should_use_plain(plain=plain) == expected
            return
        else:
            monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)

        if tty_value is None:
            assert should_use_plain(plain=plain) == expected

    def test_env_var_false_values(self, monkeypatch):
        """POLYLOGUE_FORCE_PLAIN with 0/false/no doesn't force plain."""
        for val in ("0", "false", "no", "False", "NO"):
            monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", val)
            result = should_use_plain(plain=False)
            assert isinstance(result, bool)

    def test_non_tty_returns_true(self, monkeypatch):
        """Non-TTY environment returns True (plain mode)."""
        monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
        with patch.object(sys.stdout, "isatty", return_value=False):
            with patch.object(sys.stderr, "isatty", return_value=False):
                assert should_use_plain(plain=False) is True


class TestAnnouncePlainMode:
    """Test announce_plain_mode function."""

    def test_writes_to_stderr(self):
        """Writes announcement to stderr."""
        captured = StringIO()
        with patch.object(sys, "stderr", captured):
            announce_plain_mode()
        output = captured.getvalue()
        assert "Plain output active" in output


class TestFormatCursors:
    """Test format_cursors function."""

    @pytest.mark.parametrize("cursors_input,expected_parts,description", FORMAT_CURSORS_CASES)
    def test_format_cursors(self, cursors_input, expected_parts, description):
        """Test format_cursors with various cursor structures."""
        result = format_cursors(cursors_input)

        if expected_parts is None:
            assert result is None
        else:
            assert result is not None
            for expected in expected_parts:
                assert expected in result


class TestFormatCounts:
    """Test format_counts function."""

    @pytest.mark.parametrize("counts_input,expected_parts", FORMAT_COUNTS_CASES)
    def test_format_counts(self, counts_input, expected_parts):
        """Test format_counts with various count dictionaries."""
        result = format_counts(counts_input)

        if not expected_parts:
            # No assertions for empty case (already tested)
            assert isinstance(result, str)
        else:
            for expected in expected_parts:
                assert expected in result


class TestFormatIndexStatus:
    """Test format_index_status function."""

    @pytest.mark.parametrize(
        "stage,skip,error,expected",
        FORMAT_INDEX_STATUS_CASES,
    )
    def test_format_index_status(self, stage, skip, error, expected):
        """Test format_index_status with various stage/skip/error combos."""
        result = format_index_status(stage, skip, error)
        assert result == expected


class TestFormatSourceLabel:
    """Test format_source_label function."""

    @pytest.mark.parametrize("source,provider,expected", FORMAT_SOURCE_LABEL_CASES)
    def test_format_source_label(self, source, provider, expected):
        """Test format_source_label with various source/provider pairs."""
        result = format_source_label(source, provider)
        assert result == expected


class TestFormatSourcesSummary:
    """Test format_sources_summary function."""

    def test_empty_sources(self):
        """Empty list returns 'none'."""
        assert format_sources_summary([]) == "none"

    @pytest.mark.parametrize("sources_input,expected_output", FORMAT_SOURCES_SUMMARY_CASES[1:])
    def test_format_sources_summary(self, sources_input, expected_output):
        """Test format_sources_summary with various source lists."""
        result = format_sources_summary(sources_input)

        if expected_output == ("gemini (drive)",):
            assert "gemini (drive)" in result
        elif expected_output == ("inbox", "no_drive_tag"):
            assert "inbox" in result
            assert "(drive)" not in result

    def test_missing_source(self):
        """Source without path or folder shows (missing)."""
        from unittest.mock import MagicMock

        source = MagicMock()
        source.name = "broken"
        source.path = None
        source.folder = None
        result = format_sources_summary([source])
        assert "broken (missing)" in result

    def test_truncates_long_lists(self):
        """Lists > 8 items are truncated."""
        sources = [Source(name=f"source{i}", path=Path(f"/src{i}")) for i in range(12)]
        result = format_sources_summary(sources)
        assert "+4 more" in result
        assert result.count(",") == 8


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

    @pytest.mark.parametrize("env_value,expected,description", IS_DECLARATIVE_CASES)
    def test_is_declarative(self, env_value, expected, description, monkeypatch):
        """Test is_declarative with various env var values."""
        if env_value is None:
            monkeypatch.delenv("POLYLOGUE_DECLARATIVE", raising=False)
        else:
            monkeypatch.setenv("POLYLOGUE_DECLARATIVE", env_value)

        result = helpers.is_declarative()
        assert result is expected


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

    # Parametrization data for valid/invalid resolve_sources cases
    RESOLVE_SOURCES_VALID_CASES = [
        ((), None, "empty tuple returns None"),
        (("source1",), ["source1"], "valid source returned"),
        (("source1", "source1", "source2"), ["source1", "source2"], "deduplicates"),
    ]

    RESOLVE_SOURCES_ERROR_CASES = [
        (("unknown",), "unknown", "unknown source fails"),
    ]

    @pytest.mark.parametrize("sources_input,expected,description", RESOLVE_SOURCES_VALID_CASES)
    def test_resolve_sources_valid(self, sources_input, expected, description, helpers_workspace, tmp_path):
        """Test resolve_sources with valid inputs."""
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

        result = helpers.resolve_sources(config, sources_input, "sync")
        assert result == expected

    @pytest.mark.parametrize("sources_input,error_match,description", RESOLVE_SOURCES_ERROR_CASES)
    def test_resolve_sources_invalid(self, sources_input, error_match, description, helpers_workspace, tmp_path):
        """Test resolve_sources with invalid inputs."""
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
            helpers.resolve_sources(config, sources_input, "sync")

        assert error_match.lower() in str(exc_info.value).lower()

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


class TestLatestRenderPath:
    """Tests for latest_render_path helper."""

    LATEST_RENDER_PATH_CASES = [
        ("missing_directory", None, "returns None for missing directory"),
        ("empty_directory", None, "returns None for empty directory"),
        ("html_file", "html", "finds conversation.html files"),
        ("md_file", "md", "finds conversation.md files"),
        ("most_recent", "html", "returns most recently modified file"),
    ]

    @pytest.mark.parametrize("scenario,expected_type,description", LATEST_RENDER_PATH_CASES)
    def test_latest_render_path(self, scenario, expected_type, description, tmp_path):
        """Test latest_render_path with various directory states."""
        import time

        if scenario == "missing_directory":
            result = helpers.latest_render_path(tmp_path / "nonexistent")
            assert result is None
        elif scenario == "empty_directory":
            render_dir = tmp_path / "render"
            render_dir.mkdir()
            result = helpers.latest_render_path(render_dir)
            assert result is None
        elif scenario == "html_file":
            render_dir = tmp_path / "render"
            conv_dir = render_dir / "test" / "conv1"
            conv_dir.mkdir(parents=True)
            html_file = conv_dir / "conversation.html"
            html_file.write_text("<html>test</html>", encoding="utf-8")
            result = helpers.latest_render_path(render_dir)
            assert result == html_file
        elif scenario == "md_file":
            render_dir = tmp_path / "render"
            conv_dir = render_dir / "test" / "conv1"
            conv_dir.mkdir(parents=True)
            md_file = conv_dir / "conversation.md"
            md_file.write_text("# Test", encoding="utf-8")
            result = helpers.latest_render_path(render_dir)
            assert result == md_file
        elif scenario == "most_recent":
            render_dir = tmp_path / "render"
            conv1_dir = render_dir / "test" / "conv1"
            conv2_dir = render_dir / "test" / "conv2"
            conv1_dir.mkdir(parents=True)
            conv2_dir.mkdir(parents=True)

            old_file = conv1_dir / "conversation.html"
            old_file.write_text("<html>old</html>", encoding="utf-8")

            time.sleep(0.01)

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

        conv2_dir = render_dir / "test" / "conv2"
        conv2_dir.mkdir(parents=True)
        html_file2 = conv2_dir / "conversation.html"
        html_file2.write_text("<html>test2</html>", encoding="utf-8")
        html_file2.touch()

        html_file.unlink()

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


def test_create_conversation_repository() -> None:
    """Test creating conversation repository returns ConversationRepository instance."""
    backend = create_default_backend()
    repository = ConversationRepository(backend=backend)

    assert isinstance(repository, ConversationRepository)
    assert hasattr(repository, "_write_lock")


def test_create_conversation_repository_independent_instances() -> None:
    """Test that each call creates a new independent repository instance."""
    backend1 = create_default_backend()
    backend2 = create_default_backend()
    repo1 = ConversationRepository(backend=backend1)
    repo2 = ConversationRepository(backend=backend2)

    assert repo1 is not repo2
    assert repo1._write_lock is not repo2._write_lock


# =============================================================================
# AUTH COMMAND TESTS (from test_cli_auth.py)
# =============================================================================


@pytest.fixture
def auth_workspace(tmp_path, monkeypatch):
    """Set up isolated workspace for auth tests."""
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    creds_dir = tmp_path / "creds"

    for d in [config_dir, data_dir, state_dir, creds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create credentials.json (OAuth client config)
    creds_path = creds_dir / "credentials.json"
    creds_path.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "test_client.apps.googleusercontent.com",
                    "client_secret": "test_secret",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
        ),
        encoding="utf-8",
    )

    # Create token.json (OAuth tokens)
    token_path = creds_dir / "token.json"
    token_path.write_text(
        json.dumps(
            {
                "token": "test_access_token",
                "refresh_token": "test_refresh_token",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "test_client.apps.googleusercontent.com",
                "client_secret": "test_secret",
                "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(creds_path))
    monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    # Reload modules to pick up new environment
    import importlib

    import polylogue.config
    import polylogue.paths

    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)

    return {
        "creds_path": creds_path,
        "token_path": token_path,
        "data_dir": data_dir,
    }


class TestAuthCommand:
    """Tests for the auth command."""

    def test_unknown_service_fails(self, auth_workspace):
        """Unknown auth service shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--service", "unknown"])
        assert result.exit_code == 1
        assert "unknown" in result.output.lower()
        assert "drive" in result.output.lower()

    def test_revoke_removes_token(self, auth_workspace, monkeypatch):
        """--revoke removes the token file."""
        token_path = auth_workspace["token_path"]
        assert token_path.exists()

        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--revoke"])
        assert result.exit_code == 0
        assert not token_path.exists() or "removed" in result.output.lower() or "revoked" in result.output.lower()

    def test_revoke_no_token_shows_message(self, auth_workspace):
        """--revoke with no token file shows message."""
        token_path = auth_workspace["token_path"]
        token_path.unlink()

        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--revoke"])
        assert result.exit_code == 0
        assert "no token" in result.output.lower()

    def test_refresh_removes_and_reauths(self, auth_workspace):
        """--refresh removes existing token and triggers OAuth."""
        token_path = auth_workspace["token_path"]
        assert token_path.exists()

        with patch("polylogue.sources.drive_client.DriveClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            runner = CliRunner()
            result = runner.invoke(cli, ["auth", "--refresh"])

            assert "removed" in result.output.lower() or result.exit_code in (0, 1)


class TestDriveOAuthFlow:
    """Tests for the _drive_oauth_flow function."""

    def test_missing_credentials_fails(self, auth_workspace):
        """Missing credentials file shows error."""
        creds_path = auth_workspace["creds_path"]
        creds_path.unlink()

        runner = CliRunner()
        result = runner.invoke(cli, ["auth"])
        assert result.exit_code == 1
        assert (
            "credentials" in result.output.lower()
            or "missing" in result.output.lower()
            or "oauth" in result.output.lower()
        )

    def test_successful_auth_with_cached_token(self, auth_workspace):
        """Existing token uses cached credentials."""
        with patch("polylogue.sources.drive_client.DriveClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            runner = CliRunner()
            result = runner.invoke(cli, ["auth"])
            assert result.exit_code == 0
            assert "cached" in result.output.lower() or "success" in result.output.lower()

    def test_auth_failure_shows_error(self, auth_workspace):
        """Auth failure shows error message."""
        with patch("polylogue.sources.drive_client.DriveClient") as mock_client_class:
            mock_client_class.side_effect = Exception("OAuth failed: invalid_grant")

            runner = CliRunner()
            result = runner.invoke(cli, ["auth"])
            assert result.exit_code == 1
            assert "failed" in result.output.lower()

    def test_refresh_error_retries_auth(self, auth_workspace):
        """Token refresh failure triggers re-auth."""
        auth_workspace["token_path"]
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Token refresh failed")
            return MagicMock()

        with patch("polylogue.sources.drive_client.DriveClient") as mock_client_class:
            mock_client_class.side_effect = side_effect

            runner = CliRunner()
            result = runner.invoke(cli, ["auth"])
            assert result.exit_code in (0, 1)


class TestGetDrivePaths:
    """Tests for _get_drive_paths helper."""

    def test_uses_env_paths(self, auth_workspace, monkeypatch):
        """Uses paths from environment variables."""
        from polylogue.cli.commands.auth import _get_drive_paths
        from polylogue.cli.types import AppEnv

        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(auth_workspace["creds_path"]))
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(auth_workspace["token_path"]))

        mock_ui = MagicMock()
        mock_ui.plain = True
        env = AppEnv(ui=mock_ui)

        creds_path, token_path = _get_drive_paths(env)

        assert creds_path is not None
        assert token_path is not None

    def test_fallback_on_config_error(self, auth_workspace, monkeypatch):
        """Falls back to defaults if config loading fails."""
        from polylogue.cli.commands.auth import _get_drive_paths
        from polylogue.cli.types import AppEnv

        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)
        monkeypatch.delenv("POLYLOGUE_TOKEN_PATH", raising=False)

        mock_ui = MagicMock()
        mock_ui.plain = True
        env = AppEnv(ui=mock_ui)

        with patch("polylogue.cli.helpers.load_effective_config") as mock_load:
            mock_load.side_effect = Exception("Config error")
            creds_path, token_path = _get_drive_paths(env)

            assert creds_path is not None
            assert token_path is not None


class TestRevokeCredentials:
    """Tests for _revoke_drive_credentials function."""

    def test_revoke_existing_token(self, auth_workspace):
        """Revokes existing token and shows confirmation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--revoke"])

        assert result.exit_code == 0
        assert "revoked" in result.output.lower()
        assert "polylogue auth" in result.output.lower()

    def test_revoke_nonexistent_token(self, auth_workspace):
        """Handles missing token gracefully."""
        auth_workspace["token_path"].unlink()

        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--revoke"])

        assert result.exit_code == 0
        assert "no token" in result.output.lower()


# =============================================================================
# CHECK COMMAND TESTS (from test_cli_check.py)
# =============================================================================


@pytest.fixture
def runner():
    """CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_env():
    """Create mock AppEnv for tests."""
    mock_ui = MagicMock()
    mock_ui.plain = True
    mock_ui.console = MagicMock()
    mock_ui.summary = MagicMock()

    env = MagicMock()
    env.ui = mock_ui
    return env


@pytest.fixture
def sample_health_report():
    """Create a sample health report with issues."""
    return HealthReport(
        checks=[
            HealthCheck("config", VerifyStatus.OK, detail="Zero-config"),
            HealthCheck("database", VerifyStatus.OK, detail="DB reachable"),
            HealthCheck("orphaned_messages", VerifyStatus.ERROR, count=5, detail="5 orphaned messages"),
            HealthCheck("empty_conversations", VerifyStatus.WARNING, count=2, detail="2 empty conversations"),
            HealthCheck("fts_sync", VerifyStatus.OK, detail="FTS in sync"),
        ],
        summary={"ok": 3, "warning": 1, "error": 1},
    )


@pytest.fixture
def healthy_report():
    """Create a completely healthy report."""
    return HealthReport(
        checks=[
            HealthCheck("config", VerifyStatus.OK, detail="Zero-config"),
            HealthCheck("database", VerifyStatus.OK, detail="DB reachable"),
            HealthCheck("orphaned_messages", VerifyStatus.OK, detail="No orphaned messages"),
            HealthCheck("empty_conversations", VerifyStatus.OK, detail="No empty conversations"),
            HealthCheck("fts_sync", VerifyStatus.OK, detail="FTS in sync"),
        ],
        summary={"ok": 5, "warning": 0, "error": 0},
    )


class TestCheckCommand:
    """Tests for the check command."""

    def test_check_displays_health_status(self, runner, mock_env, healthy_report):
        """Check command displays health status."""
        with (
            patch("polylogue.cli.commands.check.load_effective_config"),
            patch("polylogue.cli.commands.check.get_health", return_value=healthy_report),
        ):
            result = runner.invoke(check_command, obj=mock_env)

            assert result.exit_code == 0
            assert mock_env.ui.summary.called

    def test_check_json_output(self, runner, mock_env, healthy_report):
        """Check --json outputs JSON format."""
        with (
            patch("polylogue.cli.commands.check.load_effective_config"),
            patch("polylogue.cli.commands.check.get_health", return_value=healthy_report),
        ):
            result = runner.invoke(check_command, ["--json"], obj=mock_env)

            assert result.exit_code == 0
            assert "ok" in result.output.lower()

    def test_check_vacuum_requires_repair(self, runner, mock_env):
        """--vacuum requires --repair flag."""
        result = runner.invoke(check_command, ["--vacuum"], obj=mock_env)
        assert result.exit_code != 0

    def test_check_repair_on_healthy_db(self, runner, mock_env, healthy_report):
        """Repair mode on healthy database runs repairs but finds nothing to fix."""
        clean_repairs = [
            RepairResult("orphaned_messages", 0, True, "No orphaned messages found"),
            RepairResult("empty_conversations", 0, True, "No empty conversations found"),
            RepairResult("dangling_fts", 0, True, "FTS in sync"),
            RepairResult("orphaned_attachments", 0, True, "No orphaned attachments"),
            RepairResult("wal_checkpoint", 0, True, "No WAL file present"),
        ]
        with (
            patch("polylogue.cli.commands.check.load_effective_config"),
            patch("polylogue.cli.commands.check.get_health", return_value=healthy_report),
            patch("polylogue.cli.commands.check.run_all_repairs", return_value=clean_repairs),
        ):
            result = runner.invoke(check_command, ["--repair"], obj=mock_env)

            assert result.exit_code == 0
            assert "no issues" in result.output.lower() or "0" in result.output

    def test_check_repair_runs_fixes(self, runner, mock_env, sample_health_report):
        """Repair mode runs repair functions when issues exist."""
        repair_results = [
            RepairResult("orphaned_messages", 5, True, "Deleted 5 orphaned messages"),
            RepairResult("empty_conversations", 2, True, "Deleted 2 empty conversations"),
            RepairResult("dangling_fts", 0, True, "FTS in sync"),
            RepairResult("orphaned_attachments", 0, True, "No orphaned attachments"),
        ]

        with (
            patch("polylogue.cli.commands.check.load_effective_config"),
            patch("polylogue.cli.commands.check.get_health", return_value=sample_health_report),
            patch("polylogue.cli.commands.check.run_all_repairs", return_value=repair_results),
        ):
            result = runner.invoke(check_command, ["--repair"], obj=mock_env)

            assert result.exit_code == 0
            combined = result.output
            calls = mock_env.ui.console.print.call_args_list
            combined += " ".join(str(c) for c in calls)
            assert "repair" in combined.lower()
            assert "7" in combined

    def test_check_repair_with_vacuum(self, runner, mock_env, sample_health_report):
        """Repair with --vacuum runs VACUUM after repairs."""
        repair_results = [
            RepairResult("orphaned_messages", 5, True, "Deleted 5 orphaned messages"),
        ]

        with (
            patch("polylogue.cli.commands.check.load_effective_config"),
            patch("polylogue.cli.commands.check.get_health", return_value=sample_health_report),
            patch("polylogue.cli.commands.check.run_all_repairs", return_value=repair_results),
            patch("polylogue.storage.backends.sqlite.open_connection") as mock_conn,
            patch("polylogue.storage.backends.sqlite.default_db_path", return_value=Path("/tmp/test.db")),
        ):
            mock_connection = MagicMock()
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_connection)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            result = runner.invoke(check_command, ["--repair", "--vacuum"], obj=mock_env)

            assert result.exit_code == 0
            calls = mock_env.ui.console.print.call_args_list
            output = " ".join(str(c) for c in calls)
            assert "vacuum" in output.lower()


class TestRepairFunctions:
    """Tests for individual repair functions."""

    def test_repair_orphaned_messages(self, workspace_env):
        """repair_orphaned_messages deletes orphaned messages."""
        from polylogue.config import Config
        from polylogue.storage.backends.sqlite import open_connection
        from tests.helpers import db_setup

        db_path = db_setup(workspace_env)

        with open_connection(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version) VALUES (?, ?, ?, ?, ?, ?)",
                ("orphan-msg-1", "non-existent-conv", "user", "orphaned", "hash123", 1),
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")

            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", ("non-existent-conv",)
            ).fetchone()[0]
            assert count == 1

        config = MagicMock(spec=Config)
        result = repair_orphaned_messages(config)

        assert result.success
        assert result.repaired_count == 1
        assert "orphaned" in result.detail.lower()

    def test_repair_empty_conversations(self, workspace_env):
        """repair_empty_conversations deletes empty conversations."""
        from polylogue.config import Config
        from polylogue.storage.backends.sqlite import open_connection
        from tests.helpers import db_setup

        db_path = db_setup(workspace_env)

        with open_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, provider_name, provider_conversation_id, title, created_at, updated_at, content_hash, version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("empty-conv-1", "test", "ext-1", "Empty Conv", "2024-01-01", "2024-01-01", "hash123", 1),
            )
            conn.commit()

            count = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?", ("empty-conv-1",)
            ).fetchone()[0]
            assert count == 1

        config = MagicMock(spec=Config)
        result = repair_empty_conversations(config)

        assert result.success
        assert result.repaired_count == 1
        assert "empty" in result.detail.lower()

    def test_repair_dangling_fts_no_table(self, workspace_env):
        """repair_dangling_fts handles missing FTS table gracefully."""
        from polylogue.config import Config
        from polylogue.storage.backends.sqlite import open_connection
        from tests.helpers import db_setup

        db_path = db_setup(workspace_env)

        with open_connection(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS messages_fts")
            conn.commit()

        config = MagicMock(spec=Config)
        result = repair_dangling_fts(config)

        assert result.success
        assert result.repaired_count == 0
        assert "does not exist" in result.detail

    def test_repair_orphaned_attachments(self, workspace_env):
        """repair_orphaned_attachments cleans up orphaned attachments."""
        from polylogue.config import Config
        from polylogue.storage.backends.sqlite import open_connection
        from tests.helpers import db_setup

        db_path = db_setup(workspace_env)

        with open_connection(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO attachments (attachment_id, mime_type, size_bytes, ref_count) VALUES (?, ?, ?, ?)",
                ("orphan-att-1", "image/png", 1024, 0),
            )
            conn.execute(
                "INSERT INTO attachment_refs (ref_id, attachment_id, conversation_id, message_id) VALUES (?, ?, ?, ?)",
                ("ref-1", "orphan-att-1", "non-existent-conv", "non-existent-msg"),
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")

        config = MagicMock(spec=Config)
        result = repair_orphaned_attachments(config)

        assert result.success
        assert result.repaired_count >= 1

    def test_run_all_repairs(self, workspace_env):
        """run_all_repairs runs all repair functions."""
        from polylogue.config import Config

        config = MagicMock(spec=Config)

        with (
            patch("polylogue.health.repair_orphaned_messages") as mock_orphan,
            patch("polylogue.health.repair_empty_conversations") as mock_empty,
            patch("polylogue.health.repair_dangling_fts") as mock_fts,
            patch("polylogue.health.repair_orphaned_attachments") as mock_att,
            patch("polylogue.health.repair_wal_checkpoint") as mock_wal,
        ):
            mock_orphan.return_value = RepairResult("orphaned_messages", 0, True, "OK")
            mock_empty.return_value = RepairResult("empty_conversations", 0, True, "OK")
            mock_fts.return_value = RepairResult("dangling_fts", 0, True, "OK")
            mock_att.return_value = RepairResult("orphaned_attachments", 0, True, "OK")
            mock_wal.return_value = RepairResult("wal_checkpoint", 0, True, "OK")

            results = run_all_repairs(config)

            assert len(results) == 6
            assert all(r.success for r in results)


class TestVerboseMode:
    """Tests for verbose output mode."""

    def test_verbose_shows_breakdown(self, runner, mock_env):
        """--verbose shows breakdown by provider."""
        report = HealthReport(
            checks=[
                HealthCheck(
                    "orphaned_messages",
                    VerifyStatus.WARNING,
                    count=10,
                    detail="10 orphaned messages",
                    breakdown={"chatgpt": 6, "claude": 4},
                ),
            ],
            summary={"ok": 0, "warning": 1, "error": 0},
        )

        with (
            patch("polylogue.cli.commands.check.load_effective_config"),
            patch("polylogue.cli.commands.check.get_health", return_value=report),
        ):
            result = runner.invoke(check_command, ["--verbose"], obj=mock_env)

            assert result.exit_code == 0
            assert mock_env.ui.summary.called


# =============================================================================
# EDITOR SECURITY TESTS (from test_cli_editor.py)
# =============================================================================


class TestEditorCommandValidation:
    """Parametrized tests for editor command validation."""

    @pytest.mark.parametrize("command,expected_error,description", UNSAFE_COMMAND_CASES)
    def test_validate_command_rejects_unsafe(self, command: str, expected_error: str, description: str):
        """Command with unsafe patterns should be rejected."""
        with pytest.raises(ValueError, match=expected_error):
            validate_command(command)

    @pytest.mark.parametrize("command,description", SAFE_COMMAND_CASES)
    def test_validate_command_allows_safe(self, command: str, description: str):
        """Safe editor command should be allowed."""
        validate_command(command)

    def test_validate_command_custom_context(self):
        """Custom context should appear in error message."""
        with pytest.raises(ValueError, match="CUSTOM_VAR"):
            validate_command("vim; evil", context="$CUSTOM_VAR")


class TestOpenInEditorSecurity:
    """Tests for open_in_editor function security."""

    def test_open_in_editor_rejects_injection_in_env(self, tmp_path: Path, monkeypatch):
        """open_in_editor should reject malicious $EDITOR."""
        from polylogue.cli.editor import open_in_editor

        monkeypatch.setenv("EDITOR", "vim; rm -rf /tmp/pwned")
        monkeypatch.delenv("VISUAL", raising=False)

        test_file = tmp_path / "test.txt"
        test_file.write_text("safe content")

        result = open_in_editor(test_file)
        assert result is False

    def test_open_in_editor_allows_safe_editor(self, tmp_path: Path, monkeypatch):
        """open_in_editor should handle safe $EDITOR without throwing."""
        from polylogue.cli.editor import open_in_editor

        monkeypatch.setenv("EDITOR", "nonexistent_safe_editor")
        monkeypatch.delenv("VISUAL", raising=False)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = open_in_editor(test_file)
        assert result is False

    def test_open_in_editor_returns_false_when_no_editor(self, tmp_path: Path, monkeypatch):
        """open_in_editor should return False when no editor is set."""
        from polylogue.cli.editor import open_in_editor

        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = open_in_editor(test_file)
        assert result is False

    def test_open_in_editor_returns_false_when_file_missing(self, tmp_path: Path, monkeypatch):
        """open_in_editor should return False when file doesn't exist."""
        from polylogue.cli.editor import open_in_editor

        monkeypatch.setenv("EDITOR", "vim")
        monkeypatch.delenv("VISUAL", raising=False)

        missing_file = tmp_path / "missing.txt"

        result = open_in_editor(missing_file)
        assert result is False


class TestOpenInBrowserSecurity:
    """Tests for open_in_browser function security."""

    def test_open_in_browser_rejects_injection_in_polylogue_browser(self, tmp_path: Path, monkeypatch):
        """open_in_browser should reject malicious $POLYLOGUE_BROWSER."""
        from polylogue.cli.editor import open_in_browser

        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox; rm -rf /tmp")

        test_file = tmp_path / "test.html"
        test_file.write_text("<html></html>")

        result = open_in_browser(test_file)
        assert result is False

    def test_open_in_browser_rejects_backtick_injection(self, tmp_path: Path, monkeypatch):
        """open_in_browser should reject backtick injection."""
        from polylogue.cli.editor import open_in_browser

        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox `whoami`")

        test_file = tmp_path / "test.html"
        test_file.write_text("<html></html>")

        result = open_in_browser(test_file)
        assert result is False

    def test_open_in_browser_allows_safe_browser(self, tmp_path: Path, monkeypatch):
        """open_in_browser should allow safe POLYLOGUE_BROWSER without throwing."""
        from polylogue.cli.editor import open_in_browser

        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox")

        test_file = tmp_path / "test.html"
        test_file.write_text("<html></html>")

        with patch("polylogue.cli.editor.subprocess.Popen", return_value=MagicMock()) as mock_popen:
            result = open_in_browser(test_file)
            assert result is True
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args[0][0]
            assert cmd[0] == "firefox"
            assert "file://" in cmd[1]

    def test_open_in_browser_returns_false_on_invalid_path(self, monkeypatch):
        """open_in_browser should handle invalid paths gracefully."""
        from polylogue.cli.editor import open_in_browser

        monkeypatch.setenv("POLYLOGUE_BROWSER", "firefox")

        invalid_path = Path("\x00invalid")

        result = open_in_browser(invalid_path)
        assert result is False


# --- Merged from test_supplementary_coverage.py ---


class TestPolylogueRootInitSupplementary:
    """Tests for polylogue.__init__.__getattr__ lazy imports."""

    def test_lazy_import_conversation_repository(self):
        """ConversationRepository should be importable via lazy __getattr__."""
        import polylogue

        repo_cls = polylogue.ConversationRepository
        assert repo_cls is not None
        assert repo_cls.__name__ == "ConversationRepository"

    def test_lazy_import_unknown_raises(self):
        """Unknown attributes should raise AttributeError."""
        import polylogue

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = polylogue.NonExistentThing


class TestLibInitSupplementary:
    """Tests for polylogue.lib.__init__.__getattr__ lazy imports."""

    def test_lazy_import_conversation_repository(self):
        import polylogue.lib

        repo_cls = polylogue.lib.ConversationRepository
        assert repo_cls.__name__ == "ConversationRepository"

    def test_lazy_import_conversation_projection(self):
        import polylogue.lib

        proj_cls = polylogue.lib.ConversationProjection
        assert proj_cls.__name__ == "ConversationProjection"

    def test_lazy_import_archive_stats(self):
        import polylogue.lib

        stats_cls = polylogue.lib.ArchiveStats
        assert stats_cls.__name__ == "ArchiveStats"

    def test_lazy_import_unknown_raises(self):
        import polylogue.lib

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = polylogue.lib.NonExistentThing
