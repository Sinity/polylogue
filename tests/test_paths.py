"""Tests for polylogue.paths module.

Covers:
- safe_path_component() sanitization and edge cases
- is_within_root() path containment check
- Source dataclass validation
- XDG path resolution (_xdg_path)
- Module-level path constants
- IndexConfig.from_env() env var resolution
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.paths import (
    DriveConfig,
    IndexConfig,
    Source,
    is_within_root,
    safe_path_component,
)

# =============================================================================
# safe_path_component() Tests
# =============================================================================


class TestSafePathComponent:
    """Tests for filesystem-safe path component generation."""

    def test_simple_safe_string(self):
        """Simple alphanumeric strings pass through unchanged."""
        assert safe_path_component("hello") == "hello"
        assert safe_path_component("test-file") == "test-file"
        assert safe_path_component("v2.0.1") == "v2.0.1"

    def test_special_characters_replaced(self):
        """Strings with special chars get hashed."""
        result = safe_path_component("hello world")
        assert "-" in result  # prefix-hash format
        assert len(result) > 10  # has hash suffix

    def test_empty_string_uses_fallback(self):
        """Empty string returns fallback."""
        result = safe_path_component("")
        assert result == "item"

    def test_custom_fallback(self):
        """Custom fallback is used for empty input."""
        result = safe_path_component("", fallback="default")
        assert result == "default"

    def test_none_uses_fallback(self):
        """None input returns fallback."""
        result = safe_path_component(None)
        assert result == "item"

    def test_whitespace_only_uses_fallback(self):
        """Whitespace-only input returns fallback."""
        result = safe_path_component("   ")
        assert result == "item"

    def test_dot_returns_fallback(self):
        """Single dot returns fallback (dangerous path component)."""
        result = safe_path_component(".")
        assert "item" in result

    def test_dotdot_returns_fallback(self):
        """Double dot returns fallback (path traversal)."""
        result = safe_path_component("..")
        assert "item" in result

    def test_path_separator_triggers_hash(self):
        """Path separators trigger hashed output."""
        result = safe_path_component("foo/bar")
        assert "-" in result
        # Should NOT contain the literal separator
        assert "/" not in result

    def test_unicode_triggers_hash(self):
        """Unicode characters trigger hashed output."""
        result = safe_path_component("café")
        assert "-" in result

    def test_deterministic(self):
        """Same input always produces same output."""
        r1 = safe_path_component("hello world")
        r2 = safe_path_component("hello world")
        assert r1 == r2

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs."""
        r1 = safe_path_component("hello world")
        r2 = safe_path_component("goodbye world")
        assert r1 != r2

    def test_long_prefix_truncated(self):
        """Long prefixes are truncated to 12 chars."""
        result = safe_path_component("this_is_a_very_long_name with spaces")
        prefix = result.split("-")[0]
        assert len(prefix) <= 12


# =============================================================================
# is_within_root() Tests
# =============================================================================


class TestIsWithinRoot:
    """Tests for path containment check."""

    def test_path_within_root(self, tmp_path):
        """Path inside root returns True."""
        root = tmp_path / "root"
        root.mkdir()
        child = root / "subdir" / "file.txt"
        assert is_within_root(child, root) is True

    def test_path_outside_root(self, tmp_path):
        """Path outside root returns False."""
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "other" / "file.txt"
        assert is_within_root(outside, root) is False

    def test_path_is_root(self, tmp_path):
        """Root itself is within root."""
        root = tmp_path / "root"
        root.mkdir()
        assert is_within_root(root, root) is True

    def test_path_traversal_blocked(self, tmp_path):
        """Path traversal (../) is correctly evaluated."""
        root = tmp_path / "root"
        root.mkdir()
        traversal = root / ".." / "other"
        assert is_within_root(traversal, root) is False


# =============================================================================
# Source Dataclass Tests
# =============================================================================


class TestSource:
    """Tests for Source dataclass validation."""

    def test_source_with_path(self, tmp_path):
        """Source with path is valid."""
        src = Source(name="test", path=tmp_path)
        assert src.name == "test"
        assert src.path == tmp_path
        assert not src.is_drive

    def test_source_with_folder(self):
        """Source with folder (Drive) is valid."""
        src = Source(name="gemini", folder="Google AI Studio")
        assert src.name == "gemini"
        assert src.is_drive

    def test_source_empty_name_raises(self):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Source(name="", path=Path("/tmp"))

    def test_source_whitespace_name_raises(self):
        """Whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Source(name="   ", path=Path("/tmp"))

    def test_source_no_path_no_folder_raises(self):
        """Source without path or folder raises ValueError."""
        with pytest.raises(ValueError, match="must have either"):
            Source(name="broken")

    def test_source_both_path_and_folder_raises(self):
        """Source with both path and folder raises ValueError."""
        with pytest.raises(ValueError, match="cannot have both"):
            Source(name="confused", path=Path("/tmp"), folder="Drive Folder")

    def test_source_name_stripped(self):
        """Source name is stripped of whitespace."""
        src = Source(name="  test  ", path=Path("/tmp"))
        assert src.name == "test"

    def test_source_folder_stripped(self):
        """Source folder is stripped of whitespace."""
        src = Source(name="test", folder="  My Folder  ")
        assert src.folder == "My Folder"


# =============================================================================
# DriveConfig Tests
# =============================================================================


class TestDriveConfig:
    """Tests for DriveConfig defaults."""

    def test_default_retry_count(self):
        """Default retry count is 3."""
        config = DriveConfig()
        assert config.retry_count == 3

    def test_default_timeout(self):
        """Default timeout is 30 seconds."""
        config = DriveConfig()
        assert config.timeout == 30

    def test_credentials_path_is_in_config(self):
        """Default credentials path is in polylogue config dir."""
        config = DriveConfig()
        assert "polylogue" in str(config.credentials_path)


# =============================================================================
# IndexConfig Tests
# =============================================================================


class TestIndexConfig:
    """Tests for IndexConfig from environment."""

    def test_from_env_defaults(self, monkeypatch):
        """Default IndexConfig has FTS enabled, no external services."""
        monkeypatch.delenv("POLYLOGUE_VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("POLYLOGUE_VOYAGE_MODEL", raising=False)
        monkeypatch.delenv("POLYLOGUE_VOYAGE_DIMENSION", raising=False)
        monkeypatch.delenv("POLYLOGUE_AUTO_EMBED", raising=False)
        config = IndexConfig.from_env()
        assert config.fts_enabled is True
        assert config.voyage_api_key is None
        assert config.voyage_model == "voyage-4"
        assert config.voyage_dimension is None
        assert config.auto_embed is False

    def test_from_env_polylogue_prefixed(self, monkeypatch):
        """POLYLOGUE_* env vars are picked up."""
        monkeypatch.setenv("POLYLOGUE_VOYAGE_API_KEY", "voyage-key")
        monkeypatch.setenv("POLYLOGUE_VOYAGE_MODEL", "voyage-4-large")
        monkeypatch.setenv("POLYLOGUE_VOYAGE_DIMENSION", "512")
        monkeypatch.setenv("POLYLOGUE_AUTO_EMBED", "true")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "voyage-key"
        assert config.voyage_model == "voyage-4-large"
        assert config.voyage_dimension == 512
        assert config.auto_embed is True

    def test_from_env_unprefixed_fallback(self, monkeypatch):
        """Unprefixed env vars used when POLYLOGUE_* not set."""
        monkeypatch.delenv("POLYLOGUE_VOYAGE_API_KEY", raising=False)
        monkeypatch.setenv("VOYAGE_API_KEY", "fallback-key")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "fallback-key"

    def test_from_env_prefixed_takes_precedence(self, monkeypatch):
        """POLYLOGUE_* vars take precedence over unprefixed."""
        monkeypatch.setenv("POLYLOGUE_VOYAGE_API_KEY", "preferred-key")
        monkeypatch.setenv("VOYAGE_API_KEY", "fallback-key")
        config = IndexConfig.from_env()
        assert config.voyage_api_key == "preferred-key"


# =============================================================================
# XDG Path Constants Tests
# =============================================================================


class TestXDGPaths:
    """Tests for XDG path resolution."""

    def test_xdg_data_home_respected(self, monkeypatch):
        """XDG_DATA_HOME env var overrides default."""
        monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")

        import importlib

        import polylogue.paths

        importlib.reload(polylogue.paths)

        assert Path("/custom/data") == polylogue.paths.DATA_ROOT

        # Clean up — reload with original env
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        importlib.reload(polylogue.paths)

    def test_db_path_under_data_home(self, workspace_env):
        """DB_PATH is under XDG_DATA_HOME/polylogue/."""
        import polylogue.paths

        assert "polylogue" in str(polylogue.paths.DB_PATH)
        assert polylogue.paths.DB_PATH.name == "polylogue.db"
