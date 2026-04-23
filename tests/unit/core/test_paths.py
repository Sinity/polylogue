"""Tests for path utility functions.

Consolidated from test_paths.py.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.paths.sanitize import is_within_root, safe_path_component


class TestSafePathComponent:
    """Tests for filesystem-safe path component generation."""

    def test_simple_safe_string(self) -> None:
        """Simple alphanumeric strings pass through unchanged."""
        assert safe_path_component("hello") == "hello"
        assert safe_path_component("test-file") == "test-file"
        assert safe_path_component("v2.0.1") == "v2.0.1"

    def test_special_characters_replaced(self) -> None:
        """Strings with special chars get hashed."""
        result = safe_path_component("hello world")
        assert "-" in result
        assert len(result) > 10

    def test_empty_string_uses_fallback(self) -> None:
        """Empty string returns fallback."""
        result = safe_path_component("")
        assert result == "item"

    def test_custom_fallback(self) -> None:
        """Custom fallback is used for empty input."""
        result = safe_path_component("", fallback="default")
        assert result == "default"

    def test_none_uses_fallback(self) -> None:
        """None input returns fallback."""
        result = safe_path_component(None)
        assert result == "item"

    def test_whitespace_only_uses_fallback(self) -> None:
        """Whitespace-only input returns fallback."""
        result = safe_path_component("   ")
        assert result == "item"

    def test_dot_returns_fallback(self) -> None:
        """Single dot returns fallback (dangerous path component)."""
        result = safe_path_component(".")
        assert "item" in result

    def test_dotdot_returns_fallback(self) -> None:
        """Double dot returns fallback (path traversal)."""
        result = safe_path_component("..")
        assert "item" in result

    def test_path_separator_triggers_hash(self) -> None:
        """Path separators trigger hashed output."""
        result = safe_path_component("foo/bar")
        assert "-" in result
        assert "/" not in result

    def test_unicode_triggers_hash(self) -> None:
        """Unicode characters trigger hashed output."""
        result = safe_path_component("café")
        assert "-" in result

    def test_deterministic(self) -> None:
        """Same input always produces same output."""
        r1 = safe_path_component("hello world")
        r2 = safe_path_component("hello world")
        assert r1 == r2

    def test_different_inputs_different_outputs(self) -> None:
        """Different inputs produce different outputs."""
        r1 = safe_path_component("hello world")
        r2 = safe_path_component("goodbye world")
        assert r1 != r2

    def test_long_prefix_truncated(self) -> None:
        """Long prefixes are truncated to 12 chars."""
        result = safe_path_component("this_is_a_very_long_name with spaces")
        prefix = result.split("-")[0]
        assert len(prefix) <= 12


class TestIsWithinRoot:
    """Tests for path containment check."""

    def test_path_within_root(self, tmp_path: Path) -> None:
        """Path inside root returns True."""
        root = tmp_path / "root"
        root.mkdir()
        child = root / "subdir" / "file.txt"
        assert is_within_root(child, root) is True

    def test_path_outside_root(self, tmp_path: Path) -> None:
        """Path outside root returns False."""
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "other" / "file.txt"
        assert is_within_root(outside, root) is False

    def test_path_is_root(self, tmp_path: Path) -> None:
        """Root itself is within root."""
        root = tmp_path / "root"
        root.mkdir()
        assert is_within_root(root, root) is True

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Path traversal (../) is correctly evaluated."""
        root = tmp_path / "root"
        root.mkdir()
        traversal = root / ".." / "other"
        assert is_within_root(traversal, root) is False


class TestPathsPublicBoundary:
    def test_paths_root_exports_only_directory_layout_symbols(self) -> None:
        import polylogue.paths as paths

        assert set(paths.__all__) == {
            "GEMINI_DRIVE_FOLDER",
            "archive_root",
            "blob_store_root",
            "cache_home",
            "cache_root",
            "claude_code_path",
            "codex_path",
            "config_home",
            "config_root",
            "data_home",
            "data_root",
            "db_path",
            "drive_cache_path",
            "drive_credentials_path",
            "drive_token_path",
            "inbox_root",
            "render_root",
            "state_home",
            "state_root",
        }

    def test_paths_root_does_not_reexport_sanitization_helpers(self) -> None:
        import polylogue.paths as paths

        forbidden = {
            "conversation_render_root",
            "is_within_root",
            "safe_path_component",
        }
        assert forbidden.isdisjoint(set(paths.__all__))
        for name in forbidden:
            assert not hasattr(paths, name)
