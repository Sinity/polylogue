"""Tests for path utility functions.

Consolidated from test_paths.py.
"""

from __future__ import annotations

from polylogue.paths import is_within_root, safe_path_component


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
        assert "-" in result
        assert len(result) > 10

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
