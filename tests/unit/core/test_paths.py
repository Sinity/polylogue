"""Tests for path utility functions.

Consolidated from test_paths.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import get_config
from polylogue.paths.sanitize import is_within_root, safe_path_component


class TestArchiveRootHonoursConfigFile:
    """polylogue-4ma3: paths.archive_root() must not resolve from the
    environment alone -- it has to fall back to ``polylogue.toml``'s
    ``[archive] root`` before the bare XDG default, matching the precedence
    :func:`polylogue.config.load_polylogue_config` documents and implements.

    Reverted-mutation witness: replace ``archive_root()`` in
    ``polylogue/paths/_roots.py`` with the pre-fix
    ``return _xdg_path("POLYLOGUE_ARCHIVE_ROOT", data_home())`` -- every test
    below except ``test_env_var_overrides_config_file`` and
    ``test_default_with_neither_env_nor_config_lands_under_xdg_data_home``
    then fails because the TOML-configured root is silently ignored.

    Every test here relies on the autouse ``_clear_polylogue_env`` fixture
    (``tests/conftest.py``) to strip inherited ``POLYLOGUE_*`` vars and repoint
    ``XDG_CONFIG_HOME``/``XDG_DATA_HOME`` at a fresh ``tmp_path``, so no test
    can read the real operator's ``~/.config/polylogue/polylogue.toml``.
    """

    def _write_user_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, root: Path) -> None:
        config_dir = tmp_path / "xdg-config" / "polylogue"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "polylogue.toml").write_text(f'[archive]\nroot = "{root.as_posix()}"\n', encoding="utf-8")

    def test_config_file_only_resolution_works_with_no_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from polylogue.paths import archive_root

        monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
        configured_root = tmp_path / "configured-archive"
        self._write_user_config(monkeypatch, tmp_path, configured_root)

        assert archive_root() == configured_root

    def test_env_var_overrides_config_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from polylogue.paths import archive_root

        configured_root = tmp_path / "configured-archive"
        self._write_user_config(monkeypatch, tmp_path, configured_root)
        env_root = tmp_path / "env-override-archive"
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(env_root))

        assert archive_root() == env_root

    def test_per_test_env_override_still_isolates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A monkeypatched POLYLOGUE_ARCHIVE_ROOT must take effect immediately
        and un-set again just as fast -- archive_root() must not cache a
        resolution across calls within one test (or across tests), which
        would break the many fixtures/tests that isolate scratch archives via
        a per-test override."""
        from polylogue.paths import archive_root

        configured_root = tmp_path / "configured-archive"
        self._write_user_config(monkeypatch, tmp_path, configured_root)

        monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
        assert archive_root() == configured_root

        scratch_a = tmp_path / "scratch-a"
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(scratch_a))
        assert archive_root() == scratch_a

        scratch_b = tmp_path / "scratch-b"
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(scratch_b))
        assert archive_root() == scratch_b

        monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
        assert archive_root() == configured_root

    def test_default_with_neither_env_nor_config_lands_under_xdg_data_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from polylogue.paths import archive_root, data_home

        monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
        # No polylogue.toml written under XDG_CONFIG_HOME for this test.

        assert archive_root() == data_home()
        assert archive_root() == Path(tmp_path / "xdg-data" / "polylogue")

    def test_site_config_lower_precedence_than_user_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from polylogue.paths import archive_root

        monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
        site_root = tmp_path / "site-archive"
        site_path = tmp_path / "site.toml"
        site_path.write_text(f'[archive]\nroot = "{site_root.as_posix()}"\n', encoding="utf-8")
        monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(site_path))

        # Only the site layer is configured so far.
        assert archive_root() == site_root

        user_root = tmp_path / "user-archive"
        self._write_user_config(monkeypatch, tmp_path, user_root)

        # The user layer must win over the site layer once both are present.
        assert archive_root() == user_root


def test_external_active_pointer_changes_only_the_index_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    canonical = tmp_path / "canonical" / "index.db"
    canonical.parent.mkdir()
    canonical.touch()
    (root / ".index-active-pointer").write_text(str(canonical), encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    config = get_config()

    assert config.archive_root == root
    assert config.db_path == canonical
    assert config.archive_root / "source.db" == root / "source.db"
    assert config.archive_root / "user.db" == root / "user.db"
    assert config.archive_root / "ops.db" == root / "ops.db"


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

    def test_unicode_nfc_normalization_collapses_confusables(self) -> None:
        """NFC normalization collapses decomposed equivalents to one stable form.

        ``café`` written as NFC ("é" U+00E9) and as NFD ("e" + U+0301) must
        produce the same sanitized output, so a confusable/decomposed input
        cannot bypass an existing path by hashing to a different prefix.
        """
        precomposed = "caf\u00e9"  # café (NFC)
        decomposed = "cafe\u0301"  # café (NFD: e + combining acute)
        assert safe_path_component(precomposed) == safe_path_component(decomposed)

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
            "antigravity_path",
            "archive_root",
            "blob_store_root",
            "browser_capture_pairing_state_path",
            "browser_capture_receiver_identity_path",
            "browser_capture_receiver_token_path",
            "browser_capture_spool_root",
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
            "embeddings_db_path",
            "gemini_cli_path",
            "hermes_sessions_path",
            "hooks_sidecar_dir",
            "index_db_path",
            "render_root",
            "archive_file_set_index_available_for_paths",
            "source_db_path",
            "state_home",
            "state_root",
        }
