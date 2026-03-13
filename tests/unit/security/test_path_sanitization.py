"""Regression tests for path sanitization and token store security.

Covers bugs from:
- 3796914: Drive path traversal, token file perms (0644→0600)
- FileTokenStore key sanitization
- sanitize_path traversal blocking
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from polylogue.lib.security import sanitize_path
from polylogue.sources.token_store import FileTokenStore

# =============================================================================
# sanitize_path: traversal prevention (3796914)
# =============================================================================

class TestSanitizePathTraversal:
    """Path traversal attacks must be blocked."""

    @pytest.mark.parametrize(
        "raw_path",
        [
            "../../../etc/passwd",
            "safe/../../etc/passwd",
            "..",
            "..\\..\\system32",
        ],
    )
    def test_traversal_inputs_are_blocked(self, raw_path):
        result = sanitize_path(raw_path)
        assert result is not None
        assert result.startswith("_blocked_")

    @pytest.mark.parametrize(
        "raw_path",
        [
            "file\x00.txt",
            "file\x01\x02name",
            "file\x7fname",
        ],
    )
    def test_control_characters_are_removed(self, raw_path):
        result = sanitize_path(raw_path)
        assert result is not None
        assert not any((ord(c) < 32 or ord(c) == 127) for c in result)

    @pytest.mark.parametrize(
        "raw_path,checks",
        [
            (
                "conversations/chatgpt/export.json",
                ("conversations", "chatgpt"),
            ),
            (
                "/home/user/conversations/export.json",
                ("/", "conversations"),
            ),
        ],
    )
    def test_safe_paths_remain_usable(self, raw_path, checks):
        result = sanitize_path(raw_path)
        assert result is not None
        for check in checks:
            assert check in result or result.startswith(check)

    def test_none_returns_none(self):
        assert sanitize_path(None) is None


# =============================================================================
# FileTokenStore: key sanitization and permissions
# =============================================================================

class TestFileTokenStoreSecurity:
    """Token store must prevent path traversal via key and set secure permissions."""

    def test_slash_in_key_sanitized(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        path = store._path_for_key("../../etc/passwd")
        # Must stay within the store directory
        assert tmp_path in path.parents or path.parent == tmp_path

    def test_backslash_in_key_sanitized(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        path = store._path_for_key("..\\..\\secret")
        assert tmp_path in path.parents or path.parent == tmp_path

    def test_dot_dot_in_key_sanitized(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        path = store._path_for_key("../secret_token")
        # Double-dot should be replaced with underscore
        assert ".." not in str(path.name)

    def test_save_creates_file_with_0600(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        store.save("test-token", '{"access_token": "secret"}')

        path = store._path_for_key("test-token")
        assert path.exists()
        mode = oct(stat.S_IMODE(os.stat(path).st_mode))
        assert mode == "0o600"

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        data = '{"access_token": "abc123", "refresh_token": "xyz789"}'
        store.save("oauth-token", data)
        loaded = store.load("oauth-token")
        assert loaded == data

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        assert store.load("nonexistent") is None

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        store.save("deleteme", "data")
        path = store._path_for_key("deleteme")
        assert path.exists()
        store.delete("deleteme")
        assert not path.exists()

    def test_delete_nonexistent_no_error(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        store.delete("doesnt-exist")  # Should not raise

    def test_key_with_special_chars(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path)
        store.save("token/with/slashes", "data")
        loaded = store.load("token/with/slashes")
        assert loaded == "data"


# =============================================================================
# ReDoS safety: regex patterns must not exhibit catastrophic backtracking
# =============================================================================

class TestRegexReDoSSafety:
    """Regex patterns in production code must complete in bounded time.

    Even though the patterns use negated character classes (which are safe by
    construction), we test with adversarial inputs as defense-in-depth. Each
    test has a 1-second implicit timeout — if any pattern takes longer, the
    test runner will flag it.
    """

    def test_path_pattern_adversarial_input(self):
        """_PATH_PATTERN must handle long strings without backtracking."""
        import time

        from polylogue.lib.viewports import _PATH_PATTERN

        # Adversarial: long string of dots and slashes that could cause
        # backtracking in a poorly-written path regex
        adversarial = "./" + "a" * 10_000 + " " + "/b" * 1_000
        start = time.monotonic()
        _PATH_PATTERN.findall(adversarial)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"_PATH_PATTERN took {elapsed:.2f}s on adversarial input"

    def test_path_pattern_many_quoted_segments(self):
        """Quoted path segments must not cause exponential matching."""
        import time

        from polylogue.lib.viewports import _PATH_PATTERN

        # Many alternating quote/space boundaries
        adversarial = " ".join(f'"/path/segment{i}"' for i in range(500))
        start = time.monotonic()
        results = _PATH_PATTERN.findall(adversarial)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"_PATH_PATTERN took {elapsed:.2f}s on quoted paths"
        assert len(results) == 500

    def test_uuid_pattern_adversarial_input(self):
        """UUID_PATTERN must reject near-miss inputs quickly."""
        import time

        from polylogue.schemas.schema_inference import UUID_PATTERN

        # Almost-UUIDs that require scanning to reject
        near_misses = [
            "a" * 36,  # right length, no hyphens
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaX",  # trailing non-hex
            "-".join(["abcdef01"] * 5),  # wrong segment lengths
        ]
        start = time.monotonic()
        for nm in near_misses * 1000:
            UUID_PATTERN.match(nm)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"UUID_PATTERN took {elapsed:.2f}s on 3000 near-miss inputs"

    def test_fts5_special_adversarial_input(self):
        """FTS5 escape patterns must handle pathological input."""
        import time

        from polylogue.storage.search import _FTS5_SPECIAL

        # Long string of special characters
        adversarial = "'" * 5_000 + '"' * 5_000 + ":" * 5_000
        start = time.monotonic()
        _FTS5_SPECIAL.sub("", adversarial)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"_FTS5_SPECIAL took {elapsed:.2f}s on special chars"
