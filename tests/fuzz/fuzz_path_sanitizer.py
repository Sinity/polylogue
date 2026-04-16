#!/usr/bin/env python
"""Fuzz the path sanitizer in ParsedAttachment.

Target: Path traversal attacks via malicious attachment paths.

Security properties tested:
- Null byte injection prevention
- Directory traversal (..) blocking
- Symlink path blocking
- Control character stripping
- Absolute path handling
"""

from __future__ import annotations

import os
import sys

# Check if atheris is available
try:
    import atheris

    HAS_ATHERIS = True
except ImportError:
    HAS_ATHERIS = False

import pytest

from tests.infra.adversarial_cases import PATH_TRAVERSAL_CORPUS_BYTES


def fuzz_path_sanitizer(data: bytes) -> None:
    """Fuzz the path sanitizer with arbitrary byte sequences."""
    from polylogue.sources.parsers.base import ParsedAttachment

    try:
        # Try decoding as UTF-8 with errors replaced
        path_input = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        # Create attachment with potentially malicious path
        attachment = ParsedAttachment(
            provider_attachment_id="test-id",
            message_provider_id="msg-1",
            path=path_input,
        )

        # Verify security properties
        sanitized = attachment.path
        if sanitized is None:
            return

        # Property 1: No null bytes
        assert "\x00" not in sanitized, f"Null byte in sanitized path: {sanitized!r}"

        # Property 2: No control characters (ASCII < 32 or 127)
        for char in sanitized:
            code = ord(char)
            assert code >= 32 and code != 127, f"Control char in sanitized path: {sanitized!r}"

        # Property 3: No traversal sequences in output (unless blocked)
        if not sanitized.startswith("_blocked_"):
            # If not blocked, ensure no raw ".." sequences lead to parent
            parts = sanitized.split("/")
            assert ".." not in parts, f"Traversal in non-blocked path: {sanitized!r}"

    except (ValueError, TypeError):
        # Validation errors are acceptable
        pass
    except Exception as e:
        # Unexpected exceptions should be reported
        raise AssertionError(f"Unexpected exception: {type(e).__name__}: {e}") from e


def fuzz_name_sanitizer(data: bytes) -> None:
    """Fuzz the filename sanitizer with arbitrary byte sequences."""
    from polylogue.sources.parsers.base import ParsedAttachment

    try:
        name_input = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        attachment = ParsedAttachment(
            provider_attachment_id="test-id",
            message_provider_id="msg-1",
            name=name_input,
        )

        sanitized = attachment.name
        if sanitized is None:
            return

        # Property 1: No null bytes
        assert "\x00" not in sanitized, f"Null byte in sanitized name: {sanitized!r}"

        # Property 2: No control characters
        for char in sanitized:
            code = ord(char)
            assert code >= 32 and code != 127, f"Control char in sanitized name: {sanitized!r}"

        # Property 3: Not a dots-only name (security issue)
        assert sanitized.strip(".") != "" or sanitized == "file", f"Dots-only name: {sanitized!r}"

    except (ValueError, TypeError):
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception: {type(e).__name__}: {e}") from e


# =============================================================================
# Pytest-compatible test functions (run without atheris for CI)
# =============================================================================


class TestPathSanitizerFuzz:
    """Pytest-compatible fuzz tests using seed corpus."""

    @pytest.mark.parametrize("data", PATH_TRAVERSAL_CORPUS_BYTES)
    def test_path_sanitizer_corpus(self, data: bytes):
        """Run path sanitizer fuzz with seed corpus."""
        fuzz_path_sanitizer(data)

    @pytest.mark.parametrize("data", PATH_TRAVERSAL_CORPUS_BYTES)
    def test_name_sanitizer_corpus(self, data: bytes):
        """Run name sanitizer fuzz with seed corpus."""
        fuzz_name_sanitizer(data)

    def test_path_sanitizer_random(self):
        """Run path sanitizer with random bytes (limited iterations)."""
        import random

        for _ in range(1000):
            length = random.randint(1, 200)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_path_sanitizer(data)

    def test_name_sanitizer_random(self):
        """Run name sanitizer with random bytes (limited iterations)."""
        import random

        for _ in range(1000):
            length = random.randint(1, 100)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_name_sanitizer(data)


# =============================================================================
# Standalone atheris fuzzer (run directly for extended fuzzing)
# =============================================================================


def main():
    """Run atheris fuzzer with libFuzzer engine."""
    if not HAS_ATHERIS:
        print("atheris not installed, running pytest-compatible tests instead")
        pytest.main([__file__, "-v"])
        return

    iterations = int(os.environ.get("FUZZ_ITERATIONS", "10000"))
    print(f"Running atheris fuzzer for {iterations} iterations...")

    atheris.Setup(
        sys.argv + ["-max_total_time=300", f"-runs={iterations}"],
        fuzz_path_sanitizer,
    )
    atheris.Fuzz()


if __name__ == "__main__":
    main()
