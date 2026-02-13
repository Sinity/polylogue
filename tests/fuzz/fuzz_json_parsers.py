#!/usr/bin/env python
"""Fuzz the parser parse functions with malformed JSON.

Target: Crashes and unexpected exceptions in parser parsing.

Security properties tested:
- No crashes on malformed JSON
- No unhandled exceptions that could leak internal state
- Graceful handling of deeply nested structures
- Memory safety with large inputs
"""

from __future__ import annotations

import json
import os
import sys

# Check if atheris is available
try:
    import atheris
    HAS_ATHERIS = True
except ImportError:
    HAS_ATHERIS = False

import pytest


def fuzz_chatgpt_importer(data: bytes) -> None:
    """Fuzz the ChatGPT importer with arbitrary JSON-like data."""
    from polylogue.sources.parsers import chatgpt

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        # Try parsing as JSON
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        # Invalid JSON is expected, not a bug
        return

    try:
        # Check if it looks like ChatGPT format
        if not chatgpt.looks_like(payload):
            return

        # Parse it
        result = chatgpt.parse(payload, "fuzz-fallback-id")

        # Verify result structure
        assert result is not None, "parse() returned None"
        assert hasattr(result, "messages"), "Missing messages attribute"
        assert hasattr(result, "provider_name"), "Missing provider_name attribute"
        assert result.provider_name == "chatgpt", f"Wrong provider: {result.provider_name}"

    except (ValueError, TypeError, KeyError, AttributeError):
        # These are acceptable handling of malformed input
        pass
    except RecursionError:
        # Deep nesting is acceptable to reject
        pass
    except MemoryError:
        # Large input rejection is acceptable
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception in chatgpt.parse: {type(e).__name__}: {e}") from e


def fuzz_codex_importer(data: bytes) -> None:
    """Fuzz the Codex importer with arbitrary JSONL-like data."""
    from polylogue.sources.parsers import codex

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    # Parse as JSONL (one JSON object per line)
    payload = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            payload.append(obj)
        except (json.JSONDecodeError, ValueError, RecursionError):
            # Invalid line, skip
            continue

    if not payload:
        return

    try:
        # Check if it looks like Codex format
        if not codex.looks_like(payload):
            return

        # Parse it
        result = codex.parse(payload, "fuzz-fallback-id")

        # Verify result structure
        assert result is not None, "parse() returned None"
        assert hasattr(result, "messages"), "Missing messages attribute"
        assert hasattr(result, "provider_name"), "Missing provider_name attribute"
        assert result.provider_name == "codex", f"Wrong provider: {result.provider_name}"

    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    except RecursionError:
        pass
    except MemoryError:
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception in codex.parse: {type(e).__name__}: {e}") from e


def fuzz_claude_code_importer(data: bytes) -> None:
    """Fuzz the Claude Code importer with arbitrary JSONL-like data."""
    from polylogue.sources.parsers import claude

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    # Parse as JSONL
    payload = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            payload.append(obj)
        except (json.JSONDecodeError, ValueError, RecursionError):
            continue

    if not payload:
        return

    try:
        # Check if it looks like Claude Code format
        if not claude.looks_like_code(payload):
            return

        # Parse it
        result = claude.parse_code(payload, "fuzz-fallback-id")

        # Verify result structure
        assert result is not None, "parse_code() returned None"
        assert hasattr(result, "messages"), "Missing messages attribute"
        assert hasattr(result, "provider_name"), "Missing provider_name attribute"
        assert result.provider_name == "claude-code", f"Wrong provider: {result.provider_name}"

    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    except RecursionError:
        pass
    except MemoryError:
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception in claude.parse_code: {type(e).__name__}: {e}") from e


def fuzz_claude_ai_importer(data: bytes) -> None:
    """Fuzz the Claude AI importer with arbitrary JSON data."""
    from polylogue.sources.parsers import claude

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return

    try:
        # Check if it looks like Claude AI format
        if not claude.looks_like_ai(payload):
            return

        # Parse it
        result = claude.parse_ai(payload, "fuzz-fallback-id")

        # Verify result structure
        assert result is not None, "parse_ai() returned None"
        assert hasattr(result, "messages"), "Missing messages attribute"
        assert hasattr(result, "provider_name"), "Missing provider_name attribute"
        assert result.provider_name == "claude", f"Wrong provider: {result.provider_name}"

    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    except RecursionError:
        pass
    except MemoryError:
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception in claude.parse_ai: {type(e).__name__}: {e}") from e


def fuzz_all_importers(data: bytes) -> None:
    """Combined fuzzer that tries all importers."""
    fuzz_chatgpt_importer(data)
    fuzz_codex_importer(data)
    fuzz_claude_code_importer(data)
    fuzz_claude_ai_importer(data)


# =============================================================================
# Seed corpus generation
# =============================================================================

# Valid ChatGPT-like structures
CHATGPT_CORPUS = [
    # Valid minimal
    b'{"mapping": {}}',
    b'{"mapping": {"m1": {"message": {"content": {"parts": ["hello"]}}}}}',
    # Edge cases
    b'{"mapping": null}',
    b'{"mapping": []}',
    b'{"mapping": "string"}',
    b'{"mapping": {"m1": null}}',
    b'{"mapping": {"m1": {"message": null}}}',
    b'{"mapping": {"m1": {"message": {"content": null}}}}',
    b'{"mapping": {"m1": {"message": {"content": {"parts": null}}}}}',
    # Type confusion
    b'{"mapping": {"m1": {"message": {"content": {"parts": "not-array"}}}}}',
    b'{"mapping": {"m1": {"message": {"content": {"parts": 123}}}}}',
    # Large structures
    b'{"mapping": {"m0": {}, "m1": {}, "m2": {}, "m3": {}, "m4": {}}}',
    # Deeply nested
    b'{"mapping": {"m1": {"message": {"content": {"parts": [{"nested": {"deep": "value"}}]}}}}}',
]

# Valid Codex-like structures
CODEX_CORPUS = [
    # Session meta
    b'{"type":"session_meta","payload":{"id":"test","timestamp":"2024-01-01"}}',
    # Response item
    b'{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}',
    # Intermediate format
    b'{"id":"test","timestamp":"2024-01-01"}',
    b'{"type":"message","role":"user","content":[]}',
    # Edge cases
    b'{"type":"session_meta","payload":null}',
    b'{"type":"response_item","payload":"string"}',
    b'{"type":"message","role":null,"content":null}',
]

# Valid Claude Code-like structures
CLAUDE_CODE_CORPUS = [
    # Init message
    b'{"type":"init","sessionId":"test"}',
    # User message
    b'{"type":"user","uuid":"u1","message":{"content":"hello"}}',
    # Assistant message
    b'{"type":"assistant","uuid":"a1","message":{"content":[{"type":"text","text":"hi"}]}}',
    # With tool use
    b'{"type":"assistant","uuid":"a2","message":{"content":[{"type":"tool_use","name":"Read","id":"t1","input":{}}]}}',
    # With thinking
    b'{"type":"assistant","uuid":"a3","message":{"content":[{"type":"thinking","thinking":"hmm..."}]}}',
    # Edge cases
    b'{"type":"user","uuid":"u1","message":null}',
    b'{"type":"assistant","uuid":"a1","message":{"content":null}}',
    b'{"parentUuid":"p1","leafUuid":"l1"}',
]

# Malformed JSON
MALFORMED_CORPUS = [
    # Truncated
    b'{"key": "val',
    b'{"key": [1, 2, ',
    b'{"key": {"nested":',
    # Invalid syntax
    b"{'single': 'quotes'}",
    b'{key: "no quotes on key"}',
    b'{"trailing": "comma",}',
    # Not JSON
    b"",
    b"   ",
    b"\n\n\n",
    b"null",
    b"true",
    b"42",
    b'"just a string"',
    # Deep nesting
    b'{"a":' * 50 + b'"deep"' + b'}' * 50,
    # Large strings
    b'{"key": "' + b'x' * 10000 + b'"}',
    # Unicode edge cases
    b'{"text": "\\u0000\\u001f\\u007f"}',
    b'{"text": "\\uD800"}',  # Invalid surrogate
]


# =============================================================================
# Pytest-compatible test functions
# =============================================================================


class TestImporterFuzz:
    """Pytest-compatible fuzz tests using seed corpus."""

    @pytest.mark.parametrize("data", CHATGPT_CORPUS + MALFORMED_CORPUS)
    def test_chatgpt_importer_corpus(self, data: bytes):
        """Run ChatGPT importer fuzz with seed corpus."""
        fuzz_chatgpt_importer(data)

    @pytest.mark.parametrize("data", CODEX_CORPUS + MALFORMED_CORPUS)
    def test_codex_importer_corpus(self, data: bytes):
        """Run Codex importer fuzz with seed corpus."""
        fuzz_codex_importer(data)

    @pytest.mark.parametrize("data", CLAUDE_CODE_CORPUS + MALFORMED_CORPUS)
    def test_claude_code_importer_corpus(self, data: bytes):
        """Run Claude Code importer fuzz with seed corpus."""
        fuzz_claude_code_importer(data)

    def test_chatgpt_importer_random(self):
        """Run ChatGPT importer with random bytes."""
        import random

        for _ in range(500):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_chatgpt_importer(data)

    def test_codex_importer_random(self):
        """Run Codex importer with random bytes."""
        import random

        for _ in range(500):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_codex_importer(data)

    def test_claude_code_importer_random(self):
        """Run Claude Code importer with random bytes."""
        import random

        for _ in range(500):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_claude_code_importer(data)


# =============================================================================
# Standalone atheris fuzzer
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
        fuzz_all_importers,
    )
    atheris.Fuzz()


if __name__ == "__main__":
    main()
