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


def fuzz_chatgpt_parser(data: bytes) -> None:
    """Fuzz the ChatGPT parser with arbitrary JSON-like data."""
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


def fuzz_codex_parser(data: bytes) -> None:
    """Fuzz the Codex parser with arbitrary JSONL-like data."""
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


def fuzz_claude_code_parser(data: bytes) -> None:
    """Fuzz the Claude Code parser with arbitrary JSONL-like data."""
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


def fuzz_claude_ai_parser(data: bytes) -> None:
    """Fuzz the Claude AI parser with arbitrary JSON data."""
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
        assert result.provider_name == "claude-ai", f"Wrong provider: {result.provider_name}"

    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    except RecursionError:
        pass
    except MemoryError:
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception in claude.parse_ai: {type(e).__name__}: {e}") from e


def fuzz_drive_parser(data: bytes) -> None:
    """Fuzz the Drive / Gemini chunkedPrompt parser with arbitrary JSON data."""
    from polylogue.sources.parsers import drive

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return

    if not isinstance(payload, dict):
        return

    try:
        if not drive.looks_like(payload):
            return
        result = drive.parse_chunked_prompt("gemini", payload, "fuzz-fallback-id")
        assert result is not None, "parse_chunked_prompt() returned None"
        assert hasattr(result, "messages"), "Missing messages attribute"
        assert hasattr(result, "provider_name"), "Missing provider_name attribute"
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeDecodeError):
        pass
    except RecursionError:
        pass
    except MemoryError:
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception in drive.parse_chunked_prompt: {type(e).__name__}: {e}") from e


def fuzz_antigravity_parser(data: bytes) -> None:
    """Fuzz the Antigravity markdown-export parser with arbitrary JSON data."""
    from polylogue.sources.parsers import antigravity

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return

    if not isinstance(payload, dict):
        return

    try:
        if not antigravity.looks_like_markdown_export(payload):
            return
        result = antigravity.parse_markdown_export_payload(payload, "fuzz-fallback-id")
        assert result is not None, "parse_markdown_export_payload() returned None"
        assert hasattr(result, "messages"), "Missing messages attribute"
        assert hasattr(result, "provider_name"), "Missing provider_name attribute"
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeDecodeError):
        pass
    except RecursionError:
        pass
    except MemoryError:
        pass
    except antigravity.AntigravityExportError:
        pass
    except Exception as e:
        raise AssertionError(
            f"Unexpected exception in antigravity.parse_markdown_export_payload: {type(e).__name__}: {e}"
        ) from e


def fuzz_browser_capture_parser(data: bytes) -> None:
    """Fuzz the browser_capture envelope parser with arbitrary JSON data."""
    from pydantic import ValidationError

    from polylogue.sources.parsers import browser_capture

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return

    try:
        if not browser_capture.looks_like(payload):
            return
        result = browser_capture.parse(payload, "fuzz-fallback-id")
        assert result is not None, "parse() returned None"
        assert hasattr(result, "messages"), "Missing messages attribute"
        assert hasattr(result, "provider_name"), "Missing provider_name attribute"
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeDecodeError, ValidationError):
        pass
    except RecursionError:
        pass
    except MemoryError:
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception in browser_capture.parse: {type(e).__name__}: {e}") from e


def fuzz_local_agent_parser(data: bytes) -> None:
    """Fuzz the local-agent (Gemini CLI / Hermes) parsers with arbitrary JSON data."""
    from polylogue.sources.parsers import local_agent

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return

    if not isinstance(payload, dict):
        return

    try:
        if local_agent.looks_like_gemini_cli(payload):
            result = local_agent.parse_gemini_cli(payload, "fuzz-fallback-id")
            assert result is not None, "parse_gemini_cli() returned None"
            assert hasattr(result, "messages"), "Missing messages attribute"
            assert hasattr(result, "provider_name"), "Missing provider_name attribute"
        if local_agent.looks_like_hermes(payload):
            result = local_agent.parse_hermes(payload, "fuzz-fallback-id")
            assert result is not None, "parse_hermes() returned None"
            assert hasattr(result, "messages"), "Missing messages attribute"
            assert hasattr(result, "provider_name"), "Missing provider_name attribute"
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeDecodeError):
        pass
    except RecursionError:
        pass
    except MemoryError:
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception in local_agent parsers: {type(e).__name__}: {e}") from e


def fuzz_all_parsers(data: bytes) -> None:
    """Combined fuzzer that tries all parsers."""
    fuzz_chatgpt_parser(data)
    fuzz_codex_parser(data)
    fuzz_claude_code_parser(data)
    fuzz_claude_ai_parser(data)
    fuzz_drive_parser(data)
    fuzz_antigravity_parser(data)
    fuzz_browser_capture_parser(data)
    fuzz_local_agent_parser(data)


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

# Valid Drive / Gemini chunkedPrompt-like structures
DRIVE_CORPUS = [
    b'{"chunkedPrompt": {"chunks": []}}',
    b'{"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}}',
    b'{"chunkedPrompt": {"chunks": [{"author": "user", "text": "hi"}]}}',
    b'{"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi", "id": "c1"}]}}',
    b'{"chunkedPrompt": {"chunks": [{"role": "model", "text": "ok"}, {"role": "user", "text": "more"}]}}',
    b'{"chunkedPrompt": {"chunks": "not-a-list"}}',
    b'{"chunkedPrompt": null}',
    b'{"chunkedPrompt": {}}',
    b'{"chunkedPrompt": {"chunks": [null, 42, "string"]}}',
    b'{"chunkedPrompt": {"chunks": [{}]}}',
    b'{"chunkedPrompt": {"chunks": [{"role": "", "text": "x"}]}}',
    b'{"chunkedPrompt": {"chunks": [{"role": "user"}]}}',
    b'{"chunkedPrompt": {"chunks": [{"role": "user", "text": null}]}}',
    b'{"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}, "title": "Test", "createTime": "2024-01-01"}',
    b'{"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}, "displayName": "Display"}',
]

# Valid Antigravity markdown-export envelopes
ANTIGRAVITY_CORPUS = [
    b'{"source": "antigravity_language_server", "cascadeId": "c1", "markdown": "# user\\nhi"}',
    b'{"source": "antigravity_language_server", "cascadeId": "c1", "markdown": ""}',
    b'{"source": "antigravity_language_server", "cascadeId": "c1", "markdown": "# user\\nhi\\n# assistant\\nok"}',
    b'{"source": "antigravity_language_server", "cascadeId": "c1", "markdown": "x", "title": "T"}',
    b'{"source": "antigravity_language_server", "cascadeId": "c1", "markdown": "x", "workspaceName": "ws"}',
    b'{"source": "antigravity_language_server", "cascadeId": "c1", "markdown": "x", "snippet": "s", "lastModifiedTime": "2024-01-01"}',
    b'{"source": "other", "cascadeId": "c1", "markdown": "x"}',
    b'{"source": "antigravity_language_server", "cascadeId": null, "markdown": "x"}',
    b'{"source": "antigravity_language_server", "cascadeId": "c1", "markdown": null}',
    b'{"source": "antigravity_language_server"}',
]

# Valid browser_capture envelope-like structures
BROWSER_CAPTURE_CORPUS = [
    b'{"source": "browser_capture", "captureId": "c1", "session": {"turns": []}, "provenance": {"capturedAt": "2024-01-01T00:00:00Z"}}',
    b'{"source": "browser_capture", "captureId": "c1", "session": {"turns": [{"providerTurnId": "t1", "role": "user", "text": "hi"}]}, "provenance": {"capturedAt": "2024-01-01T00:00:00Z"}}',
    b"{}",
    b'{"source": "unknown"}',
    b'{"source": "browser_capture", "captureId": "c1", "session": null, "provenance": {}}',
    b'{"source": "browser_capture", "captureId": "c1", "session": {"turns": "not-a-list"}, "provenance": {}}',
]

# Valid local-agent (Gemini CLI / Hermes) structures
LOCAL_AGENT_CORPUS = [
    b'{"sessionId": "s1", "messages": [], "startTime": "2024-01-01"}',
    b'{"sessionId": "s1", "messages": [{"role": "user", "content": "hi"}], "kind": "chat"}',
    b'{"sessionId": "s1", "messages": [{"role": "assistant", "content": "ok"}], "lastUpdated": "2024-01-01"}',
    b'{"session_id": "s1", "messages": [], "platform": "hermes"}',
    b'{"session_id": "s1", "messages": [{"role": "user", "content": "hi"}], "session_start": "2024-01-01"}',
    b'{"session_id": "s1", "messages": [{"role": "assistant", "content": "ok"}], "system_prompt": "be nice", "last_updated": "2024-01-01"}',
    b'{"sessionId": "s1", "messages": null}',
    b'{"session_id": "s1", "messages": [null, {}, {"role": null}]}',
    b'{"sessionId": "", "messages": []}',
    b"{}",
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
    b'{"a":' * 50 + b'"deep"' + b"}" * 50,
    # Large strings
    b'{"key": "' + b"x" * 10000 + b'"}',
    # Unicode edge cases
    b'{"text": "\\u0000\\u001f\\u007f"}',
    b'{"text": "\\uD800"}',  # Invalid surrogate
]


# =============================================================================
# Pytest-compatible test functions
# =============================================================================


class TestParserFuzz:
    """Pytest-compatible fuzz tests using seed corpus."""

    @pytest.mark.parametrize("data", CHATGPT_CORPUS + MALFORMED_CORPUS)
    def test_chatgpt_parser_corpus(self, data: bytes) -> None:
        """Run ChatGPT parser fuzz with seed corpus."""
        fuzz_chatgpt_parser(data)

    @pytest.mark.parametrize("data", CODEX_CORPUS + MALFORMED_CORPUS)
    def test_codex_parser_corpus(self, data: bytes) -> None:
        """Run Codex parser fuzz with seed corpus."""
        fuzz_codex_parser(data)

    @pytest.mark.parametrize("data", CLAUDE_CODE_CORPUS + MALFORMED_CORPUS)
    def test_claude_code_parser_corpus(self, data: bytes) -> None:
        """Run Claude Code parser fuzz with seed corpus."""
        fuzz_claude_code_parser(data)

    def test_chatgpt_parser_random(self) -> None:
        """Run ChatGPT parser with random bytes."""
        import random

        for _ in range(500):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_chatgpt_parser(data)

    def test_codex_parser_random(self) -> None:
        """Run Codex parser with random bytes."""
        import random

        for _ in range(500):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_codex_parser(data)

    def test_claude_code_parser_random(self) -> None:
        """Run Claude Code parser with random bytes."""
        import random

        for _ in range(500):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_claude_code_parser(data)

    @pytest.mark.parametrize("data", DRIVE_CORPUS + MALFORMED_CORPUS)
    def test_drive_parser_corpus(self, data: bytes) -> None:
        """Run Drive / Gemini parser fuzz with seed corpus."""
        fuzz_drive_parser(data)

    @pytest.mark.parametrize("data", ANTIGRAVITY_CORPUS + MALFORMED_CORPUS)
    def test_antigravity_parser_corpus(self, data: bytes) -> None:
        """Run Antigravity parser fuzz with seed corpus."""
        fuzz_antigravity_parser(data)

    @pytest.mark.parametrize("data", BROWSER_CAPTURE_CORPUS + MALFORMED_CORPUS)
    def test_browser_capture_parser_corpus(self, data: bytes) -> None:
        """Run browser_capture parser fuzz with seed corpus."""
        fuzz_browser_capture_parser(data)

    @pytest.mark.parametrize("data", LOCAL_AGENT_CORPUS + MALFORMED_CORPUS)
    def test_local_agent_parser_corpus(self, data: bytes) -> None:
        """Run local-agent parser fuzz with seed corpus."""
        fuzz_local_agent_parser(data)

    def test_drive_parser_random(self) -> None:
        """Run Drive parser with ≥1000 random byte inputs."""
        import random

        for _ in range(1000):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_drive_parser(data)

    def test_antigravity_parser_random(self) -> None:
        """Run Antigravity parser with ≥1000 random byte inputs."""
        import random

        for _ in range(1000):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_antigravity_parser(data)

    def test_browser_capture_parser_random(self) -> None:
        """Run browser_capture parser with ≥1000 random byte inputs."""
        import random

        for _ in range(1000):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_browser_capture_parser(data)

    def test_local_agent_parser_random(self) -> None:
        """Run local-agent parser with ≥1000 random byte inputs."""
        import random

        for _ in range(1000):
            length = random.randint(1, 500)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_local_agent_parser(data)


# =============================================================================
# Standalone atheris fuzzer
# =============================================================================


def main() -> None:
    """Run atheris fuzzer with libFuzzer engine."""
    if not HAS_ATHERIS:
        print("atheris not installed, running pytest-compatible tests instead")
        pytest.main([__file__, "-v"])
        return

    iterations = int(os.environ.get("FUZZ_ITERATIONS", "10000"))
    print(f"Running atheris fuzzer for {iterations} iterations...")

    atheris.Setup(
        sys.argv + ["-max_total_time=300", f"-runs={iterations}"],
        fuzz_all_parsers,
    )
    atheris.Fuzz()


if __name__ == "__main__":
    main()
