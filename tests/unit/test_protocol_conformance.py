"""Protocol conformance tests.

Verifies that every declared implementor of VectorProvider, OutputRenderer,
and ConsoleLike satisfies the contract defined in polylogue.protocols and
related Protocol classes, and that parser modules export their expected
public API.

No mocks for the happy-path contracts — real instances with temp paths only, so
gaps in constructor signatures or missing methods fail here rather than silently
at runtime.

Findings addressed:
  - Finding 1: SearchProvider / VectorProvider / OutputRenderer (@runtime_checkable, no tests)
    SearchProvider (FTS5Provider, HybridSearchProvider) was removed
    (polylogue-a7xr.10): both implementations had zero production consumers.
  - Finding 2: Parser module interface drift (chatgpt/codex vs claude/drive)
  - Finding 3: ConsoleLike (ui/facade.py) untested
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.protocols import VectorProvider
from polylogue.ui.facade import ConsoleLike, PlainConsole

# ---------------------------------------------------------------------------
# VectorProvider (Finding 1 continued)
# ---------------------------------------------------------------------------


class TestVectorProviderConformance:
    """SqliteVecProvider must satisfy the VectorProvider protocol."""

    def test_isinstance(self, tmp_path: Path) -> None:
        """isinstance() is a structural check — sqlite-vec extension not required."""
        from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider

        provider = SqliteVecProvider(voyage_key="dummy", db_path=tmp_path / "vec.db")
        assert isinstance(provider, VectorProvider)

    def test_extension_loadable(self, tmp_path: Path) -> None:
        """sqlite-vec extension must load — skip only if package is absent, fail if it can't load."""
        import sqlite3

        try:
            import sqlite_vec
        except ImportError:
            pytest.skip("sqlite-vec Python package not installed")

        conn = sqlite3.connect(str(tmp_path / "probe.db"))
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            v = conn.execute("SELECT vec_version()").fetchone()
            assert v is not None
        except (OSError, sqlite3.OperationalError) as exc:
            pytest.fail(f"sqlite-vec extension failed to load: {exc}")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# ConsoleLike (Finding 3)
# ---------------------------------------------------------------------------


class TestConsoleLikeConformance:
    """PlainConsole must satisfy the ConsoleLike @runtime_checkable Protocol."""

    def test_plain_console_isinstance(self) -> None:
        assert isinstance(PlainConsole(), ConsoleLike)

    def test_plain_console_print_does_not_raise(self) -> None:
        PlainConsole().print("hello", "world")


# ---------------------------------------------------------------------------
# Parser module interface (Finding 2)
# ---------------------------------------------------------------------------


class TestParserModuleInterface:
    """Each parser module must export its expected public functions.

    Known asymmetries (documented here to prevent silent regressions):
    - claude.py: parse_code / parse_ai (two formats); parse / looks_like are symmetric aliases
    - drive.py: parse_chunked_prompt with an extra ``provider`` argument

    Adding a fifth provider should prompt updating this file to enforce the
    new module's interface.
    """

    def test_chatgpt_exports_parse_and_looks_like(self) -> None:
        import polylogue.sources.parsers.chatgpt as m

        assert hasattr(m, "parse") and hasattr(m, "looks_like")

    def test_codex_exports_parse_and_looks_like(self) -> None:
        import polylogue.sources.parsers.codex as m

        assert hasattr(m, "parse") and hasattr(m, "looks_like")

    def test_claude_exports_both_parse_functions_and_detectors(self) -> None:
        import polylogue.sources.parsers.claude as m

        assert hasattr(m, "parse_code") and hasattr(m, "parse_ai")
        assert hasattr(m, "looks_like_code") and hasattr(m, "looks_like_ai")
        # Symmetric aliases for interface parity with chatgpt/codex
        assert hasattr(m, "parse") and hasattr(m, "looks_like")

    def test_drive_exports_parse_chunked_prompt_and_looks_like(self) -> None:
        import polylogue.sources.parsers.drive as m

        assert callable(m.parse_chunked_prompt)
        assert callable(m.looks_like)

    def test_all_parse_functions_annotate_return_type(self) -> None:
        """Enforce the return annotation convention across all parse functions."""
        import polylogue.sources.parsers.chatgpt as chatgpt
        import polylogue.sources.parsers.claude as claude
        import polylogue.sources.parsers.codex as codex
        import polylogue.sources.parsers.drive as drive

        parse_functions: list[Callable[..., object]] = [
            chatgpt.parse,
            codex.parse,
            claude.parse_code,
            claude.parse_ai,
            drive.parse_chunked_prompt,
        ]
        for fn in parse_functions:
            assert inspect.signature(fn).return_annotation is not inspect.Signature.empty, (
                f"{fn.__qualname__} is missing a return type annotation"
            )

    def test_parse_functions_return_parsed_session(self) -> None:
        """Each parse function with a minimal valid payload must return ParsedSession."""
        import polylogue.sources.parsers.chatgpt as chatgpt
        import polylogue.sources.parsers.claude as claude
        import polylogue.sources.parsers.codex as codex
        import polylogue.sources.parsers.drive as drive
        from polylogue.sources.parsers.base import ParsedSession

        cases: list[tuple[Callable[..., ParsedSession], tuple[object, ...]]] = [
            (chatgpt.parse, ({"mapping": {}}, "fallback-id")),
            (codex.parse, ([], "fallback-id")),
            (claude.parse_code, ([], "fallback-id")),
            (claude.parse_ai, ({"chat_messages": [], "id": "ai-test"}, "fallback-id")),
            (drive.parse_chunked_prompt, ("gemini", {"chunkedPrompt": {"chunks": []}}, "fallback-id")),
        ]
        for fn, args in cases:
            result = fn(*args)
            assert isinstance(result, ParsedSession), (
                f"{fn.__qualname__} returned {type(result).__name__!r}, expected ParsedSession"
            )
