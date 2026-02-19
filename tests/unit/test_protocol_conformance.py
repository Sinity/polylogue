"""Protocol conformance tests.

Verifies that every declared implementor of SearchProvider, OutputRenderer, and
ConsoleLike satisfies the contract defined in polylogue.protocols and related
Protocol classes, and that parser modules export their expected public API.

No mocks for the happy-path contracts — real instances with temp paths only, so
gaps in constructor signatures or missing methods fail here rather than silently
at runtime.

Findings addressed:
  - Finding 1: SearchProvider / OutputRenderer (@runtime_checkable, no tests)
  - Finding 2: Parser module interface drift (chatgpt/codex vs claude/drive)
  - Finding 3: ConsoleLike (ui/facade.py) untested
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polylogue.protocols import OutputRenderer, SearchProvider, VectorProvider
from polylogue.rendering.renderers.html import HTMLRenderer
from polylogue.rendering.renderers.markdown import MarkdownRenderer
from polylogue.storage.search_providers.fts5 import FTS5Provider
from polylogue.storage.search_providers.hybrid import HybridSearchProvider
from polylogue.ui.facade import ConsoleLike, PlainConsole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fts5(tmp_path: Path) -> FTS5Provider:
    return FTS5Provider(db_path=tmp_path / "fts5.db")


def _hybrid(tmp_path: Path) -> HybridSearchProvider:
    fts5 = _fts5(tmp_path)
    vector_mock: VectorProvider = MagicMock(spec=VectorProvider)
    vector_mock.query.return_value = []
    return HybridSearchProvider(fts_provider=fts5, vector_provider=vector_mock)


# ---------------------------------------------------------------------------
# SearchProvider (Finding 1)
# ---------------------------------------------------------------------------

class TestSearchProviderConformance:
    """FTS5Provider and HybridSearchProvider must satisfy the SearchProvider protocol."""

    @pytest.mark.parametrize("factory", [_fts5, _hybrid], ids=["fts5", "hybrid"])
    def test_isinstance(self, factory, tmp_path: Path) -> None:
        assert isinstance(factory(tmp_path), SearchProvider)

    def test_fts5_search_returns_list(self, tmp_path: Path) -> None:
        result = _fts5(tmp_path).search("anything")
        assert isinstance(result, list)

    def test_fts5_index_accepts_empty_list(self, tmp_path: Path) -> None:
        _fts5(tmp_path).index([])  # Must not raise

    def test_hybrid_search_returns_list(self, tmp_path: Path) -> None:
        result = _hybrid(tmp_path).search("anything")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# OutputRenderer (Finding 1)
# ---------------------------------------------------------------------------

class TestOutputRendererConformance:
    """MarkdownRenderer and HTMLRenderer must satisfy the OutputRenderer protocol."""

    @pytest.mark.parametrize(
        "cls,expected_format",
        [(MarkdownRenderer, "markdown"), (HTMLRenderer, "html")],
        ids=["markdown", "html"],
    )
    def test_isinstance(self, cls, expected_format, tmp_path: Path) -> None:
        renderer = cls(archive_root=tmp_path)
        assert isinstance(renderer, OutputRenderer)

    @pytest.mark.parametrize(
        "cls,expected_format",
        [(MarkdownRenderer, "markdown"), (HTMLRenderer, "html")],
        ids=["markdown", "html"],
    )
    def test_supports_format(self, cls, expected_format, tmp_path: Path) -> None:
        assert cls(archive_root=tmp_path).supports_format() == expected_format

    @pytest.mark.parametrize("cls", [MarkdownRenderer, HTMLRenderer], ids=["markdown", "html"])
    def test_render_is_coroutine_function(self, cls, tmp_path: Path) -> None:
        """OutputRenderer.render() must be async — the protocol contract is async."""
        renderer = cls(archive_root=tmp_path)
        assert asyncio.iscoroutinefunction(renderer.render)


# ---------------------------------------------------------------------------
# ConsoleLike (Finding 3)
# ---------------------------------------------------------------------------

class TestConsoleLikeConformance:
    """PlainConsole must satisfy ConsoleLike (print-capable shim).

    ConsoleLike is not @runtime_checkable (it's used for static type-checking
    only), so we verify structural conformance by checking the required method
    exists and is callable rather than using isinstance().
    """

    def test_plain_console_has_print_method(self) -> None:
        assert callable(getattr(PlainConsole, "print", None))

    def test_plain_console_print_does_not_raise(self) -> None:
        PlainConsole().print("hello", "world")


# ---------------------------------------------------------------------------
# Parser module interface (Finding 2)
# ---------------------------------------------------------------------------

class TestParserModuleInterface:
    """Each parser module must export its expected public functions.

    Known asymmetries (documented here to prevent silent regressions):
    - claude.py: parse_code / parse_ai (two formats) instead of a single parse()
    - drive.py: parse_chunked_prompt with an extra ``provider`` argument
    - drive.py: no looks_like — dispatch is handled by content-type detection upstream

    Adding a fifth provider should prompt updating this file to enforce the
    new module's interface.
    """

    def test_chatgpt_exports_parse_and_looks_like(self) -> None:
        import polylogue.sources.parsers.chatgpt as m
        assert callable(m.parse) and callable(m.looks_like)

    def test_codex_exports_parse_and_looks_like(self) -> None:
        import polylogue.sources.parsers.codex as m
        assert callable(m.parse) and callable(m.looks_like)

    def test_claude_exports_both_parse_functions_and_detectors(self) -> None:
        import polylogue.sources.parsers.claude as m
        assert callable(m.parse_code) and callable(m.parse_ai)
        assert callable(m.looks_like_code) and callable(m.looks_like_ai)

    def test_drive_exports_parse_chunked_prompt(self) -> None:
        import polylogue.sources.parsers.drive as m
        assert callable(m.parse_chunked_prompt)

    def test_all_parse_functions_annotate_return_type(self) -> None:
        """Enforce the return annotation convention across all parse functions."""
        import polylogue.sources.parsers.chatgpt as chatgpt
        import polylogue.sources.parsers.claude as claude
        import polylogue.sources.parsers.codex as codex
        import polylogue.sources.parsers.drive as drive

        parse_functions = [
            chatgpt.parse,
            codex.parse,
            claude.parse_code,
            claude.parse_ai,
            drive.parse_chunked_prompt,
        ]
        for fn in parse_functions:
            assert "return" in fn.__annotations__, (
                f"{fn.__qualname__} is missing a return type annotation"
            )
