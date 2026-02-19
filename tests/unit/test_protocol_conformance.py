"""Protocol conformance tests.

Verifies that every declared implementor of SearchProvider, VectorProvider,
OutputRenderer, and ConsoleLike satisfies the contract defined in
polylogue.protocols and related Protocol classes, and that parser modules
export their expected public API.

No mocks for the happy-path contracts — real instances with temp paths only, so
gaps in constructor signatures or missing methods fail here rather than silently
at runtime.

Findings addressed:
  - Finding 1: SearchProvider / VectorProvider / OutputRenderer (@runtime_checkable, no tests)
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

    @pytest.mark.parametrize("cls", [MarkdownRenderer, HTMLRenderer], ids=["markdown", "html"])
    async def test_render_produces_output(self, cls, tmp_path: Path) -> None:
        """render() must write a non-empty output file for a seeded conversation."""
        from tests.infra.helpers import upsert_conversation, upsert_message
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.store import ConversationRecord, MessageRecord

        db_path = tmp_path / "test.db"
        conv_id = "smoke-conv-0001"

        with open_connection(db_path) as conn:
            upsert_conversation(conn, ConversationRecord(
                conversation_id=conv_id,
                provider_name="chatgpt",
                provider_conversation_id=conv_id,
                title="Smoke Test Conversation",
                content_hash="hash-smoke-001",
            ))
            upsert_message(conn, MessageRecord(
                message_id="smoke-msg-001",
                conversation_id=conv_id,
                role="user",
                text="Hello world",
                content_hash="hash-smoke-msg-001",
            ))
            upsert_message(conn, MessageRecord(
                message_id="smoke-msg-002",
                conversation_id=conv_id,
                role="assistant",
                text="Hi there!",
                content_hash="hash-smoke-msg-002",
            ))
            conn.commit()

        output_dir = tmp_path / "out"
        renderer = cls(archive_root=output_dir)
        renderer.formatter.db_path = db_path

        result_path = await renderer.render(conv_id, output_dir)

        assert result_path.exists()
        assert result_path.stat().st_size > 0


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
        assert callable(m.parse) and callable(m.looks_like)

    def test_codex_exports_parse_and_looks_like(self) -> None:
        import polylogue.sources.parsers.codex as m
        assert callable(m.parse) and callable(m.looks_like)

    def test_claude_exports_both_parse_functions_and_detectors(self) -> None:
        import polylogue.sources.parsers.claude as m
        assert callable(m.parse_code) and callable(m.parse_ai)
        assert callable(m.looks_like_code) and callable(m.looks_like_ai)
        # Symmetric aliases for interface parity with chatgpt/codex
        assert callable(m.parse) and callable(m.looks_like)

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

    def test_parse_functions_return_parsed_conversation(self) -> None:
        """Each parse function with a minimal valid payload must return ParsedConversation."""
        import polylogue.sources.parsers.chatgpt as chatgpt
        import polylogue.sources.parsers.claude as claude
        import polylogue.sources.parsers.codex as codex
        import polylogue.sources.parsers.drive as drive
        from polylogue.sources.parsers.base import ParsedConversation

        cases = [
            (chatgpt.parse, ({"mapping": {}}, "fallback-id")),
            (codex.parse, ([], "fallback-id")),
            (claude.parse_code, ([], "fallback-id")),
            (claude.parse_ai, ({"chat_messages": [], "id": "ai-test"}, "fallback-id")),
            (drive.parse_chunked_prompt, ("gemini", {"chunkedPrompt": {"chunks": []}}, "fallback-id")),
        ]
        for fn, args in cases:
            result = fn(*args)
            assert isinstance(result, ParsedConversation), (
                f"{fn.__qualname__} returned {type(result).__name__!r}, expected ParsedConversation"
            )
