from typing import Any

import pytest
from textual.widgets import DataTable, Input, TabbedContent, Tree

from polylogue.ui.tui.app import PolylogueApp
from polylogue.ui.tui.screens.base import RepositoryBoundContainer
from polylogue.ui.tui.screens.dashboard import Dashboard, ProviderBar
from polylogue.ui.tui.widgets.stats import StatCard

pytestmark = pytest.mark.tui
_skip = pytest.mark.skipif(False, reason="Textual not installed")


# ---------------------------------------------------------------------------
# Helper: create app with injected repository
# ---------------------------------------------------------------------------


def _make_app(repo: Any) -> Any:
    """Create PolylogueApp with an injected repository."""
    return PolylogueApp(repository=repo)


async def _wait_workers(pilot: Any, *, selector: str | None = None, reject: str = "Loading...") -> None:
    """Wait for all thread workers to finish, then flush DOM (Phase 2B fix).

    Thread workers schedule call_from_thread() callbacks that haven't been
    processed by Textual's event loop yet. We poll the event loop until
    callbacks land, with an optional widget value check for robustness
    under heavy system load.

    If *selector* is given, poll until that StatCard's ``value`` no longer
    equals *reject* (default ``"Loading..."``), up to ~10 s.
    """
    import asyncio

    await pilot.app.workers.wait_for_complete()
    # call_from_thread() callbacks may take several event loop ticks to land.
    for _ in range(10):
        await pilot.pause()

    # Poll until a specific widget settles (handles system load jitter).
    # Under extreme load (4000+ tests), the event loop may need real time.
    if selector:
        for _ in range(100):
            widget = pilot.app.query_one(selector, StatCard)
            if widget.value != reject:
                break
            await asyncio.sleep(0.1)
            await pilot.pause()
        await pilot.pause()


# ===========================================================================
# Dashboard tests
# ===========================================================================


@_skip
@pytest.mark.asyncio
async def test_dashboard_stats_populated(storage_repository: Any, conversation_builder: Any) -> None:
    """Seed data → mount → wait → assert stat card values match seeded counts."""
    conversation_builder("c1").add_message("m1", text="Hello").save()
    conversation_builder("c2").add_message("m2", text="World").add_message("m3", text="!").save()

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        await _wait_workers(pilot, selector="#stat-conversations")

        convs = pilot.app.query_one("#stat-conversations", StatCard)
        msgs = pilot.app.query_one("#stat-messages", StatCard)
        assert convs.value == "2"
        assert msgs.value == "3"


@_skip
@pytest.mark.asyncio
async def test_dashboard_provider_bars(storage_repository: Any, conversation_builder: Any) -> None:
    """Seed 2 providers → assert ProviderBar widgets rendered with correct counts."""
    conversation_builder("c1").provider("chatgpt").add_message("m1", text="A").save()
    conversation_builder("c2").provider("chatgpt").add_message("m2", text="B").save()
    conversation_builder("c3").provider("claude-ai").add_message("m3", text="C").save()

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        await _wait_workers(pilot)

        bars = pilot.app.query(ProviderBar)
        assert len(bars) >= 2
        # chatgpt should have 2, claude should have 1
        texts = [bar.render() for bar in bars]
        chatgpt_bar = [t for t in texts if "chatgpt" in t]
        claude_bar = [t for t in texts if "claude-ai" in t]
        assert len(chatgpt_bar) == 1
        assert len(claude_bar) == 1
        assert "2" in chatgpt_bar[0]
        assert "1" in claude_bar[0]


@_skip
@pytest.mark.asyncio
async def test_dashboard_empty_db(storage_repository: Any) -> None:
    """Empty DB → graceful '0' display, no errors."""
    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        await _wait_workers(pilot, selector="#stat-conversations")

        convs = pilot.app.query_one("#stat-conversations", StatCard)
        msgs = pilot.app.query_one("#stat-messages", StatCard)
        assert convs.value == "0"
        assert msgs.value == "0"


# ===========================================================================
# Browser tests
# ===========================================================================


@_skip
@pytest.mark.asyncio
async def test_browser_tree_populated(storage_repository: Any, conversation_builder: Any) -> None:
    """Seed conversations → switch to browser → wait → assert tree nodes match."""
    conversation_builder("c1").provider("chatgpt").title("My Chat").add_message("m1", text="Hi").save()

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "browser"
        await _wait_workers(pilot)

        tree = pilot.app.query_one("#browser-tree", Tree)
        # Root should have at least one child (provider node)
        assert len(tree.root.children) >= 1

        # Find the chatgpt provider node
        provider_names = [str(child.label) for child in tree.root.children]
        assert "Chatgpt" in provider_names

        # Expand chatgpt node — should have our conversation
        chatgpt_node = [c for c in tree.root.children if str(c.label) == "Chatgpt"][0]
        assert len(chatgpt_node.children) >= 1


@_skip
@pytest.mark.asyncio
async def test_browser_node_selection(storage_repository: Any, conversation_builder: Any) -> None:
    """Click leaf node → assert markdown viewer shows conversation content."""
    conversation_builder("c1").provider("chatgpt").title("Test Chat").add_message("m1", text="Hello World").save()

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "browser"
        await _wait_workers(pilot)

        tree = pilot.app.query_one("#browser-tree", Tree)
        chatgpt_node = [c for c in tree.root.children if str(c.label) == "Chatgpt"][0]

        # Select the leaf node (the conversation)
        if chatgpt_node.children:
            leaf = chatgpt_node.children[0]
            tree.select_node(leaf)
            tree.action_select_cursor()
            await pilot.pause()

            # Markdown viewer should have content
            from textual.widgets import Markdown as MarkdownWidget

            viewer = pilot.app.query_one("#markdown-viewer", MarkdownWidget)
            assert viewer is not None
            assert "Test Chat" in viewer.source
            assert "Hello World" in viewer.source


@_skip
@pytest.mark.asyncio
async def test_browser_empty_db(storage_repository: Any) -> None:
    """Empty DB → fallback provider list shown."""
    import asyncio

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "browser"
        await _wait_workers(pilot)

        tree = pilot.app.query_one("#browser-tree", Tree)
        # Poll until the tree is populated (handles load jitter)
        for _ in range(50):
            if len(tree.root.children) > 0:
                break
            await asyncio.sleep(0.1)
            await pilot.pause()

        # Should have fallback providers (chatgpt, claude-ai)
        provider_names = [str(child.label) for child in tree.root.children]
        assert len(provider_names) >= 2


# ===========================================================================
# Search tests (Phase 2D: FTS setup)
# ===========================================================================


@_skip
@pytest.mark.asyncio
async def test_search_flow(storage_repository: Any, conversation_builder: Any) -> None:
    """Seed + index → type query → wait → assert DataTable rows."""
    conversation_builder("c1").add_message("m1", text="UniqueSearchTerm123").save()

    # Rebuild FTS index so search works
    from polylogue.storage.backends.connection import open_connection
    from polylogue.storage.index import rebuild_index

    with open_connection(storage_repository.backend.db_path) as conn:
        rebuild_index(conn)

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "search"
        await pilot.pause()

        inp = pilot.app.query_one("#search-input", Input)
        inp.focus()
        inp.value = "UniqueSearchTerm123"
        await pilot.press("enter")
        await pilot.pause()

        table = pilot.app.query_one("#search-results", DataTable)
        assert table.row_count > 0

        # Verify the found row key matches our conversation
        row_key = next(iter(table.rows))
        assert row_key.value == "c1"

        from textual.widgets import Markdown as MarkdownWidget

        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()

        viewer = pilot.app.query_one("#search-viewer", MarkdownWidget)
        assert "UniqueSearchTerm123" in viewer.source


@_skip
@pytest.mark.asyncio
async def test_search_no_results(storage_repository: Any, conversation_builder: Any) -> None:
    """Search non-existent term → empty results, no error."""
    conversation_builder("c1").add_message("m1", text="Hello").save()

    # Rebuild FTS index
    from polylogue.storage.backends.connection import open_connection
    from polylogue.storage.index import rebuild_index

    with open_connection(storage_repository.backend.db_path) as conn:
        rebuild_index(conn)

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "search"
        await pilot.pause()

        inp = pilot.app.query_one("#search-input", Input)
        inp.focus()
        inp.value = "NonExistentXYZ999"
        await pilot.press("enter")
        await pilot.pause()

        table = pilot.app.query_one("#search-results", DataTable)
        assert table.row_count == 0


@_skip
@pytest.mark.asyncio
async def test_search_empty_db(storage_repository: Any) -> None:
    """Empty DB with FTS table → 0 results, no crash (messages_fts always exists)."""
    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "search"
        await pilot.pause()

        inp = pilot.app.query_one("#search-input", Input)
        inp.focus()
        inp.value = "anything"
        await pilot.press("enter")
        await pilot.pause()

        table = pilot.app.query_one("#search-results", DataTable)
        # Empty DB has FTS table but no rows — should return 0 results gracefully
        assert table.row_count == 0


# ===========================================================================
# Keyboard & interaction tests
# ===========================================================================


@_skip
@pytest.mark.asyncio
async def test_keyboard_tab_switch(storage_repository: Any) -> None:
    """Press Tab key → verify tab changes (basic navigation)."""
    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        assert tabs.active == "dashboard"

        # Switch to browser
        tabs.active = "browser"
        await pilot.pause()
        assert tabs.active == "browser"

        # Switch to search
        tabs.active = "search"
        await pilot.pause()
        assert tabs.active == "search"


@_skip
@pytest.mark.asyncio
async def test_dark_mode_toggle(storage_repository: Any) -> None:
    """Press 'd' → assert dark mode toggles without crashing."""
    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        await pilot.press("d")
        await pilot.pause()
        assert pilot.app.theme == "textual-light"
        await pilot.press("d")
        await pilot.pause()
        assert pilot.app.theme == "textual-dark"
        assert pilot.app.query_one(Dashboard)


@_skip
@pytest.mark.asyncio
async def test_search_missing_index_shows_rebuild_hint(storage_repository: Any, conversation_builder: Any) -> None:
    """Dropping FTS tables yields a direct rebuild hint instead of a crash."""
    from polylogue.storage.backends.connection import open_connection

    conversation_builder("c1").add_message("m1", text="Reindex me").save()

    with open_connection(storage_repository.backend.db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.commit()

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "search"
        await pilot.pause()

        inp = pilot.app.query_one("#search-input", Input)
        inp.focus()
        inp.value = "Reindex"
        await pilot.press("enter")
        await pilot.pause()

        table = pilot.app.query_one("#search-results", DataTable)
        assert table.row_count == 1
        row = table.get_row_at(0)
        assert "Search index not built" in str(row[2])


def test_repository_bound_container_requires_injected_repo() -> None:
    class DummyScreen(RepositoryBoundContainer):
        pass

    screen = DummyScreen()

    with pytest.raises(RuntimeError, match="DummyScreen widget requires an injected repository"):
        screen._get_repo("DummyScreen")


@_skip
@pytest.mark.asyncio
async def test_quit_action(storage_repository: Any) -> None:
    """Press 'q' → app exits cleanly."""
    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        await pilot.press("q")
        # If we reach here, the app didn't crash during quit
        # The run_test context manager handles exit assertions


# ===========================================================================
# Error resilience
# ===========================================================================


@_skip
@pytest.mark.asyncio
async def test_worker_failure_recovery(storage_repository: Any, monkeypatch: Any) -> None:
    """Inject error in repo → assert app stays usable with error notification."""
    from unittest.mock import MagicMock

    # Create a repo that raises on get_archive_stats
    broken_repo = MagicMock()
    broken_repo.get_archive_stats.side_effect = RuntimeError("DB exploded")

    app = PolylogueApp(repository=broken_repo)
    async with app.run_test() as pilot:
        await _wait_workers(pilot)

        # App should still be running (Dashboard catches errors and notifies)
        assert pilot.app.query_one(Dashboard)

        # Stat cards should still exist (may show "Loading...")
        stat = pilot.app.query_one("#stat-conversations", StatCard)
        assert stat is not None


# ===========================================================================
# App startup (basic sanity)
# ===========================================================================


@_skip
@pytest.mark.asyncio
async def test_app_startup(storage_repository: Any) -> None:
    """Test that the app starts and loads the dashboard."""
    app = _make_app(storage_repository)

    async with app.run_test() as pilot:
        assert pilot.app.query_one(Dashboard)
        assert pilot.app.query_one("#stat-conversations")
        assert pilot.app.query_one("#stat-messages")

        await _wait_workers(pilot, selector="#stat-conversations")

        stats = pilot.app.query_one("#stat-conversations", StatCard)
        assert stats.value == "0"
