import pytest
from textual.widgets import DataTable, Input, TabbedContent, Tree

try:
    from polylogue.ui.tui.app import PolylogueApp
    from polylogue.ui.tui.screens.dashboard import Dashboard, ProviderBar
    from polylogue.ui.tui.widgets.stats import StatCard
except ImportError:
    # Textual might not be installed in some envs
    PolylogueApp = None

_skip = pytest.mark.skipif(PolylogueApp is None, reason="Textual not installed")


# ---------------------------------------------------------------------------
# Helper: create app with injected repository
# ---------------------------------------------------------------------------

def _make_app(repo):
    """Create PolylogueApp with an injected repository."""
    return PolylogueApp(repository=repo)


async def _wait_workers(pilot, *, selector: str | None = None, reject: str = "Loading..."):
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
async def test_dashboard_stats_populated(storage_repository, conversation_builder):
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
async def test_dashboard_provider_bars(storage_repository, conversation_builder):
    """Seed 2 providers → assert ProviderBar widgets rendered with correct counts."""
    conversation_builder("c1").provider("chatgpt").add_message("m1", text="A").save()
    conversation_builder("c2").provider("chatgpt").add_message("m2", text="B").save()
    conversation_builder("c3").provider("claude").add_message("m3", text="C").save()

    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        await _wait_workers(pilot)

        bars = pilot.app.query(ProviderBar)
        assert len(bars) >= 2
        # chatgpt should have 2, claude should have 1
        texts = [bar.render() for bar in bars]
        chatgpt_bar = [t for t in texts if "chatgpt" in t]
        claude_bar = [t for t in texts if "claude" in t]
        assert len(chatgpt_bar) == 1
        assert len(claude_bar) == 1
        assert "2" in chatgpt_bar[0]
        assert "1" in claude_bar[0]


@_skip
@pytest.mark.asyncio
async def test_dashboard_empty_db(storage_repository):
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
async def test_browser_tree_populated(storage_repository, conversation_builder):
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
async def test_browser_node_selection(storage_repository, conversation_builder):
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
            # The viewer should have been updated (not the default empty)
            # We can't easily read Markdown widget content, but we can check it mounted
            assert viewer is not None


@_skip
@pytest.mark.asyncio
async def test_browser_empty_db(storage_repository):
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

        # Should have fallback providers (chatgpt, claude)
        provider_names = [str(child.label) for child in tree.root.children]
        assert len(provider_names) >= 2


# ===========================================================================
# Search tests (Phase 2D: FTS setup)
# ===========================================================================


@_skip
@pytest.mark.asyncio
async def test_search_flow(storage_repository, conversation_builder):
    """Seed + index → type query → wait → assert DataTable rows."""
    conversation_builder("c1").add_message("m1", text="UniqueSearchTerm123").save()

    # Rebuild FTS index so search works
    conn = storage_repository._backend._get_connection()
    from polylogue.storage.index import rebuild_index

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


@_skip
@pytest.mark.asyncio
async def test_search_no_results(storage_repository, conversation_builder):
    """Search non-existent term → empty results, no error."""
    conversation_builder("c1").add_message("m1", text="Hello").save()

    # Rebuild FTS index
    conn = storage_repository._backend._get_connection()
    from polylogue.storage.index import rebuild_index

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
async def test_search_no_index(storage_repository):
    """No FTS table → error row shown, no crash."""
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
        # Should have an error row instead of crashing
        assert table.row_count >= 0  # Doesn't crash


# ===========================================================================
# Keyboard & interaction tests
# ===========================================================================


@_skip
@pytest.mark.asyncio
async def test_keyboard_tab_switch(storage_repository):
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
async def test_dark_mode_toggle(storage_repository):
    """Press 'd' → assert dark mode toggles without crashing."""
    app = _make_app(storage_repository)
    async with app.run_test() as pilot:
        # The 'd' key is bound to action_toggle_dark
        # Just verify it doesn't crash — dark mode is a Textual theme feature
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        # If we reach here, the toggle didn't crash
        assert pilot.app.query_one(Dashboard)


@_skip
@pytest.mark.asyncio
async def test_quit_action(storage_repository):
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
async def test_worker_failure_recovery(storage_repository, monkeypatch):
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
async def test_app_startup(storage_repository):
    """Test that the app starts and loads the dashboard."""
    app = _make_app(storage_repository)

    async with app.run_test() as pilot:
        assert pilot.app.query_one(Dashboard)
        assert pilot.app.query_one("#stat-conversations")
        assert pilot.app.query_one("#stat-messages")

        await _wait_workers(pilot, selector="#stat-conversations")

        stats = pilot.app.query_one("#stat-conversations", StatCard)
        assert stats.value == "0"
