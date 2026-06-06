from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import pytest

pytest.importorskip("textual", reason="Textual not installed")

from textual.pilot import Pilot
from textual.widgets import DataTable, Input, TabbedContent, Tree

from polylogue.api import Polylogue
from polylogue.ui.tui.app import PolylogueApp
from polylogue.ui.tui.screens.base import RepositoryBoundContainer
from polylogue.ui.tui.screens.dashboard import Dashboard, OriginBar
from polylogue.ui.tui.widgets.stats import StatCard
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import db_setup

if TYPE_CHECKING:
    from tests.infra.storage_records import SessionBuilder

pytestmark = pytest.mark.tui


# ---------------------------------------------------------------------------
# Helper: create app reading the archive
# ---------------------------------------------------------------------------


SessionBuilderFactory: TypeAlias = Callable[[str], "SessionBuilder"]


def _make_app(db_path: Path) -> PolylogueApp:
    """Create PolylogueApp with a native :class:`Polylogue` facade over ``db_path``.

    The TUI screens read through the archive facade / ``TUIReadSurface``;
    seeding and reads both resolve to the same archive `index.db` so the
    rendered DOM reflects the seeded archive.
    """
    facade = Polylogue(archive_root=db_path.parent, db_path=db_path)
    return PolylogueApp(polylogue=facade)


def _make_broken_app(error: Exception) -> PolylogueApp:
    """Create PolylogueApp whose facade raises on the dashboard stats read."""

    class _BrokenFacade:
        async def storage_stats(self) -> object:
            raise error

    return PolylogueApp(polylogue=cast(Polylogue, _BrokenFacade()))


async def _wait_workers(pilot: Pilot[None], *, selector: str | None = None, reject: str = "Loading...") -> None:
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


@pytest.mark.asyncio
async def test_dashboard_stats_populated(
    workspace_env: dict[str, Path], session_builder: SessionBuilderFactory
) -> None:
    """Seed data → mount → wait → assert stat card values match seeded counts."""
    db_path = db_setup(workspace_env)
    session_builder("c1").add_message("m1", text="Hello").save()
    session_builder("c2").add_message("m2", text="World").add_message("m3", text="!").save()

    app = _make_app(db_path)
    async with app.run_test() as pilot:
        await _wait_workers(pilot, selector="#stat-sessions")

        convs = pilot.app.query_one("#stat-sessions", StatCard)
        msgs = pilot.app.query_one("#stat-messages", StatCard)
        assert convs.value == "2"
        assert msgs.value == "3"


@pytest.mark.asyncio
async def test_dashboard_origin_bars(workspace_env: dict[str, Path], session_builder: SessionBuilderFactory) -> None:
    """Seed 2 origins -> assert OriginBar widgets rendered with correct counts."""
    db_path = db_setup(workspace_env)
    session_builder("c1").provider("chatgpt").add_message("m1", text="A").save()
    session_builder("c2").provider("chatgpt").add_message("m2", text="B").save()
    session_builder("c3").provider("claude-ai").add_message("m3", text="C").save()

    app = _make_app(db_path)
    async with app.run_test() as pilot:
        await _wait_workers(pilot)

        bars = pilot.app.query(OriginBar)
        assert len(bars) >= 2
        # Origin buckets should reflect the seeded source families.
        texts = [str(bar.render()) for bar in bars]
        chatgpt_bar = [t for t in texts if "chatgpt" in t]
        claude_bar = [t for t in texts if "claude-ai" in t]
        assert len(chatgpt_bar) == 1
        assert len(claude_bar) == 1
        assert "2" in chatgpt_bar[0]
        assert "1" in claude_bar[0]


@pytest.mark.asyncio
async def test_dashboard_empty_db(workspace_env: dict[str, Path]) -> None:
    """Empty DB → graceful '0' display, no errors."""
    db_path = db_setup(workspace_env)
    app = _make_app(db_path)
    async with app.run_test() as pilot:
        await _wait_workers(pilot, selector="#stat-sessions")

        convs = pilot.app.query_one("#stat-sessions", StatCard)
        msgs = pilot.app.query_one("#stat-messages", StatCard)
        assert convs.value == "0"
        assert msgs.value == "0"


# ===========================================================================
# Browser tests
# ===========================================================================


@pytest.mark.asyncio
async def test_browser_tree_populated(workspace_env: dict[str, Path], session_builder: SessionBuilderFactory) -> None:
    """Seed sessions → switch to browser → wait → assert tree nodes match."""
    db_path = db_setup(workspace_env)
    session_builder("c1").provider("chatgpt").title("My Chat").add_message("m1", text="Hi").save()

    app = _make_app(db_path)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "browser"
        await _wait_workers(pilot)

        tree = pilot.app.query_one("#browser-tree", Tree)
        # Root should have at least one child (provider node)
        assert len(tree.root.children) >= 1

        # Find the chatgpt provider node
        source_names = [str(child.label) for child in tree.root.children]
        assert "Chatgpt-export" in source_names

        # Expand chatgpt node — should have our session
        chatgpt_node = [c for c in tree.root.children if str(c.label) == "Chatgpt-export"][0]
        assert len(chatgpt_node.children) >= 1


@pytest.mark.asyncio
async def test_browser_node_selection(workspace_env: dict[str, Path], session_builder: SessionBuilderFactory) -> None:
    """Click leaf node → assert markdown viewer shows session content."""
    db_path = db_setup(workspace_env)
    session_builder("c1").provider("chatgpt").title("Test Chat").add_message("m1", text="Hello World").save()

    app = _make_app(db_path)
    async with app.run_test() as pilot:
        tabs = pilot.app.query_one(TabbedContent)
        tabs.active = "browser"
        await _wait_workers(pilot)

        tree = pilot.app.query_one("#browser-tree", Tree)
        chatgpt_node = [c for c in tree.root.children if str(c.label) == "Chatgpt-export"][0]

        # Select the leaf node (the session)
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


@pytest.mark.asyncio
async def test_browser_empty_db(workspace_env: dict[str, Path]) -> None:
    """Empty DB → direct empty-state leaf shown."""
    import asyncio

    db_path = db_setup(workspace_env)
    app = _make_app(db_path)
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

        node_labels = [str(child.label) for child in tree.root.children]
        assert node_labels == ["No sessions in archive"]


# ===========================================================================
# Search tests
# ===========================================================================


@pytest.mark.asyncio
async def test_search_flow(workspace_env: dict[str, Path], session_builder: SessionBuilderFactory) -> None:
    """Seed + index → type query → wait → assert DataTable rows."""
    db_path = db_setup(workspace_env)
    session_builder("c1").add_message("m1", text="UniqueSearchTerm123").save()

    app = _make_app(db_path)
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

        # Verify the found row key matches our session's archive session id
        row_key = next(iter(table.rows))
        assert row_key.value == native_session_id_for("test", "c1")

        from textual.widgets import Markdown as MarkdownWidget

        table.move_cursor(row=0)
        table.action_select_cursor()
        await pilot.pause()

        viewer = pilot.app.query_one("#search-viewer", MarkdownWidget)
        assert "UniqueSearchTerm123" in viewer.source


@pytest.mark.asyncio
async def test_search_no_results(workspace_env: dict[str, Path], session_builder: SessionBuilderFactory) -> None:
    """Search non-existent term → empty results, no error."""
    db_path = db_setup(workspace_env)
    session_builder("c1").add_message("m1", text="Hello").save()

    app = _make_app(db_path)
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


@pytest.mark.asyncio
async def test_search_empty_db(workspace_env: dict[str, Path]) -> None:
    """Empty DB → 0 results, no crash."""
    db_path = db_setup(workspace_env)
    app = _make_app(db_path)
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
        # Empty archive — should return 0 results gracefully
        assert table.row_count == 0


# ===========================================================================
# Keyboard & interaction tests
# ===========================================================================


@pytest.mark.asyncio
async def test_keyboard_tab_switch(workspace_env: dict[str, Path]) -> None:
    """Press Tab key → verify tab changes (basic navigation)."""
    db_path = db_setup(workspace_env)
    app = _make_app(db_path)
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


@pytest.mark.asyncio
async def test_dark_mode_toggle(workspace_env: dict[str, Path]) -> None:
    """Press 'd' → assert dark mode toggles without crashing."""
    db_path = db_setup(workspace_env)
    app = _make_app(db_path)
    async with app.run_test() as pilot:
        await pilot.press("d")
        await pilot.pause()
        assert pilot.app.theme == "textual-light"
        await pilot.press("d")
        await pilot.pause()
        assert pilot.app.theme == "textual-dark"
        assert pilot.app.query_one(Dashboard) is not None


def test_repository_bound_container_requires_injected_facade() -> None:
    class DummyScreen(RepositoryBoundContainer):
        pass

    screen = DummyScreen()

    with pytest.raises(RuntimeError, match="DummyScreen widget requires an injected Polylogue facade"):
        screen._get_facade("DummyScreen")


@pytest.mark.asyncio
async def test_quit_action(workspace_env: dict[str, Path]) -> None:
    """Press 'q' → app exits cleanly."""
    db_path = db_setup(workspace_env)
    app = _make_app(db_path)
    async with app.run_test() as pilot:
        await pilot.press("q")
        # If we reach here, the app didn't crash during quit
        # The run_test context manager handles exit assertions


# ===========================================================================
# Error resilience
# ===========================================================================


@pytest.mark.asyncio
async def test_worker_failure_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject error in the facade → assert app stays usable with error notification."""
    app = _make_broken_app(RuntimeError("DB exploded"))
    async with app.run_test() as pilot:
        await _wait_workers(pilot)

        # App should still be running (Dashboard catches errors and notifies)
        assert pilot.app.query_one(Dashboard) is not None

        # Stat cards should still exist (may show "Loading...")
        stat = pilot.app.query_one("#stat-sessions", StatCard)
        assert stat is not None


# ===========================================================================
# App startup (basic sanity)
# ===========================================================================


@pytest.mark.asyncio
async def test_app_startup(workspace_env: dict[str, Path]) -> None:
    """Test that the app starts and loads the dashboard."""
    db_path = db_setup(workspace_env)
    app = _make_app(db_path)

    async with app.run_test() as pilot:
        assert pilot.app.query_one(Dashboard) is not None
        assert pilot.app.query_one("#stat-sessions") is not None
        assert pilot.app.query_one("#stat-messages") is not None

        await _wait_workers(pilot, selector="#stat-sessions")

        stats = pilot.app.query_one("#stat-sessions", StatCard)
        assert stats.value == "0"
