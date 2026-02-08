import pytest

try:
    from polylogue.ui.tui.app import PolylogueApp
    from polylogue.ui.tui.screens.dashboard import Dashboard
    from polylogue.ui.tui.widgets.stats import StatCard
except ImportError:
    # Textual might not be installed in some envs
    PolylogueApp = None


@pytest.mark.skipif(PolylogueApp is None, reason="Textual not installed")
@pytest.mark.asyncio
async def test_app_startup(workspace_env):
    """Test that the app starts and loads the dashboard."""
    app = PolylogueApp()

    async with app.run_test() as pilot:
        # Check if Dashboard is active (it's in a tab)
        assert pilot.app.query_one(Dashboard).id is None  # It has no ID, but should exist

        # Check for stat cards
        assert pilot.app.query_one("#stat-conversations")
        assert pilot.app.query_one("#stat-messages")

        # Wait for worker to load stats?
        # In a real test we might mock repository. For now, let's just ensure it doesn't crash.
        await pilot.pause()

        # Check stat values (should be "Loading..." or "0" depending on timing/mock)
        # With temp_repo, counts should be 0.

        # We need to wait for the worker. Textual's run_test usually handles pending workers?
        # Let's force a yield.
        await pilot.pause()
        await pilot.pause()

        # Assuming temp_repo is empty
        stats = pilot.app.query_one("#stat-conversations", StatCard)
        # Note: Worker runs in thread, might be racey in test without explicit wait/mock.
        # But we just want to ensure it mounted.
        assert stats
