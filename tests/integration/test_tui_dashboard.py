"""Headless TUI launch smokes."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import pytest
from textual.pilot import Pilot
from textual.widgets import TabbedContent

from polylogue.ui.tui.app import PolylogueApp
from polylogue.ui.tui.widgets.stats import StatCard

if TYPE_CHECKING:
    from polylogue.storage.repository import SessionRepository
    from tests.infra.storage_records import SessionBuilder

SessionBuilderFactory = Callable[[str], "SessionBuilder"]
DashboardAutopilot = Callable[[Pilot[object]], Coroutine[Any, Any, None]]


def _dashboard_autopilot(repository: SessionRepository) -> DashboardAutopilot:
    async def autopilot(pilot: Pilot[object]) -> None:
        try:
            await pilot.pause()
            tabs = pilot.app.query_one(TabbedContent)
            assert tabs.active == "dashboard"

            await pilot.app.workers.wait_for_complete()
            for _ in range(20):
                await pilot.pause()
                stat = pilot.app.query_one("#stat-sessions", StatCard)
                if stat.value != "Loading...":
                    break

            assert pilot.app.query_one("#stat-sessions", StatCard).value == "1"
            assert pilot.app.query_one("#stat-messages", StatCard).value == "1"
        finally:
            await repository.backend.close()
            pilot.app.exit()

    return autopilot


@pytest.mark.tui
def test_dashboard_app_launches_headless_with_real_repository(
    storage_repository: SessionRepository,
    session_builder: SessionBuilderFactory,
) -> None:
    session_builder("tui-smoke-1").provider("chatgpt").title("Dashboard Smoke").add_message(
        "m1",
        text="Dashboard launch smoke",
    ).save()

    from polylogue.api import Polylogue

    facade = Polylogue()
    app = PolylogueApp(polylogue=facade)

    app.run(headless=True, size=(100, 30), auto_pilot=_dashboard_autopilot(storage_repository))
