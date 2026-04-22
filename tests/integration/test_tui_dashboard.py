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
    from polylogue.storage.repository import ConversationRepository
    from tests.infra.storage_records import ConversationBuilder

ConversationBuilderFactory = Callable[[str], "ConversationBuilder"]
DashboardAutopilot = Callable[[Pilot[object]], Coroutine[Any, Any, None]]


def _dashboard_autopilot(repository: ConversationRepository) -> DashboardAutopilot:
    async def autopilot(pilot: Pilot[object]) -> None:
        try:
            await pilot.pause()
            tabs = pilot.app.query_one(TabbedContent)
            assert tabs.active == "dashboard"

            await pilot.app.workers.wait_for_complete()
            for _ in range(20):
                await pilot.pause()
                stat = pilot.app.query_one("#stat-conversations", StatCard)
                if stat.value != "Loading...":
                    break

            assert pilot.app.query_one("#stat-conversations", StatCard).value == "1"
            assert pilot.app.query_one("#stat-messages", StatCard).value == "1"
        finally:
            await repository.backend.close()
            pilot.app.exit()

    return autopilot


@pytest.mark.tui
def test_dashboard_app_launches_headless_with_real_repository(
    storage_repository: ConversationRepository,
    conversation_builder: ConversationBuilderFactory,
) -> None:
    conversation_builder("tui-smoke-1").provider("chatgpt").title("Dashboard Smoke").add_message(
        "m1",
        text="Dashboard launch smoke",
    ).save()

    app = PolylogueApp(repository=storage_repository)

    app.run(headless=True, size=(100, 30), auto_pilot=_dashboard_autopilot(storage_repository))
