"""Enhanced dashboard screen with embedding stats and provider breakdown."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Grid
from textual.widgets import Static

from polylogue.config import Config
from polylogue.lib.log import get_logger
from polylogue.ui.tui.widgets.stats import StatCard

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


class ProviderBar(Static):
    """A horizontal bar showing provider distribution using Rich Text.

    Static widgets should render Rich content directly, not compose child widgets.
    This avoids NoActiveAppError when mounted dynamically.
    """

    DEFAULT_CSS = """
    ProviderBar {
        width: 100%;
        height: 1;
        margin-bottom: 0;
    }
    """

    def __init__(self, provider: str, count: int, max_count: int) -> None:
        super().__init__()
        self.provider = provider
        self.count = count
        self.max_count = max_count

    def render(self) -> str:
        """Render the provider bar as a simple text line."""
        pct = (self.count / self.max_count * 100) if self.max_count > 0 else 0
        bar_width = 20
        filled = int(pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        # Format: "provider_name    ████████░░░░░░░░░░░░  123"
        return f"{self.provider[:16]:<16} {bar} {self.count:>6}"


class Dashboard(Container):
    """Enhanced dashboard widget with embedding stats and provider breakdown."""

    DEFAULT_CSS = """
    Dashboard {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    Dashboard > #stats-grid {
        grid-size: 5;
        grid-gutter: 1;
        height: auto;
        margin-bottom: 2;
    }

    Dashboard > #providers-section {
        height: auto;
        border: solid $accent;
        padding: 1;
        margin-top: 1;
    }

    Dashboard > #providers-section > .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        config: Config | None = None,
        repository: ConversationRepository | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._repository = repository

    def _get_repo(self) -> ConversationRepository:
        """Get the repository, falling back to the service singleton."""
        if self._repository is not None:
            return self._repository
        from polylogue.services import get_repository
        return get_repository()

    def compose(self) -> ComposeResult:
        # Grid for stats
        with Grid(id="stats-grid"):
            yield StatCard("Conversations", "Loading...", id="stat-conversations")
            yield StatCard("Messages", "Loading...", id="stat-messages")
            yield StatCard("Attachments", "Loading...", id="stat-attachments")
            yield StatCard("Embeddings", "Loading...", id="stat-embeddings")
            yield StatCard("Coverage", "Loading...", id="stat-coverage")

        # Provider breakdown section
        with Container(id="providers-section"):
            yield Static("By Provider", classes="section-title")
            yield Container(id="provider-bars")

    def on_mount(self) -> None:
        """Load data when screen mounts."""
        self.run_worker(self._fetch_stats())

    async def _fetch_stats(self) -> None:
        """Fetch stats asynchronously, then update DOM."""
        try:
            repo = self._get_repo()
            stats = await repo.get_archive_stats()
        except Exception as e:
            self.notify(f"Failed to load stats: {e}", severity="error")
            return

        self._apply_stats(stats)

    def _apply_stats(self, stats) -> None:
        """Apply fetched stats to DOM widgets (runs on main thread)."""

        self.query_one("#stat-conversations", StatCard).value = str(stats.total_conversations)
        self.query_one("#stat-messages", StatCard).value = str(stats.total_messages)
        self.query_one("#stat-attachments", StatCard).value = str(stats.total_attachments)
        self.query_one("#stat-embeddings", StatCard).value = str(stats.embedded_messages)
        self.query_one("#stat-coverage", StatCard).value = f"{stats.embedding_coverage:.1f}%"

        # Mount provider bars
        bars_container = self.query_one("#provider-bars", Container)
        sorted_providers = sorted(stats.providers.items(), key=lambda x: x[1], reverse=True)
        max_count = sorted_providers[0][1] if sorted_providers else 1

        for provider, count in sorted_providers[:10]:
            bar = ProviderBar(provider or "unknown", count, max_count)
            bars_container.mount(bar)
