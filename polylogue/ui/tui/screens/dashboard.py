"""Enhanced dashboard screen with embedding stats and origin breakdown.

Routes stats through the archive ``Polylogue.storage_stats()`` facade
method instead of calling the repository directly.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Grid
from textual.widgets import Static

from polylogue.archive.stats import ArchiveStats as StorageArchiveStats
from polylogue.logging import get_logger
from polylogue.ui.tui.screens.base import RepositoryBoundContainer
from polylogue.ui.tui.widgets.stats import StatCard

logger = get_logger(__name__)


class OriginBar(Static):
    """A horizontal bar showing origin distribution using Rich Text.

    Static widgets should render Rich content directly, not compose child widgets.
    This avoids NoActiveAppError when mounted dynamically.
    """

    DEFAULT_CSS = """
    OriginBar {
        width: 100%;
        height: 1;
        margin-bottom: 0;
    }
    """

    def __init__(self, origin: str, count: int, max_count: int) -> None:
        super().__init__()
        self.origin = origin
        self.count = count
        self.max_count = max_count

    def render(self) -> str:
        """Render the origin bar as a simple text line."""
        pct = (self.count / self.max_count * 100) if self.max_count > 0 else 0
        bar_width = 20
        filled = int(pct / 100 * bar_width)
        bar = chr(0x2588) * filled + chr(0x2591) * (bar_width - filled)
        # Format: "source_name    [bar]  123"
        return f"{self.origin[:16]:<16} {bar} {self.count:>6}"


class Dashboard(RepositoryBoundContainer):
    """Enhanced dashboard widget with embedding stats and origin breakdown."""

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

    Dashboard > #origins-section {
        height: auto;
        border: solid $accent;
        padding: 1;
        margin-top: 1;
    }

    Dashboard > #origins-section > .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        # Grid for stats
        with Grid(id="stats-grid"):
            yield StatCard("Sessions", "Loading...", id="stat-sessions")
            yield StatCard("Messages", "Loading...", id="stat-messages")
            yield StatCard("Attachments", "Loading...", id="stat-attachments")
            yield StatCard("Embeddings", "Loading...", id="stat-embeddings")
            yield StatCard("Coverage", "Loading...", id="stat-coverage")

        # Origin breakdown section
        with Container(id="origins-section"):
            yield Static("By Origin", classes="section-title")
            yield Container(id="origin-bars")

    def on_mount(self) -> None:
        """Load data when screen mounts."""
        self.run_worker(self._fetch_stats())

    async def _fetch_stats(self) -> None:
        """Fetch stats asynchronously, then update DOM."""
        try:
            facade = self._get_facade("Dashboard")
            stats = await facade.storage_stats()
        except Exception as e:
            self.notify(f"Failed to load stats: {e}", severity="error")
            return

        self._apply_stats(stats)

    def _apply_stats(self, stats: StorageArchiveStats) -> None:
        """Apply fetched stats to DOM widgets (runs on main thread)."""

        self.query_one("#stat-sessions", StatCard).value = str(stats.total_sessions)
        self.query_one("#stat-messages", StatCard).value = str(stats.total_messages)
        self.query_one("#stat-attachments", StatCard).value = str(stats.total_attachments)
        self.query_one("#stat-embeddings", StatCard).value = str(stats.embedded_messages)
        self.query_one("#stat-coverage", StatCard).value = f"{stats.embedding_coverage:.1f}%"

        # Mount origin bars
        bars_container = self.query_one("#origin-bars", Container)
        sorted_origins = sorted(stats.origins.items(), key=lambda x: x[1], reverse=True)
        max_count = sorted_origins[0][1] if sorted_origins else 1

        for origin, count in sorted_origins[:10]:
            bar = OriginBar(origin or "unknown", count, max_count)
            _ = bars_container.mount(bar)
