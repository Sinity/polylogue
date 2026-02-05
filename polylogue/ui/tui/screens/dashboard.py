"""Enhanced dashboard screen with embedding stats and provider breakdown."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Grid
from textual.widgets import Static

from polylogue.config import Config
from polylogue.ui.tui.widgets.stats import StatCard


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

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config

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
        self.run_worker(self.load_stats(), thread=True)

    async def load_stats(self) -> None:
        """Fetch statistics from the repository."""
        from polylogue.storage.backends.sqlite import connection_context

        try:
            with connection_context(None) as conn:
                # Count conversations
                conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

                # Count messages
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

                # Count attachments and size
                att_row = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM attachments").fetchone()
                att_count = att_row[0]

                # Get provider breakdown
                provider_rows = conn.execute("""
                    SELECT provider_name, COUNT(*) as count
                    FROM conversations
                    GROUP BY provider_name
                    ORDER BY count DESC
                """).fetchall()
                providers = [(row["provider_name"], row["count"]) for row in provider_rows]

                # Get embedding stats
                embedded_msgs = 0
                embedded_convs = 0
                try:
                    embedded_msgs = conn.execute(
                        "SELECT COUNT(*) FROM message_embeddings"
                    ).fetchone()[0]
                    embedded_convs = conn.execute(
                        "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
                    ).fetchone()[0]
                except Exception:
                    pass

            # Update widgets
            self.query_one("#stat-conversations", StatCard).value = str(conv_count)
            self.query_one("#stat-messages", StatCard).value = str(msg_count)
            self.query_one("#stat-attachments", StatCard).value = str(att_count)
            self.query_one("#stat-embeddings", StatCard).value = str(embedded_msgs)

            # Calculate coverage
            coverage = (embedded_convs / conv_count * 100) if conv_count > 0 else 0
            self.query_one("#stat-coverage", StatCard).value = f"{coverage:.1f}%"

            # Update provider bars
            bars_container = self.query_one("#provider-bars", Container)
            max_count = providers[0][1] if providers else 1

            for provider, count in providers[:10]:  # Top 10 providers
                bar = ProviderBar(provider or "unknown", count, max_count)
                bars_container.mount(bar)

        except Exception as e:
            self.notify(f"Failed to load stats: {e}", severity="error")
