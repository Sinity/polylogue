from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Grid

from polylogue.config import Config
from polylogue.ui.tui.widgets.stats import StatCard


class Dashboard(Container):
    """Dashboard widget showing high-level stats."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        # Grid for stats
        with Grid(id="stats-grid"):
            # We'll populate these with real data later
            yield StatCard("Conversations", "Loading...", id="stat-conversations")
            yield StatCard("Messages", "Loading...", id="stat-messages")
            yield StatCard("Attachments", "Loading...", id="stat-attachments")
            yield StatCard("Storage Usage", "Loading...", id="stat-storage")

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
                att_val = att_row[1]
                att_size = att_val if att_val is not None else 0

            # Update widgets
            self.query_one("#stat-conversations", StatCard).value = str(conv_count)
            self.query_one("#stat-messages", StatCard).value = str(msg_count)
            self.query_one("#stat-attachments", StatCard).value = str(att_count)

            # Format size
            size_gb = att_size / (1024 * 1024 * 1024)
            size_mb = att_size / (1024 * 1024)
            size_str = f"{size_gb:.2f} GB" if size_gb >= 1 else f"{size_mb:.2f} MB"
            self.query_one("#stat-storage", StatCard).value = size_str

        except Exception as e:
            self.notify(f"Failed to load stats: {e}", severity="error")
