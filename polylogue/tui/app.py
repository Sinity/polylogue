"""Textual-based TUI application for browsing conversations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, ListItem, ListView, Markdown, Static

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.lib.repository import ConversationRepository


class ConversationList(ListView):
    """Widget for displaying a list of conversations."""

    def __init__(self, repository: ConversationRepository, provider: str | None = None):
        """Initialize the conversation list.

        Args:
            repository: Conversation repository for data access
            provider: Optional provider filter
        """
        super().__init__()
        self.repository = repository
        self.provider = provider
        self.conversations: list[Conversation] = []
        self.border_title = "Conversations"

    def on_mount(self) -> None:
        """Load conversations when mounted."""
        self.load_conversations()

    def load_conversations(self, query: str | None = None) -> None:
        """Load conversations from repository.

        Args:
            query: Optional search query to filter conversations
        """
        self.clear()

        if query:
            # Search conversations
            self.conversations = self.repository.search(query)
            self.border_title = f"Search: {query}"
        else:
            # List all conversations (filtered by provider if set)
            self.conversations = self.repository.list(
                limit=100,
                offset=0,
                provider=self.provider,
            )
            if self.provider:
                self.border_title = f"Conversations ({self.provider})"
            else:
                self.border_title = "Conversations"

        # Populate list
        for conv in self.conversations:
            title = conv.title or conv.id[:8]
            provider = conv.provider
            updated_str = conv.updated_at or conv.created_at or ""

            # Format timestamp
            date_str = ""
            if updated_str and isinstance(updated_str, str):
                try:
                    dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, AttributeError):
                    date_str = updated_str[:16] if len(updated_str) > 16 else updated_str

            label = f"[bold]{title}[/bold]\n[dim]{provider} • {date_str}[/dim]"
            self.append(ListItem(Static(label)))


class ConversationViewer(Vertical):
    """Widget for displaying conversation content."""

    def __init__(self) -> None:
        """Initialize the conversation viewer."""
        super().__init__()
        self.conversation: Conversation | None = None
        self.border_title = "Conversation"

    def show_conversation(self, conversation: Conversation) -> None:
        """Display a conversation.

        Args:
            conversation: Conversation to display
        """
        self.conversation = conversation
        self.border_title = f"{conversation.title or conversation.id[:8]}"

        # Build markdown content
        lines = []
        lines.append(f"# {conversation.title or conversation.id}")
        lines.append("")
        lines.append(f"**Provider:** {conversation.provider}")
        lines.append(f"**ID:** {conversation.id}")
        if conversation.created_at:
            lines.append(f"**Created:** {conversation.created_at}")
        if conversation.updated_at:
            lines.append(f"**Updated:** {conversation.updated_at}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Get all messages (substantive_only returns an iterator of messages)
        for msg in conversation.messages:
            # Skip non-substantive messages
            if not msg.is_substantive:
                continue

            role = msg.role.upper()
            lines.append(f"## {role}")
            if msg.timestamp:
                lines.append(f"*{msg.timestamp}*")
            lines.append("")
            lines.append(msg.text or "")
            lines.append("")

        content = "\n".join(lines)

        # Update viewer
        self.remove_children()
        self.mount(Markdown(content, id="conversation-markdown"))

    def show_empty(self) -> None:
        """Show empty state."""
        self.border_title = "Conversation"
        self.remove_children()
        self.mount(Static("[dim]Select a conversation to view[/dim]", id="empty-state"))


class ConversationBrowser(App[Any]):
    """TUI application for browsing AI conversations.

    Keyboard shortcuts:
        j/k: Navigate list
        /: Search conversations
        Enter: View selected conversation
        q: Quit
    """

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-rows: 1fr auto;
    }

    #main-container {
        column-span: 2;
        height: 100%;
    }

    #conversation-list {
        width: 40%;
        border: solid $primary;
    }

    #conversation-viewer {
        width: 60%;
        border: solid $secondary;
    }

    #conversation-markdown {
        height: 100%;
        overflow-y: auto;
    }

    #empty-state {
        height: 100%;
        content-align: center middle;
    }

    Footer {
        column-span: 2;
    }
    """

    BINDINGS = [
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("/", "search", "Search"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        repository: ConversationRepository,
        provider: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the browser.

        Args:
            repository: Conversation repository for data access
            provider: Optional provider filter
            **kwargs: Additional keyword arguments for App
        """
        super().__init__(**kwargs)
        self.repository = repository
        self.provider = provider
        self.conversation_list: ConversationList | None = None
        self.conversation_viewer: ConversationViewer | None = None

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            self.conversation_list = ConversationList(self.repository, self.provider)
            self.conversation_list.id = "conversation-list"
            yield self.conversation_list

            self.conversation_viewer = ConversationViewer()
            self.conversation_viewer.id = "conversation-viewer"
            yield self.conversation_viewer

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        if self.conversation_viewer:
            self.conversation_viewer.show_empty()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle conversation selection.

        Args:
            event: Selection event
        """
        if not self.conversation_list or not self.conversation_viewer:
            return

        # Get selected conversation
        index = event.list_view.index
        if index is not None and 0 <= index < len(self.conversation_list.conversations):
            conversation = self.conversation_list.conversations[index]
            self.conversation_viewer.show_conversation(conversation)

    def action_cursor_down(self) -> None:
        """Move cursor down in list."""
        if self.conversation_list:
            self.conversation_list.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up in list."""
        if self.conversation_list:
            self.conversation_list.action_cursor_up()

    def action_search(self) -> None:
        """Open search input."""
        # For simplicity, this is a placeholder
        # A full implementation would show an input modal
        self.notify("Search: Type / to search (not yet implemented)")


def run_browser(
    db_path: Path | None = None,
    provider: str | None = None,
) -> None:
    """Run the conversation browser TUI.

    Args:
        db_path: Optional path to database file
        provider: Optional provider filter
    """
    from polylogue.lib.repository import ConversationRepository
    from polylogue.storage.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)
    app = ConversationBrowser(repository=repository, provider=provider)
    app.run()


__all__ = ["ConversationBrowser", "run_browser"]
