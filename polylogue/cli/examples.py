"""Shared CLI command examples."""

from __future__ import annotations

from typing import Dict, List, Tuple

# Command Examples for --examples flag
# Each command maps to list of (description, command_line) tuples
COMMAND_EXAMPLES: Dict[str, List[Tuple[str, str]]] = {
    "render": [
        ("Render a single export", "polylogue render export.json --out ~/polylogue-data/render"),
        ("Render with HTML previews", "polylogue render export.json --html on"),
        ("Render multiple exports", "polylogue render exports/ --diff"),
    ],
    "sync": [
        ("Sync all Drive chats", "polylogue sync drive --all"),
        ("Sync from specific Drive folder", "polylogue sync drive --folder-name 'Work Chats'"),
        ("Sync Codex sessions with preview", "polylogue sync codex --dry-run"),
        ("Sync Claude Code with diff tracking", "polylogue sync claude-code --diff"),
        ("Sync and prune deleted chats", "polylogue sync drive --all --prune"),
        ("Watch Codex and auto-sync", "polylogue sync codex --watch"),
        ("Watch Claude Code with HTML", "polylogue sync claude-code --watch --html on"),
    ],
    "import": [
        ("Import ChatGPT export", "polylogue import chatgpt export.zip --html on"),
        ("Import with interactive picker", "polylogue import claude-code pick"),
        ("Import specific conversation", "polylogue import chatgpt export.zip --conversation-id abc123"),
        ("Import all from Claude export", "polylogue import claude conversations.zip --all"),
        ("Preview import without writing", "polylogue import codex session.jsonl --dry-run"),
    ],
    "search": [
        ("Search for term", "polylogue search 'error handling'"),
        ("Search with filters", "polylogue search 'API' --provider chatgpt --since 2024-01-01"),
        ("Search with limit", "polylogue search 'authentication' --limit 10"),
        ("Search and open anchored result", "polylogue search 'TODO' --limit 1 --open"),
        ("Search with attachment filter", "polylogue search 'diagram' --with-attachments"),
    ],
    "browse": [
        ("Browse branch graph", "polylogue browse branches --provider claude"),
        ("View stats", "polylogue browse stats --provider chatgpt"),
        ("Get stats with time filter", "polylogue browse stats --since 2024-01-01 --until 2024-12-31"),
        ("Show status overview", "polylogue browse status"),
        ("Watch status continuously", "polylogue browse status --watch"),
        ("List recent runs", "polylogue browse runs --limit 20"),
    ],
    "maintain": [
        ("Preview prune operation", "polylogue maintain prune --dry-run"),
        ("Prune legacy files", "polylogue maintain prune --dir ~/polylogue-data/chatgpt"),
        ("Check all providers", "polylogue maintain doctor"),
        ("Check with JSON output", "polylogue maintain doctor --json"),
        ("Validate indexes", "polylogue maintain index check"),
        ("Repair indexes", "polylogue maintain index check --repair"),
        ("Restore a snapshot", "polylogue maintain restore --from /tmp/snap --to ~/.local/share/polylogue/archive --force"),
    ],
    "config": [
        ("Interactive setup wizard", "polylogue config init"),
        ("Force re-initialization", "polylogue config init --force"),
        ("Show current configuration", "polylogue config show"),
        ("Get configuration as JSON", "polylogue config show --json"),
        ("Enable HTML previews", "polylogue config set --html on"),
        ("Set dark theme", "polylogue config set --theme dark"),
        ("Reset to defaults", "polylogue config set --reset"),
    ],
    "compare": [
        ("Compare coverage between two providers", "polylogue compare 'auth error' --provider-a codex --provider-b claude-code --limit 10"),
    ],
    "attachments": [
        ("Show top attachments by size", "polylogue attachments stats --sort size --limit 5"),
        ("Read attachment metadata from the index", "polylogue attachments stats --from-index --json"),
        ("Extract PDFs to a folder", "polylogue attachments extract --ext .pdf --out ~/desk/pdfs"),
    ],
    "status": [
        ("Watch status with JSON output", "polylogue status --json --watch --interval 10"),
        ("Dump recent runs quietly", "polylogue status --dump - --runs-limit 50 --quiet"),
        ("Summarize a single provider", "polylogue status --providers codex --summary -"),
    ],
    "prefs": [
        ("Persist a search default", "polylogue prefs set search --flag --limit --value 5"),
        ("List saved preferences", "polylogue prefs list"),
        ("Clear all saved defaults", "polylogue prefs clear"),
    ],
    "help": [
        ("Show all examples", "polylogue help --examples"),
        ("View examples for a single command", "polylogue help search --examples"),
    ],
}

__all__ = ["COMMAND_EXAMPLES"]
