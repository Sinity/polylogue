"""View command - semantic projection viewer for conversations."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import click

from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.config import ConfigError
from polylogue.lib.models import Conversation, DialoguePair, Message
from polylogue.lib.repository import ConversationRepository

# --- Projection types ---

PROJECTIONS = {
    "full": "All messages (no filtering)",
    "dialogue": "User/assistant dialogue only (no system/tool)",
    "clean": "Substantive dialogue (no noise, context dumps)",
    "pairs": "User-assistant turn pairs",
    "user": "User messages only",
    "assistant": "Assistant messages only",
    "thinking": "Thinking/reasoning traces only",
    "stats": "Statistics summary only",
}


def _apply_projection(convo: Conversation, projection: str) -> Conversation | list[DialoguePair] | list[str] | dict[str, Any]:
    """Apply semantic projection to conversation."""
    if projection == "full":
        return convo
    elif projection == "dialogue":
        return convo.dialogue_only()
    elif projection == "clean":
        return convo.substantive_only()
    elif projection == "pairs":
        return list(convo.iter_pairs())
    elif projection == "user":
        return convo.user_only()
    elif projection == "assistant":
        return convo.assistant_only()
    elif projection == "thinking":
        return list(convo.iter_thinking())
    elif projection == "stats":
        return {
            "id": convo.id,
            "title": convo.title,
            "provider": convo.provider,
            "message_count": convo.message_count,
            "user_messages": convo.user_message_count,
            "assistant_messages": convo.assistant_message_count,
            "word_count": convo.word_count,
            "substantive_words": convo.substantive_word_count,
            "cost_usd": convo.total_cost_usd,
            "duration_ms": convo.total_duration_ms,
            "created_at": convo.created_at.isoformat() if convo.created_at else None,
            "updated_at": convo.updated_at.isoformat() if convo.updated_at else None,
        }
    return convo


# --- Output formatters ---


def _format_message_text(msg: Message, include_role: bool = True, include_meta: bool = False) -> str:
    """Format a single message for text output."""
    lines = []
    if include_role:
        role_prefix = f"[{msg.role}]"
        if include_meta and msg.timestamp:
            role_prefix += f" ({msg.timestamp.isoformat()})"
        lines.append(role_prefix)
    if msg.text:
        lines.append(msg.text)
    return "\n".join(lines)


def _format_pair_text(pair: DialoguePair) -> str:
    """Format a dialogue pair for text output."""
    return f"User: {pair.user.text}\n\nAssistant: {pair.assistant.text}"


def _render_text(
    data: Conversation | list[DialoguePair] | list[str] | dict[str, Any],
    include_role: bool = True,
    include_meta: bool = False,
) -> Iterator[str]:
    """Render projection data as text lines."""
    if isinstance(data, dict):
        # Stats output
        for key, value in data.items():
            yield f"{key}: {value}"
    elif isinstance(data, Conversation):
        for msg in data.messages:
            yield _format_message_text(msg, include_role, include_meta)
            yield ""  # blank line between messages
    elif isinstance(data, list):
        if data and isinstance(data[0], DialoguePair):
            for i, pair in enumerate(data, 1):
                yield f"--- Turn {i} ---"
                yield _format_pair_text(pair)  # type: ignore[arg-type]
                yield ""
        elif data and isinstance(data[0], str):
            # Thinking traces
            for i, trace in enumerate(data, 1):
                yield f"--- Thinking {i} ---"
                yield trace  # type: ignore[misc]
                yield ""


def _serialize_message(msg: Message) -> dict[str, Any]:
    """Serialize message for JSON output."""
    return {
        "id": msg.id,
        "role": msg.role,
        "text": msg.text,
        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
        "is_user": msg.is_user,
        "is_assistant": msg.is_assistant,
        "is_tool_use": msg.is_tool_use,
        "is_thinking": msg.is_thinking,
        "is_substantive": msg.is_substantive,
        "word_count": msg.word_count,
        "cost_usd": msg.cost_usd,
        "duration_ms": msg.duration_ms,
    }


def _serialize_pair(pair: DialoguePair) -> dict[str, Any]:
    """Serialize dialogue pair for JSON output."""
    return {
        "user": _serialize_message(pair.user),
        "assistant": _serialize_message(pair.assistant),
    }


def _render_json(data: Conversation | list[DialoguePair] | list[str] | dict[str, Any]) -> dict[str, Any]:
    """Render projection data as JSON-serializable dict."""
    if isinstance(data, dict):
        return data
    elif isinstance(data, Conversation):
        return {
            "id": data.id,
            "title": data.title,
            "provider": data.provider,
            "created_at": data.created_at.isoformat() if data.created_at else None,
            "updated_at": data.updated_at.isoformat() if data.updated_at else None,
            "messages": [_serialize_message(m) for m in data.messages],
        }
    elif isinstance(data, list):
        if data and isinstance(data[0], DialoguePair):
            return {"pairs": [_serialize_pair(p) for p in data]}  # type: ignore[arg-type]
        elif data and isinstance(data[0], str):
            return {"thinking_traces": data}
    return {}


# --- Filtering ---


def _list_conversations(
    repo: ConversationRepository,
    *,
    limit: int = 50,
    offset: int = 0,
    provider: str | None = None,
    since: str | None = None,
    until: str | None = None,
    query: str | None = None,
) -> list[Conversation]:
    """List and filter conversations."""
    if query:
        conversations: list[Conversation] = repo.search(query) or []
        # Apply provider filter to search results (search doesn't filter by provider)
        if provider:
            conversations = [c for c in conversations if c.provider == provider]
    else:
        # Provider filtering happens at SQL level now
        # Fetch more if date filters are active (they're still Python-level)
        fetch_limit = limit * 20 if (since or until) else limit
        conversations = repo.list(limit=fetch_limit, offset=offset, provider=provider) or []

    # Apply date filters (still Python-level for now)
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
            conversations = [c for c in conversations if c.updated_at and c.updated_at >= since_dt]
        except ValueError as exc:
            raise ValueError(f"Invalid --since date '{since}': {exc}") from exc

    if until:
        try:
            until_dt = datetime.fromisoformat(until)
            conversations = [c for c in conversations if c.updated_at and c.updated_at <= until_dt]
        except ValueError as exc:
            raise ValueError(f"Invalid --until date '{until}': {exc}") from exc

    return list(conversations[:limit])


# --- CLI Command ---


@click.command("view")
@click.argument("conversation_id", required=False)
@click.option(
    "--projection", "-p",
    type=click.Choice(list(PROJECTIONS.keys())),
    default="clean",
    help="Semantic projection to apply",
)
@click.option("--limit", type=int, default=20, show_default=True, help="Max conversations to list")
@click.option("--offset", type=int, default=0, help="Skip first N conversations")
@click.option("--provider", help="Filter by provider (chatgpt, claude, etc)")
@click.option("--since", help="Filter conversations updated after date (ISO format)")
@click.option("--until", help="Filter conversations updated before date (ISO format)")
@click.option("--query", "-q", help="FTS search query to filter conversations")
@click.option("--json", "json_output", is_flag=True, help="Output JSON")
@click.option("--json-lines", is_flag=True, help="Output JSON Lines (one per conversation)")
@click.option("--list", "list_mode", is_flag=True, help="List matching conversations (no content)")
@click.option("--verbose", "-v", is_flag=True, help="Include metadata in output")
@click.pass_obj
def view_command(
    env: AppEnv,
    conversation_id: str | None,
    projection: str,
    limit: int,
    offset: int,
    provider: str | None,
    since: str | None,
    until: str | None,
    query: str | None,
    json_output: bool,
    json_lines: bool,
    list_mode: bool,
    verbose: bool,
) -> None:
    """View conversations with semantic projections.

    Without CONVERSATION_ID, lists matching conversations.
    With CONVERSATION_ID, displays that conversation with the selected projection.

    \b
    Projections:
      full      - All messages (no filtering)
      dialogue  - User/assistant dialogue only (no system/tool)
      clean     - Substantive dialogue (no noise, context dumps) [default]
      pairs     - User-assistant turn pairs
      user      - User messages only
      assistant - Assistant messages only
      thinking  - Thinking/reasoning traces only
      stats     - Statistics summary only

    \b
    Examples:
      polylogue view                          # List recent conversations
      polylogue view abc123                   # View conversation with clean projection
      polylogue view abc123 -p pairs          # View as dialogue pairs
      polylogue view --query "python" -p stats  # Stats for conversations matching 'python'
      polylogue view --provider claude --json # JSON output of Claude conversations
    """
    try:
        load_effective_config(env)
    except ConfigError as exc:
        fail("view", str(exc))

    repo = ConversationRepository()

    # Single conversation view
    if conversation_id:
        convo = repo.view(conversation_id)  # Supports partial ID resolution
        if not convo:
            fail("view", f"Conversation not found: {conversation_id}")

        projected = _apply_projection(convo, projection)

        if json_output:
            env.ui.console.print(json.dumps(_render_json(projected), indent=2))
        else:
            for line in _render_text(projected, include_role=True, include_meta=verbose):
                env.ui.console.print(line)
        return

    # List/search mode
    try:
        conversations = _list_conversations(
            repo,
            limit=limit,
            offset=offset,
            provider=provider,
            since=since,
            until=until,
            query=query,
        )
    except RuntimeError as exc:
        fail("view", str(exc))

    if not conversations:
        env.ui.console.print("No conversations found.")
        return

    if list_mode or (not json_output and not json_lines):
        # List mode - show conversation summaries
        env.ui.summary("Conversations", [f"Found: {len(conversations)}"])
        for convo in conversations:
            title = convo.title or (convo.id[:30] + "..." if len(convo.id) > 30 else convo.id)
            date_str = convo.updated_at.strftime("%Y-%m-%d") if convo.updated_at else "unknown"
            msg_count = convo.message_count
            # Show up to 24 chars of ID to be more usable for prefix matching
            env.ui.console.print(f"  {convo.id[:24]:24}  {date_str}  [{convo.provider:12}]  {title} ({msg_count} msgs)")

        if not json_output and not json_lines:
            return

    # JSON output for multiple conversations
    if json_output:
        results = []
        for convo in conversations:
            projected = _apply_projection(convo, projection)
            results.append(_render_json(projected))
        env.ui.console.print(json.dumps(results, indent=2))
    elif json_lines:
        for convo in conversations:
            projected = _apply_projection(convo, projection)
            env.ui.console.print(json.dumps(_render_json(projected)))
