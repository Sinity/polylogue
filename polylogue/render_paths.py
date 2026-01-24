from __future__ import annotations

from pathlib import Path

from .paths import safe_path_component


def render_root(render_root: Path, provider: str, conversation_id: str) -> Path:
    """Get the render output path for a conversation.

    Args:
        render_root: Base render directory
        provider: Provider name (e.g., "claude", "chatgpt")
        conversation_id: Unique conversation identifier

    Returns:
        Path to the conversation's render directory
    """
    safe_provider = safe_path_component(provider, fallback="provider")
    safe_conversation = safe_path_component(conversation_id, fallback="conversation")
    return render_root / safe_provider / safe_conversation


__all__ = ["render_root"]
