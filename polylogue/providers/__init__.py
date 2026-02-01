"""Provider-specific typed models for parse-don't-validate pattern.

This package contains strictly-typed models for each provider's format.
These models are derived from the JSON schemas in polylogue/schemas/providers/
and provide:

1. Type-safe parsing of provider exports
2. IDE autocompletion for provider-specific fields
3. Early error detection on malformed data
4. Self-documenting schemas

The models use Pydantic with `extra='allow'` to:
- Validate known fields strictly
- Preserve unknown fields for forward compatibility
- Enable gradual schema evolution

Usage:
    from polylogue.providers.chatgpt import ChatGPTConversation

    # Parse with validation
    conv = ChatGPTConversation.model_validate(raw_data)

    # Access typed fields
    print(conv.title, conv.create_time)

    # Unknown fields are preserved
    if hasattr(conv, 'new_field_from_future'):
        print(conv.new_field_from_future)
"""

from polylogue.providers.chatgpt import ChatGPTConversation
from polylogue.providers.claude_ai import ClaudeAIConversation
from polylogue.providers.claude_code import ClaudeCodeRecord
from polylogue.providers.codex import CodexRecord
from polylogue.providers.gemini import GeminiMessage

__all__ = [
    "ChatGPTConversation",
    "ClaudeAIConversation",
    "ClaudeCodeRecord",
    "CodexRecord",
    "GeminiMessage",
]
