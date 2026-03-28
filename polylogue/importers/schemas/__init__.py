"""Provider-specific Pydantic schemas for strict validation.

These schemas define the expected structure of data from each provider.
When a provider changes their format, Pydantic will raise a clear ValidationError
indicating exactly what changed, making it easier to adapt the importers.
"""
from .chatgpt import ChatGPTConversation, ChatGPTMessage
from .claude_ai import ClaudeAIConversation, ClaudeAIMessage

__all__ = [
    "ChatGPTConversation",
    "ChatGPTMessage",
    "ClaudeAIConversation",
    "ClaudeAIMessage",
]
