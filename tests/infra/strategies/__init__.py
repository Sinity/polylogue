"""Hypothesis strategies for polylogue property-based testing.

This module provides composable strategies for generating test data
that matches the structure of provider exports and internal models.

Usage:
    from hypothesis import given
    from tests.infra.strategies import chatgpt_message, conversation_strategy

    @given(chatgpt_message())
    def test_message_parsing(msg):
        ...

Strategies are organized by domain:
- messages: Message and content block strategies
- providers: Provider-specific export format strategies
- filters: Filter composition strategies for query testing
- adversarial: Malformed/attack data for security testing
"""

from tests.infra.strategies.adversarial import (
    malformed_json_strategy,
    path_traversal_strategy,
    sql_injection_strategy,
)
from tests.infra.strategies.filters import (
    filter_arg_strategy,
    filter_chain_strategy,
    filter_type_strategy,
)
from tests.infra.strategies.messages import (
    code_block_strategy,
    content_block_strategy,
    conversation_strategy,
    message_strategy,
    parsed_attachment_model_strategy,
    parsed_conversation_model_strategy,
    parsed_message_model_strategy,
    parsed_message_strategy,
    text_content_strategy,
    thinking_block_strategy,
    tool_use_block_strategy,
)
from tests.infra.strategies.providers import (
    chatgpt_export_strategy,
    chatgpt_message_node_strategy,
    claude_ai_export_strategy,
    claude_ai_message_strategy,
    claude_code_message_strategy,
    codex_message_strategy,
    decode_provider_payload,
    gemini_export_strategy,
    gemini_message_strategy,
    provider_export_strategy,
    provider_hint_path_strategy,
    provider_payload_case_strategy,
    provider_payload_strategy,
    provider_source_case_strategy,
)
from tests.infra.strategies.sources import (
    conversations_wrapper_bytes_strategy,
    json_array_bytes_strategy,
    json_document_strategy,
    jsonl_bytes_strategy,
)
from tests.infra.strategies.storage import (
    ConversationSpec,
    MessageSpec,
    conversation_graph_strategy,
    expected_sorted_ids,
    expected_tree_ids,
    root_index,
    seed_conversation_graph,
)

__all__ = [
    # Messages
    "code_block_strategy",
    "content_block_strategy",
    "conversation_strategy",
    "message_strategy",
    "parsed_attachment_model_strategy",
    "parsed_conversation_model_strategy",
    "parsed_message_model_strategy",
    "parsed_message_strategy",
    "text_content_strategy",
    "thinking_block_strategy",
    "tool_use_block_strategy",
    # Providers
    "chatgpt_export_strategy",
    "chatgpt_message_node_strategy",
    "claude_ai_export_strategy",
    "claude_ai_message_strategy",
    "claude_code_message_strategy",
    "codex_message_strategy",
    "decode_provider_payload",
    "gemini_export_strategy",
    "gemini_message_strategy",
    "provider_hint_path_strategy",
    "provider_payload_case_strategy",
    "provider_payload_strategy",
    "provider_export_strategy",
    "provider_source_case_strategy",
    # Source/json wire contracts
    "conversations_wrapper_bytes_strategy",
    "json_array_bytes_strategy",
    "json_document_strategy",
    "jsonl_bytes_strategy",
    # Storage
    "ConversationSpec",
    "MessageSpec",
    "conversation_graph_strategy",
    "expected_sorted_ids",
    "expected_tree_ids",
    "root_index",
    "seed_conversation_graph",
    # Filters
    "filter_arg_strategy",
    "filter_chain_strategy",
    "filter_type_strategy",
    # Adversarial
    "malformed_json_strategy",
    "path_traversal_strategy",
    "sql_injection_strategy",
]
