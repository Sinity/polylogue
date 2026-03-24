"""Canonical semantic fact API."""

from polylogue.lib.semantic_fact_builders import (
    build_conversation_semantic_facts,
    build_mcp_detail_semantic_facts,
    build_mcp_summary_semantic_facts,
    build_message_semantic_facts,
    build_projection_semantic_facts,
    build_stream_semantic_facts,
    build_summary_semantic_facts,
)
from polylogue.lib.semantic_fact_models import (
    ConversationSemanticFacts,
    MCPDetailSemanticFacts,
    MCPSummarySemanticFacts,
    MessageSemanticFacts,
    ProjectionSemanticFacts,
    StreamSemanticFacts,
    SummarySemanticFacts,
)
from polylogue.lib.semantic_fact_support import (
    message_has_text,
    message_model_name,
    message_reasoning_traces,
    message_tokens,
    message_tool_calls,
    normalized_role_label,
    sorted_counts,
)

__all__ = [
    "ConversationSemanticFacts",
    "MCPDetailSemanticFacts",
    "MCPSummarySemanticFacts",
    "MessageSemanticFacts",
    "ProjectionSemanticFacts",
    "StreamSemanticFacts",
    "SummarySemanticFacts",
    "build_conversation_semantic_facts",
    "build_message_semantic_facts",
    "build_mcp_detail_semantic_facts",
    "build_mcp_summary_semantic_facts",
    "build_projection_semantic_facts",
    "build_stream_semantic_facts",
    "build_summary_semantic_facts",
    "message_has_text",
    "message_model_name",
    "message_reasoning_traces",
    "message_tokens",
    "message_tool_calls",
    "normalized_role_label",
    "sorted_counts",
]
