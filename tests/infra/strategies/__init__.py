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
from tests.infra.strategies.cli import (
    QueryDeleteCase,
    QueryMutationCase,
    SendOutputCase,
    SummaryOutputCase,
    SummaryStatsCase,
    query_delete_case_strategy,
    query_mutation_case_strategy,
    send_output_case_strategy,
    summary_output_case_strategy,
    summary_stats_case_strategy,
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
    text_content_strategy,
    thinking_block_strategy,
    tool_use_block_strategy,
)
from tests.infra.strategies.pipeline import (
    AcquisitionInputSpec,
    ParseMergeEvent,
    ValidationCase,
    acquisition_input_batch_strategy,
    build_acquisition_raw_bytes,
    build_validation_payload,
    expected_parse_merge_totals,
    expected_validation_contract,
    parse_merge_events_strategy,
    validation_case_strategy,
)
from tests.infra.strategies.providers import (
    chatgpt_export_strategy,
    chatgpt_message_node_strategy,
    chatgpt_semantic_message_strategy,
    claude_ai_export_strategy,
    claude_ai_semantic_message_strategy,
    claude_code_message_strategy,
    claude_code_semantic_record_strategy,
    codex_message_strategy,
    codex_semantic_record_strategy,
    gemini_export_strategy,
    gemini_message_strategy,
    gemini_semantic_message_strategy,
    provider_export_strategy,
    provider_hint_path_strategy,
    provider_payload_case_strategy,
    provider_payload_strategy,
    provider_semantic_case_strategy,
    provider_source_case_strategy,
)
from tests.infra.strategies.search import (
    fts5_match_text_strategy,
    search_query_strategy,
    search_with_since_strategy,
)
from tests.infra.strategies.schema import (
    SessionJsonlFileSpec,
    dynamic_key_strategy,
    expected_session_documents,
    nested_required_schema_strategy,
    record_payload_strategy,
    record_variant_signature,
    session_jsonl_tree_strategy,
    static_key_strategy,
)
from tests.infra.strategies.site import (
    SiteArchiveSpec,
    expected_index_pages,
    site_archive_spec_strategy,
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
    TagAssignmentSpec,
    TitleSearchSpec,
    conversation_graph_strategy,
    expected_sorted_ids,
    expected_tag_counts,
    expected_tree_ids,
    literal_title_search_strategy,
    root_index,
    seed_conversation_graph,
    shortest_unique_prefix,
    tag_assignment_strategy,
)
from tests.infra.strategies.summaries import (
    ConversationSummarySpec,
    build_conversation_summary,
    build_message_counts,
    conversation_summary_batch_strategy,
    conversation_summary_spec_strategy,
)

__all__ = [
    # Messages
    "code_block_strategy",
    "content_block_strategy",
    "conversation_strategy",
    "message_strategy",
    "text_content_strategy",
    "thinking_block_strategy",
    "tool_use_block_strategy",
    # Providers
    "chatgpt_export_strategy",
    "chatgpt_message_node_strategy",
    "chatgpt_semantic_message_strategy",
    "claude_ai_export_strategy",
    "claude_ai_semantic_message_strategy",
    "claude_code_message_strategy",
    "claude_code_semantic_record_strategy",
    "codex_message_strategy",
    "codex_semantic_record_strategy",
    "gemini_export_strategy",
    "gemini_message_strategy",
    "gemini_semantic_message_strategy",
    "provider_hint_path_strategy",
    "provider_semantic_case_strategy",
    "provider_payload_case_strategy",
    "provider_payload_strategy",
    "provider_export_strategy",
    "provider_source_case_strategy",
    # Schema
    "SessionJsonlFileSpec",
    "dynamic_key_strategy",
    "expected_session_documents",
    "nested_required_schema_strategy",
    "record_payload_strategy",
    "record_variant_signature",
    "session_jsonl_tree_strategy",
    "static_key_strategy",
    # Source/json wire contracts
    "conversations_wrapper_bytes_strategy",
    "json_array_bytes_strategy",
    "json_document_strategy",
    "jsonl_bytes_strategy",
    # Storage
    "ConversationSpec",
    "MessageSpec",
    "TagAssignmentSpec",
    "TitleSearchSpec",
    "conversation_graph_strategy",
    "expected_tag_counts",
    "expected_sorted_ids",
    "expected_tree_ids",
    "literal_title_search_strategy",
    "root_index",
    "seed_conversation_graph",
    "shortest_unique_prefix",
    "tag_assignment_strategy",
    # Pipeline
    "AcquisitionInputSpec",
    "ParseMergeEvent",
    "ValidationCase",
    "acquisition_input_batch_strategy",
    "build_acquisition_raw_bytes",
    "build_validation_payload",
    "expected_parse_merge_totals",
    "expected_validation_contract",
    "parse_merge_events_strategy",
    "validation_case_strategy",
    # Filters
    "filter_arg_strategy",
    "filter_chain_strategy",
    "filter_type_strategy",
    # Adversarial
    "malformed_json_strategy",
    "path_traversal_strategy",
    "sql_injection_strategy",
    # Summary/presentation
    "ConversationSummarySpec",
    "build_conversation_summary",
    "build_message_counts",
    "conversation_summary_batch_strategy",
    "conversation_summary_spec_strategy",
    # CLI
    "QueryDeleteCase",
    "QueryMutationCase",
    "SendOutputCase",
    "SummaryOutputCase",
    "SummaryStatsCase",
    "query_delete_case_strategy",
    "query_mutation_case_strategy",
    "send_output_case_strategy",
    "summary_output_case_strategy",
    "summary_stats_case_strategy",
    # Search
    "fts5_match_text_strategy",
    "search_query_strategy",
    "search_with_since_strategy",
    # Site
    "SiteArchiveSpec",
    "expected_index_pages",
    "site_archive_spec_strategy",
]
