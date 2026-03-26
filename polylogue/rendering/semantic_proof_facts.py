"""Fact extraction and primitive semantic checks for proof surfaces."""

from __future__ import annotations

from polylogue.rendering.semantic_proof_fact_checks import (
    _critical_or_preserved,
    _declared_loss_or_preserved,
    _presence_check,
)
from polylogue.rendering.semantic_proof_fact_exports import (
    _canonical_markdown_output_facts,
    _csv_output_facts,
    _html_output_facts,
    _json_like_output_facts,
    _markdown_doc_output_facts,
    _obsidian_output_facts,
    _org_output_facts,
)
from polylogue.rendering.semantic_proof_fact_reads import (
    _mcp_detail_output_facts,
    _mcp_summary_output_facts,
    _stream_json_lines_output_facts,
    _stream_markdown_output_facts,
    _stream_plaintext_output_facts,
    _summary_csv_output_facts,
    _summary_output_facts,
    _summary_text_output_facts,
)

__all__ = [
    "_canonical_markdown_output_facts",
    "_critical_or_preserved",
    "_csv_output_facts",
    "_declared_loss_or_preserved",
    "_html_output_facts",
    "_json_like_output_facts",
    "_markdown_doc_output_facts",
    "_mcp_detail_output_facts",
    "_mcp_summary_output_facts",
    "_obsidian_output_facts",
    "_org_output_facts",
    "_presence_check",
    "_stream_json_lines_output_facts",
    "_stream_markdown_output_facts",
    "_stream_plaintext_output_facts",
    "_summary_csv_output_facts",
    "_summary_output_facts",
    "_summary_text_output_facts",
]
