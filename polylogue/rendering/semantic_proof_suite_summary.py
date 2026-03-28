"""Summary-surface collectors for semantic-proof suites."""

from __future__ import annotations

from polylogue.cli.query_output import format_summary_list
from polylogue.mcp.payloads import MCPConversationSummaryListPayload, MCPConversationSummaryPayload
from polylogue.rendering.semantic_proof_surface_reads import (
    _prove_mcp_summary_surface,
    _prove_query_summary_csv_surface,
    _prove_query_summary_json_like_surface,
    _prove_query_summary_text_surface,
)

_SUMMARY_JSON_SURFACES = {
    "query_summary_json_v1",
    "query_summary_yaml_v1",
}


def append_summary_surface_proofs(
    proofs_by_surface: dict[str, list],
    *,
    summary,
    message_count: int,
    surface_set: set[str],
) -> None:
    """Append summary-only surface proofs for a single summary row."""
    for surface in sorted(_SUMMARY_JSON_SURFACES & surface_set):
        rendered_text = format_summary_list(
            [summary],
            "json" if surface == "query_summary_json_v1" else "yaml",
            None,
            message_counts={str(summary.id): message_count},
        )
        proofs_by_surface[surface].append(
            _prove_query_summary_json_like_surface(
                summary=summary,
                message_count=message_count,
                rendered_text=rendered_text,
                surface=surface,
            )
        )

    if "query_summary_csv_v1" in surface_set:
        rendered_text = format_summary_list(
            [summary],
            "csv",
            None,
            message_counts={str(summary.id): message_count},
        )
        proofs_by_surface["query_summary_csv_v1"].append(
            _prove_query_summary_csv_surface(
                summary=summary,
                message_count=message_count,
                rendered_text=rendered_text,
            )
        )

    if "query_summary_text_v1" in surface_set:
        rendered_text = format_summary_list(
            [summary],
            "text",
            None,
            message_counts={str(summary.id): message_count},
        )
        proofs_by_surface["query_summary_text_v1"].append(
            _prove_query_summary_text_surface(
                summary=summary,
                message_count=message_count,
                rendered_text=rendered_text,
            )
        )

    if "mcp_summary_json_v1" in surface_set:
        rendered_text = MCPConversationSummaryListPayload(
            root=[
                MCPConversationSummaryPayload.from_summary(
                    summary,
                    message_count=message_count,
                )
            ]
        ).to_json()
        proofs_by_surface["mcp_summary_json_v1"].append(
            _prove_mcp_summary_surface(
                summary=summary,
                message_count=message_count,
                rendered_text=rendered_text,
            )
        )


__all__ = ["append_summary_surface_proofs"]
