"""Derived-model statuses for embeddings and retrieval bands."""

from __future__ import annotations

from polylogue.maintenance_models import DerivedModelStatus
from polylogue.storage.derived_status_support import pending_docs, pending_rows


def build_retrieval_statuses(metrics: dict[str, int | bool]) -> dict[str, DerivedModelStatus]:
    return {
        "transcript_embeddings": DerivedModelStatus(
            name="transcript_embeddings",
            ready=bool(metrics["transcript_embeddings_ready"]),
            detail=(
                f"Transcript embeddings ready ({metrics['embedded_conversations']:,}/{metrics['total_conversations']:,} conversations, {metrics['embedded_messages']:,} messages)"
                if bool(metrics["transcript_embeddings_ready"])
                else (
                    f"Transcript embeddings pending ({metrics['embedded_conversations']:,}/{metrics['total_conversations']:,} conversations, "
                    f"pending {metrics['pending_conversations']:,}, stale {metrics['stale_messages']:,}, missing provenance {metrics['missing_provenance']:,})"
                )
            ),
            source_documents=int(metrics["total_conversations"]),
            materialized_documents=int(metrics["embedded_conversations"]),
            materialized_rows=int(metrics["embedded_messages"]),
            pending_documents=int(metrics["pending_conversations"]),
            stale_rows=int(metrics["stale_messages"]),
            missing_provenance_rows=int(metrics["missing_provenance"]),
        ),
        "retrieval_evidence": DerivedModelStatus(
            name="retrieval_evidence",
            ready=bool(metrics["evidence_retrieval_ready"]),
            detail=(
                f"Evidence retrieval ready ({metrics['evidence_retrieval_rows']:,}/{metrics['expected_evidence_retrieval_rows']:,} supporting rows)"
                if bool(metrics["evidence_retrieval_ready"])
                else (
                    f"Evidence retrieval pending ({metrics['evidence_retrieval_rows']:,}/{metrics['expected_evidence_retrieval_rows']:,} supporting rows; "
                    f"profile_evidence_fts={metrics['profile_evidence_fts_rows']:,}/{metrics['profile_rows']:,}, "
                    f"action_event_fts={metrics['action_fts_rows']:,}/{metrics['action_rows']:,})"
                )
            ),
            source_documents=int(metrics["action_source_documents"]) + int(metrics["total_conversations"]),
            materialized_documents=int(metrics["action_documents"]) + int(metrics["profile_rows"]),
            source_rows=int(metrics["expected_evidence_retrieval_rows"]),
            materialized_rows=int(metrics["evidence_retrieval_rows"]),
            pending_documents=(
                pending_docs(int(metrics["action_source_documents"]), int(metrics["action_documents"]))
                + pending_docs(int(metrics["total_conversations"]), int(metrics["profile_rows"]))
            ),
            pending_rows=pending_rows(int(metrics["expected_evidence_retrieval_rows"]), int(metrics["evidence_retrieval_rows"])),
            stale_rows=int(metrics["profile_evidence_fts_duplicates"]) + int(metrics["action_stale_rows"]),
            orphan_rows=int(metrics["action_orphan_rows"]) + int(metrics["orphan_profile_rows"]),
        ),
        "retrieval_inference": DerivedModelStatus(
            name="retrieval_inference",
            ready=bool(metrics["inference_retrieval_ready"]),
            detail=(
                f"Inference retrieval ready ({metrics['inference_retrieval_rows']:,}/{metrics['expected_inference_retrieval_rows']:,} supporting rows)"
                if bool(metrics["inference_retrieval_ready"])
                else (
                    f"Inference retrieval pending ({metrics['inference_retrieval_rows']:,}/{metrics['expected_inference_retrieval_rows']:,} supporting rows; "
                    f"profile_inference_fts={metrics['profile_inference_fts_rows']:,}/{metrics['profile_rows']:,}, "
                    f"work_event_fts={metrics['work_event_fts_rows']:,}/{metrics['work_event_rows']:,}, "
                    f"phases={metrics['phase_rows']:,}/{metrics['expected_phase_rows']:,})"
                )
            ),
            source_documents=int(metrics["profile_rows"]),
            materialized_documents=int(metrics["profile_rows"]),
            source_rows=int(metrics["expected_inference_retrieval_rows"]),
            materialized_rows=int(metrics["inference_retrieval_rows"]),
            pending_rows=pending_rows(int(metrics["expected_inference_retrieval_rows"]), int(metrics["inference_retrieval_rows"])),
            stale_rows=(
                int(metrics["profile_inference_fts_duplicates"])
                + int(metrics["work_event_fts_duplicates"])
                + int(metrics["stale_work_event_rows"])
                + int(metrics["stale_phase_rows"])
            ),
            orphan_rows=(
                int(metrics["orphan_profile_rows"])
                + int(metrics["orphan_work_event_rows"])
                + int(metrics["orphan_phase_rows"])
            ),
        ),
        "retrieval_enrichment": DerivedModelStatus(
            name="retrieval_enrichment",
            ready=bool(metrics["enrichment_retrieval_ready"]),
            detail=(
                f"Enrichment retrieval ready ({metrics['enrichment_retrieval_rows']:,}/{metrics['expected_enrichment_retrieval_rows']:,} supporting rows)"
                if bool(metrics["enrichment_retrieval_ready"])
                else (
                    f"Enrichment retrieval pending ({metrics['enrichment_retrieval_rows']:,}/{metrics['expected_enrichment_retrieval_rows']:,} supporting rows; "
                    f"profile_enrichment_fts={metrics['profile_enrichment_fts_rows']:,}/{metrics['profile_rows']:,})"
                )
            ),
            source_rows=int(metrics["expected_enrichment_retrieval_rows"]),
            materialized_rows=int(metrics["enrichment_retrieval_rows"]),
            pending_rows=pending_rows(int(metrics["expected_enrichment_retrieval_rows"]), int(metrics["enrichment_retrieval_rows"])),
            stale_rows=int(metrics["profile_enrichment_fts_duplicates"]),
        ),
    }


__all__ = ["build_retrieval_statuses"]
