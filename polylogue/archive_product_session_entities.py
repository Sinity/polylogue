"""Session-scoped archive product entities."""

from __future__ import annotations

from typing import Any

from polylogue.archive_product_base import (
    ARCHIVE_PRODUCT_CONTRACT_VERSION,
    ArchiveEnrichmentProvenance,
    ArchiveInferenceProvenance,
    ArchiveProductModel,
    ArchiveProductProvenance,
)
from polylogue.archive_product_payloads import (
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.storage.store import (
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)


class SessionProfileProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_profile"
    semantic_tier: str = "merged"
    conversation_id: str
    provider_name: str
    title: str | None = None
    provenance: ArchiveProductProvenance
    evidence: SessionEvidencePayload | None = None
    inference_provenance: ArchiveInferenceProvenance | None = None
    inference: SessionInferencePayload | None = None

    @classmethod
    def from_record(
        cls,
        record: SessionProfileRecord,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct:
        include_evidence = tier in {"merged", "evidence"}
        include_inference = tier in {"merged", "inference"}
        return cls(
            semantic_tier=tier,
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            title=record.title,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            evidence=(
                SessionEvidencePayload.model_validate(record.evidence_payload)
                if include_evidence
                else None
            ),
            inference_provenance=(
                ArchiveInferenceProvenance(
                    materializer_version=record.materializer_version,
                    materialized_at=record.materialized_at,
                    source_updated_at=record.source_updated_at,
                    source_sort_key=record.source_sort_key,
                    inference_version=record.inference_version,
                    inference_family=record.inference_family,
                )
                if include_inference
                else None
            ),
            inference=(
                SessionInferencePayload.model_validate(record.inference_payload)
                if include_inference
                else None
            ),
        )


class SessionEnrichmentProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_enrichment"
    semantic_tier: str = "enrichment"
    conversation_id: str
    provider_name: str
    title: str | None = None
    provenance: ArchiveProductProvenance
    enrichment_provenance: ArchiveEnrichmentProvenance
    enrichment: SessionEnrichmentPayload

    @classmethod
    def from_record(cls, record: SessionProfileRecord) -> SessionEnrichmentProduct:
        return cls(
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            title=record.title,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            enrichment_provenance=ArchiveEnrichmentProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
                enrichment_version=record.enrichment_version,
                enrichment_family=record.enrichment_family,
            ),
            enrichment=SessionEnrichmentPayload.model_validate(record.enrichment_payload),
        )


class SessionWorkEventProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_work_event"
    semantic_tier: str = "inference"
    event_id: str
    conversation_id: str
    provider_name: str
    event_index: int
    provenance: ArchiveProductProvenance
    inference_provenance: ArchiveInferenceProvenance
    evidence: WorkEventEvidencePayload
    inference: WorkEventInferencePayload

    @classmethod
    def from_record(cls, record: SessionWorkEventRecord) -> SessionWorkEventProduct:
        return cls(
            event_id=record.event_id,
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            event_index=record.event_index,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            inference_provenance=ArchiveInferenceProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
                inference_version=record.inference_version,
                inference_family=record.inference_family,
            ),
            evidence=WorkEventEvidencePayload.model_validate(record.evidence_payload),
            inference=WorkEventInferencePayload.model_validate(record.inference_payload),
        )


class SessionPhaseProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "session_phase"
    semantic_tier: str = "inference"
    phase_id: str
    conversation_id: str
    provider_name: str
    phase_index: int
    provenance: ArchiveProductProvenance
    inference_provenance: ArchiveInferenceProvenance
    evidence: SessionPhaseEvidencePayload
    inference: SessionPhaseInferencePayload

    @classmethod
    def from_record(cls, record: SessionPhaseRecord) -> SessionPhaseProduct:
        return cls(
            phase_id=record.phase_id,
            conversation_id=record.conversation_id,
            provider_name=record.provider_name,
            phase_index=record.phase_index,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
            ),
            inference_provenance=ArchiveInferenceProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.source_updated_at,
                source_sort_key=record.source_sort_key,
                inference_version=record.inference_version,
                inference_family=record.inference_family,
            ),
            evidence=SessionPhaseEvidencePayload.model_validate(record.evidence_payload),
            inference=SessionPhaseInferencePayload.model_validate(record.inference_payload),
        )


class WorkThreadProduct(ArchiveProductModel):
    contract_version: int = ARCHIVE_PRODUCT_CONTRACT_VERSION
    product_kind: str = "work_thread"
    thread_id: str
    root_id: str
    dominant_project: str | None = None
    provenance: ArchiveProductProvenance
    thread: dict[str, Any]

    @classmethod
    def from_record(cls, record: WorkThreadRecord) -> WorkThreadProduct:
        return cls(
            thread_id=record.thread_id,
            root_id=record.root_id,
            dominant_project=record.dominant_project,
            provenance=ArchiveProductProvenance(
                materializer_version=record.materializer_version,
                materialized_at=record.materialized_at,
                source_updated_at=record.end_time or record.start_time,
                source_sort_key=None,
            ),
            thread=dict(record.payload),
        )
