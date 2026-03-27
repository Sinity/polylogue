"""Support helpers for semantic-proof suite assembly."""

from __future__ import annotations

from polylogue.rendering.semantic_proof_models import (
    SemanticProofReport,
    SemanticProofSuiteReport,
    _build_provider_reports,
)


def empty_surface_report(
    surface: str,
    *,
    record_limit: int | None,
    record_offset: int,
    provider_filters: list[str],
) -> SemanticProofReport:
    return SemanticProofReport(
        surface=surface,
        conversations=[],
        provider_reports={},
        record_limit=record_limit,
        record_offset=record_offset,
        provider_filters=provider_filters,
    )


def build_suite_report(
    *,
    surfaces: list[str],
    proofs_by_surface: dict[str, list],
    record_limit: int | None,
    record_offset: int,
    provider_filters: list[str],
) -> SemanticProofSuiteReport:
    return SemanticProofSuiteReport(
        surface_reports={
            surface: SemanticProofReport(
                surface=surface,
                conversations=proofs_by_surface[surface],
                provider_reports=_build_provider_reports(proofs_by_surface[surface]),
                record_limit=record_limit,
                record_offset=record_offset,
                provider_filters=provider_filters,
            )
            for surface in surfaces
        },
        record_limit=record_limit,
        record_offset=record_offset,
        provider_filters=provider_filters,
        surface_filters=list(surfaces),
    )


__all__ = [
    "build_suite_report",
    "empty_surface_report",
]
