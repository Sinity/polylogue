"""Small public root for semantic-proof suite entrypoints."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import archive_root as default_archive_root
from polylogue.paths import db_path as default_db_path
from polylogue.rendering.semantic_proof_models import SemanticProofReport, SemanticProofSuiteReport
from polylogue.rendering.semantic_proof_suite_runtime import prove_semantic_surface_suite_async
from polylogue.rendering.semantic_proof_suite_support import empty_surface_report
from polylogue.rendering.semantic_surface_registry import resolve_semantic_surfaces
from polylogue.sync_bridge import run_coroutine_sync

_prove_semantic_surface_suite_async = prove_semantic_surface_suite_async


def prove_semantic_surface_suite(
    *,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    providers: list[str] | None = None,
    surfaces: list[str] | tuple[str, ...] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> SemanticProofSuiteReport:
    """Run semantic preservation proof across canonical render and export surfaces."""
    effective_db_path = db_path or default_db_path()
    bounded_limit = max(1, int(record_limit)) if record_limit is not None else None
    bounded_offset = max(0, int(record_offset))
    resolved_surfaces = resolve_semantic_surfaces(surfaces)
    provider_filters = list(providers or [])

    if not effective_db_path.exists():
        return SemanticProofSuiteReport(
            surface_reports={
                surface: empty_surface_report(
                    surface,
                    record_limit=bounded_limit,
                    record_offset=bounded_offset,
                    provider_filters=provider_filters,
                )
                for surface in resolved_surfaces
            },
            record_limit=bounded_limit,
            record_offset=bounded_offset,
            provider_filters=provider_filters,
            surface_filters=list(resolved_surfaces),
        )

    return run_coroutine_sync(
        _prove_semantic_surface_suite_async(
            db_path=effective_db_path,
            archive_root=archive_root or default_archive_root(),
            providers=providers,
            surfaces=resolved_surfaces,
            record_limit=bounded_limit,
            record_offset=bounded_offset,
        )
    )


def prove_markdown_render_semantics(
    *,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    providers: list[str] | None = None,
    record_limit: int | None = None,
    record_offset: int = 0,
) -> SemanticProofReport:
    """Run semantic preservation proof over canonical markdown rendering."""
    suite = prove_semantic_surface_suite(
        db_path=db_path,
        archive_root=archive_root,
        providers=providers,
        surfaces=["canonical_markdown_v1"],
        record_limit=record_limit,
        record_offset=record_offset,
    )
    return suite.surfaces["canonical_markdown_v1"]


__all__ = [
    "prove_markdown_render_semantics",
    "prove_semantic_surface_suite",
]
