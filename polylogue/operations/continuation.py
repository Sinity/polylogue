"""Continuation products over the resident operation's original pinned reader."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING

from polylogue.analysis.resume import _rank_resume_profiles
from polylogue.archive.context_models import ContextSpec
from polylogue.archive.hydration import archive_summary_to_domain
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.query.unit_results import query_unit_envelope, query_unit_request
from polylogue.archive.resume_routing import route_resume
from polylogue.archive.session.domain_models import Session
from polylogue.context.product_image import compile_context_image
from polylogue.core.async_bridge import complete_without_suspension
from polylogue.core.errors import SessionNotFoundError
from polylogue.operations.context_image_product import PinnedContextImageSource
from polylogue.operations.daemon_protocol import (
    ContinuationCandidatesRequest,
    ContinuationContextRequest,
    ContinuationRouteRequest,
)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def successor_context_spec(session_id: str) -> ContextSpec:
    """The original continuation recipe, independent of surface syntax."""
    session_clause = f"session.id:{session_id}"
    return ContextSpec(
        purpose="continue",
        seed_refs=(f"session:{session_id}",),
        read_views=("messages",),
        unit_queries=tuple(
            f"{unit} where {session_clause}" for unit in ("runs", "observed-events", "context-snapshots", "actions")
        ),
    )


class _PinnedContinuationSource(PinnedContextImageSource):
    def __init__(self, archive: ArchiveStore, *, observed_at_ms: int, serving_identity: str) -> None:
        super().__init__(archive, observed_at_ms=observed_at_ms)
        self.serving_identity = serving_identity

    async def query_units(self, expression: str, *, limit: int) -> object:
        self.archive.check_operation_read()
        envelope = query_unit_envelope(
            self.archive,
            query_unit_request(expression=expression, limit=limit),
            serving_identity=self.serving_identity,
        )
        self.archive.check_operation_read()
        return envelope


def execute_continuation_read(
    name: str,
    payload: dict[str, object],
    *,
    archive: ArchiveStore,
    serving_identity: str,
    checkpoint: Callable[[], None],
) -> dict[str, object]:
    """Return a route, successor image or ranked window without reopening a tier."""
    checkpoint()
    archive.check_operation_read()
    if name == "continuation.candidates":
        request = ContinuationCandidatesRequest.model_validate(payload)
        profiles = archive.list_session_profile_insights(sort="last-message", tier="merged", limit=None)
        candidates = _rank_resume_profiles(
            profiles,
            repo_path=request.repo_path,
            cwd=request.cwd,
            recent_files=request.recent_files,
            limit=request.limit,
            checkpoint=checkpoint,
        )
        checkpoint()
        return {
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "returned": len(candidates),
            "limit": request.limit,
        }
    target = (
        ContinuationRouteRequest.model_validate(payload)
        if name == "continuation.route"
        else ContinuationContextRequest.model_validate(payload)
    )
    session_id = target.session_id
    try:
        summary = archive_summary_to_domain(archive.read_summary(archive.resolve_session_id(session_id)))
    except KeyError as exc:
        raise SessionNotFoundError(f"Session not found: {session_id}") from exc
    checkpoint()
    if name == "continuation.route":
        # Routing reads identity/cwd only. A transcript is not an input to the
        # existing harness route owner, so do not hydrate it for this product.
        route = route_resume(
            Session(
                id=summary.id,
                origin=summary.origin,
                messages=MessageCollection(messages=[]),
                working_directories=summary.working_directories,
            )
        )
        return {**asdict(route), "command": route.command, "argv": list(route.argv)}
    image = complete_without_suspension(
        compile_context_image(
            _PinnedContinuationSource(
                archive,
                observed_at_ms=ContinuationContextRequest.model_validate(payload).observed_at_ms,
                serving_identity=serving_identity,
            ),
            successor_context_spec(str(summary.id)),
        )
    )
    # The compiler records unsupported query-unit omissions, but an operation
    # cancellation remains a refusal, never a successful partial image.
    archive.check_operation_read()
    checkpoint()
    return {"view": "context-image", "payload": image.model_dump(mode="json", exclude_none=True)}
