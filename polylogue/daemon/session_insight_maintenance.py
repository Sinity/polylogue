"""Translate sealed insight-maintenance authority into owner work.

The operation layer seals a bounded, ordered set of targets.  This module
does not discover a replacement scope or perform storage work: the shared
session-profile owner owns the compute admission, short writer publication,
and output certification.  Keeping this translation here lets the owner stay
independent of audit and transport contracts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from polylogue.daemon.convergence import (
    SelectedSessionTarget,
    SessionProfileConvergenceOwner,
)
from polylogue.daemon.derivation import DerivationFrame
from polylogue.operations.insight_acceptance import (
    AcceptedInsightPart,
    InsightCertifiedCounts,
    InsightTargetDisposition,
    SessionInsightPartReceipt,
    SessionInsightTargetReceipt,
)
from polylogue.operations.session_profile_convergence import make_session_profile_frame

SessionProfileFrameFactory = Callable[[tuple[str, ...]], DerivationFrame]
StopRequested = Callable[[], str | None]


class SessionInsightMaintenance:
    """Run one already-authorized insight page through the shared owner."""

    def __init__(
        self,
        owner: SessionProfileConvergenceOwner,
        *,
        frame_factory: SessionProfileFrameFactory,
    ) -> None:
        self._owner = owner
        self._frame_factory = frame_factory

    def plan_binding(self, *, opened_index_path: Path) -> tuple[str, str]:
        """Return the canonical binding only when it names the pinned index.

        The caller already opened and pinned ``opened_index_path`` under the
        publication guard.  The frame factory supplies the owner vocabulary;
        this method does not open a database, resolve a root, or substitute a
        different active generation if that factory observation disagrees.
        """

        frame = self._frame_factory(())
        expected_generation = f"index-generation:{opened_index_path}"
        if frame.source_revision != expected_generation:
            raise RuntimeError("session insight frame does not name the supplied opened index generation")
        recipe = frame.recipe_version("session_profile")
        if not recipe:
            raise RuntimeError("session insight frame has no session-profile recipe binding")
        return frame.source_revision, recipe

    async def converge_part(
        self,
        part: AcceptedInsightPart,
        *,
        stop_requested: StopRequested,
    ) -> SessionInsightPartReceipt:
        """Converge exactly ``part.targets`` without scope rediscovery.

        The owner returns only an ordered attempted prefix when it observes a
        stop request.  The suffix is retained verbatim as unattempted authority
        rather than relabelled as a pending outcome.
        """

        target_refs = tuple(target.target_ref for target in part.targets)
        dispositions = tuple(target.disposition for target in part.targets)
        return await self._converge_targets(
            target_refs,
            dispositions=dispositions,
            expected_generation=part.index_generation,
            expected_recipe=part.recipe_version,
            stop_requested=stop_requested,
        )

    async def converge_ingest_sessions(
        self,
        session_ids: tuple[str, ...],
        *,
        expected_recipe: str,
        stop_requested: StopRequested,
    ) -> SessionInsightPartReceipt:
        """Converge exact source-authorized session ids without audit-page refs.

        Ingest has already established its own durable source-generation
        authority.  It therefore supplies concrete session ids, all required,
        and binds this call to a fresh canonical frame rather than fabricating
        an :class:`AcceptedInsightPart` or rediscovering archive work. The
        caller passes the recipe preserved by its accepted source-generation
        plan; the owner marks it stale if the current adapter recipe moved.
        """

        target_refs = tuple(f"session:{session_id}" for session_id in session_ids)
        exact_session_ids = tuple(_session_id(target_ref) for target_ref in target_refs)
        ingest_dispositions: tuple[InsightTargetDisposition, ...] = tuple("required" for _ in target_refs)
        frame = self._frame_factory(exact_session_ids)
        return await self._converge_targets(
            target_refs,
            dispositions=ingest_dispositions,
            expected_generation=frame.source_revision,
            expected_recipe=expected_recipe,
            stop_requested=stop_requested,
            frame=frame,
        )

    async def _converge_targets(
        self,
        target_refs: tuple[str, ...],
        *,
        dispositions: tuple[InsightTargetDisposition, ...],
        expected_generation: str,
        expected_recipe: str,
        stop_requested: StopRequested,
        frame: DerivationFrame | None = None,
    ) -> SessionInsightPartReceipt:
        if len(target_refs) != len(dispositions):
            raise ValueError("session target refs and dispositions must have equal length")
        session_ids = tuple(_session_id(target_ref) for target_ref in target_refs)
        selected = tuple(
            SelectedSessionTarget(session_id, disposition)
            for session_id, disposition in zip(session_ids, dispositions, strict=True)
        )
        resolved_frame = frame if frame is not None else self._frame_factory(session_ids)
        outcomes = await self._owner.converge_selected(
            resolved_frame,
            targets=selected,
            expected_generation=expected_generation,
            expected_recipe=expected_recipe,
            stop_requested=stop_requested,
        )
        if len(outcomes) > len(target_refs):
            raise RuntimeError("session owner returned more outcomes than the exact target sequence")

        receipts: list[SessionInsightTargetReceipt] = []
        for target_ref, outcome in zip(target_refs, outcomes, strict=False):
            expected_session_id = _session_id(target_ref)
            if outcome.session_id != expected_session_id:
                raise RuntimeError("session owner outcome does not match the accepted target order")
            receipts.append(
                SessionInsightTargetReceipt(
                    target_ref=target_ref,
                    disposition=outcome.state,
                    input_binding=outcome.input_binding,
                    output_binding=outcome.output_binding,
                    certified_counts=InsightCertifiedCounts(
                        profiles=outcome.certified_counts.profiles,
                        work_events=outcome.certified_counts.work_events,
                        phases=outcome.certified_counts.phases,
                    ),
                    publication_known_committed=outcome.publication_known_committed,
                )
            )
        return SessionInsightPartReceipt(
            targets=tuple(receipts),
            remaining_unattempted_target_refs=target_refs[len(receipts) :],
        )


def make_session_insight_maintenance(
    owner: SessionProfileConvergenceOwner,
    *,
    index_db_path: Path,
    archive_root: Path,
) -> SessionInsightMaintenance:
    """Bind the canonical frame factory without opening or discovering work."""

    def frame_factory(session_ids: tuple[str, ...]) -> DerivationFrame:
        return make_session_profile_frame(index_db_path, archive_root=archive_root, scope=session_ids)

    return SessionInsightMaintenance(owner, frame_factory=frame_factory)


def _session_id(target_ref: str) -> str:
    """Decode the already-validated operation target reference."""

    if not target_ref.startswith("session:"):
        raise ValueError("accepted insight target must be a session reference")
    session_id = target_ref.removeprefix("session:")
    if not session_id:
        raise ValueError("accepted insight target session id is empty")
    return session_id


__all__ = [
    "SessionInsightMaintenance",
    "SessionProfileFrameFactory",
    "StopRequested",
    "make_session_insight_maintenance",
]
