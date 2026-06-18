"""Shared maintenance operation envelope across CLI, daemon HTTP, and MCP.

This module owns the **single typed envelope** that every maintenance
surface — CLI ``polylogue ops maintenance plan/run``, daemon HTTP
``/api/maintenance/plan`` and ``/api/maintenance/run``, and MCP
``maintenance_preview`` / ``maintenance_execute`` — emits as its result
shape. It exists so that an operator (or a script) can call any of the
three surfaces and parse the same JSON.

The envelope is a structural superset of
:meth:`polylogue.maintenance.planner.BackfillOperation.to_dict` plus a
small set of surface-agnostic metadata:

* ``origin`` — which surface produced this envelope (``"cli"``,
  ``"daemon"``, ``"mcp"``). Useful for diagnostics and for filtering
  operation history without re-deriving the surface from a process
  name.
* ``mode`` — ``"preview"`` (read-only inventory; no mutations) or
  ``"execute"`` (an actual or dry-run repair pass). Surfaces always
  populate this so callers do not have to infer mode from the kind +
  status fields.

The :func:`envelope_from_operation` factory is the only public way to
build an envelope. It refuses unknown ``origin`` / ``mode`` strings
and copies every planner field into a frozen Pydantic model so callers
cannot accidentally mutate the result.

Parity across surfaces is pinned by
``tests/unit/maintenance/test_envelope_contracts.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict

from polylogue.core.json import JSONDocument, json_document
from polylogue.maintenance.planner import (
    BackfillOperation,
    BoundedFailureSamples,
    FailureSample,
)
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.surfaces.payloads import SurfacePayloadModel

#: Allowed values for the ``origin`` envelope field.
EnvelopeOrigin = Literal["cli", "daemon", "mcp"]

#: Allowed values for the ``mode`` envelope field.
EnvelopeMode = Literal["preview", "execute"]


class MaintenanceFailureSamplePayload(SurfacePayloadModel):
    """One bounded failure sample, mirrored from :class:`FailureSample`."""

    kind: str
    locator: str
    message: str

    @classmethod
    def from_sample(cls, sample: FailureSample) -> MaintenanceFailureSamplePayload:
        return cls(kind=sample.kind, locator=sample.locator, message=sample.message)


class MaintenanceFailureSamplesPayload(SurfacePayloadModel):
    """Bounded failure-sample envelope, mirrored from :class:`BoundedFailureSamples`."""

    samples: tuple[MaintenanceFailureSamplePayload, ...] = ()
    truncated: bool = False

    @classmethod
    def from_bounded(cls, bounded: BoundedFailureSamples) -> MaintenanceFailureSamplesPayload:
        return cls(
            samples=tuple(MaintenanceFailureSamplePayload.from_sample(s) for s in bounded.samples),
            truncated=bounded.truncated,
        )


class MaintenanceScopePayload(SurfacePayloadModel):
    """Typed scope view: target ids + typed scope filter.

    ``filter`` is the serialized form of
    :class:`polylogue.maintenance.scope.MaintenanceScopeFilter` —
    a dict of named scope dimensions (``session_ids``,
    ``provider``, ``source_family``, ``source_root``,
    ``raw_artifact_id``, ``time_range``, ``failure_kind``,
    ``parser_version``). It is typed as ``dict[str, Any]`` at the
    Pydantic boundary so the recursive ``JSONValue`` alias does not
    need a manual rebuild; the typed contract is pinned by
    :class:`MaintenanceScopeFilter` itself.
    """

    targets: tuple[str, ...]
    filter: dict[str, Any]


class MaintenanceOperationEnvelope(SurfacePayloadModel):
    """Surface-agnostic envelope for one maintenance operation.

    Carries everything in :class:`BackfillOperation.to_dict` plus an
    ``origin`` tag and a ``mode`` discriminator so the same shape is
    emitted by CLI, daemon HTTP, and MCP. The envelope is frozen and
    forbids extra fields — adding a field is a deliberate API change,
    not an accidental drift.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    kind: str
    mode: EnvelopeMode
    origin: EnvelopeOrigin
    status: str
    targets: tuple[str, ...]
    scope: MaintenanceScopePayload
    progress: float
    started_at: str | None
    completed_at: str | None
    error: str | None
    affected_rows: int
    estimated_time_s: float
    results: tuple[dict[str, Any], ...]
    reason: str | None
    resume_cursor: str | None
    failure_samples: MaintenanceFailureSamplesPayload
    metrics: dict[str, float]

    def to_dict(self) -> JSONDocument:
        """Return the envelope as a JSON-shaped dict.

        ``mode="json"`` ensures tuples and Pydantic types are reduced
        to JSON primitives so the result is byte-stable across
        surfaces.
        """
        return json_document(self.model_dump(mode="json"))


def envelope_from_operation(
    operation: BackfillOperation,
    *,
    origin: EnvelopeOrigin,
    mode: EnvelopeMode,
) -> MaintenanceOperationEnvelope:
    """Wrap a planner :class:`BackfillOperation` in the shared envelope.

    Parameters
    ----------
    operation:
        Planner result from :func:`polylogue.maintenance.planner.preview_backfill`,
        :func:`polylogue.maintenance.planner.execute_backfill`, or
        :func:`polylogue.maintenance.replay.execute_replay`.
    origin:
        Surface that produced this envelope. ``"cli"``, ``"daemon"``,
        or ``"mcp"``.
    mode:
        ``"preview"`` for read-only inventory, ``"execute"`` for an
        actual or dry-run repair pass.
    """
    scope = operation.scope
    if scope is not None:
        scope_targets = tuple(scope.targets)
        scope_filter_dict = scope.filter.to_dict()
    else:
        scope_targets = tuple(operation.targets)
        scope_filter_dict = MaintenanceScopeFilter().to_dict()
    scope_payload = MaintenanceScopePayload(
        targets=scope_targets,
        filter=scope_filter_dict,
    )
    return MaintenanceOperationEnvelope(
        operation_id=operation.operation_id,
        kind=operation.kind.value,
        mode=mode,
        origin=origin,
        status=operation.status.value,
        targets=tuple(operation.targets),
        scope=scope_payload,
        progress=operation.progress,
        started_at=operation.started_at,
        completed_at=operation.completed_at,
        error=operation.error,
        affected_rows=operation.affected_rows,
        estimated_time_s=operation.estimated_time_s,
        results=tuple(operation.results),
        reason=(operation.reason.value if operation.reason is not None else None),
        resume_cursor=operation.resume_cursor,
        failure_samples=MaintenanceFailureSamplesPayload.from_bounded(operation.failure_samples),
        metrics=dict(operation.metrics),
    )


def envelope_keys() -> frozenset[str]:
    """Return the canonical set of top-level envelope keys.

    Used by the parity test to assert that surfaces agree byte-for-byte
    on the envelope shape.
    """
    return frozenset(MaintenanceOperationEnvelope.model_fields.keys())


__all__ = [
    "EnvelopeMode",
    "EnvelopeOrigin",
    "MaintenanceFailureSamplePayload",
    "MaintenanceFailureSamplesPayload",
    "MaintenanceOperationEnvelope",
    "MaintenanceScopePayload",
    "envelope_from_operation",
    "envelope_keys",
]
