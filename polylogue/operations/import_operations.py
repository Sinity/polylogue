"""Validated browser upload metadata; execution uses the daemon operation protocol."""

from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict

from polylogue.core.json import JSONDocument, json_document
from polylogue.operations.specs import OperationKind
from polylogue.surfaces.payloads import SurfacePayloadModel


class ImportRequest(SurfacePayloadModel):
    """Typed input for one import operation.

    Carries the four scheduling fields (source_path, staged_path,
    source_name, idempotency_key). ``operation_kind`` is a fixed
    ``ClassVar`` (not a pydantic field — it does not appear in
    :meth:`to_dict`'s wire payload, matching the shape from before the
    polylogue-a7xr.14 collapse) since this model only ever carries
    :attr:`OperationKind.IMPORT`.

    Surfaces validate the request with Pydantic at the boundary — invalid
    types or unknown extra keys raise :class:`pydantic.ValidationError`
    before any work is scheduled.

    Fields
    ------
    source_path:
        Path or URI the caller wants to ingest. Required. The contract
        does not constrain the value to an existing filesystem path so
        URI-shaped sources (https, drive://) remain compatible.
    source_name:
        Caller-supplied label that names the logical source. Required so
        the surface can record per-source progress without parsing the
        path.
    staged_path:
        Optional path the surface should treat as the actual byte source.
        Surfaces that copy/upload before processing (daemon HTTP ingest,
        ``polylogue ingest --copy``) populate this; in-place surfaces
        leave it ``None``.
    idempotency_key:
        Optional caller-supplied key so retries do not schedule duplicate
        work.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_kind: ClassVar[OperationKind] = OperationKind.IMPORT

    source_path: str
    source_name: str
    staged_path: str | None = None
    idempotency_key: str | None = None

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


__all__ = ["ImportRequest"]
