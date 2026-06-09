"""Typed invalidation reasons for maintenance planning.

A derived read model can be stale for many distinct reasons; the
maintenance planner needs to surface *why* it scheduled work so that
operators and downstream tooling can reason about the resulting
:class:`~polylogue.maintenance.planner.BackfillOperation`. Until this
module landed, the planner only carried a free-form ``detail`` string
inherited from :class:`~polylogue.maintenance.models.DerivedModelStatus`,
which lost the structured reason at the surface boundary.

The enum here is intentionally small and closed. New reasons must be
added explicitly: callers persisting the value (logs, JSON envelopes,
fixtures) rely on the string form being stable.
"""

from __future__ import annotations

from enum import Enum


class InvalidationReason(str, Enum):
    """Why a derived read model needs to be (re)built.

    The values are stable wire-strings: do not rename without a
    deliberate movement of stored evidence and fixtures.
    """

    #: The materialized read model is missing entirely for some
    #: subset of the source documents.
    MISSING = "missing"

    #: The materializer that produced existing rows is older than the
    #: registered ``materializer_version``; the rows are stale even if
    #: every source document has a row.
    STALE_MATERIALIZER_VERSION = "stale_materializer_version"

    #: A source document was re-ingested with a different content hash;
    #: the derived rows attached to it must be rebuilt.
    SOURCE_CHANGED = "source_changed"

    #: A parser or provider/schema version bumped, and the upstream
    #: validation/typed-record shape changed.
    PARSER_OR_SCHEMA_CHANGED = "parser_or_schema_changed"

    #: Configuration or model snapshot used by the materializer
    #: changed (e.g. embedding model id, redaction policy version).
    CONFIG_OR_MODEL_SNAPSHOT_CHANGED = "config_or_model_snapshot_changed"

    #: Catch-all for staleness whose exact cause is not (yet)
    #: classified by the planner.
    UNKNOWN = "unknown"


__all__ = ["InvalidationReason"]
