"""Typed scope filters for maintenance backfill operations (issue #1196).

Replaces the legacy free-form ``MaintenanceScope.filter`` JSON dict with
a typed :class:`MaintenanceScopeFilter` Pydantic model so every surface
(CLI ``polylogue ops maintenance plan/run``, daemon HTTP, MCP
``maintenance_preview``/``maintenance_execute``) agrees on the same
scope dimensions.

The filter dimensions match the ones #996 AC #1 names explicitly:

* ``session_ids`` — restrict to a specific set of session ids;
* ``provider`` — restrict to one provider name (e.g. ``"claude"``);
* ``source_family`` — restrict to one source family (e.g.
  ``"claude-code-session"``);
* ``source_root`` — restrict to artifacts acquired under one runtime
  root (e.g. ``~/.claude/projects``);
* ``time_range`` — inclusive ``(since, until)`` ISO-8601 window;
* ``failure_kind`` — restrict to attempts that failed with one kind;
* ``parser_version`` — restrict to one parser/materializer version.

The filter is intentionally *target-owned* at the repair-fn boundary:
each repair fn declares which dimensions it knows how to honor and must
not advertise narrower operator behavior than it actually applies. For
example, session-insight repair honors ``session_ids``. Other dimensions
remain advisory until a target pins their contract.

The filter round-trips through :meth:`MaintenanceScopeFilter.to_dict`
/ :meth:`MaintenanceScopeFilter.from_dict` so the CLI ``--output-format
json``, the daemon HTTP body, and the MCP tool args all carry the
exact same shape.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, field_validator

from polylogue.surfaces.payloads import SurfacePayloadModel


class MaintenanceScopeFilter(SurfacePayloadModel):
    """Typed scope filter for a maintenance backfill operation.

    Every field is optional; ``None`` means "do not narrow on this
    dimension". An entirely empty filter is the canonical full-scope
    request — :func:`is_empty` returns ``True`` for that case.

    The model is frozen and forbids extra fields, so surfaces cannot
    silently introduce new dimensions without an explicit change to
    this contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_ids: tuple[str, ...] | None = None
    provider: str | None = None
    source_family: str | None = None
    source_root: Path | None = None
    time_range: tuple[datetime, datetime] | None = None
    failure_kind: str | None = None
    parser_version: str | None = None

    @field_validator("session_ids", mode="before")
    @classmethod
    def _coerce_session_ids(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        return value

    @field_validator("time_range", mode="before")
    @classmethod
    def _coerce_time_range(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("time_range must be a (since, until) pair")
            since, until = value
            return (_coerce_datetime(since), _coerce_datetime(until))
        return value

    @field_validator("source_root", mode="before")
    @classmethod
    def _coerce_source_root(cls, value: Any) -> Any:
        if value is None or isinstance(value, Path):
            return value
        return Path(str(value))

    def is_empty(self) -> bool:
        """True when no scope dimension is set (full-scope request)."""
        return (
            self.session_ids is None
            and self.provider is None
            and self.source_family is None
            and self.source_root is None
            and self.time_range is None
            and self.failure_kind is None
            and self.parser_version is None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the filter as a JSON-shaped dict.

        ``mode="json"`` coerces tuples to lists, ``Path`` to string,
        and ``datetime`` to ISO-8601 strings so the result is
        byte-stable across surfaces.
        """
        return self.model_dump(mode="json", exclude_none=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> MaintenanceScopeFilter:
        """Reconstruct a filter from a JSON-shaped dict.

        Missing keys default to ``None``. ``None`` and ``{}`` both
        round-trip to an empty filter.
        """
        if payload is None or not payload:
            return cls()
        return cls.model_validate(payload)


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(text)
    raise TypeError(f"Cannot coerce {value!r} to datetime")


__all__ = ["MaintenanceScopeFilter"]
