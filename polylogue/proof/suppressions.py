"""Suppression registry — intentional verification skips with expiry dates.

Every suppression in ``docs/plans/suppressions.yaml`` declares an expiry
date. After that date the verification lint fails, forcing review and
either renewal or removal.

Suppressions are YAML (same pattern as all evidence-driven structural
specs under ``docs/plans/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REGISTRY = _REPO_ROOT / "docs" / "plans" / "suppressions.yaml"


def _resolve_registry(*, registry: Path | None) -> Path:
    return registry if registry is not None else _DEFAULT_REGISTRY


def _string_tuple(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if isinstance(item, str) and item.strip())


@dataclass(frozen=True, slots=True)
class Suppression:
    """A declared intentional verification skip with an expiry date."""

    id: str
    reason: str
    expires_at: str
    issue: str | None = None
    paths: tuple[str, ...] = field(default_factory=tuple)
    claims: tuple[str, ...] = field(default_factory=tuple)


def _parse_date(raw: str) -> date:
    """Parse an ISO-8601 date string, raising on invalid input."""
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.date()


def load_suppressions(*, registry: Path | None = None) -> list[Suppression]:
    """Read the suppression registry and return the list of Suppressions.

    Parameters
    ----------
    registry:
        Path to the suppressions YAML file. Defaults to
        ``docs/plans/suppressions.yaml``.

    Returns
    -------
    list[Suppression]
        The declared suppressions in document order.
    """
    path = _resolve_registry(registry=registry)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []
    entries = raw.get("suppressions", [])
    if not isinstance(entries, list):
        return []
    result: list[Suppression] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        result.append(
            Suppression(
                id=str(entry.get("id", "")),
                reason=str(entry.get("reason", "")),
                expires_at=str(entry.get("expires_at", "")),
                issue=str(entry.get("issue")) if entry.get("issue") else None,
                paths=_string_tuple(entry.get("paths", [])),
                claims=_string_tuple(entry.get("claims", [])),
            )
        )
    return result


def is_expired(suppression: Suppression, *, now: date | None = None) -> bool:
    """Check whether a suppression has passed its expiry date.

    Parameters
    ----------
    suppression:
        The suppression to check.
    now:
        The reference date. Defaults to today in UTC.

    Returns
    -------
    bool
        ``True`` if the suppression has expired (or its expiry date is
        unparseable), ``False`` otherwise.
    """
    if not suppression.expires_at:
        return True
    try:
        expires = _parse_date(suppression.expires_at)
    except (ValueError, TypeError):
        return True
    reference = now if now is not None else datetime.now(timezone.utc).date()
    return expires < reference


def validate_suppressions(
    suppressions: list[Suppression] | None = None,
    *,
    now: date | None = None,
    registry: Path | None = None,
) -> list[str]:
    """Validate suppression registry for expired entries.

    Parameters
    ----------
    suppressions:
        Pre-loaded suppression list. If ``None``, loads from the registry.
    now:
        Reference date for expiry checking.
    registry:
        Path to the suppressions YAML file (only used when ``suppressions``
        is ``None``).

    Returns
    -------
    list[str]
        Error messages for every expired or invalid suppression. Empty
        means all suppressions are current.
    """
    if suppressions is None:
        suppressions = load_suppressions(registry=registry)
    errors: list[str] = []
    for suppression in suppressions:
        if not suppression.id:
            errors.append("suppression entry missing 'id'")
            continue
        if not suppression.expires_at:
            errors.append(f"suppression {suppression.id!r}: missing 'expires_at'")
            continue
        try:
            _parse_date(suppression.expires_at)
        except (ValueError, TypeError):
            errors.append(f"suppression {suppression.id!r}: invalid 'expires_at': {suppression.expires_at!r}")
            continue
        if is_expired(suppression, now=now):
            errors.append(
                f"suppression {suppression.id!r}: expired at {suppression.expires_at} — review and remove or extend"
            )
    return errors


__all__ = [
    "Suppression",
    "is_expired",
    "load_suppressions",
    "validate_suppressions",
]
