"""Canonical bounded and informative rows for session list/search surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from os.path import basename
from typing import Any

from polylogue.archive.query.search_hits import bound_display_title, bound_search_snippet

TITLE_BUDGET = 96
SNIPPET_BUDGET = 320
OUTCOME_VALUES = frozenset({"completed", "failed", "abandoned", "unknown"})


def _value(item: object, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def relative_time(value: datetime | None, *, now: datetime | None = None) -> str:
    if value is None:
        return "unknown"
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    seconds = max(0, int((current - value).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 604800:
        return f"{seconds // 86400}d ago"
    return f"{seconds // 604800}w ago"


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class SessionRowProjection:
    """One bounded list-row contract shared by JSON, text, CSV, and select."""

    id: str
    origin: str
    title: str
    date: str | None
    relative_time: str
    outcome: str
    cost_usd: float | None
    repo: str | None
    cwd_display: str | None
    message_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "origin": self.origin,
            "title": self.title,
            "date": self.date,
            "relative_time": self.relative_time,
            "outcome": self.outcome,
            "cost_usd": self.cost_usd,
            "repo": self.repo,
            "cwd_display": self.cwd_display,
            "message_count": self.message_count,
        }


def session_row(item: object, *, message_count: int | None = None) -> SessionRowProjection:
    session_id = str(_value(item, "id", _value(item, "session_id", "")))
    date_value = _as_datetime(_value(item, "display_date") or _value(item, "updated_at") or _value(item, "created_at"))
    directories = _value(item, "working_directories", ()) or ()
    cwd = next((str(path).strip() for path in directories if str(path).strip()), None)
    repo = _value(item, "git_repository_url")
    count = message_count if message_count is not None else _value(item, "message_count")
    if count is None:
        count = len(_value(item, "messages", ()) or ())
    return SessionRowProjection(
        id=session_id,
        origin=str(_value(item, "origin", "unknown")),
        title=bound_display_title(
            _value(item, "display_title") or _value(item, "display_label") or _value(item, "title"),
            session_id,
            max_chars=TITLE_BUDGET,
        ),
        date=date_value.strftime("%Y-%m-%d") if date_value else None,
        relative_time=relative_time(date_value),
        outcome=(state if state in OUTCOME_VALUES else "unknown")
        if (state := str(_value(item, "terminal_state") or "unknown"))
        else "unknown",
        cost_usd=(float(_value(item, "total_cost_usd")) if _value(item, "total_cost_usd") is not None else None),
        repo=basename(str(repo).rstrip("/")) if repo else None,
        cwd_display=basename(cwd.rstrip("/")) if cwd else None,
        message_count=int(count or 0),
    )


def search_row(hit: object, *, message_count: int | None = None) -> tuple[SessionRowProjection, str | None]:
    row = session_row(_value(hit, "summary"), message_count=message_count)
    return row, bound_search_snippet(_value(hit, "snippet"), max_chars=SNIPPET_BUDGET)


__all__ = ["TITLE_BUDGET", "SNIPPET_BUDGET", "SessionRowProjection", "relative_time", "session_row", "search_row"]
