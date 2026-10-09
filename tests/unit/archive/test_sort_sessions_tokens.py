"""``sort_sessions``'s ``tokens`` key must mirror SQL's measured/unmeasured semantics.

Mirrors ``tests/unit/storage/test_token_sort_unknown_usage.py`` at the
Python-composed-sort layer (``polylogue.archive.query.sorting.sort_sessions``,
used for lineage-composed sessions where SQL's own stored per-session counters
cover only a child's own tail).
"""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.archive.filter.types import SortField
from polylogue.archive.query.sorting import sort_sessions
from tests.infra.builders import make_conv, make_msg


@dataclass(frozen=True)
class _Plan:
    sort: SortField | None
    reverse: bool
    limit: int | None = None
    sample: int | None = None


def test_measured_zero_ranks_ahead_of_unmeasured_in_both_directions() -> None:
    unmeasured = make_conv(id="unmeasured", messages=[make_msg(id="m1", text="body")])
    measured_zero = make_conv(
        id="measured-zero",
        messages=[make_msg(id="m1", text="body", input_tokens=0, output_tokens=0)],
    )
    measured_ten = make_conv(
        id="measured-ten",
        messages=[make_msg(id="m1", text="body", input_tokens=10, output_tokens=0)],
    )
    sessions = [unmeasured, measured_zero, measured_ten]

    ascending = sort_sessions(_Plan(sort="tokens", reverse=True), sessions)
    assert [str(session.id) for session in ascending] == ["measured-zero", "measured-ten", "unmeasured"]

    descending = sort_sessions(_Plan(sort="tokens", reverse=False), sessions)
    assert [str(session.id) for session in descending] == ["measured-ten", "measured-zero", "unmeasured"]


def test_token_total_sums_every_measured_field_not_text_length() -> None:
    """A short reply with real usage outranks a long, never-measured wall of text."""
    long_unmeasured = make_conv(id="long", messages=[make_msg(id="m1", text="x" * 4000)])
    short_measured = make_conv(
        id="short",
        messages=[make_msg(id="m1", text="hi", input_tokens=50, output_tokens=50)],
    )

    descending = sort_sessions(_Plan(sort="tokens", reverse=False), [long_unmeasured, short_measured])

    assert [str(session.id) for session in descending] == ["short", "long"]


def test_composed_count_ties_break_by_sort_key_then_session_id() -> None:
    """Anti-vacuity (Codex P2, #5695): sort by the count alone and a tie keeps
    its incoming order, so the older session stays first under ``limit=1``
    although SQL breaks the tie by ``sort_key_ms`` then ``session_id``."""
    from datetime import datetime, timezone

    older = make_conv(
        id="older",
        messages=[make_msg(id="m1", text="a"), make_msg(id="m2", text="b")],
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    newer = make_conv(
        id="newer",
        messages=[make_msg(id="m1", text="a"), make_msg(id="m2", text="b")],
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    descending = sort_sessions(_Plan(sort="messages", reverse=False), [older, newer])
    assert [str(session.id) for session in descending] == ["newer", "older"]
    ascending = sort_sessions(_Plan(sort="messages", reverse=True), [newer, older])
    assert [str(session.id) for session in ascending] == ["older", "newer"]


def test_comparison_values_preserve_naive_utc_offsets_and_microseconds() -> None:
    from datetime import datetime, timedelta, timezone

    from polylogue.archive.query.sorting import session_order_values

    plan = _Plan(sort="date", reverse=True)
    naive = make_conv(id="naive", updated_at=datetime(2026, 1, 1, 0, 0, 0, 1))
    offset = make_conv(id="offset", updated_at=datetime(2026, 1, 1, 2, 0, 0, 1, tzinfo=timezone(timedelta(hours=2))))
    later = make_conv(id="later", updated_at=datetime(2026, 1, 1, 0, 0, 0, 2, tzinfo=timezone.utc))
    assert session_order_values(plan, naive)[:3] == session_order_values(plan, offset)[:3]
    assert session_order_values(plan, later)[1] == session_order_values(plan, naive)[1] + 1
    assert [str(row.id) for row in sort_sessions(plan, [later, offset, naive])] == ["naive", "offset", "later"]
