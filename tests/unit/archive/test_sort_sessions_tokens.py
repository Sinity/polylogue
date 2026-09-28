"""``sort_sessions``'s ``tokens`` key must mirror SQL's measured/unmeasured semantics.

Mirrors ``tests/unit/storage/test_token_sort_unknown_usage.py`` at the
Python-composed-sort layer (``polylogue.archive.query.sorting.sort_sessions``,
used for lineage-composed sessions where SQL's own stored per-session counters
cover only a child's own tail).
"""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.archive.query.sorting import sort_sessions
from tests.infra.builders import make_conv, make_msg


@dataclass(frozen=True)
class _Plan:
    sort: str | None
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
