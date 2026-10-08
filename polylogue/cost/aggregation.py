"""Daily-usage aggregation over typed cost insights (#1138).

The :func:`build_cycle_outlook` engine in :mod:`polylogue.cost.outlook`
consumes a sequence of typed :class:`DailyUsage` rows. The archive
already materializes per-session cost estimates through
:class:`polylogue.analysis.archive.SessionCostInsight`. This module is
the pure-function bridge: it folds a list of session-cost insights into
one :class:`DailyUsage` row per UTC day, in USD basis.

The aggregator deliberately ignores rows without a parseable
``created_at`` timestamp or a positive ``total_usd`` — the cost outlook
must not pretend an unpriced session contributed to the burn rate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime

from polylogue.analysis.archive import SessionCostInsight
from polylogue.cost.outlook import DailyUsage

__all__ = ["session_costs_to_daily_usd"]


def session_costs_to_daily_usd(
    insights: Iterable[SessionCostInsight], *, as_of: datetime | None = None
) -> list[DailyUsage]:
    """Aggregate ``insights`` into one ``DailyUsage`` row per UTC day.

    The resulting rows carry ``basis="usd"`` and ``amount`` equal to the
    sum of ``estimate.total_usd`` for that day. Rows are returned sorted
    by day, deterministically, so snapshot tests over CLI/MCP payloads
    are stable.

    ``as_of`` includes observations at that exact instant and excludes later
    timestamps before their UTC dates are folded together.

    Sessions without a parseable ``created_at`` or with
    ``total_usd <= 0`` are excluded — they cannot contribute to a cycle
    burn rate without misrepresenting coverage.
    """
    cutoff = as_of.astimezone(UTC) if as_of is not None else None
    daily_totals: dict[str, float] = defaultdict(float)
    for insight in insights:
        if insight.estimate.total_usd is None or insight.estimate.total_usd <= 0.0:
            continue
        ts = insight.created_at
        if not ts:
            continue
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            # Archive timestamps without an offset use the UTC convention.
            parsed = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
        except ValueError:
            continue
        if cutoff is not None and parsed > cutoff:
            continue
        daily_totals[parsed.date().isoformat()] += float(insight.estimate.total_usd)

    return [
        DailyUsage(day=datetime.fromisoformat(day_iso).date(), basis="usd", amount=amount)
        for day_iso, amount in sorted(daily_totals.items())
    ]
