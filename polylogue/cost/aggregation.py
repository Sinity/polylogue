"""Daily-usage aggregation over typed cost insights (#1138).

The :func:`build_cycle_outlook` engine in :mod:`polylogue.cost.outlook`
consumes a sequence of typed :class:`DailyUsage` rows. The archive
already materializes per-session cost estimates through
:class:`polylogue.insights.archive.SessionCostInsight`. This module is
the pure-function bridge: it folds a list of session-cost insights into
one :class:`DailyUsage` row per UTC day, in USD basis.

The aggregator deliberately ignores rows without a parseable
``created_at`` timestamp or a positive ``total_usd`` — the cost outlook
must not pretend an unpriced session contributed to the burn rate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime

from polylogue.cost.outlook import DailyUsage
from polylogue.insights.archive import SessionCostInsight

__all__ = ["session_costs_to_daily_usd"]


def session_costs_to_daily_usd(insights: Iterable[SessionCostInsight]) -> list[DailyUsage]:
    """Aggregate ``insights`` into one ``DailyUsage`` row per UTC day.

    The resulting rows carry ``basis="usd"`` and ``amount`` equal to the
    sum of ``estimate.total_usd`` for that day. Rows are returned sorted
    by day, deterministically, so snapshot tests over CLI/MCP payloads
    are stable.

    Sessions without a parseable ``created_at`` or with
    ``total_usd <= 0`` are excluded — they cannot contribute to a cycle
    burn rate without misrepresenting coverage.
    """
    daily_totals: dict[str, float] = defaultdict(float)
    for insight in insights:
        if insight.estimate.total_usd <= 0.0:
            continue
        ts = insight.created_at
        if not ts:
            continue
        try:
            day_iso = _parse_iso_day(ts)
        except ValueError:
            continue
        daily_totals[day_iso] += float(insight.estimate.total_usd)

    return [
        DailyUsage(day=datetime.fromisoformat(day_iso).date(), basis="usd", amount=amount)
        for day_iso, amount in sorted(daily_totals.items())
    ]


def _parse_iso_day(ts: str) -> str:
    """Return the ``YYYY-MM-DD`` UTC date for ``ts``.

    Accepts trailing ``Z`` as a synonym for ``+00:00``. Raises
    :class:`ValueError` for unparseable strings — callers are expected
    to swallow it as a coverage gap rather than aborting the whole
    aggregation.
    """
    normalized = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    parsed = datetime.fromisoformat(normalized)
    return parsed.date().isoformat()
