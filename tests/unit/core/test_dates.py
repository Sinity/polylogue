"""Relative-date parsing determinism (polylogue-cpf.6).

``parse_date`` uses ``datetime.now(tz=timezone.utc)`` as ``dateparser``'s
``RELATIVE_BASE`` on every call — not frozen at import time (an earlier,
incorrect claim). The real gap this bead closed was proving the existing
``frozen_clock`` test infrastructure actually reaches it: ``frozen_clock``
can only patch a module's ``datetime`` symbol if the test tells it to via
``@pytest.mark.frozen_clock_modules(...)``. ``polylogue.core.dates`` does
``from datetime import datetime`` at module scope, so it was always
patchable this way — the seam already existed, it just had no regression
test proving `since:7d` resolves deterministically end to end, from the
query DSL down through ``SessionQuerySpec.to_plan()`` to ``parse_date``.

These tests pin: (1) ``parse_date`` itself is deterministic under a frozen
clock and shifts only when the clock is advanced; (2) the full query-spec
lowering path (``since:7d`` -> ``SessionQuerySpec`` -> ``to_plan()``) is
equally deterministic; (3) no other query-time parsing module reaches for
its own ``datetime.now()`` outside this one chokepoint (a grep-based
audit, not a mypy/runtime check — there is no lint plugin for this in the
repo yet).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from polylogue.core.dates import parse_date
from tests.infra.frozen_clock import DEFAULT_FROZEN_EPOCH, freeze_clock


@pytest.mark.frozen_clock_modules("polylogue.core.dates")
def test_parse_date_relative_is_deterministic_under_frozen_clock(frozen_clock: object) -> None:
    first = parse_date("7 days ago")
    second = parse_date("7 days ago")
    assert first == second
    assert first is not None
    assert first.isoformat() == "2023-11-07T22:13:20+00:00"


def test_parse_date_relative_shifts_only_when_clock_advances() -> None:
    with freeze_clock(patch_datetime_in_modules=["polylogue.core.dates"]) as clock:
        before = parse_date("7 days ago")
        clock.advance(3600 * 24 * 30)
        after = parse_date("7 days ago")
    assert before != after
    assert after is not None and before is not None
    assert (after - before).days == 30


@pytest.mark.frozen_clock_modules("polylogue.core.dates")
def test_since_7d_query_lowering_is_deterministic(frozen_clock: object) -> None:
    """The DSL's since:7d -> SessionQuerySpec -> to_plan() path is
    deterministic end to end, not just the parse_date leaf call."""
    from polylogue.archive.query.spec import SessionQuerySpec

    spec = SessionQuerySpec.from_params({"since": "7 days ago"})
    plan_a = spec.to_plan()
    plan_b = spec.to_plan()

    assert plan_a.since == plan_b.since
    assert plan_a.since is not None
    assert plan_a.since.isoformat() == "2023-11-07T22:13:20+00:00"


def test_since_7d_query_lowering_shifts_only_with_injected_clock() -> None:
    from polylogue.archive.query.spec import SessionQuerySpec

    spec = SessionQuerySpec.from_params({"since": "7 days ago"})
    with freeze_clock(patch_datetime_in_modules=["polylogue.core.dates"]) as clock:
        before = spec.to_plan().since
        clock.set_time(DEFAULT_FROZEN_EPOCH + 3600 * 24 * 30)
        after = spec.to_plan().since
    assert before != after
    assert after is not None and before is not None
    assert (after - before).days == 30


def test_no_query_time_module_bypasses_the_parse_date_seam() -> None:
    """Audit: no module under archive/query or archive/filter calls
    datetime.now() directly for relative-date resolution — the only
    clock read in the query-time parsing path must be inside
    polylogue.core.dates.parse_date, or frozen_clock cannot reach it."""
    package_root = Path(__file__).resolve().parents[3] / "polylogue"
    scan_dirs = [package_root / "archive" / "query", package_root / "archive" / "filter"]
    now_call = re.compile(r"datetime\.now\(")

    violations: list[str] = []
    for scan_dir in scan_dirs:
        for path in sorted(scan_dir.rglob("*.py")):
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if now_call.search(line):
                    violations.append(f"{path.relative_to(package_root.parent)}:{lineno}: {line.strip()}")

    assert violations == [], f"query-time parsing modules reading the clock outside parse_date: {violations}"
