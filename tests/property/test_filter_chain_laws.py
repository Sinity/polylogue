"""Property tests for the SessionFilter chain algebraic laws.

Four laws are verified against a real SQLite archive:

1. Subset law    — filter(X) results ⊆ unfiltered results, for every filter dimension.
2. Monotonicity  — adding more constraints never increases result count.
3. Commutativity — filter(A).filter(B) and filter(B).filter(A) return the same
                   IDs and count.
4. Idempotency   — applying the same filter twice is equivalent to applying it once.

Tests use a fixed seeded database (built once per Hypothesis example via a
temporary directory) so the archive content is deterministic per draw.  Each
test draws filter parameters from ``filter_params_strategy()`` which covers the
main independently-testable filter dimensions.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue import Polylogue
from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from tests.infra.storage_records import SessionBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROVIDERS = ["chatgpt-export", "claude-ai-export", "claude-code-session", "codex-session"]

_SUPPRESS = [
    HealthCheck.function_scoped_fixture,
    HealthCheck.too_slow,
]

# Sessions are created with timestamps distributed across 2024.
_BASE_DT = datetime(2024, 1, 1, tzinfo=timezone.utc)
_MID_DT = datetime(2024, 6, 1, tzinfo=timezone.utc)
_END_DT = datetime(2024, 12, 31, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Filter parameter type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterParams:
    """A single snapshot of filter settings drawn by Hypothesis."""

    provider: str | None
    has_tool_use: bool
    has_thinking: bool
    min_messages: int | None
    max_messages: int | None
    min_words: int | None
    since: datetime | None
    until: datetime | None


# ---------------------------------------------------------------------------
# Hypothesis strategy
# ---------------------------------------------------------------------------


@st.composite
def filter_params_strategy(draw: st.DrawFn) -> FilterParams:
    """Generate a random combination of filter parameters.

    Each dimension is independently optional so that many combinations are
    generated, including the empty filter (all None/False).
    """
    provider: str | None = draw(st.one_of(st.none(), st.sampled_from(_PROVIDERS)))
    has_tool_use: bool = draw(st.booleans())
    has_thinking: bool = draw(st.booleans())
    min_messages: int | None = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=5)))
    max_messages: int | None = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10)))
    min_words_val: int | None = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=50)))

    # Generate an optional date range; ensure since <= until when both present.
    use_dates: bool = draw(st.booleans())
    since: datetime | None = None
    until: datetime | None = None
    if use_dates:
        since_offset = draw(st.integers(min_value=0, max_value=180))
        window = draw(st.integers(min_value=0, max_value=180))
        from datetime import timedelta

        since = _BASE_DT + timedelta(days=since_offset)
        until = since + timedelta(days=window)

    return FilterParams(
        provider=provider,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words_val,
        since=since,
        until=until,
    )


# ---------------------------------------------------------------------------
# Database seeding
# ---------------------------------------------------------------------------


def _seed_diverse_archive(db_path: Path) -> None:
    """Seed a diverse archive covering all filter dimensions.

    Creates 20 sessions spanning:
    - all four providers (5 each)
    - some with tool use, some without
    - some with thinking blocks, some without
    - varying message counts (1–6)
    - varying word counts
    - timestamps spread across 2024
    """
    from datetime import timedelta

    conv_index = 0

    for provider_idx, provider in enumerate(_PROVIDERS):
        for slot in range(5):
            conv_index += 1
            cid = f"conv-{conv_index:03d}"

            # Spread timestamps: first half of year for slot 0-2, second half for 3-4
            day_offset = slot * 60 + provider_idx * 5
            ts = (_BASE_DT + timedelta(days=day_offset)).isoformat()

            seed_provider = provider_from_origin(Origin(provider)).value
            builder = (
                SessionBuilder(db_path, cid)
                .provider(seed_provider)
                .title(f"{provider} conv {slot}")
                .created_at(ts)
                .updated_at(ts)
            )

            # Vary message count: slot 0→1 msg, slot 1→2 msgs, slot 2→3 msgs, etc.
            msg_count = slot + 1
            for msg_idx in range(msg_count):
                words = " ".join(["word"] * (5 * (msg_idx + 1) + slot * 3))
                builder.add_message(
                    role="user" if msg_idx % 2 == 0 else "assistant",
                    text=words,
                    timestamp=ts,
                )

            # Add tool_use block on every other session
            if conv_index % 2 == 0:
                builder.add_message(
                    role="assistant",
                    text="tool response",
                    timestamp=ts,
                    blocks=[{"type": "tool_use", "tool_name": "bash", "tool_id": f"tid-{cid}"}],
                )

            # Add thinking block on every third session
            if conv_index % 3 == 0:
                builder.add_message(
                    role="assistant",
                    text="answer",
                    timestamp=ts,
                    blocks=[{"type": "thinking", "text": "thinking..."}],
                )

            builder.save()


# ---------------------------------------------------------------------------
# Filter application helper
# ---------------------------------------------------------------------------


def _apply_params(filter_obj: Any, params: FilterParams) -> Any:
    """Apply FilterParams to a SessionFilter instance.

    Numeric thresholds (``min_messages``, ``max_messages``, ``min_words``) and
    temporal bounds (``since``, ``until``) on the builder replace prior values
    instead of intersecting. The monotonicity law (more constraints → fewer
    results) only holds when overlapping bounds are composed as a strict
    tightening. We do that explicitly here by reading the current plan and
    keeping the more restrictive value.
    """
    plan = filter_obj._plan
    if params.provider is not None:
        # ``SessionFilter.provider`` accumulates names into an IN-list
        # (OR semantics across the tuple): a second call appends rather than
        # replaces. The commutativity / monotonicity laws here treat the
        # composition of two ``FilterParams`` as logical conjunction
        # (matches A AND matches B). When two provider values disagree,
        # that conjunction is the empty set. We achieve that by overwriting
        # the plan's providers tuple with ``(Provider.UNKNOWN,)`` — no
        # seeded session has ``provider == UNKNOWN``, so the predicate
        # yields zero rows independent of what was there before. (Simply
        # appending an extra provider name would *grow* the IN-list and
        # leak rows from the first filter into the result.)
        #
        # The plan filters on origin tokens internally (#1743); project the
        # requested provider to its origin family token for the comparison.
        params_origin = Origin.from_string(params.provider).value
        current_origins = plan.origins
        if current_origins and params_origin not in current_origins:
            filter_obj._plan = replace(plan, origins=(Origin.UNKNOWN_EXPORT.value,))
            plan = filter_obj._plan
        else:
            filter_obj = filter_obj.origin(params.provider)
    if params.has_tool_use:
        filter_obj = filter_obj.has_tool_use()
    if params.has_thinking:
        filter_obj = filter_obj.has_thinking()
    if params.min_messages is not None:
        current = plan.min_messages
        tightened = params.min_messages if current is None else max(current, params.min_messages)
        filter_obj = filter_obj.min_messages(tightened)
    if params.max_messages is not None:
        current = plan.max_messages
        tightened = params.max_messages if current is None else min(current, params.max_messages)
        filter_obj = filter_obj.max_messages(tightened)
    if params.min_words is not None:
        current = plan.min_words
        tightened = params.min_words if current is None else max(current, params.min_words)
        filter_obj = filter_obj.min_words(tightened)
    if params.since is not None:
        current = plan.since
        tightened = params.since if current is None else max(current, params.since)
        filter_obj = filter_obj.since(tightened)
    if params.until is not None:
        current = plan.until
        tightened = params.until if current is None else min(current, params.until)
        filter_obj = filter_obj.until(tightened)
    return filter_obj


# ---------------------------------------------------------------------------
# Law 1: Subset law
# ---------------------------------------------------------------------------


@given(params=filter_params_strategy())
@settings(max_examples=40, deadline=None, suppress_health_check=_SUPPRESS)
@pytest.mark.asyncio
async def test_filter_result_is_subset_of_unfiltered(params: FilterParams) -> None:
    """filter(X) result IDs must be a subset of the unfiltered result IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        _seed_diverse_archive(db_path)

        async with Polylogue(archive_root=Path(tmpdir), db_path=db_path) as archive:
            all_convs = await archive.filter().list()
            all_ids = {str(c.id) for c in all_convs}

            filtered_convs = await _apply_params(archive.filter(), params).list()
            filtered_ids = {str(c.id) for c in filtered_convs}

        extra = filtered_ids - all_ids
        assert not extra, f"Filtered result contains IDs not in unfiltered set: {extra!r}\nparams={params!r}"


# ---------------------------------------------------------------------------
# Law 2: Monotonicity
# ---------------------------------------------------------------------------


@given(
    params_a=filter_params_strategy(),
    params_b=filter_params_strategy(),
)
@settings(max_examples=40, deadline=None, suppress_health_check=_SUPPRESS)
@pytest.mark.asyncio
async def test_adding_filter_never_increases_count(params_a: FilterParams, params_b: FilterParams) -> None:
    """Applying filter B on top of an already-filtered A result must not exceed A's count.

    More constraints → same or fewer results.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        _seed_diverse_archive(db_path)

        async with Polylogue(archive_root=Path(tmpdir), db_path=db_path) as archive:
            count_a = await _apply_params(archive.filter(), params_a).count()
            # Apply both A and B together
            f_ab = archive.filter()
            f_ab = _apply_params(f_ab, params_a)
            f_ab = _apply_params(f_ab, params_b)
            count_ab = await f_ab.count()

        assert count_ab <= count_a, (
            f"Adding filter B increased count: {count_a} → {count_ab}\nparams_a={params_a!r}\nparams_b={params_b!r}"
        )


# ---------------------------------------------------------------------------
# Law 3: Commutativity
# ---------------------------------------------------------------------------


@given(
    params_a=filter_params_strategy(),
    params_b=filter_params_strategy(),
)
@settings(max_examples=40, deadline=None, suppress_health_check=_SUPPRESS)
@pytest.mark.asyncio
async def test_filter_application_order_is_commutative(params_a: FilterParams, params_b: FilterParams) -> None:
    """filter(A).filter(B) and filter(B).filter(A) must yield identical IDs and counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        _seed_diverse_archive(db_path)

        async with Polylogue(archive_root=Path(tmpdir), db_path=db_path) as archive:
            # A then B
            f_ab = archive.filter()
            f_ab = _apply_params(f_ab, params_a)
            f_ab = _apply_params(f_ab, params_b)
            convs_ab = await f_ab.list()
            ids_ab = {str(c.id) for c in convs_ab}

            # B then A
            f_ba = archive.filter()
            f_ba = _apply_params(f_ba, params_b)
            f_ba = _apply_params(f_ba, params_a)
            convs_ba = await f_ba.list()
            ids_ba = {str(c.id) for c in convs_ba}

        assert ids_ab == ids_ba, (
            f"Filter order changed results:\n"
            f"  A→B only: {ids_ab - ids_ba!r}\n"
            f"  B→A only: {ids_ba - ids_ab!r}\n"
            f"params_a={params_a!r}\n"
            f"params_b={params_b!r}"
        )


# ---------------------------------------------------------------------------
# Law 4: Idempotency
# ---------------------------------------------------------------------------


@given(params=filter_params_strategy())
@settings(max_examples=40, deadline=None, suppress_health_check=_SUPPRESS)
@pytest.mark.asyncio
async def test_filter_applied_twice_equals_once(params: FilterParams) -> None:
    """Applying the same filter params twice must yield the same result as once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        _seed_diverse_archive(db_path)

        async with Polylogue(archive_root=Path(tmpdir), db_path=db_path) as archive:
            # Apply once
            convs_once = await _apply_params(archive.filter(), params).list()
            ids_once = {str(c.id) for c in convs_once}

            # Apply twice
            f_twice = archive.filter()
            f_twice = _apply_params(f_twice, params)
            f_twice = _apply_params(f_twice, params)
            convs_twice = await f_twice.list()
            ids_twice = {str(c.id) for c in convs_twice}

        assert ids_once == ids_twice, (
            f"Applying filter twice changed results:\n"
            f"  once-only: {ids_once - ids_twice!r}\n"
            f"  twice-only: {ids_twice - ids_once!r}\n"
            f"params={params!r}"
        )
