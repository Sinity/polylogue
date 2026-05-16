"""SQL injection boundary properties for user-controlled filter inputs.

`ConversationFilter` is the single funnel for user-controlled query inputs
flowing in from the CLI, MCP, and Python API surfaces. This module
property-fuzzes that funnel with strings packed with SQL/FTS5 metacharacters
and asserts:

1. The query never raises a SQLite ``OperationalError`` / ``DatabaseError`` /
   ``ProgrammingError`` — the parameterization/escaping layer must keep all
   metacharacters as data, never as syntax.
2. Sentinel rows pre-loaded into the archive are not deleted, corrupted, or
   exfiltrated by any input. After the query, the same sentinel rows are
   still queryable by a clean-text search.
3. The number of rows returned for an arbitrary search string is bounded by
   the universe of rows in the archive (no negative counts, no row inflation
   via injected ``UNION SELECT``).
4. Empty / whitespace-only / pure-metacharacter inputs degrade to "match
   nothing or match everything", never to a SQL error.

These properties are invariants of the *whole stack*: filter chain →
storage query layer → FTS5 escape helpers → SQLite. A regression at any
layer (e.g. a future filter that interpolates a raw string into SQL)
should make a property test fail.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.archive.filter.filters import ConversationFilter
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.storage_records import ConversationBuilder

# Strings packed with SQL/FTS5 metacharacters, classic injection payloads,
# and unicode/control variants. Hypothesis composes around these via .one_of
# below; the explicit list also forms the deterministic regression spine.
_KNOWN_INJECTION_PAYLOADS: tuple[str, ...] = (
    "'; DROP TABLE conversations; --",
    "' OR '1'='1",
    '" OR ""="',
    "%' OR '1'='1' --",
    "); DELETE FROM messages WHERE ('1'='1",
    "1; SELECT * FROM sqlite_master --",
    "UNION SELECT name FROM sqlite_master",
    "*",
    "**",
    "AND OR NOT",
    "NEAR(foo)",
    '"hello',
    'foo"bar',
    "foo*bar",
    "foo:bar",
    "(((",
    ")))",
    "[]",
    "{}",
    "\x00",
    "\x00\x01\x02\x03",
    "\u202e\u202d",  # RTL/LTR override (Trojan Source style)
    "‏‏\u2066\u2067",
    "%00",
    "../../etc/passwd",
    "$" * 100,
    "'" * 50,
    '"' * 50,
    "\\",
    '\\"',
    "\n; DROP TABLE messages;",
)

_SENTINEL_SUBSTRING = "ZQXWVPLRMS_sentinel_marker"


@pytest.fixture
def seeded_repo(tmp_path: Path) -> ConversationRepository:
    """A small archive with sentinel rows used by every injection property."""
    db_path = tmp_path / "injection.db"
    with open_connection(db_path) as conn:
        rebuild_index(conn)
    for idx in range(3):
        builder = ConversationBuilder(db_path, f"sent-{idx}").provider("claude-ai")
        builder = builder.title(f"sentinel {idx} {_SENTINEL_SUBSTRING}")
        builder = builder.add_message(f"m{idx}", role="user", text=_SENTINEL_SUBSTRING)
        builder.save()
    backend = SQLiteBackend(db_path)
    return ConversationRepository(backend=backend)


async def _sentinels_still_present(repo: ConversationRepository) -> int:
    summaries = await ConversationFilter(repo).contains(_SENTINEL_SUBSTRING).list_summaries()
    return len(summaries)


injection_strings: st.SearchStrategy[str] = st.one_of(
    st.sampled_from(_KNOWN_INJECTION_PAYLOADS),
    st.text(
        alphabet=st.characters(blacklist_categories=["Cs"]),
        min_size=0,
        max_size=120,
    ),
)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=injection_strings)
@pytest.mark.asyncio
async def test_contains_never_raises_sql_error(seeded_repo: ConversationRepository, payload: str) -> None:
    """`.contains(payload)` always completes; result count is bounded."""
    sentinels_before = await _sentinels_still_present(seeded_repo)
    try:
        results = await ConversationFilter(seeded_repo).contains(payload).list_summaries()
    except Exception as exc:
        pytest.fail(
            f"ConversationFilter.contains({payload!r}) raised {type(exc).__name__}: {exc}. "
            "Filter inputs must be parameterized or escaped, never interpolated as SQL."
        )
    assert isinstance(results, list)
    assert 0 <= len(results) <= 64, (
        f"Unbounded result count {len(results)} for injection payload {payload!r}; "
        "possible UNION injection or row inflation."
    )
    sentinels_after = await _sentinels_still_present(seeded_repo)
    assert sentinels_after == sentinels_before, (
        f"Sentinel rows changed from {sentinels_before} to {sentinels_after} after "
        f"querying {payload!r}; possible destructive injection."
    )


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=injection_strings)
@pytest.mark.asyncio
async def test_title_never_raises_sql_error(seeded_repo: ConversationRepository, payload: str) -> None:
    sentinels_before = await _sentinels_still_present(seeded_repo)
    try:
        results = await ConversationFilter(seeded_repo).title(payload).list_summaries()
    except Exception as exc:
        pytest.fail(f"ConversationFilter.title({payload!r}) raised {type(exc).__name__}: {exc}")
    assert isinstance(results, list)
    sentinels_after = await _sentinels_still_present(seeded_repo)
    assert sentinels_after == sentinels_before


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=injection_strings)
@pytest.mark.asyncio
async def test_tag_filter_never_raises_sql_error(seeded_repo: ConversationRepository, payload: str) -> None:
    sentinels_before = await _sentinels_still_present(seeded_repo)
    try:
        results = await ConversationFilter(seeded_repo).tag(payload).list_summaries()
    except Exception as exc:
        pytest.fail(f"ConversationFilter.tag({payload!r}) raised {type(exc).__name__}: {exc}")
    assert isinstance(results, list)
    sentinels_after = await _sentinels_still_present(seeded_repo)
    assert sentinels_after == sentinels_before


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=injection_strings)
@pytest.mark.asyncio
async def test_provider_filter_never_raises_sql_error(seeded_repo: ConversationRepository, payload: str) -> None:
    sentinels_before = await _sentinels_still_present(seeded_repo)
    try:
        results = await ConversationFilter(seeded_repo).provider(payload).list_summaries()
    except Exception as exc:
        pytest.fail(f"ConversationFilter.provider({payload!r}) raised {type(exc).__name__}: {exc}")
    assert isinstance(results, list)
    sentinels_after = await _sentinels_still_present(seeded_repo)
    assert sentinels_after == sentinels_before


@pytest.mark.parametrize(
    "builder",
    [
        lambda repo, payload: ConversationFilter(repo).contains(payload),
        lambda repo, payload: ConversationFilter(repo).title(payload),
        lambda repo, payload: ConversationFilter(repo).tag(payload),
    ],
)
@pytest.mark.parametrize(
    "payload",
    ["", "   ", "\n", "\t", "*", "***", "()", '"', "\\", "AND", "OR", "NOT"],
)
@pytest.mark.asyncio
async def test_degenerate_inputs_complete_safely(
    seeded_repo: ConversationRepository,
    builder: Callable[[ConversationRepository, str], ConversationFilter],
    payload: str,
) -> None:
    sentinels_before = await _sentinels_still_present(seeded_repo)
    results = await builder(seeded_repo, payload).list_summaries()
    assert isinstance(results, list)
    sentinels_after = await _sentinels_still_present(seeded_repo)
    assert sentinels_after == sentinels_before
