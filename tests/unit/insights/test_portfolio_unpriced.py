"""The portfolio bundle says how much of its total it could not price.

Sessions with no usage evidence contribute 0.0 to ``total_cost_usd`` because
absent evidence is not a bill. Without a count of them, "$12.40 across 200
sessions" reads identically whether none or most were unpriced.
"""

from __future__ import annotations

from polylogue.analysis.portfolio import (
    compile_portfolio_bundle,
    render_portfolio_markdown,
    render_portfolio_plain,
)
from polylogue.analysis.postmortem import PostmortemScope
from polylogue.archive.session.models import SessionProfile


def _profile(session_id: str, *, cost: float, provenance: str) -> SessionProfile:
    return SessionProfile(
        session_id=session_id,
        origin="codex-session",
        title=None,
        created_at=None,
        updated_at=None,
        message_count=1,
        substantive_count=1,
        tool_use_count=0,
        thinking_count=0,
        attachment_count=0,
        word_count=0,
        total_cost_usd=cost,
        total_duration_ms=0,
        tool_categories={},
        repo_paths=(),
        cwd_paths=(),
        branch_names=(),
        file_paths_touched=(),
        languages_detected=(),
        repo_names=(),
        work_events=(),
        phases=(),
        first_message_at=None,
        last_message_at=None,
        wall_duration_ms=0,
        cost_is_estimated=provenance != "provider_reported",
        cost_provenance=provenance,
        total_input_tokens=100 if provenance != "unknown" else 0,
        total_output_tokens=50 if provenance != "unknown" else 0,
        total_cache_read_tokens=0,
        total_cache_write_tokens=0,
    )


def test_portfolio_counts_and_renders_unpriced_sessions() -> None:
    """Two priced sessions and one without usage evidence report unpriced=1.

    Anti-vacuity: dropping ``unpriced_session_count`` from the aggregation
    makes the field read 0 while the $0.00 session still lands in the total,
    and both renderer assertions below lose the "1 unpriced" clause -- the
    exact shape where absent evidence reads as a known zero.
    """
    profiles = [
        _profile("codex-session:a", cost=8.0, provenance="catalog_priced"),
        _profile("codex-session:b", cost=4.4, provenance="provider_reported"),
        _profile("codex-session:c", cost=0.0, provenance="unknown"),
    ]
    scope = PostmortemScope(
        since=None,
        until=None,
        query=None,
        matched_session_count=3,
        analyzed_session_count=3,
        truncated=False,
        dropped_session_count=0,
    )

    bundle = compile_portfolio_bundle(profiles, {}, scope=scope, top_n=10)

    assert bundle.estimated_cost.unpriced_session_count == 1
    # The unpriced session contributes nothing to the total, which is exactly
    # why the count has to be reported alongside it.
    assert bundle.estimated_cost.total_cost_usd == 12.4

    plain = render_portfolio_plain(bundle)
    markdown = render_portfolio_markdown(bundle)
    assert "across 3 sessions, 1 unpriced" in plain
    assert "across 3 sessions, 1 unpriced" in markdown


def test_portfolio_omits_the_unpriced_clause_when_everything_is_priced() -> None:
    """A fully priced bundle reports zero and says nothing about unpriced work.

    Anti-vacuity: a renderer that always emitted the clause would print
    "0 unpriced" here, which is noise on every healthy report.
    """
    profiles = [
        _profile("codex-session:a", cost=8.0, provenance="catalog_priced"),
        _profile("codex-session:b", cost=4.4, provenance="provider_reported"),
    ]
    scope = PostmortemScope(
        since=None,
        until=None,
        query=None,
        matched_session_count=2,
        analyzed_session_count=2,
        truncated=False,
        dropped_session_count=0,
    )

    bundle = compile_portfolio_bundle(profiles, {}, scope=scope, top_n=10)

    assert bundle.estimated_cost.unpriced_session_count == 0
    assert "unpriced" not in render_portfolio_plain(bundle)
    assert "unpriced" not in render_portfolio_markdown(bundle)
