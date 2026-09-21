"""The portfolio's pathology distribution says what population it swept.

``compile_portfolio_bundle`` documents that a *subset* of session digests is the
expected input, so the corpus-wide pathology fields are routinely computed over
a fraction of the declared scope. Reporting ``clean`` there would publish "this
corpus contains no pathology" on the strength of whatever sample happened to
have a digest (polylogue-oimqc).
"""

from __future__ import annotations

from polylogue.analysis.portfolio import (
    PORTFOLIO_SCHEMA_VERSION,
    compile_portfolio_bundle,
    render_portfolio_markdown,
    render_portfolio_plain,
)
from polylogue.analysis.postmortem import PostmortemScope
from polylogue.analysis.transforms import SessionDigest, compile_session_digest
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.archive.session.models import SessionProfile
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId


def _profile(session_id: str) -> SessionProfile:
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
        total_cost_usd=0.0,
        total_duration_ms=0,
        tool_categories={},
        repo_paths=(),
        cwd_paths=(),
        branch_names=(),
        file_paths_touched=(),
        languages_detected=(),
        repo_names=(),
        first_message_at=None,
        last_message_at=None,
        wall_duration_ms=0,
        cost_is_estimated=False,
        cost_provenance="catalog_priced",
        total_input_tokens=0,
        total_output_tokens=0,
        total_cache_read_tokens=0,
        total_cache_write_tokens=0,
    )


def _clean_digest(session_id: str) -> SessionDigest:
    session = Session(
        id=SessionId(session_id),
        origin=Origin.CODEX_SESSION,
        title="clean run",
        working_directories=("/realm/project/polylogue",),
        messages=MessageCollection(
            messages=[
                Message(id=f"{session_id}-m1", role=Role.USER, text="do the work"),
                Message(id=f"{session_id}-m2", role=Role.ASSISTANT, text="done"),
            ]
        ),
    )
    return compile_session_digest(session)


def _scope(*, matched: int, analyzed: int) -> PostmortemScope:
    return PostmortemScope(
        since=None,
        until=None,
        query=None,
        matched_session_count=matched,
        analyzed_session_count=analyzed,
        truncated=matched > analyzed,
        dropped_session_count=max(matched - analyzed, 0),
    )


def test_portfolio_sweep_over_a_digest_sample_is_not_reported_as_clean() -> None:
    """A sampled sweep and a whole-corpus sweep must not read the same.

    Anti-vacuity: restore the ``if not projections`` guard — where any non-empty
    projection list falls through to ``_pathology_field(findings)`` and an empty
    findings list becomes ``clean`` — and the sampled half goes red while the
    whole-corpus half stays green. That asymmetry is the defect.
    """
    profiles = [_profile(f"codex-session:{n}") for n in "abcde"]

    whole = compile_portfolio_bundle(
        profiles,
        {p.session_id: _clean_digest(p.session_id) for p in profiles},
        scope=_scope(matched=5, analyzed=5),
        top_n=10,
    )
    sampled = compile_portfolio_bundle(
        profiles,
        {"codex-session:a": _clean_digest("codex-session:a")},
        scope=_scope(matched=5, analyzed=5),
        top_n=10,
    )

    assert whole.schema_version == PORTFOLIO_SCHEMA_VERSION
    assert whole.pathologies.status == "clean"
    assert whole.context_loss.status == "clean"
    assert whole.pathologies.covers_whole_scope is True

    assert sampled.pathologies.status == "partial"
    assert sampled.context_loss.status == "partial"
    assert sampled.pathologies.swept_session_count == 1
    assert sampled.pathologies.scope_session_count == 5
    assert sampled.pathologies.missing_digest_count == 4
    assert sampled.pathologies.covers_whole_scope is False

    statuses = {whole.pathologies.status, sampled.pathologies.status}
    assert len(statuses) == 2

    # Both renderers carry the status and its named gap, so the artifact a
    # human attaches to a portfolio cannot read as a swept corpus either.
    plain = render_portfolio_plain(sampled)
    assert "pathologies: partial" in plain
    assert "swept 1 of 5 sessions in scope" in plain
    markdown = render_portfolio_markdown(sampled)
    assert "**Overall:** partial" in markdown
    assert "4 without a session digest" in markdown


def test_portfolio_sweep_with_no_digests_stays_unavailable() -> None:
    """No projection anywhere is still ``unavailable``, not ``partial``."""
    profiles = [_profile("codex-session:a"), _profile("codex-session:b")]
    bundle = compile_portfolio_bundle(profiles, {}, scope=_scope(matched=2, analyzed=2), top_n=10)

    assert bundle.pathologies.status == "unavailable"
    assert bundle.context_loss.status == "unavailable"
    assert bundle.pathologies.swept_session_count == 0
    assert bundle.pathologies.scope_session_count == 2
    assert bundle.top_pathologies == ()


def test_portfolio_analysis_cap_is_a_distinct_uncounted_bucket() -> None:
    """Sessions the cap never reached are separated from missing digests.

    Anti-vacuity: compute coverage from ``profiles`` alone and this goes red —
    every analyzed profile here carries a digest, so a profile-relative reading
    would call 100% coverage over a 40%-covered corpus.
    """
    profiles = [_profile(f"codex-session:{n}") for n in "ab"]
    bundle = compile_portfolio_bundle(
        profiles,
        {p.session_id: _clean_digest(p.session_id) for p in profiles},
        scope=_scope(matched=5, analyzed=2),
        top_n=10,
    )

    field = bundle.pathologies
    assert field.status == "partial"
    assert field.swept_session_count == 2
    assert field.missing_digest_count == 0
    assert field.unanalyzed_session_count == 3
    assert "3 never analyzed" in field.detail
