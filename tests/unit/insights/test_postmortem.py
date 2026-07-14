"""Pure-aggregator tests for the postmortem bundle (#2380).

These exercise ``compile_postmortem_bundle`` with constructed model inputs (no
DB). They assert every headline field is present with a drillable EvidenceRef
when a value exists, and that the never-fabricated fields (failure_mode,
wasted_loop, longest_tool_gap, and the no-signal top_expensive_session) degrade
honestly instead of inventing a number.
"""

from __future__ import annotations

from datetime import UTC, datetime

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.archive.session.models import SessionProfile
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.insights.postmortem import (
    POSTMORTEM_SCHEMA_VERSION,
    PostmortemScope,
    compile_postmortem_bundle,
    render_postmortem_markdown,
    render_postmortem_plain,
)
from polylogue.insights.transforms import SessionDigest, compile_session_digest


def _profile(
    session_id: str,
    *,
    origin: str = "codex-session",
    title: str | None = None,
    total_cost_usd: float = 0.0,
    cost_is_estimated: bool = False,
    wall_duration_ms: int = 0,
    first_message_at: datetime | None = None,
    last_message_at: datetime | None = None,
    repo_names: tuple[str, ...] = (),
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    total_cache_read_tokens: int = 0,
    total_cache_write_tokens: int = 0,
) -> SessionProfile:
    return SessionProfile(
        session_id=session_id,
        origin=origin,
        title=title,
        created_at=None,
        updated_at=None,
        message_count=1,
        substantive_count=1,
        tool_use_count=0,
        thinking_count=0,
        attachment_count=0,
        word_count=0,
        total_cost_usd=total_cost_usd,
        total_duration_ms=0,
        tool_categories={},
        repo_paths=(),
        cwd_paths=(),
        branch_names=(),
        file_paths_touched=(),
        languages_detected=(),
        repo_names=repo_names,
        work_events=(),
        phases=(),
        first_message_at=first_message_at,
        last_message_at=last_message_at,
        wall_duration_ms=wall_duration_ms,
        cost_is_estimated=cost_is_estimated,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cache_read_tokens=total_cache_read_tokens,
        total_cache_write_tokens=total_cache_write_tokens,
    )


def _digest_with_subagent(session_id: str) -> SessionDigest:
    """Build a real session digest whose run projection includes a subagent."""
    session = Session(
        id=SessionId(session_id),
        origin=Origin.CODEX_SESSION,
        title="parent",
        working_directories=("/realm/project/polylogue",),
        messages=MessageCollection(
            messages=[
                Message(id="m1", role=Role.USER, text="do the work"),
                Message(
                    id="m2",
                    role=Role.ASSISTANT,
                    text="dispatching",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "tool-task",
                            "name": "Task",
                            "tool_input": {
                                "subagent_type": "Explore",
                                "taskId": "task-1",
                                "child_session_id": "codex-session:child-1",
                                "prompt": "map the surface",
                            },
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-task",
                            "text": "Subagent done.",
                        },
                    ],
                ),
            ]
        ),
    )
    return compile_session_digest(session)


def _scope(matched: int, analyzed: int) -> PostmortemScope:
    return PostmortemScope(
        since="2024-01-01",
        until="2024-12-31",
        query=None,
        matched_session_count=matched,
        analyzed_session_count=analyzed,
        truncated=False,
        dropped_session_count=0,
    )


def test_compile_postmortem_bundle_populates_every_headline_field() -> None:
    early = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
    mid = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)
    late = datetime(2024, 3, 2, 9, 0, tzinfo=UTC)

    profile_a = _profile(
        "codex-session:a",
        title="cheap",
        total_cost_usd=1.50,
        cost_is_estimated=True,
        wall_duration_ms=60_000,
        first_message_at=early,
        last_message_at=mid,
        repo_names=("polylogue",),
        total_input_tokens=100,
        total_output_tokens=50,
        total_cache_read_tokens=10,
        total_cache_write_tokens=5,
    )
    profile_b = _profile(
        "codex-session:b",
        title="expensive",
        total_cost_usd=3.25,
        wall_duration_ms=120_000,
        first_message_at=mid,
        last_message_at=late,
        repo_names=("polylogue", "sinex"),
        total_input_tokens=200,
        total_output_tokens=80,
        total_cache_read_tokens=20,
        total_cache_write_tokens=8,
    )
    profile_c = _profile("codex-session:c", total_cost_usd=0.0)

    digests = {"codex-session:b": _digest_with_subagent("codex-session:b")}
    bundle = compile_postmortem_bundle(
        [profile_a, profile_b, profile_c],
        digests,
        scope=_scope(matched=3, analyzed=3),
    )

    assert bundle.schema_version == POSTMORTEM_SCHEMA_VERSION
    assert bundle.scope.matched_session_count == 3
    assert bundle.scope.analyzed_session_count == 3

    # session_count
    assert bundle.session_count.count == 3
    assert bundle.session_count.evidence_refs

    # wallclock_span: earliest first_message → latest last_message, summed wall
    assert bundle.wallclock_span.earliest_first_message_at == early.isoformat()
    assert bundle.wallclock_span.latest_last_message_at == late.isoformat()
    expected_span_ms = int((late - early).total_seconds() * 1000)
    assert bundle.wallclock_span.span_ms == expected_span_ms
    assert bundle.wallclock_span.summed_wall_duration_ms == 180_000
    assert bundle.wallclock_span.evidence_refs

    # estimated_cost + labelled token lanes (no undifferentiated "tokens")
    assert bundle.estimated_cost.total_cost_usd == 4.75
    assert bundle.estimated_cost.cost_is_estimated is True
    assert bundle.estimated_cost.tokens.input_tokens == 300
    assert bundle.estimated_cost.tokens.output_tokens == 130
    assert bundle.estimated_cost.tokens.cache_read_tokens == 30
    assert bundle.estimated_cost.tokens.cache_write_tokens == 13
    assert bundle.estimated_cost.evidence_refs

    # repos_touched
    repo_counts = {r.repo: r.session_count for r in bundle.repos_touched}
    assert repo_counts == {"polylogue": 2, "sinex": 1}
    for repo_metric in bundle.repos_touched:
        assert repo_metric.evidence_refs

    # top_expensive_session points at b
    assert bundle.top_expensive_session.status == "ok"
    assert bundle.top_expensive_session.session_id == "codex-session:b"
    assert bundle.top_expensive_session.total_cost_usd == 3.25
    assert bundle.top_expensive_session.evidence_refs
    assert bundle.top_expensive_session.evidence_refs[0].session_id == "codex-session:b"

    # subagent_branch_count derived from the run projection
    assert bundle.subagent_branch_count.count == 1
    assert bundle.subagent_branch_count.evidence_refs
    assert bundle.subagent_branch_count.evidence_refs[0].session_id == "codex-session:b"

    # longest_tool_gap stays degraded (no per-tool timing in v0)
    assert bundle.longest_tool_gap.value is None
    assert bundle.longest_tool_gap.status == "unavailable"
    assert bundle.longest_tool_gap.reason
    # pathology fields: detectors ran over the digest run projection (a subagent
    # with no failed tests / unaddressed reviews / lossy context) → clean
    assert bundle.wasted_loop.status == "clean"
    assert bundle.failure_mode.status == "clean"

    # AC1: every headline metric that carries a value has >=1 drillable ref.
    for metric in (
        bundle.session_count,
        bundle.wallclock_span,
        bundle.estimated_cost,
        bundle.top_expensive_session,
        bundle.subagent_branch_count,
    ):
        assert metric.evidence_refs, f"{type(metric).__name__} missing evidence"

    # Renderers consume the same payload object.
    plain = render_postmortem_plain(bundle)
    assert "top_expensive_session: codex-session:b" in plain
    markdown = render_postmortem_markdown(bundle)
    assert "# Postmortem Bundle" in markdown
    assert "subagent_branch_count" in markdown


def test_compile_postmortem_bundle_degrades_without_signal() -> None:
    """With no cost, no subagents, and no digests, nothing is fabricated."""
    profile = _profile("codex-session:only", total_cost_usd=0.0)
    bundle = compile_postmortem_bundle(
        [profile],
        {},
        scope=_scope(matched=1, analyzed=1),
    )

    # top_expensive_session degrades to no_signal rather than naming a $0 session
    assert bundle.top_expensive_session.status == "no_signal"
    assert bundle.top_expensive_session.session_id is None
    assert bundle.top_expensive_session.total_cost_usd == 0.0
    assert bundle.top_expensive_session.reason

    # no subagent evidence invented
    assert bundle.subagent_branch_count.count == 0

    # longest_tool_gap degrades; pathology fields are unavailable with no digests
    assert bundle.longest_tool_gap.value is None
    assert bundle.longest_tool_gap.status == "unavailable"
    assert bundle.wasted_loop.status == "unavailable"
    assert bundle.failure_mode.status == "unavailable"

    # no repos, no wall span
    assert bundle.repos_touched == ()
    assert bundle.wallclock_span.span_ms is None
    assert bundle.wallclock_span.summed_wall_duration_ms == 0


def test_postmortem_bundle_json_envelope_has_stable_headline_keys() -> None:
    """The JSON contract carries every named headline key (#2380 AC2)."""
    bundle = compile_postmortem_bundle(
        [_profile("codex-session:a", total_cost_usd=1.0)],
        {},
        scope=_scope(matched=1, analyzed=1),
    )
    payload = bundle.model_dump(mode="json")
    expected_keys = {
        "schema_version",
        "scope",
        "session_count",
        "wallclock_span",
        "estimated_cost",
        "repos_touched",
        "top_expensive_session",
        "subagent_branch_count",
        "longest_tool_gap",
        "wasted_loop",
        "failure_mode",
    }
    assert expected_keys <= set(payload)
    # token lanes are labelled separately, never a single "tokens" integer
    token_keys = set(payload["estimated_cost"]["tokens"])
    assert token_keys == {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    }
    # The CLI ships JSON through model_json_document(..., exclude_none=True), not
    # raw model_dump — assert the contract on that exact path. Degraded fields
    # keep their explicit status + reason and never carry a fabricated value.
    from polylogue.surfaces.payloads import model_json_document

    shipped = model_json_document(bundle, exclude_none=True)
    assert expected_keys <= set(shipped)
    # longest_tool_gap is the remaining DegradedField (status + reason, no value)
    gap = shipped["longest_tool_gap"]
    assert isinstance(gap, dict)
    assert gap["status"] in {"no_signal", "unavailable"}
    assert gap["reason"]
    assert gap.get("value") is None
    # pathology fields carry an explicit detected/clean/unavailable status
    for field in ("wasted_loop", "failure_mode"):
        pathology = shipped[field]
        assert isinstance(pathology, dict)
        assert pathology["status"] in {"detected", "clean", "unavailable"}
