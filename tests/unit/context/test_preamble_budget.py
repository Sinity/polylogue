"""The context preamble under a *finite* token budget (polylogue-uy49i).

``daemon/http.py`` passes ``max_tokens`` straight through to
``build_context_preamble_payload``, so a finite budget is a live path. Before
this fix the whole preamble crossed the scheduler as one item: the budget
decision was binary, and the exclude branch dropped
``recent_related_sessions`` and ``guidance`` wholesale -- the continuation
material -- while keeping the framing, so the result still looked well-formed.

Every test here therefore builds at a budget derived from the *measured* cost
of the segments. A test at an unbounded budget passes against the shipped code
and proves nothing; the unbounded case appears below only as the control the
constrained cases are compared against.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import polylogue.context.preamble as preamble_module
from polylogue.context.preamble import build_context_preamble_payload
from polylogue.context.scheduler import ContextItem, schedule_context
from polylogue.surfaces.payloads import ContextPreamble

_LONG_SUMMARY = (
    "A long prior-session summary that exists to make the resume-candidate "
    "section the most expensive part of the preamble, so a budget can be "
    "chosen that forces the scheduler to shed or reduce something real. "
) * 4


def _candidate(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        logical_session_id=f"codex-session:prior-{index}",
        session_id=f"codex-session:prior-{index}",
        title=f"prior work {index}",
        date="2026-09-01",
        terminal_state="abandoned",
        objective_posture="resumable",
        summary=f"{index}: {_LONG_SUMMARY}",
        origin="codex-session",
        overlap_basis=None,
    )


def _claim(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        kind=SimpleNamespace(value="decision"),
        body_text=f"Operator decision {index}: keep context as refs, not raw logs.",
        author_kind="agent",
        author_ref="agent:codex",
        status="active",
        context_policy={"inject": True},
        target_ref="session:seed",
        scope_ref="repo:polylogue",
        evidence_refs=["session:seed"],
    )


def _poly(*, candidates: int = 4, claims: int = 3) -> MagicMock:
    lineage = SimpleNamespace(
        root_id="codex-session:root",
        nodes=(),
        edges=(),
    )
    poly = MagicMock()
    poly.config = SimpleNamespace(archive_root=None)
    poly.get_session = AsyncMock(
        return_value=SimpleNamespace(
            git_repository_url="https://example.invalid/repo",
            git_branch="main",
            origin="codex-session",
            model="gpt-test",
            permission_mode="default",
        )
    )
    poly.compact_lineage = AsyncMock(return_value=lineage)
    poly.find_resume_candidates = AsyncMock(return_value=[_candidate(i) for i in range(candidates)])
    poly.list_assertion_claim_payloads = AsyncMock(return_value=[_claim(i) for i in range(claims)])
    return poly


async def _segment_costs(monkeypatch: pytest.MonkeyPatch, poly: MagicMock) -> dict[str, int]:
    """Measure what each segment actually costs at this build."""

    seen: dict[str, int] = {}

    def capture(sources: Any, **kwargs: Any) -> Any:
        for source in sources:
            for item in source.candidates(moment=kwargs["moment"], target_session=kwargs["target_session"]):
                seen[item.ref.rsplit(":", 1)[-1]] = item.token_cost
        return schedule_context(sources, **kwargs)

    monkeypatch.setattr(preamble_module, "schedule_context", capture)
    await _build(poly, token_budget=None)
    monkeypatch.setattr(preamble_module, "schedule_context", schedule_context)
    return seen


async def _build(poly: MagicMock, *, token_budget: int | None) -> ContextPreamble:
    preamble = await build_context_preamble_payload(
        poly,
        session_id="seed",
        cwd=None,
        source_tool_calls={"build_context_preamble_payload": "polylogue-test"},
        token_budget=token_budget,
    )
    assert preamble is not None
    return preamble


def _shed(preamble: ContextPreamble) -> dict[str, str]:
    prefix = preamble_module._BUDGET_FAILURE_PREFIX
    return {
        key.removeprefix(prefix): value for key, value in preamble.component_failures.items() if key.startswith(prefix)
    }


@pytest.mark.asyncio
async def test_unbounded_budget_admits_every_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control case: an unbounded build is complete and says so.

    This one passes against the shipped code too. It is here so the
    constrained cases below have something to differ from.
    """
    preamble = await _build(_poly(), token_budget=None)

    assert len(preamble.recent_related_sessions) == 4
    assert preamble.guidance is not None and not isinstance(preamble.guidance, str)
    assert len(preamble.guidance.assertions) == 3
    assert preamble.session_lineage is not None
    assert preamble.project_state is not None
    assert _shed(preamble) == {}


@pytest.mark.asyncio
async def test_a_finite_budget_sheds_the_replaceable_sections_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The continuation material is the LAST thing to go, not the first.

    Anti-vacuity (executed): this goes red both when ``ordinal_score`` is
    zeroed -- candidates are emitted in name order, so ``project_state`` then
    outranks ``recent_related_sessions`` -- and when the ranks are inverted.
    It also goes red against the shipped single-item preamble, whose exclusion
    branch emptied ``recent_related_sessions`` and ``guidance`` while leaving
    ``project_state`` populated: exactly the two assertions below.
    """
    poly = _poly()
    costs = await _segment_costs(monkeypatch, poly)
    # Enough for guidance and the related sessions, not for the rest.
    budget = costs["guidance"] + costs["recent_related_sessions"]

    preamble = await _build(poly, token_budget=budget)

    # Kept: the two sections a continuation cannot be reconstructed without.
    assert len(preamble.recent_related_sessions) == 4
    assert preamble.guidance is not None and not isinstance(preamble.guidance, str)
    assert len(preamble.guidance.assertions) == 3
    # Shed: the sections the agent can re-derive itself.
    assert preamble.project_state is None
    assert preamble.source_tool_calls == {}
    shed = _shed(preamble)
    assert set(shed) == {"session_lineage", "project_state", "source_tool_calls"}
    for reason in shed.values():
        assert "context token budget" in reason
    # The recorded cost is the section's own, not whatever last crossed the
    # admission boundary, so the reason is usable for choosing a real budget.
    assert f"the section costs {costs['project_state']} tokens" in shed["project_state"]


@pytest.mark.asyncio
async def test_a_degrade_callback_keeps_a_smaller_true_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """A section too big to admit whole is reduced, not dropped.

    Anti-vacuity (executed): removing ``degrade=`` from the preamble's
    ``ContextItem`` construction leaves the scheduler only include/exclude --
    ``recent_related_sessions`` comes back empty and the recorded reason
    changes from "reduced" to "omitted". That is precisely the shipped state,
    and it is the only test here that mutation reddens.
    """
    poly = _poly()
    costs = await _segment_costs(monkeypatch, poly)
    # Guidance whole, plus less than the related section costs but more than
    # its reduced form: the degrade callback is the only way to keep anything.
    budget = costs["guidance"] + costs["recent_related_sessions"] // 2

    preamble = await _build(poly, token_budget=budget)

    assert preamble.guidance is not None and not isinstance(preamble.guidance, str)
    assert len(preamble.guidance.assertions) == 3
    assert len(preamble.recent_related_sessions) == 1
    kept = preamble.recent_related_sessions[0]
    # The reduced form is a true, smaller version: real identity, no bulk.
    assert kept.session_id == "codex-session:prior-0"
    assert kept.title == "prior work 0"
    assert kept.summary is None
    shed = _shed(preamble)
    assert "reduced by the context token budget" in shed["recent_related_sessions"]
    assert "kept the top 1 of 4 candidates" in shed["recent_related_sessions"]


@pytest.mark.asyncio
async def test_the_smallest_budget_still_protects_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """When only one section fits, it is the operator guidance.

    Anti-vacuity (executed): inverting ``ordinal_score`` turns this red --
    ``source_tool_calls`` and ``project_state`` win the budget and the guidance
    is shed. Zeroing ``ordinal_score`` does not, because ``guidance`` also sorts
    first by name; the sibling test below is the one that catches that.
    """
    poly = _poly()
    costs = await _segment_costs(monkeypatch, poly)

    preamble = await _build(poly, token_budget=costs["guidance"])

    assert preamble.guidance is not None and not isinstance(preamble.guidance, str)
    assert len(preamble.guidance.assertions) == 3
    assert preamble.recent_related_sessions == []
    assert preamble.session_lineage is None
    assert preamble.project_state is None
    shed = _shed(preamble)
    assert "guidance" not in shed
    assert {"recent_related_sessions", "session_lineage", "project_state", "source_tool_calls"} == set(shed)


@pytest.mark.asyncio
async def test_a_starved_budget_names_every_omission_instead_of_looking_complete() -> None:
    """A preamble emptied by the budget is distinguishable from an empty one.

    AC2: the distinction is in the payload itself, not in comparing sizes. An
    archive with nothing to say yields no ``context_budget:`` entries at all;
    this build, which had five populated sections, names each one it could not
    carry.
    """
    poly = _poly()

    starved = await _build(poly, token_budget=0)
    assert starved.recent_related_sessions == []
    assert starved.guidance is None
    assert starved.project_state is None
    assert set(_shed(starved)) == {
        "guidance",
        "recent_related_sessions",
        "session_lineage",
        "project_state",
        "source_tool_calls",
    }

    empty = await _build(
        _poly(candidates=0, claims=0),
        token_budget=0,
    )
    # Nothing to carry is not the same fact as nothing could be carried.
    assert empty.recent_related_sessions == []
    assert empty.guidance is None
    assert "recent_related_sessions" not in _shed(empty)
    assert "guidance" not in _shed(empty)


@pytest.mark.asyncio
async def test_degraded_sections_keep_their_admission_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scheduler must still own authority when a degrader fires.

    The reduced candidate goes back through ``_validate_degraded_item``, so a
    reducer that changed source, ref, or trust would be rejected outright
    rather than admitted. Assert the contract holds on the real route.
    """
    poly = _poly()
    costs = await _segment_costs(monkeypatch, poly)

    captured: list[ContextItem] = []
    real = schedule_context

    def capture(sources: Any, **kwargs: Any) -> Any:
        assembly = real(sources, **kwargs)
        captured.extend(assembly.quoted_evidence)
        return assembly

    monkeypatch.setattr(preamble_module, "schedule_context", capture)
    await _build(poly, token_budget=costs["guidance"] + costs["recent_related_sessions"] // 2)

    assert captured
    for item in captured:
        assert item.source == "context-session_start"
        assert item.trust_class == "quoted"
        assert item.material_class == "evidence"
        assert item.ref.startswith("context-preamble:session_start:seed:")
