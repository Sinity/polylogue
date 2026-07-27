"""Real-evidence ActorRef/ExecutionContextRef adapters (polylogue-h6r).

Anti-vacuity note: several tests below deliberately implement the *wrong*
adapter inline (the exact shortcuts h6r AC6 names -- actor=model-name folded
with config, actor=session, context=prompt-only) and assert the real adapter
in :mod:`polylogue.insights.actor_context` produces *different*,
distinguishable behavior from that naive implementation on the same fixture.
A test that only exercised the real adapter in isolation could not prove the
anti-pattern is rejected; showing the two implementations diverge is the
proof.
"""

from __future__ import annotations

import hashlib

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.refs import ActorRef, EvidenceRef, ObjectRef
from polylogue.insights.actor_context import (
    actor_ref_from_run,
    actor_ref_from_session,
    execution_context_ref_from_run,
    execution_context_ref_from_session,
)
from polylogue.insights.run_projection import ProjectedRun
from polylogue.sources.parsers.base_models import ParsedMessage, ParsedSession


def _session(
    *,
    provider_session_id: str,
    source_name: Provider = Provider.CODEX,
    models_used: list[str] | None = None,
    instructions_text: str | None = None,
) -> ParsedSession:
    return ParsedSession(
        source_name=source_name,
        provider_session_id=provider_session_id,
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hi")],
        models_used=models_used or [],
        instructions_text=instructions_text,
    )


def _run(
    *,
    object_id: str,
    harness: str = "codex",
    provider_origin: str = "codex-session",
    agent_ref: ObjectRef | None = None,
) -> ProjectedRun:
    return ProjectedRun(
        run_ref=ObjectRef(kind="run", object_id=object_id),
        harness=harness,  # type: ignore[arg-type]
        provider_origin=provider_origin,
        agent_ref=agent_ref,
        evidence_refs=(EvidenceRef(session_id=object_id),),
    )


# --- AC1: same actor, two prompts/configs -> one ActorRef, two ExecutionContextRefs ---


def test_same_actor_under_two_instruction_sets_yields_one_actor_two_contexts() -> None:
    session_a = _session(
        provider_session_id="s-a",
        models_used=["claude-sonnet-5"],
        instructions_text="You are a careful reviewer.",
    )
    session_b = _session(
        provider_session_id="s-b",
        models_used=["claude-sonnet-5"],
        instructions_text="You are a terse implementer.",
    )

    actor_a = actor_ref_from_session(session_a)
    actor_b = actor_ref_from_session(session_b)
    context_a = execution_context_ref_from_session(session_a)
    context_b = execution_context_ref_from_session(session_b)

    assert actor_a == actor_b
    assert actor_a.format() == "agent:claude-sonnet-5"
    assert context_a.context_id != context_b.context_id
    assert "instructions_hash" in context_a.known_fields
    assert "instructions_hash" in context_b.known_fields


def test_provider_neutral_across_codex_and_claude_code_origins() -> None:
    codex_session = _session(provider_session_id="s-codex", source_name=Provider.CODEX, models_used=["gpt-5.6-terra"])
    claude_session = _session(
        provider_session_id="s-claude", source_name=Provider.CLAUDE_CODE, models_used=["gpt-5.6-terra"]
    )

    # Same model family reported by two different provider harnesses: actor
    # identity is model-keyed (provider-neutral), but the execution context
    # differs because harness is real, distinguishing evidence.
    assert actor_ref_from_session(codex_session) == actor_ref_from_session(claude_session)
    assert (
        execution_context_ref_from_session(codex_session).context_id
        != execution_context_ref_from_session(claude_session).context_id
    )


# --- AC2: two distinct actors in one runtime remain distinct; session != actor/context ---


def test_distinct_models_in_one_runtime_remain_distinct_actors() -> None:
    session_a = _session(provider_session_id="s-a", models_used=["model-a"], instructions_text="shared prompt")
    session_b = _session(provider_session_id="s-b", models_used=["model-b"], instructions_text="shared prompt")

    assert actor_ref_from_session(session_a) != actor_ref_from_session(session_b)
    # Same harness + same instructions -> execution context is genuinely
    # identical for both, even though the actors differ. This is expected:
    # context content-addresses the environment, not who is operating in it.
    assert (
        execution_context_ref_from_session(session_a).context_id
        == execution_context_ref_from_session(session_b).context_id
    )


def test_session_identity_never_leaks_into_actor_or_context_identity() -> None:
    session_a = _session(provider_session_id="attempt-1", models_used=["model-x"], instructions_text="p")
    session_b = _session(provider_session_id="attempt-2", models_used=["model-x"], instructions_text="p")

    assert session_a.provider_session_id != session_b.provider_session_id
    assert actor_ref_from_session(session_a) == actor_ref_from_session(session_b)
    assert (
        execution_context_ref_from_session(session_a).context_id
        == execution_context_ref_from_session(session_b).context_id
    )
    assert "attempt-1" not in actor_ref_from_session(session_a).format()
    assert "attempt-1" not in execution_context_ref_from_session(session_a).context_id


# --- AC3: partial/absent evidence -> explicit unknown, stable identity, no fabrication ---


def test_session_with_no_model_or_instructions_yields_stable_unknown_actor_and_marked_context() -> None:
    session = _session(provider_session_id="s-bare")

    actor = actor_ref_from_session(session)
    context = execution_context_ref_from_session(session)

    assert actor == ActorRef(kind="agent", identity="unknown")
    assert not context.is_complete
    assert context.unknown_fraction > 0.5
    assert "instructions_hash" in context.unknown_fields
    assert "tools_profile" in context.unknown_fields
    assert "permissions" in context.unknown_fields
    # Stable and re-derivable: the same bare evidence always yields the same context id.
    assert context.context_id == execution_context_ref_from_session(_session(provider_session_id="s-bare-2")).context_id


def test_run_projection_actor_context_marks_uncaptured_fields_unknown() -> None:
    run = _run(object_id="codex-session:demo")

    context = execution_context_ref_from_run(run)
    assert set(context.unknown_fields) == {
        "tools_profile",
        "mcp_profile",
        "permissions",
        "runtime_build",
        "sampling_params",
    }
    assert context.known_fields == ("harness", "provider_origin")


def test_run_with_explicit_agent_ref_uses_it_as_real_actor_identity() -> None:
    explore_ref = ObjectRef(kind="agent", object_id="codex/Explore")
    run = _run(object_id="codex-session:demo:subagent:0", agent_ref=explore_ref)

    actor = actor_ref_from_run(run)
    assert actor.identity == "codex/Explore"


def test_run_without_agent_ref_yields_explicit_unknown_actor_not_harness() -> None:
    run = _run(object_id="codex-session:demo", agent_ref=None)

    actor = actor_ref_from_run(run)
    # Harness ("codex") must never leak into actor identity as a fallback --
    # that would be exactly the context-collapsed-into-actor shortcut.
    assert actor == ActorRef(kind="agent", identity="unknown")
    assert actor.identity != run.harness


# --- AC6: mutation tests proving the naive wrong adapters are rejected ---


def test_actor_equals_model_plus_config_shortcut_is_distinguishable_and_rejected() -> None:
    """Naive: actor identity folds in instructions (actor=model+config).

    The correct adapter keeps the same actor across two instruction sets for
    one model; the naive shortcut below fractures identity on every prompt
    change, which is exactly what h6r AC1 forbids.
    """

    session_a = _session(provider_session_id="s-a", models_used=["model-x"], instructions_text="prompt A")
    session_b = _session(provider_session_id="s-b", models_used=["model-x"], instructions_text="prompt B")

    def naive_actor_equals_model_and_config(session: ParsedSession) -> ActorRef:
        model = session.models_used[0]
        config_hash = hashlib.sha256((session.instructions_text or "").encode()).hexdigest()[:8]
        return ActorRef(kind="agent", identity=f"{model}:{config_hash}")

    naive_a = naive_actor_equals_model_and_config(session_a)
    naive_b = naive_actor_equals_model_and_config(session_b)
    real_a = actor_ref_from_session(session_a)
    real_b = actor_ref_from_session(session_b)

    assert naive_a != naive_b  # the naive shortcut fractures actor identity on prompt change
    assert real_a == real_b  # the real adapter correctly keeps one actor across configs
    assert real_a != naive_a  # the two implementations are behaviorally distinguishable


def test_actor_equals_session_shortcut_is_distinguishable_and_rejected() -> None:
    """Naive: actor identity keyed on the session/attempt id (actor=session)."""

    session_a = _session(provider_session_id="attempt-1", models_used=["model-x"])
    session_b = _session(provider_session_id="attempt-2", models_used=["model-x"])

    def naive_actor_equals_session(session: ParsedSession) -> ActorRef:
        return ActorRef(kind="agent", identity=session.provider_session_id)

    naive_a = naive_actor_equals_session(session_a)
    naive_b = naive_actor_equals_session(session_b)
    real_a = actor_ref_from_session(session_a)
    real_b = actor_ref_from_session(session_b)

    assert naive_a != naive_b  # naive shortcut makes every session/attempt its own actor
    assert real_a == real_b  # the real adapter correctly recognizes the same actor
    assert real_a != naive_a


def test_context_equals_prompt_only_shortcut_is_distinguishable_and_rejected() -> None:
    """Naive: execution context hashes only the prompt text (context=prompt-only).

    Two sessions with identical instructions but different provider harnesses
    are genuinely different execution environments; a prompt-only context
    would incorrectly collapse them into one.
    """

    session_a = _session(provider_session_id="s-a", source_name=Provider.CODEX, instructions_text="same prompt")
    session_b = _session(provider_session_id="s-b", source_name=Provider.CLAUDE_CODE, instructions_text="same prompt")

    def naive_context_equals_prompt_only(session: ParsedSession) -> str:
        return hashlib.sha256((session.instructions_text or "").encode()).hexdigest()

    naive_a = naive_context_equals_prompt_only(session_a)
    naive_b = naive_context_equals_prompt_only(session_b)
    real_a = execution_context_ref_from_session(session_a)
    real_b = execution_context_ref_from_session(session_b)

    assert naive_a == naive_b  # naive shortcut incorrectly collapses two different harnesses
    assert real_a.context_id != real_b.context_id  # the real adapter correctly distinguishes them
