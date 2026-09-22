"""polylogue-ksgg AC10: a parser may not contradict its own topology declaration.

``fill_linear_parent_chain`` chains each message to the previous one BY LIST
POSITION. For an origin whose ``OriginSpec`` declares ``message_parent``
structurally-absent, calling it manufactures a provider edge nothing expressed.
Gemini CLI did exactly that: ``parse_gemini_cli`` gap-filled parents while
``_gemini_cli_spec`` declared the field absent, and the parser reads no parent
key from the wire at all.

The resolution is the code's, not the declaration's. That a turn sequence is
linear is a property of the READ model -- recoverable from ``position`` -- not
evidence that the provider asserted a reply-to edge. The campaign measured the
same substitution on Codex and Hermes: dropping position-derived parents left
session-level topology bit-identical (session_links 38 = 38) while all 98,892
message parent edges went to zero, proving every one was position-derived.

AC10 permits the remaining caller: ``parsers/drive.py`` runs the gap-fill for
``aistudio-drive``, whose spec declares ``message_parent`` "carried", which is
the explicit non-absent declaration AC10 names as the alternative to removal.

Anti-vacuity: restoring ``fill_linear_parent_chain(messages)`` in
``parse_gemini_cli`` makes both the behavioural and the structural test red.
``test_gap_fill_helper_still_chains`` pins the opposite direction so deleting
the helper's body -- rather than its wrong caller -- cannot pass.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Origin, Provider
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.origin_specs import (
    OriginSpec,
    TopologyCapabilities,
    _absent_topology,
    _executable_spec,
    origin_specs,
)
from polylogue.sources.parsers.base import ParsedMessage, fill_linear_parent_chain

_HELPER = "fill_linear_parent_chain"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _parser_modules_by_parent_state() -> tuple[dict[str, set[str]], set[str]]:
    """Group declared parser modules by the message_parent states they serve.

    A module declared by several origins can serve both an absent and a
    non-absent declaration (``claude/ai_parser.py`` does). Such a module cannot
    be judged by a module-level rule, so it is returned as its own explicitly
    unmeasured term rather than folded into either answer.
    """
    states_by_module: dict[str, set[str]] = defaultdict(set)
    for spec in origin_specs():
        state = spec.topology_capabilities.message_parent.state
        for parser_path in spec.parser_paths:
            states_by_module[parser_path].add(state)
    absent_only = {path for path, states in states_by_module.items() if states == {"structurally-absent"}}
    mixed = {path for path, states in states_by_module.items() if len(states) > 1}
    return {path: states_by_module[path] for path in absent_only}, mixed


def test_absent_parent_origins_never_gap_fill() -> None:
    """No parser serving only absent-parent origins may synthesise parents."""
    absent_only, mixed = _parser_modules_by_parent_state()
    assert absent_only, "no origin declares message_parent structurally-absent; the rule would be vacuous"

    offenders = []
    for parser_path in sorted(absent_only):
        source = (_REPO_ROOT / parser_path).read_text()
        if _HELPER in source:
            offenders.append(parser_path)
    assert offenders == []

    # The exclusion is visible rather than silent: a module that starts serving
    # both an absent and a non-absent origin leaves this rule's scope, and this
    # assertion is where that becomes a decision instead of a quiet pass.
    assert mixed == {"polylogue/sources/parsers/claude/ai_parser.py"}


def test_gemini_cli_emits_no_synthesised_parents() -> None:
    """The Gemini CLI parser agrees with its own declaration, on a real parse."""
    spec = next(spec for spec in origin_specs() if spec.origin is Origin.GEMINI_CLI_SESSION)
    assert spec.topology_capabilities.message_parent.state == "structurally-absent"

    payload = {
        "sessionId": "gemini-topology-1",
        "projectHash": "project-hash",
        "startTime": "2026-04-08T20:45:00.000Z",
        "lastUpdated": "2026-04-08T20:47:00.000Z",
        "kind": "chat",
        "summary": "Topology declaration",
        "messages": [
            {"id": "u1", "timestamp": "2026-04-08T20:45:01.000Z", "type": "user", "content": ["first"]},
            {"id": "a1", "timestamp": "2026-04-08T20:45:02.000Z", "type": "gemini", "content": "second"},
            {"id": "u2", "timestamp": "2026-04-08T20:45:03.000Z", "type": "user", "content": ["third"]},
        ],
    }

    [session] = parse_payload("gemini-cli", payload, "fallback")

    assert session.source_name is Provider.GEMINI_CLI
    assert [message.provider_message_id for message in session.messages] == ["u1", "a1", "u2"]
    # Ordering survives -- it is what a linear read model is built from ...
    assert [message.position for message in session.messages] == [0, 1, 2]
    assert session.active_leaf_message_provider_id == "u2"
    # ... while no message claims a parent the provider never expressed.
    assert [message.parent_message_provider_id for message in session.messages] == [None, None, None]
    assert [message.parent_message_position for message in session.messages] == [None, None, None]


def test_gap_fill_helper_still_chains() -> None:
    """The helper is intact: removal was a routing decision, not a gutting.

    ``parsers/drive.py`` still calls it for ``aistudio-drive``, which declares
    ``message_parent`` carried, so the helper must keep chaining.
    """
    messages = [ParsedMessage(provider_message_id=f"m{index}", role=Role.USER, text=str(index)) for index in range(3)]

    filled = fill_linear_parent_chain(messages)

    assert [message.parent_message_provider_id for message in filled] == [None, "m0", "m1"]


def test_topology_declaration_has_no_silent_default() -> None:
    """ksgg AC11: an Origin cannot inherit structurally-absent by omission.

    The recorded residual said ``_no_topology_capabilities`` "still defaults
    EVERY dimension to structurally-absent for any origin that does not
    override", leaving a produced-but-undeclared capability silently
    declarable as absent. At this head that is not reachable by omission:
    ``OriginSpec.topology_capabilities`` and ``_executable_spec``'s
    keyword-only parameter both carry no default, and ``TopologyCapabilities``
    requires all five dimensions. Calling ``_no_topology_capabilities`` is an
    explicit declaration, not a fallback.

    Nothing pinned that, so a later ``= _no_topology_capabilities(origin)``
    default would silently reopen the hole. This is the pin. Anti-vacuity:
    give either parameter a default and the corresponding assertion goes red;
    the construction attempts below stay red-on-omission independently.
    """
    for owner, parameter in (
        (OriginSpec, "topology_capabilities"),
        (_executable_spec, "topology_capabilities"),
    ):
        signature = inspect.signature(owner)
        assert signature.parameters[parameter].default is inspect.Parameter.empty, owner

    # Not only the signature: omitting it actually refuses.
    with pytest.raises(TypeError):
        _executable_spec(  # type: ignore[call-arg]
            Origin.UNKNOWN_EXPORT,
            provider=Provider.UNKNOWN,
            tightness=1,
            discovery="probe",
            acquisition_modes=("probe",),
            parser_paths=(),
            fixture_paths=(),
            display_description="probe",
        )
    with pytest.raises(TypeError):
        TopologyCapabilities(  # type: ignore[call-arg]
            message_parent=_absent_topology("probe"),
            message_branch_state=_absent_topology("probe"),
            session_parent_target=_absent_topology("probe"),
            inheritance_branch_point=_absent_topology("probe"),
        )
