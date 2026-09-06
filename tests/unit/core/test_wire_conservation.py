"""Content conservation between a wire record and its parsed session.

Anti-vacuity: ``test_truncated_rendering_is_not_conserved`` is red only
because the check compares by equality over multiplicities. Reshape
``check_conservation`` to ask whether the planted value merely *contains* or
is *contained by* some parsed block text and that test goes green while the
preferred-truncation defect ships. Reshape the multiset comparison to "at
least once" and ``test_value_emitted_twice_is_duplication`` goes green the
same way. Let ``excluded_paths_from_pins`` accept a reject pin with no written
reason and ``test_unexplained_reject_pin_excludes_nothing`` goes green while
undocumented drops read as conserved. Drop the ``conservation=`` argument at
the receipt's witness construction and
``test_receipt_measures_conservation_for_every_provider`` goes red, which is
the only test here that fails when the production route stops measuring.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.schemas.pinning import PinDecision, PinSet
from polylogue.schemas.synthetic.conservation import (
    check_conservation,
    collect_planted_values,
    excluded_paths_from_pins,
    normalise_path,
)
from polylogue.schemas.synthetic.wire_formats import (
    CONSERVATION_BLOCKING_PROVIDERS,
    WireParserWitness,
)
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage, ParsedSession
from tests.infra.wire_support import shared_wire_support_receipt

BODY_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "x-polylogue-semantic-role": "message_body"},
                    "at": {"type": "string", "x-polylogue-semantic-role": "message_timestamp"},
                    "note": {"type": "string"},
                },
            },
        }
    },
}


def _session(*texts: str, title: str | None = None) -> ParsedSession:
    """One parsed session whose single message carries ``texts`` as blocks.

    The production type, not a stub: the check reads the same fields the
    writer lowers, so a rename there must reach this test.
    """
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="conservation-fixture",
        title=title,
        messages=[
            ParsedMessage(
                provider_message_id="conservation-fixture:0",
                role=Role.ASSISTANT,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text) for text in texts],
            )
        ],
    )


def _payload(*texts: str) -> dict[str, object]:
    return {"turns": [{"text": text, "at": "2026-01-01", "note": "ignored"} for text in texts]}


def test_body_reaching_one_block_is_conserved() -> None:
    result = check_conservation(BODY_SCHEMA, [_payload("alpha", "beta")], [_session("alpha", "beta")])

    assert result.planted_count == 2
    assert result.conserved
    assert result.findings == ()


def test_absent_body_is_loss() -> None:
    result = check_conservation(BODY_SCHEMA, [_payload("alpha", "beta")], [_session("alpha")])

    assert not result.conserved
    assert [(f.verdict, f.role) for f in result.findings] == [("loss", "message_body")]
    assert "$.turns[1].text" in result.findings[0].path


def test_body_parsed_into_empty_block_is_loss() -> None:
    """The compaction-summary shape: a message exists, its content does not."""
    result = check_conservation(BODY_SCHEMA, [_payload("alpha")], [_session("")])

    assert not result.conserved
    assert result.findings[0].verdict == "loss"


TITLE_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string", "x-polylogue-semantic-role": "session_title"}},
}


def test_title_is_conserved_against_titles_not_blocks() -> None:
    """A title reaching a block instead of the session title is still loss."""
    payload: dict[str, object] = {"name": "The Session"}

    conserved = check_conservation(TITLE_SCHEMA, [payload], [_session(title="The Session")])
    assert conserved.conserved

    misplaced = check_conservation(TITLE_SCHEMA, [payload], [_session("The Session")])
    assert not misplaced.conserved
    assert [(f.verdict, f.role) for f in misplaced.findings] == [("loss", "session_title")]


def test_value_emitted_twice_is_duplication() -> None:
    """The Drive shape: one wire value, two blocks."""
    result = check_conservation(BODY_SCHEMA, [_payload("alpha")], [_session("alpha", "alpha")])

    assert not result.conserved
    assert result.findings[0].verdict == "duplication"
    assert "expected 1" in result.findings[0].detail


def test_truncated_rendering_is_not_conserved() -> None:
    """The preferred-truncation shape, and this module's anti-vacuity case.

    A containment test would find "alpha" inside the planted value and call
    it conserved. Equality is what makes the truncation visible.
    """
    planted = "alpha beta gamma delta"
    result = check_conservation(BODY_SCHEMA, [_payload(planted)], [_session("alpha")])

    assert not result.conserved
    assert result.findings[0].verdict == "mutation"


def test_repeated_body_must_appear_once_per_plant() -> None:
    """A body planted twice is conserved only when emitted twice."""
    conserved = check_conservation(BODY_SCHEMA, [_payload("same", "same")], [_session("same", "same")])
    assert conserved.conserved

    halved = check_conservation(BODY_SCHEMA, [_payload("same", "same")], [_session("same")])
    assert not halved.conserved
    assert halved.findings[0].verdict == "loss"
    assert "expected 2" in halved.findings[0].detail


def test_non_content_roles_are_out_of_scope() -> None:
    """Scope is the content-bearing annotation, not the field list.

    ``at`` carries ``message_timestamp`` and ``note`` carries no role at all;
    neither reaching a block is not loss.
    """
    planted = collect_planted_values(BODY_SCHEMA, _payload("alpha"))

    assert [item.path for item in planted] == ["$.turns[0].text"]


def test_reject_pin_excludes_a_path_from_conservation() -> None:
    """A declared exclusion is honored; silence is still loss."""
    payload = _payload("alpha")

    undeclared = check_conservation(BODY_SCHEMA, [payload], [_session()])
    assert not undeclared.conserved

    declared = check_conservation(
        BODY_SCHEMA,
        [payload],
        [_session()],
        excluded_paths=frozenset({"$.turns"}),
    )
    assert declared.conserved
    assert declared.excluded_paths == ("$.turns[0].text",)


@pytest.mark.parametrize("enforced", [True, False])
def test_conservation_gates_health_only_where_enforced(enforced: bool) -> None:
    """Report-only providers are measured without deciding an exit code."""
    failing = check_conservation(BODY_SCHEMA, [_payload("alpha")], [_session()])
    witness = WireParserWitness(
        index=-1,
        exercised_keywords=("properties",),
        parsed_session_count=1,
        parsed_message_count=1,
        artifact_evidence=("evidence",),
        conservation=failing,
        conservation_enforced=enforced,
    )

    assert not failing.conserved
    assert witness.healthy is not enforced


def _pinned(monkeypatch: pytest.MonkeyPatch, *pins: PinDecision) -> None:
    """Answer ``load_pins`` with ``pins`` for whatever provider is asked."""
    monkeypatch.setattr(
        "polylogue.schemas.pinning.load_pins",
        lambda provider: PinSet(provider=str(provider), pins=list(pins)),
    )


def test_unexplained_reject_pin_excludes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The written reason is the exclusion, not the ``reject`` action."""
    explained = PinDecision(
        path="$.turns[].text",
        role="message_body",
        action="reject",
        reason="turn text is a render of the block below it",
    )
    _pinned(monkeypatch, replace(explained, reason=""))
    assert excluded_paths_from_pins(Provider.CLAUDE_CODE) == frozenset()

    _pinned(monkeypatch, replace(explained, reason="   "))
    assert excluded_paths_from_pins(Provider.CLAUDE_CODE) == frozenset()

    _pinned(monkeypatch, explained)
    assert excluded_paths_from_pins(Provider.CLAUDE_CODE) == frozenset({"$.turns[].text"})


def test_a_pin_covers_every_element_the_schema_generates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pin names a schema position; planted values carry array indices.

    Without the index erased, a pin written against an array position matches
    no generated value and the exclusion vocabulary is inert.
    """
    assert normalise_path("$.turns[0].text") == "$.turns[].text"

    _pinned(
        monkeypatch,
        PinDecision(
            path="$.turns[].text",
            role="message_body",
            action="reject",
            reason="turn text is a render of the block below it",
        ),
    )
    excluded = excluded_paths_from_pins(Provider.CLAUDE_CODE)

    result = check_conservation(
        BODY_SCHEMA,
        [_payload("alpha", "beta")],
        [_session()],
        excluded_paths=excluded,
    )

    assert result.conserved
    assert result.planted_count == 0
    assert result.excluded_paths == ("$.turns[0].text", "$.turns[1].text")


def test_confirmed_pin_is_not_an_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirming an annotation asserts the value is content, not that it may vanish."""
    _pinned(
        monkeypatch,
        PinDecision(path="$.turns[].text", role="message_body", action="confirm", reason="this is the body"),
    )

    assert excluded_paths_from_pins(Provider.CLAUDE_CODE) == frozenset()


def test_receipt_measures_conservation_for_every_provider() -> None:
    """The production route measures, reports per provider, and gates nobody yet."""
    receipt = shared_wire_support_receipt()

    witnesses = [witness for entry in receipt.entries for witness in entry.parser_witnesses]
    assert witnesses, "receipt produced no parser witnesses to measure"
    assert all(witness.conservation is not None for witness in witnesses)

    planted_by_provider = {
        entry.provider: sum(
            witness.conservation.planted_count for witness in entry.parser_witnesses if witness.conservation
        )
        for entry in receipt.entries
        if entry.parser_witnesses
    }
    assert planted_by_provider, "no provider produced a witness"
    assert any(count > 0 for count in planted_by_provider.values()), (
        f"conservation is vacuous everywhere: {planted_by_provider}"
    )

    # Unadjudicated providers are measured without deciding the exit code.
    assert not CONSERVATION_BLOCKING_PROVIDERS
    enforced = {
        entry.provider
        for entry in receipt.entries
        for witness in entry.parser_witnesses
        if witness.conservation_enforced
    }
    assert enforced <= CONSERVATION_BLOCKING_PROVIDERS
    unconserved = [
        witness
        for entry in receipt.entries
        for witness in entry.parser_witnesses
        if witness.conservation is not None and not witness.conservation.conserved
    ]
    assert unconserved, "no finding measured; the check would report nothing to adjudicate"
    assert all(witness.conservation_conserved for witness in unconserved)


def test_receipt_payload_carries_the_findings_it_measured() -> None:
    """A finding no reader can see is not a report."""
    payload = shared_wire_support_receipt().to_dict()

    entries = payload["entries"]
    assert isinstance(entries, list)
    reported: list[tuple[object, object]] = []
    for entry in entries:
        assert isinstance(entry, Mapping)
        witnesses = entry["parser_witnesses"]
        assert isinstance(witnesses, list)
        for witness in witnesses:
            assert isinstance(witness, Mapping)
            conservation = witness["conservation"]
            assert isinstance(conservation, Mapping)
            findings = conservation["findings"]
            assert isinstance(findings, list)
            reported.extend((entry["provider"], finding) for finding in findings)

    assert reported, "receipt payload carries no conservation findings"
    assert all(isinstance(finding, str) and finding for _provider, finding in reported)
