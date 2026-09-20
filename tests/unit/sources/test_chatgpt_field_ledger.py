"""The ChatGPT field register is enforced, not merely written down.

polylogue-9193q.  ``sources/parsers/chatgpt.py`` carries a READ / EXCLUDED /
NOT PRESENT register naming what the parser does with every key it sees.  As
prose it detected nothing: a new upstream field was dropped with no entry
anywhere, which is how ~1.9 MB per export went missing.  The register now also
exists as ``CHATGPT_READ_KEYS`` / ``CHATGPT_EXCLUDED_KEYS``, and these tests
hold it against every committed synthetic ChatGPT export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.sources.parsers.chatgpt import (
    CHATGPT_EXCLUDED_KEYS,
    CHATGPT_FIELD_SCOPES,
    CHATGPT_READ_KEYS,
)
from tests.infra.chatgpt_field_census import census_export_file, census_export_keys

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "chatgpt"

#: Every committed synthetic ChatGPT conversation export.  A new fixture is
#: picked up without editing this file, which is the point: the next export
#: shape someone commits is censused the moment it lands.
_EXPORT_FIXTURES = sorted(
    path
    for path in _FIXTURE_DIR.glob("*.json")
    if isinstance(json.loads(path.read_text()), dict) and "mapping" in json.loads(path.read_text())
)


def _classify(observed: dict[str, set[str]]) -> dict[str, set[str]]:
    """Return ``scope -> keys in neither register``."""

    unclassified: dict[str, set[str]] = {}
    for scope, keys in observed.items():
        known = set(CHATGPT_READ_KEYS.get(scope, frozenset())) | set(CHATGPT_EXCLUDED_KEYS.get(scope, {}))
        if remainder := keys - known:
            unclassified[scope] = remainder
    return unclassified


def test_the_repository_has_committed_chatgpt_exports_to_census() -> None:
    """A walker over an empty corpus proves nothing.

    Anti-vacuity: this is the guard that stops every other test in the file
    from passing on zero fixtures.
    """

    assert _EXPORT_FIXTURES


@pytest.mark.parametrize("fixture", _EXPORT_FIXTURES, ids=lambda path: path.name)
def test_every_key_in_a_committed_export_is_read_or_excluded(fixture: Path) -> None:
    """The register covers every key the committed exports actually state.

    ``native-conversation-v1.json`` was authored independently of the
    register, so it is real evidence; ``native-conversation-field-census-v1``
    is built from the register's own vocabulary and its job is to keep the
    EXCLUDED entries and every scope of the walker exercised.

    Anti-vacuity: the sibling test below adds one novel key to the same walked
    structure and requires this check to report it.  A register that merely
    existed, or a test that only asserted the exclusion set is non-empty,
    would not survive that.
    """

    assert _classify(census_export_file(fixture)) == {}


def test_a_novel_upstream_key_is_reported_rather_than_dropped() -> None:
    """A field no one has classified must fail loudly.

    This is polylogue-9193q's own anti-vacuity condition executed as a test:
    add one metadata key to a fixture without touching the parser or the
    exclusion structure, and the check goes red.

    Anti-vacuity for this test: widen either register to accept unknown keys
    (for instance by classifying on a prefix, or by defaulting an unknown
    scope to "allow") and the assertions below stop holding.
    """

    conversation = json.loads(_EXPORT_FIXTURES[0].read_text())
    node = next(iter(conversation["mapping"].values()))
    node["message"].setdefault("metadata", {})["a_brand_new_upstream_field"] = "value"
    conversation["another_brand_new_conversation_field"] = "value"

    unclassified = _classify(census_export_keys(conversation))

    assert unclassified["message.metadata"] == {"a_brand_new_upstream_field"}
    assert unclassified["conversation"] == {"another_brand_new_conversation_field"}


def test_the_two_registers_are_disjoint_and_scoped() -> None:
    """A key cannot be both read and excluded, and neither invents a scope.

    Anti-vacuity: declare a key in both registers, or under a scope the walker
    never produces, and this goes red -- both are silent ways for the register
    to stop describing the parser.
    """

    assert set(CHATGPT_READ_KEYS) <= set(CHATGPT_FIELD_SCOPES)
    assert set(CHATGPT_EXCLUDED_KEYS) <= set(CHATGPT_FIELD_SCOPES)
    for scope in CHATGPT_FIELD_SCOPES:
        read = set(CHATGPT_READ_KEYS.get(scope, frozenset()))
        excluded = set(CHATGPT_EXCLUDED_KEYS.get(scope, {}))
        assert not (read & excluded), scope


def test_every_exclusion_states_a_reason() -> None:
    """An exclusion with no reason is the unexamined third state again.

    Anti-vacuity: add an entry with an empty reason and this goes red.
    """

    for scope, entries in CHATGPT_EXCLUDED_KEYS.items():
        for key, reason in entries.items():
            assert reason.strip(), f"{scope}.{key}"


def test_the_prioritised_fields_still_reach_their_destinations() -> None:
    """polylogue-9193q criterion 4, asserted through the production parser.

    ``finish_details`` -> ``messages.stop_reason``, ``default_model_slug`` ->
    ``messages.model_name`` and ``sessions.models_used``, and
    ``targeted_reply`` -> the ``chatgpt_targeted_reply`` event.  These are the
    three meaning-bearing fields the bead prioritised; the census fixture
    states all three.

    Anti-vacuity: delete any one of the three readers in
    ``sources/parsers/chatgpt.py`` and its own assertion goes red -- the
    register tests above would not notice, because a register entry is a
    claim about the parser and this is the check on the claim.
    """

    from polylogue.sources.parsers.chatgpt import parse

    fixture = _FIXTURE_DIR / "native-conversation-field-census-v1.json"
    session = parse(json.loads(fixture.read_text()), "field-census-v1")

    assert [message.stop_reason for message in session.messages] == ["max_tokens"]
    assert [message.model_name for message in session.messages] == ["synthetic-conversation-default-slug"]
    assert "synthetic-conversation-default-slug" in session.models_used
    targeted = [event for event in session.session_events if event.event_type == "chatgpt_targeted_reply"]
    assert len(targeted) == 1
    assert targeted[0].payload["targeted_reply_label"] == "Replying to"
