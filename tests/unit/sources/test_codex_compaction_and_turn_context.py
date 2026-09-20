"""The two codex-parser branches polylogue-xdr8w could not reach.

The bead filed both defects as "plain on inspection, no reproducing input
found", and its acceptance made exhibiting the reaching record shape the first
deliverable.  Both shapes are exhibited here, built as records the provider
writes rather than as a guess at internal state:

* the COMPACTED branch is entered by a ``{"type": "compacted", "payload":
  {"replacement_history": [...]}}`` record whose re-embedded content text is
  longer than ``_CODEX_REPLACEMENT_CONTEXT_MAX_CHARS`` (256 KiB), which is
  what routes it into the digest-only channel;
* the TURN_CONTEXT branch is entered by ``{"type": "turn_context", "payload":
  {"user_instructions": ...}}`` records that restate a *different* prompt,
  which is what makes the parser consult its revision register at all.
"""

from __future__ import annotations

import json

import pytest

from polylogue.sources.parsers.codex import parse

_OVERSIZED = "A" * (256 * 1024 + 10)


def _rollout(*records: dict[str, object]) -> list[object]:
    return [
        {"type": "session_meta", "payload": {"id": "xdr8w", "timestamp": "2026-01-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "m0",
                "role": "user",
                "timestamp": "2026-01-01T00:00:01Z",
                "content": [{"type": "input_text", "text": "first turn"}],
            },
        },
        *records,
    ]


def _compacted(text: str) -> dict[str, object]:
    return {
        "type": "compacted",
        "timestamp": "2026-01-01T00:00:02Z",
        "payload": {
            "message": "compaction summary",
            "replacement_history": [
                {
                    "type": "message",
                    "role": "user",
                    "phase": "pre",
                    "content": [{"type": "input_text", "text": text}],
                }
            ],
        },
    }


def test_the_compacted_branch_is_reachable_and_digests_oversized_history() -> None:
    """polylogue-xdr8w criterion 1, compacted half.

    Anti-vacuity: shorten the content text below
    ``_CODEX_REPLACEMENT_CONTEXT_MAX_CHARS`` and no
    ``codex_replacement_context_omitted`` event is produced at all -- the
    branch under test is not entered, which is exactly the state that left
    this bead open.
    """

    session = parse(_rollout(_compacted(_OVERSIZED)), "xdr8w")

    omitted = [event for event in session.session_events if event.event_type == "codex_replacement_context_omitted"]
    assert len(omitted) == 1
    assert omitted[0].payload["content_chars"] == len(_OVERSIZED)
    assert omitted[0].payload["content_sha256"]


def test_a_lone_surrogate_in_replacement_history_does_not_break_the_parse() -> None:
    """polylogue-xdr8w criterion 2.

    ``json.loads`` admits a lone surrogate as a ``str``; the digest then
    encoded it as strict UTF-8.  The text is built by decoding real JSON bytes
    carrying a ``\\ud800`` escape, so this is a rollout the provider could
    write, not a hand-assembled Python string.

    Anti-vacuity: drop ``errors="surrogatepass"`` from the digest encodes in
    ``sources/parsers/codex.py`` and this raises ``UnicodeEncodeError``
    ("surrogates not allowed") instead of parsing -- verified by reverting.
    """

    encoded = json.dumps(_compacted(_OVERSIZED + "\\ud800")).replace("\\\\ud800", "\\ud800")
    record = json.loads(encoded)
    text = record["payload"]["replacement_history"][0]["content"][0]["text"]
    assert text.endswith("\ud800")

    session = parse(_rollout(record), "xdr8w")

    omitted = [event for event in session.session_events if event.event_type == "codex_replacement_context_omitted"]
    assert len(omitted) == 1
    assert omitted[0].payload["content_chars"] == len(text)


def test_the_turn_context_branch_is_reachable_and_numbers_revisions() -> None:
    """polylogue-xdr8w criterion 1, turn_context half.

    Anti-vacuity: restate the SAME ``user_instructions`` on every turn and no
    ``codex_instructions_changed`` event is produced -- the register is never
    consulted, which is why the triage's synthetic rollouts saw nothing here.
    """

    records: list[dict[str, object]] = []
    for index in range(4):
        records.append(
            {
                "type": "turn_context",
                "timestamp": f"2026-01-01T00:01:0{index}Z",
                "payload": {"cwd": "/repo", "user_instructions": f"prompt revision {index}"},
            }
        )
    session = parse(_rollout(*records), "xdr8w")

    changed = [
        event
        for event in session.session_events
        if event.event_type == "codex_instructions_changed"
        and event.payload["instructions_kind"] == "user_instructions"
    ]
    assert [event.payload["revision"] for event in changed] == [2, 3, 4]
    assert [event.payload["instructions"] for event in changed] == [
        "prompt revision 1",
        "prompt revision 2",
        "prompt revision 3",
    ]


class _CountingKey(str):
    """A value that records every equality comparison made against it."""

    __slots__ = ("comparisons",)

    comparisons: int

    def __new__(cls, value: str) -> _CountingKey:
        key = super().__new__(cls, value)
        key.comparisons = 0
        return key

    def __eq__(self, other: object) -> bool:
        self.comparisons += 1
        return str.__eq__(self, other)

    def __hash__(self) -> int:
        return str.__hash__(self)


def test_revision_membership_does_not_scan_the_prior_revisions() -> None:
    """polylogue-xdr8w criterion 3: observe the work, not the result.

    A set and a list produce identical parser output, so the only honest
    assertion is on the work done.  ``_CodexInstructionRevisions`` is the
    production container the parser now consults; filling it with values that
    count equality comparisons against themselves shows a membership miss
    costs no comparisons at all.

    Anti-vacuity: give ``_CodexInstructionRevisions.__contains__`` the list
    (``return value in self._order``) and the total climbs to one comparison
    per stored revision -- 64 here, and O(turns x revisions) over a real
    rollout.
    """

    from polylogue.sources.parsers.codex import _CodexInstructionRevisions

    register = _CodexInstructionRevisions()
    keys = [_CountingKey(f"revision {index}") for index in range(64)]
    for key in keys:
        register.add(key)
    for key in keys:
        key.comparisons = 0

    assert "a revision never seen before" not in register

    assert sum(key.comparisons for key in keys) == 0
    assert len(register) == 64
    assert register.values()[0] == "revision 0"


@pytest.mark.parametrize("revisions", [8, 64])
def test_the_register_answers_identically_whatever_its_size(revisions: int) -> None:
    """The dedup's observable contract is unchanged by the container swap.

    Anti-vacuity: make ``add`` skip the ordered list and the revision numbers
    below collapse, because a revision number is a first-seen position.
    """

    from polylogue.sources.parsers.codex import _CodexInstructionRevisions

    register = _CodexInstructionRevisions()
    for index in range(revisions):
        value = f"revision {index}"
        assert value not in register
        register.add(value)
        assert value in register

    assert len(register) == revisions
    assert register.values() == tuple(f"revision {index}" for index in range(revisions))
