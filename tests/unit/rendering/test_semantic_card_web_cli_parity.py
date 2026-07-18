"""Anti-fragmentation parity harness (webui-04 / polylogue-ap7).

Both the WebUI v2 transcript renderer and the CLI markdown reader project the
SAME ``SemanticTranscript`` (``polylogue.rendering.semantic_cards.build_semantic_transcript``)
rather than reclassifying tools independently. This module proves that for one
session mixing a failed shell result, a forked lineage relationship, and an
unknown provider tool, the web-serving path (``semantic_card_placement_for_messages``,
which daemon/http.py calls verbatim for every archive-backed session read) and
the CLI's own markdown projection (``render_semantic_transcript_markdown``)
agree on card count, family (kind), and structural outcome. A regression here
means one surface silently reclassified a tool or dropped a card the other
still shows - exactly the drift ap7's acceptance criteria forbid.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, BranchType, Provider
from polylogue.rendering.semantic_card_models import SemanticCardKind
from polylogue.rendering.semantic_card_placement import semantic_card_placement_for_messages
from polylogue.rendering.semantic_cards import build_semantic_transcript, lineage_descriptor_from_archive_envelope
from polylogue.rendering.semantic_markdown import render_semantic_transcript_markdown
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _seed_parity_fixture(archive_root: Path) -> tuple[str, str]:
    with ArchiveStore(archive_root) as archive:
        parent_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="parity-parent",
                title="Parity parent",
                messages=[
                    ParsedMessage(provider_message_id="p0", role=Role.USER, text="run the flaky suite"),
                    ParsedMessage(
                        provider_message_id="p1",
                        role=Role.ASSISTANT,
                        text="running pytest",
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="shell",
                                tool_id="tool-shell",
                                tool_input={"command": "pytest -k flaky"},
                            ),
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="tool-shell",
                                text="2 failed, 8 passed",
                                is_error=True,
                                exit_code=2,
                            ),
                        ],
                    ),
                ],
            )
        )
        child_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="parity-child",
                title="Parity child",
                parent_session_provider_id="parity-parent",
                branch_type=BranchType.FORK,
                messages=[
                    ParsedMessage(provider_message_id="p0", role=Role.USER, text="run the flaky suite"),
                    ParsedMessage(
                        provider_message_id="p1",
                        role=Role.ASSISTANT,
                        text="running pytest",
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="shell",
                                tool_id="tool-shell",
                                tool_input={"command": "pytest -k flaky"},
                            ),
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="tool-shell",
                                text="2 failed, 8 passed",
                                is_error=True,
                                exit_code=2,
                            ),
                        ],
                    ),
                    ParsedMessage(
                        provider_message_id="c0",
                        role=Role.ASSISTANT,
                        text="trying an unfamiliar tool",
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="mystery_tool",
                                tool_id="tool-mystery",
                                tool_input={"payload": "not json-shaped"},
                            ),
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="tool-mystery",
                                text="unrecognized response",
                            ),
                        ],
                    ),
                ],
            )
        )
    return parent_id, child_id


def test_web_json_and_cli_markdown_agree_on_card_family_and_outcome(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _parent_id, child_id = _seed_parity_fixture(archive_root)

    with ArchiveStore(archive_root) as archive:
        resolved_child_id = archive.resolve_session_id(child_id)
        envelope = archive.read_session(resolved_child_id)

    # The web-serving path: exactly what daemon/http.py's
    # ``_archive_semantic_card_placement``/``_do_archive_get_session`` calls
    # for every archive-backed session read.
    lineage = lineage_descriptor_from_archive_envelope(envelope)
    web_placement = semantic_card_placement_for_messages(
        envelope.messages,
        session_id=envelope.session_id,
        provider_family=envelope.origin,
        lineage=lineage,
    )
    web_cards: list[dict[str, object]] = [
        cast(dict[str, object], entry["card"])
        for message in envelope.messages
        for entry in web_placement.entries_for(str(message.message_id))
        if entry.get("entry_type") == "card"
    ] + [
        cast(dict[str, object], entry["card"])
        for entry in web_placement.session_entries
        if entry.get("entry_type") == "card"
    ]

    # The CLI path: build_semantic_transcript + render_semantic_transcript_markdown,
    # the exact pair polylogue/cli/messages.py's markdown branch calls.
    transcript = build_semantic_transcript(
        envelope.messages,
        session_id=envelope.session_id,
        provider_family=envelope.origin,
        lineage=lineage,
    )
    cli_markdown = render_semantic_transcript_markdown(transcript)

    web_kinds = sorted(cast(str, card["kind"]) for card in web_cards)
    transcript_kinds = sorted(card.kind.value for card in transcript.cards)
    assert web_kinds == transcript_kinds, "web JSON card kinds must match the shared transcript's card kinds"
    assert len(web_cards) == len(transcript.cards)

    def _web_outcome(card: dict[str, object]) -> tuple[object, object, object]:
        outcome = card.get("outcome")
        outcome_map = cast(dict[str, object], outcome) if isinstance(outcome, dict) else {}
        return cast(str, card["kind"]), outcome_map.get("state"), outcome_map.get("exit_code")

    web_outcomes = sorted(_web_outcome(card) for card in web_cards)
    transcript_outcomes = sorted(
        (
            card.kind.value,
            card.outcome.state.value if card.outcome else None,
            card.outcome.exit_code if card.outcome else None,
        )
        for card in transcript.cards
    )
    assert web_outcomes == transcript_outcomes, "web JSON outcomes (state/exit_code) must match the shared transcript"

    assert SemanticCardKind.SHELL.value in web_kinds, "the failed shell result must classify as a shell card"
    assert SemanticCardKind.FALLBACK.value in web_kinds, "the unknown tool must classify as a fallback card, not drop"
    assert SemanticCardKind.LINEAGE.value in web_kinds, "the fork relationship must surface a lineage card"

    for card in transcript.cards:
        # Every card the shared transcript produces must be independently
        # visible in the CLI's markdown projection - by title (unique per
        # card in this fixture) so neither surface silently absorbs one
        # family's card into another's rendering.
        assert card.title in cli_markdown, f"CLI markdown dropped the {card.kind.value} card {card.title!r}"

    shell_card = next(card for card in transcript.cards if card.kind is SemanticCardKind.SHELL)
    assert shell_card.outcome is not None
    assert shell_card.outcome.state.value == "failed"
    assert shell_card.outcome.exit_code == 2
    assert "exit_code=2" in cli_markdown

    fallback_card = next(card for card in transcript.cards if card.kind is SemanticCardKind.FALLBACK)
    assert fallback_card.raw_evidence is not None
