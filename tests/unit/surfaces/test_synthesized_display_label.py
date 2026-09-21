"""A recognized prompt echo is never rendered as if it were a title.

polylogue-4p1.6: ``sessions.title_source = 'heuristic'`` records that the
parser recognized the stored title as an echo of the user's own opening
prompt. Every list surface rendered it verbatim anyway, because the read-time
display projection never consulted ``title_source``.

These tests run the production route -- ``ArchiveStore.read_summary`` /
``list_summaries`` -> ``archive_summary_to_domain`` -> the surface envelope --
rather than calling the label helper directly.

Anti-vacuity: restore ``display_label = provider_title or display_name or
structural_label`` in ``_summary_from_row`` (or drop the ``title_source``
branch from ``DisplayTitleTagsMixin.explicit_display_title``) and
``test_heuristic_title_is_replaced_by_a_composed_label`` goes red, because
the echo reappears as the rendered title.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.hydration import archive_summary_to_domain
from polylogue.core.enums import BlockType, DisplayLabelSource, Provider, Role, TitleSource
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.surfaces.payloads import (
    session_list_envelope_from_summary,
    session_summary_envelope_from_summary,
)

ECHOED_PROMPT = "please look at the failing test in the parser and tell me what is wrong"


def _write_session(
    db_path: Path,
    *,
    native_id: str,
    title: str | None,
    title_source: TitleSource | None,
    tool_calls: tuple[str, ...] = (),
) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        messages = [
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text=ECHOED_PROMPT,
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=ECHOED_PROMPT)],
            )
        ]
        for index, tool_name in enumerate(tool_calls, start=1):
            messages.append(
                ParsedMessage(
                    provider_message_id=f"t{index}",
                    role=Role.ASSISTANT,
                    text="",
                    position=index,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name=tool_name,
                            tool_id=f"call-{index}",
                            tool_input={"file_path": f"/repo/src/mod{index}.py"},
                        )
                    ],
                )
            )
        write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                title=title,
                title_source=title_source,
                messages=messages,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _bootstrap(tmp_path: Path) -> Path:
    with ArchiveStore(tmp_path, initialize=True, read_only=False):
        pass
    return tmp_path / "index.db"


def test_heuristic_title_is_replaced_by_a_composed_label(tmp_path: Path) -> None:
    db_path = _bootstrap(tmp_path)
    _write_session(
        db_path,
        native_id="codex-echo-1",
        title=ECHOED_PROMPT,
        title_source=TitleSource.HEURISTIC,
        tool_calls=("Edit", "Edit", "Read"),
    )

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        session_id = archive.resolve_session_id("codex-echo-1")
        summary = archive.read_summary(session_id)

        # The stored evidence is preserved: this is not a deletion.
        assert summary.title == ECHOED_PROMPT
        assert summary.title_source == "heuristic"

        # ...but the rendered label is composed, and says so.
        assert summary.display_label != ECHOED_PROMPT
        assert summary.display_label_source == DisplayLabelSource.SYNTHESIZED.value

        domain = archive_summary_to_domain(summary)
        assert domain.display_title == summary.display_label
        assert domain.display_title_is_synthesized is True

        envelope = session_summary_envelope_from_summary(domain)
        assert envelope.title == summary.display_label
        assert envelope.title_is_synthesized is True
        # The payload still reports where the *stored* title came from, which
        # is exactly why the synthesized flag has to travel beside it.
        assert envelope.title_source == "heuristic"

        row = session_list_envelope_from_summary(domain, message_count=4)
        assert row.title == summary.display_label
        assert row.title_is_synthesized is True


def test_composed_label_names_the_dominant_action_family(tmp_path: Path) -> None:
    db_path = _bootstrap(tmp_path)
    _write_session(
        db_path,
        native_id="codex-echo-2",
        title=ECHOED_PROMPT,
        title_source=TitleSource.HEURISTIC,
        tool_calls=("Edit", "Edit", "Read"),
    )

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        summary = archive.read_summary(archive.resolve_session_id("codex-echo-2"))

    label = summary.display_label or ""
    # The dominant family (two Edit calls beat one Read) is a component of the
    # label, so sibling sessions in one repo differ by what they did.
    assert "file_edit" in label, label
    assert "msgs" in label


def test_origin_titled_sessions_are_unchanged(tmp_path: Path) -> None:
    db_path = _bootstrap(tmp_path)
    _write_session(
        db_path,
        native_id="codex-origin-1",
        title="Fix the flaky parser test",
        title_source=TitleSource.ORIGIN,
    )

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        summary = archive.read_summary(archive.resolve_session_id("codex-origin-1"))
        assert summary.display_label == "Fix the flaky parser test"
        assert summary.display_label_source == DisplayLabelSource.ORIGIN.value

        domain = archive_summary_to_domain(summary)
        assert domain.display_title == "Fix the flaky parser test"
        assert domain.display_title_is_synthesized is False

        envelope = session_summary_envelope_from_summary(domain)
        assert envelope.title == "Fix the flaky parser test"
        assert envelope.title_is_synthesized is False


def test_sibling_echo_sessions_do_not_collapse_to_one_label(tmp_path: Path) -> None:
    """Two sessions that opened with the same boilerplate prompt differ.

    The composed label carries no prompt text at all, so a shared boilerplate
    prefix cannot make siblings identical; they separate on their own
    structural evidence.
    """
    db_path = _bootstrap(tmp_path)
    _write_session(
        db_path,
        native_id="codex-sib-1",
        title=ECHOED_PROMPT,
        title_source=TitleSource.HEURISTIC,
        tool_calls=("Edit",),
    )
    _write_session(
        db_path,
        native_id="codex-sib-2",
        title=ECHOED_PROMPT,
        title_source=TitleSource.HEURISTIC,
        tool_calls=("Edit", "Edit", "Bash", "Bash", "Bash"),
    )

    with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
        first = archive.read_summary(archive.resolve_session_id("codex-sib-1"))
        second = archive.read_summary(archive.resolve_session_id("codex-sib-2"))

    assert first.display_label != second.display_label
    assert ECHOED_PROMPT not in (first.display_label or "")
    assert ECHOED_PROMPT not in (second.display_label or "")


def test_no_network_or_llm_call_in_the_derivation_path(tmp_path: Path) -> None:
    """The label is composed from SQL reads only.

    Anti-vacuity: an LLM or HTTP call added to the composition would trip the
    socket guard this test installs.
    """
    import socket

    db_path = _bootstrap(tmp_path)
    _write_session(
        db_path,
        native_id="codex-echo-3",
        title=ECHOED_PROMPT,
        title_source=TitleSource.HEURISTIC,
        tool_calls=("Edit",),
    )

    original = socket.socket

    class _Refused(socket.socket):
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("the display-label derivation opened a socket")

    socket.socket = _Refused  # type: ignore[misc]
    try:
        with ArchiveStore(tmp_path, initialize=False, read_only=True) as archive:
            summary = archive.read_summary(archive.resolve_session_id("codex-echo-3"))
    finally:
        socket.socket = original  # type: ignore[misc]

    assert summary.display_label_source == DisplayLabelSource.SYNTHESIZED.value
