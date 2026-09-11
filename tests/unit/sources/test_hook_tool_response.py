"""Hook ``tool_response`` recovery for tool results the sidecar join left truncated.

Synthetic fixtures only -- no real command output, paths, or content (this
repo is public). The shapes mirror what a live corpus measurement found: a
``Bash`` response whose ``stdout`` is capped while ``persistedOutputSize``
reports the real size, and a whole-result response (WebFetch/MCP) that
recovers in full.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import BlockType, MaterialOrigin, Origin, Provider, Role
from polylogue.sources.live.hook_tool_response import (
    BASH_STDOUT_CAP_CHARS,
    HOOK_TOOL_RESPONSE_EVENT_TYPE,
    HookToolResponse,
    apply_hook_tool_responses,
    hook_response_text,
    recover_persisted_tool_results,
    resolve_hook_tool_responses,
    unresolved_persisted_truncations,
)
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent

_RECOVERED_NEEDLE = "zz_sentinel_needle_only_in_the_hook_payload"
_UNREACHABLE_SIDECAR = "/gone/projects/-p/sess-recover/tool-results/toolu_AAA.txt"


def _truncated_inline(pointer: str) -> str:
    return (
        "<persisted-output>\n"
        f"Output too large (5.0KB). Full output saved to: {pointer}\n\n"
        "Preview (first 2KB):\nshort preview text, does not carry the real payload"
    )


def _tool_result(tool_use_id: str, text: str) -> ParsedContentBlock:
    return ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id=tool_use_id, text=text, is_error=False)


def _session(*blocks: ParsedContentBlock, native_id: str = "sess-recover") -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=native_id,
        messages=[
            ParsedMessage(
                provider_message_id=f"m-{index}",
                role=Role.USER,
                material_origin=MaterialOrigin.TOOL_RESULT,
                position=index,
                blocks=[block],
            )
            for index, block in enumerate(blocks)
        ],
    )


def _write_post_tool_use(
    archive_root: Path,
    *,
    session_native_id: str,
    tool_use_id: str,
    tool_response: object,
    event_id: str = "e1",
) -> None:
    payload: dict[str, object] = {
        "event_type": "PostToolUse",
        "session_id": session_native_id,
        "payload": {"tool_use_id": tool_use_id, "tool_name": "Bash", "tool_response": tool_response},
    }
    with ArchiveStore(archive_root) as archive:
        archive.write_hook_event(
            provider=Provider.CLAUDE_CODE,
            payload=b"{}",
            source_path=f"/spool/{event_id}.json",
            acquired_at_ms=1,
            hook_event=ArchiveHookEvent(
                hook_event_id=f"hook:{event_id}",
                origin=Origin.CLAUDE_CODE_SESSION,
                source_path=f"/spool/{event_id}.json",
                event_type="PostToolUse",
                payload=payload,
                observed_at_ms=1,
                native_id=f"{session_native_id}:PostToolUse:{event_id}",
                session_native_id=session_native_id,
            ),
            carrier_source_id="spool",
            carrier_relative_path=f"{event_id}.json",
        )


def test_only_a_leading_envelope_counts_as_an_unresolved_truncation() -> None:
    """A tool result that merely *quotes* a transcript is not itself truncated.

    Anti-vacuity: drop the head bound in ``unresolved_persisted_truncations``
    and the quoting block is claimed too, so a full-text ``Read`` result would
    be overwritten with a hook payload belonging to a different call.
    """
    quoting = _tool_result(
        "toolu_QUOTE",
        "x" * 8000 + f'{{"content":"<persisted-output>\\nFull output saved to: {_UNREACHABLE_SIDECAR}"}}',
    )
    resolved = _tool_result("toolu_DONE", "the sidecar join already put the whole output here")
    truncated = _tool_result("toolu_AAA", _truncated_inline(_UNREACHABLE_SIDECAR))

    found = unresolved_persisted_truncations(_session(quoting, resolved, truncated))

    assert [item.tool_use_id for item in found] == ["toolu_AAA"]
    assert found[0].pointer == _UNREACHABLE_SIDECAR
    assert found[0].inline_chars == len(truncated.text or "")


def test_error_shape_envelope_pointer_drops_the_sentence_period() -> None:
    block = _tool_result(
        "toolu_MCP",
        f"Error: result (99,000 characters) exceeds maximum allowed tokens. "
        f"Output has been saved to {_UNREACHABLE_SIDECAR}.\nFormat: JSON",
    )
    (found,) = unresolved_persisted_truncations(_session(block))
    assert found.pointer == _UNREACHABLE_SIDECAR


def test_hook_response_text_reads_the_shapes_it_declares_and_no_others() -> None:
    capped = {
        "stdout": "o" * BASH_STDOUT_CAP_CHARS,
        "stderr": "",
        "persistedOutputSize": 342022,
    }
    assert hook_response_text(capped) == ("o" * BASH_STDOUT_CAP_CHARS, 342022)
    assert hook_response_text({"stdout": "out", "stderr": "err"}) == ("out\nerr", None)
    assert hook_response_text({"result": "whole result", "code": 200}) == ("whole result", None)
    assert hook_response_text("plain string response") == ("plain string response", None)
    # An Edit response carries no declared content key: guessing at
    # ``originalFile`` would replace a tool result with an unrelated file body.
    assert hook_response_text({"filePath": "/x.py", "originalFile": "unrelated body"}) is None
    assert hook_response_text({"stdout": ""}) is None
    assert hook_response_text(None) is None


def test_recovery_records_whether_the_hook_copy_was_whole() -> None:
    bounded = _tool_result("toolu_BASH", _truncated_inline(_UNREACHABLE_SIDECAR))
    whole = _tool_result("toolu_FETCH", _truncated_inline(_UNREACHABLE_SIDECAR))
    session = _session(bounded, whole)
    truncations = unresolved_persisted_truncations(session)

    recovered = apply_hook_tool_responses(
        session,
        truncations,
        {
            "toolu_BASH": HookToolResponse(
                tool_use_id="toolu_BASH",
                hook_event_id="hook:b",
                text=f"capped hook copy {_RECOVERED_NEEDLE} " + "o" * 4000,
                full_size=342022,
            ),
            "toolu_FETCH": HookToolResponse(
                tool_use_id="toolu_FETCH",
                hook_event_id="hook:f",
                text=f"whole hook copy {_RECOVERED_NEEDLE} " + "r" * 4000,
                full_size=None,
            ),
        },
    )

    texts = {block.tool_id: block.text for message in recovered.messages for block in message.blocks}
    assert _RECOVERED_NEEDLE in (texts["toolu_BASH"] or "")
    assert _RECOVERED_NEEDLE in (texts["toolu_FETCH"] or "")

    events = {
        str(event.payload["tool_use_id"]): event.payload
        for event in recovered.session_events
        if event.event_type == HOOK_TOOL_RESPONSE_EVENT_TYPE
    }
    assert events["toolu_BASH"]["recovery_complete"] is False
    assert events["toolu_BASH"]["reported_full_size"] == 342022
    assert events["toolu_FETCH"]["recovery_complete"] is True
    assert all(payload["content_replaced"] is True for payload in events.values())
    assert len(recovered.messages) == len(session.messages)


def test_recovery_keeps_the_transcript_preview_when_the_hook_adds_nothing() -> None:
    block = _tool_result("toolu_AAA", _truncated_inline(_UNREACHABLE_SIDECAR))
    session = _session(block)
    truncations = unresolved_persisted_truncations(session)

    recovered = apply_hook_tool_responses(
        session,
        truncations,
        {
            "toolu_AAA": HookToolResponse(
                tool_use_id="toolu_AAA",
                hook_event_id="hook:a",
                text="tiny",
                full_size=342022,
            )
        },
    )

    (kept,) = [block for message in recovered.messages for block in message.blocks]
    assert kept.text == block.text
    (event,) = recovered.session_events
    assert event.payload["content_replaced"] is False


def test_absent_hook_evidence_is_recorded_not_guessed() -> None:
    session = _session(_tool_result("toolu_AAA", _truncated_inline(_UNREACHABLE_SIDECAR)))
    recovered = apply_hook_tool_responses(session, unresolved_persisted_truncations(session), {})
    (event,) = recovered.session_events
    assert event.payload["acquisition_status"] == "absent"
    assert event.payload["pointer"] == _UNREACHABLE_SIDECAR


def test_hook_responses_resolve_from_the_durable_source_tier(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    _write_post_tool_use(
        archive_root,
        session_native_id="sess-parent",
        tool_use_id="toolu_AAA",
        tool_response={"stdout": f"hook stdout {_RECOVERED_NEEDLE}", "stderr": "", "persistedOutputSize": 900},
    )

    resolved = resolve_hook_tool_responses(
        archive_root,
        origin=Origin.CLAUDE_CODE_SESSION.value,
        session_native_ids=("sess-parent",),
        tool_use_ids=("toolu_AAA",),
    )

    assert _RECOVERED_NEEDLE in resolved["toolu_AAA"].text
    assert resolved["toolu_AAA"].full_size == 900
    assert resolved["toolu_AAA"].complete is False
    assert (
        resolve_hook_tool_responses(
            archive_root,
            origin=Origin.CLAUDE_CODE_SESSION.value,
            session_native_ids=("sess-parent",),
            tool_use_ids=("toolu_OTHER",),
        )
        == {}
    )


def test_missing_source_tier_degrades_to_no_evidence(tmp_path: Path) -> None:
    assert (
        resolve_hook_tool_responses(
            tmp_path / "no-archive-here",
            origin=Origin.CLAUDE_CODE_SESSION.value,
            session_native_ids=("sess-parent",),
            tool_use_ids=("toolu_AAA",),
        )
        == {}
    )


def test_subagent_session_recovers_from_its_parents_hook_journal(tmp_path: Path) -> None:
    """A subagent's tool calls are journalled under the parent session's id.

    Anti-vacuity: look up only the session's own native id and this recovery
    finds nothing, because the hook never carried the ``:agent-`` suffix.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    _write_post_tool_use(
        archive_root,
        session_native_id="sess-parent",
        tool_use_id="toolu_AAA",
        # Recovery never replaces a longer provider preview with a shorter
        # hook copy, so this whole-result payload must exceed the envelope.
        tool_response={"result": f"whole hook copy {_RECOVERED_NEEDLE} " + "r" * 512},
    )
    session = _session(
        _tool_result("toolu_AAA", _truncated_inline(_UNREACHABLE_SIDECAR)),
        native_id="sess-parent:agent-abc123",
    )

    recovered = recover_persisted_tool_results(session, archive_root=archive_root)

    (block,) = [block for message in recovered.messages for block in message.blocks]
    assert _RECOVERED_NEEDLE in (block.text or "")
    (event,) = recovered.session_events
    assert event.payload["acquisition_status"] == "matched"
    assert event.payload["recovery_complete"] is True


def test_a_session_with_no_truncation_is_returned_untouched(tmp_path: Path) -> None:
    """No unresolved pointer means no source-tier read at all.

    Anti-vacuity: an archive root that does not exist would still be opened
    (and would still return ``{}``) if the guard were removed -- the identity
    assertion is what proves the read was skipped.
    """
    session = _session(_tool_result("toolu_DONE", "the sidecar join already put the whole output here"))
    assert recover_persisted_tool_results(session, archive_root=tmp_path / "absent") is session
