"""Claude Code ``tool-results/`` sidecar acquisition (polylogue-rujy).

Synthetic fixtures only -- no real command output, paths, or content (this
repo is public). Mirrors the two real Claude Code sidecar shapes measured
against a live corpus: a truncated "output too large" overflow pointer, and a
never-truncated mirror of content already fully present inline.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import BlockType
from polylogue.sources.live.tool_result_sidecars import (
    SidecarDebt,
    SidecarMatch,
    join_tool_result_sidecars,
)
from polylogue.sources.parsers.claude.code_parser import apply_tool_result_sidecars, parse_code

_TRUNCATED_NEEDLE = "zz_sentinel_needle_only_in_full_output"


def _write_sidecar(tool_results_dir: Path, name: str, text: str) -> None:
    tool_results_dir.mkdir(parents=True, exist_ok=True)
    (tool_results_dir / name).write_text(text, encoding="utf-8")


def _record(uuid: str, tool_use_id: str, inline_content: str) -> dict[str, object]:
    return {
        "type": "user",
        "uuid": uuid,
        "sessionId": "sess-sidecar",
        "timestamp": 1704067200,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": inline_content,
                }
            ],
        },
    }


def _truncated_inline(sidecar_path: Path) -> str:
    return (
        "<persisted-output>\n"
        f"Output too large (5.0KB). Full output saved to: {sidecar_path}\n\n"
        "Preview (first 2KB):\nshort preview text, does not contain the real payload"
    )


def test_join_tool_result_sidecars_classifies_truncated_full_mirror_and_debt(tmp_path: Path) -> None:
    tool_results_dir = tmp_path / "tool-results"
    truncated_path = tool_results_dir / "toolu_AAA.txt"
    _write_sidecar(tool_results_dir, "toolu_AAA.txt", f"full output text {_TRUNCATED_NEEDLE}")
    _write_sidecar(tool_results_dir, "toolu_BBB.txt", "The file /x.py has been updated successfully.")
    _write_sidecar(tool_results_dir, "orphan123.txt", "no owning tool_result block references this file")
    _write_sidecar(tool_results_dir, "hook-deadbeef-stdout.txt", "hook stdout capture, a distinct mechanism")

    payload = [
        _record("m-aaa", "toolu_AAA", _truncated_inline(truncated_path)),
        _record("m-bbb", "toolu_BBB", "The file /x.py has been updated successfully."),
    ]

    result = join_tool_result_sidecars(payload, tool_results_dir)

    matched_by_id = {match.tool_use_id: match for match in result.matched}
    assert set(matched_by_id) == {"toolu_AAA", "toolu_BBB"}

    truncated_match = matched_by_id["toolu_AAA"]
    assert truncated_match.was_truncated is True
    assert _TRUNCATED_NEEDLE in truncated_match.full_text
    assert truncated_match.byte_size == len(truncated_match.full_text.encode("utf-8"))

    full_mirror_match = matched_by_id["toolu_BBB"]
    assert full_mirror_match.was_truncated is False
    assert full_mirror_match.full_text == "The file /x.py has been updated successfully."

    # The hook-*.txt capture is a distinct, already-tracked mechanism (raw hook
    # stdout, not tool_result content) -- never surfaced as debt.
    debt_filenames = {debt.filename for debt in result.debt}
    assert debt_filenames == {"orphan123.txt"}
    assert result.debt[0].reason == "no_owning_tool_result_block"


def test_join_tool_result_sidecars_returns_empty_result_when_dir_absent(tmp_path: Path) -> None:
    result = join_tool_result_sidecars([], tmp_path / "does-not-exist")
    assert result.matched == ()
    assert result.debt == ()


def test_apply_tool_result_sidecars_replaces_truncated_block_text_only(tmp_path: Path) -> None:
    tool_results_dir = tmp_path / "tool-results"
    truncated_path = tool_results_dir / "toolu_AAA.txt"
    _write_sidecar(tool_results_dir, "toolu_AAA.txt", f"full output text {_TRUNCATED_NEEDLE}")
    _write_sidecar(tool_results_dir, "toolu_BBB.txt", "The file /x.py has been updated successfully.")
    _write_sidecar(tool_results_dir, "orphan123.txt", "no owning tool_result block references this file")

    payload = [
        _record("m-aaa", "toolu_AAA", _truncated_inline(truncated_path)),
        _record("m-bbb", "toolu_BBB", "The file /x.py has been updated successfully."),
    ]

    join_result = join_tool_result_sidecars(payload, tool_results_dir)

    baseline = parse_code(payload, "fallback-sidecar")
    acquired = parse_code(payload, "fallback-sidecar", tool_result_sidecars=join_result)

    # AC1: acquisition attaches to existing blocks; it never changes session
    # shape (message count, provider ids) -- a test asserting this fails if a
    # future change starts materializing sidecar files as their own messages.
    assert len(acquired.messages) == len(baseline.messages)
    assert [m.provider_message_id for m in acquired.messages] == [m.provider_message_id for m in baseline.messages]

    by_id = {message.provider_message_id: message for message in acquired.messages}
    aaa_block = next(block for block in by_id["m-aaa"].blocks if block.type is BlockType.TOOL_RESULT)
    bbb_block = next(block for block in by_id["m-bbb"].blocks if block.type is BlockType.TOOL_RESULT)

    # AC2: the term that only exists in the full output is now the block's
    # searchable text (FTS indexes `blocks.search_text`, populated from block
    # content) -- fails if the truncated preview is kept instead of the
    # acquired full text.
    assert _TRUNCATED_NEEDLE in (aaa_block.text or "")
    assert "Preview (first 2KB)" not in (aaa_block.text or "")

    # A never-truncated mirror sidecar must NOT overwrite already-full content.
    assert bbb_block.text == "The file /x.py has been updated successfully."

    events_by_status = {
        (event.payload.get("acquisition_status"), event.payload.get("tool_use_id") or event.payload.get("filename"))
        for event in acquired.session_events
        if event.event_type == "claude_tool_result_sidecar"
    }
    assert ("matched", "toolu_AAA") in events_by_status
    assert ("matched", "toolu_BBB") in events_by_status
    # AC3: an unmatched sidecar is recorded as typed acquisition debt, not
    # silently dropped -- fails if debt entries stop being turned into events.
    assert ("debt", "orphan123.txt") in events_by_status

    matched_aaa_event = next(
        event
        for event in acquired.session_events
        if event.event_type == "claude_tool_result_sidecar" and event.payload.get("tool_use_id") == "toolu_AAA"
    )
    assert matched_aaa_event.payload["content_replaced"] is True
    assert isinstance(matched_aaa_event.payload["content_hash"], str)
    assert len(matched_aaa_event.payload["content_hash"]) == 64


def test_apply_tool_result_sidecars_is_a_noop_without_matches_or_debt() -> None:
    from polylogue.sources.live.tool_result_sidecars import SidecarJoinResult

    payload = [_record("m-ccc", "toolu_CCC", "small inline result")]
    parsed = parse_code(payload, "fallback-noop")
    result = apply_tool_result_sidecars(parsed, SidecarJoinResult())
    assert result is parsed


def test_sidecar_dataclasses_are_frozen() -> None:
    match = SidecarMatch(
        tool_use_id="toolu_X",
        filename="toolu_X.txt",
        byte_size=1,
        content_hash="0" * 64,
        was_truncated=True,
        full_text="x",
    )
    debt = SidecarDebt(filename="orphan.txt", byte_size=1, reason="no_owning_tool_result_block")
    assert match.tool_use_id == "toolu_X"
    assert debt.reason == "no_owning_tool_result_block"
