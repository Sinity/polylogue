"""Claude Code ``tool-results/`` sidecar acquisition (polylogue-rujy).

Synthetic fixtures only -- no real command output, paths, or content (this
repo is public). Mirrors the two real Claude Code sidecar shapes measured
against a live corpus: a truncated "output too large" overflow pointer, and a
never-truncated mirror of content already fully present inline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from polylogue.config import Source
from polylogue.core.enums import BlockType, Provider
from polylogue.sources import value_bounds
from polylogue.sources.live.sidecar_resolution import FilesystemSidecarResolver
from polylogue.sources.live.tool_result_sidecars import (
    SidecarDebt,
    SidecarMatch,
    join_tool_result_sidecars,
    join_tool_result_sidecars_session_scoped,
    resolve_sibling_transcript_paths,
    sidecar_files_from_directory,
)
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.sources.parsers.claude.code_parser import apply_tool_result_sidecars, parse_code
from polylogue.sources.revision_backfill import _parse_one
from polylogue.sources.sidecar_evidence import (
    CapturedSidecarResolver,
    RetainedSidecarFile,
    RetainedSidecarScope,
    SiblingTranscript,
)
from polylogue.sources.source_parsing import iter_source_sessions_with_raw

_TRUNCATED_NEEDLE = "zz_sentinel_needle_only_in_full_output"


def test_captured_sidecar_resolver_has_no_ambient_path_fallback(tmp_path: Path) -> None:
    """Detached parsing sees only the explicitly captured scope for a path."""
    source_path = tmp_path / "project" / "session.jsonl"
    staged = tmp_path / "captured-output.txt"
    staged.write_text("captured", encoding="utf-8")
    scope = RetainedSidecarScope(
        scope_key=str(tmp_path / "project" / "session" / "tool-results"),
        files=(RetainedSidecarFile("output.txt", 8, None, lambda: staged.read_text()),),
        available=True,
        witness=(("file", "raw-captured", "a" * 64),),
    )
    resolver = CapturedSidecarResolver({source_path.as_posix(): scope})

    assert resolver.claude_code_scope(source_path) is scope
    unresolved = resolver.claude_code_scope(tmp_path / "project" / "other.jsonl")
    assert unresolved.available is False
    assert unresolved.files == ()
    assert unresolved.scope_key == ""


def _dir_scope(tool_results_dir: Path) -> RetainedSidecarScope:
    """The scope an acquisition-time resolver reports for one directory.

    The join no longer enumerates a path (polylogue-cq1ql); these
    single-transcript cases still exercise the directory shape, so they build
    the scope the filesystem resolver would.
    """
    return RetainedSidecarScope(
        scope_key=str(tool_results_dir),
        files=sidecar_files_from_directory(tool_results_dir),
        available=tool_results_dir.is_dir(),
    )


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

    result = join_tool_result_sidecars(payload, _dir_scope(tool_results_dir))

    matched_by_id = {match.tool_use_id: match for match in result.matched}
    assert set(matched_by_id) == {"toolu_AAA", "toolu_BBB"}

    truncated_match = matched_by_id["toolu_AAA"]
    assert truncated_match.was_truncated is True
    assert _TRUNCATED_NEEDLE in truncated_match.read_text()
    assert truncated_match.byte_size == len(truncated_match.read_text().encode("utf-8"))

    full_mirror_match = matched_by_id["toolu_BBB"]
    assert full_mirror_match.was_truncated is False
    assert full_mirror_match.read_text() == "The file /x.py has been updated successfully."

    # The hook-*.txt capture is a distinct, already-tracked mechanism (raw hook
    # stdout, not tool_result content) -- never surfaced as debt.
    debt_filenames = {debt.filename for debt in result.debt}
    assert debt_filenames == {"orphan123.txt"}
    assert result.debt[0].reason == "no_owning_tool_result_block"


def test_join_tool_result_sidecars_returns_empty_result_when_dir_absent(tmp_path: Path) -> None:
    result = join_tool_result_sidecars([], _dir_scope(tmp_path / "does-not-exist"))
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

    join_result = join_tool_result_sidecars(payload, _dir_scope(tool_results_dir))

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


def test_sidecar_changed_after_join_becomes_typed_debt_without_replacing_inline_text() -> None:
    payload = [_record("m-race", "toolu_RACE", "preview")]
    reads = 0

    def read_once_then_disappear() -> str:
        nonlocal reads
        reads += 1
        if reads == 1:
            return "complete output"
        raise FileNotFoundError("sidecar disappeared after the join")

    scope = RetainedSidecarScope(
        scope_key="changed-after-join",
        files=(
            RetainedSidecarFile(
                filename="toolu_RACE.txt",
                byte_size=len("complete output"),
                file_mtime_ms=1_719_878_400_000,
                read_text=read_once_then_disappear,
            ),
        ),
        available=True,
    )
    joined = join_tool_result_sidecars(payload, scope)
    assert len(joined.matched) == 1

    parsed = apply_tool_result_sidecars(parse_code(payload, "fallback-race"), joined)
    [block] = [
        block
        for message in parsed.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT and block.tool_id == "toolu_RACE"
    ]
    assert block.text == "preview"
    [event] = [event for event in parsed.session_events if event.event_type == "claude_tool_result_sidecar"]
    assert event.payload["acquisition_status"] == "debt"
    assert event.payload["reason"] == "read_error:FileNotFoundError"


def test_sidecar_dataclasses_are_frozen() -> None:
    match = SidecarMatch(
        tool_use_id="toolu_X",
        filename="toolu_X.txt",
        byte_size=1,
        content_hash="0" * 64,
        was_truncated=True,
        read_text=lambda: "x",
    )
    debt = SidecarDebt(filename="orphan.txt", byte_size=1, reason="no_owning_tool_result_block")
    assert match.tool_use_id == "toolu_X"
    assert debt.reason == "no_owning_tool_result_block"


def test_apply_tool_result_sidecars_sets_event_timestamp_from_file_mtime(tmp_path: Path) -> None:
    """occurred_at_ms must come from the sidecar file's own mtime, not stay NULL.

    Before this fix ``ParsedSessionEvent.timestamp`` was never set for sidecar
    events, so every ``claude_tool_result_sidecar`` event landed with
    ``occurred_at_ms IS NULL`` in the archive -- with no timestamp at all,
    nobody could tell a closed historical debt cohort from one still actively
    accruing. Setting a specific, verifiable mtime and asserting the derived
    ISO timestamp round-trips to it is the load-bearing check here.
    """
    tool_results_dir = tmp_path / "tool-results"
    _write_sidecar(tool_results_dir, "toolu_AAA.txt", "small full text")
    _write_sidecar(tool_results_dir, "orphan123.txt", "no owning tool_result block references this file")

    fixed_epoch_s = 1719878400  # 2024-07-02T00:00:00Z, arbitrary but exact
    os.utime(tool_results_dir / "toolu_AAA.txt", (fixed_epoch_s, fixed_epoch_s))
    os.utime(tool_results_dir / "orphan123.txt", (fixed_epoch_s, fixed_epoch_s))

    payload = [_record("m-aaa", "toolu_AAA", "small full text")]
    join_result = join_tool_result_sidecars(payload, _dir_scope(tool_results_dir))
    acquired = parse_code(payload, "fallback-sidecar", tool_result_sidecars=join_result)

    sidecar_events = [event for event in acquired.session_events if event.event_type == "claude_tool_result_sidecar"]
    assert len(sidecar_events) == 2
    for event in sidecar_events:
        assert event.timestamp is not None
        assert event.timestamp.startswith("2024-07-02T00:00:00")


def _write_transcript(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_resolve_sibling_transcript_paths_finds_parent_and_subagents(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    session_dir = project_dir / "sess-1"
    parent_path = project_dir / "sess-1.jsonl"
    subagent_a = session_dir / "subagents" / "agent-a.jsonl"
    subagent_b = session_dir / "subagents" / "agent-b.jsonl"
    _write_transcript(parent_path, [{"type": "summary"}])
    _write_transcript(subagent_a, [{"type": "summary"}])
    _write_transcript(subagent_b, [{"type": "summary"}])

    from_root = resolve_sibling_transcript_paths(parent_path)
    assert set(from_root) == {subagent_a, subagent_b}

    from_subagent_a = resolve_sibling_transcript_paths(subagent_a)
    assert set(from_subagent_a) == {parent_path, subagent_b}


def test_session_scoped_join_resolves_sibling_owned_file_without_duplicating_debt(tmp_path: Path) -> None:
    """The scope bug this fixes: a naive per-transcript join over-counts debt by the subagent fanout.

    ``toolu_SUBAGENT`` is a tool call issued *by* the subagent -- its
    tool_result block, and therefore its owning index entry, lives only in
    the subagent's own transcript, never the parent's. A single-transcript
    join of the parent against the shared tool-results dir would misclassify
    that file as ``no_owning_tool_result_block`` even though it's genuinely
    owned (verified against 3 live multi-subagent sessions, see the module
    docstring). The session-scoped join must resolve it via the union index,
    attribute the match to the subagent (the transcript whose own payload
    actually owns the id), and emit nothing at all from the parent's pass for
    that file -- not a duplicate debt entry, not a duplicate match.
    """
    project_dir = tmp_path / "project"
    session_dir = project_dir / "sess-1"
    parent_path = project_dir / "sess-1.jsonl"
    subagent_path = session_dir / "subagents" / "agent-a.jsonl"
    tool_results_dir = session_dir / "tool-results"

    _write_sidecar(tool_results_dir, "toolu_PARENT.txt", "owned by the parent transcript")
    _write_sidecar(tool_results_dir, "toolu_SUBAGENT.txt", "owned by the subagent transcript")
    _write_sidecar(tool_results_dir, "orphan999.txt", "owned by nobody, anywhere in the session")

    parent_payload = [_record("m-parent", "toolu_PARENT", "owned by the parent transcript")]
    subagent_payload = [_record("m-sub", "toolu_SUBAGENT", "owned by the subagent transcript")]
    _write_transcript(parent_path, parent_payload)
    _write_transcript(subagent_path, subagent_payload)

    resolver = FilesystemSidecarResolver()
    parent_result = join_tool_result_sidecars_session_scoped(
        parent_payload, resolver.claude_code_scope(parent_path), parent_path
    )
    subagent_result = join_tool_result_sidecars_session_scoped(
        subagent_payload, resolver.claude_code_scope(subagent_path), subagent_path
    )

    parent_matched_ids = {match.tool_use_id for match in parent_result.matched}
    subagent_matched_ids = {match.tool_use_id for match in subagent_result.matched}

    # Each transcript matches only what it owns -- no duplicate match for the
    # sibling's file, no misattributed match either.
    assert parent_matched_ids == {"toolu_PARENT"}
    assert subagent_matched_ids == {"toolu_SUBAGENT"}

    # The parent's pass is the sole source of debt for the shared directory;
    # the subagent's pass never reports debt for files it doesn't own.
    parent_debt_filenames = {debt.filename for debt in parent_result.debt}
    assert parent_debt_filenames == {"orphan999.txt"}
    assert subagent_result.debt == ()


def test_session_scoped_join_never_emits_debt_for_subagent_meta_companion_files(tmp_path: Path) -> None:
    """A ``subagents/agent-*.meta.json`` companion path must never originate debt either.

    Discovered live: Claude Code's ``agent-*.meta.json`` subagent metadata
    sidecar (a distinct capture surface, ``artifact_taxonomy.AGENT_SIDECAR_META``)
    also gets ingested as its own quasi-session, using the SAME shared
    ``tool-results/`` directory as its ``.jsonl`` sibling -- and, before this
    fix, would independently re-enumerate and re-report the whole directory as
    debt a second time per subagent, on top of the ``.jsonl`` fanout. It lives
    under ``subagents/`` exactly like an ``agent-*.jsonl`` transcript, so the
    same root/non-root path-shape check that fixes the ``.jsonl`` fanout
    covers it for free: it carries no ``tool_result`` blocks of its own, so it
    matches nothing and -- the property this test locks in -- reports no debt.
    """
    session_dir = tmp_path / "project" / "sess-1"
    meta_path = session_dir / "subagents" / "agent-a.meta.json"
    tool_results_dir = session_dir / "tool-results"
    _write_sidecar(tool_results_dir, "toolu_OWNED_ELSEWHERE.txt", "owned by a sibling, not this meta file")

    result = join_tool_result_sidecars_session_scoped(
        [], FilesystemSidecarResolver().claude_code_scope(meta_path), meta_path
    )

    assert result.matched == ()
    assert result.debt == ()


def test_tool_results_sidecar_never_becomes_a_session_on_either_chokepoint(tmp_path: Path) -> None:
    """polylogue-b508: a ``tool-results/`` file is raw-only, whatever it contains.

    A tool call's own output can reproduce a genuine session-document shape --
    a messages list, even an ``id`` field shaped like the ``toolu_*`` id the
    sidecar is named for. Content heuristics alone therefore cannot refuse this
    family; the ``tool_result_sidecar`` path rule is the gate, and both parse
    chokepoints must honour it: the discovery/acquisition walk that the one-shot
    importer and the live watcher share, and the offline replay engine that
    rebuilds from retained raws.

    Anti-vacuity: widening ``tool_result_sidecar``'s ``path_pattern`` so it no
    longer matches, or relaxing its ``parse_policy`` from ``raw-only``, admits
    a session whose ``provider_session_id`` is the ``toolu_*`` fragment id --
    the phantom shape this law forbids.
    """
    body = json.dumps(
        {
            "id": "toolu_01ABCDEFGHIJKLMNOPQRSTUV",
            "messages": [
                {"role": "user", "content": "tool output that happens to look like a chat"},
                {"role": "assistant", "content": "reply"},
            ],
        }
    ).encode("utf-8")

    sidecar = (
        tmp_path / ".claude" / "projects" / "proj" / "sess" / "tool-results" / "toolu_01ABCDEFGHIJKLMNOPQRSTUV.json"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(body)

    rule = artifact_rule_for_path(Provider.CLAUDE_CODE, str(sidecar))
    assert rule is not None
    assert (rule.kind, rule.parse_policy) == ("tool_result_sidecar", "raw-only")

    replayed = _parse_one(Provider.CLAUDE_CODE, body, str(sidecar), sidecar_resolver=None)
    assert replayed == []

    acquired = list(iter_source_sessions_with_raw(Source(name="claude-code", path=sidecar), capture_raw=False))
    assert [session.provider_session_id for _raw, session in acquired] == []


def test_sidecar_event_time_stays_unknown_when_the_file_carries_no_mtime_evidence() -> None:
    """polylogue-x1gd: absent time evidence stays typed unknown, never invented.

    The sidecar file's own mtime is the only timestamp evidence this join has --
    sidecars carry no embedded time, and for genuine debt the owning
    ``tool_result`` block is by definition unresolvable. When ``stat`` cannot
    supply it (``file_mtime_ms`` stays ``None``), the emitted
    ``claude_tool_result_sidecar`` event must carry no timestamp, leaving
    ``occurred_at_ms`` NULL, rather than an ingestion-time stamp that would
    read downstream as "this sidecar was written at import".

    Anti-vacuity: defaulting the missing-mtime branch to the current clock or
    to the transcript's own timestamp makes both ``event.timestamp`` values
    non-None and turns this red.
    """
    from polylogue.sources.live.tool_result_sidecars import SidecarJoinResult

    payload = [_record("m-aaa", "toolu_AAA", "small full text")]
    join_result = SidecarJoinResult(
        matched=(
            SidecarMatch(
                tool_use_id="toolu_AAA",
                filename="toolu_AAA.txt",
                byte_size=15,
                content_hash="0" * 64,
                was_truncated=False,
                read_text=lambda: "small full text",
            ),
        ),
        debt=(SidecarDebt(filename="orphan123.txt", byte_size=7, reason="no_owning_tool_result_block"),),
    )

    acquired = parse_code(payload, "fallback-sidecar", tool_result_sidecars=join_result)

    sidecar_events = [event for event in acquired.session_events if event.event_type == "claude_tool_result_sidecar"]
    assert len(sidecar_events) == 2
    assert [event.timestamp for event in sidecar_events] == [None, None]


def test_session_scoped_join_reads_neither_sibling_owned_nor_unstorable_sidecars(tmp_path: Path) -> None:
    """polylogue-9k62p: filter first; refuse only what SQLite cannot store.

    The parent's pass must not read a sidecar a sibling owns (it discarded the
    text afterwards anyway) and must not read one whose retained size exceeds
    SQLite's physical value limit, which no block text can hold -- that file
    becomes typed ``value_bound_refused`` debt instead of an allocation.
    ``read_text`` raises here, so any read of either file fails the test.
    Anti-vacuity: dropping the ownership skip, or the physical-limit gate,
    makes the corresponding read fire and the test red.
    """
    project_dir = tmp_path / "project"
    session_dir = project_dir / "sess-bounded"
    parent_path = project_dir / "sess-bounded.jsonl"
    subagent_path = session_dir / "subagents" / "agent-a.jsonl"

    parent_payload = [_record("m-parent", "toolu_PARENT_BIG", "inline preview")]
    subagent_payload = [_record("m-sub", "toolu_SUBAGENT", "inline preview")]

    def _forbidden_read() -> str:
        raise AssertionError("sidecar contents were read before the ownership/size filter")

    scope = RetainedSidecarScope(
        scope_key=str(session_dir / "tool-results"),
        files=(
            RetainedSidecarFile(
                filename="toolu_SUBAGENT.txt",
                byte_size=16,
                file_mtime_ms=None,
                read_text=_forbidden_read,
            ),
            RetainedSidecarFile(
                filename="toolu_PARENT_BIG.txt",
                byte_size=value_bounds.MAX_STORABLE_VALUE_BYTES + 1,
                file_mtime_ms=None,
                read_text=_forbidden_read,
            ),
        ),
        siblings=(SiblingTranscript(coordinate=str(subagent_path), open_records=lambda: iter(subagent_payload)),),
        available=True,
    )
    _write_transcript(parent_path, parent_payload)
    _write_transcript(subagent_path, subagent_payload)

    result = join_tool_result_sidecars_session_scoped(parent_payload, scope, parent_path)

    assert result.matched == ()
    assert {(debt.filename, debt.reason) for debt in result.debt} == {("toolu_PARENT_BIG.txt", "value_bound_refused")}


#: One synthetic sidecar's size: above the former 64 MiB per-file refusal, and
#: four of them exceed the former 256 MiB per-transcript refusal.
_LARGE_SIDECAR_BYTES = 65 * 1024 * 1024
_LARGE_SIDECAR_COUNT = 4


def test_large_sidecars_join_completely_with_memory_bounded_by_one_file() -> None:
    """Sidecars of any size and number are joined, one file in memory at a time.

    Four 65 MiB sidecars used to become ``size_exceeded`` debt (64 MiB per
    file, 256 MiB per transcript) and never reach the index. Each is now
    matched with the hash of its full text, and the join's traced peak stays
    near one file's size although the files together are four times that.

    Anti-vacuity: restore either cap and the matches become debt; keep each
    file's text on its match (the former ``full_text``) and the traced peak
    grows with the total instead of staying under two files' worth.
    """
    import tracemalloc

    from polylogue.core.hashing import hash_text

    ids = [f"toolu_LARGE{index}" for index in range(_LARGE_SIDECAR_COUNT)]

    def generated(index: int) -> str:
        return chr(ord("a") + index) * _LARGE_SIDECAR_BYTES

    def reader(index: int):  # type: ignore[no-untyped-def]
        return lambda: generated(index)

    scope = RetainedSidecarScope(
        scope_key="large-sidecars",
        files=tuple(
            RetainedSidecarFile(
                filename=f"{tool_use_id}.txt",
                byte_size=_LARGE_SIDECAR_BYTES,
                file_mtime_ms=None,
                read_text=reader(index),
            )
            for index, tool_use_id in enumerate(ids)
        ),
        available=True,
    )
    payload = [_record(f"m-{tool_use_id}", tool_use_id, "inline preview") for tool_use_id in ids]

    tracemalloc.start()
    try:
        result = join_tool_result_sidecars(payload, scope)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.debt == ()
    assert [match.tool_use_id for match in result.matched] == ids
    assert all(match.was_truncated for match in result.matched)
    total = _LARGE_SIDECAR_BYTES * _LARGE_SIDECAR_COUNT
    assert peak < 2 * _LARGE_SIDECAR_BYTES + 16 * 1024 * 1024 < total, peak
    for match in result.matched:
        text = match.read_text()
        assert len(text) == _LARGE_SIDECAR_BYTES
        assert match.content_hash == hash_text(text)
        del text


def test_large_sidecar_replaces_its_truncated_block_in_full() -> None:
    """The applied session carries the whole large sidecar, and its event says so.

    Anti-vacuity: refuse a file above 64 MiB again and the block keeps its
    inline preview while the event reports debt.
    """
    tool_use_id = "toolu_LARGE_APPLY"
    text = "z" * _LARGE_SIDECAR_BYTES
    scope = RetainedSidecarScope(
        scope_key="large-sidecar-apply",
        files=(
            RetainedSidecarFile(
                filename=f"{tool_use_id}.txt",
                byte_size=_LARGE_SIDECAR_BYTES,
                file_mtime_ms=None,
                read_text=lambda: text,
            ),
        ),
        available=True,
    )
    payload = [_record("m-large", tool_use_id, "inline preview")]

    session = parse_code(payload, "fallback-large", tool_result_sidecars=join_tool_result_sidecars(payload, scope))

    [block] = [
        block
        for message in session.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT and block.tool_id == tool_use_id
    ]
    assert block.text is not None and len(block.text) == _LARGE_SIDECAR_BYTES
    [event] = [event for event in session.session_events if event.event_type == "claude_tool_result_sidecar"]
    assert event.payload["acquisition_status"] == "matched"
    assert event.payload["content_replaced"] is True
