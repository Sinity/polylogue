"""Original-file scanning behavior retained when ownership moved to Polylogue."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.operations.raw_sessions.page import CompactJSONPage
from polylogue.operations.raw_sessions.sessions import SessionError, SessionLogService, SessionSource


def session_service(tmp_path: Path) -> tuple[SessionLogService, Path]:
    root = tmp_path / "claude"
    root.mkdir()
    service = SessionLogService(
        sources=(SessionSource("claude-code", root),),
    )
    return service, root


def test_default_codex_source_is_the_canonical_sessions_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_root = home / ".codex" / "sessions"
    session_root.mkdir(parents=True)
    (session_root / "session.jsonl").write_text('{"text":"session"}\n')
    (home / ".codex" / "history.jsonl").write_text('{"text":"history"}\n')
    sources = SessionLogService.default_sources(home)
    service = SessionLogService(
        sources=sources,
    )

    listed = service.list("codex")

    assert sources[1] == SessionSource("codex", session_root)
    assert [entry["reference"] for entry in listed["sessions"]] == ["codex:session.jsonl"]


def test_session_list_read_and_search_preserve_provider_reference(
    tmp_path: Path,
) -> None:
    service, root = session_service(tmp_path)
    session = root / "project" / "session.jsonl"
    session.parent.mkdir()
    session.write_text('{"text":"gateway demonstration"}\n{"text":"second"}\n')

    listed = service.list("claude-code")
    reference = listed["sessions"][0]["reference"]
    read = service.read(reference, max_bytes=32)
    search = service.search("claude-code", "demonstration", max_results=1)

    assert reference == "claude-code:project/session.jsonl"
    assert read["reference"] == reference
    assert "gateway" in read["content"]
    assert search["matches"][0]["reference"] == reference
    assert search["truncated"] is False


def test_session_references_survive_a_provider_root_symlink(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical-claude"
    canonical_root.mkdir()
    alias_root = tmp_path / "claude"
    alias_root.symlink_to(canonical_root, target_is_directory=True)
    session = canonical_root / "project" / "session.jsonl"
    session.parent.mkdir()
    session.write_text('{"text":"gateway demonstration"}\n')
    service = SessionLogService(
        sources=(SessionSource("claude-code", alias_root),),
    )

    reference = service.list("claude-code")["sessions"][0]["reference"]

    assert reference == "claude-code:project/session.jsonl"
    assert service.read(reference, max_bytes=1)["reference"] == reference


def test_session_reference_rejects_path_escape(tmp_path: Path) -> None:
    service, root = session_service(tmp_path)
    (root / "session.jsonl").write_text("{}\n")

    with pytest.raises(SessionError, match="remain within"):
        service.read("claude-code:../session.jsonl")


def test_session_search_covers_a_long_line_beyond_the_old_prefix(
    tmp_path: Path,
) -> None:
    service, root = session_service(tmp_path)
    (root / "session.jsonl").write_text("x" * 70_000 + "needle\n")

    result = service.search("claude-code", "needle")

    assert result["matches"][0]["offset"] == 69_800
    assert "needle" in result["matches"][0]["text"]
    assert result["truncated"] is False


def test_session_search_continues_across_chunk_and_unicode_boundaries(
    tmp_path: Path,
) -> None:
    service, root = session_service(tmp_path)
    session = root / "large.jsonl"
    session.write_bytes(b"x" * (64 * 1_024 - 1) + "éneedle\n".encode())
    key = b"s" * 32

    first = service.search("claude-code", "éneedle", scan_bytes=64 * 1_024, cursor_key=key)
    assert first["matches"] == [] and first["next_cursor"]
    second = service.search(
        "claude-code",
        "éneedle",
        scan_bytes=64 * 1_024,
        cursor=first["next_cursor"],
        cursor_key=key,
    )

    match = second["matches"][0]
    assert match["line"] == 1 and "éneedle" in match["text"]
    assert "éneedle" in service.read(match["reference"], match["offset"], 512)["content"]
    assert second["next_cursor"] is None


def test_session_search_continuation_rejects_a_changed_source(tmp_path: Path) -> None:
    service, root = session_service(tmp_path)
    session = root / "large.jsonl"
    session.write_text("x" * (64 * 1_024) + "needle\n")
    key = b"s" * 32
    first = service.search("claude-code", "needle", scan_bytes=64 * 1_024, cursor_key=key)
    session.write_text(session.read_text() + "changed\n")

    with pytest.raises(SessionError, match="source changed"):
        service.search("claude-code", "needle", cursor=first["next_cursor"], cursor_key=key)


def test_newest_sessions_are_selected_across_the_entire_tree(tmp_path: Path) -> None:
    service, root = session_service(tmp_path)
    for name, modified in (("old/a", 1), ("old/b", 2), ("new/c", 9)):
        path = root / f"{name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}\n")
        os.utime(path, ns=(modified, modified))

    result = service.list("claude-code", limit=1)

    assert result["sessions"][0]["reference"] == "claude-code:new/c.jsonl"
    assert result["truncated"] is True


def test_timeline_filters_before_discovery_limit_and_reports_exact_end(
    tmp_path: Path,
) -> None:
    service, root = session_service(tmp_path)
    for index in range(1_002):
        path = root / f"{index:04}.jsonl"
        path.write_text("{}\n")
        os.utime(path, ns=(index + 1, index + 1))

    result = service.timeline("claude-code", 1, 1, None, 1)

    assert [row["reference"] for row in result["entries"]] == ["claude-code:0000.jsonl"]
    assert result["truncated"] is False


def test_session_discovery_cannot_search_symlinked_external_content(
    tmp_path: Path,
) -> None:
    service, root = session_service(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"text":"external needle"}\n')
    (root / "linked.jsonl").symlink_to(outside)

    assert service.list("claude-code")["sessions"] == []
    assert service.search("claude-code", "needle")["matches"] == []


def test_search_hit_contains_query_and_a_readable_byte_offset(tmp_path: Path) -> None:
    service, root = session_service(tmp_path)
    path = root / "long.jsonl"
    path.write_text('{"text":"' + "é" * 3_000 + 'needle"}\n')

    match = service.search("claude-code", "needle")["matches"][0]

    assert "needle" in match["text"]
    assert "needle" in service.read(match["reference"], match["offset"], 2_000)["content"]


def test_missing_session_source_is_unavailable(tmp_path: Path) -> None:
    service, root = session_service(tmp_path)
    root.rmdir()
    with pytest.raises(SessionError, match="unavailable"):
        service.list("claude-code")


def test_read_preserves_raw_offsets_and_replaces_malformed_utf8(tmp_path: Path) -> None:
    service, root = session_service(tmp_path)
    (root / "bytes.jsonl").write_bytes("é".encode() + b"\xff\xe2\x82")

    read = service.read("claude-code:bytes.jsonl", offset=1)

    assert read["offset"] == 1 and read["bytes"] == 4
    assert read["content"] == "\ufffd\ufffd\ufffd"
    assert read["next_offset"] is None


def test_file_timeline_continuation_resumes_after_the_consumed_file(tmp_path: Path) -> None:
    service, root = session_service(tmp_path)
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    first.write_text("needle\nneedle\n")
    second.write_text("needle\n")
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    key = b"timeline-key"

    page = service.timeline("claude-code", None, None, "needle", 1, cursor_key=key)
    next_page = service.timeline("claude-code", None, None, "needle", 1, cursor=page["next_cursor"], cursor_key=key)

    assert [entry["reference"] for entry in page["entries"]] == ["claude-code:second.jsonl"]
    assert [entry["reference"] for entry in next_page["entries"]] == ["claude-code:first.jsonl"]
    matches = service.search("claude-code", "needle", 10)["matches"]
    assert len(matches) == 3


def test_exact_reference_search_skips_unrelated_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, root = session_service(tmp_path)
    selected = root / "selected.jsonl"
    selected.write_text("needle\n")
    (root / "unrelated.jsonl").write_text("not selected\n")

    def inventory_must_not_run(source: SessionSource) -> list[tuple[Path, os.stat_result]]:
        raise AssertionError(f"unexpected inventory for {source.provider}")

    monkeypatch.setattr(service, "_files", inventory_must_not_run)

    result = service.search("claude-code", "needle", reference="claude-code:selected.jsonl")

    assert [row["reference"] for row in result["matches"]] == ["claude-code:selected.jsonl"]


@pytest.mark.parametrize(
    "rows",
    [
        [{"text": "plain"}, {"text": "é\\nquoted"}],
        [{"nested": [None, {"emoji": "☃"}]}],
    ],
)
def test_compact_json_page_matches_complete_list_size(rows: list[dict[str, object]]) -> None:
    page = CompactJSONPage(10_000)

    for row in rows:
        assert page.try_append(row)

    import json

    assert page.encoded_bytes == len(json.dumps(rows, separators=(",", ":")).encode())


def test_compact_json_page_rejects_without_mutating_state() -> None:
    row = {"text": "x" * 50}
    page = CompactJSONPage(2 + len('{"text":"x"}') + 1)

    assert page.try_append({"text": "x"})
    before = (list(page.items), page.encoded_bytes)
    assert not page.try_append(row)
    assert (page.items, page.encoded_bytes) == before
