"""Parser tests for ``~/.claude/history.jsonl`` paste-evidence sidecar (#1583)."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.sources.parsers.claude.history import (
    HistoryEntry,
    HistoryPaste,
    build_session_paste_index,
    parse_history_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_parse_yields_no_rows_for_missing_path(tmp_path: Path) -> None:
    assert list(parse_history_jsonl(tmp_path / "nope.jsonl")) == []


def test_parse_handles_empty_pasted_contents(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    _write_jsonl(
        path,
        [
            {
                "display": "ordinary prompt",
                "pastedContents": {},
                "timestamp": 1762953171392,
                "project": "/realm/project",
                "sessionId": "abc",
            }
        ],
    )
    entries = list(parse_history_jsonl(path))
    assert entries == [
        HistoryEntry(
            display="ordinary prompt",
            timestamp_ms=1762953171392,
            project="/realm/project",
            session_id="abc",
            pastes=(),
        )
    ]
    assert entries[0].has_paste is False


def test_parse_extracts_full_content_paste(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    _write_jsonl(
        path,
        [
            {
                "display": "[Pasted text #1 +6 lines]",
                "pastedContents": {
                    "1": {"id": 1, "type": "text", "content": "alpha\nbravo"},
                },
                "timestamp": 1762953171392,
                "project": "/realm/project",
                "sessionId": "abc",
            }
        ],
    )
    entries = list(parse_history_jsonl(path))
    assert len(entries) == 1
    entry = entries[0]
    assert entry.has_paste is True
    assert entry.pastes == (HistoryPaste(paste_id="1", paste_type="text", content="alpha\nbravo", has_content=True),)
    assert entry.pastes[0].is_hash_only is False


def test_parse_marks_hash_only_paste_as_existing_but_unrecoverable(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    _write_jsonl(
        path,
        [
            {
                "display": "[Pasted text #1]",
                "pastedContents": {"1": {"id": 1, "type": "text"}},
                "timestamp": 1762953171000,
                "sessionId": "abc",
            }
        ],
    )
    entry = next(parse_history_jsonl(path))
    assert entry.has_paste is True
    assert entry.pastes[0].is_hash_only is True
    assert entry.pastes[0].content == ""


def test_parse_skips_malformed_lines_without_aborting(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        "\n".join(
            [
                "not json at all",
                '{"display": "valid", "pastedContents": {}, "timestamp": 1}',
                '{"display": "missing_brace"',
                '{"display": "also valid", "pastedContents": {}, "sessionId": "xyz"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    entries = list(parse_history_jsonl(path))
    assert [entry.display for entry in entries] == ["valid", "also valid"]


def test_parse_skips_non_dict_rows(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text('"just a string"\n[1, 2, 3]\n{"display": "ok"}\n', encoding="utf-8")
    entries = list(parse_history_jsonl(path))
    assert [entry.display for entry in entries] == ["ok"]


def test_parse_handles_missing_optional_fields(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    _write_jsonl(path, [{"display": "no metadata"}])
    entry = next(parse_history_jsonl(path))
    assert entry == HistoryEntry(
        display="no metadata",
        timestamp_ms=None,
        project=None,
        session_id=None,
        pastes=(),
    )


def test_build_session_paste_index_only_includes_paste_rows_with_session_id(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    _write_jsonl(
        path,
        [
            # No sessionId — dropped.
            {
                "display": "paste",
                "pastedContents": {"1": {"id": 1, "type": "text", "content": "x"}},
                "timestamp": 1,
            },
            # No pastes — dropped.
            {
                "display": "no paste",
                "pastedContents": {},
                "timestamp": 2,
                "sessionId": "sess-a",
            },
            # Paste + sessionId — kept.
            {
                "display": "[Pasted text #1]",
                "pastedContents": {"1": {"id": 1, "type": "text", "content": "y"}},
                "timestamp": 3,
                "sessionId": "sess-a",
            },
            # Paste + different sessionId — separate bucket.
            {
                "display": "[Pasted text #1]",
                "pastedContents": {"1": {"id": 1, "type": "text", "content": "z"}},
                "timestamp": 4,
                "sessionId": "sess-b",
            },
        ],
    )
    index = build_session_paste_index(path)
    assert set(index.keys()) == {"sess-a", "sess-b"}
    assert [entry.timestamp_ms for entry in index["sess-a"]] == [3]
    assert [entry.timestamp_ms for entry in index["sess-b"]] == [4]


def test_build_session_paste_index_for_missing_path_returns_empty(tmp_path: Path) -> None:
    assert build_session_paste_index(tmp_path / "absent.jsonl") == {}


def test_parse_handles_non_utf8_file_gracefully(tmp_path: Path) -> None:
    """A single bad-byte file must not abort enrichment for any session.

    Production-safety contract: ``parse_history_jsonl`` returns no rows
    when the file cannot be decoded as UTF-8 (logged at debug). Asserts the
    swallow pattern matches the rest of the parser's malformed-line policy.
    """
    path = tmp_path / "history.jsonl"
    path.write_bytes(b"\x80\x81\x82 not utf8")
    assert list(parse_history_jsonl(path)) == []
