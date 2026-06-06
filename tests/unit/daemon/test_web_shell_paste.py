"""Unit tests for the reader paste-spans slice (#1201).

Covers:

- ``detect_paste_spans`` server-side diff detector — hunk shape,
  file-header-pair shape, prose rejection, and contiguous-region
  termination;
- ``envelope_paste_spans`` formatting helper;
- ``snippet_for_paste`` short-line extraction;
- ``build_paste_browser_payload`` envelope shape.
"""

from __future__ import annotations

from polylogue.daemon.web_shell_attachments import LibraryEntry, build_library_payload
from polylogue.daemon.web_shell_paste import (
    PasteBrowserEntry,
    build_paste_browser_payload,
    detect_paste_spans,
    envelope_paste_spans,
    render_paste_browser_page,
    snippet_for_paste,
)


def test_detect_paste_spans_unified_diff_hunk() -> None:
    text = (
        "Here is a diff:\n@@ -1,3 +1,4 @@\n context line\n-old line\n+new line\n+added line\n\nAnd some prose after.\n"
    )
    spans = detect_paste_spans(text)
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == "diff"
    assert span.confidence >= 0.9
    # Span starts at the hunk header, not the prose introduction.
    assert text[span.start : span.start + 2] == "@@"
    # Span includes the "+added line" line (last diff line).
    assert "added line" in text[span.start : span.end]


def test_detect_paste_spans_file_header_pair() -> None:
    text = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    spans = detect_paste_spans(text)
    assert len(spans) == 1
    assert spans[0].kind == "diff"
    # The file-header pair plus hunk all live in one span.
    body = text[spans[0].start : spans[0].end]
    assert body.startswith("--- ")
    assert "+++" in body
    assert "@@" in body


def test_detect_paste_spans_rejects_prose_with_dashes() -> None:
    # Em-dash separator + "+++" emphasis in prose must not be flagged.
    text = "This is a long --- sentence with +++ markers but no diff structure."
    assert detect_paste_spans(text) == []


def test_detect_paste_spans_terminates_on_prose() -> None:
    text = "@@ -1,2 +1,2 @@\n-old\n+new\n\nNow back to prose without diff structure.\nMore prose.\n"
    spans = detect_paste_spans(text)
    assert len(spans) == 1
    body = text[spans[0].start : spans[0].end]
    # Body must include the diff lines and stop before the prose.
    assert "Now back to prose" not in body
    assert "+new" in body


def test_envelope_paste_spans_returns_dicts() -> None:
    text = "@@ -1 +1 @@\n-a\n+b\n"
    spans = envelope_paste_spans(text, has_paste=True)
    assert len(spans) == 1
    assert spans[0]["kind"] == "diff"
    assert spans[0]["start"] == 0
    end_val = spans[0]["end"]
    assert isinstance(end_val, int) and end_val > 0
    assert "confidence" in spans[0]


def test_envelope_paste_spans_empty_when_no_diff() -> None:
    # Whole-message paste with no detectable diff yields empty spans —
    # the client falls back to a banner.
    assert envelope_paste_spans("plain pasted text", has_paste=True) == []
    assert envelope_paste_spans("", has_paste=True) == []
    assert envelope_paste_spans(None, has_paste=True) == []


def test_snippet_for_paste_uses_first_span_line() -> None:
    text = "intro line\n@@ -1 +1 @@\n-old\n+new\n"
    spans = envelope_paste_spans(text, has_paste=True)
    snippet = snippet_for_paste(text, spans)
    assert snippet.startswith("@@")


def test_snippet_for_paste_falls_back_to_first_line() -> None:
    text = "first line\nsecond line"
    snippet = snippet_for_paste(text, [])
    assert snippet == "first line"


def test_snippet_for_paste_truncates_long_lines() -> None:
    text = "x" * 500
    snippet = snippet_for_paste(text, [], limit=64)
    assert len(snippet) == 65  # 64 chars + ellipsis
    assert snippet.endswith("\u2026")


def test_build_paste_browser_payload_envelope_shape() -> None:
    entry = PasteBrowserEntry(
        session_id="c1",
        session_title="t1",
        origin="claude-ai-export",
        message_id="m1",
        message_anchor="message-m1",
        role="user",
        timestamp=None,
        word_count=5,
        snippet="@@ -1 +1 @@",
        paste_spans=[{"kind": "diff", "start": 0, "end": 10, "confidence": 0.95}],
        has_diff=True,
    )
    payload = build_paste_browser_payload([entry], total=1)
    assert payload["total"] == 1
    items = payload["items"]
    assert isinstance(items, list)
    assert items[0]["session_id"] == "c1"
    assert items[0]["origin"] == "claude-ai-export"
    assert "provider" not in items[0]
    assert items[0]["has_diff"] is True
    assert items[0]["message_anchor"] == "message-m1"
    assert items[0]["paste_spans"][0]["kind"] == "diff"


def test_build_attachment_library_payload_uses_origin() -> None:
    entry = LibraryEntry(
        envelope={
            "attachment_id": "att-1",
            "session_id": "c1",
            "message_id": "m1",
            "name": "note.txt",
            "mime_type": "text/plain",
            "size_bytes": 12,
            "path": None,
            "state": "available",
        },
        session_title="t1",
        origin="claude-ai-export",
        message_anchor="message-m1",
    )
    payload = build_library_payload([entry], total=1)
    items = payload["items"]
    assert isinstance(items, list)
    assert items[0]["origin"] == "claude-ai-export"
    assert "provider" not in items[0]


def test_render_paste_browser_page_inlines_js_and_dom_targets() -> None:
    html = render_paste_browser_page()
    # Bootstrap script invocation present.
    assert "initPasteBrowser();" in html
    # DOM targets that the JS expects.
    assert 'id="paste-list"' in html
    assert 'id="paste-empty"' in html
    # JS module pieces embedded inline.
    assert "_polyDetectDiffSpans" in html
    assert "/api/paste-browser" in html
