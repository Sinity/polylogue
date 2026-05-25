"""MK3 reader paste-spans rendering + paste browser MVP (#1201).

Pins the daemon web shell's contract for the paste-spans slice:

- per-message paste indication driven by ``has_paste`` (whole-message
  fallback) and by ``paste_spans`` (per-span highlight + diff fold);
- copy menu gains ``copy-typed`` and ``copy-paste`` actions, disabled
  when no paste spans are present;
- ``/p`` paste-browser route serves a standalone page that lists
  paste-flagged messages grouped by conversation, with anchor links
  back to the source message;
- ``/api/paste-browser`` returns the envelope shape the page consumes,
  including empty-state behaviour against a fresh archive.

Evidence is shape-based — we assert the literal class names, CSS
selectors, function names, and DOM targets land in the served HTML so
the contract surface is browserless-verifiable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.visual.conftest import (
    ReaderWorkspace,
    archive_db_path,
    assert_no_private_paths,
    get_json,
    get_text,
    parse_dom,
    running_reader_server,
    write_evidence_manifest,
)


def _seed_diff_paste_message(workspace: ReaderWorkspace) -> None:
    """Add a synthetic message containing a unified-diff paste to the
    archive seeded by ``running_reader_server``."""

    db = archive_db_path(workspace)
    conn = sqlite3.connect(str(db))
    diff_text = "Before the diff:\n@@ -1,3 +1,4 @@\n context\n-old\n+new\n+added\n"
    conn.execute(
        """
        INSERT INTO messages(
            message_id, conversation_id, role, text, sort_key, source_name,
            content_hash, version, word_count, has_tool_use, has_thinking,
            has_paste, message_type
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "reader-c3-diff",
            "reader-c3",
            "user",
            diff_text,
            1.0,
            "claude-ai",
            "hash-reader-c3-diff",
            1,
            len(diff_text.split()),
            0,
            0,
            1,
            "message",
        ),
    )
    conn.commit()
    conn.close()


def test_reader_paste_spans_contract(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        _seed_diff_paste_message(reader_workspace)
        status, content_type, body = get_text(base_url, "/")
        # Conversation envelope must carry ``paste_spans``.
        conv_payload = get_json(base_url, "/api/conversations/reader-c3")
        assert isinstance(conv_payload, dict)
        messages = conv_payload["messages"]
        # The seeded diff-paste message is present with non-empty spans;
        # the original "synthetic paste-like block" message has has_paste
        # set but no detectable diff, so spans are empty (banner path).
        diff_msg = next(m for m in messages if m["id"] == "reader-c3-diff")
        plain_paste_msg = next(m for m in messages if m["id"] == "reader-c3-m1")
        # Paste browser page + envelope.
        p_status, p_ctype, p_body = get_text(base_url, "/p")
        payload = get_json(base_url, "/api/paste-browser?limit=100")

    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(body, context="reader shell HTML")

    assert diff_msg["has_paste"] is True
    assert isinstance(diff_msg["paste_spans"], list)
    assert len(diff_msg["paste_spans"]) == 1
    assert diff_msg["paste_spans"][0]["kind"] == "diff"
    assert plain_paste_msg["has_paste"] is True
    # Whole-message fallback: no per-span data for the prose paste.
    assert plain_paste_msg["paste_spans"] == []

    dom = parse_dom(body)
    for required in ("msg-list", "help-overlay"):
        assert required in dom.ids, f"missing required id {required!r}"

    # --- Paste-span renderer wiring -------------------------------------
    for phrase in (
        "_polyRenderPasteBody",
        "_polyDetectDiffSpans",
        "_polyEffectivePasteSpans",
        "_polyPasteBannerHtml",
        "_polyHasPaste",
        "_polyDiffFoldHtml",
        "_polyClassifyDiffLine",
        ".msg-paste-banner",
        ".msg-paste-span",
        ".msg-diff-fold",
        ".diff-line.add",
        ".diff-line.del",
        ".diff-line.hunk",
        ".diff-line.file",
    ):
        assert phrase in body, f"paste-span renderer missing {phrase!r}"

    # --- Copy menu actions ---------------------------------------------
    for phrase in (
        'data-act="copy-typed"',
        'data-act="copy-paste"',
        "copyTypedOnly",
        "copyPasteOnly",
        "_polyTypedOnlyText",
        "_polyPasteOnlyText",
        # Disabled-reason token for the disabled-when-no-spans path.
        'data-disabled-reason="no_paste_spans"',
    ):
        assert phrase in body, f"copy menu missing {phrase!r}"

    # --- /p paste browser page -----------------------------------------
    assert p_status == 200
    assert "text/html" in p_ctype
    assert_no_private_paths(p_body, context="paste-browser HTML")
    for phrase in (
        'id="paste-list"',
        'id="paste-empty"',
        "initPasteBrowser",
        "/api/paste-browser",
        "_polyPasteBrowserRender",
    ):
        assert phrase in p_body, f"paste-browser page missing {phrase!r}"

    # --- /api/paste-browser populated envelope -------------------------
    assert isinstance(payload, dict)
    items = payload["items"]
    assert isinstance(items, list)
    # Two paste-flagged messages in the seeded archive (one diff, one
    # whole-message prose paste).
    ids = {item["message_id"] for item in items}
    assert "reader-c3-diff" in ids
    assert "reader-c3-m1" in ids
    diff_entry = next(item for item in items if item["message_id"] == "reader-c3-diff")
    assert diff_entry["has_diff"] is True
    assert diff_entry["conversation_id"] == "reader-c3"
    assert diff_entry["message_anchor"].startswith("message-")
    assert any(span["kind"] == "diff" for span in diff_entry["paste_spans"])

    write_evidence_manifest(
        tmp_path / "reader-paste-spans-evidence.json",
        artifact_id="polylogue.local_reader.paste_spans",
        route="/p",
        fixture_id="reader-visual-synthetic-v1+diff",
        checks={
            "status": status,
            "content_type": content_type,
            "paste_spans_in_envelope": True,
            "whole_message_fallback": True,
            "copy_actions": ["copy-typed", "copy-paste"],
            "diff_fold_classes": ["msg-diff-fold", "diff-line add", "diff-line del"],
            "paste_browser_route_status": p_status,
            "paste_browser_items": len(items),
            "private_path_safe": True,
        },
    )


def test_paste_browser_empty_state(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    # Seed an archive without any has_paste rows by clearing the seeded
    # paste flag before listing. We use ``conversations=False`` so the
    # archive starts empty, then the paste-browser must return an empty
    # envelope and the page must still render its DOM targets.
    with running_reader_server(reader_workspace, conversations=False) as (_, base_url):
        payload = get_json(base_url, "/api/paste-browser")
        p_status, _ctype, p_body = get_text(base_url, "/p")

    assert isinstance(payload, dict)
    assert payload["items"] == []
    assert payload["total"] == 0
    assert p_status == 200
    assert 'id="paste-list"' in p_body
    assert 'id="paste-empty"' in p_body

    write_evidence_manifest(
        tmp_path / "reader-paste-browser-empty-evidence.json",
        artifact_id="polylogue.local_reader.paste_browser_empty",
        route="/api/paste-browser",
        fixture_id="reader-visual-empty-archive",
        checks={
            "items": 0,
            "total": 0,
            "page_status": p_status,
            "page_has_list_target": True,
            "page_has_empty_target": True,
        },
    )
