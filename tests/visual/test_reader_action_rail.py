"""MK3 reader message-card action rail + fold policies + shortcuts (#1202).

Pins the daemon web shell's contract for the new reader slice:

- per-message ``msg-actions`` rail with the five named actions
  (copy, copy-link, open-raw, view-provenance, jump-to-anchor);
- default fold policies for tool calls / tool output / thinking blocks
  via the ``msg-fold`` family and per-block code folds via
  ``msg-code-fold``;
- the documented keyboard shortcuts ``j``/``k``/``o``/``c``/``/``/``?``/``Esc``.

Evidence is shape-based — we assert the literal class names, ``data-act``
tokens, and helper function names land in the served HTML so that
downstream agents can rely on the contract without scraping rendered
output.
"""

from __future__ import annotations

from pathlib import Path

from tests.visual.conftest import (
    ReaderWorkspace,
    assert_no_private_paths,
    get_text,
    parse_dom,
    running_reader_server,
    write_evidence_manifest,
)


def test_reader_action_rail_contract(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, body = get_text(base_url, "/")

    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(body, context="reader shell HTML")

    dom = parse_dom(body)
    # All pre-existing IDs still present — the slice extends, doesn't
    # rearrange.
    for required in ("msg-list", "help-overlay", "help-panel"):
        assert required in dom.ids, f"missing required id {required!r}"

    # --- Action rail: per-message buttons -------------------------------
    # The rail itself plus each of the five named actions per #1202.
    for phrase in (
        '"msg-actions"',
        'data-act="copy-text"',
        'data-act="copy-link"',
        'data-act="open-raw"',
        'data-act="view-provenance"',
        'data-act="jump-anchor"',
        "copyMessageById",
        "copyMessageLink",
        "openProvenanceTab",
        "jumpToAnchor",
        "/api/sessions/' + encodeURIComponent(sessionId) + '/raw",
    ):
        assert phrase in body, f"action rail missing {phrase!r}"

    # --- Fold policies --------------------------------------------------
    # Tool calls / tool output / thinking blocks → ``msg-fold`` family,
    # code blocks → ``msg-code-fold`` per block.
    for phrase in (
        '"msg-fold"',
        '"msg-fold-body"',
        '"msg-code-fold"',
        "togglePolyFold",
        "toggleCodeFold",
        "_polyToolFoldHtml",
        "_polyThinkingFoldHtml",
        "_polySplitCodeBlocks",
        # Thinking summary marker per #1202 spec.
        "[thinking]",
    ):
        assert phrase in body, f"fold policy missing {phrase!r}"

    # --- Renderer wiring -------------------------------------------------
    # ``messageBlocksHtml`` must delegate to the slice so we don't
    # accidentally regress to the legacy single-blob renderer.
    assert "return renderMessageBlocks(messages);" in body

    # --- Keyboard shortcuts ---------------------------------------------
    # The full /j/k/o/c/?/Esc set is wired and the help overlay names
    # the new bindings.
    for phrase in (
        "installReaderShortcuts",
        "_polyHandleNavigateMessages",
        "_polyHandleCopyFocused",
        "_polyHandleOpenSession",
        "Open focused session",
        "Copy focused message text",
        "Focus search",
        "Next session",
        "Previous session",
        "Toggle this help",
    ):
        assert phrase in body, f"keyboard wiring missing {phrase!r}"

    # The base j/k handler now defers to the reader slice when a
    # session is loaded (prevents double-navigation).
    assert "if (convOpen) return;" in body

    write_evidence_manifest(
        tmp_path / "reader-action-rail-evidence.json",
        artifact_id="polylogue.local_reader.message_card",
        route="/",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "status": status,
            "content_type": content_type,
            "action_rail_present": True,
            "actions": [
                "copy-text",
                "copy-link",
                "open-raw",
                "view-provenance",
                "jump-anchor",
            ],
            "fold_classes_present": ["msg-fold", "msg-fold-body", "msg-code-fold"],
            "keyboard_shortcuts": ["j", "k", "o", "c", "/", "?", "Esc"],
            "renderer_delegates_to_slice": True,
            "private_path_safe": True,
        },
    )
