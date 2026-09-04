"""Paste evidence remains readable through typed SSR and its API envelope."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from tests.visual.conftest import (
    READER_C3,
    READER_C3_DIFF,
    READER_C3_M1,
    ReaderWorkspace,
    assert_no_private_paths,
    get_json,
    get_text,
    running_reader_server,
    seed_reader_diff_paste,
    write_evidence_manifest,
)


def test_paste_evidence_is_linked_from_typed_routes(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        seed_reader_diff_paste(reader_workspace)
        session_status, session_type, session_page = get_text(base_url, f"/sessions/{READER_C3}")
        browser_status, browser_type, browser_page = get_text(base_url, "/p")
        session_payload = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C3}"))
        messages = cast(list[dict[str, object]], session_payload["messages"])
        payload = cast(dict[str, object], get_json(base_url, "/api/paste-browser?limit=100"))

    assert session_status == 200
    assert "text/html" in session_type
    assert f'id="msg-{READER_C3_DIFF}"' in session_page
    assert 'data-flag="paste"' in session_page
    assert browser_status == 200
    assert "text/html" in browser_type
    assert "<h1>Paste evidence</h1>" in browser_page
    assert_no_private_paths(session_page, context="paste session HTML")
    assert_no_private_paths(browser_page, context="paste browser HTML")

    diff_message = next(item for item in messages if item["id"] == READER_C3_DIFF)
    plain_message = next(item for item in messages if item["id"] == READER_C3_M1)
    assert diff_message["has_paste_evidence"] is True
    assert any(span["kind"] == "diff" for span in cast(list[dict[str, object]], diff_message["paste_spans"]))
    assert plain_message["has_paste_evidence"] is True
    assert plain_message["paste_spans"] == []
    items = cast(list[dict[str, object]], payload["items"])
    assert {item["message_id"] for item in items} >= {READER_C3_DIFF, READER_C3_M1}

    write_evidence_manifest(
        tmp_path / "typed-webui-paste-evidence.json",
        artifact_id="polylogue.webui.paste_evidence",
        route="/p",
        fixture_id="reader-visual-synthetic-v1+diff",
        checks={"session_status": session_status, "browser_status": browser_status, "items": len(items)},
    )


def test_paste_browser_has_an_honest_empty_state(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace, sessions=False) as (_, base_url):
        status, content_type, page = get_text(base_url, "/p")
        payload = get_json(base_url, "/api/paste-browser")

    assert status == 200
    assert "text/html" in content_type
    assert "<h1>Paste evidence</h1>" in page
    assert "No paste evidence is available in this archive." in page
    assert payload == {"items": [], "total": 0}

    write_evidence_manifest(
        tmp_path / "typed-webui-paste-empty-evidence.json",
        artifact_id="polylogue.webui.paste_empty",
        route="/p",
        fixture_id="reader-visual-empty-archive",
        checks={"status": status, "empty": True},
    )
