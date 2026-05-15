"""Daemon-served reader visual/DOM smoke evidence (#865)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from tests.visual.conftest import (
    ReaderWorkspace,
    assert_no_private_paths,
    get_json,
    get_text,
    parse_dom,
    running_reader_server,
    write_evidence_manifest,
)


def test_reader_search_shell_dom_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, body = get_text(base_url, "/")

    assert status == 200
    assert "text/html" in content_type
    assert len(body) > 20_000
    assert "https://cdn" not in body
    assert_no_private_paths(body, context="reader shell HTML")

    dom = parse_dom(body)
    expected_ids = {
        "app",
        "status-strip",
        "status-dot",
        "sidebar",
        "search",
        "facet-bar",
        "conv-list",
        "main",
        "conv-header",
        "msg-list",
        "inspector",
        "inspector-tabs",
        "footer",
        "help-overlay",
    }
    assert expected_ids <= dom.ids
    assert dom.meta_viewport is True
    assert dom.scripts == 1
    assert dom.styles == 1
    for phrase in ("Select a conversation", "Keyboard Shortcuts", "Focus search", "Local"):
        assert phrase in body

    checks = {
        "status": status,
        "content_type": content_type,
        "html_bytes": len(body.encode()),
        "required_ids": sorted(expected_ids),
        "script_tags": dom.scripts,
        "style_tags": dom.styles,
        "viewport_meta": dom.meta_viewport,
        "private_path_safe": True,
        "runtime_cdn_free": True,
    }
    manifest = write_evidence_manifest(
        tmp_path / "reader-search-dom-evidence.json",
        artifact_id="polylogue.local_reader.search",
        route="/",
        fixture_id="reader-visual-synthetic-v1",
        checks=checks,
    )
    assert manifest["evidence_kind"] == "browserless-dom"


def test_reader_conversation_deeplink_and_detail_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, shell = get_text(base_url, "/c/reader-c1")
        detail = cast(dict[str, object], get_json(base_url, "/api/conversations/reader-c1"))
        messages = cast(dict[str, object], get_json(base_url, "/api/conversations/reader-c1/messages"))
        raw = cast(dict[str, object], get_json(base_url, "/api/conversations/reader-c1/raw"))

    assert status == 200
    assert "text/html" in content_type
    assert "getConvIdFromURL" in shell
    assert_no_private_paths(shell, context="reader /c deeplink shell")
    assert_no_private_paths(json.dumps(detail), context="reader detail JSON")
    assert_no_private_paths(json.dumps(messages), context="reader messages JSON")

    target_ref = cast(dict[str, object], detail["target_ref"])
    assert target_ref["identity_key"] == "conversation:reader-c1"
    assert detail["anchor"] == "conversation-reader-c1"
    assert detail["title"] == "MK3 reader target contract"

    message_items = cast(list[dict[str, object]], messages["messages"])
    assert messages["total"] == 3
    assert message_items[0]["target_ref"] == {
        "target_type": "message",
        "target_id": "reader-c1-m1",
        "conversation_id": "reader-c1",
        "message_id": "reader-c1-m1",
        "identity_key": "message:reader-c1:reader-c1-m1",
    }
    assert message_items[2]["message_type"] == "tool_result"
    tool_actions = cast(dict[str, dict[str, object]], message_items[2]["actions"])
    assert tool_actions["copy_text"]["enabled"] is True

    assert raw["id"] == "reader-c1"
    assert "raw_artifacts" in raw

    checks = {
        "status": status,
        "content_type": content_type,
        "shell_bytes": len(shell.encode()),
        "conversation_target": target_ref["identity_key"],
        "message_total": messages["total"],
        "tool_message_present": True,
        "raw_endpoint_present": True,
        "private_path_safe": True,
    }
    write_evidence_manifest(
        tmp_path / "reader-conversation-dom-evidence.json",
        artifact_id="polylogue.local_reader.conversation",
        route="/c/reader-c1",
        fixture_id="reader-visual-synthetic-v1",
        checks=checks,
    )


def test_reader_search_query_no_results_and_facets_evidence(
    reader_workspace: ReaderWorkspace,
    tmp_path: Path,
) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        query = cast(dict[str, object], get_json(base_url, "/api/conversations?query=Hello"))
        no_results = cast(dict[str, object], get_json(base_url, "/api/conversations?query=zzzz_no_match"))
        facets = cast(dict[str, object], get_json(base_url, "/api/facets?query=Hello"))

    assert query["total"] == 1
    hits = cast(list[dict[str, object]], query["hits"])
    hit_target = cast(dict[str, object], hits[0]["target_ref"])
    hit_match = cast(dict[str, object], hits[0]["match"])
    match_target = cast(dict[str, object], hit_match["target_ref"])
    assert hit_target["identity_key"] == "conversation:reader-c1"
    assert match_target["identity_key"] == "message:reader-c1:reader-c1-m1"
    assert no_results["total"] == 0
    assert "diagnostics" in no_results
    assert facets["scoped_to_query"] is True
    assert facets["providers"] == {"claude-code": 1}
    assert_no_private_paths(json.dumps(query), context="reader search JSON")
    assert_no_private_paths(json.dumps(no_results), context="reader no-results JSON")
    assert_no_private_paths(json.dumps(facets), context="reader query facets JSON")

    write_evidence_manifest(
        tmp_path / "reader-query-dom-evidence.json",
        artifact_id="polylogue.local_reader.search.query",
        route="/api/conversations?query=Hello",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "query_total": query["total"],
            "no_results_total": no_results["total"],
            "diagnostics_present": "diagnostics" in no_results,
            "facet_scope": facets["scoped_to_query"],
            "private_path_safe": True,
        },
    )


def test_reader_empty_and_degraded_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace, conversations=False) as (_, base_url):
        empty_list = cast(dict[str, object], get_json(base_url, "/api/conversations"))
        empty_facets = cast(dict[str, object], get_json(base_url, "/api/facets"))

    with running_reader_server(reader_workspace, conversations=True, message_fts=False) as (_, base_url):
        degraded_status, _, degraded_body = get_text(base_url, "/api/conversations?query=Hello")

    assert empty_list["total"] == 0
    assert empty_list["items"] == []
    assert empty_facets["total_conversations"] == 0
    assert degraded_status == 503
    degraded_payload = json.loads(degraded_body)
    assert degraded_payload["ok"] is False
    assert degraded_payload["error"] == "DatabaseError"
    assert "Search index" in degraded_payload["detail"]
    assert "Traceback" not in degraded_body
    assert_no_private_paths(json.dumps(empty_list), context="reader empty list JSON")
    assert_no_private_paths(degraded_body, context="reader degraded JSON")

    write_evidence_manifest(
        tmp_path / "reader-degraded-dom-evidence.json",
        artifact_id="polylogue.local_reader.degraded",
        route="/api/conversations?query=Hello",
        fixture_id="reader-visual-synthetic-empty-and-degraded-v1",
        checks={
            "empty_total": empty_list["total"],
            "empty_facets_total": empty_facets["total_conversations"],
            "degraded_status": degraded_status,
            "sanitized_error": True,
            "private_path_safe": True,
        },
    )
