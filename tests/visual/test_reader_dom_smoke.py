"""Typed WebUI route evidence over a synthetic archive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

from polylogue.surfaces.payloads import reader_anchor
from tests.visual.conftest import (
    READER_C1,
    READER_C1_M1,
    READER_C2,
    ReaderWorkspace,
    assert_no_private_paths,
    get_json,
    get_text,
    running_reader_server,
    write_evidence_manifest,
)


def _send_json(base_url: str, method: str, path: str, payload: dict[str, object]) -> tuple[int, object]:
    req = Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(req, timeout=10) as response:
        return response.status, json.loads(response.read())


def test_archive_overview_is_semantic_ssr(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, body = get_text(base_url, "/")

    assert status == 200
    assert "text/html" in content_type
    assert "<h1>Archive overview</h1>" in body
    assert 'id="archive-activity-list"' in body
    assert 'data-island="archive-overview"' in body
    assert 'src="/assets/archive-overview-' in body
    assert_no_private_paths(body, context="archive overview HTML")

    write_evidence_manifest(
        tmp_path / "typed-webui-overview-evidence.json",
        artifact_id="polylogue.webui.overview",
        route="/",
        fixture_id="reader-visual-synthetic-v1",
        checks={"status": status, "semantic_ssr": True, "private_path_safe": True},
    )


def test_workspace_routes_render_typed_projections(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        stack_status, stack_type, stack = get_text(
            base_url, f"/w/stack?ids={READER_C1},{READER_C2},missing-conv&focus={READER_C1}"
        )
        compare_status, compare_type, compare = get_text(
            base_url, f"/w/compare?left={READER_C1}&right={READER_C2}&align=prompt"
        )
        stack_payload = cast(
            dict[str, object], get_json(base_url, f"/api/stack?ids={READER_C1},{READER_C2},missing-conv")
        )
        compare_payload = cast(
            dict[str, object], get_json(base_url, f"/api/compare?left={READER_C1}&right={READER_C2}&align=prompt")
        )

    assert (stack_status, compare_status) == (200, 200)
    assert "text/html" in stack_type and "text/html" in compare_type
    assert "<h1>Workspace stack</h1>" in stack
    assert "<h1>Workspace compare</h1>" in compare
    assert stack_payload["resolved_count"] == 2
    assert stack_payload["degraded_count"] == 1
    assert compare_payload["mode"] == "compare"
    assert_no_private_paths(stack, context="workspace stack HTML")
    assert_no_private_paths(compare, context="workspace compare HTML")

    write_evidence_manifest(
        tmp_path / "typed-webui-workspace-evidence.json",
        artifact_id="polylogue.webui.workspace",
        route=f"/w/stack?ids={READER_C1},{READER_C2},missing-conv",
        fixture_id="reader-visual-synthetic-workspace-v1",
        checks={"stack_status": stack_status, "compare_status": compare_status, "private_path_safe": True},
    )


def test_session_reader_and_read_envelopes_agree(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, page = get_text(base_url, f"/s/{READER_C1}")
        detail = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}"))
        messages = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/read?view=messages"))
        raw = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/read?view=raw"))

    assert status == 200
    assert "text/html" in content_type
    assert "<h1>MK3 reader target contract</h1>" in page
    assert 'id="message-flow"' in page
    assert f'id="msg-{READER_C1_M1}"' in page
    assert detail["anchor"] == reader_anchor("session", READER_C1)
    assert messages["view"] == "messages"
    assert raw["view"] == "raw"
    assert_no_private_paths(page, context="session reader HTML")
    assert_no_private_paths(json.dumps(raw), context="raw read envelope")

    write_evidence_manifest(
        tmp_path / "typed-webui-session-evidence.json",
        artifact_id="polylogue.webui.session",
        route=f"/sessions/{READER_C1}",
        fixture_id="reader-visual-synthetic-v1",
        checks={"status": status, "message_anchor": READER_C1_M1, "private_path_safe": True},
    )


def test_search_and_dedicated_pages_use_typed_routes(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        search_status, search_type, search = get_text(base_url, "/search?q=Hello")
        cost_status, cost_type, cost = get_text(base_url, "/cost")
        query = cast(dict[str, object], get_json(base_url, "/api/sessions?query=Hello"))
        no_results = cast(dict[str, object], get_json(base_url, "/api/sessions?query=zzzz_no_match"))

    assert (search_status, cost_status) == (200, 200)
    assert "text/html" in search_type and "text/html" in cost_type
    assert "<h1>Search</h1>" in search
    assert 'data-island="search"' in search
    assert "<h1>Cost &amp; usage</h1>" in cost
    assert query["total"] == 1
    assert no_results["total"] == 0
    assert_no_private_paths(search, context="search HTML")
    assert_no_private_paths(cost, context="cost HTML")

    write_evidence_manifest(
        tmp_path / "typed-webui-search-evidence.json",
        artifact_id="polylogue.webui.search",
        route="/search?q=Hello",
        fixture_id="reader-visual-synthetic-v1",
        checks={"status": search_status, "query_total": query["total"], "private_path_safe": True},
    )


def test_overlay_operations_remain_route_backed(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, page_type, page = get_text(base_url, f"/sessions/{READER_C1}")
        mark_status, mark = _send_json(
            base_url, "POST", "/api/user/marks", {"session_id": READER_C1, "mark_type": "archive"}
        )
        annotation_status, annotation = _send_json(
            base_url,
            "POST",
            "/api/user/annotations",
            {
                "annotation_id": "reader-visual-flow-note",
                "session_id": READER_C1,
                "target_type": "message",
                "message_id": READER_C1_M1,
                "note_text": "Visual smoke overlay note",
            },
        )
        marks = cast(dict[str, object], get_json(base_url, f"/api/user/marks?session_id={READER_C1}"))
        annotations = cast(dict[str, object], get_json(base_url, f"/api/user/annotations?session_id={READER_C1}"))

    assert status == 200
    assert "text/html" in page_type
    assert "<h1>MK3 reader target contract</h1>" in page
    assert mark_status == 201
    assert cast(dict[str, object], mark)["operation"] == "mark.add"
    assert annotation_status == 201
    assert cast(dict[str, object], annotation)["operation"] == "annotation.save"
    assert "archive" in {str(item["mark_type"]) for item in cast(list[dict[str, object]], marks["items"])}
    assert "reader-visual-flow-note" in {
        str(item["annotation_id"]) for item in cast(list[dict[str, object]], annotations["items"])
    }

    write_evidence_manifest(
        tmp_path / "typed-webui-overlay-evidence.json",
        artifact_id="polylogue.webui.overlay_operations",
        route=f"/sessions/{READER_C1}",
        fixture_id="reader-visual-synthetic-v1",
        checks={"page_status": status, "mark_status": mark_status, "annotation_status": annotation_status},
    )


def test_unavailable_overview_and_degraded_search_are_explicit(
    reader_workspace: ReaderWorkspace, tmp_path: Path
) -> None:
    with running_reader_server(reader_workspace, sessions=False) as (_, base_url):
        empty_status, _, empty_page = get_text(base_url, "/")
        empty_list = cast(dict[str, object], get_json(base_url, "/api/sessions"))
    with running_reader_server(reader_workspace, sessions=True, message_fts=False) as (_, base_url):
        degraded_status, _, degraded_body = get_text(base_url, "/api/sessions?query=Hello")

    assert empty_status == 503
    assert "<h1>Archive overview</h1>" in empty_page
    assert empty_list["total"] == 0
    assert degraded_status == 200
    degraded_payload = json.loads(degraded_body)
    assert degraded_payload["route_state"]["state"] == "degraded"
    assert "Traceback" not in degraded_body

    write_evidence_manifest(
        tmp_path / "typed-webui-degraded-evidence.json",
        artifact_id="polylogue.webui.unavailable_and_degraded",
        route="/api/sessions?query=Hello",
        fixture_id="reader-visual-synthetic-empty-and-degraded-v1",
        checks={"unavailable_status": empty_status, "degraded_status": degraded_status, "sanitized": True},
    )
