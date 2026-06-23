"""Daemon-served reader visual/DOM smoke evidence (#865)."""

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
    parse_dom,
    running_reader_server,
    seed_reader_assertion_claims,
    write_evidence_manifest,
)


def _send_json(base_url: str, method: str, path: str, payload: dict[str, object] | None = None) -> tuple[int, object]:
    body = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    with urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


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
        "status-browser-capture",
        "sidebar",
        "search",
        "facet-bar",
        "conv-list",
        "main",
        "conv-header",
        "msg-list",
        "inspector",
        "inspector-tabs",
        "workspace-toolbar",
        "workspace-mode-switcher",
        "workspace-save-btn",
        "workspace-restore-select",
        "workspace-create-recall-pack-btn",
        "footer",
        "help-overlay",
    }
    assert expected_ids <= dom.ids
    assert dom.meta_viewport is True
    assert dom.scripts == 1
    assert dom.styles == 1
    for phrase in (
        "Select a session",
        "Keyboard Shortcuts",
        "Focus search",
        "Local",
        "/api/user/marks",
        "/api/user/annotations",
        "toggleMark",
        "saveAnnotation",
        "No annotations on this session",
        "Save current view",
        "Saved Views",
        "Save workspace",
        "Restore workspace",
        "Recall pack",
        "/api/user/workspaces",
        "/api/user/recall-packs",
        "/api/stack",
        "/api/compare",
    ):
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


def test_reader_stack_workspace_dom_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, shell = get_text(
            base_url, f"/w/stack?ids={READER_C1},{READER_C2},missing-conv&focus={READER_C1}"
        )
        stack = cast(
            dict[str, object],
            get_json(base_url, f"/api/stack?ids={READER_C1},{READER_C2},missing-conv&focus={READER_C1}"),
        )

    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(shell, context="reader stack workspace shell")
    for phrase in (
        "getWorkspaceRouteFromURL",
        "renderStackWorkspace",
        "stack-view",
        "stack-items",
        "stack-focus",
        "workspace-degraded-count",
        "workspace-save-btn",
        "workspace-create-recall-pack-btn",
    ):
        assert phrase in shell
    assert stack["mode"] == "stack"
    assert stack["resolved_count"] == 2
    assert stack["degraded_count"] == 1
    items = cast(list[dict[str, object]], stack["items"])
    assert items[2]["disabled_reason"] == "session_not_found"

    write_evidence_manifest(
        tmp_path / "reader-stack-workspace-dom-evidence.json",
        artifact_id="polylogue.local_reader.workspace.stack",
        route=f"/w/stack?ids={READER_C1},{READER_C2},missing-conv&focus={READER_C1}",
        fixture_id="reader-visual-synthetic-workspace-v1",
        checks={
            "status": status,
            "content_type": content_type,
            "resolved_count": stack["resolved_count"],
            "degraded_count": stack["degraded_count"],
            "workspace_controls_present": True,
            "private_path_safe": True,
        },
    )


def test_reader_compare_workspace_dom_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, shell = get_text(base_url, f"/w/compare?left={READER_C1}&right={READER_C2}&align=prompt")
        compare = cast(
            dict[str, object],
            get_json(base_url, f"/api/compare?left={READER_C1}&right={READER_C2}&align=prompt"),
        )

    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(shell, context="reader compare workspace shell")
    for phrase in (
        "renderCompareWorkspace",
        "compare-view",
        "compare-left-pane",
        "compare-right-pane",
        "compare-pairs",
        "compare-align-select",
        "compare-degraded-banner",
    ):
        assert phrase in shell
    assert compare["mode"] == "compare"
    assert compare["align"] == "prompt"
    assert cast(dict[str, object], compare["left"])["id"] == READER_C1
    assert cast(dict[str, object], compare["right"])["id"] == READER_C2
    assert cast(list[dict[str, object]], compare["pairs"])

    write_evidence_manifest(
        tmp_path / "reader-compare-workspace-dom-evidence.json",
        artifact_id="polylogue.local_reader.workspace.compare",
        route=f"/w/compare?left={READER_C1}&right={READER_C2}&align=prompt",
        fixture_id="reader-visual-synthetic-workspace-v1",
        checks={
            "status": status,
            "content_type": content_type,
            "align": compare["align"],
            "pair_count": len(cast(list[dict[str, object]], compare["pairs"])),
            "workspace_controls_present": True,
            "private_path_safe": True,
        },
    )


def test_reader_session_deeplink_and_detail_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, shell = get_text(base_url, f"/s/{READER_C1}")
        detail = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}"))
        messages = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/messages"))
        raw = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/raw"))
        marks = cast(dict[str, object], get_json(base_url, f"/api/user/marks?session_id={READER_C1}"))
        annotations = cast(dict[str, object], get_json(base_url, f"/api/user/annotations?session_id={READER_C1}"))
        saved_views = cast(dict[str, object], get_json(base_url, "/api/user/saved-views"))

    assert status == 200
    assert "text/html" in content_type
    # Deeplink resolver in the reader shell JS. Renamed getConvIdFromURL ->
    # getSessionIdFromURL in the conversation->session terminology sweep (#1810).
    assert "getSessionIdFromURL" in shell
    assert_no_private_paths(shell, context="reader /s deeplink shell")
    assert_no_private_paths(json.dumps(detail), context="reader detail JSON")
    assert_no_private_paths(json.dumps(messages), context="reader messages JSON")

    target_ref = cast(dict[str, object], detail["target_ref"])
    assert target_ref["identity_key"] == f"session:{READER_C1}"
    assert detail["anchor"] == reader_anchor("session", READER_C1)
    assert detail["title"] == "MK3 reader target contract"

    message_items = cast(list[dict[str, object]], messages["messages"])
    assert messages["total"] == 3
    first_target = cast(dict[str, object], message_items[0]["target_ref"])
    assert first_target["target_type"] == "message"
    assert first_target["target_id"] == READER_C1_M1
    assert first_target["session_id"] == READER_C1
    assert first_target["message_id"] == READER_C1_M1
    assert first_target["identity_key"] == f"message:{READER_C1}:{READER_C1_M1}"
    assert message_items[2]["message_type"] == "tool_result"
    tool_actions = cast(dict[str, dict[str, object]], message_items[2]["actions"])
    assert tool_actions["copy_text"]["enabled"] is True

    assert raw["id"] == READER_C1
    assert "raw_artifacts" in raw
    mark_types = {str(item["mark_type"]) for item in cast(list[dict[str, object]], marks["items"])}
    assert mark_types == {"pin", "star"}
    annotation_items = cast(list[dict[str, object]], annotations["items"])
    assert annotation_items[0]["annotation_id"] == "reader-ann-c1"
    assert annotation_items[0]["note_text"] == "This session anchors the MK3 reader evidence."
    view_items = cast(list[dict[str, object]], saved_views["items"])
    assert view_items[0]["name"] == "Claude Code reader fixtures"
    assert cast(dict[str, object], view_items[0]["query"])["provider"] == "claude-code"

    checks = {
        "status": status,
        "content_type": content_type,
        "shell_bytes": len(shell.encode()),
        "session_target": target_ref["identity_key"],
        "message_total": messages["total"],
        "tool_message_present": True,
        "raw_endpoint_present": True,
        "mark_types": sorted(mark_types),
        "annotation_count": len(annotation_items),
        "saved_view_count": len(view_items),
        "private_path_safe": True,
    }
    write_evidence_manifest(
        tmp_path / "reader-session-dom-evidence.json",
        artifact_id="polylogue.local_reader.session",
        route=f"/s/{READER_C1}",
        fixture_id="reader-visual-synthetic-v1",
        checks=checks,
    )


def test_reader_search_query_no_results_and_facets_evidence(
    reader_workspace: ReaderWorkspace,
    tmp_path: Path,
) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        query = cast(dict[str, object], get_json(base_url, "/api/sessions?query=Hello"))
        no_results = cast(dict[str, object], get_json(base_url, "/api/sessions?query=zzzz_no_match"))
        facets = cast(dict[str, object], get_json(base_url, "/api/facets?query=Hello"))

    assert query["total"] == 1
    hits = cast(list[dict[str, object]], query["hits"])
    hit_session = cast(dict[str, object], hits[0]["session"])
    hit_target = cast(dict[str, object], hit_session["target_ref"])
    hit_match = cast(dict[str, object], hits[0]["match"])
    match_target = cast(dict[str, object], hit_match["target_ref"])
    assert hit_target["identity_key"] == f"session:{READER_C1}"
    assert match_target["identity_key"] == f"message:{READER_C1}:{READER_C1_M1}"
    assert no_results["total"] == 0
    assert "diagnostics" in no_results
    assert facets["scoped_to_query"] is True
    assert facets["origins"] == {"claude-code-session": 1}
    assert "generated_at" in facets
    deferred_families = cast(dict[str, object], facets["deferred_families"])
    family_status = cast(dict[str, dict[str, object]], facets["family_status"])
    assert deferred_families == {"repos": "deferred_by_default", "action_types": "deferred_by_default"}
    assert family_status["repos"]["state"] == "deferred"
    assert_no_private_paths(json.dumps(query), context="reader search JSON")
    assert_no_private_paths(json.dumps(no_results), context="reader no-results JSON")
    assert_no_private_paths(json.dumps(facets), context="reader query facets JSON")

    write_evidence_manifest(
        tmp_path / "reader-query-dom-evidence.json",
        artifact_id="polylogue.local_reader.search.query",
        route="/api/sessions?query=Hello",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "query_total": query["total"],
            "no_results_total": no_results["total"],
            "diagnostics_present": "diagnostics" in no_results,
            "facet_scope": facets["scoped_to_query"],
            "facet_deferred_families": sorted(deferred_families.keys()),
            "private_path_safe": True,
        },
    )


def test_reader_cost_panel_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    """Cost panel endpoint surfaces typed cost shape with confidence chip vocabulary (#1122).

    Pins:
    - existing session returns the typed cost-panel payload with a
      confidence chip from the MK3 vocabulary;
    - unknown session returns 404 (not a blank panel);
    - shell HTML carries the new ``Cost`` tab + ``renderInspectorCost``
      so the panel is reachable through the inspector tab strip.
    """
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, shell = get_text(base_url, f"/s/{READER_C1}")
        known_cost = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/cost"))
        unknown_status, _, unknown_body = get_text(base_url, "/api/sessions/does-not-exist/cost")

    assert status == 200
    assert "text/html" in content_type
    # The Cost tab is wired into the inspector strip and its renderer is
    # present in the shell payload.
    for phrase in (
        'data-tab="cost"',
        "renderInspectorCost",
        "loadCostPanel",
        "q-canonical",
        "q-estimated",
        "q-heuristic",
        "q-unavailable",
        "Basis split",
        "Per-model",
    ):
        assert phrase in shell

    # Known session: typed envelope, with a chip from the closed vocabulary.
    assert known_cost["session_id"] == READER_C1
    assert known_cost["confidence_tag"] in {
        "q-canonical",
        "q-estimated",
        "q-heuristic",
        "q-unavailable",
    }
    assert isinstance(known_cost["basis"], dict)
    assert isinstance(known_cost["usage"], dict)
    assert isinstance(known_cost["per_model_breakdown"], list)
    assert "missing_reasons" in known_cost

    # Unknown session: 404, not a blank panel.
    assert unknown_status == 404
    unknown_payload = json.loads(unknown_body)
    assert unknown_payload.get("error") in {"not_found", None}

    write_evidence_manifest(
        tmp_path / "reader-cost-panel-dom-evidence.json",
        artifact_id="polylogue.local_reader.cost_panel",
        route=f"/api/sessions/{READER_C1}/cost",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "shell_status": status,
            "cost_endpoint_status": 200,
            "unknown_endpoint_status": unknown_status,
            "confidence_tag": known_cost["confidence_tag"],
            "has_basis_split": True,
            "has_per_model_block": True,
            "chip_vocabulary_in_shell": True,
            "private_path_safe": True,
        },
    )


def test_reader_evidence_panel_endpoint_and_shell_smoke(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    """Evidence tab uses shared recovery/assertion DTOs without raw leakage (#1846)."""

    with running_reader_server(reader_workspace) as (_, base_url):
        seed_reader_assertion_claims(reader_workspace)
        status, content_type, shell = get_text(base_url, f"/s/{READER_C1}")
        recovery = cast(
            dict[str, object],
            get_json(base_url, f"/api/sessions/{READER_C1}/recovery?report=work-packet&format=json"),
        )
        assertions = cast(
            dict[str, object],
            get_json(base_url, f"/api/assertions?target_ref=session%3A{READER_C1}&limit=10"),
        )
        context = cast(
            dict[str, object],
            get_json(base_url, f"/api/sessions/{READER_C1}/read?view=context"),
        )
        context_pack = cast(
            dict[str, object],
            get_json(base_url, f"/api/sessions/{READER_C1}/read?view=context-pack&max_messages=5"),
        )

    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(shell, context="reader evidence shell")
    for phrase in (
        'data-tab="evidence"',
        "renderInspectorEvidence",
        "renderBrowserCaptureChip",
        "renderReadViewExecution",
        "renderContextReadView",
        "renderContextPackReadView",
        "loadEvidencePanel",
        "/api/assertions?target_ref=",
        "/recovery?report=work-packet&format=json",
        "/read?view=",
    ):
        assert phrase in shell, f"missing evidence shell hook {phrase!r}"

    assert recovery["report"] == "work-packet"
    packet = cast(dict[str, object], recovery["work_packet"])
    assert packet["session_id"] == READER_C1
    assert packet["evidence_refs"]
    assert "raw_artifacts" not in json.dumps(recovery)

    claim_items = cast(list[dict[str, object]], assertions["items"])
    assert [item["assertion_id"] for item in claim_items] == ["reader-evidence-decision"]
    assert claim_items[0]["target_ref"] == f"session:{READER_C1}"
    assert context["view"] == "context"
    context_payload = cast(dict[str, object], context["payload"])
    assert context_payload["preamble_version"] == "1.0"
    assert context_pack["view"] == "context-pack"
    context_pack_payload = cast(dict[str, object], context_pack["payload"])
    context_sessions = cast(list[dict[str, object]], context_pack_payload["sessions"])
    assert context_sessions[0]["session_id"] == READER_C1
    context_provenance = cast(dict[str, object], context_pack_payload["provenance"])
    assert context_provenance["redacted"] is True

    write_evidence_manifest(
        tmp_path / "reader-evidence-panel-dom-evidence.json",
        artifact_id="polylogue.local_reader.evidence_panel",
        route=f"/s/{READER_C1}",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "shell_status": status,
            "recovery_report": recovery["report"],
            "work_packet_evidence_refs": len(cast(list[object], packet["evidence_refs"])),
            "assertion_count": assertions["total"],
            "context_pack_sessions": context_pack_payload["total_sessions"],
            "raw_artifacts_absent_from_recovery": True,
            "private_path_safe": True,
        },
    )


def test_reader_overlay_mutation_flow_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    """Reader overlay actions use route-backed mutation envelopes (#1846)."""

    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, shell = get_text(base_url, f"/s/{READER_C1}")
        mark_status, mark = _send_json(
            base_url,
            "POST",
            "/api/user/marks",
            {"session_id": READER_C1, "mark_type": "archive"},
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
    assert "text/html" in content_type
    assert_no_private_paths(shell, context="reader overlay mutation shell")
    for phrase in (
        "toggleMark",
        "saveAnnotation",
        "annotation-composer",
        "annotation-target-select",
        "annotation-note-input",
        "/api/user/marks",
        "/api/user/annotations",
    ):
        assert phrase in shell, f"missing overlay shell hook {phrase!r}"

    mark_payload = cast(dict[str, object], mark)
    annotation_payload = cast(dict[str, object], annotation)
    assert mark_status == 201
    assert mark_payload["operation"] == "mark.add"
    assert mark_payload["affected_count"] == 1
    assert mark_payload["target_id"] == READER_C1
    assert annotation_status == 201
    assert annotation_payload["operation"] == "annotation.save"
    assert annotation_payload["affected_count"] == 1
    assert annotation_payload["target_type"] == "message"
    assert annotation_payload["target_id"] == READER_C1_M1

    mark_types = {str(item["mark_type"]) for item in cast(list[dict[str, object]], marks["items"])}
    assert "archive" in mark_types
    annotation_ids = {str(item["annotation_id"]) for item in cast(list[dict[str, object]], annotations["items"])}
    assert "reader-visual-flow-note" in annotation_ids

    write_evidence_manifest(
        tmp_path / "reader-overlay-mutation-flow-evidence.json",
        artifact_id="polylogue.local_reader.overlay_mutations",
        route=f"/s/{READER_C1}",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "shell_status": status,
            "mark_status": mark_status,
            "annotation_status": annotation_status,
            "mark_operation": mark_payload["operation"],
            "annotation_operation": annotation_payload["operation"],
            "archive_mark_visible": "archive" in mark_types,
            "message_annotation_visible": "reader-visual-flow-note" in annotation_ids,
            "private_path_safe": True,
        },
    )


def test_reader_operator_flow_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    """Representative workbench flow stays route-backed and privacy-bounded (#1846)."""

    with running_reader_server(reader_workspace) as (_, base_url):
        shell_status, content_type, shell = get_text(base_url, f"/s/{READER_C1}")
        search = cast(dict[str, object], get_json(base_url, "/api/sessions?query=Hello&limit=5"))
        messages = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/read?view=messages"))
        recovery = cast(
            dict[str, object],
            get_json(base_url, f"/api/sessions/{READER_C1}/read?view=recovery&format=json"),
        )
        context = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/read?view=context"))
        raw_view = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/read?view=raw"))
        work_packet = cast(
            dict[str, object],
            get_json(base_url, f"/api/sessions/{READER_C1}/recovery?report=work-packet&format=json"),
        )
        assertions = cast(
            dict[str, object],
            get_json(base_url, f"/api/assertions?target_ref=session%3A{READER_C1}&limit=10"),
        )
        mark_status, mark = _send_json(
            base_url,
            "POST",
            "/api/user/marks",
            {"session_id": READER_C1, "mark_type": "star"},
        )
        annotation_status, annotation = _send_json(
            base_url,
            "POST",
            "/api/user/annotations",
            {
                "annotation_id": "reader-operator-flow-message-note",
                "session_id": READER_C1,
                "target_type": "message",
                "message_id": READER_C1_M1,
                "note_text": "Operator flow note",
            },
        )
        annotations = cast(dict[str, object], get_json(base_url, f"/api/user/annotations?session_id={READER_C1}"))
        provenance = cast(dict[str, object], get_json(base_url, f"/api/sessions/{READER_C1}/provenance"))

    assert shell_status == 200
    assert "text/html" in content_type
    assert_no_private_paths(shell, context="reader operator-flow shell")
    for phrase in (
        "read-profile-selector",
        "renderReadViewExecution",
        "renderInspectorEvidence",
        "annotation-composer",
        "loadRawData",
    ):
        assert phrase in shell

    assert search["total"] == 1
    assert messages["view"] == "messages"
    assert recovery["view"] == "recovery"
    assert context["view"] == "context"
    assert raw_view["view"] == "raw"
    assert work_packet["report"] == "work-packet"
    assert isinstance(assertions["total"], int)
    assert assertions["total"] >= 0
    assert mark_status in {200, 201}
    assert cast(dict[str, object], mark)["operation"] == "mark.add"
    assert annotation_status == 201
    annotation_payload = cast(dict[str, object], annotation)
    assert annotation_payload["operation"] == "annotation.save"
    assert annotation_payload["target_type"] == "message"
    assert annotation_payload["target_id"] == READER_C1_M1

    annotation_ids = {str(item["annotation_id"]) for item in cast(list[dict[str, object]], annotations["items"])}
    assert "reader-operator-flow-message-note" in annotation_ids
    assert_no_private_paths(json.dumps(raw_view), context="reader raw read-view JSON")
    assert_no_private_paths(json.dumps(provenance), context="reader provenance JSON")

    write_evidence_manifest(
        tmp_path / "reader-operator-flow-evidence.json",
        artifact_id="polylogue.local_reader.operator_flow",
        route=f"/s/{READER_C1}",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "shell_status": shell_status,
            "search_total": search["total"],
            "read_views": [messages["view"], recovery["view"], context["view"], raw_view["view"]],
            "work_packet_report": work_packet["report"],
            "mark_status": mark_status,
            "annotation_status": annotation_status,
            "annotation_target": annotation_payload["target_id"],
            "raw_provenance_private_path_safe": True,
            "private_path_safe": True,
        },
    )


def test_reader_insights_browser_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    """Insights browser endpoint surfaces typed envelope + readiness chips (#1120).

    Pins:
    - existing session returns the four-kind envelope (profile/timeline/
      phases/threads), each with a chip from the closed q-ready/q-partial/
      q-missing vocabulary;
    - unknown session returns 404 (not a blank panel);
    - include= subset honors the request and drops unknown tokens;
    - shell HTML carries the new ``Insights`` tab + ``renderInspectorInsights``
      so the panel is reachable through the inspector tab strip and uses
      the readiness chip CSS classes.
    """
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, shell = get_text(base_url, f"/s/{READER_C1}")
        known = cast(dict[str, object], get_json(base_url, f"/api/insights/sessions/{READER_C1}"))
        subset = cast(
            dict[str, object],
            get_json(base_url, f"/api/insights/sessions/{READER_C1}?include=profile,bogus,phases"),
        )
        unknown_status, _, unknown_body = get_text(base_url, "/api/insights/sessions/does-not-exist")

    assert status == 200
    assert "text/html" in content_type
    for phrase in (
        'data-tab="insights"',
        "renderInspectorInsights",
        "loadInsightsPanel",
        "q-ready",
        "q-partial",
        "q-missing",
        "Work events",
        "Phases",
        "Work threads",
    ):
        assert phrase in shell, f"missing phrase in shell: {phrase!r}"

    assert known["session_id"] == READER_C1
    assert isinstance(known["kinds"], dict)
    kinds = cast(dict[str, dict[str, object]], known["kinds"])
    assert set(kinds.keys()) == {"profile", "timeline", "phases", "threads"}
    for kind, body in kinds.items():
        assert body["readiness_tag"] in {"q-ready", "q-partial", "q-missing"}, kind
        assert "materialized" in body

    subset_kinds = cast(dict[str, dict[str, object]], subset["kinds"])
    assert set(subset_kinds.keys()) == {"profile", "phases"}
    assert subset["include"] == ["profile", "phases"]

    assert unknown_status == 404
    unknown_payload = json.loads(unknown_body)
    assert unknown_payload.get("error") in {"not_found", None}

    assert_no_private_paths(json.dumps(known), context="insights browser envelope")
    assert_no_private_paths(shell, context="reader shell HTML")

    write_evidence_manifest(
        tmp_path / "reader-insights-browser-dom-evidence.json",
        artifact_id="polylogue.local_reader.insights_browser",
        route=f"/api/insights/sessions/{READER_C1}",
        fixture_id="reader-visual-synthetic-v1",
        checks={
            "shell_status": status,
            "insights_endpoint_status": 200,
            "unknown_endpoint_status": unknown_status,
            "kinds_present": sorted(kinds.keys()),
            "profile_readiness": kinds["profile"]["readiness_tag"],
            "timeline_readiness": kinds["timeline"]["readiness_tag"],
            "phases_readiness": kinds["phases"]["readiness_tag"],
            "threads_readiness": kinds["threads"]["readiness_tag"],
            "subset_honored": list(subset_kinds.keys()),
            "chip_vocabulary_in_shell": True,
            "private_path_safe": True,
        },
    )


def test_reader_empty_and_degraded_evidence(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace, sessions=False) as (_, base_url):
        empty_list = cast(dict[str, object], get_json(base_url, "/api/sessions"))
        empty_facets = cast(dict[str, object], get_json(base_url, "/api/facets"))

    with running_reader_server(reader_workspace, sessions=True, message_fts=False) as (_, base_url):
        degraded_status, _, degraded_body = get_text(base_url, "/api/sessions?query=Hello")

    assert empty_list["total"] == 0
    assert empty_list["items"] == []
    assert empty_facets["total_sessions"] == 0
    assert set(cast(list[str], empty_facets["complete_families"])) >= {"total_counts", "origins", "tags"}
    assert empty_facets["deferred_families"] == {"repos": "deferred_by_default", "action_types": "deferred_by_default"}
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
        route="/api/sessions?query=Hello",
        fixture_id="reader-visual-synthetic-empty-and-degraded-v1",
        checks={
            "empty_total": empty_list["total"],
            "empty_facets_total": empty_facets["total_sessions"],
            "empty_facets_deferred": sorted(cast(dict[str, object], empty_facets["deferred_families"]).keys()),
            "degraded_status": degraded_status,
            "sanitized_error": True,
            "private_path_safe": True,
        },
    )
