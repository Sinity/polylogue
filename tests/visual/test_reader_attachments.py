"""MK3 reader attachment product surface (#1199).

Pins the daemon web shell's contract for the attachment surface:

- per-message attachment cards rendered inline above the message body;
- session envelope carries an ``attachments`` list and each
  message envelope embeds its own ``attachments`` array;
- ``/api/sessions/<id>/attachments`` returns the same envelope
  shape, scoped to one session;
- ``/api/attachments`` returns the archive-wide library payload;
- ``/a`` library page serves a standalone HTML page with filters for
  mime / state / session;
- attachment states cover ``available``, ``missing-blob``,
  ``unsupported-kind``, ``too-large``, and ``quarantined`` derived
  purely from substrate fields;
- raw HTML/SVG attachments do not produce inline ``<script>`` blocks —
  the renderer always renders the card as escaped metadata.

Evidence is shape-based — we assert the literal class names, CSS
selectors, function names, DOM targets, and envelope keys land in the
served HTML / JSON so the contract surface is browserless-verifiable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.daemon.web_shell_attachments import (
    PREVIEW_SIZE_BUDGET,
    attachment_to_envelope,
    classify_state,
)
from tests.visual.conftest import (
    ATT_MISSING,
    ATT_OK,
    ATT_QUARANTINED,
    ATT_RAWHTML,
    ATT_TOOLARGE,
    ATT_UNSUPPORTED,
    READER_C1,
    READER_C1_M1,
    READER_C2,
    ReaderWorkspace,
    assert_no_private_paths,
    get_json,
    get_text,
    parse_dom,
    running_reader_server,
    seed_reader_attachments,
    write_evidence_manifest,
)


def test_classify_state_pure_function() -> None:
    """``classify_state`` derives the MK3 state token without I/O."""

    assert classify_state(path="blob/aa/x", size_bytes=10, mime_type="text/plain", provider_meta=None) == "available"
    assert classify_state(path=None, size_bytes=10, mime_type="text/plain", provider_meta=None) == "missing-blob"
    assert (
        classify_state(path="blob/x", size_bytes=10, mime_type="application/zip", provider_meta=None)
        == "unsupported-kind"
    )
    assert (
        classify_state(
            path="blob/x",
            size_bytes=PREVIEW_SIZE_BUDGET + 1,
            mime_type="video/mp4",
            provider_meta=None,
        )
        == "too-large"
    )
    assert (
        classify_state(path="blob/x", size_bytes=10, mime_type="text/html", provider_meta={"quarantined": True})
        == "quarantined"
    )
    # Quarantine wins over missing-blob — safety boundary first.
    assert (
        classify_state(path=None, size_bytes=10, mime_type="text/html", provider_meta={"quarantined": True})
        == "quarantined"
    )


def test_attachment_to_envelope_shape() -> None:
    """The envelope helper carries only typed, JSON-safe fields."""

    class _Stub:
        id = "att-1"
        mime_type = "text/plain"
        size_bytes = 256
        path = "blob/aa/aaaa"
        provider_meta = {"name": "n.txt"}
        name = "n.txt"

    envelope = attachment_to_envelope(_Stub(), session_id="c1", message_id="m1")
    assert envelope == {
        "attachment_id": "att-1",
        "session_id": "c1",
        "message_id": "m1",
        "name": "n.txt",
        "mime_type": "text/plain",
        "size_bytes": 256,
        "path": "blob/aa/aaaa",
        "state": "available",
    }


def test_reader_attachment_surface_contract(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        seed_reader_attachments(reader_workspace)
        status, content_type, body = get_text(base_url, "/")
        conv_payload = get_json(base_url, f"/api/sessions/{READER_C1}")
        per_conv = get_json(base_url, f"/api/sessions/{READER_C1}/attachments")
        library = get_json(base_url, "/api/attachments?limit=100")
        lib_status, lib_ctype, lib_body = get_text(base_url, "/a")

    # Reader shell carries the attachment slice.
    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(body, context="reader shell HTML")
    for phrase in (
        "_polyAttachmentCardHtml",
        "_polyRenderAttachmentCards",
        "_polyAttachmentStripHtml",
        "_polyAttachmentInspectorHtml",
        "renderInspectorAttachments",
        ".msg-attachment",
        ".att-inspector-list",
        'data-tab="attachments"',
    ):
        assert phrase in body, f"reader shell missing {phrase!r}"

    # Session envelope embeds attachments at both levels.
    assert isinstance(conv_payload, dict)
    conv_atts = conv_payload["attachments"]
    assert isinstance(conv_atts, list)
    by_id = {a["attachment_id"]: a for a in conv_atts}
    assert by_id[ATT_OK]["state"] == "available"
    assert by_id[ATT_MISSING]["state"] == "missing-blob"
    assert by_id[ATT_UNSUPPORTED]["state"] == "unsupported-kind"
    assert by_id[ATT_TOOLARGE]["state"] == "too-large"
    assert by_id[ATT_QUARANTINED]["state"] == "quarantined"

    messages = conv_payload["messages"]
    msg = next(m for m in messages if m["id"] == READER_C1_M1)
    assert "attachments" in msg
    assert {a["attachment_id"] for a in msg["attachments"]} >= {
        ATT_OK,
        ATT_MISSING,
        ATT_UNSUPPORTED,
        ATT_TOOLARGE,
        ATT_QUARANTINED,
        ATT_RAWHTML,
    }

    # Per-session endpoint returns the same envelope shape.
    assert isinstance(per_conv, dict)
    assert per_conv["total"] >= 6
    assert any(item["state"] == "missing-blob" for item in per_conv["items"])

    # Library envelope carries session context per row.
    assert isinstance(library, dict)
    lib_items = library["items"]
    assert isinstance(lib_items, list)
    assert lib_items, "library should expose seeded attachments"
    sample = lib_items[0]
    for required_key in (
        "attachment_id",
        "session_id",
        "session_title",
        "origin",
        "state",
        "message_anchor",
    ):
        assert required_key in sample, f"library entry missing {required_key!r}"

    # Library page DOM has the filter controls and the targets the
    # bootstrap script writes into.
    assert lib_status == 200
    assert "text/html" in lib_ctype
    assert_no_private_paths(lib_body, context="attachment library HTML")
    for phrase in (
        'id="att-list"',
        'id="att-empty"',
        'id="att-filter-mime"',
        'id="att-filter-state"',
        'id="att-filter-session"',
        "initAttachmentLibrary",
        "/api/attachments",
    ):
        assert phrase in lib_body, f"library page missing {phrase!r}"

    dom = parse_dom(lib_body)
    assert "att-list" in dom.ids
    assert "att-filter-state" in dom.ids

    write_evidence_manifest(
        tmp_path / "reader-attachments-evidence.json",
        artifact_id="polylogue.local_reader.attachment_surface",
        route="/a",
        fixture_id="reader-visual-attachments-v1",
        checks={
            "status": status,
            "content_type": content_type,
            "conv_envelope_attachments": len(conv_atts),
            "states_covered": sorted({a["state"] for a in conv_atts}),
            "library_route_status": lib_status,
            "library_items": len(lib_items),
            "private_path_safe": True,
        },
    )


def test_library_filters_apply(reader_workspace: ReaderWorkspace) -> None:
    """``mime``, ``state``, and ``session`` query params filter."""

    with running_reader_server(reader_workspace) as (_, base_url):
        seed_reader_attachments(reader_workspace)
        only_missing = get_json(base_url, "/api/attachments?state=missing-blob")
        only_text = get_json(base_url, "/api/attachments?mime=text/")
        only_other = get_json(base_url, f"/api/attachments?session={READER_C2}")

    assert isinstance(only_missing, dict)
    states_missing = {item["state"] for item in only_missing["items"]}
    assert states_missing == {"missing-blob"}

    assert isinstance(only_text, dict)
    assert only_text["items"], "text/ filter should match seeded text attachments"
    for item in only_text["items"]:
        assert (item["mime_type"] or "").startswith("text/")

    # reader-c2 has no attachments seeded — payload is empty.
    assert isinstance(only_other, dict)
    assert only_other["items"] == []


def test_attachment_library_empty_state(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    """``/api/attachments`` returns an empty envelope when nothing is seeded."""

    with running_reader_server(reader_workspace) as (_, base_url):
        payload = get_json(base_url, "/api/attachments")
        lib_status, _ctype, lib_body = get_text(base_url, "/a")

    assert isinstance(payload, dict)
    assert payload["items"] == []
    assert payload["total"] == 0
    assert lib_status == 200
    assert 'id="att-list"' in lib_body
    assert 'id="att-empty"' in lib_body

    write_evidence_manifest(
        tmp_path / "reader-attachments-empty-evidence.json",
        artifact_id="polylogue.local_reader.attachment_library_empty",
        route="/api/attachments",
        fixture_id="reader-visual-attachments-empty",
        checks={
            "items": 0,
            "total": 0,
            "page_status": lib_status,
            "page_has_list_target": True,
            "page_has_empty_target": True,
        },
    )


def test_raw_html_attachment_renders_no_inline_script(
    reader_workspace: ReaderWorkspace,
) -> None:
    """The safety boundary: a raw HTML attachment with a script-shaped
    name must not surface as an executable ``<script>`` tag in any
    served HTML."""

    with running_reader_server(reader_workspace) as (_, base_url):
        seed_reader_attachments(reader_workspace)
        _status, _ctype, shell_body = get_text(base_url, "/")
        _ls, _lc, library_body = get_text(base_url, "/a")
        payload = get_json(base_url, "/api/attachments?limit=200")

    # Name was injected — verify the bare ``<script>alert`` sequence
    # never appears in any served HTML. The JSON payload retains the
    # raw string (safe — it's a JSON value, not HTML), but the page
    # bodies must not contain an unescaped script tag derived from it.
    assert "<script>alert('xss')</script>.html" not in shell_body
    assert "<script>alert('xss')</script>.html" not in library_body
    # The raw JSON envelope still preserves the original value so
    # downstream consumers can detect / triage the suspect name.
    assert isinstance(payload, dict)
    names = {item.get("name") for item in payload["items"]}
    assert any("xss" in (n or "") for n in names)


@pytest.mark.parametrize(
    "size,expected",
    [
        (0, "available"),
        (PREVIEW_SIZE_BUDGET, "available"),
        (PREVIEW_SIZE_BUDGET + 1, "too-large"),
    ],
)
def test_size_budget_boundary(size: int, expected: str) -> None:
    """The size-budget boundary is inclusive of ``PREVIEW_SIZE_BUDGET``."""

    assert classify_state(path="blob/x", size_bytes=size, mime_type="text/plain", provider_meta=None) == expected
