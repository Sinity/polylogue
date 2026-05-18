"""MK3 reader attachment product surface (#1199).

Pins the daemon web shell's contract for the attachment surface:

- per-message attachment cards rendered inline above the message body;
- conversation envelope carries an ``attachments`` list and each
  message envelope embeds its own ``attachments`` array;
- ``/api/conversations/<id>/attachments`` returns the same envelope
  shape, scoped to one conversation;
- ``/api/attachments`` returns the archive-wide library payload;
- ``/a`` library page serves a standalone HTML page with filters for
  mime / state / conversation;
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

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from polylogue.daemon.web_shell_attachments import (
    PREVIEW_SIZE_BUDGET,
    attachment_to_envelope,
    classify_state,
)
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


def _insert_attachment(
    conn: sqlite3.Connection,
    *,
    attachment_id: str,
    conversation_id: str,
    message_id: str,
    mime_type: str | None,
    size_bytes: int | None,
    path: str | None,
    provider_meta: dict[str, object] | None = None,
) -> None:
    meta_blob = json.dumps(provider_meta or {})
    conn.execute(
        """
        INSERT INTO attachments(
            attachment_id, mime_type, size_bytes, path, ref_count, provider_meta
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (attachment_id, mime_type, size_bytes, path, 1, meta_blob),
    )
    conn.execute(
        """
        INSERT INTO attachment_refs(
            ref_id, attachment_id, conversation_id, message_id, provider_meta
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), attachment_id, conversation_id, message_id, meta_blob),
    )


def _seed_attachments(workspace: ReaderWorkspace) -> None:
    """Insert one attachment of each MK3 state under ``reader-c1``.

    The seed produces five distinct attachment rows linked to the
    first message of the seeded ``reader-c1`` conversation so the
    library + inspector + inline-card surfaces all have evidence to
    render at once.
    """

    db = archive_db_path(workspace)
    conn = sqlite3.connect(str(db))
    _insert_attachment(
        conn,
        attachment_id="att-ok",
        conversation_id="reader-c1",
        message_id="reader-c1-m1",
        mime_type="text/plain",
        size_bytes=1024,
        path="blob/aa/aaaa-ok",
        provider_meta={"name": "notes.txt"},
    )
    _insert_attachment(
        conn,
        attachment_id="att-missing",
        conversation_id="reader-c1",
        message_id="reader-c1-m1",
        mime_type="image/png",
        size_bytes=2048,
        path=None,
        provider_meta={"name": "screenshot.png"},
    )
    _insert_attachment(
        conn,
        attachment_id="att-unsupported",
        conversation_id="reader-c1",
        message_id="reader-c1-m1",
        mime_type="application/zip",
        size_bytes=4096,
        path="blob/bb/bbbb-zip",
        provider_meta={"name": "bundle.zip"},
    )
    _insert_attachment(
        conn,
        attachment_id="att-toolarge",
        conversation_id="reader-c1",
        message_id="reader-c1-m1",
        mime_type="video/mp4",
        size_bytes=PREVIEW_SIZE_BUDGET + 1,
        path="blob/cc/cccc-video",
        provider_meta={"name": "recording.mp4"},
    )
    _insert_attachment(
        conn,
        attachment_id="att-quarantined",
        conversation_id="reader-c1",
        message_id="reader-c1-m1",
        mime_type="text/html",
        size_bytes=512,
        path="blob/dd/dddd-html",
        provider_meta={"name": "suspect.html", "quarantined": True},
    )
    # A raw HTML attachment for the safety-boundary check. The name
    # carries injection markers — the renderer must escape them.
    _insert_attachment(
        conn,
        attachment_id="att-rawhtml",
        conversation_id="reader-c1",
        message_id="reader-c1-m1",
        mime_type="text/html",
        size_bytes=2048,
        path="blob/ee/eeee-html",
        provider_meta={"name": "<script>alert('xss')</script>.html"},
    )
    conn.commit()
    conn.close()


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

    envelope = attachment_to_envelope(_Stub(), conversation_id="c1", message_id="m1")
    assert envelope == {
        "attachment_id": "att-1",
        "conversation_id": "c1",
        "message_id": "m1",
        "name": "n.txt",
        "mime_type": "text/plain",
        "size_bytes": 256,
        "path": "blob/aa/aaaa",
        "state": "available",
    }


def test_reader_attachment_surface_contract(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        _seed_attachments(reader_workspace)
        status, content_type, body = get_text(base_url, "/")
        conv_payload = get_json(base_url, "/api/conversations/reader-c1")
        per_conv = get_json(base_url, "/api/conversations/reader-c1/attachments")
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

    # Conversation envelope embeds attachments at both levels.
    assert isinstance(conv_payload, dict)
    conv_atts = conv_payload["attachments"]
    assert isinstance(conv_atts, list)
    by_id = {a["attachment_id"]: a for a in conv_atts}
    assert by_id["att-ok"]["state"] == "available"
    assert by_id["att-missing"]["state"] == "missing-blob"
    assert by_id["att-unsupported"]["state"] == "unsupported-kind"
    assert by_id["att-toolarge"]["state"] == "too-large"
    assert by_id["att-quarantined"]["state"] == "quarantined"

    messages = conv_payload["messages"]
    msg = next(m for m in messages if m["id"] == "reader-c1-m1")
    assert "attachments" in msg
    assert {a["attachment_id"] for a in msg["attachments"]} >= {
        "att-ok",
        "att-missing",
        "att-unsupported",
        "att-toolarge",
        "att-quarantined",
        "att-rawhtml",
    }

    # Per-conversation endpoint returns the same envelope shape.
    assert isinstance(per_conv, dict)
    assert per_conv["total"] >= 6
    assert any(item["state"] == "missing-blob" for item in per_conv["items"])

    # Library envelope carries conversation context per row.
    assert isinstance(library, dict)
    lib_items = library["items"]
    assert isinstance(lib_items, list)
    assert lib_items, "library should expose seeded attachments"
    sample = lib_items[0]
    for required_key in (
        "attachment_id",
        "conversation_id",
        "conversation_title",
        "provider",
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
        'id="att-filter-conversation"',
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
    """``mime``, ``state``, and ``conversation`` query params filter."""

    with running_reader_server(reader_workspace) as (_, base_url):
        _seed_attachments(reader_workspace)
        only_missing = get_json(base_url, "/api/attachments?state=missing-blob")
        only_text = get_json(base_url, "/api/attachments?mime=text/")
        only_other = get_json(base_url, "/api/attachments?conversation=reader-c2")

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
        _seed_attachments(reader_workspace)
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
