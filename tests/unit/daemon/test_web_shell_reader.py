"""Web shell reader contract tests — anchors, folds, density, topology (#1518).

Pins the reader slice asset contracts added in #1518 (slices 1a–1d, 4a–4c).
Tests assert on the module-level string constants and on the assembled HTML
served through the daemon HTTP surface.

The module is organized by slice:

- ``TestMessageAnchors``          — slice 1a: stable message anchors
- ``TestContentFolds``             — slice 1b: fold/collapse for long content
- ``TestDensityToggle``            — slice 1c: dense/comfortable view toggle
- ``TestKeyboardNavigation``       — slice 1d: g g / G keyboard shortcuts
- ``TestTopologyEdgeDetail``       — slice 4a/4b/4c: edge-detail view
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from polylogue.insights.topology import SessionTopology

# All test classes in this module that start HTTP servers share an xdist
# group to prevent cross-worker port interference.
pytestmark = pytest.mark.xdist_group("web-reader")

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from polylogue.daemon import web_shell_lineage, web_shell_reader

# ---------------------------------------------------------------------------
# HTTP harness — mirrors tests/unit/daemon/test_web_reader.py
# ---------------------------------------------------------------------------


@contextmanager
def _running_server(
    workspace: dict[str, Path],
    *,
    seeded: bool = True,
) -> Iterator[tuple[HTTPServer, str]]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    if seeded:
        _seed_test_db(workspace)
    else:
        _seed_empty_schema(workspace)

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="web-shell-reader-test", daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _index_db_path(workspace: dict[str, Path]) -> Path:
    return workspace["data_root"] / "polylogue" / "index.db"


def _seed_test_db(workspace: dict[str, Path]) -> None:
    import sqlite3

    from polylogue.core.enums import Provider
    from polylogue.core.sources import origin_from_provider
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    db = _index_db_path(workspace)
    db.parent.mkdir(parents=True, exist_ok=True)
    initialize_archive_database(db, ArchiveTier.INDEX)
    conn = sqlite3.connect(str(db))
    for cid, prov, title in [
        ("c1", "claude-code", "Claude Code session about authentication"),
        ("c2", "chatgpt", "ChatGPT debugging session"),
        ("c3", "claude-ai", "Claude AI brainstorm thread"),
    ]:
        provider = Provider.from_string(prov)
        origin = origin_from_provider(provider).value
        conn.execute(
            """
            INSERT INTO sessions(native_id, origin, title, content_hash, created_at_ms, updated_at_ms)
            VALUES(?, ?, ?, ?, 1770000000000, 1770000000000)
            """,
            (cid, origin, title, f"hash-{cid}".encode().ljust(32, b"x")[:32]),
        )
        session_id = f"{origin}:{cid}"
        conn.execute(
            """
            INSERT INTO messages(session_id, native_id, position, role, content_hash)
            VALUES(?, ?, 0, 'user', ?)
            """,
            (session_id, f"m-{cid}", f"mhash-{cid}".encode().ljust(32, b"y")[:32]),
        )
        conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, text)
            VALUES(?, ?, 0, 'text', 'Hello reader')
            """,
            (f"{session_id}:m-{cid}", session_id),
        )
    conn.commit()
    conn.close()


def _seed_empty_schema(workspace: dict[str, Path]) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    db = _index_db_path(workspace)
    db.parent.mkdir(parents=True, exist_ok=True)
    initialize_archive_database(db, ArchiveTier.INDEX)


def _get_text(base_url: str, path: str) -> tuple[int, str, str]:
    req = Request(f"{base_url}{path}")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
    except HTTPError as exc:
        body = exc.read().decode()
        return exc.code, exc.headers.get("Content-Type", ""), body


# ---------------------------------------------------------------------------
# Slice 1a: Stable message anchors
# ---------------------------------------------------------------------------


class TestMessageAnchors:
    """Slice 1a: message anchor IDs and permalink generation."""

    def test_module_exports_reader_assets(self) -> None:
        """Reader module exports CSS, JS, and help HTML strings."""
        assert isinstance(web_shell_reader.READER_CSS, str)
        assert isinstance(web_shell_reader.READER_JS, str)
        assert isinstance(web_shell_reader.READER_HELP_HTML, str)
        assert set(web_shell_reader.__all__) == {"READER_CSS", "READER_JS", "READER_HELP_HTML"}

    def test_css_declares_anchor_permalink_styles(self) -> None:
        """The CSS must include hover-reveal styles for the # permalink."""
        css = web_shell_reader.READER_CSS
        assert ".msg-anchor-link" in css
        assert ".msg-block:hover .msg-anchor-link" in css
        assert ".msg-anchor-target" in css

    def test_js_declares_copy_message_anchor(self) -> None:
        """The JS must expose copyMessageAnchor for clicking # links."""
        js = web_shell_reader.READER_JS
        assert "function copyMessageAnchor" in js
        assert "msg-anchor-link" in js

    def test_assembled_html_contains_message_anchor_markup(self, workspace_env: dict[str, Path]) -> None:
        """The full assembled shell HTML must contain msg-anchor-link."""
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "msg-anchor-link" in body
        assert "msg-anchor-target" in body

    def test_render_message_blocks_produces_anchor_id_and_permalink(self) -> None:
        """Rendering a message produces an anchor target and permalink."""
        js = web_shell_reader.READER_JS
        # The renderMessageBlocks function embeds msg-anchor-link in each card.
        assert "msg-anchor-link" in js
        assert "msg-anchor-target" in js
        # The anchor ID format uses data-anchor attribute.
        assert "data-anchor=" in js


# ---------------------------------------------------------------------------
# Slice 1b: Fold/collapse for long content
# ---------------------------------------------------------------------------


class TestContentFolds:
    """Slice 1b: long-text fold toggle markup and JS support."""

    def test_css_declares_text_fold_styles(self) -> None:
        """The CSS must include styles for long-text fold wrapper."""
        css = web_shell_reader.READER_CSS
        assert ".msg-text-fold" in css
        assert ".text-fold-body" in css

    def test_js_declares_text_fold_functions(self) -> None:
        """The JS must expose toggleTextFold and the post-render hook."""
        js = web_shell_reader.READER_JS
        assert "function toggleTextFold" in js
        assert "_polyApplyTextFolds" in js
        assert "_polyInstallTextFoldObserver" in js
        assert "_TEXT_FOLD_LINE_LIMIT" in js
        assert "_TEXT_FOLD_CHAR_LIMIT" in js

    def test_text_fold_thresholds_are_configured(self) -> None:
        """The fold thresholds match the spec: 40 lines, 500 chars."""
        js = web_shell_reader.READER_JS
        assert "_TEXT_FOLD_LINE_LIMIT = 40" in js
        assert "_TEXT_FOLD_CHAR_LIMIT = 500" in js

    def test_text_fold_renders_fold_bar_with_metadata(self) -> None:
        """The fold bar must render line/char counts so the operator sees
        how much content is hidden. The fold-label is "text" and the
        fold-meta shows the counts."""
        js = web_shell_reader.READER_JS
        assert "lines, " in js  # "N lines, M chars"
        assert "chars" in js
        assert "fold-label" in js

    def test_assembled_html_contains_text_fold_markup(self, workspace_env: dict[str, Path]) -> None:
        """The assembled shell HTML must contain text-fold CSS classes."""
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "msg-text-fold" in body


# ---------------------------------------------------------------------------
# Slice 1c: Dense/comfortable view density toggle
# ---------------------------------------------------------------------------


class TestDensityToggle:
    """Slice 1c: density toggle with CSS custom properties and localStorage."""

    def test_css_declares_density_rules(self) -> None:
        """Density CSS must use [data-density] attribute selectors."""
        css = web_shell_reader.READER_CSS
        assert '[data-density="compact"]' in css
        assert ".density-toggle" in css

    def test_css_reduces_padding_in_compact_mode(self) -> None:
        """Compact mode must reduce message block padding and text size."""
        css = web_shell_reader.READER_CSS
        # Compact padding is 3px vs comfortable 7px.
        assert "padding: 3px 12px" in css
        assert "font-size: 10px" in css  # compact font-size reduction

    def test_js_declares_density_toggle_function(self) -> None:
        """The JS must expose installDensityToggle."""
        js = web_shell_reader.READER_JS
        assert "function installDensityToggle" in js
        assert "_DENSITY_KEY" in js

    def test_density_preference_persisted_to_localstorage(self) -> None:
        """The JS must use localStorage to persist density preference."""
        js = web_shell_reader.READER_JS
        assert "localStorage.setItem(_DENSITY_KEY" in js
        assert "localStorage.getItem(_DENSITY_KEY)" in js

    def test_density_defaults_to_comfortable(self) -> None:
        """Default density is 'comfortable' when nothing is stored."""
        js = web_shell_reader.READER_JS
        assert "'comfortable'" in js  # default value
        assert "saved = 'comfortable'" in js

    def test_assembled_html_contains_density_toggle_markup(self, workspace_env: dict[str, Path]) -> None:
        """The assembled shell HTML must contain the density-toggle class."""
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "density-toggle" in body


# ---------------------------------------------------------------------------
# Slice 1d: Keyboard navigation (g g / G)
# ---------------------------------------------------------------------------


class TestKeyboardNavigation:
    """Slice 1d: g g and G keyboard shortcuts."""

    def test_js_declares_scroll_shortcuts(self) -> None:
        """The keyboard handler must handle g g and G."""
        js = web_shell_reader.READER_JS
        assert "_doubleGtimer" in js
        # G (shift-g) for scroll-to-bottom.
        assert "scrollTo({top: msgList2.scrollHeight" in js
        # g g for scroll-to-top.
        assert "scrollTo({top: 0, behavior: 'smooth'})" in js

    def test_help_html_lists_new_shortcuts(self) -> None:
        """The help overlay must document g g and G."""
        help_html = web_shell_reader.READER_HELP_HTML
        assert "g g" in help_html
        assert "Scroll to top" in help_html
        assert "Scroll to bottom" in help_html

    def test_double_g_timer_resets_on_single_wait(self) -> None:
        """The double-g timer (500ms) is declared in the JS."""
        js = web_shell_reader.READER_JS
        assert "500" in js  # timer threshold in _doubleGtimer comparison

    def test_assembled_footer_lists_new_shortcuts(self, workspace_env: dict[str, Path]) -> None:
        """The footer hint strip must include g g and G."""
        with _running_server(workspace_env, seeded=False) as (_, base_url):
            _, _, body = _get_text(base_url, "/")
        assert "g g" in body
        assert "top" in body
        assert "bottom" in body


# ---------------------------------------------------------------------------
# Slice 4a/4b/4c: Topology edge detail view
# ---------------------------------------------------------------------------


def _make_topology() -> SessionTopology:
    """Build a small SessionTopology covering parent+target+sibling+child+unresolved."""
    from polylogue.core.types import SessionId
    from polylogue.insights.topology import (
        SessionTopology,
        TopologyEdge,
        TopologyEdgeKind,
        TopologyNode,
    )

    cid = SessionId

    nodes = (
        TopologyNode(
            session_id=cid("root"),
            origin="claude-code",
            title="Root",
            depth=0,
            is_root=True,
        ),
        TopologyNode(
            session_id=cid("target"),
            origin="claude-code",
            title="Target",
            depth=1,
            is_root=False,
        ),
        TopologyNode(
            session_id=cid("sibling"),
            origin="claude-code",
            title="Sibling",
            depth=1,
            is_root=False,
        ),
        TopologyNode(
            session_id=cid("child"),
            origin="claude-code",
            title="Child",
            depth=2,
            is_root=False,
        ),
        TopologyNode(
            session_id=cid("orphan"),
            origin="claude-code",
            title="Orphan",
            depth=1,
            is_root=False,
        ),
    )
    edges = (
        TopologyEdge(
            child_id=cid("target"),
            parent_id=cid("root"),
            parent_native_id="root-native",
            kind=TopologyEdgeKind.CONTINUATION,
            resolved=True,
        ),
        TopologyEdge(
            child_id=cid("sibling"),
            parent_id=cid("root"),
            parent_native_id="root-native",
            kind=TopologyEdgeKind.SIDECHAIN,
            resolved=True,
        ),
        TopologyEdge(
            child_id=cid("child"),
            parent_id=cid("target"),
            parent_native_id="target-native",
            kind=TopologyEdgeKind.SUBAGENT,
            resolved=True,
        ),
        # Unresolved edge — parent not yet ingested.
        TopologyEdge(
            child_id=cid("orphan"),
            parent_id=None,
            parent_native_id="missing-parent-native",
            kind=TopologyEdgeKind.CONTINUATION,
            resolved=False,
        ),
    )
    return SessionTopology(
        target_id=cid("target"),
        root_id=cid("root"),
        nodes=nodes,
        edges=edges,
        cycle_detected=False,
    )


class TestTopologyEdgeDetail:
    """Slice 4a/4b/4c: topology edge detail in the lineage inspector."""

    def test_lineage_js_renders_edges_section(self) -> None:
        """The lineage JS must render an 'Edges' header with a count."""
        js = web_shell_lineage.LINEAGE_JS
        assert "Edges (" in js
        assert "allEdges.length" in js

    def test_lineage_js_renders_resolved_status_chip(self) -> None:
        """Resolved edges must get a q-canonical/resolved chip."""
        js = web_shell_lineage.LINEAGE_JS
        assert "q-canonical" in js
        assert "resolved" in js

    def test_lineage_js_renders_unresolved_status_chip(self) -> None:
        """Unresolved edges must get a q-unresolved chip."""
        js = web_shell_lineage.LINEAGE_JS
        assert "q-unresolved" in js

    def test_lineage_js_renders_edge_type_chip(self) -> None:
        """Each edge must render its kind in a chip via lineageEdgeKindClass."""
        js = web_shell_lineage.LINEAGE_JS
        assert "lineageEdgeKindClass" in js
        assert "kindChip" in js

    def test_lineage_js_renders_source_to_target_with_clickable_links(self) -> None:
        """The edge detail must show source → target with clickable links."""
        js = web_shell_lineage.LINEAGE_JS
        assert "selectSession" in js  # clickable links use this
        assert "sourceLink" in js
        assert "targetLink" in js
        assert "→" in js  # arrow between source and target

    def test_lineage_js_renders_unresolved_placeholder(self) -> None:
        """Unresolved edges must render a placeholder — not silently absent (#1518 slice 4c)."""
        js = web_shell_lineage.LINEAGE_JS
        assert "Not yet ingested" in js
        assert "parent_native_id" in js

    def test_lineage_js_renders_confidence_when_available(self) -> None:
        """The edge detail must render confidence when present."""
        js = web_shell_lineage.LINEAGE_JS
        assert "edge.confidence" in js
        assert "confidence:" in js

    def test_lineage_js_renders_evidence_summary(self) -> None:
        """The edge detail must render an evidence summary when present."""
        js = web_shell_lineage.LINEAGE_JS
        assert "edge.evidence" in js

    def test_lineage_js_renders_reason_when_available(self) -> None:
        """The edge detail must render a reason when present."""
        js = web_shell_lineage.LINEAGE_JS
        assert "edge.reason" in js
        assert "reason:" in js

    def test_unresolved_edge_in_envelope_preserves_parent_native_id(self) -> None:
        """build_topology_envelope must include parent_native_id on unresolved edges."""
        from polylogue.daemon.topology_http import build_topology_envelope

        topology = _make_topology()
        envelope = build_topology_envelope(topology)
        edges: list[dict[str, object]] = envelope["edges"]  # type: ignore[assignment]
        # Find the unresolved edge.
        unresolved = [e for e in edges if not e["resolved"]]
        assert len(unresolved) >= 1
        assert unresolved[0]["parent_native_id"] == "missing-parent-native"
        assert unresolved[0]["parent_id"] is None

    def test_lineage_js_consumes_all_edge_fields_from_envelope(self) -> None:
        """Every field the edge-detail section reads from edge objects
        must be present in the build_topology_envelope output."""
        from polylogue.daemon.topology_http import build_topology_envelope

        topology = _make_topology()
        envelope = build_topology_envelope(topology)
        edges: list[dict[str, object]] = envelope["edges"]  # type: ignore[assignment]
        assert len(edges) >= 1
        edge_keys = set(edges[0].keys())
        for key in ("child_id", "parent_id", "parent_native_id", "kind", "resolved"):
            assert key in edge_keys, f"envelope edge must include '{key}'"

    def test_lineage_module_exports_only_lineage_js(self) -> None:
        """The lineage module contract is a single JS string constant."""
        assert isinstance(web_shell_lineage.LINEAGE_JS, str)
        assert web_shell_lineage.LINEAGE_JS.strip()
        assert web_shell_lineage.__all__ == ["LINEAGE_JS"]
