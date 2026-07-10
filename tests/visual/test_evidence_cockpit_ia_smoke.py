"""Evidence-cockpit IA redesign smoke evidence (feature/web/evidence-cockpit-redesign).

Companion to ``test_route_state_interaction_smoke.py`` (#2673 truthful-state
machinery, left unmodified by this redesign). These tests pin the new
information architecture built on top of it:

- a primary nav organized by the product's four verbs (Search / Analyze /
  Audit / Remember -- README.md, docs/demos.md) instead of leaving
  navigation implicit in route-name-shaped inspector tabs;
- an archive landing view (snapshot stats + recent sessions + verb entry
  points) rendered in #main instead of a bare "Select a session" placeholder;
- a session evidence strip rendered above the transcript, sourced from the
  already-fetched Insights/Cost/Lineage panel caches, so a session reads as
  tool activity and structural outcomes first, raw messages second;
- the Analyze/Audit/Remember panels wire in three routes that were already
  registered in route_contracts.py (``/api/provider-usage``,
  ``/api/archive-debt``, ``/api/assertions`` with no ``target_ref``) but were
  never called by this shell before this change -- no new API surface.

Two evidence tiers, matching the established pattern:

- Browserless DOM contract (always runs): the served shell HTML must carry
  the new IA building blocks.
- Best-effort real JS execution (skipped if ``node`` is not on PATH): the
  ACTUAL extracted functions from the real, assembled ``WEB_SHELL_HTML``,
  driven through a harness that fakes ``document``/``fetch``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.visual.conftest import (
    ReaderWorkspace,
    assert_no_private_paths,
    get_text,
    parse_dom,
    running_reader_server,
)

_NODE = shutil.which("node")


# ---------------------------------------------------------------------------
# Browserless DOM contract (always runs, no node dependency)
# ---------------------------------------------------------------------------


def test_verb_nav_and_ia_contract_in_served_shell(reader_workspace: ReaderWorkspace) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, body = get_text(base_url, "/")

    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(body, context="reader shell HTML")

    dom = parse_dom(body)
    assert "verb-nav" in dom.ids

    for phrase in (
        # The four product verbs (README.md / docs/demos.md), each a real
        # click target wired to setActiveView -- not just section headings.
        'data-verb="search"',
        'data-verb="analyze"',
        'data-verb="audit"',
        'data-verb="remember"',
        "function setActiveView",
        "function syncVerbNavButtons",
        # Archive landing replaces the bare "Select a session" placeholder.
        "function renderLandingView",
        'class="landing"',
        # Analyze/Audit/Remember panels and the routes they wire in.
        "function renderAnalyzePanel",
        "/api/provider-usage",
        "function renderAuditPanel",
        "/api/archive-debt",
        "function renderRememberPanel",
        # Session evidence strip (work-not-chat redesign).
        "function renderEvidenceStrip",
        "evidence-strip",
        # Failure handling for the three new panels follows the established
        # error-sentinel pattern (no infinite retry loop on a failed fetch).
        "function retryAnalyzePanel",
        "function retryAuditPanel",
        "function retryAssertionsPanel",
    ):
        assert phrase in body, f"missing evidence-cockpit IA marker: {phrase!r}"

    # Truthful-state machinery from #2673 must survive the redesign verbatim.
    for phrase in ("daemon-banner", "function renderDaemonBanner", "function debugDisclosure"):
        assert phrase in body, f"missing pre-existing truthful-state marker: {phrase!r}"


# ---------------------------------------------------------------------------
# Real JS execution against the actual extracted functions
# ---------------------------------------------------------------------------


def _extract_function(source: str, name: str) -> str:
    for prefix in (f"async function {name}(", f"function {name}("):
        idx = source.find(prefix)
        if idx == -1:
            continue
        brace_start = source.index("{", idx)
        depth = 0
        i = brace_start
        while True:
            ch = source[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return source[idx : i + 1]
            i += 1
    raise AssertionError(f"could not locate function {name} in WEB_SHELL_HTML")


def _extract_functions(source: str, names: list[str]) -> str:
    return "\n".join(_extract_function(source, name) for name in names)


_DOM_HARNESS_PRELUDE = r"""
var API = '';
var elements = {};
function makeElement(id) {
  var classes = [];
  return {
    id: id, innerHTML: '', textContent: '', className: '', title: '', style: {}, dataset: {},
    classList: {
      add: function(c) { if (classes.indexOf(c) === -1) classes.push(c); },
      remove: function(c) { var i = classes.indexOf(c); if (i !== -1) classes.splice(i, 1); },
      toggle: function(c, force) {
        var has = classes.indexOf(c) !== -1;
        var want = force === undefined ? !has : !!force;
        if (want && !has) classes.push(c);
        if (!want && has) classes.splice(classes.indexOf(c), 1);
      },
      contains: function(c) { return classes.indexOf(c) !== -1; }
    },
    addEventListener: function() {},
    querySelectorAll: function() { return []; }
  };
}
var document = {
  getElementById: function(id) { if (!elements[id]) elements[id] = makeElement(id); return elements[id]; },
  querySelector: function() { return null; },
  querySelectorAll: function() { return []; },
  addEventListener: function() {}
};
var window = { performance: { now: function() { return Date.now(); } } };
"""

_LANDING_FUNCS = [
    "esc",
    "escAttr",
    "escJsAttr",
    "statTile",
    "landingVerbCard",
    "readinessLabel",
    "renderLandingView",
]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_landing_view_renders_snapshot_and_verb_cards(tmp_path: Path) -> None:
    """renderLandingView() reads only already-loaded state (status snapshot +
    session list) -- no fetch of its own -- and must render real numbers
    (never inventing them) plus all four verb entry points."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(WEB_SHELL_HTML, _LANDING_FUNCS)
    harness = f"""
{_DOM_HARNESS_PRELUDE}
var state = {{
  status: {{total_sessions: 42, total_messages: 1337, component_readiness: {{
    search: {{state: 'ready'}}, embeddings: {{state: 'missing'}}
  }}}},
  sessions: [
    {{id: 's1', title: 'Repair the failing parser test', origin: 'codex-session',
      created_at: '2026-07-01T00:00:00Z', message_count: 12}}
  ],
  routeStates: {{}}
}};
{functions_src}
console.log(JSON.stringify({{html: renderLandingView()}}));
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    html = json.loads(proc.stdout)["html"]

    assert 'class="landing"' in html
    # Real numbers from state.status, not invented copy.
    assert "42" in html
    assert "1,337" in html
    assert "Repair the failing parser test" in html
    assert "codex-session" in html
    # All four verb entry points reachable from the landing.
    assert "focusSearchBox()" in html
    assert "setActiveView(&#39;analyze&#39;)" in html or "setActiveView('analyze')" in html
    assert "setActiveView(&#39;audit&#39;)" in html or "setActiveView('audit')" in html
    assert "setActiveView(&#39;remember&#39;)" in html or "setActiveView('remember')" in html


_EVIDENCE_STRIP_FUNCS = [
    "esc",
    "escAttr",
    "escJsAttr",
    "formatUsd",
    "evidenceStripToolCounts",
    "renderTopologyBranchChip",
    "renderEvidenceStrip",
]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_evidence_strip_surfaces_structural_tool_outcomes(tmp_path: Path) -> None:
    """The evidence strip must summarize the SAME structural is_error/exit_code
    -derived outcome kinds the Insights tab already reads (command_succeeded/
    command_failed/test_passed/test_failed from insights/transforms.py), not
    a re-derived or prose-based count -- and must render a failure count
    distinctly from a success count instead of collapsing them."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(WEB_SHELL_HTML, _EVIDENCE_STRIP_FUNCS)
    harness = f"""
{_DOM_HARNESS_PRELUDE}
function loadLineage() {{}}
var state = {{
  insightsPanels: {{
    's1': {{
      kinds: {{
        profile: {{profile: {{tool_use_count: 3}}}},
        timeline: {{events: [
          {{inference: {{kind: 'command_succeeded'}}}},
          {{inference: {{kind: 'command_failed'}}}},
          {{inference: {{kind: 'test_passed'}}}}
        ]}}
      }}
    }}
  }},
  costPanels: {{}},
  lineage: undefined
}};
{functions_src}
console.log(JSON.stringify({{html: renderEvidenceStrip({{id: 's1'}})}}));
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    html = json.loads(proc.stdout)["html"]

    assert 'class="evidence-strip"' in html
    assert "3 tool calls" in html
    assert "2 ok" in html
    assert "1 failed" in html
    assert "q-unresolved" in html  # the failed-count chip uses the failure quality class


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_evidence_strip_reports_missing_evidence_honestly(tmp_path: Path) -> None:
    """A session with zero materialized work events must say so explicitly
    (q-missing) -- never a bare empty strip that reads as "nothing happened"
    for an unmaterialized session (polylogue-bby.1 doctrine: no zero for
    unevaluated evidence)."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(WEB_SHELL_HTML, _EVIDENCE_STRIP_FUNCS)
    harness = f"""
{_DOM_HARNESS_PRELUDE}
function loadLineage() {{}}
var state = {{
  insightsPanels: {{'s1': {{kinds: {{profile: {{profile: {{tool_use_count: 0}}}}, timeline: {{events: []}}}}}}}},
  costPanels: {{}},
  lineage: undefined
}};
{functions_src}
console.log(JSON.stringify({{html: renderEvidenceStrip({{id: 's1'}})}}));
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    html = json.loads(proc.stdout)["html"]

    assert "no work events" in html
    assert "q-missing" in html


_NAV_FUNCS = ["esc", "escAttr", "escJsAttr", "syncVerbNavButtons", "setActiveView"]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_set_active_view_switches_state_and_repaints(tmp_path: Path) -> None:
    """setActiveView() is the only place the active verb changes. Proves it
    updates state.activeView and triggers a repaint (guarded so it stays a
    safe no-op in isolated extraction harnesses like this one)."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(WEB_SHELL_HTML, _NAV_FUNCS)
    harness = f"""
{_DOM_HARNESS_PRELUDE}
var state = {{activeView: 'search'}};
var renderMainCalls = 0;
function renderMain() {{ renderMainCalls += 1; }}
{functions_src}
setActiveView('audit');
console.log(JSON.stringify({{activeView: state.activeView, renderMainCalls: renderMainCalls}}));
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    result = json.loads(proc.stdout)

    assert result["activeView"] == "audit"
    assert result["renderMainCalls"] == 1


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_verb_nav_exposes_active_view_to_assistive_technology(tmp_path: Path) -> None:
    """Visual and accessible active state must move together between verbs."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(WEB_SHELL_HTML, _NAV_FUNCS)
    harness = f"""
function button(verb) {{
  return {{
    dataset: {{verb: verb}}, attributes: {{}},
    classList: {{toggle: function() {{}}}},
    setAttribute: function(name, value) {{ this.attributes[name] = value; }},
    removeAttribute: function(name) {{ delete this.attributes[name]; }}
  }};
}}
var buttons = [button('search'), button('analyze'), button('audit'), button('remember')];
var document = {{getElementById: function() {{ return {{querySelectorAll: function() {{ return buttons; }}}}; }}}};
var state = {{activeView: 'search'}};
function renderMain() {{}}
{functions_src}
syncVerbNavButtons();
setActiveView('audit');
console.log(JSON.stringify(buttons.map(function(btn) {{ return btn.attributes['aria-current'] || null; }})));
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    assert json.loads(proc.stdout) == [None, None, "page", None]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_workspace_route_returns_to_search_before_rendering(tmp_path: Path) -> None:
    """Stack/compare actions remain visible when launched from another verb."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(WEB_SHELL_HTML, ["loadWorkspaceRoute"])
    harness = f"""
var state = {{activeView: 'remember', mode: 'single', selected: {{id: 's1'}}, selectedRaw: {{}}}};
var navSyncCalls = 0, renderSnapshots = [];
function syncVerbNavButtons() {{ navSyncCalls += 1; }}
function pushWorkspaceURL() {{}}
function fetchJSON() {{ return Promise.resolve({{items: []}}); }}
function renderMain() {{ renderSnapshots.push({{activeView: state.activeView, mode: state.mode}}); }}
function renderInspector() {{}}
function renderSessions() {{}}
{functions_src}
(async function() {{
  await loadWorkspaceRoute({{mode: 'stack', ids: ['s1'], focus: 's1'}}, true);
  console.log(JSON.stringify({{activeView: state.activeView, navSyncCalls: navSyncCalls, renders: renderSnapshots}}));
}})();
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    result = json.loads(proc.stdout)
    assert result == {
        "activeView": "search",
        "navSyncCalls": 1,
        "renders": [{"activeView": "search", "mode": "stack"}],
    }


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_remember_saved_view_uses_wire_view_id(tmp_path: Path) -> None:
    """Remember must pass the saved-view API key, never an absent legacy id."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(
        WEB_SHELL_HTML,
        ["esc", "escAttr", "escJsAttr", "renderRouteStateNotice", "statTile", "renderRememberPanel"],
    )
    harness = f"""
var state = {{
  marks: {{}}, annotations: {{}}, assertionsPanel: {{items: []}},
  savedViews: [{{view_id: 'view-42', name: 'Review queue'}}], routeStates: {{}}
}};
{functions_src}
console.log(JSON.stringify({{html: renderRememberPanel()}}));
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    html = json.loads(proc.stdout)["html"]
    assert "applySavedView('view-42')" in html
    assert "undefined" not in html


_ANALYZE_FUNCS = [
    "esc",
    "escAttr",
    "escJsAttr",
    "nowMs",
    "nextApiRequestId",
    "summarizeApiBody",
    "renderApiDebugChip",
    "rememberApiDebug",
    "requestJSON",
    "fetchJSON",
    "fallbackCommand",
    "debugDisclosure",
    "routeErrorDetails",
    "statTile",
    "formatUsd",
    "renderInlineRouteFailure",
    "loadAnalyzePanel",
    "retryAnalyzePanel",
    "renderAnalyzePanel",
]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_analyze_panel_failure_is_terminal_not_a_retry_loop(tmp_path: Path) -> None:
    """A failed /api/provider-usage fetch must render one truthful failure
    panel with a retry action -- not re-issue the request on every render
    pass. This is the regression the error-sentinel pattern (matching
    loadCostPanel/loadInsightsPanel) exists to prevent."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(WEB_SHELL_HTML, _ANALYZE_FUNCS)
    harness = f"""
{_DOM_HARNESS_PRELUDE}
var state = {{activeView: 'analyze', apiDebug: {{counter: 0, last: null}}, analyzePanel: undefined}};
var fetchCallCount = 0;
global.fetch = function() {{ fetchCallCount += 1; return Promise.reject(new TypeError('fetch failed')); }};
function renderMain() {{}}
{functions_src}

(async function() {{
  var firstHtml = renderAnalyzePanel();
  await new Promise(function(r) {{ setTimeout(r, 20); }});
  var secondHtml = renderAnalyzePanel();
  var thirdHtml = renderAnalyzePanel();
  console.log(JSON.stringify({{
    firstHtml: firstHtml, secondHtml: secondHtml, thirdHtml: thirdHtml, fetchCallCount: fetchCallCount
  }}));
}})();
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    result = json.loads(proc.stdout)

    assert "Loading usage accounting" in result["firstHtml"]
    assert "Usage accounting unavailable" in result["secondHtml"]
    assert "retryAnalyzePanel" in result["secondHtml"]
    assert "Usage accounting unavailable" in result["thirdHtml"]
    # The core regression check: three render passes, exactly one fetch.
    assert result["fetchCallCount"] == 1
