"""Truthful route-state interaction smoke evidence (polylogue-bby.1).

Live-probe finding: the web shell rendered a populated session list under
status chips claiming "checking / 0 convs / 0 msgs", a facets panel stuck
showing a literal developer curl command in the product panel, and every
widget independently degrading to "Failed to fetch" with no single visible
"the daemon is unreachable" signal when the daemon died mid-session.

Two evidence tiers, matching the established pattern in
``tests/unit/daemon/test_web_shell_xss_escaping.py``:

- Browserless DOM contract (always runs): the served shell HTML must carry
  the truthful-state building blocks (``#daemon-banner``, the collapsed
  ``route-debug`` disclosure, the panel-failure helper, the reconnect loop).
- Best-effort real JS execution (skipped if ``node`` is not on PATH): the
  ACTUAL extracted ``loadSessions``/``renderSessions``/``loadStatus``/
  ``renderDaemonBanner``/``renderInlinePanelFailure`` from the real,
  assembled ``WEB_SHELL_HTML`` are driven through a harness that fakes
  ``document``/``fetch`` and proves the real runtime behavior, not a
  hand-copied mirror of the logic:

  1. ``test_session_list_survives_daemon_failure_seeded`` -- seeds a real
     archive, captures a REAL ``/api/sessions`` envelope from the running
     daemon HTTP server (contract fidelity), replays it as the shell's
     first fetch, then fails the second fetch (daemon died mid-session).
     Asserts the previously-rendered session rows are still visible (not
     blanked to "0 convs") and a truthful failed-route banner is now
     prepended above them.
  2. ``test_daemon_banner_gates_on_health_probe`` -- every fetch fails from
     the start; asserts the single ``/api/health`` probe drives one visible
     "daemon unreachable" banner with a reason, a collapsed debug
     disclosure (not an inline curl command), and an automatic reconnect
     timer -- the negative control for "slow/missing routes degrade
     visibly".
  3. ``test_panel_failure_carries_reason_and_retry`` -- proves the
     Insights/Cost/Evidence panel failure renderer now carries the route,
     status, and failure reason plus a retry action instead of the old
     static "<X> surface unavailable" string.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from tests.visual.conftest import (
    READER_C1,
    ReaderWorkspace,
    assert_no_private_paths,
    get_json,
    get_text,
    parse_dom,
    running_reader_server,
)

WEB_SHELL_MODULE_DIR = Path(__file__).resolve().parents[2] / "polylogue" / "daemon"
_NODE = shutil.which("node")


# ---------------------------------------------------------------------------
# Browserless DOM contract (always runs, no node dependency)
# ---------------------------------------------------------------------------


def test_truthful_route_state_contract_in_served_shell(
    reader_workspace: ReaderWorkspace,
) -> None:
    with running_reader_server(reader_workspace) as (_, base_url):
        status, content_type, body = get_text(base_url, "/")

    assert status == 200
    assert "text/html" in content_type
    assert_no_private_paths(body, context="reader shell HTML")

    dom = parse_dom(body)
    assert "daemon-banner" in dom.ids

    for phrase in (
        # Global daemon-liveness gate + reconnect loop.
        "renderDaemonBanner",
        "scheduleDaemonRetry",
        "retryDaemonHealth",
        "daemon unreachable",
        # Curl fallback collapsed behind a debug disclosure, not printed
        # inline in the primary product panel.
        "function debugDisclosure",
        'class="route-debug"',
        "<summary>debug</summary>",
        # Panel-scoped failure renderer with route/status/reason + retry,
        # replacing the old static "<X> surface unavailable" strings.
        "function renderInlinePanelFailure",
        "retryInsightsPanel",
        "retryCostPanel",
        "retryEvidencePanel",
        # loadSessions() no longer blanks state on a failed refresh.
        "Preserve the last-known session rows",
    ):
        assert phrase in body, f"missing truthful-route-state marker: {phrase!r}"

    # Regression guard: the fallback curl command must never be printed as a
    # bare inline <code> sibling of a route-state notice again -- it must
    # always be routed through debugDisclosure()'s collapsed <details>.
    assert "'<br><code>' + esc(rs.fallback" not in body
    assert "'<p><code>' + esc(details.fallback" not in body


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


def _extract_var_block(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    return source[start:end]


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
  documentElement: { dataset: {} },
  getElementById: function(id) { if (!elements[id]) elements[id] = makeElement(id); return elements[id]; },
  querySelector: function() { return null; },
  querySelectorAll: function() { return []; },
  addEventListener: function() {}
};
var window = {
  performance: { now: function() { return Date.now(); } },
  dispatchEvent: function() {}
};
function CustomEvent(name, init) { this.type = name; this.detail = init && init.detail; }
"""

_SESSION_LIST_FUNCS = [
    "esc",
    "escAttr",
    "escJsAttr",
    "nowMs",
    "nextApiRequestId",
    "summarizeApiBody",
    "renderApiDebugChip",
    "rememberApiDebug",
    "setWebAuthState",
    "bootstrapWebCredential",
    "ensureWebCredential",
    "requestJSON",
    "fetchJSON",
    "fallbackCommand",
    "debugDisclosure",
    "routeErrorDetails",
    "setRouteState",
    "routeStateQuality",
    "renderRouteStateNotice",
    "updateStatusCountsUnknown",
    "setChipQuality",
    "markSetFor",
    "hasMark",
    "sessionsFromListPayload",
    "renderSidebarState",
    "renderSessions",
    "isSelectionSelected",
    "selectionSelectedIds",
    "selectionSelectedCount",
    "renderSelectionToolbar",
    "renderSelectionStatus",
    "cssEscape",
    "maybeAnimateExistingRow",
    "loadSessions",
]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_session_list_survives_daemon_failure_seeded(reader_workspace: ReaderWorkspace, tmp_path: Path) -> None:
    """Seed a real archive, capture the REAL /api/sessions envelope from the
    running daemon, then prove the actual shipped loadSessions()/
    renderSessions() keep the previously-rendered rows visible (with a
    truthful failed-route banner) when a second refresh's fetch fails --
    the exact live-probe regression named in polylogue-bby.1."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    with running_reader_server(reader_workspace) as (_, base_url):
        captured = cast(dict[str, object], get_json(base_url, "/api/sessions?limit=100&offset=0"))

    assert isinstance(captured, dict)
    items = cast(list[dict[str, object]], captured.get("items") or [])
    assert items, "seeded reader archive must yield at least one session row"
    assert any(item.get("id") == READER_C1 for item in items)

    functions_src = _extract_functions(WEB_SHELL_HTML, _SESSION_LIST_FUNCS)
    web_auth_failures_src = _extract_var_block(WEB_SHELL_HTML, "var WEB_AUTH_FAILURES", "];")
    harness = f"""
{_DOM_HARNESS_PRELUDE}
{web_auth_failures_src}
var state = {{
  sessions: [], selected: null, origin: '', query: '', offset: 0, limit: 100, total: 0,
  routeStates: {{}}, marks: {{}}, selectionSet: {{}}, lastSelectionResult: null,
  apiDebug: {{counter: 0, last: null}}, actionAffordances: [],
  webAuth: {{state: 'unknown', expiresAt: null, bootstrapPromise: null, lastFailure: null}}
}};
{functions_src}

var CAPTURED_PAYLOAD = {json.dumps(captured)};
var fetchCallCount = 0;
global.fetch = function(url, opts) {{
  if (url === '/api/web-auth/session') {{
    return Promise.resolve({{
      ok: true, status: 200,
      json: function() {{ return Promise.resolve({{credential: {{state: 'ready'}}}}); }}
    }});
  }}
  fetchCallCount += 1;
  if (fetchCallCount === 1) {{
    return Promise.resolve({{
      ok: true, status: 200,
      headers: {{ get: function() {{ return ''; }} }},
      text: function() {{ return Promise.resolve(JSON.stringify(CAPTURED_PAYLOAD)); }}
    }});
  }}
  return Promise.reject(new TypeError('fetch failed'));
}};

(async function() {{
  await loadSessions();
  var afterFirstHtml = document.getElementById('conv-list').innerHTML;
  var sessionsAfterFirst = (state.sessions || []).length;

  await loadSessions();
  var afterSecondHtml = document.getElementById('conv-list').innerHTML;
  var sessionsAfterSecond = (state.sessions || []).length;

  console.log(JSON.stringify({{
    after_first_html: afterFirstHtml,
    after_second_html: afterSecondHtml,
    sessions_after_first: sessionsAfterFirst,
    sessions_after_second: sessionsAfterSecond,
    route_state: state.routeStates.sessionList
  }}));
  process.exit(0);
}})();
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    result = json.loads(proc.stdout)

    first_title = "MK3 reader target contract"
    assert first_title in result["after_first_html"], "first successful load must render seeded session rows"
    assert result["sessions_after_first"] == len(items)

    # The core regression: a failed refresh must not blank previously-loaded
    # rows to a false "0 conversations" state.
    assert result["sessions_after_second"] == len(items), (
        "loadSessions() must retain the last-known session rows on a failed refresh"
    )
    assert first_title in result["after_second_html"], "stale rows must remain visible after a failed refresh"
    assert "No sessions in archive" not in result["after_second_html"]
    assert "0 sessions" not in result["after_second_html"]

    # A truthful failed-route banner must be visible above the stale rows.
    assert 'data-route-state-name="sessionList"' in result["after_second_html"]
    assert 'data-route-state="failed"' in result["after_second_html"]
    assert "q-unavailable" in result["after_second_html"]
    assert result["route_state"]["state"] == "failed"
    assert result["route_state"]["stale_available"] is True


_STATUS_FUNCS = [
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
    "setRouteState",
    "routeStateQuality",
    "renderRouteStateNotice",
    "updateStatusCountsUnknown",
    "setChipQuality",
    "renderFacets",
    "renderDevLoopChip",
    "renderDaemonBanner",
    "retryDaemonHealth",
    "scheduleDaemonRetry",
    "loadStatus",
]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_daemon_banner_gates_on_health_probe(tmp_path: Path) -> None:
    """Negative control: every route fails from the start (daemon fully
    unreachable). Proves the single /api/health probe drives one visible
    banner with a reason and a collapsed debug disclosure, plus schedules
    an automatic reconnect -- not N independent silent per-widget
    failures."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    daemon_retry_block = _extract_var_block(
        WEB_SHELL_HTML, "var daemonRetryTimer = null;", "var DAEMON_RETRY_MAX_MS = 30000;"
    )
    functions_src = _extract_functions(WEB_SHELL_HTML, _STATUS_FUNCS)
    harness = f"""
{_DOM_HARNESS_PRELUDE}
var state = {{
  routeStates: {{}}, facets: null, total: 0, apiDebug: {{counter: 0, last: null}}
}};
{daemon_retry_block}
{functions_src}

global.fetch = function() {{ return Promise.reject(new TypeError('fetch failed')); }};

(async function() {{
  await loadStatus();
  var banner = document.getElementById('daemon-banner');
  console.log(JSON.stringify({{
    banner_visible: banner.classList.contains('visible'),
    banner_html: banner.innerHTML,
    health_state: state.routeStates.health,
    retry_scheduled: daemonRetryTimer !== null,
    retry_delay_ms: daemonRetryDelayMs
  }}));
  process.exit(0);
}})();
"""
    (tmp_path / "harness.js").write_text(harness, encoding="utf-8")
    proc = subprocess.run([_NODE, str(tmp_path / "harness.js")], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    result = json.loads(proc.stdout)

    assert result["banner_visible"] is True
    assert "daemon unreachable" in result["banner_html"]
    assert "Retry now" in result["banner_html"]
    # The fallback curl hint is present but collapsed, not printed inline.
    assert '<details class="route-debug">' in result["banner_html"]
    assert "curl -fsS" in result["banner_html"]
    assert result["health_state"]["state"] == "error"
    # An automatic reconnect must be scheduled -- no silent dead end.
    assert result["retry_scheduled"] is True
    assert result["retry_delay_ms"] == 6000  # doubled from the 3000ms base after the first failure


_PANEL_FAILURE_FUNCS = [
    "esc",
    "escAttr",
    "fallbackCommand",
    "debugDisclosure",
    "routeErrorDetails",
    "renderInlinePanelFailure",
]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_panel_failure_carries_reason_and_retry(tmp_path: Path) -> None:
    """Insights/Cost/Evidence/Raw panel failures used to collapse into a
    static "<X> surface unavailable" string -- indistinguishable from a
    route that legitimately returned zero rows. Proves the real
    renderInlinePanelFailure() now surfaces route, status, and reason, a
    retry action, and keeps the curl fallback collapsed."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    functions_src = _extract_functions(WEB_SHELL_HTML, _PANEL_FAILURE_FUNCS)
    harness = f"""
{functions_src}
var err = new Error('500');
err.status = 500;
err.response_summary = 'materialization backlog exceeded budget';
err.request_id = 'web-abc123';
var details = routeErrorDetails(err, '/api/insights/sessions/example-session');
var html = renderInlinePanelFailure('Insights unavailable', details, "retryInsightsPanel('example-session')");
console.log(JSON.stringify({{details: details, html: html}}));
"""
    proc = subprocess.run([_NODE, "-e", harness], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    result = json.loads(proc.stdout)
    html = result["html"]

    assert "Insights unavailable" in html
    assert "route /api/insights/sessions/example-session" in html
    assert "status 500" in html
    assert "materialization backlog exceeded budget" in html
    assert (
        'onclick="retryInsightsPanel(&#39;example-session&#39;)"' in html
        or "retryInsightsPanel('example-session')" in html
    )
    assert "Retry" in html
    # The curl fallback must be collapsed behind the debug disclosure, not a
    # bare inline <code> line in the panel body.
    assert '<details class="route-debug">' in html
    assert re.search(r"<p><code>|<br><code>", html) is None
