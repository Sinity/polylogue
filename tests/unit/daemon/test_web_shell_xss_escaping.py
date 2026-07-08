"""Web-shell stored-XSS escaping regression tests (polylogue-2n39).

Three ``<script>`` fragments (``web_shell.py``, ``web_shell_attachments.py``,
``web_shell_reader.py``, and several more ``web_shell_*.py`` modules) get
concatenated into ONE inline script and share one JS top-level scope. Before
this fix, ``web_shell_attachments.py`` defined its own ``esc``/``escAttr``
(HTML-entity escaping only) which, due to ``function`` redeclaration
semantics, silently overrode ``web_shell.py``'s earlier, differently-broken
``escAttr`` for the WHOLE page. Both were wrong for the same reason from two
different angles:

- ``web_shell.py``'s original ``escAttr`` escaped quotes for a JS string
  but never escaped backslashes first, so a value ending in ``\\`` could
  consume the escaping backslash of the template's own closing quote.
- ``web_shell_attachments.py``'s ``escAttr`` (== ``esc``) HTML-entity-escaped
  quotes (``'`` -> ``&#39;``), which does NOT protect a JS string literal
  embedded in an HTML attribute: the browser's HTML parser decodes entities
  in the attribute value BEFORE the JS engine parses that decoded string as
  the event handler's source, so the escaping is invisible to the JS parser
  and the attacker's raw quote reappears right where it can break out.

Both are exploitable stored XSS: a hostile/malformed captured provider
session can carry a crafted ``mime_type``/``session_id``/``message_id``/
``view_id``/``annotation_id`` etc. that, once rendered into an
``onclick="fn('VALUE')"`` sink, executes arbitrary JS in the operator's own
browser session the next time the web shell renders that content.

The fix: a single canonical ``escJsAttr`` (JS-string-escape the value FIRST,
backslash before quote, THEN HTML-attribute-escape the result) used at
every sink that interpolates a value into a JS string literal inside an
event-handler attribute, plus a fixed ``escAttr`` for plain (non-JS)
HTML-attribute contexts and an ``esc`` for HTML text.

These tests are split into two tiers:

- Static, dependency-free (always run): grep every ``web_shell*.py`` module
  for the exact bug shape (an ``escAttr(`` call immediately followed by a
  JS-string closing quote, i.e. inside ``\\'...\\'``) and fail if any
  remain -- this is the direct regression guard for the bug class.
- Best-effort JS execution (skipped if ``node`` is not on PATH -- Node.js
  is not a declared flake dependency of this project, only incidentally
  present in some dev environments): actually run the real, extracted
  ``esc``/``escAttr``/``escJsAttr`` from ``WEB_SHELL_HTML`` through a
  harness that simulates the browser's HTML-attribute-value entity
  decoding step, then parses the decoded string as JS -- proving the fix
  holds at the actual runtime semantics the bead's AC calls for, not just
  "the source doesn't have the old pattern".
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_SHELL_MODULE_DIR = Path(__file__).resolve().parents[3] / "polylogue" / "daemon"

# ---------------------------------------------------------------------------
# Static, dependency-free regression guard
# ---------------------------------------------------------------------------

# Matches `escAttr(` immediately preceded by a JS single-quote inside a JS
# string literal boundary (`\'...escAttr(...)...\'`) -- the exact shape of
# "a value is being interpolated into a JS string literal but escaped with
# the wrong function". A legitimate JS-string sink now uses `escJsAttr(`
# instead; any remaining `escAttr(` in this shape is the bug reappearing.
_JS_STRING_ESCATTR_SINK = re.compile(r"\\'[^\\']*escAttr\(")


def _web_shell_source_files() -> list[Path]:
    return sorted(WEB_SHELL_MODULE_DIR.glob("web_shell*.py"))


def test_no_escattr_call_remains_inside_a_js_string_literal_context() -> None:
    """Regression guard: escAttr (HTML-attribute escaping) must never be
    used where a value is interpolated into a JS string literal embedded
    in an event-handler attribute -- that needs escJsAttr instead."""
    violations: list[str] = []
    for path in _web_shell_source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _JS_STRING_ESCATTR_SINK.search(line):
                violations.append(f"{path.name}:{lineno}: {line.strip()}")
    assert violations == [], "escAttr() used inside a JS-string-literal sink (should be escJsAttr()): " + "; ".join(
        violations
    )


def test_esc_js_attr_helper_is_defined_exactly_once() -> None:
    """The canonical escJsAttr definition lives in web_shell.py; other
    modules sharing its concatenated scope must not redefine it (the
    exact redeclaration-shadowing hazard that caused this bug)."""
    definitions = []
    for path in _web_shell_source_files():
        text = path.read_text()
        if "function escJsAttr(" in text:
            definitions.append(path.name)
    assert definitions == ["web_shell.py"], f"escJsAttr must be defined exactly once, in web_shell.py: {definitions}"


def test_esc_attr_helper_is_defined_exactly_once_in_the_shared_scope() -> None:
    """web_shell_attachments.py's ATTACHMENT_JS (concatenated into the
    shared scope) must not redefine esc/escAttr -- only its standalone
    ATTACHMENT_LIBRARY_HTML page (a separate document, separate JS scope,
    no onclick sinks) may have its own copy."""
    from polylogue.daemon.web_shell_attachments import ATTACHMENT_JS

    assert "function esc(" not in ATTACHMENT_JS
    assert "function escAttr(" not in ATTACHMENT_JS

    shared_scope_definitions = []
    for path in _web_shell_source_files():
        if path.name == "web_shell_attachments.py":
            continue
        text = path.read_text()
        if "function escAttr(" in text:
            shared_scope_definitions.append(path.name)
    assert shared_scope_definitions == ["web_shell.py"]


# ---------------------------------------------------------------------------
# Best-effort JS execution against the real extracted functions
# ---------------------------------------------------------------------------

_NODE = shutil.which("node")

_HARNESS_TEMPLATE = r"""
%(functions)s

function htmlAttrDecode(s) {
  return s.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'");
}

const payloads = %(payloads_json)s;
const results = [];
let calls;
function probe(arg) { calls.push(arg); }

for (const payload of payloads) {
  calls = [];
  const attr = "probe('message-" + escJsAttr(payload) + "')";
  const decoded = htmlAttrDecode(attr);
  let outcome;
  try {
    const fn = new Function('probe', decoded);
    fn(probe);
    outcome = (calls.length === 1 && calls[0] === ('message-' + payload)) ? 'safe_roundtrip' : 'unsafe_' + JSON.stringify(calls);
  } catch (e) {
    outcome = 'threw:' + e.message;
  }
  results.push({payload: payload, outcome: outcome});
}
console.log(JSON.stringify(results));
"""


def _extract_escaping_functions() -> str:
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    esc_match = re.search(r"function esc\(s\) \{.*?\}", WEB_SHELL_HTML)
    esc_attr_match = re.search(r"function escAttr\(s\) \{.*?\}", WEB_SHELL_HTML)
    esc_js_attr_match = re.search(r"function escJsAttr\(s\) \{.*?\n\}", WEB_SHELL_HTML, re.DOTALL)
    assert esc_match and esc_attr_match and esc_js_attr_match, (
        "could not locate esc/escAttr/escJsAttr in WEB_SHELL_HTML"
    )
    return "\n".join([esc_match.group(0), esc_attr_match.group(0), esc_js_attr_match.group(0)])


_ADVERSARIAL_PAYLOADS = [
    "');document.title='pwned';('",
    "\\",
    "\\'; alert(document.cookie); //",
    "<script>alert(1)</script>",
    "normal-session-id-123",
    "a\"b<c>d&e'f",
    "\\\\\\'",
]


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_esc_js_attr_neutralizes_onclick_breakout_against_the_real_shipped_function() -> None:
    """Execute the ACTUAL escJsAttr extracted from WEB_SHELL_HTML (not a
    hand-copied mirror) through a harness simulating the browser's
    HTML-attribute-value entity decoding step, then parses the decoded
    string as JS. Every adversarial payload must round-trip as exactly
    one call with the original value as its sole argument -- proving no
    extra JS executes and no syntax error masks a partial injection."""
    assert _NODE is not None
    functions_src = _extract_escaping_functions()
    harness = _HARNESS_TEMPLATE % {
        "functions": functions_src,
        "payloads_json": json.dumps(_ADVERSARIAL_PAYLOADS),
    }
    proc = subprocess.run([_NODE, "-e", harness], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    results = json.loads(proc.stdout)
    failures = [r for r in results if r["outcome"] != "safe_roundtrip"]
    assert failures == [], f"escJsAttr failed to neutralize: {failures}"


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_old_html_entity_escaping_would_have_been_exploitable() -> None:
    """Documents WHY escJsAttr exists: proves the classic quote-breakout
    payload defeats plain HTML-entity escaping (esc/the old escAttr) when
    used for a JS-string-in-attribute sink, via the exact decode-then-parse
    mechanism the browser applies. This is a characterization test for the
    vulnerability, not a check on current shipped code."""
    assert _NODE is not None
    harness = r"""
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function oldBrokenEscAttr(s) { return esc(s).replace(/'/g, '&#39;'); }
function htmlAttrDecode(s) {
  return s.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'");
}
const exploit = "');globalThis.__pwned = true;('";
const attr = "probe('message-" + oldBrokenEscAttr(exploit) + "')";
const decoded = htmlAttrDecode(attr);
let pwned = false;
function probe() {}
new Function('probe', decoded)(probe);
console.log(JSON.stringify({pwned: globalThis.__pwned === true}));
"""
    proc = subprocess.run([_NODE, "-e", harness], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    result = json.loads(proc.stdout)
    assert result["pwned"] is True, (
        "expected the characterization payload to prove HTML-entity escaping is exploitable "
        "for a JS-string-in-attribute sink; if this now fails, the vulnerability mechanism "
        "this test documents may no longer apply and the test should be revisited"
    )


# ---------------------------------------------------------------------------
# Raw innerHTML-injection sinks (a second, distinct bug class found during
# this audit): _polyAttachmentLibraryRender (web_shell_attachments.py) and
# its near-identical sibling in web_shell_paste.py built attachment/paste
# row HTML from mime_type/state/origin/title/role/snippet fields with EITHER
# zero escaping (att-origin, pb-origin, pb-role, att-row's state-* class,
# att-state's state label) OR a partial regex escape covering only `<`/`&`
# (name, att-group-title, pb-snippet, pb-group-title) -- missing `>`, `"`,
# `'`, enough to break out of an attribute or, for the fully-unescaped
# fields, inject a raw <script> tag directly via listEl.innerHTML = html.
# No JS-string nesting subtlety here: esc()/escAttr() are correctly
# protective for plain HTML text/attribute contexts, proven above.
# ---------------------------------------------------------------------------


def test_no_partial_lt_amp_only_escape_remains() -> None:
    """Regression guard for the second bug class: a `.replace(/[<&]/g, ...)`
    escaping only `<` and `&` is not a real HTML escaper (misses `>`, `"`,
    `'`) and must never reappear as a substitute for esc()/escAttr()."""
    violations: list[str] = []
    pattern = re.compile(r"\.replace\(/\[<&\]/g")
    for path in _web_shell_source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                violations.append(f"{path.name}:{lineno}: {line.strip()}")
    assert violations == [], f"partial <>&-only escape reappeared: {violations}"


@pytest.mark.parametrize(
    ("filename", "field_snippets"),
    [
        (
            "web_shell_attachments.py",
            [
                "escAttr(it.state || 'unknown')",
                "esc(name)",
                "esc(meta.join(",
                "esc(_polyAttachmentStateLabel(it.state || 'unknown'))",
                "esc(g.title || cid)",
                "esc(g.origin || '')",
            ],
        ),
        (
            "web_shell_paste.py",
            [
                "esc(it.role || '')",
                "esc(it.snippet || '')",
                "esc(g.title || cid)",
                "esc(g.origin || '')",
            ],
        ),
    ],
)
def test_previously_unescaped_row_builder_fields_are_now_wrapped(filename: str, field_snippets: list[str]) -> None:
    """Each field the audit found unescaped or partially escaped in the
    attachment/paste library row builders is now wrapped in esc()/escAttr()
    in the actual shipped source -- not just in a hand-copied test mirror."""
    source = (WEB_SHELL_MODULE_DIR / filename).read_text()
    missing = [snippet for snippet in field_snippets if snippet not in source]
    assert missing == [], f"{filename} missing expected escaping: {missing}"


@pytest.mark.skipif(_NODE is None, reason="node not on PATH (not a declared flake dependency)")
def test_esc_renders_script_tags_and_attribute_breakouts_inert() -> None:
    """AC: attachment/session content with mime_type/origin/meta containing
    quotes, backslashes, angle brackets, and script tags must render inert.
    Runs the real esc()/escAttr() extracted from WEB_SHELL_HTML against the
    literal payloads the bead names, then confirms the output -- if it were
    dropped into innerHTML -- creates no script element and cannot break out
    of a double-quoted HTML attribute."""
    assert _NODE is not None
    from polylogue.daemon.web_shell import WEB_SHELL_HTML

    esc_match = re.search(r"function esc\(s\) \{.*?\}", WEB_SHELL_HTML)
    esc_attr_match = re.search(r"function escAttr\(s\) \{.*?\}", WEB_SHELL_HTML)
    assert esc_match and esc_attr_match

    payloads = [
        "<script>alert(document.cookie)</script>",
        'image/png"><img src=x onerror=alert(1)>',
        "normal/mime-type",
        "a'b\"c<d>e&f\\g",
    ]
    harness = f"""
{esc_match.group(0)}
{esc_attr_match.group(0)}
const payloads = {json.dumps(payloads)};
const results = payloads.map(function(p) {{
  var textCtx = '<span class="att-meta">' + esc(p) + '</span>';
  // Exactly 4 double-quotes expected: class="..." and href="..." each
  // contribute an open+close pair. Any extra/fewer quote means escAttr
  // let a raw `"` through and the payload broke out of the class attribute.
  var attrCtx = '<a class="att-row state-' + escAttr(p) + '" href="/x">';
  var quoteCount = (attrCtx.match(/"/g) || []).length;
  return {{
    payload: p,
    text_has_live_script: /<script[ >]/i.test(textCtx),
    attr_breaks_out: quoteCount !== 4,
  }};
}});
console.log(JSON.stringify(results));
"""
    proc = subprocess.run([_NODE, "-e", harness], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    results = json.loads(proc.stdout)
    for r in results:
        assert r["text_has_live_script"] is False, f"unescaped <script> survived for payload {r['payload']!r}"
        assert r["attr_breaks_out"] is False, f"attribute breakout for payload {r['payload']!r}"
