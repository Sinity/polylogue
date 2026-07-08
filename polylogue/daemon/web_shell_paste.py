"""MK3 reader paste-spans slice for the daemon web shell (#1201).

Owns:

- paste-span CSS for inline highlight, diff syntax colors, and
  collapsible diff blocks;
- paste-aware renderer additions used by ``web_shell_reader.py``
  (``_polyRenderPasteSpans``, ``_polyDetectDiff``, ``copyTypedOnly`` /
  ``copyPasteOnly``);
- the ``/p`` paste-browser page HTML and bootstrap JS, which fetches
  ``/api/paste-browser`` and renders a grouped list of paste messages
  with anchor links back to the source session.

The substrate exposes paste evidence per message (#1313) but does not yet
expose per-character paste-span offsets (those land in #839/#864). This
slice ships **whole-message fallback** driven by ``has_paste_evidence`` plus
**heuristic diff-span detection** (unified-diff format), so the visual
acceptance criteria can be met today without waiting on substrate work.

Per-message envelope contract (in ``daemon/http.py:_do_get_session``):

- ``has_paste_evidence`` — true when the message carries paste evidence.
- ``paste_spans`` — new list of ``{kind, start, end, confidence}``
  records. Empty unless the server-side heuristic detects a diff
  embedded in the message text; see ``detect_paste_spans``.

The renderer treats the envelope as the source of truth: when
``paste_spans`` is empty but ``has_paste_evidence`` is true, the whole message
gets a paste banner; when ``paste_spans`` contains entries, those
ranges are highlighted inline and (for diffs) folded with syntax color.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Server-side paste-span detection
# ---------------------------------------------------------------------------

# Unified diff fingerprints. A message qualifies as containing a diff
# paste when at least one ``@@ -... +... @@`` hunk header is present, or
# when both ``--- `` and ``+++ `` file headers appear adjacent. This
# matches the output of ``git diff`` / ``diff -u`` and is conservative
# enough that prose containing "---" alone is not flagged.
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)
_DIFF_FILE_HEADER_RE = re.compile(r"^--- [^\n]+\n\+\+\+ [^\n]+", re.MULTILINE)

# Minimum collapsed length: pastes longer than this default to folded.
DIFF_COLLAPSE_THRESHOLD = 500


@dataclass(frozen=True)
class PasteSpan:
    """A detected paste range within a message body.

    ``kind`` is one of ``"diff"`` (heuristic unified-diff match) or
    ``"text"`` (generic paste — currently unused server-side; reserved
    for #839/#864).
    """

    kind: str
    start: int
    end: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


def detect_paste_spans(text: str) -> list[PasteSpan]:
    """Server-side paste-span detector.

    Currently detects only one shape — embedded unified-diff blocks —
    because that is the only span shape the substrate does not yet
    expose elsewhere. Returns one span per contiguous diff block.

    The detector is deliberately conservative: it requires either a
    hunk header (``@@ -... +... @@``) or a paired file header
    (``--- foo`` / ``+++ bar``) before claiming a region. The span
    starts at the earliest qualifying marker and extends to the end of
    the contiguous diff lines (lines beginning with ``+``, ``-``,
    ``@@``, ``---``, ``+++``, or blank lines between them).
    """

    if not text:
        return []
    # Quick reject: neither fingerprint present.
    if not _DIFF_HUNK_RE.search(text) and not _DIFF_FILE_HEADER_RE.search(text):
        return []

    spans: list[PasteSpan] = []
    lines = text.split("\n")
    # Precompute line offsets to convert (line index → absolute offset).
    line_offsets: list[int] = []
    cursor = 0
    for line in lines:
        line_offsets.append(cursor)
        cursor += len(line) + 1  # +1 for the newline

    def _is_diff_line(line: str) -> bool:
        if not line:
            return False
        return line.startswith(("@@", "---", "+++", "+", "-", " ", "\\"))

    i = 0
    while i < len(lines):
        line = lines[i]
        is_hunk = _DIFF_HUNK_RE.match(line) is not None
        is_file_pair = False
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            is_file_pair = True
        if not (is_hunk or is_file_pair):
            i += 1
            continue
        # Walk forward gathering contiguous diff-shaped lines. Allow at
        # most one blank line gap before terminating — diffs can have
        # blank context lines but not arbitrary prose interleaved.
        start_line = i
        end_line = i
        blank_run = 0
        j = i
        while j < len(lines):
            current = lines[j]
            if _is_diff_line(current):
                end_line = j
                blank_run = 0
            elif current.strip() == "":
                blank_run += 1
                if blank_run > 1:
                    break
            else:
                break
            j += 1
        start_offset = line_offsets[start_line]
        # End is exclusive — point one past the last newline of the
        # final diff line so callers can slice ``text[start:end]``.
        last_line_end = line_offsets[end_line] + len(lines[end_line])
        spans.append(
            PasteSpan(
                kind="diff",
                start=start_offset,
                end=last_line_end,
                confidence=0.95 if is_hunk else 0.75,
            )
        )
        i = end_line + 1
    return spans


def envelope_paste_spans(text: str | None, *, has_paste: bool) -> list[dict[str, object]]:
    """Compute the ``paste_spans`` value for a message envelope.

    Returns an empty list when no spans are detected — the renderer
    falls back to whole-message paste banner whenever the internal
    ``has_paste`` storage flag is
    set and ``paste_spans`` is empty.
    """

    if not text:
        return []
    spans = detect_paste_spans(text)
    return [span.to_dict() for span in spans]


# ---------------------------------------------------------------------------
# Paste browser route plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PasteBrowserEntry:
    """One row in the paste-browser listing."""

    session_id: str
    session_title: str
    origin: str | None
    message_id: str
    message_anchor: str
    role: str
    timestamp: str | None
    word_count: int
    snippet: str
    paste_spans: list[dict[str, object]]
    has_diff: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_title": self.session_title,
            "origin": self.origin,
            "message_id": self.message_id,
            "message_anchor": self.message_anchor,
            "role": self.role,
            "timestamp": self.timestamp,
            "word_count": self.word_count,
            "snippet": self.snippet,
            "paste_spans": self.paste_spans,
            "has_diff": self.has_diff,
        }


def build_paste_browser_payload(entries: Iterable[PasteBrowserEntry], *, total: int) -> dict[str, object]:
    """Wrap a sequence of paste-browser entries in the envelope shape
    consumed by the client. Grouping by session happens
    client-side so the server contract stays a flat list."""

    items = [entry.to_dict() for entry in entries]
    return {"items": items, "total": total}


def snippet_for_paste(text: str, spans: list[dict[str, object]], *, limit: int = 160) -> str:
    """Return a single-line snippet of the first paste span, or the
    first line of the message when no spans were detected."""

    if spans:
        first = spans[0]
        start_val = first.get("start", 0)
        end_val = first.get("end", min(int(start_val) + limit, len(text)) if isinstance(start_val, int) else len(text))
        start = int(start_val) if isinstance(start_val, (int, float)) else 0
        end = int(end_val) if isinstance(end_val, (int, float)) else len(text)
        body = text[start:end]
    else:
        body = text
    first_line = body.strip().split("\n", 1)[0] if body else ""
    if len(first_line) > limit:
        return first_line[:limit] + "\u2026"
    return first_line


# ---------------------------------------------------------------------------
# Client-side: paste-span CSS, JS helpers, paste-browser page
# ---------------------------------------------------------------------------

PASTE_CSS = r"""
.msg-paste-banner {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 1px 6px; margin: 2px 0 4px 0;
  border: 1px dashed var(--text-dim); border-radius: 3px;
  color: var(--text-muted);
  font-family: var(--font-mono); font-size: 10px; line-height: 1.4;
}
.msg-paste-banner .pb-label { color: var(--accent); }

.msg-paste-span {
  background: var(--panel-subtle);
  border-left: 2px solid var(--accent);
  padding: 0 4px;
  border-radius: 2px;
}
.msg-paste-span::before {
  content: '\1F4CB';
  color: var(--text-dim); margin-right: 4px; font-size: 10px;
}

/* Diff syntax colors — minimal palette tuned to the reader theme. */
.msg-diff-fold {
  margin: 4px 0; border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--panel-subtle);
}
.msg-diff-fold .fold-bar {
  display: flex; gap: 6px; padding: 3px 8px; cursor: pointer;
  color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;
  user-select: none; border-bottom: 1px solid transparent;
}
.msg-diff-fold .fold-bar:hover { color: var(--text); }
.msg-diff-fold.open .fold-bar { border-bottom-color: var(--border); }
.msg-diff-fold .fold-arrow { color: var(--text-dim); width: 10px; display: inline-block; }
.msg-diff-fold.open .fold-arrow::before { content: '\25BE'; }
.msg-diff-fold:not(.open) .fold-arrow::before { content: '\25B8'; }
.msg-diff-fold .fold-label { color: var(--accent); }
.msg-diff-fold .fold-meta { color: var(--text-dim); margin-left: auto; }
.msg-diff-fold .diff-body { display: none; }
.msg-diff-fold.open .diff-body { display: block; }
.msg-diff-fold pre {
  margin: 0; padding: 8px 12px; font-family: var(--font-mono); font-size: var(--code);
  white-space: pre-wrap; word-break: break-word;
}
.msg-diff-fold .diff-line { display: block; }
.msg-diff-fold .diff-line.add { color: #5fb350; background: rgba(95,179,80,0.08); }
.msg-diff-fold .diff-line.del { color: #d05050; background: rgba(208,80,80,0.08); }
.msg-diff-fold .diff-line.hunk { color: var(--accent); font-weight: 600; }
.msg-diff-fold .diff-line.file { color: var(--text-muted); }

.msg-actions .act-btn[disabled] {
  opacity: 0.4; cursor: not-allowed;
}
"""

# JS injected into the reader bootstrap. Exports:
#
#   - ``_polyDetectDiffSpans(text)`` — fallback client-side detector
#     mirroring ``detect_paste_spans`` for messages the server
#     envelope did not annotate (older endpoints, smoke fixtures);
#   - ``_polyRenderPasteBody(m)`` — replaces ``_polyBodyHtml`` when
#     paste spans are present; renders diff folds and span highlights;
#   - ``copyTypedOnly(messageId)`` / ``copyPasteOnly(messageId)`` —
#     copy menu actions that respect paste spans;
#   - ``initPasteBrowser()`` — bootstrap for the ``/p`` route page.
PASTE_JS = r"""
// MK3 reader paste-spans slice (#1201).

var _POLY_DIFF_COLLAPSE = 500;

function _polyDetectDiffSpans(text) {
  // Fallback client-side detector mirroring the server-side one.
  if (!text) return [];
  var lines = text.split('\n');
  var spans = [];
  var i = 0;
  var offsets = [];
  var off = 0;
  for (var k = 0; k < lines.length; k++) { offsets.push(off); off += lines[k].length + 1; }
  function isDiffLine(l) {
    if (!l) return false;
    return /^(@@|---|\+\+\+|\+|-| |\\)/.test(l);
  }
  while (i < lines.length) {
    var line = lines[i];
    var isHunk = /^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@/.test(line);
    var isFilePair = line.indexOf('--- ') === 0 && i + 1 < lines.length && lines[i+1].indexOf('+++ ') === 0;
    if (!isHunk && !isFilePair) { i++; continue; }
    var startLine = i;
    var endLine = i;
    var blankRun = 0;
    var j = i;
    while (j < lines.length) {
      var cur = lines[j];
      if (isDiffLine(cur)) { endLine = j; blankRun = 0; }
      else if (cur.trim() === '') { blankRun++; if (blankRun > 1) break; }
      else break;
      j++;
    }
    var startOff = offsets[startLine];
    var endOff = offsets[endLine] + lines[endLine].length;
    spans.push({kind: 'diff', start: startOff, end: endOff,
                confidence: isHunk ? 0.95 : 0.75});
    i = endLine + 1;
  }
  return spans;
}

function _polyEffectivePasteSpans(m) {
  // Prefer server envelope; fall back to client detection so old
  // payloads still render correctly.
  if (m && Array.isArray(m.paste_spans) && m.paste_spans.length) return m.paste_spans;
  if (m && m.has_paste_evidence) {
    var detected = _polyDetectDiffSpans(m.text || '');
    if (detected.length) return detected;
  }
  return [];
}

function _polyClassifyDiffLine(line) {
  if (!line) return '';
  if (/^@@/.test(line)) return 'hunk';
  if (line.indexOf('--- ') === 0 || line.indexOf('+++ ') === 0) return 'file';
  if (line.charAt(0) === '+') return 'add';
  if (line.charAt(0) === '-') return 'del';
  return '';
}

function _polyDiffFoldHtml(body) {
  var lineCount = body.split('\n').length;
  var collapsed = body.length > _POLY_DIFF_COLLAPSE;
  var classNames = 'msg-diff-fold' + (collapsed ? '' : ' open');
  var addCount = 0, delCount = 0;
  var lines = body.split('\n');
  var linesHtml = lines.map(function(line) {
    var cls = _polyClassifyDiffLine(line);
    if (cls === 'add') addCount++;
    if (cls === 'del') delCount++;
    return '<span class="diff-line ' + cls + '">' + esc(line) + '</span>';
  }).join('\n');
  var meta = lineCount + ' line' + (lineCount === 1 ? '' : 's')
    + ' \u00B7 +' + addCount + ' / -' + delCount;
  return ''
    + '<div class="' + classNames + '">'
    +   '<div class="fold-bar" onclick="toggleCodeFold(this.parentElement)">'
    +     '<span class="fold-arrow"></span>'
    +     '<span class="fold-label">diff</span>'
    +     '<span class="fold-meta">' + meta + '</span>'
    +   '</div>'
    +   '<div class="diff-body"><pre>' + linesHtml + '</pre></div>'
    + '</div>';
}

function _polyRenderPasteBody(m) {
  // Renders message text with paste spans highlighted. Each ``diff``
  // span becomes a foldable diff block; remaining text runs render as
  // plain text. Non-diff spans (kind === 'text') render as inline
  // highlighted ranges with the ``msg-paste-span`` class.
  var text = m.text || '';
  var spans = _polyEffectivePasteSpans(m);
  if (!spans.length) return _polyBodyHtml(m);
  // Sort spans by start to walk linearly; clamp to text length.
  var ordered = spans.slice().sort(function(a, b) { return a.start - b.start; });
  var out = '';
  var cursor = 0;
  ordered.forEach(function(span) {
    var s = Math.max(0, Math.min(text.length, span.start | 0));
    var e = Math.max(s, Math.min(text.length, span.end | 0));
    if (s > cursor) {
      var pre = text.substring(cursor, s);
      if (pre.trim().length) out += '<div class="msg-text">' + esc(pre) + '</div>';
    }
    var body = text.substring(s, e);
    if (span.kind === 'diff') {
      out += _polyDiffFoldHtml(body);
    } else {
      out += '<div class="msg-text"><span class="msg-paste-span">' + esc(body) + '</span></div>';
    }
    cursor = e;
  });
  if (cursor < text.length) {
    var tail = text.substring(cursor);
    if (tail.trim().length) out += '<div class="msg-text">' + esc(tail) + '</div>';
  }
  return out;
}

function _polyHasPaste(m) {
  if (m && m.has_paste_evidence) return true;
  return _polyEffectivePasteSpans(m).length > 0;
}

function _polyPasteBannerHtml(m) {
  // Whole-message indication: used when paste evidence is set but no
  // per-span data was produced (substrate-only signal).
  var spans = _polyEffectivePasteSpans(m);
  if (spans.length) return '';
  if (!_polyHasPaste(m)) return '';
  return ''
    + '<div class="msg-paste-banner">'
    +   '<span class="pb-label">paste</span>'
    +   '<span>whole message flagged as paste</span>'
    + '</div>';
}

function _polyTypedOnlyText(m) {
  var text = m.text || '';
  var spans = _polyEffectivePasteSpans(m);
  if (!spans.length) return text;
  var ordered = spans.slice().sort(function(a, b) { return a.start - b.start; });
  var out = '';
  var cursor = 0;
  ordered.forEach(function(span) {
    if (span.start > cursor) out += text.substring(cursor, span.start);
    cursor = Math.max(cursor, span.end);
  });
  if (cursor < text.length) out += text.substring(cursor);
  return out;
}

function _polyPasteOnlyText(m) {
  var text = m.text || '';
  var spans = _polyEffectivePasteSpans(m);
  if (!spans.length) return '';
  return spans
    .slice()
    .sort(function(a, b) { return a.start - b.start; })
    .map(function(s) { return text.substring(s.start, s.end); })
    .join('\n\n');
}

function copyTypedOnly(messageId) {
  var msgs = (state.selected && state.selected.messages) || [];
  var hit = null;
  for (var i = 0; i < msgs.length; i++) {
    if (String(msgs[i].id) === String(messageId)) { hit = msgs[i]; break; }
  }
  if (!hit) return;
  var btn = document.querySelector('.msg-block[data-msg-id="'
    + (messageId || '').replace(/"/g, '\\"') + '"] .act-btn[data-act="copy-typed"]');
  _polyClipboardWrite(_polyTypedOnlyText(hit), btn);
}

function copyPasteOnly(messageId) {
  var msgs = (state.selected && state.selected.messages) || [];
  var hit = null;
  for (var i = 0; i < msgs.length; i++) {
    if (String(msgs[i].id) === String(messageId)) { hit = msgs[i]; break; }
  }
  if (!hit) return;
  var btn = document.querySelector('.msg-block[data-msg-id="'
    + (messageId || '').replace(/"/g, '\\"') + '"] .act-btn[data-act="copy-paste"]');
  _polyClipboardWrite(_polyPasteOnlyText(hit), btn);
}

function _polyCopyActionRailHtml(m) {
  // Extra copy-menu actions appended to the per-message rail when
  // paste spans are present (or paste evidence is set with whole-message
  // fallback). When spans are absent the actions render disabled so
  // the contract surface is stable.
  var hasSpans = _polyEffectivePasteSpans(m).length > 0;
  var disabledAttr = hasSpans ? '' : 'disabled aria-disabled="true" data-disabled-reason="no_paste_spans"';
  var msgId = escJsAttr(String(m.id || ''));
  return ''
    + '<button class="act-btn" data-act="copy-typed" '
    +         'title="Copy typed only (excludes paste spans)" '
    +         (hasSpans ? 'onclick="copyTypedOnly(\'' + msgId + '\')"' : '') + ' '
    +         disabledAttr + '>typed</button>'
    + '<button class="act-btn" data-act="copy-paste" '
    +         'title="Copy paste only (paste spans only)" '
    +         (hasSpans ? 'onclick="copyPasteOnly(\'' + msgId + '\')"' : '') + ' '
    +         disabledAttr + '>paste</button>';
}

// ---------------------------------------------------------------------
// Paste browser bootstrap (served as a separate page at /p).
// ---------------------------------------------------------------------

function _polyPasteBrowserRender(items) {
  var listEl = document.getElementById('paste-list');
  var emptyEl = document.getElementById('paste-empty');
  if (!listEl || !emptyEl) return;
  if (!items || !items.length) {
    listEl.innerHTML = '';
    emptyEl.style.display = 'block';
    emptyEl.textContent = 'No paste-flagged messages in the archive.';
    return;
  }
  emptyEl.style.display = 'none';
  // Group by session.
  var groups = {};
  var order = [];
  items.forEach(function(item) {
    var cid = item.session_id;
    if (!groups[cid]) { groups[cid] = {title: item.session_title, origin: item.origin, items: []}; order.push(cid); }
    groups[cid].items.push(item);
  });
  var html = order.map(function(cid) {
    var g = groups[cid];
    var rows = g.items.map(function(it) {
      var href = '/s/' + encodeURIComponent(cid) + '#' + encodeURIComponent(it.message_anchor);
      var badges = '';
      if (it.has_diff) badges += '<span class="pb-badge pb-diff">diff</span>';
      var spanCount = (it.paste_spans || []).length;
      if (spanCount > 1) badges += '<span class="pb-badge">' + spanCount + ' spans</span>';
      return ''
        + '<a class="pb-row" href="' + href + '">'
        +   '<span class="pb-role">' + esc(it.role || '') + '</span>'
        +   '<span class="pb-snippet">' + esc(it.snippet || '') + '</span>'
        +   '<span class="pb-meta">' + (it.word_count || 0) + ' words ' + badges + '</span>'
        + '</a>';
    }).join('');
    return ''
      + '<div class="pb-group">'
      +   '<div class="pb-group-title">'
      +     '<a href="/s/' + encodeURIComponent(cid) + '">'
      +     esc(g.title || cid)
      +     '</a>'
      +     '<span class="pb-origin">' + esc(g.origin || '') + '</span>'
      +   '</div>'
      +   rows
      + '</div>';
  }).join('');
  listEl.innerHTML = html;
}

function initPasteBrowser() {
  var listEl = document.getElementById('paste-list');
  var emptyEl = document.getElementById('paste-empty');
  if (!listEl || !emptyEl) return;
  emptyEl.style.display = 'block';
  emptyEl.textContent = 'Loading paste-flagged messages\u2026';
  fetch('/api/paste-browser?limit=200').then(function(r) {
    return r.json();
  }).then(function(payload) {
    _polyPasteBrowserRender(payload.items || []);
  }).catch(function(err) {
    emptyEl.textContent = 'Failed to load paste browser: ' + (err && err.message ? err.message : err);
  });
}
"""

PASTE_BROWSER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polylogue \u00B7 Paste Browser</title>
<style>
:root {
  --bg: #15171a; --panel: #1d2025; --panel-subtle: #23262c;
  --border: #2c2f36; --text: #d6d9de; --text-muted: #8b919b;
  --text-dim: #5a5f68; --accent: #5fb3b3; --radius: 4px;
  --font-mono: ui-monospace,Menlo,Consolas,monospace;
  --font-ui: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--font-ui); font-size: 13px; }
header { padding: 12px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: baseline; gap: 16px; }
header h1 { margin: 0; font-size: 16px; font-weight: 600; }
header a { color: var(--accent); text-decoration: none; }
header a:hover { text-decoration: underline; }
main { padding: 16px 24px; max-width: 1100px; }
#paste-empty { color: var(--text-muted); padding: 24px 0; }
.pb-group { margin-bottom: 24px; }
.pb-group-title {
  display: flex; align-items: baseline; gap: 10px;
  padding: 6px 8px; border-bottom: 1px solid var(--border);
  font-weight: 600;
}
.pb-group-title a { color: var(--text); text-decoration: none; }
.pb-group-title a:hover { color: var(--accent); }
.pb-origin { color: var(--text-dim); font-weight: 400; font-size: 11px; font-family: var(--font-mono); }
.pb-row {
  display: grid; grid-template-columns: 80px 1fr 220px;
  gap: 12px; padding: 6px 8px; align-items: center;
  border-bottom: 1px solid var(--panel-subtle);
  color: var(--text); text-decoration: none;
}
.pb-row:hover { background: var(--panel-subtle); }
.pb-role { color: var(--text-muted); font-family: var(--font-mono); font-size: 11px; }
.pb-snippet { font-family: var(--font-mono); font-size: 12px; color: var(--text);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pb-meta { color: var(--text-dim); font-size: 11px; text-align: right; }
.pb-badge {
  display: inline-block; margin-left: 6px; padding: 1px 6px;
  border: 1px solid var(--border); border-radius: 3px;
  color: var(--text-muted); font-family: var(--font-mono); font-size: 10px;
}
.pb-badge.pb-diff { color: var(--accent); border-color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Paste browser</h1>
  <a href="/">\u2190 Back to reader</a>
</header>
<main>
  <div id="paste-empty"></div>
  <div id="paste-list"></div>
</main>
<script>
__PASTE_BROWSER_JS__
initPasteBrowser();
</script>
</body>
</html>
"""


def render_paste_browser_page() -> str:
    """Return the standalone HTML served at ``/p``."""

    return PASTE_BROWSER_HTML.replace("__PASTE_BROWSER_JS__", PASTE_JS)


__all__ = [
    "DIFF_COLLAPSE_THRESHOLD",
    "PASTE_CSS",
    "PASTE_JS",
    "PasteBrowserEntry",
    "PasteSpan",
    "build_paste_browser_payload",
    "detect_paste_spans",
    "envelope_paste_spans",
    "render_paste_browser_page",
    "snippet_for_paste",
]
