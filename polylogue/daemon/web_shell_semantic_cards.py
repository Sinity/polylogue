"""Web reader rendering for provider-neutral semantic transcript cards (#ap7).

The daemon session-detail routes (``daemon/http.py``,
``_do_get_session``/``_do_archive_get_session``) attach a ``semantic_cards``
array (and a ``semantic_card_suppressed`` flag) to every message payload,
built by :mod:`polylogue.rendering.semantic_card_placement` from the exact
same :class:`polylogue.rendering.semantic_card_models.SemanticCard` registry
the CLI renders to Markdown (``cli/messages.py`` /
``rendering/semantic_markdown.py``). This module is the HTML backend for that
one shared registry: it turns ``SemanticCard.to_document()`` JSON straight
into DOM, with no independent tool-classification logic of its own.

Follows the existing reader-slice pattern (``web_shell_reader.py``,
``web_shell_attachments.py``): a CSS block, a JS block installed as a
``_polyXxxHtml`` hook that ``renderMessageBlocks`` calls when present, wired
into ``web_shell.py`` through ``__SEMANTIC_CARD_CSS__``/``__SEMANTIC_CARD_JS__``
placeholders.
"""

from __future__ import annotations

SEMANTIC_CARD_CSS = r"""
.sem-cards { display: flex; flex-direction: column; gap: 6px; margin: 4px 0; }
.sem-card {
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--panel-subtle); padding: 6px 10px;
  border-left: 3px solid var(--role-tool);
}
.sem-card.sem-card-shell { border-left-color: var(--accent); }
.sem-card.sem-card-file_edit { border-left-color: var(--active); }
.sem-card.sem-card-task { border-left-color: var(--warn); }
.sem-card.sem-card-lineage { border-left-color: var(--text-dim); }
.sem-card.sem-card-attachment { border-left-color: var(--text-dim); }
.sem-card.sem-card-fallback { border-left-color: var(--border-strong); }
.sem-card-header {
  display: flex; align-items: center; gap: 6px; font-family: var(--font-mono);
  font-size: 11px; color: var(--text-muted);
}
.sem-card-kind {
  text-transform: uppercase; letter-spacing: 0.04em; color: var(--accent);
  font-size: 10px;
}
.sem-card-title { color: var(--text); font-weight: 500; }
.sem-outcome {
  margin-left: auto; padding: 0 6px; border-radius: 3px; font-size: 10px;
  border: 1px solid var(--border); color: var(--text-dim);
}
.sem-outcome.state-succeeded { color: var(--ok); border-color: var(--ok); background: var(--ok-bg); }
.sem-outcome.state-failed { color: var(--err); border-color: var(--err); background: var(--err-bg); }
.sem-outcome.state-unknown { color: var(--text-dim); border-color: var(--border); }
.sem-card-anchor {
  color: var(--text-dim); cursor: pointer; user-select: none; padding: 0 2px;
}
.sem-card-anchor:hover { color: var(--accent); }
.sem-card-summary {
  font-family: var(--font-mono); font-size: var(--code); color: var(--text);
  margin: 4px 0; white-space: pre-wrap; word-break: break-word;
}
.sem-card-fields { display: flex; flex-direction: column; gap: 2px; margin: 4px 0; }
.sem-card-field { display: flex; gap: 6px; font-size: var(--small); }
.sem-field-label { color: var(--text-dim); min-width: 64px; }
.sem-field-value {
  font-family: var(--font-mono); color: var(--text); white-space: pre-wrap;
  word-break: break-word;
}
.sem-card-preview { margin-top: 4px; }
.sem-card-preview .code-body pre {
  margin: 0; padding: 6px 10px; font-family: var(--font-mono); font-size: var(--code);
  white-space: pre-wrap; word-break: break-word; color: var(--text);
}
.sem-diff .diff-add { color: var(--ok); }
.sem-diff .diff-del { color: var(--err); }
.sem-diff .diff-hunk { color: var(--accent); }
.sem-diff .diff-ctx { color: var(--text-dim); }
.sem-card-caveats {
  margin: 4px 0 0; padding-left: 16px; font-size: var(--small); color: var(--text-dim);
}
.sem-card-caveats li { margin: 1px 0; }
"""


# Drop-in renderer for the ``renderMessageBlocks`` semantic-card hooks.
# Exposes ``_polySemanticCardsHtml(m)`` (the public hook name the reader
# slice looks up) plus its private helpers.
SEMANTIC_CARD_JS = r"""
var SEM_CARD_LABEL = {
  shell: 'shell', file_edit: 'edit', task: 'task', lineage: 'lineage',
  attachment: 'attachment', fallback: 'tool'
};

function _polyDiffLinesHtml(text) {
  return (text || '').split('\n').map(function(line) {
    var cls = 'diff-ctx';
    if (line.charAt(0) === '+' && line.substring(0, 3) !== '+++') cls = 'diff-add';
    else if (line.charAt(0) === '-' && line.substring(0, 3) !== '---') cls = 'diff-del';
    else if (line.charAt(0) === '@') cls = 'diff-hunk';
    return '<span class="' + cls + '">' + esc(line) + '</span>';
  }).join('\n');
}

function _polySemCardOutcomeHtml(card) {
  if (!card.outcome) return '';
  var state = card.outcome.state || 'unknown';
  var label = state === 'succeeded' ? 'ok' : (state === 'failed' ? 'FAILED' : 'unknown');
  var bits = [];
  if (typeof card.outcome.exit_code === 'number') bits.push('exit ' + card.outcome.exit_code);
  var title = bits.length ? (label + ' (' + bits.join(', ') + ')') : label;
  return '<span class="sem-outcome state-' + esc(state) + '" title="' + escAttr(title) + '">'
    + esc(label) + '</span>';
}

function _polySemCardFieldsHtml(card) {
  if (!card.fields || !card.fields.length) return '';
  return '<div class="sem-card-fields">' + card.fields.map(function(f) {
    return '<div class="sem-card-field"><span class="sem-field-label">' + esc(f.label) + '</span>'
      + '<code class="sem-field-value">' + esc(f.value) + '</code></div>';
  }).join('') + '</div>';
}

function _polySemCardPreviewHtml(preview) {
  var lines = preview.line_count || 0;
  var meta = lines + ' line' + (lines === 1 ? '' : 's');
  if (preview.truncated) meta += ', truncated';
  var body = preview.kind === 'diff'
    ? '<pre class="sem-diff">' + _polyDiffLinesHtml(preview.text) + '</pre>'
    : '<pre>' + esc(preview.text || '') + '</pre>';
  return ''
    + '<div class="msg-code-fold sem-card-preview" data-preview-kind="' + escAttr(preview.kind || '') + '">'
    +   '<div class="fold-bar" onclick="toggleCodeFold(this.parentElement)">'
    +     '<span class="fold-arrow"></span>'
    +     '<span class="fold-label" style="color:var(--accent)">' + esc(preview.kind || 'preview') + '</span>'
    +     '<span class="fold-meta" style="color:var(--text-dim);margin-left:auto">' + esc(meta) + '</span>'
    +   '</div>'
    +   '<div class="code-body">' + body + '</div>'
    + '</div>';
}

function _polySemCardCaveatsHtml(card) {
  if (!card.caveats || !card.caveats.length) return '';
  return '<ul class="sem-card-caveats">' + card.caveats.map(function(c) {
    return '<li>' + esc(c) + '</li>';
  }).join('') + '</ul>';
}

function _polySemCardAnchor(card) {
  var src = card.source || {};
  var token = src.block_id || src.result_block_id
    || (src.message_id ? (src.message_id + ':' + (src.block_index != null ? src.block_index : '0')) : null);
  return token ? 'card-' + String(token).replace(/[^a-zA-Z0-9_:.-]/g, '_') : '';
}

function _polyCopySemanticCardAnchor(anchor, btn) {
  if (!anchor) return;
  var url = window.location.origin + window.location.pathname + '#' + anchor;
  _polyClipboardWrite(url, btn);
}

function _polySemanticCardHtml(card) {
  var kind = card.kind || 'fallback';
  var label = SEM_CARD_LABEL[kind] || kind;
  var anchor = _polySemCardAnchor(card);
  var hasSummaryField = (card.fields || []).some(function(f) { return f.value === card.summary; });
  var summaryHtml = (card.summary && !hasSummaryField)
    ? '<div class="sem-card-summary">' + esc(card.summary) + '</div>' : '';
  var previewsHtml = (card.previews || []).map(_polySemCardPreviewHtml).join('');
  return ''
    + '<div class="sem-card sem-card-' + esc(kind) + '"' + (anchor ? ' id="' + escAttr(anchor) + '"' : '') + '>'
    +   '<div class="sem-card-header">'
    +     '<span class="sem-card-kind">' + esc(label) + '</span>'
    +     '<span class="sem-card-title">' + esc(card.title || '') + '</span>'
    +     _polySemCardOutcomeHtml(card)
    +     (anchor
            ? '<span class="sem-card-anchor" title="Copy card link" '
              + 'onclick="_polyCopySemanticCardAnchor(\'' + escJsAttr(anchor) + '\', this)">#</span>'
            : '')
    +   '</div>'
    +   summaryHtml
    +   _polySemCardFieldsHtml(card)
    +   previewsHtml
    +   _polySemCardCaveatsHtml(card)
    + '</div>';
}

// Exposed under the public hook name renderMessageBlocks looks up.
function _polySemanticCardsHtml(m) {
  if (!m || !Array.isArray(m.semantic_cards) || !m.semantic_cards.length) return '';
  return '<div class="sem-cards">' + m.semantic_cards.map(_polySemanticCardHtml).join('') + '</div>';
}
"""

__all__ = ["SEMANTIC_CARD_CSS", "SEMANTIC_CARD_JS"]
