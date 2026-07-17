"""HTML backend for the provider-neutral semantic transcript document.

Daemon routes attach ``semantic_entries`` produced by the same pure renderer
that the CLI presents as Markdown. This leaf formats those typed documents; it
does not classify raw roles, prose, tool names, or outcomes.
"""

from __future__ import annotations

SEMANTIC_CARD_CSS = r"""
.sem-entries { display: flex; flex-direction: column; gap: 6px; margin: 4px 0; }
.sem-session-entries { margin: 0 10px 8px; }
.sem-cards { display: flex; flex-direction: column; gap: 6px; margin: 4px 0; }
.sem-card {
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--panel-subtle); padding: 6px 10px;
  border-left: 3px solid var(--role-tool);
}
.sem-card.sem-card-shell { border-left-color: var(--accent); }
.sem-card.sem-card-file_read { border-left-color: var(--accent-soft); }
.sem-card.sem-card-file_edit { border-left-color: var(--active); }
.sem-card.sem-card-search { border-left-color: var(--role-user); }
.sem-card.sem-card-web { border-left-color: var(--role-user); }
.sem-card.sem-card-task { border-left-color: var(--warn); }
.sem-card.sem-card-mcp { border-left-color: var(--role-tool); }
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
.sem-source { margin-top: 4px; font-size: 10px; color: var(--text-dim); }
.sem-source summary { cursor: pointer; user-select: none; }
.sem-source code { display: block; margin-top: 2px; white-space: pre-wrap; word-break: break-word; }
.sem-prose { min-width: 0; }
.sem-prose-meta {
  display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 3px;
  font-family: var(--font-mono); font-size: 10px; color: var(--text-dim);
}
.sem-prose-meta .chip { padding: 0 4px; }
.sem-notice {
  border: 1px dashed var(--border-strong); border-radius: var(--radius);
  background: var(--panel-subtle); color: var(--role-thinking);
  padding: 5px 9px; font-size: var(--small);
}
.sem-notice strong { font-family: var(--font-mono); font-weight: 500; }
.sem-notice details { margin-top: 2px; }
.sem-notice code { display: block; white-space: pre-wrap; word-break: break-word; }
"""


SEMANTIC_CARD_JS = r"""
var SEM_CARD_LABEL = {
  shell: 'shell', file_read: 'read', file_edit: 'edit', search: 'search',
  web: 'web', task: 'task', mcp: 'mcp', lineage: 'lineage',
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
  if (typeof card.outcome.is_error === 'boolean') bits.push('is_error=' + card.outcome.is_error);
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
  if (preview.omitted_lines) meta += ', ' + preview.omitted_lines + ' omitted';
  if (preview.omitted_characters) meta += ', ' + preview.omitted_characters + ' chars omitted';
  if (preview.encoding_replacements) meta += ', ' + preview.encoding_replacements + ' replacements';
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

function _polySemSourceBits(source) {
  source = source || {};
  var keys = [
    'session_id', 'provider_family', 'origin', 'message_id', 'block_id', 'block_index',
    'tool_name', 'tool_id', 'attachment_id', 'material_origin', 'occurred_at', 'duration_ms',
    'parent_message_id', 'variant_index', 'is_active_path', 'is_active_leaf', 'inherited_prefix',
    'result_message_id', 'result_block_id', 'result_block_index', 'result_duration_ms',
    'result_material_origin', 'result_inherited_prefix'
  ];
  return keys.filter(function(key) { return source[key] !== undefined && source[key] !== null; })
    .map(function(key) { return key + '=' + String(source[key]); });
}

function _polySemSourceHtml(source) {
  var bits = _polySemSourceBits(source);
  if (!bits.length) return '';
  return '<details class="sem-source"><summary>evidence</summary><code>'
    + esc(bits.join('\n')) + '</code></details>';
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
    +   _polySemSourceHtml(card.source)
    + '</div>';
}

function _polySemProseMetaBits(prose) {
  var bits = [];
  if (prose.block_type) bits.push(prose.block_type);
  if (prose.language) bits.push(prose.language);
  if (prose.material_origin) bits.push('material:' + prose.material_origin);
  if (prose.variant_index !== undefined && prose.variant_index !== null) bits.push('variant:' + prose.variant_index);
  if (prose.is_active_path === true) bits.push('active path');
  if (prose.is_active_leaf === true) bits.push('active leaf');
  if (prose.inherited_prefix === true) bits.push('inherited prefix');
  return bits;
}

function _polySemanticProseHtml(prose, entryIndex) {
  prose = prose || {};
  var type = prose.block_type || '';
  var text = prose.text || '';
  var metaBits = _polySemProseMetaBits(prose);
  var meta = metaBits.length
    ? '<div class="sem-prose-meta">' + metaBits.map(function(bit) {
        return '<span class="chip">' + esc(bit) + '</span>';
      }).join('') + '</div>'
    : '';
  var body;
  if (type === 'thinking' || type === 'reasoning') {
    body = ''
      + '<div class="msg-fold" onclick="togglePolyFold(this)">'
      +   '<span class="fold-arrow"></span><span class="fold-label">[' + esc(type) + ']</span>'
      +   '<span class="fold-meta">typed block</span>'
      + '</div>'
      + '<div class="msg-fold-body"><div class="msg-text">' + esc(text) + '</div></div>';
  } else if (type === 'code') {
    body = _polyCodeBlockHtml({lang: prose.language || 'code', body: text}, entryIndex || 0);
  } else {
    body = text ? '<div class="msg-text">' + esc(text) + '</div>' : '';
  }
  var sourceBits = _polySemSourceBits({
    provider_family: prose.provider_family,
    origin: prose.origin,
    message_id: prose.message_id,
    block_id: prose.block_id,
    block_index: prose.block_index,
    material_origin: prose.material_origin,
    occurred_at: prose.occurred_at,
    duration_ms: prose.duration_ms,
    parent_message_id: prose.parent_message_id,
    variant_index: prose.variant_index,
    is_active_path: prose.is_active_path,
    is_active_leaf: prose.is_active_leaf,
    inherited_prefix: prose.inherited_prefix
  });
  var source = sourceBits.length
    ? '<details class="sem-source"><summary>source</summary><code>' + esc(sourceBits.join('\n')) + '</code></details>'
    : '';
  return '<div class="sem-prose" data-semantic-block-type="' + escAttr(type) + '">'
    + meta + body + source + '</div>';
}

function _polySemanticNoticeHtml(notice) {
  notice = notice || {};
  var count = Number(notice.count || ((notice.sources || []).length));
  var label = notice.kind === 'empty_thinking' ? 'thinking absent' : (notice.kind || 'notice');
  var refs = (notice.sources || []).map(function(source) {
    var coord = source.block_id || ('index:' + source.block_index);
    return 'message:' + source.message_id + '/block:' + coord + '/type:' + source.block_type;
  });
  var detail = refs.length
    ? '<details><summary>' + esc(String(count)) + ' source coordinate' + (count === 1 ? '' : 's')
      + '</summary><code>' + esc(refs.join('\n')) + '</code></details>'
    : '';
  return '<div class="sem-notice" data-notice-kind="' + escAttr(notice.kind || '') + '">'
    + '<strong>' + esc(label) + '</strong> · ' + esc(String(count)) + ' typed block' + (count === 1 ? '' : 's')
    + detail + '</div>';
}

function _polySemanticEntryHtml(entry, entryIndex) {
  if (!entry) return '';
  if (entry.entry_type === 'card' && entry.card) return _polySemanticCardHtml(entry.card);
  if (entry.entry_type === 'prose' && entry.prose) return _polySemanticProseHtml(entry.prose, entryIndex);
  if (entry.entry_type === 'notice' && entry.notice) return _polySemanticNoticeHtml(entry.notice);
  return '';
}

function _polySemanticEntriesHtml(m) {
  if (!m || !Array.isArray(m.semantic_entries)) return '';
  return '<div class="sem-entries">' + m.semantic_entries.map(_polySemanticEntryHtml).join('') + '</div>';
}

function _polySemanticSessionEntriesHtml(c) {
  if (!c || !Array.isArray(c.semantic_entries) || !c.semantic_entries.length) return '';
  return '<div class="sem-entries sem-session-entries">'
    + c.semantic_entries.map(_polySemanticEntryHtml).join('') + '</div>';
}

// Compatibility hook for consumers that still request card-only payloads.
function _polySemanticCardsHtml(m) {
  if (!m || !Array.isArray(m.semantic_cards) || !m.semantic_cards.length) return '';
  return '<div class="sem-cards">' + m.semantic_cards.map(_polySemanticCardHtml).join('') + '</div>';
}
"""

__all__ = ["SEMANTIC_CARD_CSS", "SEMANTIC_CARD_JS"]
