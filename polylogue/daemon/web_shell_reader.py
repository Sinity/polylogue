"""MK3 reader message-card slice for the daemon web shell (#1202).

Owns the per-message action rail, the default fold policies for
tool-call / tool-output / thinking / code blocks, and the
keyboard-shortcut handler that lets an operator drive the reader from
the keyboard alone.

The slice is rendered into ``web_shell.WEB_SHELL_HTML`` via three
``__READER_*__`` placeholders so the bulk of the reader HTML stays in
one file and the message-card logic stays grouped here.
"""

from __future__ import annotations

# Per-message-card visuals: a thin right-aligned rail of icon buttons,
# fold/unfold affordances for tool/thinking/code blocks, and a focus
# outline for the message currently driven by the keyboard handler.
READER_CSS = r"""
.msg-block { position: relative; }
.msg-block.focused { background: var(--bg-raised); box-shadow: inset 3px 0 0 var(--accent); }
.msg-actions { display: flex; gap: 4px; margin-left: auto; }
.msg-actions .act-btn {
  background: var(--panel-elevated); border: 1px solid var(--border);
  color: var(--text-muted); padding: 1px 6px; border-radius: 3px;
  font-family: var(--font-ui); font-size: 10px; line-height: 1.4;
  cursor: pointer; user-select: none; text-decoration: none;
}
.msg-actions .act-btn:hover { color: var(--text); border-color: var(--text-dim); }
.msg-actions .act-btn.flash { color: var(--ok); border-color: var(--ok); }
.msg-fold {
  display: flex; align-items: center; gap: 6px; padding: 4px 8px; margin: 4px 0;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--panel-subtle); color: var(--text-muted);
  font-family: var(--font-mono); font-size: var(--small); cursor: pointer;
  user-select: none;
}
.msg-fold:hover { color: var(--text); border-color: var(--text-dim); }
.msg-fold .fold-arrow { color: var(--text-dim); width: 10px; display: inline-block; }
.msg-fold .fold-label { color: var(--accent); }
.msg-fold .fold-meta { color: var(--text-dim); margin-left: auto; }
.msg-fold-body { display: none; }
.msg-fold.open .fold-arrow::before { content: '\25BE'; }
.msg-fold:not(.open) .fold-arrow::before { content: '\25B8'; }
.msg-fold.open + .msg-fold-body { display: block; }
.msg-code-fold {
  margin: 4px 0; border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--panel-subtle);
}
.msg-code-fold .fold-bar {
  display: flex; gap: 6px; padding: 3px 8px; cursor: pointer;
  color: var(--text-muted); font-family: var(--font-mono); font-size: 11px;
  user-select: none; border-bottom: 1px solid transparent;
}
.msg-code-fold .fold-bar:hover { color: var(--text); }
.msg-code-fold.open .fold-bar { border-bottom-color: var(--border); }
.msg-code-fold .fold-arrow { color: var(--text-dim); width: 10px; display: inline-block; }
.msg-code-fold.open .fold-arrow::before { content: '\25BE'; }
.msg-code-fold:not(.open) .fold-arrow::before { content: '\25B8'; }
.msg-code-fold .code-body { display: none; }
.msg-code-fold.open .code-body { display: block; }
.msg-code-fold .code-body pre {
  margin: 0; padding: 8px 12px; font-family: var(--font-mono); font-size: var(--code);
  white-space: pre-wrap; word-break: break-word; color: var(--text);
}
"""


# Drop-in renderers for ``messageBlocksHtml`` and the keyboard handler.
# Exposes the following globals:
#   - ``renderMessageBlocks(messages)``       — installed as ``messageBlocksHtml``
#   - ``togglePolyFold(el)``                  — fold/unfold a tool/thinking block
#   - ``toggleCodeFold(el)``                  — fold/unfold a code block
#   - ``copyMessageById(messageId)``          — copy message text to clipboard
#   - ``jumpToAnchor(anchor)``                — set URL hash and scroll into view
#   - ``focusMessageByIndex(idx)``            — focus a message card (j/k drive)
#   - ``installReaderShortcuts()``            — registers the keydown handler
#
# The fold policy:
#   tool_use / tool_result / role==='tool'   → fold by default, show summary
#   thinking block (heuristic)               → fold by default, show "[thinking N tokens]"
#   code blocks (\u200b\u200bdetected via    → fold per block, button per block
#     fenced ``` markers in the text)
READER_JS = r"""
// MK3 reader message-card slice (#1202).

// Word-count proxy used in fold-summary chips. Token-true counts would
// require a tokenizer round-trip; the word count is what the daemon
// already exposes on the message payload (``word_count``).
function _polyTokenHint(m) {
  if (typeof m.word_count === 'number') return m.word_count + ' words';
  var text = m.text || '';
  if (!text) return '0 words';
  return text.trim().split(/\s+/).filter(Boolean).length + ' words';
}

function _polyToolSummary(m) {
  var text = m.text || '';
  var first = text.split('\n', 1)[0] || '';
  if (first.length > 140) first = first.substring(0, 140) + '\u2026';
  var label = m.message_type && m.message_type !== 'message' ? m.message_type : (m.role || 'tool');
  return {label: label, snippet: first};
}

function _polySplitCodeBlocks(text) {
  // Splits the text into a sequence of {kind: 'text'|'code', body, lang}.
  // Code blocks are detected via Markdown fenced ``` markers. The fence
  // language tag (if any) is preserved.
  if (!text || text.indexOf('```') === -1) return [{kind: 'text', body: text || ''}];
  var parts = [];
  var lines = text.split('\n');
  var buf = [];
  var inCode = false;
  var lang = '';
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var match = line.match(/^```(.*)$/);
    if (match) {
      if (!inCode) {
        if (buf.length) parts.push({kind: 'text', body: buf.join('\n')});
        buf = [];
        inCode = true;
        lang = match[1].trim();
      } else {
        parts.push({kind: 'code', body: buf.join('\n'), lang: lang});
        buf = [];
        inCode = false;
        lang = '';
      }
      continue;
    }
    buf.push(line);
  }
  if (inCode) {
    // Unterminated fence — treat as code anyway so we don't lose content.
    parts.push({kind: 'code', body: buf.join('\n'), lang: lang});
  } else if (buf.length) {
    parts.push({kind: 'text', body: buf.join('\n')});
  }
  return parts;
}

function _polyActionRailHtml(m, conversationId) {
  // Five actions per #1202: copy text, copy link, open raw, view
  // provenance, jump-to-anchor. Each carries data attributes the
  // keyboard handler reads when it needs to operate on the focused
  // message card.
  var anchor = m.anchor || ('message-' + (m.id || ''));
  var rawHref = '/api/conversations/' + encodeURIComponent(conversationId) + '/raw';
  return ''
    + '<div class="msg-actions" data-anchor="' + escAttr(anchor) + '">'
    +   '<button class="act-btn" data-act="copy-text" title="Copy message text (c)" '
    +           'onclick="copyMessageById(\'' + escAttr(String(m.id || '')) + '\')">copy</button>'
    +   '<button class="act-btn" data-act="copy-link" title="Copy anchor link" '
    +           'onclick="copyMessageLink(\'' + escAttr(anchor) + '\')">link</button>'
    +   '<a class="act-btn" data-act="open-raw" target="_blank" rel="noopener" '
    +     'title="Open raw conversation JSON" href="' + escAttr(rawHref) + '">raw</a>'
    +   '<button class="act-btn" data-act="view-provenance" title="View provenance (Raw tab)" '
    +           'onclick="openProvenanceTab()">prov</button>'
    +   '<button class="act-btn" data-act="jump-anchor" title="Jump to anchor in URL" '
    +           'onclick="jumpToAnchor(\'' + escAttr(anchor) + '\')">#</button>'
    +   (typeof _polyCopyActionRailHtml === 'function' ? _polyCopyActionRailHtml(m) : '')
    + '</div>';
}

function _polyToolFoldHtml(m) {
  var hint = _polyTokenHint(m);
  var sum = _polyToolSummary(m);
  var bodyHtml = '<div class="msg-text">' + esc(m.text || '') + '</div>';
  return ''
    + '<div class="msg-fold" onclick="togglePolyFold(this)">'
    +   '<span class="fold-arrow"></span>'
    +   '<span class="fold-label">' + esc(sum.label) + '</span>'
    +   '<code style="color:var(--text-muted);font-family:var(--font-mono);font-size:11px">'
    +     esc(sum.snippet) + '</code>'
    +   '<span class="fold-meta">' + esc(hint) + '</span>'
    + '</div>'
    + '<div class="msg-fold-body">' + bodyHtml + '</div>';
}

function _polyThinkingFoldHtml(m) {
  var hint = _polyTokenHint(m);
  return ''
    + '<div class="msg-fold" onclick="togglePolyFold(this)">'
    +   '<span class="fold-arrow"></span>'
    +   '<span class="fold-label">[thinking]</span>'
    +   '<span class="fold-meta">' + esc(hint) + '</span>'
    + '</div>'
    + '<div class="msg-fold-body"><div class="msg-text">' + esc(m.text || '') + '</div></div>';
}

function _polyCodeBlockHtml(part, blockIdx) {
  var langLabel = part.lang ? part.lang : 'code';
  var lineCount = (part.body || '').split('\n').length;
  return ''
    + '<div class="msg-code-fold" data-code-block="' + blockIdx + '">'
    +   '<div class="fold-bar" onclick="toggleCodeFold(this.parentElement)">'
    +     '<span class="fold-arrow"></span>'
    +     '<span class="fold-label" style="color:var(--accent)">' + esc(langLabel) + '</span>'
    +     '<span class="fold-meta" style="color:var(--text-dim);margin-left:auto">'
    +       lineCount + ' line' + (lineCount === 1 ? '' : 's') + '</span>'
    +   '</div>'
    +   '<div class="code-body"><pre>' + esc(part.body || '') + '</pre></div>'
    + '</div>';
}

function _polyBodyHtml(m) {
  // Splits non-tool/thinking bodies into text + foldable code blocks.
  var parts = _polySplitCodeBlocks(m.text || '');
  if (parts.length === 1 && parts[0].kind === 'text') {
    return parts[0].body ? '<div class="msg-text">' + esc(parts[0].body) + '</div>' : '';
  }
  var html = '';
  var codeIdx = 0;
  parts.forEach(function(part) {
    if (part.kind === 'text') {
      if (part.body && part.body.trim().length) {
        html += '<div class="msg-text">' + esc(part.body) + '</div>';
      }
    } else {
      html += _polyCodeBlockHtml(part, codeIdx++);
    }
  });
  return html;
}

function _polyIsTool(m) {
  var role = (m.role || '').toLowerCase();
  return role === 'tool'
    || m.message_type === 'tool_use'
    || m.message_type === 'tool_result'
    || m.has_tool_use === true;
}

function _polyIsThinking(m) {
  var role = (m.role || '').toLowerCase();
  return role === 'thinking'
    || m.message_type === 'thinking'
    || m.has_thinking === true;
}

function renderMessageBlocks(messages) {
  var convId = (state.selected && state.selected.id) || '';
  return (messages || []).map(function(m, idx) {
    var role = (m.role || '').toLowerCase();
    var isTool = _polyIsTool(m);
    var isThinking = _polyIsThinking(m);
    var tsHtml = m.timestamp
      ? '<span class="msg-ts" title="' + esc(m.timestamp) + '">'
        + new Date(m.timestamp).toLocaleTimeString() + '</span>'
      : '';
    var typeTag = (m.message_type && m.message_type !== 'message')
      ? '<span class="msg-type">' + esc(m.message_type) + '</span>' : '';
    var roleClass = 'msg-role ' + (role || 'unknown');
    var blockClass = 'msg-block';
    if (isTool) blockClass += ' tool-block';
    var anchor = m.anchor || ('message-' + (m.id || ''));
    var body;
    if (isTool) body = _polyToolFoldHtml(m);
    else if (isThinking) body = _polyThinkingFoldHtml(m);
    else if (typeof _polyHasPaste === 'function' && _polyHasPaste(m)) {
      var bannerHtml = typeof _polyPasteBannerHtml === 'function' ? _polyPasteBannerHtml(m) : '';
      body = bannerHtml + _polyRenderPasteBody(m);
    }
    else body = _polyBodyHtml(m);
    // MK3 attachment strip (#1199). Prepended above the body so the
    // cards stay visible regardless of fold variant. The hook is
    // optional so legacy harnesses without the attachment slice still
    // render cleanly.
    var attachmentStrip = (typeof _polyAttachmentStripHtml === 'function')
      ? _polyAttachmentStripHtml(m) : '';
    if (attachmentStrip) body = attachmentStrip + body;
    var rail = _polyActionRailHtml(m, convId);
    return ''
      + '<div class="' + blockClass + '" id="msg-' + idx + '" data-msg-id="' + escAttr(String(m.id || ''))
      +   '" data-anchor="' + escAttr(anchor) + '" tabindex="-1">'
      +   '<a id="' + escAttr(anchor) + '" style="position:absolute;visibility:hidden"></a>'
      +   '<div class="msg-header">'
      +     '<span class="' + roleClass + '">' + esc(role || '?') + '</span>'
      +     typeTag + tsHtml + rail
      +   '</div>'
      +   body
      + '</div>';
  }).join('');
}

function togglePolyFold(headerEl) {
  if (!headerEl) return;
  headerEl.classList.toggle('open');
}

function toggleCodeFold(wrapperEl) {
  if (!wrapperEl) return;
  wrapperEl.classList.toggle('open');
}

function _flashActButton(btn) {
  if (!btn) return;
  btn.classList.add('flash');
  setTimeout(function() { btn.classList.remove('flash'); }, 800);
}

function copyMessageById(messageId) {
  var msgs = (state.selected && state.selected.messages) || [];
  var hit = null;
  for (var i = 0; i < msgs.length; i++) {
    if (String(msgs[i].id) === String(messageId)) { hit = msgs[i]; break; }
  }
  var text = hit ? (hit.text || '') : '';
  var btn = document.querySelector('.msg-block[data-msg-id="' + (messageId || '').replace(/"/g, '\\"') + '"] .act-btn[data-act="copy-text"]');
  _polyClipboardWrite(text, btn);
}

function copyMessageLink(anchor) {
  if (!anchor) return;
  var url = window.location.origin + window.location.pathname + '#' + anchor;
  var btn = document.querySelector('.msg-actions[data-anchor="' + anchor.replace(/"/g, '\\"') + '"] .act-btn[data-act="copy-link"]');
  _polyClipboardWrite(url, btn);
}

function _polyClipboardWrite(text, btn) {
  var done = function() { _flashActButton(btn); };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, function() { _polyFallbackCopy(text); done(); });
  } else {
    _polyFallbackCopy(text);
    done();
  }
}

function _polyFallbackCopy(text) {
  // Older WebView / locked-down browser fallback. Hidden textarea +
  // execCommand('copy'); silently no-ops if unavailable.
  var ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.top = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); } catch (_) {}
  document.body.removeChild(ta);
}

function openProvenanceTab() {
  // Provenance lives behind the Raw inspector tab (#1125). Switch to
  // it and trigger the existing renderer if present.
  state.inspectorTab = 'raw';
  document.querySelectorAll('#inspector-tabs button').forEach(function(b) {
    b.classList.toggle('active', b.dataset.tab === 'raw');
  });
  if (typeof renderInspector === 'function') renderInspector();
}

function jumpToAnchor(anchor) {
  if (!anchor) return;
  var path = window.location.pathname + window.location.search;
  history.replaceState(null, '', path + '#' + anchor);
  var el = document.getElementById(anchor);
  if (el) el.scrollIntoView({block: 'center', behavior: 'smooth'});
  // Briefly focus the parent message card so j/k pick up from here.
  var block = el ? el.closest('.msg-block') : null;
  if (block) focusMessageBlock(block);
}

function focusMessageBlock(blockEl) {
  if (!blockEl) return;
  document.querySelectorAll('.msg-block.focused').forEach(function(b) {
    if (b !== blockEl) b.classList.remove('focused');
  });
  blockEl.classList.add('focused');
  state._focusedMessageIndex = parseInt(blockEl.id.replace('msg-', ''), 10);
}

function focusMessageByIndex(idx) {
  var blocks = document.querySelectorAll('#msg-list .msg-block');
  if (!blocks.length) return;
  if (idx < 0) idx = 0;
  if (idx >= blocks.length) idx = blocks.length - 1;
  var target = blocks[idx];
  focusMessageBlock(target);
  target.scrollIntoView({block: 'nearest'});
}

function _polyFocusedMessageIndex() {
  var n = state._focusedMessageIndex;
  return (typeof n === 'number' && !isNaN(n)) ? n : -1;
}

function _polyHandleNavigateMessages(direction) {
  var blocks = document.querySelectorAll('#msg-list .msg-block');
  if (!blocks.length) return false;
  var idx = _polyFocusedMessageIndex();
  if (idx < 0) idx = direction > 0 ? -1 : blocks.length;
  focusMessageByIndex(idx + direction);
  return true;
}

function _polyHandleCopyFocused() {
  var idx = _polyFocusedMessageIndex();
  if (idx < 0) return false;
  var block = document.getElementById('msg-' + idx);
  if (!block) return false;
  var msgId = block.dataset.msgId;
  if (msgId) { copyMessageById(msgId); return true; }
  return false;
}

function _polyHandleOpenConversation() {
  // ``o`` opens the focused conversation in the sidebar. When the
  // operator is already inside a conversation but the sidebar still
  // has selection, this re-loads the highlighted entry — matches
  // existing j/k semantics on the sidebar.
  var sel = document.querySelector('.conv-item.selected') || document.querySelector('.conv-item');
  if (sel) { sel.click(); return true; }
  return false;
}

function installReaderShortcuts() {
  // The base handler in web_shell.py owns ``/``, ``?``, ``Esc``, ``n/p`` and
  // delegates ``j/k`` to a hook so this slice can switch between sidebar
  // navigation (no conversation open) and message-card navigation
  // (conversation open). We attach a capture-phase handler so we can
  // suppress the base behavior when we drive messages.
  document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var hasConversation = !!(state.selected && state.selected.messages && state.selected.messages.length);
    if (e.key === 'j' && hasConversation) {
      if (_polyHandleNavigateMessages(1)) { e.preventDefault(); e.stopPropagation(); }
    } else if (e.key === 'k' && hasConversation) {
      if (_polyHandleNavigateMessages(-1)) { e.preventDefault(); e.stopPropagation(); }
    } else if (e.key === 'c') {
      if (_polyHandleCopyFocused()) { e.preventDefault(); e.stopPropagation(); }
    } else if (e.key === 'o') {
      if (_polyHandleOpenConversation()) { e.preventDefault(); e.stopPropagation(); }
    }
  }, true);

  // After a fresh render of the message list the focus index is stale.
  // Reset it so j/k start from the top of the new conversation.
  var msgList = document.getElementById('msg-list');
  if (msgList && typeof MutationObserver !== 'undefined') {
    var obs = new MutationObserver(function() { state._focusedMessageIndex = -1; });
    obs.observe(msgList, {childList: true});
  }

  // Honor an anchor in the URL on initial load.
  if (window.location.hash) {
    setTimeout(function() {
      var anchor = window.location.hash.replace(/^#/, '');
      if (anchor) jumpToAnchor(anchor);
    }, 400);
  }
}

// Install on load — both the original ``DOMContentLoaded`` and the
// already-loaded case are covered.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', installReaderShortcuts);
} else {
  installReaderShortcuts();
}
"""


# Extra rows inserted into the keyboard-shortcut help overlay so the
# operator can discover the new ``c`` / ``o`` bindings and the message-
# navigation reinterpretation of ``j`` / ``k``.
READER_HELP_HTML = r"""
      <kbd>o</kbd><span>Open focused conversation</span>
      <kbd>c</kbd><span>Copy focused message text</span>
"""

__all__ = ["READER_CSS", "READER_JS", "READER_HELP_HTML"]
