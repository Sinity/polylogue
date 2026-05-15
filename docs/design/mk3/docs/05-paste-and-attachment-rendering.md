# Paste and attachment rendering

## Paste rendering

MK3 should not render pasted payloads as ordinary text. They are often the important context, but they destroy readability when expanded inline.

### Paste block anatomy

A paste block shows:

- label: `paste #1`, `pasted file`, `chatlog paste`, or heuristic label.
- source: hook, history, provider marker, heuristic, manual.
- size: lines, chars, estimated tokens.
- confidence: explicit/high/medium/low.
- content kind: text/code/image/file/unknown.
- dedup: “seen in 4 conversations” when hash exists.
- actions: expand, copy paste, copy typed-only message, open paste, find repeats.

### Rendering modes

Collapsed inline:

```text
▸ paste #1 · explicit · 218 lines · 14.2k chars · copied from UserPromptSubmit
```

Match window:

```text
paste #1 · match 3/8
  120  ... preceding context
  121  matched line
  122  following context ...
```

Expanded:

- monospace, scroll-limited, syntax-highlight optional only if safe/local.
- line numbers on demand.
- copy buttons sticky in the fold header.

Whole-message fallback:

When only `has_paste=1` exists, render the whole user message normally but add a warning chip: “paste likely · boundaries unknown.” The “typed-only” copy action is disabled until paste spans exist.

### Paste browser

Archive-wide paste browser groups by content hash when available. Useful slices:

- largest pastes.
- repeated pastes.
- chatlog-forwarding pastes.
- code-heavy pastes.
- low-confidence heuristic pastes.
- pastes with attachment/image markers.

## Attachment rendering

Attachments must appear in three places:

1. Inline inside the message where they matter.
2. Conversation inspector tab for all attachments in the current conversation.
3. Global attachments library for search/browse.

### Attachment card anatomy

- name or generated label.
- mime type and size.
- provider/source icon.
- availability state.
- preview thumbnail/text snippet when safe.
- refs: message count/conversation count.
- actions: open preview, copy metadata, copy extracted text, open raw, show refs, copy path when allowed.

### Attachment states

Available: local path/blob present and safe preview exists.

Missing: ref exists but blob/path unavailable. Show provider id and source path if redaction permits.

Remote-only: provider id exists, local content not archived.

Unsupported: content exists but no preview renderer.

Too large: preview skipped by budget.

Quarantined: extraction or sanitizer blocked content.

Shared: same attachment identity or hash appears in multiple conversations/messages.

Path-redacted: hide local path unless explicit copy action is allowed.

### Preview policy

Images: thumbnail + metadata.

PDFs: metadata + extracted text/thumbnail if derivative exists.

Text/code: extracted text preview with language guess.

HTML: text-only or sandboxed strict-CSP preview; never trusted main-DOM HTML.

Archives/binaries: metadata only.

### Attachment library view

Facets:

- provider/source.
- mime family.
- availability state.
- size bucket.
- duplicate/shared.
- has preview / no preview.
- referenced by marked/pinned conversations.

Rows:

- attachment identity.
- current availability.
- ref count.
- latest conversation.
- actions.

Inspector:

- preview.
- refs table.
- raw provider ids.
- derivative status.
- provenance.
