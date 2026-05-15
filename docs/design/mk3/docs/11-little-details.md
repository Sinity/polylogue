# Little details that make MK3 feel real

## Copy feedback

Copy buttons should not just flash. Show exactly what was copied: “message text copied,” “markdown with role header copied,” “typed-only copy unavailable: paste boundaries unknown.” Keep feedback near the target, not only as global toast.

## Permalinks

Every conversation, message, content block, attachment, paste span, and topology edge should have a stable anchor once its target identity is implemented. Conversation/message anchors come first.

## Disabled actions

Disabled actions are part of the design. Tooltips should say “needs paste spans,” “attachment preview unavailable,” “topology edge unresolved,” “annotation target not implemented,” or “daemon write API unavailable,” not generic disabled text.

## Data quality chips

Useful chip vocabulary:

- canonical
- inferred
- heuristic
- explicit hook evidence
- unresolved
- repaired
- stale
- partial
- estimated
- unavailable
- redacted

## Header chip order

Conversation header chip order:

1. provider/source.
2. live/stale.
3. repo/cwd/branch.
4. topology state.
5. message/tool/thinking/paste/attachment counts.
6. cost/tokens.
7. derived model availability.
8. user marks/tags.

## Search row chip order

Search result chip order:

1. provider/source.
2. date/age.
3. match lane/rank.
4. repo/cwd/branch.
5. topology marker.
6. flags: tool/thinking/paste/attachment.
7. marks/tags.

## Line height and density

Comfortable: large transcript, full headers, inline metadata.

Compact: smaller headers, folded metadata, action rail on hover/focus.

Dense: outline/list biased, message bodies collapsed unless selected.

## Panel persistence

Persist per view:

- left rail width/collapsed.
- inspector width/collapsed.
- selected inspector tab.
- density mode.
- fold defaults.
- stack lane widths.

Use durable user settings once available; use local persistence only as a temporary implementation detail, not the product contract.

## Privacy and safety

Local paths are sensitive. Redact by default in raw/provenance surfaces, but make deliberate copy/open actions available for the local owner. Raw panels render text only. Attachment/HTML previews must not inject trusted HTML into the main DOM.

## Keyboard shape

Global:

- `/` search.
- `Ctrl-K` command palette.
- `?` help.
- `g h/s/l/a/p/t/i/status` route shortcuts.

List:

- `j/k` move row.
- `Enter` open.
- `Space` select.
- `o` open in tab.
- `S` open selected as stack.

Reader:

- `j/k` next/prev message.
- `[` / `]` prev/next search match.
- `f` fold target.
- `y` copy menu.
- `m` mark menu.
- `a` annotate.
- `r` raw/provenance.
- `l` lineage.
- `x` open compare chooser.

Stack:

- `Alt-1..9` lane/tab focus.
- `gt/gT` tab navigation.
- `s` save workspace.
- `v` switch tabs/stack/compare/timeline.
