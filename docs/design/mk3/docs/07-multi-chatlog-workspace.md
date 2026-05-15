# Multi-chatlog rendering: tabs, stack, compare, timeline

The user explicitly asked for dynamic rendering of several chatlogs at once. MK3 should support four modes instead of choosing one.

## Tabs

Use for lightweight multitasking. A tab is an open target: conversation, saved view, topology cluster, paste, attachment, or status page.

Details:

- Tabs show provider icon, short title, dirty/unsaved workspace marker, live/stale dot.
- Cmd/Ctrl-click on a result opens in background tab.
- `gt`/`gT` or Alt-left/right switches.
- Tabs persist in a workspace when saved; otherwise they are ephemeral URL state.

## Stack

Use for related conversations that should be read together.

Layout:

- 2–5 lanes, each with mini header and independent transcript scroll.
- Shared top bar with query/facet context and stack actions.
- Shared right inspector can lock to active lane or selected target.
- Lanes can be compact, normal, or outline-only.

Useful stack builders:

- open parent chain.
- open siblings/forks.
- open selected search results.
- open same work thread.
- open same paste hash or shared attachment.
- open recent conversations in repo/cwd.

States:

- too many lanes: collapse older lanes into tabs.
- mixed live/stale: each lane has own live chip.
- missing topology: stack can still open selected explicit conversations.
- unsaved workspace: show save affordance.

## Compare

Use for two conversations. It is not a code diff; it is a transcript comparison.

Alignment strategies:

- by user prompt ordinal.
- by topology fork point.
- by timestamp.
- by shared file/attachment/tool action.
- by search term occurrences.

Compare row types:

- same/similar user prompt.
- shared pasted context.
- divergent assistant strategy.
- shared tool/file action.
- result/decision difference.

Actions:

- copy comparison summary.
- create recall pack from both.
- pin paired messages.
- open raw evidence for either side.

## Timeline

Use when chronological reconstruction matters.

Sources:

- multiple conversations.
- hook events.
- tool/action events.
- attachments/pastes.
- source ingestion events.

States:

- timestamp gaps/unknown timestamps.
- provider-relative timestamps only.
- clock skew.
- dense event bursts.

Rendering:

- major events as cards.
- minor tool bursts collapsed.
- conversation/message origin always visible.
- switch between time order and conversation order.

## Workspace persistence

First slice can use URL params:

```text
/w/stack?ids=convA,convB,convC&mode=stack&focus=convB:msg42
```

Durable workspaces later use `reader_workspaces` with `open_targets_json` and `layout_json`.

Recommended saved workspace fields:

- name.
- mode.
- query spec that created it.
- open targets.
- layout mode and lane widths.
- pinned target refs.
- inspector lock state.
- created/updated timestamps.
