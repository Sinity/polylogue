# MK3 implementation slices

## Slice 0 — contract audit and target refs

Goal: make later UI work reuse one action/target vocabulary.

Work:

- Define `TargetRef` model for conversation and message first.
- Add deterministic message anchors/permalinks.
- Extend reader payload enough to include message flags, content block summary counts, attachment counts, paste status, provenance status.
- Add disabled-reason vocabulary for future target types.

Verification:

- HTTP contract tests for target refs and message anchors.
- Visual smoke: copy menu disabled/enabled states.

## Slice 1 — message card + copy actions + folds

Goal: make the single-conversation reader excellent before adding more views.

Work:

- Message card component with action rail.
- Copy menu: text, markdown, permalink, raw when available.
- Fold policies for tool/thinking/code/paste-like long blocks.
- Search match windows inside folds.
- Keyboard actions `y`, `f`, `r`, `a`, `m`.

Verification:

- DOM smoke for populated, huge, folded, copied, denied clipboard fallback.
- Snapshot/visual evidence for message card variants.

## Slice 2 — paste spans and paste rendering

Goal: stop treating paste-heavy messages as ordinary text.

Work:

- Add `message_paste_spans` or equivalent derived projection.
- Render explicit spans and whole-message fallback.
- Copy typed-only / paste-only when spans exist.
- Paste browser MVP grouped by conversation/message.

Verification:

- Fixture with explicit paste span.
- Fixture with heuristic whole-message fallback.
- Query path still supports `has_paste` and `typed_only`.

## Slice 3 — attachments product surface

Goal: make attachment facts visible and safe.

Work:

- Extend conversation/message payload with attachment refs.
- Attachment inline card and inspector tab.
- Attachment library MVP.
- Attachment derivative/status model if previews are produced.

Verification:

- Available/missing/unsupported/too-large/quarantined visual states.
- Raw HTML safety regression.

## Slice 4 — user state: marks, annotations, saved views

Goal: make reader state durable.

Work:

- Replace or extend current `user_marks` with target-ref based marks.
- Add annotations table/API.
- Saved-view query spec roundtrip.
- Reader toggles and notes panel.

Verification:

- Idempotent CRUD tests.
- Reimport/identity preservation test.
- Content-hash exclusion test.
- Visual smoke for marked, annotated, pending, failed mutation states.

## Slice 5 — topology substrate

Goal: solve continuation/fork/subagent modeling before designing graph-heavy screens.

Work:

- `topology_edges` table and archive operations.
- Preserve unresolved native parents.
- Late repair.
- Cycle detection/quarantine.
- Topology read model: ancestors, descendants, siblings, root/members.

Verification:

- Storage tests for unresolved, repair, cycle, provenance.
- Parser fixtures for Claude Code sidechain/subagent, Codex continuation, ChatGPT message branches.

## Slice 6 — topology UI and stack builder

Goal: let topology shape the reading flow.

Work:

- Branch chips and lineage rail.
- Full topology explorer for selected cluster.
- “Open parent chain as stack.”
- “Compare with parent/fork.”

Verification:

- Visual smoke: resolved parent, unresolved parent, subagent cluster, fork compare.

## Slice 7 — multi-chat workspace

Goal: tabs/stack/compare/timeline.

Work:

- Workspace tab strip.
- Stack lanes for selected IDs.
- Compare two conversations with simple alignment.
- Timeline view for selected IDs.
- Ephemeral URL persistence first; durable `reader_workspaces` later.

Verification:

- Visual smoke for tabs, stack, compare, timeline, too-many-lanes state.

## Slice 8 — insights, cost, similarity, provenance integration

Goal: surface derived models where they help reading.

Work:

- Insights browser sections with truthful availability chips.
- Cost panel with estimated/known/unknown labeling.
- Similarity “more like this” panel.
- Provenance drilldown for raw artifacts, source run, hook event, parser evidence.

Verification:

- Partial/unavailable/stale/rebuilding states.
- No fake charts; every chart has data availability evidence.

## Slice 9 — realtime

Goal: live/stale/disconnected behavior across list, reader, stack, and status.

Work:

- SSE endpoint with topic subscription.
- Snapshot coalescing/backpressure.
- ETag polling fallback.
- Live chips and new-row animations.

Verification:

- Event contract tests.
- Reader/list DOM smoke for appended message, stale state, disconnected fallback.

## Slice 10 — design assurance

Goal: keep MK3 from rotting.

Work:

- `docs/design/mk3` committed with screenshots.
- Visual smoke harness covers every primary view and degraded state.
- Accessibility checks for keyboard-only navigation.
- Style tokens shared with CLI/theme docs.
