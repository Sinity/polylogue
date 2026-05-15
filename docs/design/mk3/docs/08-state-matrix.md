# MK3 state matrix

## Application shell states

Ready: daemon reachable, archive loaded, read surfaces healthy.

First run: no archive or no sources. Show setup path and ingest commands.

Offline: daemon unavailable. Show last known local state only if cached; otherwise explain `polylogued run`/service state.

Auth failed: API token required or invalid. Do not render raw/local paths until authenticated.

Critical schema mismatch: reader remains available in degraded mode; ingest disabled; show runtime/db schema versions and fix hints.

Live stale: SSE/polling last tick exceeds threshold; keep snapshot visible and mark data age.

Disconnected: SSE failed and polling failed; keep current view but stop optimistic live claims.

## Search states

No query: show saved views, recent conversations, pinned items.

Loading: skeleton rows preserve layout.

No results: show active filters and one-click clear; suggest nearby saved views/facets.

FTS stale: show results but mark ranking/facets as partial.

Hybrid lane missing: show FTS-only or list-only chip.

Query syntax/escape issue: structured error, not traceback.

Large result set: show pagination/virtualization and result count confidence.

## Conversation states

Canonical: parsed messages and stats available.

Raw-only: raw artifact exists but parse/materialization failed.

Partial: some messages/content blocks unavailable or truncated.

No messages: distinguish empty conversation, manifest-only repaired, and parse failure.

Huge: virtualized message window with outline.

Live tail: appended messages appear with “new” chips and preserved scroll.

Unresolved topology: branch chip shows missing parent/target.

Derived stale: insights/cost/topology/attachments preview outdated.

## Message states

Normal text.

Tool use/result.

Thinking/protocol/context block.

Paste present with explicit spans.

Paste present with unknown boundaries.

Attachments present.

Raw/provenance missing.

Copied/copy denied.

Marked/annotated/pending mutation.

Selected/focused/search match.

## Paste states

No paste.

Explicit paste: hook/history/provider marker.

Heuristic paste: message-level only or inferred spans.

Deduped paste: content hash repeated.

Large paste: folded with match windows.

Conflict: explicit marker contradicts heuristic; show data-quality chip.

## Attachment states

Available.

Missing.

Remote-only.

Unsupported.

Too large.

Quarantined.

Preview pending.

Preview failed.

Duplicate/shared.

Path redacted.

## Topology states

No topology.

Resolved parent.

Resolved child/fork sibling.

Unresolved parent.

Ambiguous parent.

Late repaired edge.

Cycle rejected/quarantined.

Subagent cluster.

Sidechain cluster.

Graph too large.

## User-state states

Unmarked/marked.

Annotation absent/present/editing/dirty/pending/error.

Saved view valid/invalid query spec/stale count.

Recall pack ready/building/failed/oversized.

Workspace ephemeral/saved/dirty/conflicting target refs.

## Realtime states

Live: streaming, last tick recent.

Stale: stream silent beyond threshold.

Backpressure: daemon sends snapshot event and resets local queue.

Polling fallback: ETag polling after SSE unavailable.

Disconnected: no stream or polling.

Event conflict: optimistic mutation or live update does not match server snapshot; reconcile and show toast.
