# MK3 view inventory

This is the complete intended view set. Each view has a stable job, primary data, main states, and initial implementation note.

## 1. Home / command center

Purpose: answer “what can I do now?” without forcing a query.

Primary data: daemon health, live ingestion freshness, recently updated conversations, saved views, pinned items, recent recall packs, degraded-state alerts, and top facets.

States: ready, first-run/no archive, daemon offline, schema/ingest critical, partial derived models, stale live feed, auth failed, empty archive.

Details: The command center should not be a marketing splash. It is an operational launchpad with one search box, saved-view cards, a “live sessions” strip, and “needs attention” chips.

## 2. Search / result cockpit

Purpose: find conversations, inspect why they matched, and select sets.

Primary data: query text, active filter chain, ranked hits, facet counts scoped to current query, score components, matched terms, tags, marks, flags, provider/source/repo/cwd/branch facets.

States: empty archive, no query, loading, no results, query syntax issue, FTS stale, hybrid lanes disabled, partial facet counts, large result set, selected result set.

Details: Search rows need two tiers: compact list metadata and expanded hit explanation. The selection bar supports bulk tag/export/re-embed/delete when those mutation surfaces are available. Until then, disabled actions show the exact missing contract.

## 3. Conversation reader

Purpose: read one transcript accurately and act on messages.

Primary data: header chips, message timeline, content blocks, paste segments, attachments, message actions, outline, notes, raw/provenance, lineage context, derived insights.

States: loading, no messages, raw-only/parse-failed, message window partial, live tail attached, stale derived models, unresolved parent, huge transcript virtualized, copy denied, mutation pending/error.

Details: The main pane shows message cards and folded content. The right inspector changes with selection: conversation, message, block, paste, attachment, raw artifact, topology edge.

## 4. Long-session map

Purpose: survive 100+ message coding-agent sessions.

Primary data: message windows, section outline, user turns, tool runs, file references, paste markers, attachment markers, token/cost bands, error/tool-result markers, branch points.

States: virtualized window, folded bulk output, selected anchor, unread/live appended messages, scroll restored, anchor not found after reimport.

Details: This is not a separate URL at first; it is a reader mode. The left rail becomes an outline of “user prompt → assistant work → tool burst → summary/decision.”

## 5. Stack workspace

Purpose: render several chatlogs at once in ways that make sense.

Primary data: open conversations, saved workspace title, synchronized filter context, active topology cluster, pinned messages, shared files/attachments, comparison anchors.

Modes:

- Tabs: many open conversations, one active transcript.
- Stack: 2–5 related conversations as vertical lanes, with compact headers and independently scrollable transcripts.
- Compare: two conversations side by side with aligned prompts, shared attachments, and divergent outcomes.
- Timeline: multiple conversations merged chronologically by timestamp/source.

States: empty workspace, unsaved changes, too many lanes, mixed providers with incompatible timestamps, missing topology, loading lane, degraded lane, shared inspector locked/unlocked.

Details: Stack is the MK3 answer to “several chatlogs at once.” It should be workspace-persistent, but the first slice can store only URL params and a local ephemeral layout while backend `reader_workspaces` is designed.

## 6. Topology / lineage

Purpose: explain continuations, forks, sidechains, subagents, unresolved parents, and late repairs.

Primary data: topology edges, conversation nodes, message branch nodes, provider-native IDs, confidence, provenance, resolution status, cycle/quarantine evidence, work-thread membership.

States: no edges, single parent, unresolved parent, ambiguous parent, repaired late parent, cycle rejected, subagent cluster, sidechain cluster, topology stale, graph too large.

Details: Not a generic graph UI. The default is a compact branch rail beside the conversation; the full view is opened only for clusters.

## 7. Attachments

Purpose: make file/image/document references inspectable, previewable, and safe.

Primary data: attachments, refs, messages that cite them, mime/size/path/provider IDs, preview/extraction status, missing/quarantined states, duplicates/shared refs, raw/source links.

States: available, missing local file, remote-only, unsupported preview, huge file, quarantined, duplicate/shared, text extracted, thumbnail ready, hash mismatch, path redacted.

Details: Attachments render inline as cards in messages, as an inspector tab in conversation view, and as a library view for archive-wide browsing.

## 8. Paste browser

Purpose: turn pasted context into a first-class navigable artifact.

Primary data: paste spans, message refs, content hash, line/char counts, detection source, confidence, dedup count, typed-vs-paste split, language/code-ratio hints.

States: explicit hook/history marker, heuristic-only, whole-message fallback, deduped across sessions, hidden large paste, detection conflict.

Details: In the transcript, paste blocks fold by default and expose “copy typed only,” “copy paste,” “open paste,” and “find repeats.” In the browser, pastes can be searched and grouped by hash/source.

## 9. Insights browser

Purpose: put derived read models next to reading instead of in a detached island.

Primary data: session profile, work events, phases, work threads, day/week summaries, tool usage, timing, cost/outlook, embeddings/similarity, resume brief.

States: unavailable, stale, rebuilding, partial coverage, estimated, source disabled, cost cap exhausted.

Details: Derived panels are issue-linked and statusful. They should never show fake charts. Each section says “live,” “partial,” “stale,” “not configured,” or “needs rebuild.”

## 10. Provenance / raw evidence

Purpose: make every row explainable and recoverable.

Primary data: raw artifacts, source path, raw id, acquisition time, parser/validator state, content hash, session events, hook events, blob/attachment links.

States: raw missing, raw quarantined, parse failed, validation drift, path redacted, source offline, blob missing, raw too large.

Details: Raw panels are text-only and never trusted HTML. The UI should show redacted paths by default with a deliberate “copy path” action when permitted.

## 11. Live / capture

Purpose: show active ingestion and browser capture without opening status internals.

Primary data: sources, cursors, last event, capture receiver state, browser extension status, live tail, pending queue, recent events.

States: live, stale, disconnected, receiver offline, source permission issue, schema critical, backpressure snapshot, polling fallback.

## 12. Status / maintenance

Purpose: answer “is Polylogue healthy?” and “what repair/backfill is pending?”

Primary data: daemon status, health alerts, source status, FTS/readiness, embedding readiness, raw failures, maintenance plan/run, SLO/read-surface evidence, visual smoke artifacts.

States: healthy, warning, critical, degraded, maintenance running, maintenance failed, read-only schema mismatch, no sources, first-run.

## 13. Settings / design controls

Purpose: tune density, theme, privacy, keyboard, and source behavior.

Primary data: theme, density, panel persistence, shortcuts, redaction policy, API auth state, live update mode, per-view defaults.

States: saved, unsaved, invalid shortcut, browser storage disabled, daemon write disabled, auth required.

## 14. Command palette / help

Purpose: make every action discoverable and keyboard-first.

Primary data: commands scoped by current selection, shortcuts, disabled reasons, recent commands, copy modes.

States: no selection, conversation selection, message selection, block selection, attachment selection, multi-select result set, offline/disabled actions.
