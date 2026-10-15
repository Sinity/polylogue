# Decision: how subagents, resumes, forks and copies are stored

Status: **decided** (storage option B; link kinds from real data,
`evidence/lineage-model.md`). Decided by the spec authors from the measured
cost (9.3% copied records, text stored once) and from the lineage marks
living in the rebuildable `index.db`, which makes the choice reversible; the
operator may overrule.

## The problem

Most sessions are related to others: about three Claude Code subagents per
main session, half of all Codex sessions are spawned threads, and resumed or
forked sessions begin with a copy of an earlier conversation. The operator's
rulings: content must not be duplicated where it matters ("the data model we
were supposed to have was not duplicating content like this", 2026-09-06);
counts, costs and search must not double-count; links come only from
evidence.

## Storage: option B

Each session stores its full message list exactly as its own files have it.
A message the child copied from an earlier session carries `inherited_from`
(the source message's ID). Text lives in a content-addressed table, so copied
text is stored once. Counts, costs and search exclude inherited messages by
default; `inherited:*` includes them (`inherited:true` selects only copies).

Rejected:

- **A**, the Python design, recorded in polylogue-4ts ("prefix-sharing
  children store only divergent tails", composed on read, one lineage state
  machine): children store only their new messages and are recomposed on
  read. A parent change cascaded into its children, reading a child needed
  its parent, and it took about 2,000 lines of special code, while
  deduplicating storage for only about 537 linked sessions. Option B
  reverses that Bead's design on purpose and says so: reads never compose,
  and the cost of full lists is small, measured on the operator's data
  (`evidence/ideas-storage.md`): copied records are 9.3% of Claude Code
  records, and with copied text stored once they add only a few percent to
  the index. The Bead's invariants carry over as tests (counted once,
  ordering, citations, a parent arriving late, compaction; the lineage laws (`plan/testing/laws.md`: 4 and 11
  for copies and late parents, 25, 29–34, 39, 99)). Question 1 is settled
  (`questions.md`); the operator may overrule.
- **C**, full lists without marks: double-counts.

## Three link kinds

Real data showed that "fork" is mostly an attribute of a spawn: 1,006 of
1,007 Codex forks are spawned subagents started with the parent's context,
and Claude Code marks a forked subagent with `isFork` in its meta file. So a
link has a kind (who started the child) and a `how` (how the child got the
parent's context):

- `subagent` (`fresh`, `copied`, `summarized`): started by a tool call;
- `continuation` (`resume`, `fork`, `after_compaction`): a conversation
  continued in a new session;
- `reference` (`briefing`, `ping`, `mention`): the child's text names the
  parent.

`link:fork` means either a forked continuation or a copied subagent.
`session.kind` has no `continuation`; that fact lives in the links.

## How copies are detected

- **Claude Code.** Copied records keep the source's `sessionId` (main files)
  or `agentId` (subagent files); `forkedFrom` names the source message of a
  forked conversation. Matching is by native message ID, and for an
  `assistant` record also by `message.id` with `requestId`: fork subagents
  re-emit the parent's spawning response under a new `uuid` (40 of 61
  sampled forks), and that response is counted once, in the parent. Both
  values are stored on model messages (`message.response_id`,
  `message.request_id`, indexed) so the derive step can make this match
  when the parent arrives.
- **Codex.** Some forks replay the parent's items without item IDs (March
  2026, and some later paginated-history forks); they are matched to the
  parent by content, as the longest run of leading items equal in order, and
  `session_meta.subagent_history_start_ordinal` (when present) bounds where
  the child's own records start. July 2026 forks carry a summary instead (a
  compaction event, nothing inherited); from September 2026 nothing is
  copied. A spawned thread replays its parent only when it carries
  `forked_from_id` or `subagent_history_start_ordinal`; one with neither
  starts empty (`how: fresh`).

## When marks are set

The parser reports what each record itself says about its source
(`copy_evidence`); it never sets `inherited_from`. Every mechanism that sets
it, and who runs it:

- **The writer, from the record's own evidence**, in the session's own
  transaction: a copied `sessionId` with the record's original `uuid`, an
  `agentId` that differs from the file's, `forkedFrom.messageUuid` (Claude
  Code); a replay marker the record carries (Codex `forked_from_id`,
  `subagent_history_start_ordinal`); a ChatGPT node ID the record names as
  shared with another conversation. Morrow: L5's fork copies name their
  source in `forkedFrom`, L8's copies name an absent session by its
  `sessionId`.
- **The derive step, by matching**, once the parent session is stored: a
  parent message's `uuid` (Claude Code copies without other evidence);
  `message.id` plus `requestId` (Claude Code re-emitted responses,
  `response_id`/`request_id`); content equal to the parent's in order (Codex
  replays and older resumes, AI Studio chunks equal to the parent's); a
  ChatGPT node ID found in another conversation (`native_message_id`,
  stored and indexed, never split from the ID).

When the parent arrives, the writer bumps the child's `changed` counter and
the marks are recomputed. Until then, and forever if the provider deleted
the parent before Polylogue saw it (about 680 Claude Code resumes on this
machine), the copies of one native message (same origin, same native
message ID, same named source session) in several sessions count once: in
the session whose earliest non-copy record is earliest (by `at`, then
session ID); copies themselves never decide it, so the choice cannot depend
on which session the copies are given to. The others get `counted_as`
naming that copy. Every copy whose evidence names its source carries that
`inherited_from`, present or not: genealogy and counting are separate
fields, because the counted copy is not necessarily the one the others
descend from (an outside review, 2026-10-12). When the source arrives,
every copy's `counted_as` becomes the source's message. A copy whose source
the provider does not name has `copy_origin: unknown` and is counted
nowhere until a source is identified. A copy of a message of an excised
session (listed in the excision's `message_ids_json`) is counted nowhere:
its usage is in the excision's `usage_json`
(`spec/storage/archive-db.md`, how excision works).
The copy's foreign session ID stays as the evidence of a pending link. This is part of slice 1 for Claude Code, because resumed
sessions are common there and would otherwise be counted twice.

## Consequences

- `spec/model.md` (Link), `spec/sources/` (evidence per provider and era),
  `spec/storage/index-db.md` (`link.how`, `link.attrs_json`,
  `message.from_session`, `session.team`), `spec/analysis.md` (tree cost) and `spec/query.md`
  (`parent`, `root`, `depth`, `continues`, `lineage:`, `tree_cost`,
  `child.<field>`, `from_session`) follow this.
- The lineage laws (`plan/testing/laws.md`: 4 and 11 for copies and late
  parents, 29–34 and 99 in slice 1; 25 and 39 in slice 6): no double count in a subagent tree, a copy
  is never counted twice, arrival order and late parents do not change
  results, every link has evidence.
- The Morrow lineage set (`examples/archive/FACTS.md`) exercises a nested
  subagent, a sibling copy, an explicit fork, a team with a peer message, a
  missing parent, the same session in two folders and an older-layout
  sidechain.
