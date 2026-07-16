---
created: 2026-07-16
purpose: Decide lineage normalization, unresolved-parent, cycle, and snapshot semantics for L05-L06
status: recommended-decision
project: polylogue
---

# Lineage composition and snapshots

## Decision

Retain the current normalized representation: each child stores only its
canonical divergent tail plus a typed `session_links` edge. Reads compose the
parent prefix through `branch_point_message_id` with the child tail. Every
composed read runs inside one deferred read transaction and returns both content
and `LineageCompleteness`.

Arrival order must not change the final composed transcript. Missing or
ambiguous parents do not make the child unreadable and do not authorize invented
prefixes: the reader returns the provable child material with an explicit
incomplete state. Later parent arrival or replacement re-normalizes atomically.

## Canonical edge states

An edge is one of:

- `resolved` — one parent and branch point are proven;
- `unresolved_parent` — parent native identity is known but not present;
- `dangling_branch_point` — parent exists but the recorded message cannot be
  found in the current parent snapshot;
- `ambiguous_parent` — more than one candidate satisfies incomplete identity;
- `repaired` — a formerly incomplete edge was rebound with recorded proof;
- `quarantined_cycle` — resolving it would introduce a cycle;
- `invalid_inheritance` — claimed prefix sharing is not supported by transcript
  evidence.

`inheritance=spawned-fresh` never composes a parent prefix. A prefix-sharing
edge composes only to an existing proven branch point.

## Write normalization

After every accepted child full replacement, parent full replacement, or
link-resolution event, run one atomic lineage normalization transaction:

1. capture the affected parent/child composed signatures;
2. retain the newest accepted sibling variants under their existing variant
   identity rules;
3. find the longest exact provider-semantic parent prefix supported by evidence;
4. bind an existing branch-point message id or record a typed incomplete state;
5. store only the child's divergent physical rows;
6. update edge and completeness state in the same transaction;
7. verify direct physical rows, link row, and composed signature before commit.

Failure rolls back physical rows and edge state together. Retrying the same
normalized envelope is idempotent.

## Read snapshot contract

`read_archive_session_envelope`, message pagination, transcript views, and all
other composed readers must share one `LineageReadSession`:

- open one read-only connection and `BEGIN DEFERRED` before resolving topology;
- resolve the full bounded ancestor chain and all message pages through it;
- detect repeated session ids and enforce a defensive maximum depth;
- compose each ancestor at most once per read;
- return content, participating session/version identities, and completeness;
- close/rollback the read transaction on completion or cancellation.

The depth bound is a resource guard, not a semantic truncation. Hitting it
returns `depth_exceeded` completeness plus a continuation/detail reference; it
does not pretend the returned transcript is complete.

## Parent replacement

`branch_point_message_id` remains a non-FK logical reference. A parent
full-replace may delete and recreate deterministic message ids without cascade
nulling the edge. Within the same normalization transaction:

- if the exact branch point survives, composition is unchanged;
- if equivalent prefix content maps deterministically to a replacement id, the
  edge is repaired with proof;
- otherwise the child remains readable as its physical tail and completeness
  becomes `dangling_branch_point` until repaired or judged.

Never silently attach to “nearest timestamp”, ordinal, or similar message.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Store every child's replayed full transcript | Simple reads | Duplicates large prefixes, makes parent corrections diverge, obscures logical message identity |
| Reject child ingest until parent exists | Simple complete-state invariant | Loses arrival independence and blocks valid offline/import ordering |
| Resolve missing branch points heuristically | Maximizes apparently complete reads | Fabricates history and hides evidence loss |
| Repair edges lazily during reads | Avoids writer work | Read paths mutate authority, race each other, and disagree across surfaces |
| Read each ancestor in separate transactions | Minimal refactor | Produces impossible mixed snapshots during parent replacement |

## Migration and implementation seams

- Centralize the existing normalization fragments in
  `storage/sqlite/archive_tiers/write.py`; do not add a second lineage model.
- Keep `session_links` as topology and lineage authority.
- Make completeness a first-class domain value consumed by CLI/API/MCP/daemon
  projections; do not reduce it to a warning string.
- `polylogue-4ts.3` parser classification must require positive provider
  evidence before marking auto-compaction prefix-sharing.
- Rebuild replay (`polylogue-5q2u`) must order or retry lineage normalization so
  child-before-parent produces the same result as parent-before-child.

## Required proof

- stateful parent/child/sibling operations over all arrival orders converge to
  identical physical tails, edges, composed content, and completeness;
- parent replacement preserves or explicitly degrades branch points atomically;
- deterministic barriers prove a composed read is wholly old or wholly new;
- cycle and depth mutations never hang or invent content;
- retry after rollback leaves no half-normalized rows;
- the `polylogue-866e` historical seed fails when normalization is removed.

Primary evidence: `polylogue-866e`, `polylogue-4ts.4`, `polylogue-4ts.6`,
`polylogue-4ts.3`, `polylogue-5q2u`;
`polylogue/storage/sqlite/archive_tiers/write.py`,
`polylogue/storage/sqlite/queries/session_links.py`, and
`polylogue/storage/sqlite/queries/message_query_reads.py`.
