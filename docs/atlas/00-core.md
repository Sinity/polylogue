# Polylogue Atlas

This is the cold-start map for the repository. It is orientation, not a
second project contract: `CLAUDE.md` owns semantics and task authority lives
in the external task backend.

## What the system is

Polylogue is a local, single-writer archive for AI coding and chat sessions.
It acquires heterogeneous exports and live captures, parses them into a
normalized session tree, stores durable evidence in split SQLite tiers, and
serves query-first reads through the CLI, MCP, Python API, and daemon
(`polylogue/daemon/cli.py:1956-1999`; `polylogue/storage/sqlite/archive_tiers/bootstrap.py:49-85`).

The useful mental model is a flight recorder: every derived answer should be
able to resolve to stored source bytes, structured records, and provenance.
The archive is local and provider-agnostic; it is not a hosted observability
dashboard or an unbounded semantic-memory promise.

## Four jobs

1. **Search** — find sessions, messages, blocks, actions, assertions, and
   insight units with the query DSL (`polylogue/archive/query/expression.py:1-80`).
2. **Analyze** — read descriptor-driven insights and measurements whose
   evidence boundaries are explicit (`polylogue/analysis/registry.py:1-100`).
3. **Audit** — inspect operation previews, authorization, attempts, and
   continuity, with durable audit records separate from rebuildable indexes
   (`polylogue/storage/sqlite/archive_tiers/audit.py:53-67`).
4. **Remember** — retain user assertions and context-delivery provenance in
   the durable user tier; claims remain typed and evidence-linked
   (`polylogue/storage/sqlite/archive_tiers/user.py:19-40`).

## Data flow

```text
source acquisition → detection → parsing → archive write → derived reads
       source.db                         index.db / insights / embeddings
                                             ↓
                         CLI · MCP · API · daemon readers
```

The parsed-session write choke point computes public origin, native identity,
session identity, and parser fingerprints before lowering records
(`polylogue/storage/sqlite/archive_tiers/write.py:1063-1068`). The daemon owns
the normal live write path and serializes admitted mutations; read surfaces
adapt through operations and insights (`polylogue/daemon/write_coordinator.py:288-344`).

Session summaries, profiles, raw observations, FTS, and embeddings use
domain-owned inspection and publication through the derivation kernel. The
daemon sheet identifies the remaining stage-based owners; the presence of the
kernel does not imply their retirement (`polylogue/daemon/session_profile_composition.py:37-62`;
`polylogue/daemon/convergence_stages.py:552-563`).

## Identity you must preserve

Sessions, messages, and blocks form the core tree. Identity is generated, not
duplicated in caller metadata:

- `session_id = origin:native_id` (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:692-705`).
- `message_id` uses explicit native (`:n:`) or content-derived (`:c:`) namespaces (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:281-300`).
- `block_id = message_id:position` (`polylogue/storage/sqlite/archive_tiers/archive_tiers_specs.py:478-495`).

`material_origin` is independent from role and expresses authoredness. Tool
outcomes use the canonical `blocks.tool_outcome` enum; deliberate unknown
outcomes preserve `tool_result_outcome_unknown_reason`. Lineage
children physically store only their divergent tail and reads recompose the
parent prefix (`polylogue/storage/sqlite/archive_tiers/write.py:755-801`;
`polylogue/storage/sqlite/archive_tiers/write.py:1903-1951`).

## Where to start

Read the area sheet before opening broad source trees:

| Question | Sheet | First implementation area |
| --- | --- | --- |
| How are files detected and normalized? | `sources-parsers.md` | `polylogue/sources/`, `polylogue/pipeline/` |
| How do query and read surfaces work? | `query-read-path.md` | `polylogue/archive/query/`, `polylogue/operations/` |
| How are durable and derived records stored? | `storage.md` | `polylogue/storage/` |
| Who owns writes and convergence? | `daemon.md` | `polylogue/daemon/` |
| How does MCP dispatch? | `mcp.md` | `polylogue/mcp/` |
| Which invariant owns this rule, and what enforces it? | `doctrine.md` | `devtools/gate.py`, the owners it names |

Then consult `docs/architecture.md` for the ring model, the specific area
sheet for anchors and gotchas, and `devtools --list-commands` for executable
verification. `devtools gate atlas` checks that every citation on these
sheets still resolves to a real file and line range.

## Non-negotiable boundaries

- Durable source, user, and audit state is authority; indexes and ops state
  are rebuildable or disposable.
- New semantics belong in storage, insights, or the product layer first;
  surfaces adapt rather than owning substrate rules.
- The daemon is the live write owner. A direct writable store entry point is
  not evidence that a new surface may bypass its coordinator.
- Public filters use `origin`; `provider` is a parser/raw-wire concept.
- Tests use synthetic fixtures and managed `devtools test` commands; ambient
  personal archives never enter tracked files.
