# Session continuation, forking, sidechains, and subagents

MK3 needs a data model and UI vocabulary that distinguish four concepts users care about:

Continuation: a new conversation continues a previous session, usually after restart, compaction, crash, or manual resume.

Fork: a new path branches from a prior conversation or message and explores a different continuation.

Sidechain: provider-created side context that is related but not the main session path.

Subagent: a child agent or delegated worker stream spawned from a parent session.

## Why parent columns are insufficient

A single `parent_conversation_id` loses:

- unresolved native parent IDs when the parent row is missing at ingest time.
- multiple evidence sources for the same edge.
- distinction between resolved/inferred/unresolved/ambiguous/cyclic edges.
- message-level fork origin.
- raw/hook/artifact provenance for the relationship.
- late repair history.

## Reader topology components

### Branch chip

Appears in conversation header:

```text
continuation · parent resolved · depth 3
fork · from message 42 · inferred
subagent · parent unresolved · native id preserved
```

Click opens the lineage rail.

### Lineage rail

A compact side rail in conversation reader:

- parent chain above current conversation.
- sibling/fork lanes below.
- sidechains/subagents grouped separately.
- unresolved parent displayed as a dashed ghost node with provider-native ID.
- current node highlighted.

### Full topology view

For clusters:

- graph/list hybrid, not force-directed graph first.
- left: tree/edge list grouped by root.
- center: selected edge and nearby nodes.
- right: provenance, confidence, raw evidence, repair status.

### Fork/continuation actions

Reader actions should be conservative:

- copy continuation context: selected parent chain + current transcript excerpt.
- create recall pack from branch chain.
- open stack: parent + current + sibling side by side.
- compare with parent.
- mark unresolved edge for review.

Actual provider-level “continue session” actions should remain disabled until a separate launcher/control-layer contract exists.

## Edge states

Resolved: source and target canonical IDs exist.

Unresolved: target provider ID exists but target canonical row is absent.

Inferred: parser/materializer inferred relationship without explicit provider edge.

Ambiguous: multiple candidate parents match.

Repaired: previously unresolved edge was resolved after late parent arrival.

Cyclic/invalid: edge rejected or quarantined with cycle path evidence.

## UI state examples

Unresolved parent:

```text
Parent: claude-code session 7e4... not yet ingested
[copy native id] [search possible parents] [show raw evidence]
```

Late repair:

```text
Parent resolved 2026-05-11 02:14 from native id match
confidence: resolved · provenance: repaired
```

Fork from message:

```text
Forked from user message 42 in parent session
[open parent at message] [compare from fork point]
```

Subagent:

```text
Subagent stream · spawned by Task tool · parent message 88
[open spawn tool call] [stack with parent]
```
