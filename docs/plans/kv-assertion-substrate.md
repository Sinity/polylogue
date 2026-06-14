# KV assertion substrate execution spec

## Purpose

Unify user annotations and agent memory as one typed overlay model. The user should experience highlights, marks, notes, corrections, decisions, lessons, blockers, and run-state as distinct UI actions; the storage model should treat them as evidence-linked assertions over typed targets.

## Storage boundary

Raw transcript rows, source records, tool calls, spans, blobs, FTS rows, vector rows, and deterministic projections are not KV. KV is for authored or interpretive overlays: things a user, agent, transform, or system asserts about a target.

## Entry shape

A KV entry needs at minimum: id, scope ref, target ref, key, kind, value JSON, optional body text, author ref, evidence refs, confidence if applicable, status, visibility, context policy, staleness policy, supersedes refs, created time, updated time.

Common kinds: highlight, mark, annotation, correction, decision, lesson, blocker, handoff, RunState, prompt evaluation, transform candidate, saved query, recall pack.

## Scope and target refs

The same model should work at many granularities: global, project, repo, issue, PR, branch, worktree, file, symbol, session, run, span, message, tool call, context envelope, prompt artifact, transform run, work packet, and bead.

## Migration path

1. Add a generic KV table in the user tier.
2. Add helpers for typed keys and target refs.
3. Write new marks/annotations/corrections through KV first.
4. Keep old APIs as thin adapters or views until all callers move.
5. Add query/read surfaces for KV by kind, scope, target, author, and status.
6. Add transform outputs as KV candidates, not automatically promoted memory.
7. Add context-policy fields only when used by recovery/continue flows.

## Acceptance criteria

- User highlights, marks, notes, and corrections can be represented as KV entries.
- Agent lessons, decisions, blockers, handoffs, and RunState can use the same substrate.
- Every entry can point back to raw evidence or be explicitly marked as unevidenced.
- Supersession and status are modeled, so stale assertions can be retired without erasing history.
- Context injection can filter by scope, kind, status, visibility, and policy.
- Old overlay tables stop growing new semantics.

## Non-goals

- Do not move raw evidence or deterministic indexes into KV.
- Do not make KV the task tracker; Beads or GitHub may own work-item state.
- Do not auto-inject every memory into prompts.
- Do not create separate human-note and agent-memory databases.