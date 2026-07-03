# Fables Poly Findings

Source: `/realm/inbox/fables-poly.md`, reviewed 2026-07-03.

This note keeps only findings that survived quick source review or are useful
devloop backlog shape. The original inbox transcript is not a durable artifact.

## Integrated Immediately

- Message-level `messages.content_hash` was identity-shaped while embedding
  freshness treated it as content-shaped. The archive writer now hashes the
  stored message role/type/material-origin and text/block content, and archive
  embedding pending selection reselects clean sessions whose
  `message_embeddings_meta.content_hash` is stale.

## High-Value Backlog

- Replace remaining keyword outcome/pathology heuristics with structural
  evidence where available, especially `tool_result_is_error` and
  `tool_result_exit_code`.
- Build outcome-conditioned analytics: cost, duration, retries, and tool usage
  grouped by structural success/failure, with coverage caveats per origin.
- Promote claim-vs-evidence demos: PR/session claims should link to structural
  tool outcome rows and raw evidence, not assistant prose.
- Improve DSL projection/composition: unit queries can filter through
  `session.*`, but result shape is still fixed. Add declared projections and
  attached unit composition where it helps read packages and demos.
- Surface query completions and DSL explanations in the web UI search box; the
  backend already has query-completion and expression-explain substrate.
- Treat `.agent` conductor state as a product dogfood target: focus
  transitions, handoffs, and run state should eventually be archive/assertion
  data rather than only markdown sidecars.

## Follow-Up Audits

- Schema-policy drift: `fts_freshness_state` and telemetry tables were flagged
  as possible duplicates or policy exceptions worth verifying against current
  DDL and readers.
- Provider/origin vocabulary: continue retiring provider tokens from non-wire
  public/read surfaces.
- Renderer drift: web reader JavaScript rendering and exported HTML rendering
  should converge on one message-rendering contract.
