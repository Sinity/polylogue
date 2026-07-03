# Fables Poly Findings

Source: `/realm/inbox/fables-poly.md`, reviewed 2026-07-03.

This note keeps only findings that survived quick source review or are useful
devloop backlog shape. The restored transcript remains at
`/realm/inbox/fables-poly.md`; this is the shorter operational extraction.

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

## Strong Product Reading

- Polylogue's stable center is not "chat export to Markdown"; it is a
  local-first evidence system for AI work: source truth, derived archive,
  query/read surfaces, assertions, provenance, and agent resume context.
- The strongest moat is construct validity. Claims should resolve to structural
  evidence: tool outcomes, provider usage events, raw source refs, message IDs,
  and explicit coverage tiers. Any demo or report that relies on prose-mined
  assistant self-report needs to say so or be replaced.
- The current breadth is high: CLI, MCP, API, daemon web reader, read packages,
  demos, devloop scripts, assertions, embeddings, run projections. The leverage
  now is last-mile composition: make the existing substrate visible, fast, and
  opinionated in the surfaces people actually use.
- The devloop itself is a product testcase. If the conductor needs active loop
  state, handoffs, focus transitions, proof claims, and velocity notes, those
  should become normal archive/query/assertion concepts rather than permanent
  sidecar-only lore.

## Demo Priorities

1. Claim-vs-evidence on real PR/session claims. Pair durable text claims with
   structural tool-result rows and raw references. This is the current active
   proof campaign.
2. Behavioral archaeology via DSL: outcome-conditioned tool/model/repo queries,
   thrash-loop sequence queries, abandoned-session queries, and direct read
   composition from query results.
3. Two-arm uplift experiment: same abandoned/interrupted task with and without
   compiled Polylogue handoff/context. Measure structural deltas: rediscovery
   actions, failures, turns to productive action, wall time, and token/cost.
4. Outcome-conditioned cost and tool analytics: cost by origin/model/tool and
   by success/failure state, with exact/estimated/unsupported coverage chips.
5. Live self-watching session: a running agent queries its own captured session
   with ingest-cursor latency and produces its own postmortem.
6. Candidate assertions on trial: detected lessons/pathologies become
   promotion-required candidates; accepted ones affect future context, rejected
   ones do not.

## DSL And Composition Direction

- The query DSL already has the right extension seams: unit fields,
  `session.*` upward predicates, `EXISTS`, `SEQ`, terminal stages, transform
  stages, and attached-unit machinery. It needs those reserved slots filled
  rather than a second language.
- Highest-leverage DSL additions:
  - terminal projections such as `| read view:temporal`, `| context-image`,
    and `| bundle:handoff`;
  - `with messages(...)` / `with actions(...)` attached evidence beyond the
    current assertion-only path;
  - a `fields`/`select` transform so unit queries can emit parent fields such
    as `session.repo` and `session.title`;
  - real aggregates (`sum`, `avg`, `p90`, multi-field `group by`,
    time-bucket groups);
  - correlated child counts such as `sessions where count(action where
    is_error:true) >= 5`;
  - SEQ span capture, adjacency/window modifiers, and repetition so sequence
    matches can flow into read/context projections.

## UX And Performance Direction

- Web UI: expose DSL completions and query explanation at the search box,
  add an analytics route over the same query/projection substrate, add live
  follow mode for active sessions, and converge web/export message rendering.
- CLI: completions should be obvious on init and in interactive hints; fzf
  selection should cover ambiguous read/action moments, not only `select`;
  empty-result guidance should explain which clause killed the result.
- Performance: a CLI-to-daemon read fast path is the biggest single UX win.
  It would reuse daemon DSL compilation/read endpoints, avoid cold Python
  storage imports, avoid cold SQLite opens, and keep direct SQLite as fallback.
- Operational perf checks should be normal status/doctor facts: WAL size,
  planner-stat presence, FTS freshness verdict cost, and query plan drift.

## Follow-Up Audits

- Schema-policy drift: `fts_freshness_state` and telemetry tables were flagged
  as possible duplicates or policy exceptions worth verifying against current
  DDL and readers.
- Provider/origin vocabulary: continue retiring provider tokens from non-wire
  public/read surfaces.
- Renderer drift: web reader JavaScript rendering and exported HTML rendering
  should converge on one message-rendering contract.
