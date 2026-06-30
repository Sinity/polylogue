# Polylogue Conductor Devloop

This is the tracked contextless resume entrypoint for the active Polylogue
dogfood/demo loop.

The process docs and helper scripts are tracked. Current loop state is local and
ignored: `ACTIVE-LOOP.md`, `OPERATING-LOG.md`, `EVENTS.jsonl`, `DEMO-RADAR.md`,
generated manifests, demos, scratch notes, and task history are not meant to be
committed.

If the current chat history is cleared and the operator says only:

```text
continue the devloop setup in .agent
```

then do this:

1. From `/realm/project/polylogue`, run `.agent/scripts/devloop-status`.
2. Run `.agent/scripts/devloop-review`.
3. Read `RUNBOOK.md`.
4. If local `ACTIVE-LOOP.md` exists, read it and continue that slice.
5. If local state is absent, start in `Direction`: choose a slice from the goal,
   current repo state, and any available archive/daemon evidence.
6. If review reports stale generated scaffold state, run
   `.agent/scripts/devloop-sync` and review again.
7. Use the focus modes in `RUNBOOK.md`: Direction, Evidence, Construction,
   Proof, Artifact, Velocity, and Meta.
8. Record material transitions with `.agent/scripts/devloop-focus`; start new
   slices with `.agent/scripts/devloop-start`.
9. Before ending, refresh/review and leave `ACTIVE-LOOP.md` plus
   `OPERATING-LOG.md` resumable.

The process goal is indefinite: continuously choose the highest-value
live-archive capability slice, produce inspectable artifacts proving Polylogue
improves agents with real history, collapse silos into shared
acquisition/query/projection/rendering substrate, verify on the active archive
or live capture, maintain logs and handoffs, and reprioritize by evidence.

Default state lives here, not in `/realm/inbox`. The default demo shelf is
`.agent/demos`, and it is current-curated rather than append-only.

## How The Devloop Works

The conductor loop is a lightweight operating system for long-running
Polylogue dogfood work. It is not a one-shot task plan. Its job is to keep the
agent oriented around live evidence, current demos, reusable substrate, and
process health across context loss and across many small slices.

The loop has seven focus modes:

- `Direction` chooses the next capability slice.
- `Evidence` inspects the current source tree, active archive, daemon state,
  logs, issues, docs, and demo artifacts.
- `Construction` edits code, docs, scripts, or artifacts.
- `Proof` verifies the exact claim with the narrowest sufficient command.
- `Artifact` makes the result inspectable outside chat, usually under
  `.agent/demos` or in the conductor packet.
- `Velocity` removes or records friction that slows the next loop.
- `Meta` improves the devloop itself when the process drifts or the operator
  corrects the agent.

Material focus changes should be logged with:

```bash
.agent/scripts/devloop-focus <from> <to> "<trigger>" "<decision>"
```

The normal slice flow is:

1. Run `.agent/scripts/devloop-status` and `.agent/scripts/devloop-review`.
2. Read `ACTIVE-LOOP.md` when present.
3. Pick one slice from current evidence, not from stale memory.
4. Record the slice with `devloop-start` or `devloop-log`.
5. Gather evidence before editing.
6. Make the smallest coherent shared-substrate or artifact change.
7. Prove the specific claim.
8. Update the demo shelf or conductor state when the result should survive.
9. Run `devloop-sync` and `devloop-review` before claiming a clean checkpoint.

The proof ladder is:

- source review proves shape;
- focused tests prove parser/storage semantics;
- CLI/API/MCP/daemon probes prove surface contracts;
- real archive artifacts prove operator value;
- broad `devtools verify` proves phase readiness, not every small edit.

## State Files And Their Roles

- `ACTIVE-LOOP.md` — current slice, focus transition, accepted warnings, and
  next action.
- `OPERATING-LOG.md` — timestamped log of decisions, evidence, actions, proof,
  and next decisions.
- `EVENTS.jsonl` — structured event sidecar generated from the operating log.
- `DEMO-RADAR.md` — demo candidates, selected artifact, proof/caveat, and next
  demo question.
- `RUNBOOK.md` — operational protocol and focus-mode rules.
- `PROCESS.md`, `TACTICS.md`, `VELOCITY.md` — compact process, tactical, and
  speed rules.
- `SELF-PROMPTS.md` — durable goal, primary self-prompt, adversarial prompt,
  and tactical prompt.
- `ADVERSARIAL-REVIEW.md` — known failure modes and mitigations.
- `INDEX.md` — routing guide for the historical notes and audit files in this
  packet.

Current local state is intentionally ignored by git. The tracked scaffold tells
a future agent how to resume; the ignored state tells it where this exact loop
currently is.

## Backlog And Prioritization Model

The devloop does keep future workload, but it is deliberately evidence-shaped
rather than a static ticket queue. The current priority comes from:

1. `ACTIVE-LOOP.md` — immediate slice and next action.
2. `DEMO-RADAR.md` — current demo pressure and the next demo question.
3. `OPERATING-LOG.md` — recent findings and next decisions.
4. `INDEX.md` plus the audit notes — larger known debt and architectural
   threads.

Prioritize work in this order unless fresh evidence contradicts it:

1. Fix anything that makes live archive evidence false or ambiguous: wrong
   archive root, stale schema, duplicate daemon, stale generated state, or
   misleading counts.
2. Fix user-facing lies and construct-invalid wording: e.g. saying "pasted
   content" where the archive only proves paste evidence.
3. Repair query/projection/rendering substrate when a useful demo or CLI/API
   workflow would otherwise be bespoke or false.
4. Produce or refresh a current demo that proves a capability on the canonical
   archive.
5. Improve performance/velocity when command latency or host pressure is
   slowing repeated loops.
6. Do meta/scaffold work only when it leaves an executable consequence or a
   materially better next-slice decision.

Known future workload classes:

- Audit and clean the CLI by running real commands, especially help, status,
  query, read, select, facets, and timing-sensitive paths.
- Continue collapsing legacy read/export/recovery flags into query,
  projection, and rendering primitives.
- Clarify and implement the projection/render composition model.
- Keep `.agent/demos` as a current curated shelf, not an append-only archive.
- Extend temporal devloop analytics so agents can reason about progress and
  friction from normal Polylogue surfaces.
- Improve affordance/tool-use analytics, especially distinguishing MCP calls,
  shell/process evidence, edit-content mentions, and documentation mentions.
- Revisit insights for construct validity, unnecessary materialization, and
  delayed/duplicated computation.
- Keep root/dotfolder/docs organization clean enough that a future agent can
  resume without inbox mirrors or stale scratch ambiguity.

## Projection / Render Composition Direction

The target model is two-layer composition:

1. A query expression selects sessions or query units.
2. A projection/render expression describes what to attach, how to window it,
   and how to render it.

Selection examples:

```text
sessions where repo:polylogue and has:paste
actions where tool:bash and text:pytest
messages where role:user and time >= 2026-06-01
```

Projection/render examples that should become first-class rather than
flag-sprawl:

```text
with messages window around matches limit tokens 4000
with actions grouped by tool
with assertions attached to sessions
render layout:context-image timestamps:include-available
render format:markdown destination:browser
```

Current product reality is partway there:

- the query DSL has session predicates and terminal query-unit sources;
- `read --spec` exposes selection/projection/render contracts for several
  views;
- `ProjectionSpec` and `RenderSpec` are already visible in read outputs;
- context-image, temporal, chronicle, dialogue, neighbors, and correlation
  views exercise pieces of the model;
- many view-specific flags still mix selection, projection policy, rendering,
  and delivery.

Open design questions:

- Should projection syntax live inside the query DSL (`with ...`) or as a
  second expression passed to `read`?
- Which controls are selection cardinality versus projection body/window/edge
  limits?
- How should query-unit pipelines attach related units to sessions without
  one-off context/recovery/export DTOs?
- Which render controls belong in a stable `RenderSpec` instead of per-view
  flags?
- How do machine-readable JSON/YAML outputs preserve enough projection/render
  metadata without bloating dialogue/export artifacts?

Working answer until contradicted by implementation evidence:

- Keep **selection** in the query DSL: origin, repo, time, text, lineage,
  session ids, query-unit predicates, sort, and selected-session cardinality.
- Keep **projection policy** in a typed projection expression/spec: attached
  units, body/window/edge limits, token budgets, omission policy, evidence
  families, and per-view computation knobs.
- Keep **rendering** in `RenderSpec`: layout, output format, timestamp policy,
  redaction, destination, and machine-vs-readable shape.
- Allow CLI flags only as ergonomic shorthands that compile into those typed
  contracts. A flag that cannot be represented in the spec is either a real
  substrate gap or compatibility debt.

Suggested conceptual grammar:

```text
<query> then read --projection "<projection-expr>" --render "<render-expr>"

projection:
  with assertions
  with actions where tool:bash
  with messages window around matches limit tokens 4000
  with temporal buckets hour families session,git,devloop

render:
  layout context-image
  format markdown
  timestamps include-available
  redact paths
  destination browser
```

This does not require the first implementation to add literal
`--projection`/`--render` flags. It does require every new read capability to
be explainable as:

```text
SelectionSpec + ProjectionSpec + RenderSpec -> payload + readable rendering
```

### Projection/Render Workload Map

Use these vertical slices to turn the model into product behavior:

1. **Inventory and classify current read controls.**
   - Input: live `read --help`, `read --views --format json`, representative
     `read --spec` outputs.
   - Output: table classifying each option as selection, projection, render,
     delivery, or compatibility debt.
   - Proof: artifact under `.agent/demos` plus a small test or verifier that
     `read --views` still exposes ownership metadata.

2. **Attach one related unit through projection substrate.**
   - Candidate: `with assertions` first, then `with actions`.
   - Required shape: selected sessions carry `attached_units` or equivalent
     query-unit envelopes; fetching is page-batched, not per-session bespoke
     loops.
   - Proof: live command showing the same session selection with attached
     assertions/actions in JSON and readable Markdown.

3. **Unify query-set read cardinality.**
   - Clarify when `limit` means selected sessions versus projection body/edge
     count.
   - Ensure `read --spec` reports the distinction.
   - Proof: focused tests plus live examples for `dialogue`, `temporal`,
     `chronicle`, `context-image`, and `messages`.

4. **Make render layout explicit everywhere.**
   - Ensure context-image, temporal+chronicle handoff, dialogue, and raw
     outputs carry stable `render.layout`, format, timestamp, and redaction
     policy where applicable.
   - Proof: generated JSON/YAML stays compact enough for large sessions while
     Markdown remains inspectable.

5. **Remove obsolete compatibility surfaces.**
   - Recovery/export/context-pack/read flags should disappear when their
     behavior is represented by query + projection + render specs.
   - Proof: source search for old public tokens, regenerated docs, and one live
     replacement command.

### Acceptance Criteria For A Good Slice

A projection/render slice is good only if it satisfies all of these:

- It uses the canonical active archive or a clearly named fixture.
- It states root, schema version, and relevant counts when quoting data.
- It can be inspected outside chat.
- It improves or removes a user-facing command, spec, payload, doc, or demo.
- It avoids new one-off DTOs unless the DTO is the general projection/render
  contract.
- It records what the proof does not show.

Useful next slices:

1. Audit current `read --help`, `read --views`, and `read --spec` output to
   classify every option as selection, projection, render, delivery, or
   compatibility debt.
2. Pick one concrete projection family, likely `with assertions` or
   `with actions`, and route it through shared query-unit payload machinery.
3. Make one live demo prove the same artifact can be regenerated from query +
   projection + render specs instead of a bespoke report.
4. Remove or rename any public flag/DTO/route whose only remaining purpose is
   compatibility with an obsolete recovery/export/context-pack surface.

## Meta And Self-Improvement

The devloop has an explicit `Meta` mode. Use it when the operator corrects the
agent, repeated friction appears, archive/process state becomes confusing, or a
loop feels vague. Meta work is only valuable if it changes future behavior.

Acceptable meta outcomes:

- a stronger `devloop-review` check;
- a sharper `devloop-status` or `devloop-velocity` signal;
- a corrected README/RUNBOOK/SELF-PROMPTS instruction;
- a cleaned scratch/current-state boundary;
- a new executable tripwire for stale archive roots, duplicate daemons, stale
  generated state, or empty log entries;
- a better prioritization rule grounded in evidence.

Non-outcomes:

- apology prose;
- broad process notes with no executable consequence;
- adding ceremony that does not speed up or harden the next slice;
- preserving compatibility trash under a new name.

When in doubt, make the next loop harder to derail and easier to resume.
