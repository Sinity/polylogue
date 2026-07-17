# Ordered next actions

## P0 — before the next broad campaign wave

### 1. Close the historical capture-evidence gap or mark it irrecoverable

**Owner:** `polylogue-3v1`, with generic action correlation under `polylogue-ptx`.

Deploy/replay the #2930 replacement/reingest path against the 27-conversation campaign census. Produce one immutable receipt per conversation containing current provider/native identity, observed terminal turn, offered output attachments, captured attachment refs/blobs, and replay outcome. For each of the 28 package revisions, either match the exact SHA-256 to canonical capture or record that historical recovery is impossible and preserve the manual-download receipt as degraded custody.

**Acceptance:** No campaign conversation is silently absent; all 27 have an explicit terminal/capture outcome. Every offered output is acquired or has a typed failure. The index/raw projection matches the current source files after replay, and unchanged poison observations remain excluded while changed replacements revive. `polylogue-3v1` is not closed until its remaining Gemini and durable capture-gap criteria are independently adjudicated.

**Local verification:** run the production watcher/membership/replacement tests from #2930; run a closed-tab live output fixture through ordinary `/v1/browser-captures`; query the durable capture-gap projection; compare canonical artifact hashes to manual custody.

### 2. Land result schema v2 as an extension of the existing campaign model

**Owner:** `polylogue-yyvg.6`.

Replace `.agent/handoffs/external-agent-campaigns/schemas/result.schema.json` with the reviewed version derived from `RESULT-TELEMETRY-V2.schema.json`. Keep existing stable IDs and fields; add action/run identity, prompt shape, snapshot details, package composition, immutable stage events, structured triage, repair, verification, PR, and Bead outcomes.

**Acceptance:** Existing v1 fixtures have an explicit migration path. New results validate with `additionalProperties: false`. Event timestamps require source and evidence refs. Unknown historical events are omitted or represented by a typed uncertainty record, never fabricated.

**Local verification:** JSON Schema validation for a normal implementation, same-chat repair, analysis-only response, manual-download degraded acquisition, rejection, and later Bead closure. Mutation fixtures must fail when terminal, artifact, merge, or closure events are deleted.

### 3. Backfill this campaign into immutable attempt receipts

**Owner:** `polylogue-yyvg.6`.

Create a one-time importer from `INTEGRATION-LEDGER.json`, `SHA256SUMS`, package ZIPs, Git/PR history, and Beads. Represent 26 package-bearing conversations, 28 package revisions, 45 download observations, four direct merges, the CaptureJobs reimplementation, and the parser-drift repair. Record the 27/28-session discrepancy as uncertainty rather than choosing one count.

**Acceptance:** Rebuilding the projection reproduces `FUNNEL.json`. Package 24 retains its original preserved-draft decision while its current owner projects closed through #2953. Direct PRs and merge commits are normalized. All imported timestamps state their source kind; canonical download timestamps are not labeled terminal or acquisition-start time.

### 4. Upgrade package triage without executing unreviewed tests

**Owner:** `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/triage-package.py` and then the shared campaign tool.

Retain the current required-member, copied-input, placeholder, and detached-worktree `git apply --check` behavior. Add result-v2 receipt writing, content composition, largest-member declarations, snapshot/prompt hash checks, duplicate hash detection, expected filename/revision checks, and route to one of implementation, repair, analysis/research, duplicate, rejected, or blocked.

**Acceptance:** Package 13’s synthetic SQL bulk and package 24’s oversized rollout-plan document are visible as dominant content; package 27’s 432-byte patch is not obscured by its fixture. The tool does not run `TESTS.md`. It creates `result.json` under `results/<job>/<attempt>/`, stores raw evidence immutably, and records a deterministic extraction receipt.

### 5. Make dispatch gating depend on campaign evidence state

**Owner:** `check-dispatch.py`, then the shared campaign orchestrator.

Add gates for: authoritative snapshot hash; prompt hash; owning Bead/dependency status; no duplicate active mission footprint; prior-wave terminal/acquisition reconciliation; integration WIP; and result-schema compatibility. Keep the current foundation and context-manifest checks.

**Acceptance:** The command names every unmet gate. A missing terminal artifact, stale snapshot, closed/superseded owner, overlapping write set, or over-cap integration queue blocks only affected jobs. Analysis/research jobs are not blocked for lacking a patch lane.

## P1 — prompt, routing, and integration changes

### 6. Add explicit mission shape and integration contract to rendered prompts

**Owner:** shared contracts under `.agent/handoffs/external-agent-campaigns/contracts/` and `render_prompts.py`.

For implementation prompts, require: current snapshot commit/archive hash; one primary Bead and any dependency owners; named production route; allowed and forbidden paths; existing interfaces that must be reused; exact residual scope; one anti-vacuity mutation; expected result filename/revision; and a machine-readable result receipt. State that a coherent smaller slice is preferred to a parallel framework.

For repair prompts, include the prior attempt/package hash, exact failed review/test receipt, current source delta, and `supersedes_attempt_id`. For analysis/research prompts, require decision ownership, falsification evidence, and no placeholder patch.

**Acceptance:** A prompt renderer check proves every job has stable title, result identity, prompt hash, owner, shape, and snapshot. Same-chat continuation renders a new attempt/revision rather than overwriting the old result.

### 7. Route before generation

**Owner:** workload coordinator and owning Beads.

Use implementation only when dependencies are landed and the production route is current. Route cross-cutting identity, provider-policy, architecture reconciliation, and dependency-premature kernels to analysis/research. Route known review failures to same-chat repair rather than fresh duplicate chats. Reject missions that ask an external package to certify live deployment, daemon state, provider policy, performance, or destructive-route completeness without the required local evidence.

**Acceptance:** Every job has one disposition hypothesis before launch and a named local owner. The hypothesis can be disproved at triage but cannot be absent. Competing seams and duplicate mission footprints are detected before dispatch.

### 8. Standardize the local integration receipt

**Owner:** integration coordinator; `results/<job>/<attempt>/integration/`.

Record: fresh base commit; worktree/branch; package patch apply status; current-source reconciliation; retained, rewritten, and discarded paths/lines; repair reason; focused production-route tests; anti-vacuity mutation; broader gate outcome and inherited failures; PR; merge; deployment if in scope; and Bead outcome.

**Acceptance:** Package-to-final retention and local repair effort become measurable. A reimplementation such as #2953 is visibly different from a direct patch merge. A later correction such as #2957 links to the original delivery track without inflating package merge count.

### 9. Synchronize Bead closure with delivered scope, not PR existence

**Owner:** owning Bead maintainers and projection builder.

At merge, record which acceptance criteria are satisfied, which residual Bead owns the rest, and whether closure follows. Keep partial slices open. Rebuild current campaign outcomes from Bead state rather than rewriting immutable package decisions.

**Acceptance:** #2922 projects as merged with owner open; #2924 as merged with owner in progress; #2925 as merged with owner closed; #2953 as local reimplementation with owner closed. No report treats those as equivalent closure outcomes.

## P2 — campaign cadence and reporting

### 10. Run a 3–5 attempt evidence pilot, then expand conditionally

Launch a mixed pilot after P0: two narrow implementation jobs, one same-chat repair, and one or two analysis/research jobs. Do not expand until every attempt is terminally reconciled, every expected asset has a typed outcome, every package is validated/triaged, and the integration queue is at or below the 3–4 active-worktree cap.

After that gate, 8–12 dependency-disjoint implementation attempts are reasonable. Generation can be parallel; shared architecture decisions, fixtures, local execution, mutation proof, and publication remain serialized or lane-bounded. A second wave launches only when prior results have current owners and no unresolved capture gap.

### 11. Publish a campaign effectiveness report from receipts, not prose

**Owner:** `polylogue-yyvg.6`.

Generate counts and duration distributions for submission success, first/terminal turn, acquisition, validation, triage, repair, verification, PR, merge, and Bead closure. Stratify by prompt shape, dependency readiness, current-source apply result, package content composition, repair/fresh attempt, and owning area. Keep descriptive association separate from causal claims.

**Acceptance:** The report can reproduce this campaign’s known counts while marking missing terminal evidence. It must reject impossible chronology, such as a PR supposedly starting after an acquisition timestamp that occurs later, unless an alternate evidence source is explicitly recorded.

## Stop conditions

Pause dispatch when any of the following occurs:

- terminal output exists without a canonical acquisition or typed failure;
- snapshot or prompt hashes drift after launch;
- duplicate mission footprints or same owner/write set are active without declared coordination;
- the untriaged result queue exceeds available integration lanes;
- package validation finds copied inputs, undeclared bulk, placeholders, unsafe paths, or a patch that cannot apply to the claimed snapshot;
- current Beads/source invalidate the mission;
- provider rate/safety circuits lack typed receipts or cooldown evidence.

## Expected value

The P0 work has high value because it converts the principal campaign uncertainty—terminal/output custody and stage timing—into measurable evidence and prevents recurrence. Prompt wording refinements alone have moderate value; the current contracts are already strong. Scaling generation before capture replay, immutable receipts, and integration gating would mostly increase unmeasured queue depth rather than trustworthy delivered yield.
