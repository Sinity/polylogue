---
created: "2026-06-29T00:00:00+02:00"
purpose: "Extract development intel from ChatGPT browser-capture sessions relevant to Polylogue/Sinex devloops"
status: "active"
project: "polylogue"
---

# Browser Capture Devloop Intel

## Scope

Read the ChatGPT browser-capture directory at:

`/home/sinity/.local/share/polylogue/browser-capture/chatgpt/`

There are 126 captured JSON sessions. I did not treat every file as equal.
The sensible split is:

- Strategic long sessions: read/ranked/extracted for design and devloop doctrine.
- Short implementation-result sessions: treated as provenance receipts for lanes
  already attempted by other agents.
- Unrelated or generic personal sessions: skipped unless keyword matches showed
  direct Polylogue/Sinex relevance.

Most relevant captures:

- `6a2ea930-87a8-83eb-adb7-4cf38ccaa6a3-05bdfeaa1571.json`
  - `AAA Provisional central planning center for polylogue and sinex`
  - 3475 turns; central strategic session.
- `6a2d9333-347c-83ed-b8ef-ff3cf99f1d78-cef138c6c70f.json`
  - `Analyze Chatlog Against Projects`
  - 1708 turns; broad repo/project analysis.
- `6a232355-ac3c-83eb-a93d-9c70697bfc18-ed8cb11eeb05.json`
  - `Analysis and Report Request`
  - 577 turns.
- `6a398eb4-15ac-83eb-b987-9a0ef2596eb7-c857dc95533d.json`
  - `High Lifetime Token Use`
  - 297 turns.
- `6a3ae33a-ca4c-83ed-986e-d717099436a9-49f8112b86c3.json`
  - `Lynchpin Structure Overview`
  - 86 turns.
- Short but important lane receipts:
  - `tt - Polylogue Browser Capture Update`
  - `tt - Polylogue agent identity patch`
  - `tt - Source identity demotion`
  - `tt - Prototyped ResourceBudget slice`
  - many `Implemented ...` Polylogue patch receipts from 2026-06-23/24.

## High-Level Understanding

The captured sessions reinforce the user's corrected devloop goal:

> Build Polylogue and Sinex rapidly toward useful, impressive, compounding demos,
> using real development work and agent transcripts as the steering signal, while
> maximizing dev-loop velocity.

Dogfooding and self-analysis are instruments, not the goal. The goal is a stack
that becomes more capable because each demo, issue pass, agent run, context
recovery, and failure analysis leaves behind reusable capability.

## Polylogue Direction

Polylogue is not merely an archive. In the intended shape, it is the memory,
identity, observability, and recovery layer for AI work.

The strongest product lens is:

- raw transcript capture is ground truth;
- transformed read models are the normal working surface;
- the raw archive is the appeal court;
- the public/demo surface should show agent work as recoverable, inspectable,
  accountable runs rather than opaque chat logs.

Key constructs from the captures:

- `Session` is the atomic agent identity more than a prompt template is.
- Durable agent identity is likely an equivalence class of sessions plus naming,
  lineage, memory scope, and communication affordances.
- `RoleSpec` / prompt template is closer to executable image.
- `ContextEnvelope` is closer to argv/env/mounted config.
- `Run` is process lifetime tied to wallclock, cwd/worktree, harness, model,
  terminal/browser, and tool surface.
- `ObservedEvent` / tool spans / messages are the flight recorder.
- OTel fits because it already models traces, spans, parent-child execution,
  timing, attributes, and cross-service correlation. It should be used as an
  execution tree vocabulary, not as generic metrics decoration.

The user rejected bloated over-modeling several times. In current implementation
work, prefer the smallest executable vocabulary that gives lineage and recovery:

- run/session refs;
- object refs;
- context snapshots;
- observed events/tool spans;
- selected KV state;
- clear transforms/read views.

Avoid inventing heavyweight Principal/Mailbox/Actor systems unless the code
actually needs them.

## RunState / KV

The captures repeatedly converge on a structured mutable state layer:

- Transcript = full append-only log.
- RunState/KV = current working memory, checkpoints, annotations, marks, task
  state, and recovery facts.

The useful shape is probably not many bespoke side tables for every note-like
thing. A typed/evidence-linked KV can represent:

- highlights;
- notes;
- marks;
- agent self-state;
- task/checklist state;
- recovery breadcrumbs;
- user annotations;
- issue/PR coordination overlays.

This is not a reason to replace all durable GitHub/project state. It is a better
agent-facing substrate that can sync to GitHub/issues/docs where appropriate.

## Context Window / Transforms

The sessions identify transforms as a core missing/opportunity area:

- Transform raw sessions into smaller evidence-linked recovery images.
- Separate user intent, decisions, code changes, verification, failures,
  tool results, and low-value chatter.
- Make "context injection every turn" possible, but only through a salience and
  recovery compiler that knows what belongs in the current window.

Important tension:

- Removing/reordering high-context text can hurt prompt-cache economics.
- But compaction can still be valuable for cognition, recovery, and relevance.

The right demo is not "summarize chat." It is:

1. recover a stopped agent run;
2. identify what it was trying to do;
3. show decisions, changed files, blockers, verification, residuals;
4. produce a next-agent context pack small enough to use immediately;
5. link back to raw evidence.

## Browser Capture Implications

Browser capture is not just another importer. It is a live proof source for:

- ChatGPT project/session provenance;
- web UI sessions not present in local CLI logs;
- external model runs that generated patches/reports;
- demo-ready evidence that Polylogue can ingest agent work across harnesses.

The short `tt` lane receipts show intended concrete closure around browser
capture:

- browser-capture dataflow should be documented end to end;
- receiver boundary should be typed and tested;
- malformed JSON and semantically invalid envelopes should fail distinctly;
- status payloads should expose whether auth is required;
- capture IDs, source, schema version, and receiver identity should be stable
  DTO fields.

This matters for demos because a browser-captured ChatGPT project can be used as
material for a recovery/context demo without needing native ChatGPT export first.

## Sinex Direction

The Sinex-relevant captures sharpen the "admission-first" reading:

- Source is often too overloaded.
- Source/source_id should not stand for material origin, parser identity, schema
  identity, privacy authority, occurrence identity, deployment grouping, or
  admission authority.
- Source can remain a namespace/display/index coordinate where useful.

The repeated "silo vs algebra" point applies directly here:

- Do not create one-off source-specific pathways when an algebraic admission,
  provenance, view, budget, or transform primitive can serve many surfaces.
- Refactoring is good when it removes divergent local handling and promotes
  repeated behavior into shared compositional primitives.

Relevant lane receipts:

- Source identity demotion: map source/source_id usages and fence authority
  away from source.
- ResourceBudget: make pressure response executable and operator-visible,
  not just metrics. It may throttle/defer/sample/pause/create debt under policy,
  but must not silently change semantics or censor data.

## Cross-Stack Demo Implications

Polylogue alone is superior for transcript-native recall. Sinex+Polylogue is
superior only when the demo joins transcript evidence with non-transcript facts:

- git commits and diffs;
- shell/terminal history;
- system pressure;
- browser/window focus;
- test/build results;
- source-material/provenance/coverage/caveats.

Therefore the cross-stack demo should not be "Sinex imports Polylogue and shows
the same chat better." It should be:

- Polylogue gives the agent-run/session/read-view truth.
- Sinex contributes event/provenance/time/resource/source evidence.
- The combined view answers a question neither project answers alone:
  "What happened in this development loop, what changed, what evidence proves it,
  what failed, and what should the next agent do?"

## Implementation Receipts Worth Mining

The capture directory contains many short "Implemented ..." sessions. Treat
them as patch/provenance receipts, not as primary doctrine:

- query DSL patch;
- query grammar completion;
- provider usage coverage/accounting;
- read-view HTTP contract;
- route readiness update;
- dev-loop ergonomics;
- assertion review slice;
- residual-map lane;
- schema hygiene;
- config diagnostics;
- browser-capture DTO/auth boundary;
- agent identity orchestration;
- source identity demotion;
- ResourceBudget slice.

Action: before rebuilding any similar feature, search these captures and the
current repo for whether the patch was already applied, partly applied, or
rejected. The current Polylogue working tree is dirty with many related files,
so do not overwrite blindly.

## Current Work Order Implications

1. Keep the demos packet alive and improve it with real, runnable examples.
2. For Polylogue, prioritize read-view/context/recovery surfaces that compress
   raw sessions into external consumer artifacts.
3. For Sinex, reuse/promote shared view primitives and admission/provenance
   algebra rather than adding CLI-private or source-specific shapes.
4. For cross-stack work, keep the bridge thin until Sinex is actually ready to
   become a backend. Do not build a provisional import layer that merely
   duplicates Polylogue's own read strengths.
5. Use browser-captured ChatGPT sessions as demo material and as cross-harness
   evidence, especially for the project `AAA Provisional central planning center
   for polylogue and sinex`.
6. When uncertain, prefer one narrow executable closure slice over a design-only
   memo, but preserve the reasoning in scratchpads.

## Caveats

- This note is a sensible extraction, not a complete semantic read of every
  captured token.
- Browser captures include reasoning/tool artifacts and duplicate branches.
  They need transforms before they become a clean product read model.
- Some short patch receipts may describe patches that were never applied or were
  superseded. Treat them as leads until verified against the current repo.
