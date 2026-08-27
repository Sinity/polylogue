# Full-archive dogfooding

This is the public operating record for sustained dogfooding of a sealed
Polylogue archive. It defines what to ask, how to preserve evidence, and how
observed value changes product priorities. Private archive content, identifiers,
transcripts, paths, and raw exports stay outside this repository.

## State

Queue version: `2026-08-27.v1`

The queue is prepared before reindex completion. The three live cycles are
launch-gated on final reindex proof and a healthy archive read path. No live
cycle is claimed by this document until its complete redacted packet exists in
the operator's private evidence store.

## Question queue

Each question has a required evidence standard and an independent fallback.
The fallback is a falsifier or comparison method, not a second source of
authority.

| ID | Question | Class | Evidence standard | Independent fallback |
| --- | --- | --- | --- | --- |
| Q01 | Which sessions contain the decision that changed the current project direction? | Session archaeology | Ordered authored messages, session refs, lineage status, and the decision's source evidence | Search retained raw exports by distinctive terms, then verify each candidate in the archive |
| Q02 | What did an agent actually do for a named task, including commands, outcomes, and retries? | Agent-process analysis | Paired tool-use and tool-result blocks, structured exit/error fields, timestamps, and session refs | Inspect the provider JSONL event stream and reconcile command IDs manually |
| Q03 | Why did the current implementation take its present shape? | Project history | Linked sessions, authored rationale, file references, and chronology with fork/resume provenance | Compare Git commits and contemporaneous session exports |
| Q04 | What did this work cost, and which usage fields support the number? | Costs and usage honesty | Provider usage events, cost-model version, currency assumptions, and an explicit unknown bucket | Recompute from raw provider usage records with a separate spreadsheet or script |
| Q05 | Which failures were observed, and which were only asserted in prose? | Failures | Structural tool outcomes, exit codes, raw evidence refs, and a prose-only negative control | Re-run the relevant command or inspect raw events without keyword matching |
| Q06 | What work was left unresolved across sessions? | Continuity | Declared open/close evidence where present, lineage composition, right-censoring, and missing-evidence status | Build a chronological list from raw exports and mark every inferred closure as unknown |
| Q07 | Which prior answer or assertion should be trusted, and why? | Annotated findings | Assertion identity, provenance, judge or operator decision, schema version, and contradiction links | Read the cited source messages and independently classify the claim |
| Q08 | What changed in the project during a selected period, and which session explains each change? | Project history | Git refs joined to session/file evidence, with unmatched changes reported separately | Use Git history and raw session search without the archive join |
| Q09 | Did a proposed fix actually resolve the failure? | Failures | Before/after structural outcomes, target linkage, and a reproducible verification result | Re-run the test or command in a clean checkout and compare its result |
| Q10 | How much operator context was repeated, omitted, or re-established? | Continuity | Authored message boundaries, lineage, context-delivery provenance, and an unknown category for unobserved context | Compare a bounded raw export with a separately compiled context package |
| Q11 | Which model or workflow was most useful for this kind of work? | Agent-process analysis | Fixed question frame, usage events, outcome evidence, and an independently recorded quality judgment | Blindly compare matched raw transcripts under equal budget and task order |
| Q12 | Which archive result is incomplete or misleading, and what product change would prevent that? | Annotated findings | Reproduction, exact query/input, missing evidence, unsupported claim, and owner mapping | Attempt the same question through raw data, provider tools, or export-to-model analysis |

## Cycle packet

Every cycle produces one private packet with this fixed record. A public note
may cite only the question ID, method, product gap, and redacted finding.

```yaml
queue_version: 2026-08-27.v1
cycle_id: <operator-private-id>
question_id: Q01
archive_state: sealed-full-archive | seeded-corpus | unavailable
surfaces:
  - cli | api | mcp | web | raw-export | provider-tool | export-to-model
query_or_prompt: <exact redacted input>
inputs: <filters, bounds, seed, model, and budget>
evidence_refs: <stable public refs or private receipt refs>
result: <answer or explicit unsupported result>
elapsed_effort: <wall time and operator effort>
unsupported_portions: <unknowns and excluded evidence>
product_friction: <wrong turns, confusing output, or missing affordance>
comparison: <method, answer quality, traceability, repeatability, interaction>
priority_review: <changed or unchanged priority and reason>
gap_disposition: existing-owner:<bead> | new-bead:<bead> | declined:<reason>
```

The exact query, inputs, evidence refs, result, elapsed effort, unsupported
portions, and friction are mandatory. A refusal is a valid result when the
archive lacks the required evidence. Numbers must resolve to structural
outcomes, usage events, provenance refs, or raw bytes. Prose is context, not an
oracle.

## Required cycles

After final reindex proof, run three complete cycles against the sealed full
archive. Select one question from each of these groups so the cycles cover
different value and failure modes:

1. Archaeology or project history: Q01, Q03, or Q08.
2. Process, failure, cost, or continuity: Q02, Q04, Q05, Q06, Q09, or Q10.
3. Findings or comparative value: Q07, Q11, or Q12.

At least one cycle must compare Polylogue with an export-to-model or direct
raw-data method. Report quality, traceability, repeatability, and interaction
separately. The comparison is allowed to favor the independent method.

The final cycle review must answer two questions: did the active product
priorities earn their current order from observed value and friction, and what
should be changed next? Record a reason for every priority change or
non-change. Repeated friction becomes an existing-owner mapping or one
distinct Bead. A symptom list is not a disposition.

## Initial priority ruling and simplification

Until the three cycles exist, keep archive truth, bounded queryability, and
evidence traceability ahead of presentation work. This follows from the
reindex gate and from the fact that a polished result cannot repair missing or
untrusted source evidence.

The first simplification is to use this queue and packet as the sole recurring
dogfood record. Do not build a bespoke dashboard, usage counter, or parallel
receipt database for this practice. Existing CLI/API/MCP/web reads, raw-source
fallbacks, and the durable audit/read contracts are sufficient inputs. Any
missing product capability is filed against its mechanism owner or as one
distinct Bead before a demo-specific workaround is retained.

## Public finding policy

Public records may contain question classes, methods, product gaps, redacted
findings, and citations to stable contracts. They must not contain private
transcripts, raw payloads, live identifiers, absolute paths, personal counts,
or archive-derived claims that cannot be reproduced from a committed fixture.
