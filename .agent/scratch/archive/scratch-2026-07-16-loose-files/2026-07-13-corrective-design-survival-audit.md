---
created: 2026-07-13
purpose: Verify that the multi-round corrective design survived Beads synchronization incidents
status: verified against merged PR #2834 and current live Beads state after gxjh fix
project: polylogue
source_proposal: .agent/scratch/2026-07-13-corrective-beads-proposal.md
git_baseline: merged origin/master 4e787df02 (PR #2834)
---

# Corrective design survival audit

## Conclusion

The design program survived. This is not inferred from the issue count.

- PR #2834 merged the final corrective state as `4e787df02`. Its Beads delta is
  50 semantic records (49 replacements plus new follow-up `gxjh.1`), not a
  whole-export rewrite.

- The 29 rows changed by corrective commit `c2948bc08` were compared field by
  field with the current 777-row export. Every non-empty corrective
  description, design, acceptance-criteria, and notes value from that commit
  remains an exact substring of the current corresponding field: **29/29**.
- The current export passes JSON parsing, the native dependency graph lint,
  and the broader backlog-hygiene lint with **zero findings**. The graph has
  zero cycles, zero priority inversions, zero duplicate labels, and zero
  missing acceptance criteria.
- The final inherited-old-binary replay was inspected through Dolt history.
  Relative to the immediately preceding real update, it changed no semantic
  issue fields. One `rxdo.5` content hash churned while its description,
  design, AC, notes, status, priority, timestamps, metadata, and close state
  remained equal. No unrecoverable live-only design row was lost.
- `gxjh` is fixed at the package source, deployed, and closed with a real
  stale-snapshot harness. The broader monotonic-import/receipt work was
  preserved on `gxjh.1`, so narrowing the incident did not discard scope.
- A final verification command inherited the pre-switch `bd` path and replayed
  the stale main-checkout JSONL once more. The Git artifact was unaffected. The
  50 merged records were then explicitly re-imported with the patched binary;
  all 50 matched the merged title/description/design/AC/notes/status/priority/
  type/labels fields exactly. The 29 previously at-risk corrective rows again
  passed exact non-empty field retention (29/29).

## Named-contract matrix

| Corrective contract | Durable owner(s) | Verified current state |
|---|---|---|
| Five-axis unification test | `cpf`, `o21` | identity, lifecycle, authority, access shape, and durability veto are all present; scaffolds default to shared protocol rather than unjustified durable object |
| Frame-exact result honesty | `rxdo.3`, `3uw`, `rxdo.9.8`, `9l5.7`, `bkzv` | enumeration, frame/coverage ref, measurement authority, definition/evaluation refs, privacy/degradation axes, and exact-with-incomplete-frame fixture all remain |
| Protocol-versioned query identity | `rxdo.2` | one language-wide version plus component refs, unsupported-version fail-closed behavior, planner-owned evaluator, dynamic definition/resolved run split, and no reverse compiler remain |
| Persist on promotion | `rxdo.2`, `rxdo.3` | ad-hoc execution does not create durable user-tier literals/member copies; saved/watched/cited/finding/experiment/pinned consumers promote explicitly |
| Provenance privacy and excision | `27m`, `kwsb`, `303r.6` | definitions, `@last`, relation members, findings, experiments, reports, vectors, exports, replicas, and backups are named excision surfaces under one lifecycle vocabulary |
| Three independent tag axes | `uh6c`, `dve1` | `tagged`, `tag_affinity`, and `tag_confidence` remain distinct and fail closed across axes; ontology facts retain separate schema/batch authority |
| Order-explicit patterns | `avna`, `avna.2` | embedded `EventOrderSpec`, typed units, partition/lineage/tie policy, observed/checkpointed/replay-verified grades, horizon, and actions-only PACK-A/B prerequisite remain |
| Horizon-honest goal state | `7yk5`, `9l5.9` | `unresolved_inactive(H)` replaces metaphysical abandonment; explicit close/block outrank inference; survival measures carry censoring/frame/authority |
| Context execution firewall | `37t.11`, `37t.15`, `cpf.3`, `xv1u` | disclosure and instruction authority are separate; ordinary knowledge is quoted evidence; only scoped, adopted, revocable `AssertionKind.POLICY` can enter the instruction partition |
| Non-total judgment | `rxdo.9.11`–`.15` | tie, incomparable, abstain, insufficient-evidence, partial order, disconnected components, exploration quotas, and actor-plus-execution-context calibration remain |
| Typed experiments without premature table | `stc`, `60i5` | `ExperimentDefinition` begins as a versioned typed assertion payload; assignment/exposure/stopping/exclusion/outcome semantics remain; schema promotion waits for two materially different consumers |
| Reusable improvement loops | `rxdo.11`, `37t.17` | `ImprovementLoopSpec` keeps real operational state; only L1 recall relevance and L2 classifier residue are initial pilots; no thirteen-loop daemon proliferation |
| Smallest measurable curriculum arm | `xv1u`, `stc`, `37t.11` | candidate-only renderer, source/exclusion receipts, no self-injection, matched experiment, and deferral of adaptive/autonomous optimization remain |
| Audit and continuity wedges | `hg8n`, `3tl.16`, `bby.15` | claims remain a finding/evidence view; cold-reader export is named; operator-flow receipts remain legitimate; AI-D3 is the first external activation |
| Honest demo claims | `rxdo.10.2`, `rxdo.10.3`, `212.6` | “prior observed recovery candidate,” “repeated-context mass/cost,” and descriptive actual resume remain; stronger fix/savings/productivity claims require state linkage or matched experiments |
| Qualified demo namespaces | `rxdo.10`, `212` | AI-D* and PF-D* identifiers remain distinct; historical unqualified names are local aliases only |
| Metric identity consolidation | `rxdo.9.1`, `9l5.7` | `MetricDefinition` owns identity; the statistics/measure layer consumes it and does not create a competing `MeasureSpec` identity |
| Claims, judgment, basket consolidation | `3tl.16`, `37t.12`, `bby.15` | claims are a finding view, judgment inbox is a workflow over one lifecycle, evidence basket is an export/workspace profile over selections and annotations |
| Marker syntax, not parallel objects | `37t.2`, `7yk5`, `stc` | marker kinds lower into owning services; markers are optional/advisory; declaration recall is measured before any mandatory policy; absence never blocks Stop or completion |
| Contention-class lane admission | `ei94`, `p155`, `2yax` | one migration writer per tier/window, one live archive writer, bounded generated-surface writers, four-heavy-lane backstop, read-only exemptions, and throughput-aware falsification remain |
| Cold-reader phase-2 artifact | `bby.15`, `hg8n` | minimal verified export shape is explicitly required for no-context audit readers; it is not promoted into a universal live bundle object |
| Falsification receipts | the owning corrective Beads | frame, identity, privacy, tag-axis, order, injection, judgment, loop reuse, curriculum, and lane-throughput claims each retain behavior-level failure cases in AC/design |

## Provisional operator decisions

All six provisional choices are present in current Beads:

1. `@last`: 48-hour non-sliding TTL, one slot per `(workspace, surface)`,
   independently excisable (`rxdo.3`).
2. Instruction authority: distinct `AssertionKind.POLICY`, not a boolean flag;
   kind alone does not self-authorize (`37t.11`).
3. Lane budget: adaptive per contention class, with four heavy lanes as the
   backstop and read-only work exempt only when resource-safe (`ei94`).
4. External activation: AI-D3 before PF-D8 (`hg8n`, `212.6`).
5. Experiments: typed assertion payload first; dedicated schema only after two
   consumers stabilize the lifecycle (`stc`, `60i5`).
6. Protocol identity: language-wide version plus component refs (`rxdo.2`).

## Dependency-shape check

The over-blocking warning from the script review was corrected rather than
ignored:

- `rxdo.2` relates to `27m`; the entire excision program does not hard-block
  the protocol-version P1.
- `rxdo.3` relates to `27m` and `3uw`; those contracts inform the result
  envelope without making the full programs a serial spine.
- `hg8n` uses related edges to claims, receipts, cold-reader export, and AI-D3,
  so the adoption proof is not hidden behind the whole design graph.
- `ei94`/`2yax`/`p155` share related edges rather than pretending their
  implementation lifecycles are identical.
- The native graph lint reports no cycles or priority inversions.

## The 29 exact-retention rows

`212`, `212.2`, `212.3`, `212.4`, `212.5`, `212.6`, `37t.2`, `3tl.5`,
`70qb`, `7yk5`, `avna`, `avna.1`, `bby.3`, `cijx`, `dve1`, `h6r`, `hg8n`,
`kph`, `rxdo.10`, `rxdo.10.1`, `rxdo.10.2`, `rxdo.10.3`, `rxdo.11`,
`rxdo.9.10`, `stc`, `xyel`, `yyvg.1`, `yyvg.2`, and `yyvg.3`.

This exact-retention test is intentionally narrower than the named-contract
matrix: it proves the rows known to have been overwritten once were restored
and stayed restored; the matrix proves the larger proposal’s semantics still
exist across all owners.
