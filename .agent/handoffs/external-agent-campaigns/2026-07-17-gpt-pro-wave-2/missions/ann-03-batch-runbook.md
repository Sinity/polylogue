Title: "Mass-annotation campaign runbook: how agent judges annotate a 13k-session archive without producing unusable labels"

Result ZIP: `ann-03-batch-runbook-r01.zip`

## Mission

Analysis/adjudication job (no patch expected; use the analysis contract).
Design the complete operational runbook for mass agent annotation of the
operator's live archive (~13.2k sessions, 3.8M messages, multi-provider),
building on the annotation substrate and the calibration design (job ann-02
runs in parallel — state interface assumptions where you depend on it).

Answer, with evidence from the attached snapshot (source, Beads, demos):

1. **Construct priority**: which annotation constructs pay off first, given
   the outreach goal (claim-vs-evidence credibility) and product goals?
   Candidates visible in the codebase: terminal-state/outcome labels
   (`vhjs` — terminal_state_method NULL for all 8,507 labels; `wofr` marathon
   blindness), failure-acknowledgment labels (claim-vs-evidence calibration),
   session-quality/derailment, task-completion-vs-claimed, pathology tags
   (existing pathology detectors), title/topic quality (`ih67`). Rank with
   reasons; name the report each construct unlocks.
2. **Batch design**: batch sizes, stratification (origin/model/era/length),
   gold ratios, judge counts per item, escalation rules; cost model per
   construct at realistic token prices for local (Ollama), subscription, and
   API lanes — with arithmetic shown.
3. **Provenance + reproducibility**: exactly which annotation_batches fields
   pin judge model/version/prompt-hash; how a batch is re-runnable; how label
   schema evolution is handled without invalidating old batches.
4. **Failure modes**: label leakage, construct drift, judge sycophancy,
   distribution shift across providers, and the specific mitigation for each.
5. **The first campaign, concretely**: a ready-to-execute plan (construct,
   N items, strata, judges, gold set source, expected cost, success criteria,
   verification queries) that the operator could launch within a day using
   local agent runtimes.

## Deliverable emphasis

REPORT.md (the runbook), DECISIONS.md (ranked construct order + rationale),
NEXT-ACTIONS.md (the first campaign as an executable checklist), and an
EVIDENCE.md citing the exact source/bead/demo files that ground each claim.
