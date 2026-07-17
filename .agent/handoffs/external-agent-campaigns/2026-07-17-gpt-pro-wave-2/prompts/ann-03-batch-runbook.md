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


---

## Context and authority

You are a long-running ChatGPT Pro analysis worker. A recent, complete
Polylogue project-state archive will be attached. Retrieve and inspect it
broadly; attachment size alone is not a reason to ignore evidence. This prompt
defines the question. The snapshot's current source, repository instructions,
complete relevant Beads records, and cited history are the evidence authority,
in that order when older plans drift.

## Working contract

- Investigate the actual source and tracker state before recommending changes.
- Separate observed facts, source-supported inference, unresolved uncertainty,
  and recommendation. Quote paths/symbols/Bead ids precisely but do not fill the
  report with copied source.
- Adjudicate contradictions and duplicates; do not create a parallel product
  model or generic architecture merely to make the report look complete.
- Translate findings into decision-ready actions: exact owning areas, ordering,
  acceptance criteria, falsification evidence, and what a local implementer
  should verify.
- Do not claim live browser, daemon, archive, deployment, or test evidence you
  cannot access.

## Deliverable

Create the exact `Result ZIP` named near the top under `/mnt/data/`. It must
contain `REPORT.md`, `EVIDENCE.md`, `DECISIONS.md`, and `NEXT-ACTIONS.md`.
Include compact machine-readable tables as JSON/CSV only when they add genuine
integration value. Do not copy the input archive into the result. Attach the
finished ZIP to the conversation through a working user-clickable link; files
left only in an internal temporary directory are not delivered.

Reopen and validate the ZIP, then report its SHA-256, size, and members. The
final chat answer must itself explain the important conclusions and decisions,
limitations, missing evidence, and the likely value of another iteration before
linking the package.

Do not perform an adversarial review unless explicitly requested. On an
ordinary **iterate/continue** request, preserve sound findings, resolve the
highest-value remaining uncertainty, and regenerate a complete package
revision. On an explicit **adversarial review** request, try to falsify the
prior report: seek contrary source/history evidence, unsupported certainty,
missed stakeholders/call sites, duplicate or incompatible designs, weak
acceptance criteria, and recommendations that do not survive current code.
Repair legitimate findings, regenerate the cohesive package, and report the
delta, residual disputes, and expected value of another pass.
