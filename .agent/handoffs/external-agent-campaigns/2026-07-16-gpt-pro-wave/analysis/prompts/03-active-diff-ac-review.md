Title: "[analysis 03] Adversarial AC review of active large diffs"

Job ID: `analysis-03`
Result ZIP: `analysis-03-active-diff-ac-review-r01.zip`

Review every currently active substantial Polylogue branch/worktree/PR against
its complete owning Beads and current master. Build a per-AC matrix, trace every
claim to changed production routes and tests, identify uncommitted/stranded
value, overlaps, stale-base conflicts, invented parallel mechanisms, vacuous
tests, missing generated surfaces, and unsafe merge ordering. This mission is
an adversarial audit by definition; do not merely summarize diffs. Return
repair prompts/actions per branch and an integration order, but do not claim
checks ran or recommend wholesale application without source evidence.

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
