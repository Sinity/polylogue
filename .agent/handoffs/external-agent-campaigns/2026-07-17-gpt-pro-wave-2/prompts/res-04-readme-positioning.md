Title: "README and positioning rewrite: evidence-led, demo-first, zero slop"

Result ZIP: `res-04-readme-positioning-r01.zip` (analysis contract; the
deliverable is a complete README draft + positioning kit, not a patch to
anything else).

## Mission

Polylogue's README undersells the tool (operator's own standing note) and
external readers are imminent: the outreach thesis is "look how rigorously
this is built and audited", to an audience (agentic-coding practitioners,
open-model/eval people) that mocks hype and rewards concrete artifacts and
numbers. Rewrite it evidence-first.

Ground everything in the attached snapshot: read the current README, `docs/
architecture.md`, `docs/search.md`, `docs/cost-model.md`,
`docs/provider-origin-identity.md`, the demo shelf (`.agent/demos/` —
especially `claim-vs-evidence/` with its README/PUBLIC_REPRODUCTION/
COLD_READER_GATE and `CURATED_CATALOG.md`), the CLI reference, and the MCP
reference. Note: an MCP six-tool replacement and new demos are landing in
parallel local lanes — write the README parameterized so those sections
have clearly-marked slots ("[six-tool table]", "[demo numbers]") the
integrator fills, with current-state text as fallback.

Deliver:

1. **The README draft**, structured roughly: one-paragraph what-it-is (a
   local-first archive + forensics layer for AI/agent sessions — concrete,
   no vision-speak); a 90-second quickstart that is REAL (verify every
   command against the snapshot CLI: install route, `polylogue import
   --demo --wait`, a find/read/search sequence — every example must be
   copy-paste runnable; use the demo corpus so no personal data is
   implied); "what it captures" table (providers/origins × what fidelity —
   from provider-origin docs); the demos section (each demo: claim, one
   number, reproduction command, honesty caveat — model it on the
   cold-reader-gate style); architecture-in-one-diagram (adapt the rings
   diagram from CLAUDE.md/docs); honest limitations section (single
   writer, rebuild semantics, what's not covered); MCP/agent-integration
   section (slot for six-tool era).
2. **Style discipline**: no em-dash-as-connector cadence, no rule-of-three
   AI tells, no unverifiable superlatives, no "blazingly fast". Numbers
   only with provenance. Every feature claim must name the command that
   demonstrates it.
3. **Positioning kit**: 3 short descriptions (GitHub about-line ≤120
   chars; 2-sentence version; 1-paragraph version) + a "what it is NOT"
   list (not an agent, not a cloud service, not a memory-retrieval
   benchmark player) that preempts miscategorization.
4. **Gap report**: every README claim you WANTED to make but the snapshot
   couldn't evidence — as a table (claim → missing evidence → which
   demo/feature would supply it). This routes future demo work.

## Deliverable emphasis

REPORT.md (README draft + positioning kit), EVIDENCE.md (per-claim
provenance: file/command for every number and feature), DECISIONS.md
(structural/framing choices), NEXT-ACTIONS.md (integration checklist:
slots to fill, commands to re-verify on the live repo before merging).


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
