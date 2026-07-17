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
