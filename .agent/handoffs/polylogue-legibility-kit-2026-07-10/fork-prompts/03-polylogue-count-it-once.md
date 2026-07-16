# Fork prompt 03 — Implement Polylogue “Count It Once”

Use the uploaded Polylogue repository and prior analysis. Design and implement a new construct-valid demo under the `polylogue-212` portfolio: **Count It Once**.

Primary claim: Polylogue preserves physical provider artifacts while distinguishing copied transcript replay from logical unique work.

Build a deterministic scenario with:

- a parent session containing a known prefix;
- a fork or continuation that physically repeats that prefix and adds a unique tail;
- a fresh subagent containing superficially similar text but no copied-prefix relationship;
- a compaction summary that is real new material and must not be deduplicated;
- independently declared expected physical and logical counts;
- usage/token lanes sufficient to show physical total, logical high-water total, and replay delta.

The demo should visibly transform a naïve physical total into a lineage-aware logical view while keeping raw evidence accessible. It must explain that logical accounting is not provider billing and that physical and logical views answer different questions.

Required controls:

1. copied prefix is charged once in logical unique work;
2. fresh subagent remains separate;
3. compaction summary remains real new content;
4. near-match text without an identity/topology edge is not fuzzy-deduplicated;
5. every headline total resolves to the exact session/message/edge ranges that contribute to it.

Use existing topology, session-profile, logical-root, usage, query, and demo-packet machinery. Do not hide a core-lineage defect in demo-specific arithmetic. If the current product lacks a required read, implement a reusable view or create a precise blocker with a failing test and deliver the strongest honest partial demo.

Produce:

- a proposed Bead specification for the new child under `polylogue-212`;
- implementation patch and tests;
- validated demo packet and report;
- a compact visual storyboard or generated terminal tape;
- exact reproduction commands;
- a claim/scope statement suitable for the public claims ledger.

Store outputs under `/mnt/data/polylogue-count-it-once/`. Use Beads, not GitHub Issues. Return links to the patch, packet, and handoff.
