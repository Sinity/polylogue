# Fork prompt 04 — Land the first semantic transcript renderer slice

Use the uploaded Polylogue repository and prior analysis. Implement the highest-value vertical slice of `polylogue-ap7`: a provider-agnostic semantic transcript renderer shared by CLI and web.

The first slice must cover:

1. shell/test command cards;
2. structural result cards with success/failure, exit status, duration, and folded output;
3. file edit cards rendered as diffs where evidence supports it;
4. a generic unknown-tool fallback that preserves all data;
5. evidence, derived, candidate/reviewed, missing, degraded, and not-supported badges where the current model exposes those states.

The core rule is that CLI and web consume a shared surface-neutral descriptor. Neither surface may independently decide whether a tool failed. Normalize provider spellings into semantic families while preserving raw tool identity and refs.

Inspect `polylogue/rendering`, CLI read views, web/site reader paths, surface payloads, and current snapshots. Design a renderer registry with explicit fallback and bounded output behavior. Add snapshots and behavioral tests, including huge output folding, invalid/missing fields, unknown tools, and structural failure that contradicts prose.

Make the change visually forceful enough to support The Receipts demo. Produce before/after screenshots or deterministic HTML/terminal fixtures using only public-safe data. Avoid attempting every tool family; document the extension contract and finish the shell/test/edit slice completely.

Run targeted rendering, CLI, and web/site tests. Do not edit README or marketing copy except a narrow developer note if required. Do not modify active token-accounting work.

Store outputs under `/mnt/data/polylogue-semantic-renderer/`, including the patch, design note, screenshots, snapshots, and verification receipt. Return links and call out any files likely to conflict with demo branches.
