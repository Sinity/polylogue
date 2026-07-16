# Fork prompt 15 — Red-team every public claim and demo

Use both uploaded repositories, the prior analysis, and any legibility artifacts already produced in this fork. Act as a hostile but technically fair external reviewer. Do not implement the preferred story. Try to falsify it.

Audit:

- READMEs and site taglines;
- proof-artifact pages;
- claims ledger;
- all demo packets and recordings;
- install commands;
- private field findings;
- Sinex–Polylogue present-state versus target wording;
- security/privacy statements;
- Beads-backed roadmap claims;
- logical versus physical cost/accounting language;
- context/memory uplift language;
- source coverage and no-loss language.

Search for:

1. circular oracles;
2. deterministic fixture results presented as prevalence;
3. private archive observations presented as benchmarks;
4. silent denominator changes;
5. present-tense aspirations;
6. stale generated assets;
7. claims that cannot resolve to evidence;
8. missing negative or missing-evidence controls;
9. inaccessible install paths;
10. private paths, hostnames, secrets, emails, or transcript text;
11. GitHub Issues used as current roadmap authority;
12. “privacy” wording that actually means redacted view while originals remain;
13. universal no-loss claims contradicted by recovery history;
14. full-backend claims contradicted by the metadata-only current bridge;
15. UI outcomes that differ from CLI semantics.

Run static scans and available validators. Build a claim-by-claim table with verdicts: supported, overbroad, stale, ambiguous, or false. For every problem, propose exact replacement wording or a falsifying test. Rank launch blockers separately from polish.

Also perform a cold-reader exercise: explain both projects and the joint architecture using only public files, then list where you had to infer missing context.

Produce `/mnt/data/adversarial-legibility-audit/` with a detailed report, machine-readable findings, secret/PII scan receipt, suggested patches, and a go/no-go verdict. Do not soften the conclusion to preserve momentum.
