# GPT Pro wave — 2026-07-16

Prepared input portfolio for manual-first, automation-ready dispatch after the
test-harness foundation is merged and a fresh Lynchpin Chisel project-state
archive is generated.

Workloads:

- `testdiet/`: 16 staged implementation-draft lanes over the Test Suite Diet;
- `beads/`: high-fit implementation clusters selected from current Beads;
- `analysis/`: custom cross-cutting audits and adjudication;
- `deep-research/`: source-cited external research tied to concrete decisions.

Every prompt begins with a literal `Title: "..."`, declares a readable job and
exact unique output filename, and requires a substantive answer plus a verified
package. The same project-state attachment may be reused across jobs only while
its snapshot identity remains authoritative.

Do not launch the Test Diet implementation prompts until the foundation mission
below has landed, the project-state archive has been regenerated, and the job's
dependency state in `campaign.json` is satisfied.

Launch gating and result intake are mechanical, not prose:

- `./check-dispatch.py [--workload W] [--job J]` validates, per job, the
  foundation receipt, `depends_on` satisfaction against the results ledger,
  and testdiet context-manifest freshness. Exit 0 = dispatchable. When the
  foundation merges, write `foundation-receipt.json`
  (`{"merged_commit": "<sha>", "verified_at": "<iso8601>"}`) at this
  directory's root.
- `./triage-package.py <result.zip> --workload W --job J --attempt aNN
  --package-revision rNN --prompt-sha256 <hash> --snapshot <commit> --write`
  validates an incoming package (identity, required members, placeholder scan,
  `git apply --check` in a throwaway worktree), preserves it in canonical raw
  custody, and writes one immutable attempt receipt. It then rebuilds the
  index from receipts. It never runs a package's own test commands — that
  happens later in a reviewed lane.
- `python ../reconcile_results.py . --check` is the required post-intake
  integrity gate: it rejects a missing, ambiguous, stale, or second-outcome
  projection rather than silently trusting the index.
- `testdiet/owning-beads.json` maps each testdiet job to the Beads records
  owning its durable product contracts (persisted by PR #2947); paste the
  job's line into the dispatch chat with the prompt. `check-dispatch.py`
  flags stale/closed/in-progress owners at launch time.

The active local test-harness agent should receive
[`test-harness-agent-foundation-mission.md`](test-harness-agent-foundation-mission.md).
