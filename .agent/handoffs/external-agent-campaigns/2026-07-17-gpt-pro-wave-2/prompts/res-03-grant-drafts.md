Title: "Grant application drafts: Emergent Ventures and Manifund, built around the local-first agent-context substrate and its first demos"

Result ZIP: `res-03-grant-drafts-r01.zip` (use the analysis contract).

## Mission

Draft complete, submission-ready applications for two funders, grounded in
the attached Polylogue snapshot (inspect the repo, demos under
`.agent/demos/`, docs, and Beads to ground every claim — nothing may be
asserted about the project that the snapshot doesn't evidence).

Applicant context you may use (public-safe; do not embellish): independent
software engineer in Poland, ~6 years commercial experience (C++/ERP, then
telecom/distributed systems), since mid-2025 full-time independent R&D on
AI-agent tooling driven almost entirely through coding agents; Polylogue is
the public MIT-licensed flagship (local-first archive/forensics for AI
sessions across ChatGPT/Claude/Codex/Gemini/Hermes; ~13k sessions / 3.8M
messages live corpus; content-addressed SQLite tiers, provenance, cost
accounting, structured tool-outcome capture, MCP server); private siblings
(Rust event-capture substrate, personal-data analysis hub) demonstrate the
broader local-first agent-context thesis. Do NOT reference or imply any
pseudonymous online identity; do not include names of other individuals.

1. **Emergent Ventures**: their application is short (project description +
   what the grant enables + budget-ish framing). Research the CURRENT
   application questions (mercatus.gmu.edu; cite what you find, note if the
   form changed). Draft: (a) the 1-2 paragraph project description pitched
   as the local-first agent-context substrate — agents that recover their
   own state and audit their own work from a user-owned archive; (b) the
   "what would you do with the grant" answer (harden the archive substrate
   into a distributable product; ship the claim-vs-evidence audit line and
   the agent-recovery demos; fund N months of full-time work); (c) the
   "most unusual thing about you" style answer if the form asks it —
   grounded in the built-through-agents methodology with honest numbers
   read from the repo (commit/PR shape, scale), framed as method not
   boast. Provide 2 tonal variants: matter-of-fact and slightly bolder.
2. **Manifund**: research the current project-posting format (manifund.org
   — regrantor norms, typical ask sizes for individual OSS/AI-tooling
   projects; cite examples). Draft the full project post: title, summary,
   what the money does (milestone-shaped: demo shelf completion, WebUI v2,
   packaged release, annotation/eval program), track record section
   (repo-evidenced), risks-and-honesty section (write this one genuinely —
   single-maintainer risk, adoption risk, the "demos exist but external
   users don't yet" truth — Manifund's audience rewards calibrated
   self-assessment), and funding tiers.
3. **Shared evidence appendix**: a fact sheet both applications draw from —
   every number (LOC, sessions, providers, tests, demo names) extracted
   from the snapshot with the file/command it came from, so the operator
   can verify each before submitting.

## Deliverable emphasis

REPORT.md (both drafts, clearly sectioned, ready to paste), EVIDENCE.md
(the fact sheet with per-claim provenance), DECISIONS.md (framing choices
made + the 2-3 the operator should review), NEXT-ACTIONS.md (submission
checklist per funder incl. anything to verify/update day-of).


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
