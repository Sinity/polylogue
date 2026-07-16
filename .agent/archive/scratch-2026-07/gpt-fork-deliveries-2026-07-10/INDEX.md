# GPT-5.6 Pro Fork Deliveries — Extraction Escrow (2026-07-10)

Source: 22 ChatGPT conversations captured via browser extension into
`/realm/inbox/polylogue-browser-spool-2026-07-10/chatgpt/*.json` (the mission brief said 18;
the spool actually contained 22 — 4 more arrived in a later capture batch, all processed).
These GPT-5.6 Pro work forks delivered analysis/implementation packages whose sandbox files
(`sandbox:/mnt/data/...` download links) are now DEAD. This escrow preserves everything
recoverable from the message text, verbatim.

Layout per fork: `delivery.md` (final delivery message(s), verbatim), `inline-artifacts.md`
(substantial recoverable content from earlier turns, or "none recovered" + why), `STATUS.md`
(ask / delivered / recoverable-vs-lost / regeneration value).

Special folder: `00-shared-branch-root-polylogue-sinex-overview/` — 11 of the 22 captures are
ChatGPT branches sharing a byte-identical prefix (Polylogue overview 62.5K + Sinex overview
94.4K + external-legibility summary 17.9K); it is captured once there instead of 11×.

Sinex-scoped forks are additionally staged (delivery.md + inline-artifacts.md only, no STATUS)
at `/realm/inbox/gpt-pro-sol/sinex-fork-deliveries/<slug>/`.

## Recovery-status legend
- **complete-inline** — the full deliverable content is in the transcript; nothing meaningful lost.
- **summary-only** — the final message is a summary pointing at dead sandbox package files; the
  package content itself is LOST.
- **needs-regeneration** — the fork never completed (or its lost package is the actual point);
  re-running in the ChatGPT UI (or a fresh session) is required to get the deliverable.

## Master table

| Slug | Conversation URL | Target | Asked for | Delivered | Recovery | Package files LOST |
|---|---|---|---|---|---|---|
| 00-shared-branch-root-polylogue-sinex-overview | (shared prefix of the 11 `Branch ·` forks below) | polylogue+sinex | explain Polylogue at length; same for Sinex + interop; improve external legibility | Polylogue presentation (62.5K), Sinex presentation + maximal interop architecture (94.4K), external-legibility pack summary | complete-inline (2 reports) / summary-only (legibility pack) | 219-file external-legibility kit |
| 01-github-publication-feasibility-sinnix-lynchpin-sinex | https://chatgpt.com/c/6a4433f6-8604-83eb-a40c-7cc2f641e158 | sinnix+lynchpin+sinex | publication feasibility review ×3 repos + git-history cleanup | consolidated 3-repo publish-feasibility verdict ("publish all three, but not the same way"); earlier: v5 history-cleanup delivery + standalone Sinex review | complete-inline (analyses) / summary-only (cleanup bundles) | output tarball, sinnix+lynchpin cleaned bundles, make_clean_git_exports_v5.py, git-cleanup-v5-report.md, audit.json, SHA256SUMS.txt |
| 02-polylogue-overview-and-agent-interop-landscape | https://chatgpt.com/c/6a4ebe8b-8f50-83ed-9867-7b77a8e567b0 | polylogue | project overview; rename brainstorm; A2A reaction; broad interop-standards research | 4 prose deliveries incl. rename verdict (collision with polylogue.page) and interop-landscape survey (A2A/MCP/AGNTCY/ANP/AGENTS.md/llms.txt/VC-DID/OAuth/OCI/LSP...) | complete-inline | none (prose-only thread) |
| 03-polylogue-project-presentation-technical-assessment | https://chatgpt.com/c/6a506bcf-852c-83eb-82e6-e23ac8a418e1 | polylogue | explain/present the project at length | 62.5K-char presentation + technical assessment (same text reused as shared branch root) | complete-inline | none |
| 04-polylogue-bead-execution-process-current-state-review | https://chatgpt.com/g/g-p-6a40343a1f9881918dee375ded0971a4-a/c/6a506c24-ec1c-83eb-b97d-0b78b8b69833 | polylogue | bead execution ordering, plan-quality assessment, upgraded beads setup, static prework packets, current-state re-review | final Jul-6→Jul-10 delta review (+113 commits, +76 closed beads, P1 32→13, verdict upgraded) + 4 earlier planning deliveries | complete-inline (judgments) / summary-only (packages) | upgraded-beads-setup ZIP, static-prework ZIP, v2 ZIP, current_state_review ZIP (md+5 CSVs+JSON), evocative-narrative md |
| 05-polylogue-strategy-falsification-decision-memo | https://chatgpt.com/c/6a507c1a-940c-83eb-9600-f8449aeda538 | polylogue | execute 06-strategy-falsification.md → decision memo (narrow/pivot/stop) | verdict NARROW w/ 30-day stop gate; amendment after new Jul-10 notes: still NARROW | complete-inline | none (amendment heredoc text captured) |
| 06-sinex-derivation-kernel-red-team | https://chatgpt.com/c/6a507e40-46bc-83eb-9c9f-0e0815dc142a | sinex | execute sinex-01-derivation-kernel-red-team.md | kernel-derivation REJECTED: 3 incompatible execution models; only commonality is a shared defect (pre-settlement commit) | complete-inline | none |
| 07-sinex-replay-completion-contract | https://chatgpt.com/c/6a507e8f-49d8-83eb-9fdc-29a4761588a0 | sinex | execute sinex-02-replay-completion-contract.md | contract FAIL: best-effort replay executor, no completion barrier/settlement ledger | complete-inline | none |
| 08-sinex-output-axes-reconciliation | https://chatgpt.com/c/6a507e9d-c600-83ed-8f9a-3eece5fea417 | sinex | execute sinex-03-output-axes-reconciliation.md | spec identifies real category error but wrong fix; needs axis separation, not cleaner enum | complete-inline | none |
| 09-sinex-semantic-fingerprint-convergence | https://chatgpt.com/c/6a507eaf-a91c-83eb-afed-39b3c2033f77 | sinex | execute sinex-04-semantic-fingerprint-convergence.md | 71K-char semantic-fingerprint + comparison protocol for changed-only convergence (16-automaton audit) | complete-inline | none |
| 10-sinex-coverage-obligation-compiler | https://chatgpt.com/c/6a507ebd-5df8-83ed-87f6-5cf829136f00 | sinex | execute sinex-05-coverage-obligation-compiler.md | 59K-char coverage-obligation compiler review (10 orphan tests found, registry-driven obligation design) | complete-inline | none |
| 11-sinex-beads-campaign-surgery-incomplete | https://chatgpt.com/c/6a507ecd-96b8-83eb-ad31-56a9df0fa81d | sinex | execute sinex-06-beads-campaign-surgery.md (dry-run ledger + script) | NOTHING FINAL — capture cuts off mid-task; only 2 short per-bead rulings (sinex-qky drift, r6d.9 stays open) | needs-regeneration | the entire requested deliverable was never produced |
| 12-sinity-personal-profile-PRIVATE | https://chatgpt.com/c/6a50b7cc-0b24-83eb-bd15-2edadd846f2b | PERSONAL | **PRIVATE** — personal-profile dossier work (scratch-only; not staged, not quoted) | 2 personal reports escrowed in scratch only | complete-inline (scratch-only) | n/a |
| 13-polylogue-semantic-transcript-renderer | https://chatgpt.com/c/6a5112e7-46a0-83eb-8486-d0865595094d | polylogue | mission 01-semantic-renderer-epic.md | semantic transcript renderer "implementation delivered" + 17 proof/follow-on doc files recovered inline | complete-inline (docs) / summary-only (code) | renderer code patch + test suite (docs recovered) |
| 14-polylogue-web-evidence-cockpit-v2 | https://chatgpt.com/c/6a5112f3-7798-83eb-b1ee-96ef29477c12 | polylogue | mission 04-web-evidence-cockpit.md | Web Evidence Cockpit v2 kit (delivered as sandbox ZIP) | summary-only | entire cockpit kit (UI code/assets) |
| 15-polylogue-demo-packet-v2-flagship-proof | https://chatgpt.com/c/6a5112f5-d4c8-83eb-8efd-84aa9ace5836 | polylogue | mission 03-demo-packet-v2-and-flagships.md | Demo Packet v2 + flagship proof package; full test_demo_packet.py recovered inline | summary-only (patch) / complete-inline (1 test file) | demo-packet patch, apply-patches.sh, verify-package.sh |
| 16-polylogue-query-dsl-expansion | https://chatgpt.com/c/6a5112f9-f200-83ed-a96e-a083dd7d4a46 | polylogue | mission 05-query-dsl-expansion.md (polylogue-fnm) | Query DSL expansion package w/ executable test vectors (sandbox ZIP) | summary-only | entire DSL implementation + test vectors |
| 17-polylogue-incident-14-32-implementation-proof | https://chatgpt.com/c/6a5112fd-1ee0-83eb-b12a-27317f452bc5 | polylogue | mission 02-incident-1432-corpus.md | packet assembled BUT model explicitly could NOT certify the patch as passing | needs-regeneration | patch + proof packet (and it was uncertified anyway) |
| 18-sinex-beads-graph-surgery | https://chatgpt.com/c/6a5113e8-3f48-83ed-931a-da42be13baea | sinex | mission 10-beads-program-surgery.md | Sinex Beads graph surgery completed (24-file adjudication zip, sha256 recorded) + 6 template files recovered inline | summary-only | sinex-beads-graph-adjudication.zip (24 files, dry-run ledger + script) |
| 19-polylogue-category-launch-research-package | https://chatgpt.com/c/6a511407-631c-83eb-b5e4-9fc3b6852eee | polylogue | mission 09-research-category-and-launch.md | category/launch research package ("lead with falsifiable structural receipts") | complete-inline | supporting research pack (main guidance inline) |
| 20-polylogue-context-memory-loop-implementation | https://chatgpt.com/c/6a51140b-8f70-83eb-bbff-c2e9c1ee77a8 | polylogue | mission 08-context-memory-loop.md | context/memory loop implementation + experiment package | summary-only | implementation code + experiment data |
| 21-sinex-test-vacuity-audit | https://chatgpt.com/c/6a511416-6d5c-83eb-9a00-159fa983aabf | sinex | mission 07-test-vacuity-audit.md | repo-wide test-vacuity audit (executed pack, ZIP) | summary-only | audit pack ZIP incl. the flagged-test enumeration |
| 22-sinex-proof-obligation-compiler | https://chatgpt.com/c/6a511425-dc34-83eb-a46b-83d6790f4c90 | sinex | mission 06-proof-obligation-compiler.md | working proof-obligation compiler, executed | summary-only | compiler implementation + execution output |

## Counts

- Forks processed: **22** real conversation captures, represented by folders 01-22, plus 1
  shared-root synthetic folder (00). Of the 22: 1 PRIVATE (12), 9 sinex-scoped (06-11, 18,
  21, 22), and 12 polylogue/portfolio (01-05, 13-17, 19-20).
- **complete-inline: 10** — 02, 03, 05, 06, 07, 08, 09, 10, 19, plus 12 (PRIVATE, scratch-only).
- **dual (analysis inline / packages lost): 2** — 01, 04.
- **summary-only: 8** — 13\*, 14, 15\*, 16, 18, 20, 21, 22. Forks 13 and 15 each have one
  substantial artifact recovered inline (13: 17 proof/follow-on docs; 15: full
  `test_demo_packet.py`) but lost their main code package.
- **needs-regeneration: 2** — 11 (never completed) and 17 (explicitly uncertified).

## Notes / surprises

- **The spool had 22 files, not 18.** Timestamps show two capture batches (08:20 and 20:05-20:06);
  the 4 extra were same-day later captures. All were processed.
- **11 captures are branches of ONE conversation** ("Branch · Project Explanation and
  Relevance"): byte-identical 175K-char shared prefix (sha256-verified), then per-branch
  divergence via a different attached mission file (01-...md through 10-...md). Captured once in
  `00-shared-branch-root.../`.
- The 9 mission files (01-semantic-renderer-epic, 02-incident-1432-corpus,
  03-demo-packet-v2-and-flagships, 04-web-evidence-cockpit, 05-query-dsl-expansion,
  06-proof-obligation-compiler, 07-test-vacuity-audit, 08-context-memory-loop,
  09-research-category-and-launch, 10-beads-program-surgery) were user-authored attachments —
  if their source files still exist locally, re-running any lost package is straightforward.
- **Forks whose full report IS inline:** 09 (71K semantic fingerprint) and 10 (59K coverage
  obligation) are complete high-value Sinex reports needing no regeneration; same for the
  06/07/08 shorter verdicts.
- **Fork 11 vs 18 confusion hazard:** two different Sinex Beads-surgery missions exist.
  `sinex-06-beads-campaign-surgery` (fork 11) never delivered; `10-beads-program-surgery`
  (fork 18) completed but its 24-file zip is lost.
- **Fork 17 is the only self-declared failure:** the model explicitly refused to certify its
  Incident-14:32 patch as passing.
- Fork 01's turn 19 is a capture artifact: the assistant's delivery duplicated with role=user
  (streaming DOM-capture glitch); the canonical role=assistant copy was used.
- Fork 12 (PRIVATE) is titled "Branch · Project Attachment Comparison" and actually STARTS as a
  legitimate multi-project technical comparison before drifting into personal profiling; only
  the delivery + status were escrowed, scratch-only, per privacy rule.
