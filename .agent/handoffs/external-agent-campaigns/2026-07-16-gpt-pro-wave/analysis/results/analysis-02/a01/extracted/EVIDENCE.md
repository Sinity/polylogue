# Evidence and reconstruction record

## Authority and method

The analysis followed this order when sources disagreed: current tracked source at snapshot commit; repository instructions and current Beads records; canonical campaign ledgers and corpus README; Git/PR history; older notes. Prose that had been explicitly corrected was not treated as current fact.

The all-refs Git bundle from the project-state archive was verified and cloned. The inspected tracked snapshot is `master` at `f654480cadb7cc4c194704e24dfd483199547b35`. The snapshot metadata says it was generated at `2026-07-17T043202Z` and was dirty. The reduced working-tree tar omitted the full tracked `.agent` campaign corpus, so the all-refs bundle was necessary to recover the canonical package tree and history.

For the package corpus, all 28 paths named by `.agent/handoffs/polylogue-sol-pro-2026-07-15/INTEGRATION-LEDGER.json` were reopened as ZIPs, hashed, and compared to the ledger. Member counts, uncompressed sizes, content categories, and largest members were recomputed from the actual archives. Bead states were joined from `.beads/issues.jsonl`. Direct package PRs and later repair PRs were matched against all-ref Git history and the archived PR records.

No package code or tests were executed. Verification claims in this report are attributed to PR/Bead receipts. No live browser, daemon, provider, archive database, deployment, or remote service was queried.

## Primary evidence inventory

| Evidence | Role |
| --- | --- |
| `polylogue-overview.json` / `polylogue-snapshot-audit.json` | Snapshot time, branch, commit, dirty flag, repository/Beads/PR census. |
| `polylogue-all-refs.bundle` | Complete tracked source and all-ref Git history. |
| `.agent/handoffs/polylogue-sol-pro-2026-07-15/README.md` | Acquisition history, capture incident summary, session correlation, corpus status. |
| `.agent/handoffs/polylogue-sol-pro-2026-07-15/INTEGRATION-LEDGER.json` | Canonical 28-package identity and adjudication. |
| `.agent/handoffs/polylogue-sol-pro-2026-07-15/SHA256SUMS` and `raw/*.zip` | Package byte custody and validation. |
| `.beads/issues.jsonl` | Current owners, corrections, acceptance criteria, closure state. |
| PRs #2913, #2918, #2919, #2922–#2930, #2953, #2957 and their merge commits | Implementation, review, verification, chronology, architectural correction. |
| `.agent/handoffs/external-agent-campaigns/README.md` | Current external campaign model: stable campaign/workload/job/attempt/revision identities, immutable `result.json`, rebuildable index. |
| `.agent/handoffs/external-agent-campaigns/schemas/{campaign,result}.schema.json` | Existing schema authority that recommendations must extend. |
| `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/{check-dispatch.py,triage-package.py}` | Current dispatch and intake mechanics. |
| `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/*/results/index.json` | Prepared next-wave indexes; all are empty in the snapshot. |

## Observed facts

1. The corpus README records 45 manual browser downloads deduplicated to 28 unique ZIPs, with 17 exact duplicate re-downloads.
2. All 28 canonical packages are present, hash-correct, ZIP-valid, and marked `processing_state: processed`; none is unknown.
3. Ledger dispositions are: 4 merged, 4 research-incorporated, 12 incorporated pending delivery, 2 already subsumed, 4 superseded/duplicate, and 2 rejected.
4. The 28 package revisions map to 26 unique conversation IDs. Package pairs 6/7 and 23/26 share conversation IDs.
5. Every ledger `iteration` field is null. All ledger `pr` fields and all direct `merge_commit` fields are null except one subsumed package’s historical merge reference; direct PR mapping lives in prose/current-master evidence rather than normalized fields.
6. The corpus contains 65,938,477 compressed bytes and 68,696,641 uncompressed bytes. Ten packages are at least 1 MiB and none directly merged.
7. The four direct-merge package ZIPs total 192,209 bytes. Their PRs are #2922, #2923, #2924, and #2925.
8. Current owner state at the snapshot shows two of the four direct merge owners closed: `polylogue-lkrc.4` and `polylogue-303r.2.1`; `polylogue-866e` is open and `polylogue-1xc.13` is in progress.
9. The CaptureJobs package is preserved on `feature/integration/capture-job-authority` at `ba340c71a` and `8ecc34ecc`, but its own Bead states that the fallback `paired:<provider>` is unsafe for exact account recovery and conflicts with generic BrowserAction. PR #2953 later delivered a reconciled current-source implementation and closed `polylogue-06zm.1`.
10. PR #2957 later repaired parser-drift replay under `polylogue-lkrc.4`, after the direct raw-authority package PR #2923.
11. `polylogue-3v1` reports a 27-conversation production census with 2,551 messages in current extension files versus 205 exposed by the index, 24/27 projection mismatches, and 22/27 cursor paths permanently excluded after transient failures. PR #2930 addresses replacement/reingest and browser snapshot membership; live replay/deployed parity remained required.
12. The corpus README records zero `*launch-handoff*` output attachment refs when the live archive was checked, despite all source sessions and 29 input attachments being present.

## Source-supported inferences

- Generation was not the principal loss point: 28 valid package revisions survived manually. The evidence loss occurred in canonical output-attachment custody and campaign event recording.
- Integration capacity and architecture stability constrained direct yield. Twelve packages were retained but gated, while the launch/capture architecture changed repeatedly during the campaign.
- The direct-merge missions shared current-source specificity and bounded claims. This association is stronger than the raw ZIP-size association, because large packages were dominated by heterogeneous fixtures and documents.
- Same-chat repair/revision behavior existed, but the campaign model failed to record it. The duplicate conversation IDs and null `iteration` values are direct evidence of that gap.
- The broad 28-session temporal set likely includes the later `6a587a8c` proof conversation outside the authoritative 27-conversation campaign census. This remains an inference until the raw roster is available.

## Contradictions and adjudication

| Earlier or superficial claim | Contrary evidence | Adjudication |
| --- | --- | --- |
| 28 files were 28 iterations of one mission. | Manifests, transcript references, patch digests, and Bead correlation identify a coordinated multi-mission batch. | Superseded. Treat as 28 package revisions across many missions. |
| 28 sessions, 28 deliverables, one-to-one. | 28 packages map to 26 unique conversation IDs; `polylogue-3v1` reports 27 campaign conversations; README lists 28 sessions in a broader time envelope. | Do not use a one-to-one funnel. Preserve all four entity counts. |
| Assistant output acquisition capability did not exist. | `launch_jobs.py`, work-package code, PR #2913, and later corrections show a mechanism existed. | Root cause was stale payload/tab-lifetime/reingest/exclusion defects, not total capability absence. |
| PR #2918/#2919 fixed the historical campaign outputs. | The audit still found zero output package refs, and `polylogue-3v1` requires live replay/deployed parity. | Future-path fixes are not retroactive recovery evidence. |
| Ledger `incorporated_pending_delivery` is current delivery truth. | Current Beads/PRs show CaptureJobs delivered later by #2953 and `polylogue-06zm.1` closed. | The ledger is immutable package adjudication; current delivery must be a rebuilt projection from Beads/Git/PR evidence. |
| Canonical download timestamp marks package completion or integration start. | Three direct PRs were created before the canonical timestamp; #2924 merged before it. | Timestamp is custody/dedup evidence only. |
| Package self-verification establishes product correctness. | Direct PRs narrowed claims, rejected competing seams, corrected design errors, and ran independent production-route tests. | Package receipts are triage input, never the publication authority. |
| Large packages contain proportionally more implementation value. | Large packages are dominated by snapshots, fixtures, replacement archives, or oversized documents; none directly merged. | Record content composition and declared necessity; do not infer quality from total bytes. |

## Direct merge evidence

### PR #2922 — lineage order independence

Package 2 maps to `polylogue-866e`. The PR merged `b55f3fd9697083d44466613091604a21c7324ae6`, using canonical sibling-variant ordering and a missing-cut guard. It explicitly refused the separate continuity package because that package covered only two of seven routes and introduced a competing scenario seam. Focused lineage/storage tests and quick gates are recorded; the Bead remains open for semantic cut witnesses, crash/rollback proof, exhaustive examples, and nested-ancestor completeness.

### PR #2923 — complete live bundle admission

Package 12 maps to the raw-authority path and merged `81142d1dce7d8e896ef340783ab943cdf59143f6`. The PR claims only the reproduced complete-bundle admission defect and real taxonomy/parser/storage/retry path. It reports 54 passes and 10 unrelated failures in the broader file, and explicitly does not claim the full immutable census/conservation program. PR #2957 later merged `a62d2f972ea43dbaa4cb34e897eed07e674e0e30` for parser-drift membership replay and closed `polylogue-lkrc.4`.

### PR #2924 — exact-source freshness

Package 23 maps to `polylogue-1xc.13` and merged `b6c78adfcd666358307daf64ac97e8d695a8b854`. It adds bounded, read-only exact-source freshness across cursor/raw/authority/parse/index/FTS/insight stages and records 189 focused tests plus affected and quick verification. It intentionally does not run remediation. The owning Bead remains in progress pending live receipts and broader work.

### PR #2925 — Sinex publication convergence

Package 25 maps to `polylogue-303r.2.1` and merged `36001d023b2cfe793cb19fdd7c42a87597356f48`. It adds durable publication obligations and mode-aware convergence but leaves real transport/deployment to `polylogue-303r.2.2`. Bead notes record a coordinator correction: source and index are separate SQLite tiers, so the enforceable invariant is ordered durability, not a single cross-database transaction. The owner is closed.

## Later repair and reimplementation evidence

### CaptureJobs

Package 24 was preserved rather than merged. Its local branch had receiver registry semantics and recorded 2 focused tests plus a 16-gate quick run, but failed the exact-account and current-transport constraints. PR #2953 merged `e6698a74ea55109d20cd21a98e51e422e6f784ed` with receiver-authoritative IDs, opaque HMAC-reduced account scope, CAS revisions/checkpoints, replaceable leases, idempotent receipts, new-profile recovery, and typed legacy orphans. Its recorded verification includes receiver, daemon security, extension, lint, manifest, and quick gates, plus multiple adversarial passes. This is local reimplementation influenced by packages 4 and 24, not a direct package patch merge.

### Parser-drift replay

PR #2957 is a post-package repair on the raw-authority line. It validates replay against persisted accepted index/raw witnesses while preserving divergent/quarantined rejection. It records focused tests and a 16/16 quick gate. Its existence demonstrates why initial merge and Bead closure must be separate funnel stages.

## Capture architecture evidence

- #2913 introduced the authenticated Sol Pro launch queue and a live submission smoke.
- #2918 removed duplicate handoff-byte transport and made canonical capture authoritative; its broad fresh-worktree run stopped at 22% after unrelated failures.
- #2919 removed launch-only capture metadata and reconciled by provider-native conversation ID.
- #2926 recorded a live closed-tab canonical output ZIP proof and hardened pacing, transport ownership, and output validation.
- #2928 removed campaign-specific launch/work-package semantics from product code, retaining generic BrowserAction plus automatic freshness convergence.
- #2930 repaired changed-file reingest and compatible browser snapshot ordering after the 27-conversation incident census.

These commits show that the campaign ran across a moving control plane. Prompt, cadence, and model-output comparisons are therefore confounded by transport and ingestion changes.

## Timing evidence

Canonical package timestamps and direct PR timestamps yield the following impossible-as-latency relationships:

| Package / PR | Canonical package timestamp | PR created | PR merged | Interpretation |
| --- | --- | --- | --- | --- |
| 2 / #2922 | 02:44:07 | 03:43:21 | 04:17:18 | Creation after custody; plausible but not proof of acquisition latency. |
| 12 / #2923 | 03:55:02 | 03:43:22 | 04:13:04 | PR created 11m40s before canonical custody timestamp. |
| 23 / #2924 | 06:14:18 | 04:42:00 | 05:52:25 | PR merged 21m53s before canonical custody timestamp. |
| 25 / #2925 | 06:14:30 | 05:22:21 | 07:03:34 | PR created 52m09s before canonical custody timestamp. |

The direct PR open-to-merge durations are valid GitHub chronology. The package timestamps are not valid stage chronology.

## Missing evidence

The supplied project-state archive lacks:

- raw canonical exports or a complete roster for the 27/28 campaign sessions;
- provider action/run receipts and Retry-After/error telemetry;
- exact first-turn, terminal-turn, and artifact-offered events;
- historical canonical output attachment refs/blobs for the 28 packages;
- live replay results after #2930 and current `ops.db` capture-gap events;
- per-attempt immutable `result.json` records for this prior campaign;
- local repair start/end times, line retention, and human review effort;
- detailed machine-readable verification runs for each package integration;
- exact uncommitted snapshot delta despite the dirty flag.

Because these are absent, the report marks terminal-turn conversion, acquisition latency, repair cost, and live capture recall as unresolved rather than estimating them.
