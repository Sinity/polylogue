# External-agent campaign effectiveness postmortem

Job: `analysis-02`  
Result revision: `r01`  
Evidence snapshot: Polylogue `master` at `f654480cadb7cc4c194704e24dfd483199547b35`, snapshot generated `2026-07-17T043202Z`, marked dirty  
Primary campaign corpus: `.agent/handoffs/polylogue-sol-pro-2026-07-15/`

## Executive assessment

The campaign generated substantial usable engineering input, but generation throughput outran trustworthy capture and integration throughput. The strongest honest package-level result is **4 direct merges from 28 canonical package revisions (14.3%)**. A broader **20/28 (71.4%)** supplied new work retained as merged code, research, or an explicit downstream delivery input; two more were already subsumed by current work. The remaining six were superseded alternatives or rejected. All 28 packages were hash-valid and adjudicated, so low direct-merge yield was not an intake-completeness problem.

The direct merges were compact, current-source-anchored missions with a coherent production contract, an owning Bead, bounded residual scope, and independent local verification. Broad architecture packages, dependency-premature kernels, competing seams, and large fixture-heavy packages more often became research, pending input, superseded work, or rejection. This is an association, not evidence that small size or narrow scope alone causes merge success.

The most serious campaign failure was evidence custody. The contemporaneous audit states that Polylogue held all source sessions and 29 input attachments but no `*launch-handoff*` output attachment references, leaving 45 manual browser downloads as the durable acquisition path. The incident was first misdiagnosed as missing output capability, then corrected to stale completion payload, tab-lifetime, snapshot replacement, and cursor exclusion defects. PRs #2918, #2919, #2928, and #2930 changed the architecture and recovery path, but the supplied snapshot contains no live replay receipt proving that the historical outputs were recovered.

## The honest funnel

The campaign does not have one denominator. Conversations, package revisions, downloads, integration tracks, and Beads are different entities. Treating them as a single 28-item funnel creates false precision.

| Stage | Observed result | Honest interpretation |
| --- | ---: | --- |
| Submitted conversations | 27 campaign conversations in the authoritative `polylogue-3v1` census | The broad session-correlation note lists 28 sessions in the temporal envelope. Raw action receipts are absent, so submission success and retries cannot be independently recounted. |
| Package-bearing conversations | 26 unique provider conversation IDs | Two conversations each produced two canonical package revisions. |
| Terminal turn | Not independently countable | The archive lacks the raw canonical session exports and per-turn terminal receipts. The existence of 28 downloadable packages proves output availability, not a one-to-one terminal-turn count. |
| Raw asset downloads | 45 | Manual browser acquisitions; timing is bursty and not a reliable completion timestamp. |
| Unique canonical assets | 28 | 17 exact duplicate re-downloads were removed: 37.8% duplicate-download rate and 62.2% unique acquisition efficiency. |
| Hash/ZIP validation | 28/28 | Every canonical ZIP matched its ledger hash and reopened successfully. |
| Triage/adjudication | 28/28 | Four merged, four research-incorporated, twelve pending delivery, two already subsumed, four superseded/duplicate, two rejected. |
| New work retained | 20/28 (71.4%) | Merged + research-incorporated + explicit pending-delivery inputs. |
| Routed or proven redundant | 22/28 (78.6%) | Adds the two already-subsumed packages. This is not equivalent to code delivery. |
| Direct package PR/merge | 4/28 (14.3%) | PRs #2922–#2925. |
| Material post-package repair | At least 2 identifiable tracks | CaptureJobs was locally reimplemented as PR #2953; the raw-authority path received a later parser-drift repair in PR #2957. In-PR coordinator repairs were not normalized as events. |
| Independent verification | 4/4 direct merges; both later repair PRs also verified | Focused production-route tests and quick gates are recorded. Full-suite evidence was not uniformly available or green. |
| Direct-merge owner closure | 2/4 | `polylogue-lkrc.4` and `polylogue-303r.2.1` closed; `polylogue-866e` remained open and `polylogue-1xc.13` remained in progress. |
| Code-delivery track closure | 3/5 | Counting four direct merge tracks plus the locally reimplemented CaptureJob track, `polylogue-06zm.1` also closed. This is a secondary attribution view, not a package conversion rate. |

`FUNNEL.json` carries the same counts with explicit entity types, confidence, and denominator notes.

## Reconciled campaign identity

Four superficially conflicting counts refer to different scopes:

- The package ledger contains 28 canonical package revisions.
- Those revisions map to 26 unique conversation IDs because packages 6/7 and 23/26 are same-chat alternatives or revisions.
- `polylogue-3v1` reports a production audit of 27 Sol Pro campaign conversations.
- The corpus README lists 28 captured sessions from 2026-07-15 21:34 through 2026-07-16 08:30, including the later `6a587a8c` “Live Action Proof Request”.

The most plausible reconciliation is 27 campaign conversations plus one later proof session in the broad temporal set, with two package-bearing conversations producing two revisions each. That reconciliation is source-supported but not independently provable without the raw session roster. The earlier `polylogue-s2x7` one-session/one-deliverable framing is superseded.

## Direct delivery outcomes

| Package | Mission | PR / merge | Current owner state | Boundary that made the result honest |
| --- | --- | --- | --- | --- |
| 2, `5e2363a1…`, 40,844 B | Lineage order independence | #2922 / `b55f3fd…` | `polylogue-866e` open | Canonical sibling ordering and missing-cut behavior merged; persisted semantic cuts, crash proof, and complete nested propagation remained open. The competing continuity package was not applied. |
| 12, `ee58f741…`, 15,136 B | Raw-authority repair | #2923 / `81142d1…` | `polylogue-lkrc.4` closed later | A narrow complete-bundle admission defect merged; immutable census/conservation was explicitly not claimed. PR #2957 later repaired parser-drift membership replay. |
| 23, `17d8a28e…`, 68,247 B | Named-source freshness | #2924 / `b6c78ad…` | `polylogue-1xc.13` in progress | Bounded read-only exact-source projection merged; two live receipts and remediation authority remained outside the PR. |
| 25, `6b6f6718…`, 67,982 B | Sinex publication convergence | #2925 / `36001d0…` | `polylogue-303r.2.1` closed | Durable local publication obligations merged; real transport/deployment remained `polylogue-303r.2.2`. Coordinator review corrected an invalid cross-database atomicity claim to ordered durability. |

These four packages total 192,209 compressed bytes, **0.29% of the 65,938,477-byte corpus**. Their median compressed size is 54,413 bytes. The 24 non-direct-merge packages have a median size of 76,850 bytes, but that comparison is heavily confounded by mission type and large embedded fixtures.

## Package shape, content, and yield

All 28 packages contained something classifiable as patch, tests, design, and report material. Therefore formal package completeness did not discriminate mergeability.

Ten packages were at least 1 MiB. They account for 64,865,984 bytes, 98.4% of compressed corpus size, and none merged directly. The large members are mostly evidence or payload bulk rather than proportional implementation substance: an 8.05 MB replacements tar in package 1, an 8.08 MB snapshot appendix in package 4, a 16.75 MB synthetic SQL fixture in package 13, an 8.33 MB rollout-plan Markdown file in package 24, a 2.08 MB freshness fixture in package 26, and an 8.34 MB pairing fixture in package 27. Package size should therefore be measured by content composition and declared necessity, not used as a standalone quality cutoff.

Mission shape was more informative:

- Direct merges targeted a concrete current production route and admitted residual scope.
- Nine of the twelve pending packages were blocked by declared dependencies; one by deployment, one by preserved-draft reconciliation, and one by an admission prerequisite.
- Broad provider identity, pairing, and incorporation designs supplied useful constraints without a safe current patch.
- Duplicate alternatives appeared both within a chat and across missions. Same-chat lineage was not recorded because every ledger `iteration` field is null.
- Rejected work either covered too little of the claimed surface, introduced a competing seam, or depended on synthetic evidence that did not establish current product behavior.

`PACKAGES.csv` records the 28 package revisions, dominant content category, disposition, ownership, and direct PR mapping.

## Iteration and local repair

The campaign’s most valuable implementation work often happened after package generation:

1. The CaptureJobs package was preserved on `feature/integration/capture-job-authority` at `ba340c71a` / `8ecc34ecc`, with focused tests and a green quick gate, but was not merged because `paired:<provider>` could not prove exact account scope and the adapter conflicted with current generic BrowserAction transport. PR #2953 then implemented receiver-authoritative identity, HMAC-reduced provider account scope, leases, CAS checkpoints, receipts, recovery, and typed legacy orphans against current source, closing `polylogue-06zm.1`.
2. PR #2923 fixed the initial complete-bundle admission defect, while PR #2957 later fixed parser-drift replay against the persisted CAS witness and deployed that correction. The second PR is a real downstream repair, not a second package result.
3. PR #2925’s integration review corrected the package-era claim that source and index writes could be one cross-database transaction. The merged contract is ordered source durability before rebuildable index projection.
4. PR #2922 explicitly refused the continuity package’s competing scenario seam and incomplete route coverage while retaining a smaller current-source lineage fix.

The ledger cannot quantify repair effort, changed lines, elapsed integrator time, or patch-retention percentage because it records no repair events and leaves PR/merge fields null even for direct merges. This missing telemetry prevents a trustworthy “package code retained” metric.

## Timing and cadence

The broad session window spans about 10 hours 56 minutes; canonical download timestamps span 23:08 through 07:02 and occur in bursts. The four direct package PRs were open for approximately 30, 34, 70, and 101 minutes before merge, with a median of about 52 minutes.

Three direct PRs were created before the canonical package download timestamp, and PR #2924 merged before that timestamp. Therefore the filename timestamp is a custody or deduplication observation, not a generation-completion or integration-start timestamp. It cannot support acquisition-latency or time-to-PR analysis. The next campaign must emit stage events from provider submission through closure rather than derive timing from filenames and Git history.

The campaign infrastructure itself evolved during execution: #2913 introduced the Sol-specific queue; #2918 made launch orchestration advisory and canonical capture authoritative; #2919 reconciled ordinary capture; #2926 operationalized canonical missions; #2928 removed campaign concepts from product code in favor of generic BrowserAction plus external orchestration; #2930 repaired replacement/reingest failures. This moving architecture is a major confounder for any prompt-level effectiveness claim.

## Decisions

The next campaign should preserve the current external campaign workspace and generic product boundary, extend the existing `schemas/result.schema.json` rather than invent a new product protocol, and make every funnel transition an evidence-backed immutable event. Dispatch should be dependency-aware and integration-capacity-aware. Implementation jobs should require one current owner, a named production route, bounded write/avoid sets, explicit residual scope, and anti-vacuity tests. Broad or dependency-premature missions should be routed deliberately to analysis/research rather than forced to emit pseudo-implementation packages.

The detailed decisions, owners, acceptance criteria, and falsification tests are in `DECISIONS.md`. The proposed additive replacement for the existing external-agent result schema is `RESULT-TELEMETRY-V2.schema.json`.

## Limitations and missing evidence

This report did not access a live browser, provider account, daemon, deployed archive, or current `ops.db`. The project-state archive does not include the raw canonical session exports for the 27/28-session campaign set, provider action receipts, per-turn terminal markers, historical output attachment blobs, or operator interaction logs. It therefore cannot independently count terminal turns, time-to-first/terminal output, asset-offer events, acquisition retries, exact manual effort, or live replay success.

The snapshot is marked dirty, but its branch-delta patch and changed-file list are empty, so the exact uncommitted state is unavailable. The Git bundle supplies tracked history and refs; it does not recover omitted live state. PR verification is based on the recorded PR/Bead evidence, not rerun tests. The 325-unit older corpus was used only as context for intake breadth, not re-adjudicated package by package.

## Value of another iteration

A second pass over the same archive would add little beyond editorial refinement. A materially higher-value iteration requires one or more of: the canonical 27-session roster and raw turn/attachment exports; BrowserAction submission receipts; live replay results after #2930; the campaign’s `ops.db`/capture-health events; and timestamped integration/worktree/verification receipts. With those inputs, the funnel can be made genuinely time-based, output-capture recall can be measured, same-chat repair yield can be separated from fresh-chat yield, and prompt/mission associations can be tested without relying on custody filenames or prose reconstruction.
