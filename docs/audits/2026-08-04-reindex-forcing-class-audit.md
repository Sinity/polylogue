# Reindex forcing-class audit

**Date:** 2026-08-04

**Scope:** `polylogue-fsgdd`, `polylogue-wwph1`, and the current `polylogue-818fy` direct-blocker snapshot at `origin/master` `f6d20affa`
**Lane:** read-only archive and source audit. No `bd` command, Beads-state write, archive mutation, or production-source change was made.

## Decision procedure used

This audit applies fsgdd's required first-hit order: S durable corrupter, K stamp poisoner, O run step, V verification instrument, P parse content, then D derived operations. A P assignment remains gated until xselt's stamps are landed and proved. An uncertainty between S/K and P/D is kept gated.

## Evidence boundary

Evidence is a current-master source read, the supplied packet's bead prose, or a fresh read-only live receipt. Inference is the resulting forcing-class decision. The direct-edge set was read from the committed `origin/master:.beads/issues.jsonl` snapshot only, never through `bd`.

Fresh live receipt, 2026-08-04 06:12 UTC, from `verify_archive(Path("/realm/db/polylogue"), checks=[...])`:

- `tier-schema` is red: `source.db` is v24 but current source schema is v26, `index.db` is v46 but current index schema is v63, and `audit.db` is absent.
- `enum-superset-check` is red: `index.db.sessions.origin` and `index.db.session_links.dst_origin` reject `claude-design-session`.
- `pointer-coherence` is green. The active pointer and conventional `index.db` resolve to the same generation.

The complete live registry was deliberately not asserted fresh. Its corpus-scale checks did not finish in this read-only lane's command window. Rows below marked **packet baseline** are still known red from the supplied current bead material, but require a new preflight receipt.

## Current direct blockers

There are 22 non-closed direct blockers. Each has one class. “Keep” below records the classification and ordering consequence only. It does not request a dependency change or decide de-gating.

| Direct blocker | Class | Evidence | Inference and exact pre-reindex proof |
|---|---|---|---|
| `0x7nh` canary changelog differ | V | 818fy runbook step 3 and 0x7nh AC require a reviewed no-promote differential. | Instrument. Run the representative per-origin candidate, classify every diff, and attach the reviewed report with zero unclassified rows. |
| `2qx` OriginSpec | P | Its scope is source admission and normalized authorship/identity declarations. No current evidence shows the reindex itself writes an incorrect durable row. | Parse-content contract. Before relying on post-run repair, xselt must prove its per-origin stamp changes for every OriginSpec-controlled parse semantic, then the origin matrix must pass. Keep until that proof exists. |
| `4ts` lineage truth | K | 818fy says session composition changes content/hash semantics. xselt's current design fingerprints parser modules, `dispatch.py`, and `pipeline/ids.py`, but does not name the lineage write/composition implementation. | Fail closed as a stamp poisoner until xselt's lowering-fingerprint boundary explicitly covers lineage composition. Prove two equivalent fork/resume/compaction inputs produce the same composed tree and stamp invalidation on a lineage-code mutation. |
| `6e7m` title derivation | P | Title is a content-hash field, but this is title-resolution enrichment rather than the hash/comparison algorithm itself. | Parser-content repair conditional on stamps. Prove the affected Codex origin stamp changes when title derivation changes, then use a per-origin reparse plus title uniqueness census. |
| `a7gmk` final deploy sync | O | It is the runbook's explicit final package, backup, and durable-migration step. Fresh `tier-schema` is red. | Run the verified-backup, migration, package-pin, and deployment receipt. Re-run `tier-schema` and `enum-superset-check` against the live root and require green. |
| `a7xr.23` content-defined chunking decision | S | The proposed replacement changes how raw bytes are chunked and admitted into `source.db`, a durable tier. | Fail closed. Before or with the run, prove unchanged input bytes preserve raw identity and revision lineage across prefix growth, including restart/reacquisition. If deferred, document that the daemon is prevented from applying the old durable cursor path during the run. |
| `a7xr.25` redundant `session_events` payloads | K | 818fy records `session_events` as direct content-hash input. The xselt design does not yet prove its lower fingerprint covers this projection choice. | Stamp poisoner until fingerprint coverage is explicit. Prove event retention changes both the relevant stamp and the candidate's expected row diff, then obtain a reviewed differ classification. |
| `ey4ro` red-backlog map | V | Its AC is the mapping from every gating bug to an executable red instrument. | Verification instrument. Commit the current direct-edge mapping with an evidence reference per row, and run each claimed red or record `no-feasible-red` for pure run/design items. |
| `f1vg` corpus acceptance | O | 818fy names it a hard preflight requirement and f1vg owns the run-specific corpus receipt. | Run its full read-only corpus audit. Require zero absences and typed attachment/revision residues, or obtain explicit operator acceptance for each residual. |
| `fsgdd` forcing-class triage | V | It owns the ordered procedure and the current direct-edge assignments. | This report supplies an independent revalidation. Coordinator must record the assignments and the three fail-closed K/S calls before treating the classification as usable. |
| `ih67` Codex title enrichment | P | The bead is an origin-local title enrichment in canonical ingest. | Conditional parse-content. xselt parser-stamp proof for Codex plus a targeted reparse and a no-native-id-title census for the selected cohort. |
| `nas1` resume topology separation | K | It changes the same session-link and composition meaning that 818fy identifies as a reindex-sensitive lineage cluster. | Fail closed with `4ts`. Prove provider-native resume links and context-delivery evidence remain distinct under the fingerprint boundary, then run the lineage candidate differential. |
| `omsw` sidecar admission | S | Tool-results and workflow journals are acquired as independent raw sessions, which persists wrong durable source rows. | Run the real acquisition path on both artifact shapes. Assert zero standalone raw-session rows and correct sidecar association, then census existing durable repairs. |
| `qj5x` Beads-origin decision | P | It removes a source-origin parser projection. 818fy already describes it as a targetable post-run follow-up. | Conditional parse-content. Prove its parser/origin stamp changes, then reparse the Beads-origin cohort and preserve the work-evidence route. |
| `r9xsj` reconciliation receipt | V | Its AC is an explicit read-only PASS/FAIL gate over source-tier reconciliation. | Run its named source queries and corpus reconciliation census after the durable drain. Require zero quarantine, blockers, and unexplained duplicates. |
| `rrxe4` convergence property loop | V | It is a production ingest and convergence proof harness, not a product-state mutation. | Run `devtools test -k convergence_property` with an order-varying corpus and the historical anti-vacuity reproduction. |
| `rrxe4.1` inferred-corpus reindex properties | V | Its AC binds the property loop to persisted provider constructs before the production rebuild. | Build the representative corpus with unsupported-construct receipts and run the focused committed property command. |
| `s8s54` browser-capture origin repair | S | Forty-one pre-fix raws have a durably wrong `unknown-export` origin. | Run the existing repair actuator under its normal backup/receipt protocol. The read-only postcondition is zero matching browser-capture rows with `origin='unknown-export'`. |
| `slshy` positional provider ids | K | It is explicitly a K-class xselt dependency. Positional parser identifiers bypass the gysk3 fallback and destabilize identity. | Run the vintage-reorder fixture against every discovered positional-id call site. Require equal comparison-id sets and prove the lowering/parser stamp changes for the repaired code. |
| `sp72` Drive revision lineage | S | Changed Drive bytes are persisted with `revision_kind='unknown'`, no logical key, and quarantine authority. | Exercise changed-byte reacquisition through the real Drive acquisition path. Require governed revision lineage, predecessor linkage, and no new untyped/quarantined durable row. |
| `uqwd` lifecycle event anchoring | K | Cross-vintage event anchoring changes the comparison axis for otherwise identical messages. | Run the lifecycle-anchor-drift fixture and the recorded real-cohort replay. Require a stable non-conflict verdict before xselt bootstrap. |
| `wwph1` root-cause campaign | V | Its product is enumerated, class-tagged findings and graduated detectors. | For each worked class, commit the denominator/judged/confirmed ledger, the forcing class, and a registry-graduation or campaign-mortal disposition. |
| `xselt` bootstrap stamps | V | It is the mechanism that makes P repairs origin-scoped after this rebuild. | Pass schema-versioning policy, real write-path fingerprint tests, stability/change tests, and the candidate-generation 100% stamp-coverage registry check. |

## Assignments invalidated by merged work

The following direct edges still exist in the committed graph but their associated work is closed on current master. They are not live work gates. This is an audit observation, not an instruction to remove edges.

| Closed direct blocker | Prior class | Current consequence |
|---|---|---|
| `0qfy`, `7zp4` | K | Former vintage/NFC stamp-poisoner assignments are satisfied. xselt still has the closed dependencies recorded, while `slshy` remains the open K dependency. |
| `2hwl`, `5iz4`, `foee`, `mvcbi`, `xofj` | P | Former parse-content gates are satisfied. Their closed edges must not be counted as remaining preflight work. |
| `4ts.10` | P | The null lineage-status defect was fixed. It is distinct from the still-open `4ts` lineage-composition K call. |
| `2qrx`, `ix5r`, `qsagp` | D | Cursor and derived catch-up assignments are satisfied. Their historical detector coverage remains useful, but they no longer block the run. |
| `5xxmc` | O | The known dead catch-up gate is closed. A fresh post-run convergence receipt is still required by the runbook. |
| `6753s` | S | Byte-duplicate supersession is closed, but r9xsj still owns the zero-survivor reconciliation proof. |
| `gzgyl` | P | The material-origin regression repair is closed. Keep its expected-delta count as post-run evidence, not a current work gate. |
| `t0m73` | V | Registry productization is closed and the code now invokes `REINDEX_ACCEPTANCE_CHECKS` before promotion. The current implementation is real evidence, but the old Bead edge is not a remaining task. |

## Known red detector map

| Detector | Current evidence | Forcing class of failure | Owner and required proof |
|---|---|---|---|
| `tier-schema` | **Fresh red.** Source v24 vs v26, index v46 vs v63, and missing `audit.db`. | S | `a7gmk`. Verified backup, durable migration/deploy receipt, then a green live `tier-schema` receipt. |
| `enum-superset-check` | **Fresh red.** Two active-index CHECK lists miss `claude-design-session`. | S | `a7gmk` with the vocabulary/migration owner. Green live check after deployment, not merely a source test. |
| `source-index-coverage` (I1) | **Packet baseline red.** The t0m73 baseline recorded 7,200 unindexed heads with a novel-vs-byte-duplicate split. | S | `r9xsj` and `f1vg`. Re-run against the live root after drain; no untyped heads or index orphans. |
| `fts-parity` (I7) | **Packet baseline red.** 35,331 missing `messages_fts` rows and 10 thread gaps were recorded. | D | `818fy` terminal rebuild stage. Candidate generation must pass `fts-parity` before promotion. |
| `blob-refs-liveness` (I3) | **Packet baseline red.** 73,427 raw-payload plus 1,336 attachment orphans were recorded. | S | `0v4tn`, feeding `r9xsj`. A read-only cross-tier join must return zero before acceptance. |
| `embeddings-refs-liveness` (I4) | **Packet baseline red and explicitly waived.** 4,186 orphaned embedding refs were recorded. | D | `feu0`. The waiver keeps the finding visible but non-blocking. Re-run and remove the waiver only with a green receipt. |
| `message-count-projection` (I8) | **Packet baseline red.** One session had a projection mismatch. | D | `818fy`. Candidate `message-count-projection` must be green before promotion. |
| `convergence-freshness` (I6) | **Packet baseline red.** Backlog plus no recent convergence activity was the recorded condition. | D | Post-run daemon owner. Require a fresh health receipt after `polylogued run`; closed `5xxmc` is not substitute evidence. |
| capability-parity V2, active-leaf V4 | **Packet baseline red.** V2 reported zero parent links for Codex/Hermes/AISTudio; V4 reported 103 multi-leaf sessions. | P | `4ts` and `2qx`. Produce a per-origin capability matrix and a candidate differential proving the intended topology result. |
| vocabulary-honesty V1b | **Fresh supporting red plus packet baseline.** The current enum CHECK failure is concrete. | S | `a7gmk`. Same green live enum-superset proof as above, plus the declared vocabulary census if broader than these two columns. |
| `corpus-absences`, `corpus-attachment-fidelity`, `corpus-revision-fidelity` | **Packet baseline red.** f1vg records named absences and unresolved fidelity residue. | S | `f1vg` and `r9xsj`. The full corpus audit must finish with zero residue or explicit operator acceptance per residual bucket. |

## Current source confirmation

`polylogue/maintenance/archive_verification.py` now has a single class-tagged registry and an explicit waiver table. `polylogue/maintenance/rebuild_index.py` runs `REINDEX_ACCEPTANCE_CHECKS` before promotion, rather than the former FTS-only gate. The current acceptance subset is index-only: `fts-parity`, `lineage-sanity`, `enum-superset-check`, `session-lineage-acyclic`, `message-count-projection`, and `planner-stats`.

That implementation invalidates the old claim that t0m73 is unbuilt, but it does not establish the missing xselt stamp-coverage check, the source/user/embeddings cross-tier acceptance receipts, or the corpus acceptance audit. Those remain external preflight evidence, not candidate-generation checks.

## Coordinator actions

1. Record these 22 classes without changing dependency edges. Keep `4ts`, `nas1`, and `a7xr.25` K, and `a7xr.23` S, until xselt's fingerprint boundary and the raw-admission behavior are proved.
2. Treat closed direct edges as satisfied rather than active work. Do not re-open them merely because the historical edge remains in the graph.
3. Assign `a7gmk` the two fresh red receipts first. The live archive cannot pass a production reindex preflight while its source/index schemas and origin CHECK vocabulary lag master.
4. Have xselt explicitly cover title, lineage, and session-event semantics in its fingerprint boundary or obtain a narrower evidence-based reclassification before allowing P repairs to move after the bootstrap.
5. Before the full run, obtain fresh full-corpus receipts for f1vg/r9xsj, the reviewed 0x7nh canary report, and the candidate registry plus xselt coverage receipt.

## Verification

- `uv run --active python -u -c '... verify_archive(Path("/realm/db/polylogue"), checks=["tier-schema"]) ...'` reported the live schema mismatch and missing audit tier.
- `uv run --active python -u -c '... verify_archive(Path("/realm/db/polylogue"), checks=["enum-superset-check"]) ...'` reported the two missing `claude-design-session` CHECK values.
- `uv run --active python -u -c '... verify_archive(Path("/realm/db/polylogue"), checks=["pointer-coherence"]) ...'` reported pointer coherence OK.

The report itself is syntax-validated before commit with `markdown-it-py` parsing and `git diff --check`.
