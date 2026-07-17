# testdiet-03 evidence

## Snapshot and authority resolution

The project-state overview and manifest identify `master` at `b9052e09103502017c0f510ecc699aac395de23c`, generated on 2026-07-17. The all-refs bundle resolves `origin/master`, `HEAD`, and their merge base to that exact commit. `polylogue-branch-delta.patch`, `polylogue-branch-delta-files.txt`, and `polylogue-branch-delta-log.txt` are all empty.

The snapshot says `dirty=true`, but the supplied artifact does not expose a tracked dirty diff. The archived working tree's 2,503 present tracked paths compare byte-for-byte equal to the commit; 1,238 tracked paths are omitted from that artifact; one untracked `browser-extension/package-lock.json` is present. This is a contradiction in granularity, not evidence that the dirty marker is false: local ignored state or omitted tracked files may have contributed. The only honest apply baseline is the named commit, with the unresolved dirty marker explicitly retained.

Artifact fingerprints used:

- all-refs bundle: `fbbea0138040db868d03734031ebb260ddee6eb01f444dc09dcb4c5dd9d5de59`
- working-tree archive: `a6255730812ff6a2527aba566124022816f56efb3922c91e70f491612a3d290a`
- empty branch-delta patch: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Architecture findings

`architecture/05-derived-freshness.md` is direct authority for the survivor's identity model:

- currentness requires exact source identity, recipe identity, output contract, and a successful result in the active generation;
- generation, counts, timestamps, and boolean stale flags are insufficient;
- attempt identity, timing, producer/resource facts, and result integrity are separate from canonical request identity;
- privacy, authorization, retention, and deletion eligibility are lifecycle metadata rather than computational identity;
- incremental, targeted, full, synchronous, asynchronous, and restarted routes must converge to the same canonical facts for a fixed source snapshot and recipe;
- domain ledgers remain domain-specific instead of becoming one universal derivation framework.

The implementation follows that model with a test-local typed key and existing production workload receipts. It does not claim to land the shared production key owned elsewhere.

`areas/storage-durability.md` asks the first implementation slice to compare public/logical facts and declared content hashes across incremental and rebuild routes, use deterministic failpoints rather than sleeps, prove restart and overlay independence, and mutate a skipped materializer or stale FTS state. The survivor deletes real derived rows, reconnects, and repairs through production routes; no sleeps or private helper call-order assertions are used.

The dossier named these exact planned tests:

- `test_incremental_and_rebuild_public_facts_match`
- `test_overlay_state_does_not_change_rebuild_identity`

It also marked the cluster `prepared-not-execution-grade`, cited zero realized artifacts, and carried stale `git_head` `21f78b4d...`. The current supplied source is `b9052e...`; current source wins. This patch turns the proposed focused tests into realized executable artifacts, but it does not claim the upstream provider-profile or archive-wide certification work is complete.

## Beads findings

### `polylogue-1xc.14.1` — in progress, P1

Provider observations must remain workload authority; generated or existing fixtures may be known-answer oracles but must not create another workload/receipt vocabulary. The survivor reuses the existing deterministic demo provider files and existing `WorkloadEnvelopeSpec`/`WorkloadReceipt` types. It does not claim the broader archive-scale workload-profile program is complete.

### `polylogue-hjwr` — open, P2

This owns the broader deterministic differential lane: full rebuild versus incremental convergence, plus fast-forward as a third comparand, with automatic census of every derived table and volatile-column allowlists. The delivered survivor is deliberately narrower and semantic. It neither removes nor replaces this future certification lane.

### `polylogue-lyv4` — open, P3

The async targeted insight rebuild currently lacks the sync branch's scoped thread/tag refresh and internal commit parity. The survivor uses the synchronous targeted route and leaves an explicit follow-on to duplicate the law across async only after the production defect is fixed.

### `polylogue-wmsc` — open, P1

This owns a storage-neutral production `DerivationKey` separating subject, exact source identity, computational recipe, and output contract from generation, eligibility, and result integrity. The survivor mirrors those semantics locally for comparison without inventing another durable ledger or scheduler.

### `polylogue-1xc.12` — open, P1

FTS row counts cannot prove identity because SQLite rowid reuse can bind a ghost FTS row to another block. The survivor therefore joins contentless FTS to `blocks`, projects block ID and content hash, and tests an exact planted search hit rather than comparing counts.

### `polylogue-303r.7` — open, P3

Complete computational request identity is distinct from output integrity and lifecycle eligibility. The survivor keeps raw/recipe/output identity separate from user-tier overlay state and from attempt receipts.

### `polylogue-f2qv.5` — closed, P1

Provider-usage/materialized insight projections must self-heal through the insight rebuild path rather than remain ingest-only. The survivor checks the complete registered materialization set, materializer version, input watermark, and counts after targeted repair and full rebuild.

## Source findings

### Singleton Drive replay identity

The durable raw replay path parses JSON via a stream, so a one-document file reaches Drive lowering as a one-item sequence. Existing code appended `-0` whenever it recursed into a sequence. Initial source identity and fresh replay identity therefore diverged even though the source bytes were identical.

The correction preserves the fallback ID for sequence length one and retains indexed suffixes for true multi-document sources. Both `_lower_drive_like_payload` and `parse_drive_payload` are changed so the generic and direct APIs cannot drift.

### Existing production corpus is sufficient for the focused canary

`polylogue/demo/seed.py` and `polylogue/scenarios/corpus.py` already plant:

- a Gemini attachment;
- a temporary Claude session and explicit unknown semantics;
- Codex parent/fork/subagent lineage;
- tool calls/results and a terminal error;
- searchable content;
- session profiles, work events, phases, usage/materialization rows, and threads;
- durable user overlays.

The new source update adds one unambiguous search sentinel through provider-native Codex JSONL rather than inserting derived rows directly.

### Real production seams used

- Incremental source route: `parse_sources_archive`.
- Durable replay route: existing maintenance replay called by `maintenance rebuild-index`.
- FTS target repair: `repair_message_fts_index_sync`.
- Insight target repair: `repair_session_insights`.
- Generation lifecycle: `IndexGenerationStore`/`ArchiveIdentity` through the CLI command.
- Overlay writer: `upsert_assertion` into the user tier.
- Attempt records: existing workload spec/receipt types.

No API, rebuild harness, source format, or durable schema was introduced for test convenience.

## Historical findings

Relevant recent history includes:

- `f0c1b489b` — restore archive contract verification;
- `d6501ac46` — make raw replay batches component-aware;
- `d068d6482` — consolidate staleness and connection lifecycle;
- `74f045dd0` — consolidate FTS trigger DDL;
- `dfe52af4f` — split maintenance commands into lazy modules;
- `d199cb04a` — provider-usage self-heal and cost ledger;
- `df559d8b3` — authority-safe raw convergence;
- `a2bbd25d6` — offline index-generation promotion;
- `6a579d090` — monotonic raw-revision replay;
- `202a09c24` — contain authority-ambiguous raw replay;
- `25bea6f03` — align diacritic folding across FTS write paths;
- `3de00794e` and `2eee22a9f` — bound insight rebuild work and WAL.

These commits reinforce three constraints used here: replay must respect durable authority, generation promotion must be atomic and rollback-capable, and derived repair routes must share semantic outputs even when their scheduling differs.

## Contradictions and resolved choices

1. **Dossier head versus supplied source.** The dossier was generated at `21f78b4d...`; the supplied source is `b9052e...`. Current source and later Beads notes win.
2. **Prepared plan versus realized implementation.** The dossier had zero sensitivity artifacts and only proposed filenames. This package implements the proposed survivor and executes one representative production mutation, while leaving archive-wide census work open.
3. **Dirty marker versus empty branch delta.** The dirty marker is retained, but no tracked diff is recoverable. The patch targets the named commit and does not pretend to preserve unknown local bytes.
4. **Shared derivation identity versus framework creation.** Architecture calls for a small shared typed protocol, but `polylogue-wmsc` owns the production landing. The survivor needs exact identity now, so it uses a test-local immutable value and existing receipt types without a universal table or new lifecycle.
5. **Sync/async equivalence versus known async defect.** The focused mission requires incremental, targeted, restarted, and rebuild routes. The synchronous target route is production-supported. Async duplication remains certification work after `polylogue-lyv4`.
6. **Public facts versus private implementation parity.** The storage packet explicitly prefers public/logical facts and declared hashes over private row order. This patch follows that boundary and does not turn internal table-layout coincidence into product contract.

## Proposed duplicated certification checks

These are additions for later certification, not reasons to remove existing tests or create another rebuild harness:

1. Run the same derivation-key survivor through both sync and async targeted insight entry points after `polylogue-lyv4` closes.
2. Add `polylogue-hjwr`'s automatic derived-table census for incremental convergence, full rebuild A, full rebuild A-rerun, and fast-forward; require every derived table to be compared or explicitly allowlisted.
3. Add a stable ChatGPT full-export/browser-capture authority canary after ambiguity resolution, retaining the existing demo session as an independent known-answer oracle.
4. Promote the focused survivor to medium/nightly scale tiers using the shared workload-profile and receipt work from `polylogue-1xc.14.1`.

No existing test or helper is proposed for deletion in this patch. Any future deletion requires independent local dominance and mutation proof.
