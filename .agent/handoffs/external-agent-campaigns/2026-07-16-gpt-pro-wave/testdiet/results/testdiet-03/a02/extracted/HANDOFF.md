# testdiet-03 handoff — incremental and rebuild equivalence

## Mission and delivered scope

This package implements the requested substantial second pass against Polylogue's supplied project-state authority. It adds one production-route survivor that replays the same realized demo workload through incremental update/reprocess, restarted targeted repair, and a fresh blue/green index rebuild, then compares independently projected public facts under exact derivation identities. It also repairs a production replay defect exposed by that survivor: singleton Gemini Drive documents acquired through the raw replay stream were assigned a `-0`-suffixed fallback identity and therefore rebuilt under a different session identity.

The package is an apply-ready patch, not a replacement repository. It contains no supplied archive, database, generated index, or copied source snapshot.

## Snapshot identity

The supplied project-state archive reports:

- source workspace: `/realm/project/polylogue`
- generated: `2026-07-17T10:53:28Z`
- branch: `master`
- commit: `b9052e09103502017c0f510ecc699aac395de23c`
- commit subject: `fix(daemon): bound raw maintenance admission (#2975)`
- commit time: `2026-07-17T08:28:35+02:00`
- dirty marker: `true`
- `origin/master`: the same commit
- branch merge base: the same commit
- branch-delta patch, changed-file list, and commit list: empty

The archive does not contain a reconstructable tracked dirty patch. Its working-tree artifact is 7,469,508 bytes with SHA-256 `a6255730812ff6a2527aba566124022816f56efb3922c91e70f491612a3d290a`. A byte comparison against the supplied all-refs bundle at the named commit found 2,503 archived tracked paths, all identical to `HEAD`; 1,238 tracked paths were omitted by the working-tree artifact, chiefly `.agent`, plus `.beads`, `.claude`, `flake.lock`, and `uv.lock`. The artifact contains one untracked path, `browser-extension/package-lock.json`. Therefore `PATCH.diff` deliberately targets the committed snapshot `b9052e...`; the original workspace's dirty-state marker is preserved as unresolved evidence rather than guessed into the patch.

The all-refs bundle used to reconstruct Git history has SHA-256 `fbbea0138040db868d03734031ebb260ddee6eb01f444dc09dcb4c5dd9d5de59`.

## Evidence inspected

Repository instructions and test conventions:

- `CLAUDE.md` / `AGENTS.md`
- `TESTING.md`
- `pyproject.toml`
- `devtools` test configuration and pytest plugin policy

Mission and architecture authority:

- supplied `03-rebuild-equivalence(1).md`
- `architecture/05-derived-freshness.md`
- `areas/storage-durability.md`
- `dossiers/incremental-rebuild-equivalence.md` and its JSON companion

Production routes and dependencies followed:

- `polylogue/sources/dispatch.py`
- `polylogue/pipeline/services/archive_ingest.py`
- `polylogue/maintenance/replay.py`
- `polylogue/cli/commands/maintenance/_rebuild_index.py`
- `polylogue/storage/index_generation.py`
- `polylogue/storage/archive_identity.py`
- `polylogue/storage/fts/fts_lifecycle.py`
- `polylogue/storage/repair.py`
- `polylogue/storage/insights/session/rebuild.py`
- source/index/user tier schemas and writers
- `polylogue/demo/seed.py`
- `polylogue/scenarios/corpus.py`
- `polylogue/scenarios/workload.py`
- existing source-law, rebuild CLI, FTS, repair, insight, generation, overlay, and lineage tests

Beads and history inspected include `polylogue-1xc.14.1`, `polylogue-hjwr`, `polylogue-lyv4`, `polylogue-wmsc`, `polylogue-1xc.12`, `polylogue-303r.7`, and `polylogue-f2qv.5`, plus path history for source dispatch, durable replay, rebuild promotion, FTS repair, and insight materialization. Detailed findings and contradictions are in `EVIDENCE.md`.

## Mechanism

### Production correction

`polylogue/sources/dispatch.py` now preserves the source fallback identity when a Drive-like parsed sequence contains exactly one document. Multiple documents still receive deterministic `-<index>` suffixes. The change is applied in both lowering and direct Drive parsing, so acquisition-time lowering and durable raw replay agree.

Without this correction, the same Gemini file is initially represented as `aistudio-drive:demo-00` but rebuild replay sees the one-item JSON stream and produces `aistudio-drive:demo-00-0`. That drops the expected session and its attachment from the canonical comparison.

### Realized workload

The survivor uses Polylogue's existing deterministic demo archive rather than adding another corpus. The selected evidence spans six sessions:

- `aistudio-drive:demo-00`
- `claude-ai-export:demo-temporary-claude-ai`
- `codex-session:demo-lineage-parent`
- `codex-session:demo-lineage-fork`
- `codex-session:demo-lineage-subagent`
- `codex-session:demo-terminal-error`

The test appends one provider-native Codex response item to the terminal-error JSONL source. It ingests that source through `parse_sources_archive`, repeats the same reprocess to prove idempotency, inserts a user assertion overlay, deliberately removes the terminal session's FTS and materialized-insight rows, closes the connection, and repairs through newly opened production connections using scoped FTS repair and scoped insight repair.

It then projects the active incremental generation, invokes the real Click command:

```text
polylogue --plain ops maintenance rebuild-index --output-format json
```

and projects the newly promoted blue/green generation. Source and user tiers are proven to remain the same durable files while `index.db` resolves to the promoted generation.

### Exact canonical identity

Every projected fact is keyed by a test-local immutable `DerivationKey` with:

- subject and grain;
- exact source identity: raw ID, provider origin, native ID, source path/index, blob SHA-256, and blob size;
- recipe identity: SHA-256 over the exact production dispatch, ingest, replay, FTS, insight, and rebuild-command source files, plus source schema v13, index schema v37, and insight materializer v14;
- explicit output contract listing the projected columns and schema/materializer versions.

Active generation is carried by each route snapshot, not folded into the canonical key. Existing `WorkloadEnvelopeSpec` and `WorkloadReceipt` types prove that both routes retain the same workload/spec identity while producing distinct receipts, runtime identities, and generation IDs.

No production-wide derivation framework or universal table is introduced. The shared production protocol remains owned by the open `polylogue-wmsc` work; this survivor binds the required facts exactly without preempting that API.

### Independently selected facts and oracle

Each route is queried independently. The expected facts originate from the planted provider-native files and declared transform invariants, not from serializing one production result as the expected value for the other. The test asserts:

- session identity, origin, lineage fields, active leaf, kind, and message count;
- message IDs, parentage, order, roles/types, active-path flags, and content hashes;
- block IDs/types/text, tool identity/outcome, searchable text, and content hashes;
- Gemini attachment identity, media/acquisition facts, reference position, and deliberate null metadata;
- parent/fork/subagent lineage links, branch point, inheritance, resolution status/method/confidence;
- an exact FTS hit for the planted update block;
- six session profile rows and materializer/source-watermark facts;
- all nine registered insight-materialization types and their planted input counts;
- four thread projections, membership, depth, branch count, total messages, and work-event breakdown;
- deliberate `unknown` terminal-state semantics and absent attachment metadata;
- source blob digest and size computed directly from the planted JSON/JSONL bytes.

The direct incremental and rebuild projections must then be exactly equal for the same derivation keys.

### Overlay separation

The survivor records source snapshot, raw/content identity, and workload identity before and after inserting a user-tier assertion. Those identities must remain unchanged. The overlay is then compared byte-for-byte before and after index-generation promotion, proving deliberate durable rejoin rather than inclusion in source or content hashes.

## Decisions

1. **Use the real rebuild command.** No parallel rebuild harness was added. The survivor calls the production Click command, including generation creation, raw replay, materialization, readiness, and atomic promotion.
2. **Use existing demo evidence.** The selected corpus already includes attachment, temporary, lineage, terminal-error, tool, search, materialization, and unknown/absent cases. This avoids a second workload identity while retaining independent known-answer assertions.
3. **Compare public/logical facts, not private row order.** Ordered projections are used only to canonicalize public columns. Attempt timestamps, run IDs, and private representation details are excluded.
4. **Keep attempts distinct.** Canonical fact equality does not collapse execution receipts or generations.
5. **Use the synchronous targeted insight repair.** `polylogue-lyv4` documents that the public async targeted branch currently performs an unscoped archive-wide wipe and lacks internal commit parity. This patch does not hide or duplicate that defect.
6. **Delete nothing.** Existing rebuild, migration, blob, lineage, FTS, and overlay tests retain distinct obligations. The dossier's possible dominated deletions remain certification candidates only.

## Changed files

- `polylogue/sources/dispatch.py`: 4 insertions, 2 deletions. Preserve singleton Drive source identity in lowering and parsing.
- `tests/unit/sources/test_source_laws.py`: 20 insertions. Focused regression for direct and generic parsing entry points.
- `tests/unit/storage/test_incremental_rebuild_equivalence.py`: 1,313 insertions. Full production-route survivor and overlay-identity law.

`PATCH.diff` has 1,337 insertions and 2 deletions across those three files. No `FILES/` directory is included because the unified diff fully disambiguates the changes.

## Acceptance matrix

| Mission obligation | Result | Evidence |
| --- | --- | --- |
| Same realized workload through incremental and fresh rebuild | Met | Demo seed plus one provider-native update; production ingest and rebuild command |
| Incremental update and idempotent reprocess | Met | First update processes terminal session; second processes no IDs and adds no raw row |
| Targeted repair after restart | Met | FTS and insight rows removed, connection closed, repaired through fresh production connections |
| Session/message/block identity and active path | Met | Independent exact projections plus planted constants |
| Attachments | Met | Gemini attachment and reference projection; singleton replay bug repaired |
| Lineage | Met | Parent/fork/subagent links and thread topology |
| Search | Met | Exact contentless-FTS join to planted update block and content hash |
| Materialized insights | Met | Profiles, nine materialization types, and thread outputs at materializer v14 |
| Absence/unknown semantics | Met | Null attachment metadata and explicit `unknown` terminal states |
| Exact source/recipe/output identity | Met | Derivation-keyed projections and direct source blob digests |
| Active generation rather than stale flag/count | Met | Projection is bound to active generation; rebuild generation must differ and be promoted |
| Distinct attempt receipts | Met | Same workload spec ID, different generation/runtime/receipt IDs |
| User-tier overlay excluded from content/workload identity | Met | Before/after identity checks and exact preservation across promotion |
| Expected facts independent of production result | Met | Hard-coded planted invariants and raw-byte digest checks on both routes |
| Omitted-stage/stale-row mutation named | Met | Skipping insight materialization, FTS repair, or retaining stale terminal rows is named in the survivor; stale rows are physically planted |
| Representative production mutation killed | Met | Reinstating old singleton `-0` suffix makes both focused source law and full survivor fail |
| Durable/derived tier and sole-writer rules | Met locally | Durable source/user files are reused; only the inactive index generation is rebuilt and promoted through production command |
| No second rebuild harness or deleted tests | Met | Patch only adds a survivor and narrow production correction |
| Full derived-table auto-census certification | Deferred | Remains owned by open `polylogue-hjwr` and is not silently claimed here |
| Live operator deployment proof | Unverified | No access claimed to daemon, browser, secrets, NixOS deployment, or operator worktree |

## Apply order

From a checkout of the named snapshot:

```bash
git checkout b9052e09103502017c0f510ecc699aac395de23c
git apply --check PATCH.diff
git apply PATCH.diff
```

Then run the focused command recorded in `TESTS.md`. The patch was generated against that exact commit and independently applied to a fresh detached worktree before packaging.

## Risks and limitations

- The supplied workspace is marked dirty, but no tracked dirty delta can be reconstructed from the supplied artifacts. Applying to an operator worktree with undisclosed edits may conflict; inspect any local delta first.
- This is a focused semantic canary over six deliberately selected sessions, not the archive-wide automatic census of every derived table proposed by `polylogue-hjwr`.
- The survivor exercises the synchronous targeted insight path. Async targeted parity remains blocked on `polylogue-lyv4`.
- The selected canary avoids treating ambiguous ChatGPT full-export/browser-capture authority as a stable known-answer source. That authority family needs its own repair and duplicate certification before inclusion.
- No formal mutation campaign was run across every named omission. One representative production mutation was executed and killed; the survivor also plants real stale FTS and insight rows so skipped repairs fail naturally.
- Two broader pytest attempts did not complete inside a 120-second harness. Focused, adjacent, static, apply, deterministic rerun, and representative mutation checks completed; the whole repository suite remains unverified.
- No live daemon, browser receiver, archive-scale source database, NixOS deployment, secrets, or operator current worktree was used.

## Remaining value of another iteration

A **small repair pass** would be valuable only for CI/platform differences, patch conflicts against the operator's actual dirty worktree, or a focused failure discovered by the full repository gate.

A **substantial certification pass** would add the open `polylogue-hjwr` automatic derived-table census, compare full rebuild, incremental convergence, and fast-forward routes, repair and include async targeted insight parity, and add a stable ChatGPT authority canary. That would broaden coverage materially, but it should extend this survivor rather than replace it or delete existing focused tests prematurely.
