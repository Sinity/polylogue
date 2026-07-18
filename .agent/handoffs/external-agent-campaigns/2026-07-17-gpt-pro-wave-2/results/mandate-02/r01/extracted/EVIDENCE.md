# EVIDENCE — source, Beads, history, and contradictions

## Authority chain

The code authority is the supplied `polylogue` project-state archive, whose manifest records `master` at `bf8191b3f56aa40da8f271df7f3385c712825497`, generated `2026-07-18T013442Z`, with a dirty working tree. The archive contains an all-refs git bundle, a working-tree tarball, Beads export/history, repository slices, GitHub issue/PR snapshots, and audit manifests.

The implementation was based on the reconstructed working tree plus bundle history. Current source was treated as authoritative when older Beads notes named commits or phases that did not exist as independent objects in the supplied all-ref bundle.

## Repository source findings

1. `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd` introduced the current provider-neutral topology/claim substrate, Claude orchestration parser/projection, work reconciliation primitives, SQLite graph persistence, repository traversal, and focused tests in one squash merge (`feat: harden archive continuity closure (#3051)`).
2. Existing `work_evidence.py` explicitly stopped before repository effects/evaluated satisfaction. Existing `work_reconciliation.py` represented a reconciliation result as a claim-to-effect relation, which did not preserve evaluated acceptance as a separate fact.
3. The existing storage route already composed through `SessionRepository` -> repository mixin -> `SQLiteQueryStore` -> `storage/sqlite/queries/work_evidence.py`. The patch extends this route instead of creating another service/store.
4. `index.db` is rebuildable derived state. The repository policy requires canonical DDL plus a semantic-reparse declaration for meaning-changing graph fields, not a durable migration chain.
5. Existing `SessionCorrelationResult` classifies `explicit_ref`, `file_overlap`, and `time_window`; this patch projects the first as direct and forces the latter two to candidate-only associations.
6. Existing Claude orchestration facts carry admitted artifact source paths. Missing path-to-`EvidenceRef` matches previously risked borrowing coordinator evidence through shared identities. The patch retains path refs and unresolved state instead.

## Beads findings

### `polylogue-1vpm.6.2`

State: open, P1.

The Bead requires authority-bearing git/GitHub/Beads/artifact/verification effects; direct identifiers first; candidate-only overlap; repository/corpus snapshots; branch-local Beads; squash and correction support; separate evaluated judgments; a seeded production query; and mutation failures for collapsed boundaries.

The implementation directly targets those substrate and synthetic-route requirements. Its incident-specific AC 5 and full-environment portion of AC 8 remain open.

### `polylogue-1vpm.6.1`

State: in progress.

A 2026-07-17 note says the substrate landed in commit `fece60e4d`. That commit is absent from the supplied all-ref bundle. The current source is present in squash commit `bce7336d3...`, whose changed-file set includes the named work-evidence, Claude, storage, and tests. This is treated as a stale pre-squash commit reference rather than proof of a separate reachable commit.

### `polylogue-2qx.2`

State: open, P1, updated `2026-07-17T23:49:26Z`.

Its current note says: the prior claim was orphaned, the claiming session had closed, no matching commits existed on master since 2026-07-14, and the Bead was reset open in the 2026-07-18 war-room sweep. Therefore the missing coordinator/run-state/journal/transcript-meta/adopt evidence cannot be invented or marked complete. The patch exposes unresolved/degraded handling but does not claim source admission that the snapshot does not contain.

### `polylogue-t8t`

State: in progress, P0.

A note reports seven deterministic continuity scenarios and independent fixture-owned oracles in PR #3018, while the real MCP cold-model/live terminal walk remains assigned to `polylogue-z9gh.7`. This package does not duplicate that terminal gate.

### `polylogue-z9gh.7`

State: open, P0.

The terminal gate depends on `polylogue-1vpm.6.2`, `polylogue-2qx.2`, `polylogue-t8t`, `polylogue-z9gh.3`, and `polylogue-z9gh.9.1`. Its exact census requires coordinator `cf0c6474-da22-44be-af3e-666037aa5ea4`, run `wf_54d4fb2e-841`, four invocations, 50 calls, 91 attempt transcripts, 65 results over 49 completed keys, one unresolved key, final structured result, and exclusion of 38 unrelated child sessions. No admitted raw incident corpus was supplied for executing that proof.

## Fixture evidence in this patch

The synthetic Beads fixtures are committed text inputs, not live tracker output:

- `work-effects-baselines.jsonl`: baseline for `polylogue-7fj` plus a second issue used for one-PR-to-many-Beads relation coverage.
- `work-effects-interactions.jsonl`: create, edit, claim, close, and later correction/supersession records.

The route test assigns each direct change an archived `EvidenceRef` and session identity. The claim event deliberately stores its session only on the association edge, proving the production query consumes the complete relation rather than relying on one convenient node layout. A separate time-only session points to the close effect as a candidate and is never returned among direct matches.

The route also records a snapshot commit, a squash-merged PR, `PATCH.diff` artifact identity, a passing synthetic verification receipt, and an acceptance assertion. The assertion traverses through an evaluation node; it does not alter any effect or tracker baseline.

## Contradictions and uncertainty

1. Beads commit note vs bundle history: `fece60e4d` is not reachable/present; current substrate source is in `bce7336d3`. Current source wins.
2. Older z9gh notes mention checkpoint commit/PR identities not independently present in the supplied bundle. The current master source and current Bead states are used instead.
3. `polylogue-2qx.2` describes expected incident artifacts but explicitly says the implementation claim was orphaned. Expected identities are modeled only as unresolved coverage gaps when absent.
4. Time/file overlap can be useful discovery evidence but is not causality. The patch encodes that uncertainty in the edge kind and state.
5. A source ref may be shared by multiple effects, especially a squash commit cited by a PR. Exact effect identity wins; a non-exact shared source ref remains ambiguous and creates no arbitrary direct edge.
6. A missing Beads baseline does not authorize reconstructing current title/status/assignee from an interaction. It creates an unresolved issue baseline and preserves the admitted change separately.
7. Full lifecycle-file failures are not caused solely by v40: the authoritative v39 baseline already causes the two version-37 assertions to include versions 38 and 39 as invalid. This package records rather than masks that defect.

## Evidence not available

- operator live daemon or MCP server;
- private/current archive contents;
- browser state, tokens, secrets, GitHub API session, or deployment;
- complete 2qx.2 artifact family for the incident;
- authorized live Beads/Dolt history and exact 25-open-P1 before/after snapshots;
- managed full dependency/verification environment.

No conclusions in this package depend on those unavailable sources.
