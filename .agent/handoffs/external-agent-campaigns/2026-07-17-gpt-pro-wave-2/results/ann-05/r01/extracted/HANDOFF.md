# HANDOFF — Public claims rendered view (`polylogue-3tl.16`)

## Mission completed

This patch replaces the checked-in public-claims ledger as an authority with one rendered projection over `AssertionKind.FINDING` rows, the merged assertion judgment lifecycle, supersession, bounded publication/privacy declarations, and a narrow shared evidence-integrity verdict interface owned by `polylogue-37t.14`.

There is no new durable claims table. The FINDING assertion remains the durable claim owner; canonical judgment rows remain the review owner; `polylogue-37t.14` remains the evidence ancestry/support owner; `docs/public-claims.yaml` becomes a generated compatibility view.

The patch also seeds the three current claim-vs-evidence headline findings, renders README/launch/findings-page/verified-export presets as Markdown and JSON, registers the generated surface, enforces public-copy sanitization, and makes stale, private, unsupported, circular, frame-incomplete, closed-loop, broken/unresolved, and unreviewed states fail closed.

## Snapshot identity and authority

- Project: `polylogue`
- Snapshot branch: `master`
- Snapshot commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Commit subject: `fix(repair): harden raw authority convergence (#3046)`
- Snapshot generated: `2026-07-17T180950Z`
- Snapshot source recorded by the archive: `/realm/project/polylogue`
- Local implementation branch used for the draft: `snapshot`, checked out exactly at the commit above
- Snapshot metadata reported `dirty: true`, but `polylogue-branch-delta.patch` and `polylogue-branch-delta-files.txt` are both zero bytes. The supplied tracked source therefore has no branch delta; the dirty bit reflects ignored/excluded local runtime state captured elsewhere in the project-state package.

The unified patch is against the named commit, not against a later all-refs tip.

## Authority inspected

The implementation followed these sources beyond the obvious claim files:

- Full Beads records for `polylogue-3tl.16`, `polylogue-37t.14`, and `polylogue-37t.12` from `polylogue-beads-export.jsonl`.
- Merged PR #2791 authority at commit `5aa34e6c5d231c952529174febe99b2a58f4da07`, including deterministic FINDING candidates, canonical judgments, active successor promotion, SAVEPOINT bulk semantics, idempotency/conflict behavior, and injection separation.
- Existing assertion substrate in `polylogue/storage/sqlite/archive_tiers/user_write.py` and lifecycle policy in `polylogue/core/assertion_lifecycle.py`.
- Existing finding adapter in `polylogue/storage/sqlite/finding_provenance.py`.
- Existing `polylogue/insights/measurement/evidence_ancestry.py`, whose own documentation defers live-storage wiring. It is deliberately not promoted into a second 37t.14 evaluator.
- Existing generated-surface registry, command catalog, docs-surface map, release-readiness gate, and public-claims verifier/tests.
- The old `docs/public-claims.yaml` ledger and public README/demo/finding copy.
- The project-state `.agent/demos/claim-vs-evidence/` packet, including the report, sanitized public summary, methodology, and generated timestamp.
- Repository instructions and topology/layering constraints. No new `AssertionKind`, durable table, migration, or dependency was added.

See `EVIDENCE.md` for the findings and reconciliations.

## Projection architecture

### Durable facts

`PublicClaimDeclaration` is embedded in the existing `polylogue.finding.v1` assertion value. It carries only bounded publication fields:

- public wording;
- scope;
- caveat;
- sanitized public evidence refs;
- selected presets;
- `public` or `held_private` disclosure.

The declaration is not a support ledger. It cannot mark a finding supported. The deterministic FINDING writer remains candidate-only for detector-authored seeds.

Additional finding value fields record the source epoch, evaluation receipt ref, and frame ref. They are additive JSON payload fields, so no user-tier schema migration is required.

### Storage adapter

`list_public_finding_inputs()` reads public-declared FINDING rows and canonical judgment facts from `user.db`. It resolves which judgment produced an active successor, but it does not resolve evidence refs and does not calculate support, drift, cycles, closure, frame completeness, or ancestry privacy.

Supersession is projected from the existing assertion relationships. Multiple live active findings for one stable key fail closed as `unknown` with `multiple-live-findings`.

### 37t.14 interface assumption

The snapshot does not contain the shared `polylogue-37t.14` evaluator or a persisted verdict surface. This patch therefore defines the smallest consumer interface needed:

```text
EvidenceIntegrityProvider.verdict_for(finding_ref)
    -> EvidenceIntegrityVerdict | None
```

The verdict vocabulary is copied from the current 37t.14 Bead authority:

```text
supported | partially_supported | not_supported | stale |
closed_loop | cycle | unresolved | frame_incomplete | held_private
```

The bounded verdict carries sanitized public evidence/remediation refs, stable reason and blind-spot codes, as-of epoch, frame ref, and definition ref. It intentionally does not expose or re-walk decisive ancestry paths. A missing verdict is `unresolved`, never fresh.

`devtools render public-claims --verdicts <receipt.json>` accepts a narrow export schema named `polylogue.evidence-integrity-verdicts.v1`. When 37t.14 lands, its adapter should implement this protocol or translate its authoritative receipt into it. The projection must not absorb evaluator logic during that integration.

### One projection, four presets

`project_public_claims()` computes status once. `claims_for_preset()` only filters the already-computed rows. Both Markdown and JSON renderers consume the same `PublicClaimProjection` objects.

| Preset | Markdown/JSON | Detail policy |
| --- | --- | --- |
| `readme` | `docs/generated/public-claims/readme.*` | Compact public status; omits review and reason-code detail. |
| `launch` | `docs/generated/public-claims/launch.*` | Includes bounded blocker/reason codes; omits internal review detail. |
| `findings-page` | `docs/generated/public-claims/findings-page.*` | Includes review, privacy, evidence, epoch/frame, reasons, and remediation refs. |
| `verified-export` | `docs/generated/public-claims/verified-export.*` | Full public projection; also drives generated `docs/public-claims.yaml`. |

Preset parity is tested by stable key: a claim cannot have one status in README and another in launch/findings/export.

## Status semantics

The required public status remains the closed seven-value vocabulary. `integrity_status` and the badge retain the distinct 37t.14 blocking state where several failures must map to public `unknown`.

| Public status | Required conditions and meaning |
| --- | --- |
| `supported` | One current finding; approved publication and privacy; present 37t.14 `supported` verdict; as-of epoch, frame, and definition refs all present. |
| `partially_supported` | Same review/privacy requirements, with a qualified 37t.14 `partially_supported` verdict. |
| `not_supported` | Canonically rejected finding or 37t.14 `not_supported`. |
| `stale/needs-rerun` | 37t.14 reports `stale`; the old finding row is preserved and the view degrades. |
| `held_private` | Either the publication declaration or 37t.14 verdict is held private. Privacy hold dominates lifecycle state and redacts copy, evidence, statistic, and qualifiers. |
| `unknown` | Candidate/deferred/superseded/deleted finding, pending review, live-selection conflict, missing verdict, unresolved/broken evidence, cycle, closed loop, or incomplete frame. The exact integrity state remains visible in `integrity_status` and the badge. |
| `capability-only` | Explicit non-measurement statement with implementation refs; it carries no finding, statistic, or support verdict. |

Distinct blocking renderings include:

| Evidence condition | Public status | Integrity detail / badge |
| --- | --- | --- |
| broken or missing ref | `unknown` | `unresolved` / `[UNKNOWN · UNRESOLVED]` plus bounded reason code |
| circular ancestry | `unknown` | `cycle` / `[UNKNOWN · CYCLE]` |
| agent/assertion-only closure | `unknown` | `closed_loop` / `[UNKNOWN · CLOSED LOOP]` |
| stale epoch/hash/definition | `stale/needs-rerun` | `stale` / `[STALE / NEEDS RERUN]` |
| incomplete frame | `unknown` | `frame_incomplete` / `[UNKNOWN · FRAME INCOMPLETE]` |
| private-held declaration or ancestry | `held_private` | `held_private` / `[HELD PRIVATE]`, with redaction |
| incompatible/unsupported evidence | `not_supported` | `not_supported` / `[NOT SUPPORTED]` |
| only partial compatible support | `partially_supported` | `partially_supported` / `[PARTIALLY SUPPORTED]` |

A candidate judgment or accepted finding never implies evidence support. A `supported` receipt cannot override rejected, pending, conflicting, or private-held publication state.

## Seeded claim population

Values come from the supplied 2026-07-04 claim-vs-evidence packet. The default checked-in render intentionally has no fabricated judgment or 37t.14 receipt, so the three detector-authored finding seeds remain candidate/pending and `unknown · unresolved`.

| Stable claim key | Seeded value | Default current status |
| --- | --- | --- |
| `finding.silent-proceed-lower-bound` | 1,205 silent continuations among 5,000 inspected failures; lower bound `0.241`; 3,375 ambiguous; frame 42,033. | `unknown`, integrity `unresolved`, candidate/pending |
| `finding.handler-class-split` | Consequential: 4,175 / 930 silent / 2,842 ambiguous / 22.2754%; benign recovery: 634 / 172 / 455 / 27.1293%; other: 191 / 103 / 78 / 53.9267%. | `unknown`, integrity `unresolved`, candidate/pending |
| `finding.per-origin-inspection-counts` | Claude Code 3,752 of frame 31,555; Codex 1,241 of 10,429; Claude AI 7 of 49. | `unknown`, integrity `unresolved`, candidate/pending |
| `category.local-evidence-system` | Explicit product-category statement with README/architecture/proof refs; no measurement claim. | `capability-only` |

Each measured finding cites the public finding page plus the local packet report and sanitized public summary. Its source epoch is `2026-07-04T08:55:53.667311+00:00`, with explicit evaluation and sample-frame refs.

A production-route smoke accepted all three candidates through the canonical judgment writer, supplied current `supported` receipts, and rendered all three as supported without changing or duplicating the underlying finding identity.

## How README and outreach should cite claims

Use the stable key and link to the appropriate generated view. Do not copy a number without its status qualifier.

Recommended form:

```text
The 2026-07-04 bounded packet reported a 24.1% silent-continuation lower bound
(claim `finding.silent-proceed-lower-bound`; current status: see
`docs/generated/public-claims/launch.md`).
```

For current operator outreach, the checked-in status is not supported: all three measured claim keys are `unknown · unresolved` until canonical acceptance and a live 37t.14 receipt are supplied. Historical prose in the finding page is date-qualified and directs readers to the generated status view.

The category statement may be used as `category.local-evidence-system` with `capability-only`; it must not be presented as a performance result.

## Changed files

### Production and storage

- `polylogue/insights/measurement/public_claims.py` — typed projection, 37t.14 seam, status policy, privacy redaction, sanitization, preset payloads, Markdown/JSON rendering.
- `polylogue/storage/sqlite/archive_tiers/user_write.py` — additive public declaration/epoch/evaluation/frame fields for FINDING payloads and sanitized refs.
- `polylogue/storage/sqlite/finding_provenance.py` — storage-to-projection adapter over FINDING, judgment, and supersession facts.
- `polylogue/scenarios/corpus.py` — three exact claim-vs-evidence finding seeds and scenario population.

### Generated-surface control plane

- `devtools/public_claims.py` — builds/verifies the projection, receipt loader, parity, coverage, retired-copy, and sanitization checks.
- `devtools/render_public_claims.py` — render/check command with seed or live-archive inputs.
- `devtools/generated_surfaces.py`, `devtools/command_catalog.py`, `devtools/docs_surface.py`, `devtools/release_readiness.py` — registration and release gating.

### Public and generated docs

- `README.md`, `docs/demos.md`, `docs/findings/claim-vs-evidence.md` — stable markers, date-qualified finding language, links to current status.
- `docs/generated/public-claims/{readme,launch,findings-page,verified-export}.{md,json}` — eight generated preset artifacts.
- `docs/public-claims.yaml` — generated verified-export compatibility view.
- `docs/README.md`, `docs/devtools.md`, `docs/plans/topology-target.yaml`, `docs/topology-status.md` — generated docs/topology registration.

### Tests

- `tests/unit/insights/measurement/test_public_claims.py`
- `tests/unit/storage/test_public_claims_projection.py`
- `tests/unit/devtools/test_public_claims.py`

## Acceptance matrix

| Mission requirement | Result | Proof |
| --- | --- | --- |
| Stable status projection from findings, judgments, supersession, privacy, and shared verdict | Implemented | Pure projection plus real SQLite adapter/lifecycle tests. |
| No second durable ledger | Implemented | No table/migration; YAML is generated; FINDING payload owns declaration. |
| Consume 37t.14, do not re-walk ancestry | Implemented as dependency seam | Protocol and bounded receipt loader; no graph traversal in projection/adapter. |
| Four presets over one projection | Implemented | Eight generated files; parity tests compare every key/status. |
| Markdown and JSON with badges, refs, epoch/frame qualifiers | Implemented | All preset artifacts and renderer tests. |
| Broken/circular/stale/frame/private/unsupported block supported distinctly | Implemented | Public status plus distinct integrity status/badge; privacy redaction tests. |
| Seed three current demo headline findings | Implemented | Exact values and evidence refs asserted by storage tests. |
| Evidence epoch advance degrades without duplicate/rewrite | Implemented | Same active assertion ref transitions supported to stale; row count unchanged. |
| Public surface coverage/drift check | Implemented for current README/demo/finding surfaces | Marker verifier and generated drift tests. |
| Sanitized public-repo output | Implemented | Typed ref/prose/statistic sanitization; held-private redaction; artifact scan. |
| Default claims can be honestly published as supported now | Blocked by upstream authority | No 37t.14 evaluator/receipt and no operator judgments are present; defaults are unknown/unresolved. |

## Apply order

1. Check out commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.
2. Apply with `git apply --index PATCH.diff`.
3. Run `devtools render all --check` and the focused tests from `TESTS.md`.
4. For a live operator render, run `devtools render public-claims --archive-root <archive> --verdicts <37t.14-receipt>`.

`git apply --index` is recommended because `docs/generated/` is ignored by broad repository rules; the explicit binary additions are correctly staged with `--index`. A plain `git apply` writes them but may require `git add -f docs/generated/public-claims` before commit.

No dependency lockfile, schema migration, new durable table, or new `AssertionKind` regeneration is required.

## Risks and limitations

1. **37t.14 is absent in this snapshot.** The interface is intentionally narrow and source-compatible in spirit, but its final module name, export envelope, or reason-code vocabulary may differ. Integration should add an adapter, not rewrite projection semantics.
2. **The default checked-in measured claims are not supported.** They are detector candidates with no canonical acceptance and no integrity receipt. This is the intended honest state, not an unfinished badge substitution.
3. **37t.12 residual work remains upstream.** This patch consumes merged #2791 lifecycle authority. It uses the reviewed active successor plus a bounded disclosure declaration for publication/privacy. 37t.12 still owns richer evidence previews, queue health, expiry, and public workflow consolidation.
4. **Legacy static-ledger breadth is intentionally reduced.** The previous eight-entry file mixed proven, capability, aspirational, and retired statements. Only the category capability and the mission’s three real measured findings are migrated into the new authority. Additional public statements should be migrated as FINDING-backed or explicitly capability-only rows, not copied back as static statuses.
5. **Coverage scope is current, not universal.** The verifier covers `README.md`, `docs/demos.md`, and `docs/findings/claim-vs-evidence.md`; no separate tracked launch-post source exists in the snapshot. The generated launch preset is ready, but a future launch file must be registered and marked.
6. **Demo packet refs are local artifacts.** `.agent/demos/claim-vs-evidence/` is ignored in the Git checkout but was present in the supplied project-state archive and is the requested evidence source. Public checked-in prose also cites the tracked finding page. A release proof should preserve or publish a sanitized packet artifact if external readers must resolve every byte directly.
7. **One unrelated baseline test is red.** `test_archive_tiers_api_archive_debt_reads_archive_consistency` fails on both the pristine snapshot and patched tree with `sqlite3.OperationalError: database source_debt is locked` at `archive.py:11830`.
8. **The complete repository quick gate did not finish.** Its ruff format/check phases passed, then full-repository mypy exceeded the five-minute execution cap. Focused mypy over all touched production/test modules passed.

## Value of another iteration

A small repair iteration would be worthwhile once 37t.14 lands: bind its real verdict adapter/export, regenerate the checked-in statuses after operator judgments, and adjust names if its final interface differs. That should be localized if the Bead contract remains stable.

A substantial second pass would add real value only if the scope expands to migrate every remaining legacy public statement, register an actual launch/site corpus, publish resolvable sanitized demo packets, or implement 37t.14 itself. Those are separate authority/coverage programs rather than small defects in this projection patch.

## Verification summary

- Apply-ready patch check and `git apply --index` against a fresh clone of the exact snapshot: passed, 29 paths staged.
- Focused post-apply suite in fresh clone: 31 passed.
- Final affected production/regression suite: 118 passed.
- Ruff format/check over all touched Python: passed.
- Focused mypy over all touched production and core tests: passed.
- `devtools render all --check`: all generated surfaces synchronized; site links resolve.
- `devtools verify public-claims --json`: passed; 4 claims, 9 artifacts, no problems.
- `devtools verify layering --json`: zero violations.
- `devtools verify topology --json`: non-blocking; zero orphan/missing/conflict/kernel findings. Nine pre-existing storage TBD classifications remain.
- Live-route smoke: canonical accept + 37t.14-format receipts rendered all three measured claims supported.
- `git diff --check`: passed.
- Patch placeholder/source-copy scan: zero implementation placeholder tokens; no supplied archive names or archive bytes included.

Exact commands and outcomes are in `TESTS.md`.
