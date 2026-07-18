# EVIDENCE — Public claims rendered view

## Snapshot evidence

The supplied project-state package contains a working-tree tarball, an all-refs Git bundle, Beads exports, Git history, merged-PR records, repository slices, tests, docs, and ignored local demo state.

Identity came from `polylogue-overview.json` / `polylogue-manifest.json` and was confirmed in Git:

```text
branch: master
commit: 536a53efac0cbe4a2473ad379e4db49ef3fce74d
dirty: true
generated_at: 2026-07-17T180950Z
```

`polylogue-branch-delta.patch` is zero bytes and `polylogue-branch-delta-files.txt` is empty. This resolves the apparent contradiction between `dirty: true` and the tracked patch authority: ignored/excluded local state exists, but there is no supplied tracked branch delta.

The implementation clone was checked out at the exact commit. Later refs in the all-refs bundle were not used as source authority.

## Bead authority

### `polylogue-3tl.16`

Current design requires:

- stable public keys projected from FINDING assertions, judgments, supersession, publication/privacy review, and 37t.14;
- public statuses `supported`, `partially_supported`, `not_supported`, `stale/needs-rerun`, `held_private`, `unknown`, and `capability-only`;
- README/launch/findings/verified-export as presets over one projection;
- generated/import-only `docs/public-claims.yaml`;
- no local ancestry/staleness/cycle algorithm.

Its July 13/15 reconciliation supersedes the earlier static ledger design. The old YAML ledger is therefore input history, not current fact authority.

### `polylogue-37t.14`

The current open Bead defines one provider-neutral evaluator and the verdict vocabulary:

```text
supported, partially_supported, not_supported, stale, closed_loop,
cycle, unresolved, frame_incomplete, held_private
```

It owns decisive paths/witnesses, grounding compatibility, hash/definition/frame drift, cycles, closure, privacy, as-of frame, and remediation. Missing/uncomputed is unknown. Consumers must not invent parallel graph walkers.

Source search confirmed that no 37t.14 evaluator or receipt surface exists at snapshot commit `536a53e`. The existing `polylogue/insights/measurement/evidence_ancestry.py` is a legacy bounded model whose documentation says live storage wiring is deferred. The patch therefore consumes a narrow protocol and does not use that module for public support.

### `polylogue-37t.12` and merged PR #2791

The current Bead says merged PR #2791 is authoritative for candidate lifecycle transitions. Git commit `5aa34e6c5d231c952529174febe99b2a58f4da07` records:

- machine findings as deterministic candidates;
- canonical judgment rows;
- active successor promotion;
- explicit injection authorization;
- exact retry/conflict behavior;
- per-item SAVEPOINT bulk semantics;
- `AssertionKind.FINDING` on the existing user-tier substrate.

Residual 37t.12 scope is evidence disclosure, queue health/expiry, real-route proof, and duplicate public workflow consolidation. This patch consumes the landed transaction instead of reimplementing it.

## Source routes inspected

- `polylogue/core/enums.py` and assertion lifecycle types.
- `polylogue/core/assertion_lifecycle.py`.
- `polylogue/storage/sqlite/archive_tiers/user_write.py`.
- `polylogue/storage/sqlite/finding_provenance.py`.
- `polylogue/insights/measurement/evidence_ancestry.py`.
- `polylogue/daemon/convergence_standing_queries.py` call sites for FINDING writers.
- `polylogue/scenarios/corpus.py` and demo overlay convergence tests.
- `devtools/public_claims.py` before replacement.
- `devtools/generated_surfaces.py`, command catalog/registry, docs surface, release readiness, and render support.
- `README.md`, `docs/demos.md`, `docs/findings/claim-vs-evidence.md`, `docs/proof-artifacts.md`, and the old `docs/public-claims.yaml`.
- Existing storage/assertion, devtools/generated, daemon, CLI, and API facade tests.
- Repository architecture/topology instructions and registration traps.

No new `AssertionKind` was needed, so OpenAPI/CLI schema and user-audit every-kind regeneration was not triggered by an enum addition. `render all --check` still confirmed all generated surfaces remained synchronized.

## Demo packet evidence

The ignored `.agent/demos/claim-vs-evidence/` content was recovered from the supplied project-state XML slice rather than inferred from public prose. The relevant sanitized packet values are:

```text
generated_at: 2026-07-04T08:55:53.667311+00:00
archive schema: v24
inspected: 5,000
complete frame: 42,033
unpaired: 101
silent proceed: 1,205
acknowledged: 420
ambiguous: 3,375
silent lower bound: 0.241
```

Handler split:

```text
consequential: 4,175 failed; 930 silent; 2,842 ambiguous; 0.22275449101796407
benign recovery: 634 failed; 172 silent; 455 ambiguous; 0.27129337539432175
other: 191 failed; 103 silent; 78 ambiguous; 0.5392670157068062
```

Origin allocation/frame:

```text
claude-code-session: 3,752 inspected; frame 31,555
codex-session: 1,241 inspected; frame 10,429
claude-ai-export: 7 inspected; frame 49
```

The packet’s public summary contained a local `/home/sinity` path. That path was not copied. Seeded/public refs use repository-relative `.agent/demos/claim-vs-evidence/...` refs and the tracked finding page.

## Design decisions supported by evidence

1. **FINDING value extension, not a table.** Existing FINDING rows already carry a typed JSON value and deterministic identity. Adding publication/epoch/frame/evaluation fields there preserves substrate-first architecture and avoids a migration.
2. **Candidate defaults.** `upsert_findings_as_assertions` is deliberately candidate-only for machine/detector findings. Seeding supported status would contradict merged lifecycle authority.
3. **Separate support authority.** A judgment accepts publication/canonical claim state; it does not establish evidence integrity. The projection requires both review and a 37t.14 verdict.
4. **Unknown defaults.** Since 37t.14 is absent, missing verdicts render `unresolved`, not fresh.
5. **Integrity detail alongside seven public states.** 3tl.16 fixes a seven-value public vocabulary, while 37t.14 requires cycle, closed loop, unresolved, and frame incomplete to remain distinct. `integrity_status` and badges preserve those distinctions without inventing extra public statuses.
6. **Privacy hold dominance.** Either declaration privacy or an integrity-level hold redacts all claim content/evidence/statistics/qualifiers, even while the finding is still a candidate.
7. **Generated compatibility YAML.** Existing consumers retain a YAML artifact, but its header names the projection and generator; it is no longer editable fact authority.
8. **Marker coverage.** Stable HTML markers bind public copy to claim keys and let CI detect missing/unknown coverage without attempting unreliable natural-language claim extraction.

## Contradictions and resolutions

| Apparent contradiction | Resolution |
| --- | --- |
| Snapshot says dirty, branch patch is empty. | Dirty state comes from ignored/excluded local material; tracked source authority is exact commit `536a53e`. |
| Earlier 3tl.16 notes called the static ledger enforced/satisfied. | Later Bead design reconciliation explicitly makes it a rendered FINDING view. Later Bead authority and current mission win. |
| 37t.14 is a dependency but no implementation exists. | Define only the consumer protocol/export needed; defaults fail closed; do not implement or copy an ancestry walker locally. |
| 37t.12 remains open, but mission says consume merged lifecycle. | Core lifecycle from #2791 is landed and authoritative; residual queue/disclosure work remains clearly bounded. |
| Demo values are under ignored `.agent/` paths. | The supplied project-state archive contains them and the mission explicitly names them as value authority; patch seeds sanitized refs and keeps the tracked public finding page as an evidence ref. |
| Old ledger labels the 24.1% claim `proven`. | No current 37t.14 receipt exists, so the rendered view correctly downgrades it to candidate/pending and unresolved. |

## Patch provenance and exclusion checks

- `PATCH.diff` was generated by `git diff --full-index --binary HEAD` from the implementation tree at the named snapshot commit.
- Patch size before packaging: 184,609 bytes; 4,112 lines.
- Patch SHA-256 before packaging: `b8c08e559f5130df44861b9c899891eea246acd70cf6a38aaecb08f96b9cbb19`.
- It contains 29 changed paths: 2,917 insertions and 402 deletions by Git’s summary.
- Added-line scan found zero `TODO`, `FIXME`, `PLACEHOLDER`, `XXX`, `IMPLEMENT ME`, or `NotImplementedError` implementation placeholders.
- The patch does not contain the supplied archive name, working-tree archive name, bundle, demo packet bytes, secrets, or copied source snapshot.
- Generated artifacts are included as Git binary patches because repository attributes suppress textual diffs for generated files. They were validated by `git apply --index` and by post-apply tests.
