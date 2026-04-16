---
created: "2026-04-16"
purpose: "Proposed carving of 244 branch commits into reviewable PRs for squash-merge to master"
status: "draft"
project: "polylogue"
---

# PR Carving Plan for `feature/chore/repo-cleanup-governance`

Branch is 244 commits ahead of `master`. Last PR-referencing commit on master is
`8a1e1a02` (`#164`). Repo uses squash-merge, so each PR becomes one commit on
master. Goal: carve the 244 commits into ~10 reviewable PRs that tell a coherent
history line on master.

## Strategy

- **Stacked branches, not one branch.** Each PR is a sub-branch cherry-picking
  its arc of commits from the current branch. PRs stack on each other so
  reviewers can check each arc independently.
- **Merges (i.e. commit squash inside the PR) are welcome.** The branch commits
  are iterative working commits; each PR should have a coherent single-subject
  squash-merge title.
- **Order matters.** Earlier PRs land first; later ones rebase on each merge.
- **Do not reopen decisions.** The branch is already internally consistent;
  this is just a republication layer.

## Constraints

- Squash-merge only. PR title = master commit subject.
- Protected checks: `CI`, `Nix`, `Pull Request Policy`.
- Don't split the three recent runtime fixes (`01cc2a12`, `1497d8c8`,
  `92e4b70b`) across PRs — they are a coherent semantic unit.
- Don't split the storage-runtime unification (`8c39814a`, `1677f8ee`) — same
  reason.

## Proposed PR Stack (10 PRs)

| # | Title | Commits | Size | Theme |
|---|---|---|---|---|
| 1 | `chore: consolidate repo governance, docs, and devtools control plane` | ~20 | L | repo surface, ruff normalization, CLAUDE/agent memory, control-plane elevation |
| 2 | `fix: harden CLI error contracts and machine envelopes` | ~25 | L | error boundaries, JSON envelopes, query/stats/open/tags flag ergonomics, completions |
| 3 | `fix: repair live-archive runtime semantics and session products` | ~25 | L | schema v1 baseline reset, session product freshness, repo attribution, FTS readiness alignment |
| 4 | `perf: reduce ingest, materialize, and read-path memory pressure` | ~15 | M | materialize rebuild bounds, single-record inline ingest, worker throttling, session refresh by message volume |
| 5 | `fix: harden ingest/parse robustness and progress observability` | ~12 | M | lone-surrogate tolerance, first-malformed-line reporting, reparse preview/scope, acquisition heartbeats, site progress |
| 6 | `refactor: introduce artifact graph and unify storage bootstrap` | ~4 | S | action-event & raw-ingest artifact semantics, explicit artifact graph, shared schema bootstrap |
| 7 | `refactor: compile showcase and benchmark surfaces from scenarios` | ~60 | XL | scenario substrate wave — metadata roots, operation specs, QA request model, catalog extractions |
| 8 | `refactor: unify scenario execution, corpus, and CLI surface roots` | ~50 | XL | corpus-spec routing, layer collapse, execution root unification, runtime-path modeling |
| 9 | `refactor: consolidate scenario assertion, maintenance targets, and sqlite runtime` | ~14 | M | final surface unification, maintenance target catalog, sqlite schema/db-path/connection-profile collapse |
| 10 | `fix: split raw parse/validation semantics and tighten quarantine accounting` | ~3 | S | parse-vs-validation split, action-event orphan readiness, verifier decode-error preservation |

**Total: 228 commits accounted for.** (~16 misc commits — style/test/docs fixups —
will fold into the nearest arc during republish.)

## PR-by-PR Detail

### PR 1 — `chore: consolidate repo governance, docs, and devtools control plane`

**Arc:** the earliest work on the branch. Governance, docs, devtools promotion.

**Cherry-pick range (chronological):**
```
0d81cb84  fix: repair codex ingestion and workflow contracts
cfbc3da6  chore: harden repo governance and generated docs
a56cbfa8  style: normalize ruff formatting across codebase
5dd259ea  docs: generate quality workflow reference from live registries
4afd9f45  docs: tighten CLAUDE constitution and agent guide
529d775d  chore: unify repo control plane and docs surface
0395482a  fix: fail fast on unknown audit sources
28d79d8f  docs: recast agent memory and simplify repo language
51bee968  docs: remove personal Claude session workflow drift
5cb154b5  test: remove dead config harness and prompt env hook
87e40925  refactor: normalize repo identity and tighten repo control surfaces
207ab5ad  test: remove bogus semantic property xfails
2d02d63e  fix: remove fork and schema warning drift
714c41e9  chore: streamline local tooling surfaces
6a361561  fix: tighten devtools control-plane feedback
bc2df35e  refactor: make devtools a first-class control plane
8d6bdddd  docs: tighten repo guidance and generated surfaces
342e7c72  chore: make devtools consistent across shell and ci
b0bf22ac  chore: route devshell status through stderr
22c66d1b  chore: tighten shell and local-state hygiene
b3c8bf32  chore: tighten devshell status and local output layout
1c1f488a  chore: tighten local state and status surfaces
d9a447d3  fix: restore interactive direnv motd
066b6ede  style: normalize import ordering
```

**Body summary for PR:**
- Promote `devtools` to a first-class repo control plane with consistent CI integration.
- Consolidate governance (PR template, CLAUDE.md/AGENTS.md rendering, docs map).
- Normalize ruff style and import ordering across the repo.
- Remove dead config harness, xfail drift, and personal session workflow docs.

---

### PR 2 — `fix: harden CLI error contracts and machine envelopes`

**Arc:** the first CLI-vetting wave. Error boundaries, JSON contracts, option ordering.

**Cherry-pick range:**
```
c640afa7  fix: harden CLI error boundaries and showcase baselines
40e825fc  fix: preserve grouped and structured stats output
f58df8e3  fix: isolate seeded QA and demo workflows
280c2caf  fix: report real message-role stats
ca261ad4  test: refresh terminal snapshot baselines
85fbe552  fix: stabilize synthetic chatgpt ids
828ee3a5  fix: tighten machine query contracts
203686aa  fix: read checkout git metadata without git on PATH
dfb72712  fix: rebase generated docs map links
94151042  test: fix attachment helper call in stats regression
59966299  fix: return json no-results envelopes for query surfaces
a32642da  fix: stop auto-announcing plain mode
7b44a8f8  fix: make seeded audit proof and showcase clean
359c6971  feat: let open print render paths
ae99ec7f  fix: keep open json no-results machine-readable
8ab020c4  fix: keep embedding status pending counts coherent
a2feb74d  fix: scope schema audit checks by provider
0a8c2154  docs: fix seeded env shell example
f446260e  test: refresh open help baseline
6934f49b  fix: deduplicate schema explain role summaries
31985ef5  test: align vec stats contract mock with embedding status
c5de471a  fix: align schema json output with machine envelope
83d5f1e3  refactor: drop dead doctor cache surface
a9e69b9a  fix: clarify root query option ordering
ada90e67  fix: accept format json on product commands
0170d15a  fix: accept direct conversation ids in open verb
95afd4af  fix: accept post-verb limits for stats
5844538b  feat: add archive-backed shell completions
0a327c15  fix: accept format json for tags
7858c2af  docs: refresh CLI reference and showcase baselines
afa1639d  fix: use valid hook type in project settings
```

**Body summary:**
- Every plain-mode `PolylogueError` now renders as a Click user error, not a raw traceback.
- Machine envelope unified across query/list/open/tags/products/stats surfaces.
- Verb-level flags like `--limit`, `--format json`, and direct conversation IDs now work after the verb consistently.
- Archive-backed zsh completions for IDs, tags, tools, and providers.

---

### PR 3 — `fix: repair live-archive runtime semantics and session products`

**Arc:** runtime/session-product/attribution fixes found during the first live rebuild.

**Cherry-pick range:**
```
2d462c15  fix: reset archive schema baseline to v1
c5688aca  fix: harden live archive reads and overwrites
6e2e8bd2  fix: align live provider ingestion with archive payloads
e6604e8b  test: refresh invalid option terminal snapshot
2f658821  fix: keep archive stats on a single read snapshot
f516a73f  fix: accept live codex review source metadata
4bc1deb7  fix: skip empty source artifacts during acquisition
42b63106  fix: fast-path oversized html message renders
5d7ea210  fix: initialize process-pool worker logging
2c1102da  fix: keep fresh read queries off the writer path
16ecbd8e  fix: ignore unreadable git admin paths
63e55eb3  fix: detect legacy inline-raw archives
cb1d230a  fix: repair grouped stats routing and empty-json contracts
317a36bc  fix: honor runtime-only doctor and root json products
4320db05  fix: tighten repo attribution signals
15a75b6c  fix: clarify archive stats attachment semantics
dd2b64e4  fix: restore search read-connection boundaries
f7868e83  fix: preserve repo attribution through session rebuilds
c886b0d1  fix: clarify action-event maintenance debt accounting
96b2bac2  fix: filter noisy repo attribution signals
eada263a  fix: invalidate stale session product rows
5dc6b17c  fix: ignore transcript stores in repo attribution
c81eab2d  fix: align message fts readiness with indexable rows
db695a44  fix: harden session product freshness and attribution
e63f0c93  fix: surface session product repair progress
42024481  fix: trim repo-local agent config from profile evidence
8148f1b0  fix: filter noisy path attribution in session profiles
```

**Body summary:**
- Schema baseline reset to `v1` with legacy inline-raw detection.
- Session product rows now invalidate on materializer version bump; repair path emits visible progress.
- Repo attribution tightened: no transcript-store paths, no config-backed agent directories, no dialogue-derived repo guessing.
- FTS readiness now compares against indexable rows, not all messages.

---

### PR 4 — `perf: reduce ingest, materialize, and read-path memory pressure`

**Arc:** memory/perf hardening after the live rebuild exposed the heap shape.

**Cherry-pick range:**
```
d58d8332  perf: keep archive stats off retrieval-band status
3d62e20b  perf: reduce sqlite memory pressure on read paths
27aaf31b  perf: cut latest-query memory on large archives
346bda25  perf: reduce ingest and rebuild memory pressure
a336cd68  perf: trim codex ingest heap overlap
e3250649  perf: stream grouped jsonl ingest
f6bc2773  perf: reduce pipeline memory spikes
cf23f044  fix: correct ingest batch memory telemetry
1fbd066e  perf: bound session refresh work by message volume
742c321b  perf: skip redundant action-event rebuilds during indexing
181ebf2c  perf: avoid process pools for single-record ingest batches
24d99412  perf: bound materialize rebuild memory and progress
c8620405  perf: reduce ingest and render memory pressure
11adbffe  perf: make default doctor use cheap health probes
5a1160c3  fix: slim default doctor probes and async read bootstrap
```

**Body summary:**
- Materialize rebuild now bounds pages and reports progress by chunk delta.
- Single-record ingest batches run inline; multi-record batches keep the pool.
- Session refresh chunked by `conversation_stats.message_count`, not by conversation count alone.
- Default `doctor` uses cheap probes (no exact orphan/FTS docsize counts) unless `--deep`/`--repair`.

---

### PR 5 — `fix: harden ingest/parse robustness and progress observability`

**Arc:** parser robustness and long-running UX.

**Cherry-pick range:**
```
e4fdd65a  fix: trim oversized ingest completion logs
86734ac9  fix: compact stage telemetry in run logs
904ca600  fix: stop calling maintenance changes issues
d9fc51a9  fix: label acquisition progress as scanning
14819161  fix: collapse empty-source warnings during acquisition
28b512bd  fix: emit acquisition heartbeats for slow files
3b9c1a2f  fix: refresh action events for changed conversations
cefbf796  fix: surface incomplete search indexes explicitly
781a2b5c  fix: tolerate lone-surrogate jsonl records
a6dacd99  fix: report first malformed jsonl line
179d3deb  fix: make reparse preview side-effect free
35526f1a  fix: scope reparse resets to selected sources
a9114848  fix: reparse validation-failed raws
bdb6b171  fix: bound fresh materialization and report site progress
```

**Body summary:**
- JSONL decode falls back to stdlib for lone-surrogate lines while keeping orjson fast-path.
- Malformed-line reporting now carries first-bad-line detail.
- `run --reparse --preview` is side-effect free; source-scoped resets stay scoped.
- Acquisition heartbeats break the multi-minute silent window on slow files; site build progress replaces `Building site...: 0`.

---

### PR 6 — `refactor: introduce artifact graph and unify storage bootstrap`

**Arc:** foundation for the scenario wave. Small, focused PR.

**Cherry-pick range:**
```
df27f558  docs: deduplicate, enrich, and tighten docs surfaces
ffae1444  refactor: unify action-event artifact semantics
66486937  refactor: unify raw ingest backlog semantics
ac053d5e  refactor: add explicit artifact graph
b670e53c  refactor: unify schema bootstrap semantics
```

**Body summary:**
- First shared semantic substrate: `polylogue/storage/action_event_artifacts.py` and `polylogue/storage/raw_ingest_artifacts.py` replace scattered backlog/repair constants.
- Explicit runtime artifact graph in `polylogue/artifact_graph.py` covers the action-event and raw-validation paths.
- Sync and async schema bootstrap now share one plan model in `polylogue/storage/backends/schema_bootstrap.py`.

---

### PR 7 — `refactor: compile showcase and benchmark surfaces from scenarios`

**Arc:** the first half of the scenario unification wave. Large but mechanically
coherent — each commit just moves authorship up one layer.

**Cherry-pick range (approximately 60 commits):**
```
bf231fe1  refactor: compile generated exercises from scenarios
4b22cf46  refactor: compile benchmark campaigns from scenarios
b5433195  refactor: preserve benchmark scenario metadata
f4cd5d17  refactor: preserve exercise scenario metadata
ebda54c0  refactor: unify scenario metadata roots
de1628f9  refactor: preserve benchmark registry metadata
aef67ef6  refactor: model artifact graph operations
2981f17c  refactor: extract runtime operation specs
2458e923  refactor: compile synthetic benchmark campaigns
6da8cd40  refactor: register synthetic benchmark scenarios
5f8fa23e  refactor: extract runtime artifact specs
2863d21f  refactor: add runtime-targeted maintenance scenario
731fa001  refactor: render runtime scenario coverage
2f4756f2  refactor: centralize runtime target resolution
b98c15b8  refactor: expose artifact graph in devtools
3647c5ba  refactor: report uncovered runtime coverage
e11c9400  refactor: complete authored runtime path coverage
01b4bb36  refactor: map session product repair semantics
4bede02f  refactor: share runtime scenario coverage
8d68bee2  refactor: resolve scenario runtime targets
19d0ac90  refactor: register scenario-bearing projections
ac3352fb  refactor: map runtime path coverage
3addd9db  feat: expose scenario projection inventory
70865804  feat: filter scenario projection inventory
1f2804b1  refactor: model qa session requests
9ca65254  refactor: derive qa requests from cli options
279f4d14  refactor: type scenario projection sources
ba13067c  refactor: type qa stage selection
e4c1a289  docs: catalog scenario-bearing projections
2070e6fa  refactor: resolve runtime targets through artifact graph
4b6c2dac  refactor: add runtime path targets to scenario metadata
3acf3cbe  refactor: type qa capture and snapshot intent
1ba36eff  refactor: execute qa snapshot plans
15319e04  refactor: type qa finalization flow
affe20db  refactor: type qa invocation plans
7d01d0d6  refactor: extract scenario projection types
b1b73f27  refactor: extract scenario projection catalog
aa567005  refactor: decouple scenario coverage from quality registry
3a970206  refactor: extract benchmark catalog types
06408afa  refactor: extract validation and mutation catalogs
29e16b3c  refactor: declare operation targets
c2d5615f  refactor: track declared operation coverage
b496976e  refactor: extract synthetic benchmark catalog
f4676efe  refactor: extract durable benchmark catalog
365d396f  refactor: extract mutation campaign catalog
c7fb9598  refactor: make exercise scenarios the showcase root
6c2040a4  refactor: make generated showcase cases scenario-first
e42ad9d7  refactor: centralize qa extra scenarios
7da91485  test: cover action-event repair benchmark operation
f01b4175  refactor: introduce operation catalogs
03f1a983  refactor: project validation lanes as scenarios
cd45bbd6  refactor: project mutation campaigns as scenarios
9865a098  refactor: preserve durable campaign metadata
0ddebe46  refactor: preserve synthetic benchmark metadata
b8f81c7e  refactor: centralize qa extra scenarios
7a8c2252  refactor: project benchmarks from catalog entries
06b4cbe8  refactor: route campaign runners through catalogs
dd90e823  refactor: route synthetic benchmarks through catalogs
3005ed43  refactor: share registry projection inputs
de049319  refactor: author operation path targets
```

**Body summary:**
- Introduce `ScenarioMetadata` as the shared vocabulary for showcase exercises, benchmark campaigns, mutation campaigns, and validation lanes.
- Extract separate benchmark/mutation/validation catalogs and decouple them from `QualityRegistry`.
- Introduce `OperationSpec` and `RUNTIME_OPERATION_SPECS` as the authored runtime operation substrate.
- Compile generated showcase exercises, benchmark campaigns, and synthetic benchmark campaigns from the same scenario root.
- Model QA session requests and plans as typed objects rather than loose CLI argument bags.
- **Split option:** if 60 commits is too large, split at `7a8c2252` (Arc 7a: substrate extraction, ~35 commits) and Arc 7b: catalog projection, ~25 commits).

---

### PR 8 — `refactor: unify scenario execution, corpus, and CLI surface roots`

**Arc:** second half of the scenario wave. Execution root and corpus-spec routing.

**Cherry-pick range (approximately 50 commits):**
```
3b485ffe  refactor: route synthetic workflows through corpus specs
af1ff80f  refactor: route synthetic test fixtures through corpus specs
45ad0edb  refactor: compile large archive specs into corpus specs
5c373651  refactor: route test fixtures through corpus specs
f2cb3acf  refactor: author synthetic benchmark dispatch metadata
2aa0902b  refactor: route synthetic helper tests through corpus specs
27f33c81  refactor: route parser schema tests through corpus specs
0636f794  refactor: share execution specs across validation and benchmark scenarios
cf50b9aa  refactor: project inferred schema corpora
5be03e64  refactor: persist inferred schema corpus specs in schema list
6efd382f  refactor: preserve scenario projection payloads
a1de4a2c  refactor: share multi-spec synthetic fixture writing
c347bf85  refactor: preserve execution in compiled verification entries
cda52da4  refactor: execute inferred corpus specs directly
56ffeebc  refactor: reuse shared lane config for scale lanes
f333289d  refactor: collapse durable benchmark scenario layers
e6daef87  refactor: collapse mutation campaign layers
9e41e16f  refactor: author benchmark entries directly
88d0590b  refactor: compile inferred corpus projections from specs
e71b1b4e  refactor: route synthetic benchmark cli through entries
9fb41402  refactor: unify lane entry models
00d24556  refactor: let scenario sources project themselves
55c3898c  refactor: unify synthetic workspace seeding
0e55704a  refactor: fold scale lanes into validation catalog
4d06ff7e  refactor: compile inferred corpus scenarios
283b4bdf  refactor: unify authored executable scenarios
6c776f6a  refactor: centralize authored scenario catalogs
d1b09f33  refactor: route harness seeding through corpus scenarios
8b71f922  refactor: compile large archive specs through scenarios
59481e80  refactor: model message fts runtime loop
f0ee361a  refactor: canonicalize pytest execution specs
73515159  refactor: move scenario execution into shared substrate
96c0e8bb  refactor: centralize polylogue cli execution semantics
5ea11068  refactor: unify showcase execution roots
b9293a55  refactor: serialize showcase execution payloads
aea56c33  refactor: honor authored showcase corpus specs
be0ae1e3  refactor: expand product runtime graph coverage
39c7a978  refactor: infer scenario targets from structured executions
13994f6b  refactor: collapse showcase exercise model layers
47517b0f  refactor: model archive ingest runtime path
2c9b3fc3  refactor: model validation families and schema scenario paths
035c98e9  refactor: model publication runtime path
eb86d835  refactor: model embeddings runtime path
8b9be3e5  refactor: model source acquisition runtime path
843b9e03  docs: refresh quality reference for acquisition path
fe3fe138  refactor: route runtime registries through authored catalog
22319b6d  refactor: thread corpus sources through synthetic harnesses
713b946c  refactor: route synthetic corpus selection through requests
9644ed84  refactor: model showcase seeding with corpus requests
8a7ef648  refactor: type pipeline probe execution
4914bec3  refactor: author memory budget lanes from executions
```

**Body summary:**
- `CorpusSpec` becomes the single input surface for synthetic harness seeding; all synthetic workflows, test fixtures, helpers, and parser schema tests route through it.
- `ExecutionSpec` unifies subprocess/pytest/cli execution under one substrate.
- Showcase exercises collapse to a single authored layer (`Exercise` extends `ExecutableScenario`).
- Runtime artifact paths now model archive ingest, publication, embeddings, and source acquisition declaratively.
- Memory-budget lanes now compile from shared execution specs, not hand-written command blocks.
- **Split option:** if this is too large, split at `73515159` (Arc 8a: corpus-spec routing, ~30 commits) and (Arc 8b: execution root collapse, ~20 commits).

---

### PR 9 — `refactor: consolidate scenario assertion, maintenance targets, and sqlite runtime`

**Arc:** final unification layer. Smaller, each refactor is a cleaner cap.

**Cherry-pick range:**
```
c4548b21  refactor: derive synthetic benchmark runners from campaigns
ad064f8d  refactor: author showcase generators directly from exercises
f11eaec9  refactor: centralize scenario execution runtime
22385d1d  refactor: unify synthetic runner execution runtime
538c7384  refactor: move execution metadata into execution specs
e6f0d046  refactor: cache the authored scenario catalog
036d23fc  refactor: unify scenario assertion semantics
43d42819  refactor: compile validation lane families from stages
02c04ce3  refactor: extract corpus profiles from corpus specs
3b29ed3d  refactor: centralize maintenance target semantics
cd8443f9  refactor: compile product query surfaces from families
337616ee  refactor: compile cli surface families and richer corpus profiles
8c39814a  refactor: unify sqlite schema and db path runtime
1677f8ee  refactor: share sqlite connection profiles
```

**Body summary:**
- `AssertionSpec` is now the one outcome vocabulary across showcase, validation lanes, and benchmarks.
- `ValidationLaneFamily` compiles family stages instead of hand-authoring each lane.
- `CorpusProfile` extracted from `CorpusSpec`; richer profile metadata flows through `operator_inference.py`.
- `MaintenanceTargetSpec` catalog owns repair target identity, category, destructiveness, and doctor-operation mapping for every consumer.
- `CliSurfaceFamily` compiles product-query and operational CLI surfaces from shared builders.
- Sync and async SQLite paths now share the same schema/DB-path runtime and connection profiles.

---

### PR 10 — `fix: split raw parse/validation semantics and tighten quarantine accounting`

**Arc:** the three most recent runtime fixes. Small, cleanly coherent.

**Cherry-pick range:**
```
01cc2a12  fix: separate parse failures from validation failures
1497d8c8  fix: count orphan action-event rows in repair readiness
92e4b70b  fix: preserve decode error detail in schema quarantine
```

**Body summary:**
- Strict schema-invalid raws no longer collapse into `parse_error` / quarantine; only real decode/parse/transform failures do.
- Action-event orphan rows now participate in repair readiness and debt accounting.
- Schema verification quarantine preserves the real decode error text instead of flattening to exception class names.

---

## Execution Recipe

For each PR in order:

```bash
# Start from master
git fetch origin
git checkout -b pr/01-governance origin/master

# Cherry-pick the arc
git cherry-pick 0d81cb84 cfbc3da6 a56cbfa8 ...

# Resolve conflicts inline; the arc is internally consistent so
# conflicts only come from out-of-order cherry-picks — reorder if needed.

# Push and open PR
git push -u origin pr/01-governance
gh pr create --title "chore: consolidate repo governance, docs, and devtools control plane" \
  --body-file .claude/scratch/pr-bodies/01-governance.md

# After PR 1 merges to master:
git fetch origin
git checkout -b pr/02-cli-contracts origin/master
git cherry-pick <pr-2 range>
# ...and so on
```

Alternative: stacked branches using `git range-diff` to keep PRs dependent without
waiting for earlier merges. This is worth the extra mechanics because each PR is
non-trivial and sequential merging would create a long wall-clock tail.

## Risks and Notes

1. **Cherry-pick conflicts.** The branch is linear, so conflicts only arise if a
   later commit touches a file that an earlier arc-mate needed to create first.
   The arcs above are chosen to minimize this; any remaining conflicts should be
   mechanical.
2. **CI runtime.** Each PR will run the full `pytest -q --ignore=tests/integration`
   and `nix flake check`. Stacked PRs multiply CI cost; consider turning off
   non-blocking CI for intermediate stacked PRs.
3. **Test pass rate drift.** Some earlier commits might have test failures that
   later commits fix. Verify each PR's final state with:
   ```bash
   pytest -q --ignore=tests/integration
   devtools render-all --check
   ```
4. **PR 7 and PR 8 are both ~50-60 commits.** If a reviewer pushes back on size,
   both have explicit split points noted above (`7a/7b`, `8a/8b`), taking the
   total to 12 PRs instead of 10.
5. **The three recent runtime fixes (PR 10) could merge first.** They are fully
   independent of the scenario wave and the cleanest unit on the branch. Merging
   them first gives fast feedback that the PR process works before tackling the
   larger arcs.

## Size Estimate

| PR | Commits | Files touched (est.) | LOC delta (est.) | Review difficulty |
|---|---|---|---|---|
| 1 | 24 | 60 | +500 / −800 | M — surface cleanup |
| 2 | 31 | 45 | +1500 / −600 | M — CLI contracts |
| 3 | 27 | 55 | +1800 / −800 | H — runtime semantics |
| 4 | 15 | 30 | +900 / −700 | M — memory work |
| 5 | 14 | 35 | +1000 / −500 | M — parser/UX |
| 6 | 5 | 15 | +1200 / −300 | M — new substrate |
| 7 | 60 | 120 | +4000 / −2500 | H — largest arc |
| 8 | 50 | 100 | +3500 / −2200 | H — execution root |
| 9 | 14 | 50 | +1500 / −900 | M — final caps |
| 10 | 3 | 12 | +250 / −150 | L — bug fix trio |

(LOC estimates are guesses; run `git diff --shortstat` per arc to pin down.)

## Alternative: 3 Mega-PRs

If the reviewer prefers fewer, larger PRs:

- **Mega-1 (PRs 1–5 merged):** `chore: consolidate repo surface and runtime vetting` — the "cleanup before substrate" story.
- **Mega-2 (PRs 6–9 merged):** `refactor: introduce shared scenario substrate and collapse parallel vocabularies` — the "unification wave" story.
- **Mega-3 (PR 10):** `fix: tighten raw parse and quarantine semantics` — the trailing runtime fixes.

3 master commits. Simpler history, harder review. Trade-off is up to the reviewer
pool.
