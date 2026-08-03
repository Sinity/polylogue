# Polylogue

Polylogue is a **local, single-writer archive for AI coding/chat sessions** —
Claude (web + Code), ChatGPT, Codex, Gemini/Drive, Antigravity, Hermes — that
ingests heterogeneous exports into a split SQLite file set, derives rich read
models, and serves them through a query-first CLI, an MCP server, a Python API,
and an HTTP daemon. Pure Python, no native deps beyond pre-built wheels.

This file is **standalone**: it carries the working understanding you need to
be effective here. For depth, read the referenced docs on demand (see
[Reference docs](#reference-docs)) — they are not auto-loaded.

---

## Orientation

The system has four rings; substrate owns meaning, surfaces are leaf adapters:

```
sources/ ─detect→ pipeline/ ─hash+write→ storage/{5 tiers} ─materialize→ insights/
                                              │                              │
                            surfaces: cli/  mcp/  api/  daemon/  ─read-through─┘
                            verification:   devtools/  tests/  schemas/
```

Package sizes (rough): `storage/` (largest), `daemon/`, `cli/`, `archive/`,
`sources/`, `schemas/`, `insights/`. Entry points:

| File | Role |
| --- | --- |
| `polylogue/api/__init__.py` | Async library facade (`Polylogue`) — deliberately thin |
| `polylogue/config.py` | 5-layer config resolution + inventory-driven diagnostics |
| `polylogue/cli/click_app.py` | Root query-first CLI dispatch |
| `polylogue/operations/specs.py` | High-level archive operations (`operations/archive.py` is just `ArchiveStats`; the real operation logic lives here + `import_operations.py` + contracts) |
| `polylogue/daemon/cli.py` | Daemon runner (`polylogued run`) |

**Working rule:** new semantics go into the substrate (`storage`/`insights`) or
product layer first, then surfaces adapt. New surface code should not import
substrate (`storage`/`pipeline`/`sources`) internals directly — route through
`insights`/`operations`/`api` instead. `docs/plans/layering.yaml` enforces
this as a **ratchet, not a clean boundary**: `cli`/`mcp`/`api`/`daemon` each
carry a large pre-existing baseline of direct substrate imports
(`docs/plans/layering-surface-baseline.json`, 311 entries as of
polylogue-2ciy) that `devtools verify layering` exempts, but any import not
already in that baseline fails the (required, CI-gated) check. The genuinely
clean, zero-exception direction is the reverse one: `storage`/`pipeline`/
`sources` must not import surface adapters, enforced with no baseline at all.

---

## Architecture

### The data model (know this cold)

Identity is **computed, never stored redundantly** — every id is a SQLite
generated column:

- `sessions.session_id = origin || ':' || native_id`
- `messages.message_id = session_id || ':' || COALESCE(native_id, position||'.'||variant_index)`
- `blocks.block_id = message_id || ':' || position`

Three-level content tree **sessions → messages → blocks**, all `STRICT` tables.
Load-bearing columns:

- **`messages.material_origin`** (`core/enums.py`) — the authoredness axis
  `Role` can't express (`human_authored`, `assistant_authored`,
  `operator_command`, `runtime_protocol`, `runtime_context`, `tool_result`,
  `generated_context_pack`, …). This is what makes honest cost/user-word
  accounting possible: Claude Code `role=user` protocol rows are excluded from
  authored-user counts.
- **`blocks.tool_result_is_error` / `tool_result_exit_code`** (index v16
  keystone) — provider-reported outcomes read from structure; `NULL` = unknown,
  never regex-guessed from prose.
- **`actions` is a VIEW**, not a table — it left-joins `tool_use ↔ tool_result`
  blocks by `tool_id`. The queryable "action" relation is derived on read.
- FTS5 is **contentless** (`content=''`, `contentless_delete=1`) over
  `blocks.search_text`, kept in sync by three triggers. `tokenize=unicode61`
  (no porter stemmer in this build — don't change it).
- CHECK constraints are **generated from Python types** where a call site
  exists — `check`/`nullable_check` (`storage/sqlite/archive_tiers/common.py`)
  embed a `PolylogueStrEnum`'s values via `sql_check_in`/`nullable_sql_check_in`
  (e.g. `check("origin", Origin)`, `check("role", Role)`), and `literal_check`
  does the same for `typing.Literal` columns (e.g.
  `delegation_facts.mapping_state`/`.result_status` via
  `literal_check("mapping_state", *get_args(DelegationMappingState))`).
  This is real for the ~20 enum-backed columns and the handful of
  `literal_check` call sites wired so far (polylogue-u6tl) — most
  hand-written `CHECK(col IN (...))` lists across `archive_tiers/*.py`
  still have no generator tie and can drift silently; `RunStatus`
  (`insights/run_projection.py`) is intentionally storage-free (an
  in-memory projection, never a column) and has no SQL surface to generate.

**Lineage normalization** (`session_links`, index v12+) is the sharpest design
point. Forks/resumes/subagents/auto-compaction physically replay the parent's
prefix, so the writer stores only the child's **divergent tail** plus
`branch_point_message_id` + `inheritance` (`prefix-sharing` | `spawned-fresh`);
reads recompose parent-up-to-branch + child-tail. `branch_point_message_id` is
**deliberately not a FK** — an `ON DELETE SET NULL` would null it during a
parent full-replace's DELETE step (cascade fires before re-INSERT) and
permanently break composition. `session_links` is also the topology-edge table
(the docs' older `topology_edges` name): it persists every parent reference a
parser asserts, even when the parent isn't ingested yet, keyed
`(src_session_id, dst_origin, dst_native_id, link_type)`, resolved on each save
by `_resolve_session_graph`/`_resolve_outbound_session_links`
(`storage/sqlite/archive_tiers/write.py`) — the sole production
implementation, invoked unconditionally from `write_parsed_session_to_archive`
(the single choke point both live incremental ingest and full raw
replay/reindex go through). `storage/sqlite/queries/session_links.py`'s
similarly-named async `resolve_session_links_for_session` has no production
caller; it exists only as test infrastructure (`polylogue-enium`).
`TopologyEdgeStatus` = unresolved/resolved/repaired/**quarantined**
(cycle-break).

### The five tiers (durability is the axis)

| Tier | durability | holds |
| --- | --- | --- |
| `source.db` | durable | raw acquired bytes (`raw_sessions`), artifact taxonomy, blob/GC substrate (`blob_refs`, `gc_generations`), hook events, sidecars |
| `index.db` | **rebuildable** | the whole parsed tree, FTS, `session_links`, cost tables, and all materialized insights |
| `embeddings.db` | rebuildable | `vec0` virtual table (Voyage 1024-dim), meta, status |
| `user.db` | **durable, irreplaceable** | unified `assertions`, settings/context receipts, immutable annotation schemas + batch provenance |
| `ops.db` | disposable | ingest cursors, attempts, `convergence_debt`, cursor-lag samples, daemon events, embed catch-up runs, otlp |

`user.db` is a **single unified `assertions` table** keyed by a closed
`AssertionKind` (mark / tag / correction / annotation / suppression / metadata /
saved_query / recall_pack / workspace_note / note / decision / caveat / lesson /
blocker / handoff / judgment / pathology / …). It collapsed the old separate
overlay tables; `context_policy_json` (default `{"inject":false}`) gates whether
an assertion is injected into agent context. The column is plain `TEXT` so the
vocabulary can grow without a user-tier schema bump. User corrections are
`AssertionKind.CORRECTION` rows here (a legacy `user_corrections` table survives
only as a compat read path for pre-split single-file archives). Versioned
annotation construct definitions live in `annotation_schemas`; independent
label-run provenance lives in `annotation_batches`, while the labels remain
ordinary assertion rows scoped by `annotation-batch:<id>` ObjectRefs.

### Content-hash idempotency

Archive writes are idempotent by content hash (`pipeline/ids.py`,
`core/hashing.py`): SHA-256 over an **NFC-normalized** payload with
None/empty/missing sentinels, hashing title + timestamps + messages + blocks +
attachments (sorted) + session events. It **excludes** user metadata by
construction — tagging/annotating never triggers re-import. Re-ingest with a
matching hash is skipped; a differing hash updates the session and rebuilds
dependent insights.

### Provider detection & parsing

`sources/dispatch.py:detect_provider()` is shape-based, in **tightness order**
(not filename order): structural/document detectors first (browser-capture,
gemini-cli, hermes, antigravity), then Pydantic-validated record checks (Codex,
Claude Code), then loose dict-key checks (ChatGPT, Claude web, Gemini). Insert a
new detector at the tightness level it deserves or an earlier parser claims its
records. `_lower_payload_specs` then recursively lowers a payload into typed
`LoweredPayloadSpec`s (handling bundles, grouped JSONL split by `sessionId`,
drive-like nesting, single-document providers), and `_parse_lowered_spec` routes
each to a concrete parser. A memory-bounded streaming path exists for multi-GiB
Claude Code JSONL.

---

## Runtime

The **daemon owns all writes** (`polylogued run`). Ingest stages:
**acquire → parse → materialize → index** (`reprocess` = parse+materialize+index
without re-acquire; `all` = full). Raw acquire/parse/materialize lives in
`pipeline/services/ingest_batch/`.

The **`DaemonConverger`** (`daemon/convergence.py`) drives *derived-model*
convergence (FTS repair, embedding catch-up, insights) after ingest. Each
`ConvergenceStage` has check/execute, plus optional batch (`check_many`/
`execute_many`) and session-scoped (`check_sessions`/`execute_sessions`, for
retrying `convergence_debt` without re-resolving source paths) variants. Two
deliberate tricks:

- `false_means_pending` — a stage does bounded work and returns `False` to push
  the *remaining* backlog into `convergence_debt` as retry-able, not a failure
  ("insights deferred until quiet").
- Hot-file quiet deferral (`convergence_stages.py`) batches still-appending
  Codex/Claude sessions until a quiet window; embed runs in bounded windows.

The main process is the **sole SQLite writer** — no convergence stage runs in
a worker process. Blob GC uses two independent safety invariants (leases +
snapshot reference check) to bridge the acquire-blob → commit-row window.

### Schema regimes (durability-keyed)

Two evolution regimes, enforced by `devtools lab policy schema-versioning`:

- **Durable tiers** (`source.db`, `user.db`): explicit **additive** numbered SQL
  migrations under `storage/sqlite/migrations/{source,user}/NNN_*.sql`, one
  `PRAGMA user_version` step at a time, behind a **verified backup manifest**.
  Destructive durable changes need a copy-forward design + explicit consent.
- **Derived tiers** (`index.db`, `embeddings.db`): no migration *chain*, but not
  "always rebuild" either. Every index bump above the compatibility floor
  declares a delta class in `storage/sqlite/lifecycle.py`; a declared
  non-semantic delta upgrades an existing generation **in place** through
  `index_fast_forward_plan()` on connect. Only a `SEMANTIC_REPARSE` delta — one
  whose result depends on parser semantics — routes to
  `polylogue ops reset --index && polylogued run`. A bump without a declaration
  is a policy violation, not a free rebuild: the lint fails and the archive
  silently falls back to full raw replay.

Before editing schema, classify the change: metadata-only, index-only,
additive-derived, additive-durable, or semantic-reparse-required. Batch
same-tier bumps from ready Beads before triggering a live rebuild; don't
repeatedly reset+reingest the active archive for isolated index additions.

**If you add any module/file under `polylogue/`**: regenerate the topology
projection or `render all --check` fails — run
`devtools render topology-projection` and commit the updated
`docs/plans/topology-target.yaml`.

---

## Surfaces

- **CLI is query-first** (`cli/click_app.py`): `find QUERY then ACTION`. Verbs:
  `find` / `read` / `analyze` / `mark` / `select` / `delete` / `continue`
  (+ `read --view transcript|messages|…`). Root filters go **before** `find`,
  verb options **after** the action. Use `--origin` (not `-p`/`--provider`),
  `read --all` (there is no `list`/`show`/`stats` verb).
  - **Strict command floor (#1842):** query mode needs *signalled intent* — the
    `find` keyword, a **quoted** expression (single argv token with internal
    whitespace), or **field syntax** (`repo:x`, `since:7d`). A bare *unquoted*
    plain word (`polylogue foo`) raises a `UsageError` with a did-you-mean/`find`
    hint; it does **not** silently search.
  - The query grammar (`archive/query/expression.py`) is a real Lark DSL:
    fielded predicates, booleans, `near:"…"`, count/date ranges, `with <units>`
    projection, and pipeline stages (`sessions where … | group by … | count`)
    over unit sources sessions/actions/messages/observed-events, lowered to SQL.
- **Python API** (`api/__init__.py`): the `Polylogue` facade is deliberately
  thin — it holds config/services and exposes `repository`/`backend`. The rich
  verbs live on the mixin-composed `SessionRepository`
  (`storage/repository/__init__.py`: archive reads, archive writes, raw,
  vectors, + six insight readers — profile, run-projection, timeline, thread,
  summary, topology) and on `services.py`.
- **MCP** (`mcp/`): the large agent-facing surface — consolidated into 10
  role-gated operation-dispatcher tools (`status`, `read`, `get`, `query`,
  `explain`, `context`, plus `write`/`run` behind the write role, `judge`
  behind the review role, `maintenance` behind the admin role), pinned by
  `tests/unit/mcp/test_server_surfaces.py` against `EXPECTED_TOOL_NAMES` —
  search/list/get, insights, corrections, context/recall, postmortem
  bundles. This, not the API, is the continuity surface. Adding an
  operation requires updating the dispatcher's declared verb table + a tool
  contract.
- **Insights** (`insights/registry.py`): descriptor-driven — one
  `INSIGHT_REGISTRY` where each `InsightType` declares field accessors + a
  Pydantic query model + operations method + CLI/MCP metadata, driving
  plaintext, JSON, and MCP from one place. Insight models carry canonical
  `origin` fields directly; surfaces serialize them without a vocabulary shim.
- **`SessionFilter`** (`archive/filter/filters.py`) is a fluent shell over an
  immutable `SessionQueryPlan` that separates SQL-pushdown from post-filters and
  summaries-from-full loading, plus the `with_units` projection.

---

## Vocabulary: Provider vs Origin vs Source

Three origin-related vocabularies with different scopes (`core/enums.py`,
`core/sources.py`; full table in `docs/provider-origin-identity.md`):

- **`Origin`** — the public source-origin token on query surfaces and read
  payloads: `claude-code-session`, `claude-ai-export`, `chatgpt-export`,
  `codex-session`, `gemini-cli-session`, `hermes-session`,
  `antigravity-session`, `aistudio-drive`, `grok-export`, `unknown-export`.
  **Public filters use `origin`.**
- **`Provider`** — the older provider-wire token (mixes lab/product/source-family
  identity). Still legitimate at wire boundaries (raw export parsing,
  `schemas/providers/`, provider/embedding-provider metadata) but a **leak** on
  public surfaces.
- **`Source`** — richer identity (`family`, `runtime_root`, `originating_lab`).

Normalized archive identity carries `Origin`; `source_name` in rebuildable
storage rows is a persistence detail converted while hydrating typed models,
never a second public identity vocabulary.

`Provider` deliberately remains at raw acquisition/parser/schema boundaries
and in genuinely provider-scoped concepts such as embedding backends, pricing
catalog vendors, provider-reported cost, and provider-usage billing. The
`GEMINI` + `DRIVE` → `AISTUDIO_DRIVE` mapping is non-injective, so normalized
code must not reverse an `Origin` into a guessed `Provider`; raw-wire code uses
its original acquisition evidence. Anti-goal: provider wording on
source-origin public filters or payloads.

---

## Working Rules (agent workflow)

These override default agent behavior.

### Beads issue tracking

This repo uses `bd` (Beads) for durable task state AND as the devloop: `bd
prime` -> `bd ready` -> claim -> work -> PR -> close with reasons. The former
bespoke conductor packet is retired — do not resurrect it or `devloop-*`
scripts under any name. (Its evidence may still sit in a gitignored,
untracked `.agent/archive/devloop-2026-07/` in some working checkouts —
polylogue-ocby — it is not part of the repo and a fresh clone will not have
it.) Repo agent conventions: `.agent/CONVENTIONS.md`; run
`devtools lab policy bead-graph` before shipping bead-state deltas. Run `bd prime` when task
context, ready work, blockers, or project memory matter. Use `bd ready --json`,
`bd show <id> --json`, `bd update <id> --claim --json`,
`bd close <id> --reason "…" --json`. Create linked Beads issues for discovered
follow-up work rather than leaving markdown TODOs as the source of truth.
`bd dolt push` follows the same policy as `git push` (feature branches / PR
updates after verification; no direct push to protected default).

**bd hazards are documented in the global agent instructions** (branch-switch
reimport, `bd export`'s cwd-independent output path, conflict-marker recovery).
They apply here unchanged; that text is not duplicated in this file. The one
project-specific consequence worth repeating: fold a `bd claim`/`close` into the
same branch as the code change it accompanies rather than opening a sibling
`chore(beads):` branch, because this repo's PR cadence makes divergent
bookkeeping branches the common failure.

### Issue-first for non-trivial work

Open an issue before work that is non-trivial, spans multiple PRs, or introduces
architectural decisions — it defines scope and acceptance criteria. Reference it
from the PR with neutral wording (`Ref #NNN`). Skip issues for self-contained
fixes where the PR body suffices.

**Do not use GitHub resolver keywords** (the close/fix/resolve family) next to
issue numbers in agent-authored PR bodies/comments/commits unless the operator
explicitly asks for that exact PR to change that exact issue's state. Use
`Ref #N` + explicit `Remaining #N scope:` instead.

### Verification — testmon inner loop, never blanket-run

The default path is `devtools verify`: static/generated gates +
**pytest-testmon affected-selection** (only tests whose dependency graph touches
your change; seconds-to-minutes). For a single target: `devtools test <file>` or
`devtools test -k <expr>`.

**Anti-pattern (do NOT):** `devtools test tests/unit/<dir>` over whole
directories, or blanket `pytest tests/unit`. Running broad directories is
effectively the full suite (>1h) and re-confirms tests your change never
touched. A mypy-green, behavior-preserving refactor needs only its
testmon-affected set.

- `mypy --strict` (via `devtools verify`) is the primary net for type/identifier
  refactors — trust it. Config in `pyproject.toml`, no exclude list.
- Seed testmon on a fresh checkout / after harness or dependency changes:
  `devtools verify --seed-testmon --skip-slow`.
- Reserve `devtools verify --all` (full non-integration run) for
  harness/dependency changes or a final pre-PR diagnostic.
- `devtools verify --quick` = format + lint + mypy + `render all --check`
  (no tests); it runs on `git push` via the pre-push hook. It is a fast gate,
  not a substitute for the default baseline before a PR.
- If failures land in files your change didn't touch and testmon didn't select,
  classify as pre-existing/flaky (re-run the exact node) before assuming yours.

Don't treat CI as the first verification pass — anticipate failures locally.

### Schema-touching changes

See [Schema regimes](#schema-regimes-durability-keyed). Durable tiers → numbered
additive migration + backup manifest; derived tiers → edit canonical DDL +
rebuild plan (`polylogue ops reset --index && polylogued run`), never an upgrade
helper (`devtools lab policy schema-versioning` rejects them).

### Multi-lane / merge-train tooling — use these, don't reinvent the discipline by hand

Discoverable-only tooling is inert tooling: a command that exists but isn't
named here is invisible to the next session unless it happens to remember it
from transcript, so it doesn't get used and the failure mode it exists to
prevent recurs. These four are load-bearing steps in the fanout/merge-train
workflow, not optional conveniences — use them at the point named, every time:

- **When provisioning any lane worktree (spawn time, before dispatch)**:
  `devtools workspace lane-init <path> --branch <branch> [--beads ids]` —
  creates worktree+branch, provisions the lane's OWN venv (`uv sync
  --extra dev-common --extra speed`; a shared-venv worktree cannot run
  devtools/pytest at all), guard-verifies imports resolve inside the lane,
  runs verify-worktree, registers the lane in `.cache/fanout/lanes.jsonl`,
  and prints the per-lane `POLYLOGUE_PYTEST_WORKERS` budget for the planned
  concurrency. At 16 lanes this is the difference between lanes that verify
  and lanes that idle on guard refusals.
- **Before claiming a batch of ready beads**: `devtools workspace bead-cluster`
  — footprint/overlap/contention clustering so overlapping-file beads land on
  one branch instead of colliding across parallel lanes.
- **When dispatching a worktree-isolated lane**: `devtools workspace lane-brief
  <ids> --out <path>` for its dispatch prompt (footprint, prior art, hazards).
- **Immediately after spawning a worktree-isolated lane, not after it reports
  back**: `devtools workspace verify-worktree <path> --expect-branch
  <branch>` — confirms the worktree is real and isolated before the lane has
  had a chance to run anything. Waiting until the lane reports is too late:
  the 2026-08-01 incident this check exists for was a silent worktree-escape
  where ~1700 lines of half-finished output had already landed directly in
  the coordinator's live tree by the time the lane reported back.
- **Before squash-merging any PR**: `devtools workspace merge-gate record <PR>
  --command "devtools verify"` (or a narrower test selection) against the
  PR's current head, then `devtools workspace merge-gate check <PR>` —
  BLOCKs unless a fresh receipt exists for the exact head sha and no review
  comment (inline, issue-level, or review-summary) is newer than the head
  commit's timestamp unless explicitly `ack`'d. This replaces "remember to
  grace-period-poll and remember to run the broader suite CI skips" with one
  command; it is not automatic (a coordinator still has to remember to run
  it), so treat it as a required step in the merge checklist below, not an
  optional nicety.
- Sizing/triage input: `devtools workspace backlog-calibration` for
  lead-time/discovery/staleness distributions before deciding batch size.

If you build a new tool in this family, add it here in the same sentence —
a tool without a line in this file is a tool the next session won't use.

### Commit / PR discipline

All product code lands via **feature branches + squash-merged PRs** to `master`
(protected; no direct pushes). Branch names: `feature/<category>/<desc>`.

- Conventional commit subjects (`feat:`/`fix:`/`refactor:`/`perf:`/`test:`/
  `docs:`/`chore:`). The **PR title is the squash-merge subject** on `master` —
  ≤72 chars, imperative, describes what changed. Ends up as permanent history.
- PR body sections (all required): **Summary**, **Problem** (evidence, not "user
  asked"), **Solution** (modules touched, non-obvious decisions), **Verification**
  (exact commands + the output line that matters, not "tests pass").
- Routine PRs do **not** edit `pyproject.toml` `version` or `CHANGELOG.md` —
  release-please owns those from conventional subjects on `master`.
- **Claim verification:** before writing that something is "unified"/"aligned"/
  "converged"/"complete", grep the diff and check both paths. A claim the code
  doesn't support is worse than no claim. State partial work honestly.
- **Acceptance-criteria honesty:** address each AC as satisfied / deferred (to a
  named follow-up issue) / misframed. Tests are not a substitute for missing
  runtime wiring.
- Stage by path (`git add <file>`), never `git add -A` / `-a` on significant
  changes. Never `--no-verify` unless the operator asked; a hook failure means
  fix the root cause in a **new** commit (don't `--amend` a successful one).

Issues and PR bodies are durable artifacts — write them to stand alone for a
reader with no conversation context (file paths, AC, design references).

---

## Testing essentials

Full detail in `TESTING.md`. Layout: `tests/unit` (~95%), `tests/property`
(Hypothesis), `tests/integration` (slow, protected), `tests/benchmarks`,
`tests/fuzz`. Shared infra in `tests/infra/` (`SessionBuilder`, `make_message`,
`corpus_seeded_db`, schema-driven strategies). `workspace_env` fixture gives
isolated XDG paths + archive root.

- **Prefer `devtools test`** over raw pytest — it runs through the managed
  harness (repo env, single-process by default, live output, stall/runtime
  timeouts, serialized overlapping runs). `POLYLOGUE_PYTEST_WORKERS=N` overrides.
- **Clock hygiene:** timestamp-sensitive tests use the `frozen_clock` fixture
  (`tests/infra/frozen_clock.py`), not the host wall clock. An autouse guard
  (`tests/infra/clock_guard.py`) makes direct `datetime.now`/`time.time` reads
  from test code raise immediately — there is no allowlist; genuine
  exceptions opt out inline via `@pytest.mark.uses_real_clock("reason")`.
- Pytest temp DBs pick ONE basetemp root via
  `devtools.verify_runs.resolve_pytest_basetemp_root` (shared by
  `tests/conftest.py` and the `devtools test`/`verify` preflight): `/dev/shm`
  tmpfs by default when it has ≥1 GiB free (`POLYLOGUE_PYTEST_BASETEMP_MIN_FREE_MB`
  to override), else `/realm/tmp/polylogue-pytest` (NVMe), else `/tmp/polylogue-pytest`
  only when `/realm/tmp` genuinely isn't mounted (cloud sandbox). If nothing
  clears the headroom requirement the run refuses immediately with every
  candidate's free space named, instead of an unrelated command crashing on
  `OSError: [Errno 28]` later (2026-07-30 incident: `.claude/settings.json`'s
  cloud `POLYLOGUE_PYTEST_BASETEMP_ROOT=/tmp/polylogue-pytest` leaked onto the
  workstation, where `/tmp` is a small tmpfs shared by every concurrent agent
  lane — that env value is now stripped before candidate selection whenever
  `/realm/tmp` is mounted). `seeded_db`/`corpus_seeded_db` build a shared DB
  once under a `.build.done` guard — a SIGKILL mid-build leaves a partial DB +
  set guard → `no such table: sessions`; fix with
  `rm -rf /dev/shm/pytest-polylogue-seeded-*` (and legacy
  `/realm/tmp/polylogue-pytest/pytest-polylogue-seeded-*`). Never `pkill` polylogue pytest without
  clearing this.
- Verify-run artifacts land under `.cache/verify/`
  (`current-pytest-{progress,selection,summary}.json`, `-output.log`).

Demo path (private-data-free) for read/search/reader checks:
`polylogue demo seed … && polylogue demo verify …`, or
`polylogue import --demo --wait`.

---

## devtools (the control plane)

`devtools` owns repo readiness: generated-surface rendering, verification,
validation-lane dispatch, packaging, PR-readiness. Domain semantics live in
lab/schema/scenario/insight modules; `devtools` commands are thin entrypoints.

Core loop:

- `devtools status` — repo state, generated-surface drift, next steps.
- `devtools render all [--check]` — refresh/verify every generated surface after
  changing docs, CLI help, or schema. **Gotcha:** `render all --check` can print
  per-surface `sync OK` yet still exit 1 — grep the output for `out of sync`,
  don't trust the tail line.
- `devtools verify [--quick|--all|--lab|--seed-testmon]` — see
  [Verification](#verification--testmon-inner-loop-never-blanket-run).
- `devtools test <sel>` — focused pytest through the managed harness.
- `devtools lab …` — executable schema/provider/pipeline/lane checks.
- `devtools workspace …` — task history, frontier, worktree-gc, evidence.

Adding a devtools command: add a `CommandSpec` to `devtools/command_catalog.py`,
implement in `devtools/<name>.py`, run `devtools render devtools-reference`.

Local state: `.cache/` (disposable) and `.local/` (untracked outputs). Keep new
outputs there, not new top-level roots.

---

## Cloud lane (Claude Code Web / Codex Cloud)

Well-suited to cloud sandboxes: pure Python, all paths overridable via
`POLYLOGUE_ARCHIVE_ROOT`. Bootstrap: `.claude/setup.sh`; env from
`.claude/settings.json` (`POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-archive`,
`POLYLOGUE_FORCE_PLAIN=1`, `HYPOTHESIS_PROFILE=ci`).

- **Safe:** `uv run pytest tests/unit -q` / `tests/property -q`;
  `ruff check`/`format --check`; `mypy polylogue`; `devtools verify` (slow);
  `render all --check`; `polylogued run --no-api --no-watch --no-browser-capture`
  against synthetic fixtures only.
- **Never in cloud:** uploading a real `~/.claude/projects/` or
  `~/.codex/sessions/` corpus (fixtures only); browser-capture flows; any
  `/realm/data/...` path (not mounted). Privacy tier follows the running
  account — confirm before enabling cloud lanes on sensitive repos
  (`docs/cloud-agents.md`).

---

## Gotchas (hard-won)

- `render all --check` exits 1 even while printing `sync OK` per surface — grep
  for `out of sync`.
- Adding a `polylogue/` module without regenerating the topology projection
  breaks `render all --check` (see [Schema regimes](#schema-regimes-durability-keyed)).
- New Click params on query verbs must go **last** — a positional shift silently
  reroutes args.
- New MCP tool → add its tool contract. `EXPECTED_TOOL_NAMES` is *derived*
  (`set(declared_tool_names(ALL_CAPABILITIES))` in `tests/infra/mcp.py`), so it
  needs no hand-editing; a missing contract is what fails discovery tests.
- New `AssertionKind` is schema-free (`TEXT`, no CHECK) but its enum is embedded
  in `render openapi` + `render cli-output-schemas` — regenerate them.
- Per-PR CI **skips the heavy `test` suite** (runs post-merge on master). A green
  `gh pr checks` does **not** mean tests ran — verify locally with
  `devtools test <files>`. Required merge checks are `lint` + `test`; an
  `UNSTABLE`/neutral `mergeStateStatus` is usually the test-skip, not a failure —
  inspect `statusCheckRollup`.
- Committing from a linked worktree: a hook aborts if you `cd`'d into the main
  checkout from inside a worktree (worktree-escape detector, #1211); set
  `POLYLOGUE_ALLOW_WORKTREE_ESCAPE=1` for legitimate cross-worktree flows.
- **Worktree running against the wrong checkout's `polylogue` (2026-07-31,
  guarded)**: a linked git worktree without its own `.venv` reuses the main
  checkout's shared venv on PATH. That venv's editable install (a `.pth` in
  site-packages) points at the main checkout, so a plain `import polylogue`
  with nothing else on `sys.path` silently resolves there instead of the
  worktree's own source — no ImportError, just wrong code answering every
  question while looking like a genuine result (corrupted four lanes in one
  day: a perf "after" measurement, false CLI-timeout readings, an "impossible"
  schema-version contradiction, a benchmark needing manual `sys.path`
  pinning). `devtools/checkout_guard.py` closes this for every entry point
  that can plausibly hit it: `devtools/__main__.py` → `click_dispatch.main()`
  refuses (exit 125) before dispatching any command; `devtools verify` /
  `devtools test` refuse before running any step and print the resolved
  `polylogue` package path as part of the run receipt (`polylogue_import_path`
  in `.cache/verify/current-run.json`); `tests/conftest.py` refuses via
  `pytest.UsageError` in `pytest_configure`, so even a bare `pytest`/
  `python -m pytest` invocation that skips `devtools` entirely still catches
  it. **Residual gap**: a standalone ad hoc script
  (`python3 /realm/tmp/scratch.py`) that never imports `devtools` or `pytest`
  has no hook to run the check from — the flake devShell already gives every
  checkout its own `.venv` via `direnv allow` (`uv venv` + `uv sync` keyed off
  `$PWD`, so it's correct-by-construction per worktree), but that costs a
  multi-minute sync + a few hundred MB per worktree, which is why short-lived
  agent worktrees reuse the shared venv instead. A script that wants the
  guarantee imports `devtools.checkout_guard.assert_polylogue_matches_checkout`
  itself in one line.
- `AGENTS.md` is a **symlink to this file** (`CLAUDE.md`) — edit CLAUDE.md, never
  AGENTS.md; there is no render step.

---

## Reference docs

Read on demand (paths relative to repo root):

| Topic | Doc |
| --- | --- |
| System rings, data flow, provider table | `docs/architecture.md` |
| Invariants, hot files, schema-version history, extension points | `docs/internals.md` |
| Target shape + architectural decision log | `docs/architecture-spine.md` |
| Execution control center hotspot map + decomposition sequence | `docs/architecture-hotspots.md` |
| Contributor workflow (branches, PRs, hooks, releases) | `CONTRIBUTING.md` |
| Full testing reference | `TESTING.md` |
| devtools command catalog | `docs/devtools.md` |
| Cloud-agent setup + privacy | `docs/cloud-agents.md` |
| Provider/Origin/Source vocabulary table | `docs/provider-origin-identity.md` |
| Retrieval lanes + search semantics | `docs/search.md` |
| Public Python domain models | `docs/data-model.md` |
| Daemon convergence + threat model | `docs/daemon.md`, `docs/daemon-threat-model.md` |
| Cost/usage model | `docs/cost-model.md` |
| CLI reference (generated) | `docs/cli-reference.md` |
| MCP reference | `docs/mcp-reference.md` |
| Normalized-session material protocol v1 (Sinex-independent wire format) | `docs/material-protocol-v1.md` |
