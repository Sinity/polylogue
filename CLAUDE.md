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
| `polylogue/operations/archive.py` | High-level archive operations |
| `polylogue/daemon/cli.py` | Daemon runner (`polylogued run`) |

**Working rule:** new semantics go into the substrate (`storage`/`insights`) or
product layer first, then surfaces adapt. Surfaces may not import substrate
internals directly (`docs/plans/layering.yaml` enforces this).

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
- CHECK constraints are **generated from Python types** —
  `literal_check("status", *get_args(RunStatus))` embeds `typing.Literal` args
  into SQL, so Python type ↔ SQL constraint stay in lockstep.

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
by `resolve_session_links_for_session`. `TopologyEdgeStatus` =
unresolved/resolved/repaired/**quarantined** (cycle-break).

### The five tiers (durability is the axis)

| Tier | ver | durability | holds |
| --- | --- | --- | --- |
| `source.db` | 3 | durable | raw acquired bytes (`raw_sessions`), artifact taxonomy, blob/GC substrate (`blob_refs`, `gc_generations`), hook events, sidecars |
| `index.db` | 24 | **rebuildable** | the whole parsed tree, FTS, `session_links`, cost tables, and all materialized insights |
| `embeddings.db` | 1 | rebuildable | `vec0` virtual table (Voyage 1024-dim), meta, status |
| `user.db` | 4 | **durable, irreplaceable** | unified `assertions` + `user_settings` |
| `ops.db` | 1 | disposable | ingest cursors, attempts, `convergence_debt`, cursor-lag samples, daemon events, embed catch-up runs, otlp |

`user.db` is a **single unified `assertions` table** keyed by a closed
`AssertionKind` (mark / tag / correction / annotation / suppression / metadata /
saved_query / recall_pack / workspace_note / note / decision / caveat / lesson /
blocker / handoff / judgment / pathology / …). It collapsed the old separate
overlay tables; `context_policy_json` (default `{"inject":false}`) gates whether
an assertion is injected into agent context. The column is plain `TEXT` so the
vocabulary can grow without a user-tier schema bump. User corrections are
`AssertionKind.CORRECTION` rows here (a legacy `user_corrections` table survives
only as a compat read path for pre-split single-file archives).

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
Claude Code JSONL. `grok-export` is a reserved origin token with no wired parser
yet.

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

CPU-bound stages go to a `ProcessPoolExecutor`; the main process stays the
**sole SQLite writer**. Blob GC uses two independent safety invariants (leases +
snapshot reference check) to bridge the acquire-blob → commit-row window.

### Schema regimes (durability-keyed)

Two evolution regimes, enforced by `devtools lab policy schema-versioning`:

- **Durable tiers** (`source.db`, `user.db`): explicit **additive** numbered SQL
  migrations under `storage/sqlite/migrations/{source,user}/NNN_*.sql`, one
  `PRAGMA user_version` step at a time, behind a **verified backup manifest**.
  Destructive durable changes need a copy-forward design + explicit consent.
- **Derived tiers** (`index.db`, `embeddings.db`): **no migration chain**. A
  schema mismatch rebuilds/blue-green-replaces the tier from source
  (`polylogue ops reset --index && polylogued run`). Bumping their schema edits
  the canonical DDL + a rebuild plan, never an upgrade helper.

Before editing schema, classify the change: metadata-only, index-only,
additive-derived, additive-durable, or semantic-reparse-required. Batch
same-tier bumps from ready Beads before triggering a live rebuild; don't
repeatedly reset+reingest the active archive for isolated index additions.

**If you add any module/file under `polylogue/`**: regenerate the topology
projection or `render all --check` fails — run
`devtools render topology-projection && devtools render topology-status` and
commit the updated `docs/plans/topology-target.yaml` + `docs/topology-status.md`.

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
  verbs live on the **10-mixin** `SessionRepository`
  (`storage/repository/__init__.py`: archive reads, archive writes, raw,
  vectors, + six insight readers — profile, run-projection, timeline, thread,
  summary, topology) and on `services.py`.
- **MCP** (`mcp/`): the large agent-facing surface (~130 tools across
  `server_*.py`) — search/list/get, insights, corrections, context/recall,
  postmortem bundles. This, not the API, is the continuity surface. Adding a
  tool requires updating `EXPECTED_TOOL_NAMES` + a tool contract.
- **Insights** (`insights/registry.py`): descriptor-driven — one
  `INSIGHT_REGISTRY` where each `InsightType` declares field accessors + a
  Pydantic query model + operations method + CLI/MCP metadata, driving
  plaintext, JSON, and MCP from one place. `project_origin_payload` renames
  provider-token keys → origin at the output boundary (see Vocabulary).
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

The provider→origin retirement is **in progress**, not a done rename.
`project_origin_payload` is a transitional shim: internal storage columns
(`session_profiles.source_name`, cost keys) and insight models still speak
provider vocabulary, so public surfaces project to `origin` at the boundary. A
naive rename is unsafe because `GEMINI` **and** `DRIVE` both collapse to
`AISTUDIO_DRIVE` (non-injective). Tracked in Beads **polylogue-9e5.8**
(retirement plan), **polylogue-jnj.7** (CLI help leakage), **polylogue-2qx**
(OriginSpec). Anti-goal: provider wording on source-origin public filters/payloads.

---

## Working Rules (agent workflow)

These override default agent behavior.

### Beads issue tracking

This repo uses `bd` (Beads) for durable task state AND as the devloop: `bd
prime` -> `bd ready` -> claim -> work -> PR -> close with reasons. The former
bespoke conductor packet is archived at `.agent/archive/devloop-2026-07/`
(evidence, never scaffold — do not resurrect it or `devloop-*` scripts). Repo
agent conventions: `.agent/CONVENTIONS.md`; run `.agent/scripts/bd-graph-lint`
before shipping bead-state deltas. Run `bd prime` when task
context, ready work, blockers, or project memory matter. Use `bd ready --json`,
`bd show <id> --json`, `bd update <id> --claim --json`,
`bd close <id> --reason "…" --json`. Create linked Beads issues for discovered
follow-up work rather than leaving markdown TODOs as the source of truth.
`bd dolt push` follows the same policy as `git push` (feature branches / PR
updates after verification; no direct push to protected default).

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
  (`tests/infra/frozen_clock.py`), not the host wall clock. The
  `verify-test-clock-hygiene` lint rejects new direct `datetime.now`/`time.time`
  in tests outside `docs/plans/test-clock-allowlist.yaml`.
- **Protected — never delete:** `tests/unit/sources/test_parsers_props.py`,
  `test_null_guard_properties.py`; `tests/unit/core/test_properties.py`;
  `tests/integration/`; `tests/unit/security/`;
  `tests/unit/storage/test_crud.py`.
- Pytest temp DBs default to `/realm/tmp/polylogue-pytest` (not `/dev/shm`).
  `seeded_db`/`corpus_seeded_db` build a shared DB once under a `.build.done`
  guard — a SIGKILL mid-build leaves a partial DB + set guard →
  `no such table: sessions`; fix with
  `rm -rf /realm/tmp/polylogue-pytest/pytest-polylogue-seeded-*` (and legacy
  `/dev/shm/pytest-polylogue-seeded-*`). Never `pkill` polylogue pytest without
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
- New MCP tool → update `EXPECTED_TOOL_NAMES` + tool contract, or discovery tests
  fail.
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
