# CLI Command Audit

Ref [#1681](https://github.com/Sinity/polylogue/issues/1681). Status: audit phase.
This document inventories every polylogue command surface and proposes
keep/merge/rename/deprecate decisions. Implementation is staged across
sub-issues filed from this audit.

## Command Inventory

### Query Verbs

These act on the matched session set after filter chain evaluation.
Registered in `polylogue/cli/verb_names.py` and `polylogue/cli/query_verbs.py`.

| Command | Recommendation | Rationale |
|---------|---------------|-----------|
| `list` | **Keep** | Canonical listing verb. Already the default. |
| `count` | **Keep** | Single integer output, useful for scripting. Distinct from `stats`. |
| `stats` | **Keep** | Aggregate breakdowns by origin/month/year/tool/repo. Distinct from `count`. |
| `open` | **Keep** | Opens session in daemon web reader. Clear action verb. |
| `show` | **Deprecate** | Overlaps with `list`. Bare query mode already shows full content; `list` with `--format` covers structured output. `show` adds a third path for the same operation. |
| `select` | **Deprecate** | Narrow use case (pick one field from one session). `list --format json \| jq` or `--json` covers this. |
| `recent` | **Keep** | Convenience wrapper: `--latest --sort date`. High usage, worth the alias. |
| `messages` | **Keep** | Lists messages, not sessions. Distinct operation. |
| `raw` | **Keep** | Raw archive payloads. Debugging surface, no overlap. |
| `delete` | **Keep** | Write operation, clear intent, irreversible. |
| `bulk-export` | **Rename to `export bulk`** | Overlaps with `export` top-level command. Merge under `export` group as `export bulk`. |

### Top-Level Commands (Nouns and Verb-Nouns)

Registered in `polylogue/cli/click_command_registration.py`.

| Command | Category | Recommendation | Rationale |
|---------|----------|---------------|-----------|
| `commands` | Meta | **Keep** | Discoverability tool. Delivered in #1717. |
| `status` | Diagnostics | **Keep** | Daemon and archive status. One of the most-used commands. |
| `doctor` | Archive Mgmt | **Keep (rename)** | Currently registered as `doctor` (internal name `check`). Align on `doctor` everywhere — `check` is a Polylogue legacy artifact. Update `commands` listing to say `doctor`. |
| `import` | Archive Mgmt | **Keep** | Import from configured sources. Clear intent. |
| `backup` | Archive Mgmt | **Keep** | Timestamped archive backup. Clear intent. |
| `reset` | Archive Mgmt | **Keep** | Reset archive state. Destructive but clearly named. |
| `maintenance` | Archive Mgmt | **Keep** | Preview and run maintenance backfill. Subcommands: `gc-history`, `plan`, `preview`, `run`, `status`. `maintenance run` overlaps with `doctor` conceptually — `doctor` is health check, `maintenance run` is repair/recompute. Distinction is subtle but real. |
| `config` | Configuration | **Keep** | Show resolved config. Clear intent. |
| `init` | Configuration | **Keep** | Detect sources and write starter config. Onboarding. |
| `auth` | Configuration | **Keep** | Authenticate external services (Drive OAuth). Clear intent. |
| `completions` | Configuration | **Keep** | Shell completion setup. Standard CLI pattern. |
| `dashboard` | Configuration | **Keep** | Open local dashboard. Could be `open --dashboard` but separate command is discoverable. |
| `tutorial` | Configuration | **Keep** | Interactive first-run walkthrough. Onboarding. |
| `export` | Context | **Keep (expand)** | Export sessions. Currently flat; absorb `bulk-export` as `export bulk`. |
| `context` | Context | **Keep (expand)** | Context Composer. Currently has `compose` subcommand only. Absorb `context-pack` as `context pack`. |
| `context-pack` | Context | **Merge into `context`** | Build a context pack for agent analysis. Duplicate noun with `context`. Merge as `context pack`. |
| `resume` | Insights | **Keep (expand)** | Resume from recent session context. Absorb `resume-candidates` as `resume --candidates` or `resume candidates`. |
| `resume-candidates` | Insights | **Merge into `resume`** | Rank resume candidates. Thin convenience over `resume`. Merge as `resume candidates` or `resume --candidates`. |
| `correlate` | Insights | **Keep** | Git commit/GitHub ref correlation. Distinct capability. |
| `facets` | Insights | **Keep** | Scoped/global facet aggregates. Distinct capability. |
| `neighbors` | Insights | **Keep** | Semantic neighbor candidates. Distinct capability. |
| `cost` | Analytics | **Keep** | Subscription usage and cost telemetry. Subcommands: `outlook`, `rollup`. |
| `insights` | Insights | **Keep (restructure)** | 14 subcommands. Needs grouping — see Subcommand Grouping below. |
| `diagnostics` | Diagnostics | **Keep** | Temporal session diagnostics. Subcommands: `pace`, `tools`, `turns`. |
| `embed` | Embeddings | **Keep** | Embedding pipeline management. Subcommands: `activate`, `backfill`, `disable`, `enable`, `preflight`, `status`. |
| `feedback` | User-State | **Keep** | Learning corrections for derived insights. Subcommands: `clear`, `list`, `record`. |
| `schema` | Schema | **Keep** | Schema package inspection. Subcommands: `compare`, `explain`, `list`. |
| `tags` | User-State | **Keep** | Tag management. Flat command with list/add/remove. Works as-is in the query-first model. |
| `user-state` | User-State | **Keep** | Durable reader marks, annotations, saved views. Subcommands: `annotations`, `marks`, `recall-packs`, `saved-views`, `workspaces`. |
| `blackboard` | User-State | **Keep** | Agent-addressable notes surface. Subcommands: `list`, `post`. |

### Subcommand Inventories

#### `insights` (14 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `audit` | **Keep** | Per-product rigor profile. Admin surface. |
| `cost-rollups` | **Keep** | Provider/model cost rollups. Analytics. |
| `costs` | **Keep** | Session-level cost estimates. Analytics. |
| `coverage` | **Keep** | Archive coverage buckets. Aggregate. |
| `debt` | **Keep** | Archive debt and maintenance readiness. Aggregate. |
| `export` | **Keep** | Versioned insight bundles. Admin. |
| `phases` | **Keep** | Session-phase insights. Session-level. |
| `profiles` | **Keep** | Session-profile insights. Session-level. |
| `status` | **Keep** | Insight materialization coverage. Admin. |
| `tags` | **Keep (rename to `tag-rollups`)** | Session-tag rollup insights. Current name `tags` collides with top-level `tags` command. |
| `threads` | **Keep** | Work-thread insights. Aggregate. |
| `timeline` | **Keep** | Per-session timeline. Session-level. |
| `tool-usage` | **Keep** | Per-tool per-origin rollups. Analytics. |
| `work-events` | **Keep** | Work-event insights. Session-level. |

Rename `insights tags` to `insights tag-rollups` to resolve collision with `polylogue tags`.

#### `cost` (2 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `outlook` | **Keep** | Cycle projection for subscription plan. |
| `rollup` | **Keep** | Flat cost rollup (legacy). |

#### `maintenance` (5 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `gc-history` | **Keep** | Blob-GC pass history. |
| `plan` | **Keep** | Dry-run summary. |
| `preview` | **Keep** | Staleness inventory. |
| `run` | **Keep** | Execute (or dry-run) backfill. |
| `status` | **Keep** | Persisted operation status. |

#### `diagnostics` (3 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `pace` | **Keep** | Inter-turn gap analysis. |
| `tools` | **Keep** | Top tools by invocation count. |
| `turns` | **Keep** | Per-turn cost and duration. |

#### `embed` (6 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `activate` | **Keep (alias)** | Alias for `enable`. Keep both — `activate` is the user-facing verb, `enable` is the config verb. |
| `backfill` | **Keep** | Run embedding batch. |
| `disable` | **Keep** | Disable pipeline without dropping embeddings. |
| `enable` | **Keep** | Turn on pipeline. |
| `preflight` | **Keep** | Cost estimate for pending backlog. |
| `status` | **Keep** | Embedding coverage and freshness. |

#### `schema` (3 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `compare` | **Keep** | Compare two schema package versions. |
| `explain` | **Keep** | Explain a package element schema. |
| `list` | **Keep** | List available packages and versions. |

#### `feedback` (3 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `clear` | **Keep** | Remove corrections. |
| `list` | **Keep** | List stored corrections. |
| `record` | **Keep** | Record a typed correction. |

#### `user-state` (5 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `annotations` | **Keep** | Session and message annotations. |
| `marks` | **Keep** | Session and message marks. |
| `recall-packs` | **Keep** | Recall packs with target evidence. |
| `saved-views` | **Keep** | Saved query views. |
| `workspaces` | **Keep** | Durable reader workspaces. |

#### `blackboard` (2 subcommands)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `list` | **Keep** | List blackboard notes. |
| `post` | **Keep** | Post a note. |

#### `context` (1 subcommand)

| Subcommand | Recommendation | Notes |
|-----------|---------------|-------|
| `compose` | **Keep** | Compose context preamble. Will be joined by `pack` after merge. |

---

## Verb Taxonomy

Every intent maps to exactly one verb. This table is the decision record.

| Intent | Chosen Verb | Existing Commands | Rename / Deprecate |
|--------|------------|-------------------|-------------------|
| List multiple items | `list` | `list`, `insights profiles`, `insights phases`, `feedback list`, `schema list`, `blackboard list` | Already consistent across groups. |
| Show single-item detail | `show` | *(none exposed as `show` yet)* | Keep `show` reserved for single-item detail. Do not use for listing. |
| Count items | `count` | `count` | Already standard. |
| Aggregate statistics | `stats` | `stats` | Already standard. Distinct from `count`. |
| Open in external viewer | `open` | `open` | Already standard. |
| View raw/debug data | `raw` | `raw`, `insights audit` | `audit` could become `raw` but its report shape is distinct. |
| Check health/status | `status` | `status`, `doctor`, `insights status`, `maintenance status`, `embed status` | `status` = read-only snapshot. `doctor` = active health check + repair (distinct). Keep both. |
| Ingest/import data | `import` | `import` | Already standard. |
| Export data | `export` | `export`, `bulk-export`, `insights export` | Merge `bulk-export` into `export bulk`. Keep `insights export` as distinct surface. |
| Create/initialize | `init` | `init` | Already standard. |
| Configure | `config` | `config`, `auth`, `completions` | These are distinct enough to stay separate. |
| Manage/modify state | *(noun-as-verb)* | `tags`, `feedback`, `user-state`, `blackboard` | These are object surfaces, not pure verbs. The noun-is-the-command pattern is fine. |
| Delete/remove | `delete` | `delete`, `reset` | `delete` operates on matched set. `reset` is full archive reset. Distinct. |
| Rebuild/repair | `run` | `maintenance run`, `doctor` | `maintenance run` executes backfill. `doctor` runs health checks. Close but distinct. |
| Inspect/explain | `explain` | `schema explain`, `config` | Keep per-group. |
| Record/set | `record` | `feedback record`, `blackboard post` | `post` vs `record`: `post` is for free-form notes; `record` is for typed structured corrections. OK as-is. |

### Verbs to deprecate

| Verb | Reason | Replacement |
|------|--------|-------------|
| `show` (query verb) | Overlaps with `list` and bare query mode | Use `list` or bare `polylogue "query"` |
| `select` (query verb) | Narrow use case; jq covers it | Use `list --json \| jq '.[0].field'` |

### Naming conventions

- **Subcommand names**: lowercase, hyphenated (`work-events`, `cost-rollups`, `tag-rollups`).
- **Group names**: singular noun (`insight` vs `insights`). Current convention is plural (`insights`, `diagnostics`). Changing to singular would be a large rename — defer to a separate decision.
- **Flag names**: hyphenated, no abbreviations except `-p`/`-r`/`-t` (established).

---

## Subcommand Grouping: `insights`

Current flat list (14 subcommands):

```
insights audit        insights costs          insights export
insights cost-rollups insights coverage       insights phases
insights debt         insights profiles       insights status
insights tags         insights threads        insights timeline
insights tool-usage   insights work-events
```

### Proposed grouping by axis

**Session-level** (one row per session):
- `profiles` — session-profile insights
- `work-events` — work-event insights
- `phases` — session-phase insights
- `timeline` — per-session hook-vs-sort-key timeline

**Aggregate** (cross-session rollups):
- `threads` — work-thread insights
- `tag-rollups` — session-tag rollup insights (renamed from `tags`)
- `coverage` — archive coverage buckets by origin/day/week
- `debt` — archive debt and maintenance readiness

**Analytics** (cost and tool metrics):
- `costs` — session-level cost estimates
- `cost-rollups` — origin/model cost rollups
- `tool-usage` — per-tool per-origin rollups

**Admin** (status, export, audit):
- `status` — insight materialization coverage
- `audit` — per-product rigor profile
- `export` — versioned insight bundles

### Implementation approach

Option A (recommended): Add visual section headers in `insights --help` output.
No structural change to Click command tree. Lowest risk.

Option B: Add intermediate Click groups (`insights sessions profiles`, etc.).
Cleaner namespace but doubles command depth and breaks existing muscle memory.

Recommendation: **Option A** for now. Revisit if users report the flat list as
confusing. The `polylogue commands` listing already provides discoverability.

---

## Sub-Issue Clusters

The following clusters are filed as separate GitHub issues so each can land as
an independent PR.

1. **[#1723](https://github.com/Sinity/polylogue/issues/1723)** — Deprecate `show` and `select` query verbs
2. **[#1724](https://github.com/Sinity/polylogue/issues/1724)** — Align `doctor`/`check` naming; rename `insights tags` to `insights tag-rollups`
3. **[#1725](https://github.com/Sinity/polylogue/issues/1725)** — Merge `context-pack` into `context`, `resume-candidates` into `resume`, `bulk-export` into `export`
4. **[#1726](https://github.com/Sinity/polylogue/issues/1726)** — Add section headers to `insights --help` output
5. **[#1727](https://github.com/Sinity/polylogue/issues/1727)** — Add deprecation-warning alias infrastructure with one-release grace period

---

## Backward Compatibility Window

Per acceptance criterion 5 of #1681:

1. **Deprecation release**: Renamed commands emit a deprecation warning to
   stderr listing the replacement, then delegate to the new implementation.
   The old name continues to work.
2. **Removal release** (next minor): The old alias raises "unknown command"
   with a rebuild hint.
3. **Implementation pattern**: A `DeprecatingCommand` wrapper (similar to
   `_LazyCommand`) that prints the warning on first invocation.

Commands requiring deprecation aliases:
- `show` → `list`
- `select` → `list --json | jq`
- `context-pack` → `context pack`
- `resume-candidates` → `resume --candidates`
- `bulk-export` → `export bulk`
- `check` (internal) / `doctor` (CLI): pick one, alias the other

---

## JSON Output Support

Ref [#1689](https://github.com/Sinity/polylogue/issues/1689). Every command
should accept `--json` (shortcut for `--format json`) and emit pipeable,
machine-parseable output without ANSI escape codes. This section inventories
the current state.

### Envelope Convention

Commands that produce structured output use one of two conventions:

1. **Success/error envelope** (`{"status": "ok"|"error", ...}`): Used by
   commands that can fail at runtime (query verbs, schema explain, doctor,
   tags, feedback). Success carries a `result` key; errors carry `code` and
   `message`. Defined in `polylogue/cli/shared/machine_errors.py` and
   published as `docs/schemas/cli-output/machine-success.schema.json` and
   `docs/schemas/cli-output/machine-error.schema.json`.

2. **Raw JSON dump**: Used by commands that always succeed or whose output
   is a simple Pydantic `model_dump(mode="json")` (facets, status, config,
   cost rollup). These do not wrap in the machine envelope.

### Stable JSON Output (Snapshot-Covered)

These commands have dedicated syrupy snapshot tests pinning their JSON shape.
Changes to the output schema will cause a snapshot diff and must be intentional.

| Command | Format Invocation | Snapshot Test | Envelope |
|---------|-------------------|---------------|----------|
| `list` | `polylogue --plain list --format json` | `test_json_list_snapshot` | N/A (structured rows) |
| `count` | `polylogue --plain count` | `test_json_count_snapshot` | N/A (bare integer) |
| `stats` | `polylogue --plain stats --format json` | `test_json_stats_snapshot` | N/A (dimension/rows/summary) |
| `facets` | `polylogue --plain facets --format json` | `test_json_facets_snapshot` | N/A (Pydantic model dump) |
| `status` | `polylogue --plain status --format json` | `test_json_status_snapshot` | N/A (direct JSON) |

Snapshots live in `tests/unit/cli/__snapshots__/test_plain_cli_snapshots.ambr`.

### Contract-Tested JSON Output

These commands are tested for the machine envelope contract in
`tests/unit/cli/test_json_envelope_contract.py`. They emit
`{"status": "ok"|"error", ...}` envelopes but do not have full snapshot
coverage of their result payloads.

| Command | Format Invocation | Contract Test |
|---------|-------------------|---------------|
| `doctor` | `polylogue doctor --format json` | `TestCheckJsonEnvelope` |
| `tags` | `polylogue tags --format json` | `TestTagsJsonEnvelope` |
| `schema explain` | `polylogue schema explain --provider <p> --format json` | `TestSchemaExplainJsonContract` |
| `schema list` | `polylogue schema list --format json` | `TestAllJsonCommandsProduceValidJson` |
| `config` | `polylogue config --format json` | `TestConfigJsonContract` |
| `cost rollup` | `polylogue cost rollup --format json` | `TestCostJsonContract` |

### Crash-Free (Parametrized Test)

The parametrized test in `tests/unit/cli/test_json_output.py`
(`TestAllCommandsAcceptJson`) verifies that every non-destructive
command accepts `--json --plain` without tracebacks, ANSI codes, or
broken JSON output. 78 commands are covered as of 2026-05-28.

### Known Gaps

| Gap | Detail | Issue |
|-----|--------|-------|
| Root `--json` doesn't propagate to all query verbs | `polylogue --json list` produces plain text, not JSON. The root `--json` sets `output_format` in the root context but query verbs with their own `--format` option do not inherit it. Use the per-verb `--format json` flag as a workaround. | #1689 |
| `recent` hardcodes invalid sort field | `recent_verb` passes `sort="updated_at"` which is rejected by `SessionQuerySpec`. The `--json` test excludes `recent` until this is fixed. | Pre-existing |
| `count` has no `--format` flag | The `count` verb always emits a bare integer; there is no way to request a JSON envelope. | #1689 |
| Deeply-nested groups (`user-state`, `blackboard`) don't dispatch through lazy wrappers | Commands registered as `_LazyCommand` but implemented as Click groups don't dispatch subcommands through the lazy wrapper. | #1725 |
| No `--json` test for mutation commands | `delete`, `reset`, `import`, `backup` are excluded from the parametrized test because they modify state. They should still accept `--json`/`--machine` without crashing. | #1689 |

### Commands Without JSON Support

These commands have no `--json` or `--format json` flag and do not emit
machine-parseable output:

- `completions` — emits shell scripts
- `dashboard` — side effect (opens browser)
- `open` — side effect (opens browser)
- `tutorial` — interactive
- `auth` — interactive, OAuth flow
- `init` — interactive, writes config
- `backup` — creates file, prints path
- `messages` — has `--format json` but not snapshot-covered
- `raw` — has `--format json` but not snapshot-covered
- `export` — has `--format json` but not snapshot-covered
- `insights *` — 14 subcommands, mixed JSON support (`status`/`audit` contract-tested)
- `diagnostics *` — 3 subcommands, no JSON snapshots
- `maintenance *` — 5 subcommands, contract-tested for `status`

### JSON Schema Publication

Published JSON Schemas for stable CLI output surfaces live under
`docs/schemas/cli-output/`. Run `devtools render-cli-output-schemas` to
regenerate after changing a Pydantic model that feeds a CLI output surface.

---

## Related Issues

- [#1625](https://github.com/Sinity/polylogue/issues/1625) — `insights coverage|debt|...` hang with no output
- [#1679](https://github.com/Sinity/polylogue/issues/1679) — `import` suggests `status` which shows wrong info
- [#1689](https://github.com/Sinity/polylogue/issues/1689) — generalized `--json` output
- [#1680](https://github.com/Sinity/polylogue/issues/1680) — `recent`/`resume` commands (landed)
