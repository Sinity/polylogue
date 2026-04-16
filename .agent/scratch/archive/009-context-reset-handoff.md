---
created: "2026-04-12T21:40:00+02:00"
purpose: "Reset-safe handoff for runtime vetting, live archive state, and verification strategy"
status: "archived"
project: "polylogue"
---

# Context Reset Handoff

Superseded by `active/009-operator-brief.md` as the live reset-safe entrypoint.

This note is the shortest path back into the current runtime lane after context
reset. It complements:

- `active/005-thorough-vetting-log.md` for detailed chronology and command/results
- `archive/006-architectural-drift-audit.md` for the fuller archived drift record
- `plans/007-vetting-and-hardening-plan.md` for the runtime roadmap
- `plans/008-verification-architecture-plan.md` for the systematic test/benchmark plan

## Branch and Worktree

- Repo: `/realm/project/polylogue`
- Branch: `feature/chore/repo-cleanup-governance`
- Latest architectural/control-plane changes now also on branch:
  - `06408afa` `refactor: extract validation and mutation catalogs`
  - `29e16b3c` `refactor: declare operation targets`
  - `c2d5615f` `refactor: track declared operation coverage`
  - `b496976e` `refactor: extract synthetic benchmark catalog`
  - `f4676efe` `refactor: extract durable benchmark catalog`
  - `365d396f` `refactor: extract mutation campaign catalog`
  - `c7fb9598` `refactor: make exercise scenarios the showcase root`
  - `6c2040a4` `refactor: make generated showcase cases scenario-first`
  - `e42ad9d7` `refactor: centralize qa extra scenarios`
  - `7da91485` `test: cover action-event repair benchmark operation`
  - pending local slice: shared operation catalogs now replace repeated
    tuple-to-map resolution across graph and scenario code
- Worktree is currently clean.
- The worktree is dirty because a parallel docs/natural-language worker is
  editing these files. Treat them as out of scope for the runtime lane:
  - `CLAUDE.md`
  - `CONTRIBUTING.md`
  - `README.md`
  - `TESTING.md`
  - `devtools/docs_surface.py`
  - `docs/README.md`
  - `docs/architecture.md`
  - `docs/devtools.md`
  - `docs/internals.md`
  - `.claude/includes/*`

## Non-Negotiable Operating Constraints

- Do **not** use `python -m devtools`; use `devtools ...`.
- Do **not** replace `orjson` globally with stdlib `json`.
- Keep commits atomic and coherent.
- The user wants real runtime/product vetting, not generic cleanup.
- Memory use matters. Multi-gigabyte RSS on heavy runs is still considered too
  high.
- Avoid opening many long-lived exec sessions. The environment repeatedly hit
  the 60-process ceiling.

## Live Archive Reset / Rebuild State

The user approved resetting the live archive.

- Backup of previous live DB:
  - `~/.local/share/polylogue/reset-backup-20260412T183637+0200`

### Full rebuild

- Log:
  - `.local/logs/run-all-default-20260412T183655+0200.log`
- Result:
  - wall `56m40.43s`
  - peak RSS `7109260 kB` (~6.9 GiB)
  - counts:
    - `9091` conversations changed
    - `9538` raws acquired
    - `7392` conversations materialized/rendered

### Parse rerun

- Log:
  - `.local/logs/run-parse-rerun-20260412T200000+0200.log`
- Result:
  - wall `20m54.02`
  - peak RSS `5613500 kB` (~5.35 GiB)
  - `parse_failures=0`
  - `processed_ids=270`

## Current DB Facts

Known query result:

- `raw_conversations|9538|3|9535|0`

Meaning:

- `9538` raws total
- `3` raws still have `parsed_at IS NULL`
- `9535` parsed
- `0` have `parse_error`

## The 3 Remaining Unparsed Raws

### Two are parseable now, but stuck in stale failed-validation state

Raw ids:

- `030d903225ad997c04498f9f32bf13acd21dfbad08059bc8931ba116703e528b`
- `dead737557ca50accfd9e5deb1a7660412575234f33294ae980ac400c761e74f`

Related source files:

- `/home/sinity/.claude/projects/-realm-project-sinex.pre-enrich/b8c8d990-f5c4-4d01-881a-f4af42ceb7f2.jsonl`
- `/home/sinity/.claude/projects/-realm-project-sinex/b8c8d990-f5c4-4d01-881a-f4af42ceb7f2.jsonl`

Important facts already proven:

- `orjson` rejects escaped lone surrogates on lines `2409` and `2410`
- stdlib `json.loads()` accepts those lines
- `build_raw_payload_envelope(...)` now reports no malformed lines for these
  payloads
- direct `ingest_record(...)` on one of these raws returns:
  - `validation_status='passed'`
  - `validation_error=None`
  - `error=None`
  - one conversation parsed

Interpretation:

- the parser tolerance bug is fixed
- the remaining problem is stale validation state / refresh behavior

### One is genuinely malformed

Raw id:

- `57399b8676d88e84698827874e6f7cee6700f8138841ca058205af05cc73fd7e`

Source:

- `/home/sinity/.claude/projects/-realm-project-sinex/da69faf4-16cb-46e7-94ff-ba40cfa9a346.jsonl`

Known failure:

- line `819`
- `Expecting ',' delimiter: line 1 column 1293 (char 1292)`

Interpretation:

- this should remain a real malformed/quarantine case
- it should not be conflated with the stale validation-state cases above

## Current Live Doctor State

Fresh-process command shape that worked:

```bash
nix develop -c sh -lc '/run/current-system/sw/bin/time -v polylogue --plain doctor --json > /tmp/polylogue-doctor-status.json 2> /tmp/polylogue-doctor-status.time'
```

Measured result:

- wall `13.75s`
- max RSS `124468 kB`

Top-level JSON shape:

- `status`
- `result`

Current summary:

- `31 ok`
- `4 warning`
- `0 error`

Current warnings:

- `empty_conversations`: `288`
- `action_event_fts`: `Action-event FTS pending (312,728/311,421 rows)`
- `retrieval_evidence`: pending for the same underlying reason
- `transcript_embeddings`: entirely pending (`0/7392`)

## Most Important Open Runtime Defect

Action-event FTS health/debt/repair accounting is inconsistent.

Observed live behavior:

- `action_event_read_model`: ok
- `action_event_fts`: warning
- archive debt still says action-event read model is ready
- repair/debt accounting does not surface the FTS drift coherently

Known root cause:

- `polylogue/storage/action_event_status.py`
  - only exposes readiness, not directional drift
- `polylogue/storage/derived_status_products.py`
  - models only pending rows via `max(0, source_rows - materialized_rows)`
  - extra/stale FTS rows are invisible
- `polylogue/storage/repair.py`
  - preview/debt counts only pending rows, not stale drift

Relevant files for the next fix:

- `polylogue/storage/action_event_status.py`
- `polylogue/storage/derived_status.py`
- `polylogue/storage/derived_status_products.py`
- `polylogue/storage/repair.py`
- `tests/unit/storage/test_repair.py`
- `tests/unit/storage/test_derived_status.py`

## Strategic Pivot

The branch already did the expensive discovery pass. Continuing to rely on
large live reruns as the primary loop is now low leverage.

Current recommended strategy:

1. build a shared scenario registry
2. convert known bug classes into law-driven tests and scenario acceptance tests
3. upgrade benchmarks to operator-workflow / pathology scenarios
4. keep live archive runs as periodic canaries only

See `../plans/008-verification-architecture-plan.md` for the full design.

## Existing Verification Assets Worth Reusing

Do not reinvent the world; extend what exists:

- `devtools/benchmark_campaign.py`
- `devtools/benchmark_campaigns.py`
- `devtools/quality_registry.py`
- `devtools/pipeline_probe.py`
- `devtools/query_memory_budget.py`
- `devtools/run_validation_lanes.py`
- `polylogue/showcase/`
- `tests/benchmarks/`
- `tests/integration/test_health.py`

## What The New Control-Plane Unification Already Changed

These are no longer just architectural ideas; they are implemented roots:

- scenario metadata now has a shared home in `polylogue/scenarios/metadata.py`
  and is preserved across:
  - showcase exercises
  - benchmark campaign catalogs
  - quality registry / quality-reference rendering
- runtime and control-plane operation targets are explicitly declared in
  `polylogue/operations/specs.py`
- operation lookup and resolution now also have one shared catalog root in
  `polylogue/operations/specs.py`, reused by:
  - `polylogue/artifact_graph.py`
  - `polylogue/scenarios/metadata.py`
  - `devtools/scenario_coverage.py`
- runtime scenario coverage now tracks:
  - artifacts
  - runtime operations
  - declared operation targets
- showcase is now scenario-first in two separate senses:
  - the serialized showcase catalog loads to `ExerciseScenario` first and
    compiles late
  - generated schema/format/filter/provider families are authored as scenarios
    first, then compiled to exercises
- QA extra exercises no longer choose their generated families in parallel with
  the projection catalog; both derive from one `generate_qa_extra_scenarios()`
  source
- the declared-operation coverage map is now complete; the last gap
  (`benchmark.repair.action-events`) is covered by a real benchmark path in
  `tests/benchmarks/test_pipeline.py`

## Recommended Next Execution Order

### Immediate

1. Fix action-event FTS consistency.
2. Verify:
   - `ruff check` on touched files
   - focused storage/doctor tests
   - live `polylogue --plain doctor --json`

### Then

3. Fix stale validation-state refresh for the two now-parseable raws.
4. Preserve clean malformed/quarantine handling for the one truly bad raw.

### Then

5. Start the systematic verification upgrade:
   - shared scenario registry
   - first law-driven scenario family
   - first operator-scenario benchmarks

### Then

6. Attack materialize/ingest memory with benchmarked scenario loops rather than
   full live reruns.

## Operational Lessons From This Context Window

- Too many open exec sessions caused repeated tool warnings and added noise.
- One timing attempt failed only because `/usr/bin/time` was not present in the
  devshell; use `/run/current-system/sw/bin/time`.
- Long runs must be tied to durable logs, otherwise too much evidence gets lost.
- The branch already has enough discovery. The next gains come from
  systematization.
