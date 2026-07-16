---
created: 2026-07-09
purpose: Evidence artifact for 5 sibling audit beads (polylogue-9e5.20, .21, .22, .11, .18)
status: read-only investigation, complete
project: polylogue
---

# Test-suite meta-health: flakiness, mock-depth, coverage, economics, fuzz-CI (2026-07-09)

This is one combined evidence artifact for five `polylogue-9e5` audit-lane
children that share data sources (`.cache/verify/`, `.cache/coverage/`,
`git log`, `tests/`). Per the epic contract, this document is **read-only
evidence + follow-up proposals**; no product code was changed to produce it.

Data sources used: `.cache/verify/runs/` (3,975 run directories,
2026-06-18 .. 2026-07-09), `.cache/coverage/coverage.json` (fresh from a
full-suite run completed 2026-07-09 08:47 UTC, ~7h old at time of writing),
`docs/test-economics.md` + `devtools/test_economics_report.py` (already
merged to `master` in PR #2613, 2026-07-09, as part of a prior session
closing out most of bead `.11`'s own acceptance criteria), and an AST scan
written for this investigation over `tests/unit/` (679 files, 8,948
`test_*` functions).

---

## 1. polylogue-9e5.20 — Flakiness tracking + quarantine lane

**Method.** Every `devtools verify`/`devtools test` invocation persists a run
directory under `.cache/verify/runs/<run_id>/` with a `run.json` (tier,
`git_head`, `git_dirty`, timestamps) and, for pytest-bearing steps, an
`events.jsonl` live-event log with one `test_report` event per test outcome.
I scanned all 3,975 run dirs (1.6 GB, 1.2M event lines) and built a per-test
outcome history, collapsing each test's per-run result to a single
pass/fail (fail wins over pass within the same run, matching pytest's own
call/setup/teardown precedence).

**Finding 1 — the exact ACs this bead asks for (flaky = pass+fail on
identical commit) are not reliably computable from current artifacts.**
`run.json.git_head` is populated for only 1,465/3,975 runs (37%), and the
gap is systematic, not random: it breaks down by `tier`:

| tier | runs | has `git_head` |
| --- | --- | --- |
| `focused-test` (the `devtools test <sel>` inner loop — the dominant tier) | 2,510 | **0** |
| `quick` (pre-push `--quick` gate) | 1,373 | 1,373 |
| `testmon` | 87 | 87 |
| `full` / `lab` / `seed-testmon` | 5 | 5 |

`focused-test` is exactly the tier that runs pytest with per-test granularity
(it's the one carrying `events.jsonl`), and it never records commit identity.
So the one artifact type that has per-test outcomes is precisely the one
without commit identity — a same-commit flakiness ledger cannot be built
from what's on disk today without a harness change.

**Finding 2 — collapsing across ALL runs regardless of commit (weaker
signal), 149 of 14,518 distinct `test_*` nodeids show both a pass and a fail
somewhere in the 3-week window.** The top of that list by occurrence count:

| test | observed | pass/fail | distinct known `git_head`s seen |
| --- | --- | --- | --- |
| `cli/test_plain_cli_snapshots.py::test_json_status_snapshot` | 70 | 60/10 | 24 |
| `cli/test_query_verbs_runtime.py::test_read_format_completion_comes_from_selected_view_profile` | 67 | 65/2 | 14 |
| `cli/test_query_expression.py::TestDaemonSessionIdFilter::test_id_miss_returns_typed_empty_not_500` | 56 | 53/3 | 26 |
| `mcp/test_envelope_contracts.py::...test_every_registered_tool_is_classified` | 36 | 32/4 | 12 |

Most entries in this list fail only 1-4 times out of dozens of observations,
spread across many distinct commits — the signature of a real regression
that was later fixed, not nondeterministic flakiness. `test_json_status_snapshot`
corroborates this directly: it's the exact stale-syrupy-snapshot regression
independently identified in `polylogue-w9wt` (`expected_user_version 24 vs
actual 28`), i.e. a real, fixed bug, not a flake.

**Finding 3 — zero true in-run nondeterminism found.** Grouping strictly by
`(run_id, nodeid)` (same process invocation, same code, same worktree state)
turns up **0** cases of a test reporting both pass and fail within one run —
no evidence of order-dependence or xdist-worker races in the retained
window.

**Finding 4 — the documented 3.11 concurrency flake does not appear as a
failure at all.** `test_concurrent_reads_during_writes` shows 18/18 `passed`
across every retained run. Either it hasn't recurred in this window, or the
harness's default interpreter (3.13, per `.cache/mypy/3.11` being the *only*
mypy cache present suggests otherwise — worth checking directly) doesn't hit
the original 3.11-specific path.

**Conclusion for the bead's design:** the quarantine-marker convention
(marker + required owner-bead ref, auto-expire on N green runs) is still a
good design and can be specified now, but the flakiness *ledger* it depends
on needs a harness fix first — `focused-test`-tier `run.json` must record
`git_head`/`git_dirty` (a one-line addition next to the `quick`/`testmon`
tiers that already do this) before same-commit flake detection is possible
at all.

**Follow-up proposals:**
1. *"Record `git_head`/`git_dirty` on `focused-test`-tier verify runs"* —
   small, mechanical fix to whatever assembles `run.json` for that tier;
   unblocks both this bead and the already-filed `polylogue-d45p` (flake
   ledger), which reads the same artifact base and should be sequenced
   after this fix rather than before it.
2. *"Add `pytest.mark.flaky(bead=...)` quarantine marker + lint"* — the
   marker/lint/auto-expire design itself (keeps failures non-blocking but
   visible, requires an owning bead ref, same discipline pattern as the
   existing `slow`/`load_sensitive` markers in `pyproject.toml`). Can proceed
   independent of #1 since it doesn't need the ledger to exist first, only
   to be designed against it.

---

## 2. polylogue-9e5.21 — Mock-depth measurement

**Method.** Wrote an AST scanner (kept at
`/tmp/.../scratchpad/mock_depth_scan.py` for this investigation, not
committed — re-runnable, ~30 lines of classification logic) over all 679
files / 8,948 `test_*` functions in `tests/unit/`. Per test function it: (a)
resolves every `patch(...)`/`patch.object(...)`/`monkeypatch.setattr(...)`
call (decorator, `with`, or plain statement) to a dotted target string; (b)
classifies each target's depth as `stdlib/3rd-party`, `own-module boundary`
(same top-level `polylogue.<pkg>` as the test file's own `tests/unit/<pkg>/`
directory), or `foreign-internal` (a *different* `polylogue.<pkg>`); (c)
collects local names bound to `Mock`/`MagicMock`/`AsyncMock` (via
assignment, `as` binding, or `@patch`-injected params) and classifies every
`assert` in the function as mock-directed (references a mock-bound name) or
real-output.

**Headline finding — assert-on-mock ratio is low overall (0.9%): 126
mock-directed asserts out of 13,721 total.** This is a reassuring result,
not a problem: the suite overwhelmingly asserts on real return values/DB
rows/CLI output, not on mock call bookkeeping. The AC's "assert-on-mock
ratio" concern is not a repo-wide problem.

**Patch-depth totals (tests/unit):**

| class | count |
| --- | --- |
| own-module boundary | 858 |
| other/unresolved (dynamic target, can't classify) | 639 |
| unresolved (patch target not a string literal) | 506 |
| foreign-internal | 272 |
| stdlib/3rd-party | 107 |

**Caveat that matters before acting on the ranking:** of the 272
foreign-internal patches, 51 (19%) are `polylogue.paths.db_path` /
`polylogue.paths.archive_root` redirection — a deliberate, repo-wide
test-isolation idiom (used across the CLI/daemon/storage suites to point
tests at a temp archive root), not evidence of "testing the mock." Any
worst-offender ranking should exclude this idiom or it inflates files that
use it heavily (e.g. `cli/test_status.py`'s raw foreign-internal count of 53
drops to ~11 once the 42 `paths.*` patches in that file are excluded — the
residual 11 include `polylogue.storage.embeddings.status_payload.embedding_status_payload`,
patched directly from a CLI test, which *is* the pattern this bead is
hunting for).

**Worst files by (raw) foreign-internal patch count, before the `paths.*`
correction** (see the caveat above — treat `cli/test_status.py` and any
other heavy `paths.*` user in this list as lower-priority than its raw rank
suggests until re-scored):

| file | test funcs | foreign-internal (raw) | own-module | assert-on-mock ratio |
| --- | --- | --- | --- | --- |
| `cli/test_status.py` | 44 | 53 (→ ~11 after `paths.*` correction) | 26 | 0.0% |
| `cli/test_embed_activation.py` | 35 | 21 | 17 | 11.9% |
| `core/test_operator_inference.py` | 11 | 21 | 0 | 0.0% |
| `cli/test_messages.py` | 8 | 11 | 0 | 0.0% |
| `maintenance/test_planner_contract.py` | 30 | 11 | 0 | 0.0% |
| `pipeline/test_run_stages_runtime.py` | 6 | 10 | 6 | 19.6% |
| `maintenance/test_planner_filter_narrowing.py` | 5 | 10 | 0 | 0.0% |
| `demo/test_workspace.py` | 11 | 9 | 2 | 0.0% |

`core/test_operator_inference.py` and `maintenance/test_planner_*.py` don't
use the `paths.*` idiom (confirmed by spot grep) — these are the cleanest
real candidates: 100% foreign-internal patching, zero own-module patches,
zero mock-directed asserts (so at least the assertion quality is fine; the
concern is purely "does this have to reach into another package's
internals, or could it drive the real thing via `SessionBuilder`/`DbFactory`
from `tests/infra/storage_records.py`?").

**Scope note:** the AC also asks to "convert three worst offenders to
infra-backed tests" — that is write access to test code, explicitly out of
scope for this read-only investigation per the delegation constraints. The
ranking above is the input to that follow-up, not a substitute for it.

**Follow-up proposals:**
1. *"Re-score mock-depth ranking excluding the `polylogue.paths.*`
   test-isolation idiom"* — cheap script change (exclude that one dotted
   prefix from the foreign-internal bucket, or give it its own
   `environment-seam` class) so the worst-offender list isn't dominated by a
   pattern that isn't actually the smell being hunted.
2. *"Convert `core/test_operator_inference.py` and
   `maintenance/test_planner_contract.py`/`test_planner_filter_narrowing.py`
   to `SessionBuilder`/`DbFactory`-backed tests"* — these three are the
   cleanest, highest-confidence conversion candidates from this pass (no
   `paths.*` noise, 100% foreign-internal, no own-module patches at all
   suggesting the module under test isn't being exercised directly).

---

## 3. polylogue-9e5.22 — Per-module coverage tracking

**Method.** A fresh full-suite `--cov=polylogue` run already exists on disk:
`.cache/coverage/coverage.json` (10 MB, generated 2026-07-09 08:47 UTC per
mtime, roughly 7 hours before this scan — same run that produced the
already-merged `docs/test-economics.md`, PR #2613). Per the task's own
instruction, I did **not** re-run the full suite; I parsed the existing
`coverage.json` directly (statement-level `summary.percent_covered` per
file, aggregated to top-level `polylogue/<pkg>` for context, and left at
per-file granularity for the "worst 3" ask since a package aggregate can
hide a single bad file, which is the whole point of this bead).

**Overall totals (from the same coverage.json):** 83.96% statement coverage
line-based / 87.2% "percent_statements_covered" / 73.5% branch coverage
(96,837 statements, 12,431 missing; 29,656 branches, 7,854 missing). The
repo's own `pyproject.toml` ratchet floor is `fail_under = 82` (branch=true),
explicitly a temporary reduced floor from the #1743 split-file migration,
with a note tracking the ratchet back to 90 under issue #1793 — so the repo
is currently running ~2 points above its own gate, not at a comfortable
margin.

**Worst 3 files by coverage % (excluding files with 0 statements):**

| file | coverage % | statements | missing |
| --- | --- | --- | --- |
| `polylogue/archive/semantic/outlook.py` | 0.0% | 88 | 88 |
| `polylogue/context/assertion_claims.py` | 0.0% | 5 | 5 |
| `polylogue/publication/__init__.py` | 0.0% | 38 | 38 |

(A fourth, `polylogue/storage/sqlite/queries/mappers_run_projection.py`, is
also at a literal 0.0%/16 missing — tied for worst by percentage but listed
4th since the AC asks for exactly 3; all four are equally "completely
unexercised by the suite," not "somewhat undertested.")

These three (four) are qualitatively different from a normal coverage hole:
they are **zero**, not low — every statement in the file is unexecuted by
the full suite. `docs/test-economics.md` (package-level view, already
merged) independently flags `polylogue/publication` as a whole package at
0.0% coverage and explicitly recommends confirming it's dead/unwired rather
than writing tests for it, per the repo's "reference-count is not
legitimacy" doctrine — `archive/semantic/outlook.py` and
`context/assertion_claims.py` deserve the identical question, not an
automatic "write tests" reflex.

**Follow-up proposals:**
1. *"Triage `polylogue/archive/semantic/outlook.py`,
   `polylogue/context/assertion_claims.py`,
   `polylogue/storage/sqlite/queries/mappers_run_projection.py` for
   dead/unwired vs untested-but-live"* — for each, resolve whether it's
   reachable from any surface (CLI/MCP/API) at all before deciding "write
   tests" vs "delete" vs "wire it up." (The publication package is already
   covered by this exact question via `docs/test-economics.md`.)
2. Per-package floor / ratchet-policy wiring is explicitly a product-code
   change (edits `pyproject.toml` + `devtools verify`), so it is out of
   scope for this read-only pass; `docs/test-economics.md` already gives the
   per-package numbers this floor would ratchet against, so that follow-up
   has its baseline ready.

---

## 4. polylogue-9e5.11 — Test-suite economics: coverage vs fix-density map

**This bead's own acceptance criteria are already substantially satisfied
on `master`.** A prior session (this same day, per `bd show` notes and `git
log`) shipped `devtools/test_economics_report.py` (registered as
`devtools lab test-economics`, 485 lines) and the committed table
`docs/test-economics.md`, merged in `bd4e96230` / PR #2613
("docs(quality): test-suite economics table + fix-density map"). That report
computes, per top-level `polylogue/` package: coverage %, `fix:`-commit
count, testmon wall-time cost-exposure, and testmon fan-out
(median/max distinct-tests-per-file), classified into 5 quadrants. I did not
duplicate that computation; I verified it's live, re-derived a couple of
its headline numbers as a sanity cross-check (the coverage.json worst-file
list above is consistent with `docs/test-economics.md`'s per-package
numbers — e.g. `storage` at 85.6%/201 fix-commits classified "under-tested
substrate" matches the file-level worst-3 above, which are all under
`storage`/`archive`/`context`/`publication`), and read the five follow-up
beads it already filed.

**Existing quadrant table (from `docs/test-economics.md`, reproduced for
this combined artifact — see that file for the full 29-row table and
metric-definition caveats):**

| Package | Coverage % | Fix commits | Quadrant |
| --- | --- | --- | --- |
| `storage` | 85.6 | 201 | under-tested substrate |
| `cli` | 83.0 | 151 | under-tested substrate |
| `daemon` | 81.4 | 118 | under-tested substrate |
| `sources` | 89.3 | 92 | under-tested substrate |
| `mcp` | 90.2 | 31 | well-covered risk area |
| `surfaces` | 95.4 | 11 | over-tested mechanical surface |
| `paths` | 98.2 | 3 | over-tested mechanical surface (flagged as not fitting cleanly — foundational stable plumbing) |
| `publication` | 0.0 | 0 | low-risk, low-cost (fine as-is) — but see triage proposal above |

**Already-filed follow-ups from that report (I did not re-file these,
listing for completeness of this combined artifact):** `polylogue-znwj`
(daemon/cli.py — 46 fix-commits, the single highest-churn file in the repo,
at 73.6% coverage), `polylogue-c52g` (cli/query_verbs.py + commands/status.py),
`polylogue-csg7` (testmon fingerprint-graph staleness —
`verification/manifests/models.py` shows 0 testmon-graph edges despite
79.6%/84.6% real coverage), `polylogue-ixqt` (surfaces package mechanical
over-coverage review), `polylogue-w9wt` (triage of 5 pre-existing test
failures surfaced while building the report — one of which,
`test_json_status_snapshot`'s stale snapshot, independently corroborates
finding 2 in section 1 above).

**New cross-reference this combined pass adds:** the mock-depth worst-file
list (section 2) and the economics report's under-tested-substrate packages
overlap meaningfully — `cli` and `daemon` appear in both as
high-fix-density/under-tested *and* homes to some of the highest raw
foreign-internal-patch test files (`cli/test_status.py`,
`daemon/test_daemon_cli.py`, `daemon/test_daemon_http_security.py`). That's
suggestive (packages that break often are also packages whose tests reach
deepest into other packages' internals rather than exercising real behavior)
but not proven by this pass — it would need the `paths.*`-corrected ranking
from proposal 2.1 before drawing a firm conclusion.

**Follow-up proposals:** none new beyond what `docs/test-economics.md`
already filed (`znwj`/`c52g`/`csg7`/`ixqt`/`w9wt`) — this bead's own AC is
satisfied by the existing artifact; the honest action here is to close it
citing that artifact rather than re-deriving a duplicate one.

---

## 5. polylogue-9e5.18 — Wire atheris fuzz targets into CI (read-only slice)

**Method.** Confirmed import-cleanliness of all 4 fuzz target modules via
`python3 -c "import tests.fuzz.<module>"` (all exit 0, even without
`atheris` installed in this environment — each module guards
`import atheris` and falls back to pytest-compatible mode). Then checked
whether the modules' own pytest-mode tests are actually collected and run by
the normal suite, since the module docstrings and `tests/fuzz/README.md`
both claim "pytest mode... runs in the normal test suite... on every
commit."

**All 4 targets import cleanly:** `fuzz_fts5_escape.py`,
`fuzz_json_parsers.py`, `fuzz_path_sanitizer.py`, `fuzz_timestamp.py`.

**Finding — the "runs in the normal test suite" claim in
`tests/fuzz/README.md` is currently false.** `pyproject.toml`'s
`[tool.pytest.ini_options]` sets `testpaths = ["tests"]` but does not
override the default `python_files` collection pattern (`test_*.py
*_test.py`). The fuzz modules are named `fuzz_*.py`, which that pattern does
not match:

- `pytest tests/fuzz -q` (and `pytest tests -q`, and `devtools test`'s
  underlying invocation) collect **0** tests from `tests/fuzz/`.
- Forcing collection with `-o "python_files=fuzz_*.py test_*.py"` collects
  418 tests (`TestParserFuzz`, `TestFTS5EscapeFuzz`, `TestPathSanitizerFuzz`,
  `TestTimestampFuzz`) that do pass when run this way.
- The only thing that *is* collected and run today is
  `tests/unit/sources/test_fuzz_targets_executable.py`, which — by its own
  docstring — only checks that each module still imports and exposes the
  documented target-function names + a `main()` referencing
  `atheris.Setup`/`atheris.Fuzz`. It does not execute any target against
  any input.
- Confirmed via `grep -rn "fuzz" .github/workflows/` that no CI workflow
  references `tests/fuzz` or atheris at all — consistent with the bead's
  premise that fuzzing "runs nowhere," but stronger than believed: even the
  *local* safety net the README describes isn't wired into default
  collection, so it was never running "on every commit" even before
  considering CI.

**Design proposal for the scheduled (not per-PR) wiring** (per the bead's
explicit read-only-slice scope, this is a design only — no workflow file
written):

1. **Fix the collection gap first, independent of CI.** Either (a) add
   `python_files = ["test_*.py", "fuzz_*.py"]` to
   `[tool.pytest.ini_options]` so `tests/fuzz`'s 418 pytest-mode cases join
   the normal suite (cheapest, restores the README's original claim, adds
   ~418 fast deterministic tests to the existing gate), or (b) rename the
   modules to `test_fuzz_*.py` (bigger diff, same effect, no ini change).
   Either is a small, mechanical, low-risk PR — not part of this read-only
   pass, but a natural immediate predecessor to the CI-scheduling work.
2. **New scheduled workflow** (e.g. `.github/workflows/fuzz-nightly.yml`,
   modeled on the existing `nightly-scale.yml` cron pattern already in this
   repo): `on: schedule` (e.g. weekly, matching the bead's "nightly/weekly,
   not per-PR" instruction) + `workflow_dispatch` for manual campaigns.
3. **Job body:** for each of the 4 target modules, run libFuzzer mode
   (`python tests/fuzz/fuzz_<name>.py -max_total_time=<bounded>`) inside the
   devshell (atheris is already a dev-dependency per
   `pyproject.toml [project.optional-dependencies] dev`). Use a per-target
   bounded wall-clock budget (the modules already accept
   `-max_total_time=N`), not iteration count, so total job time is
   predictable regardless of target speed.
4. **On-crash handling:** upload the libFuzzer crash/reproducer artifact
   (atheris writes a `crash-<hash>` file on failure) as a workflow artifact,
   and open or comment on a tracking issue (reuse the existing "convert
   anonymous debt into tracked debt" convention — one issue per target, not
   one per crash, to avoid issue-spam from a flapping target).
5. **Seed corpus:** `tests/fuzz/README.md` already documents seeding from
   real/sanitized provider fixtures rather than structureless bytes; confirm
   at implementation time that the referenced corpus directory exists and is
   wired into `atheris.Setup`'s corpus argument (out of scope to verify
   further here — the README section on "Seed Corpus" should be read
   alongside this proposal by whoever implements the execution half).
6. **Local entry point:** `devtools lab fuzz` (matching the
   `devtools lab test-economics` naming convention already established by
   `polylogue-9e5.11`'s shipped tooling) wrapping the same bounded-time
   invocation, so the scheduled CI job and local ad-hoc campaigns share one
   code path instead of the workflow YAML hand-duplicating target-invocation
   logic.

**Follow-up proposals:**
1. *"Fix `tests/fuzz` pytest collection gap (`python_files` pattern
   excludes `fuzz_*.py`)"* — small, immediate, unblocks everything else;
   currently the README's central safety-net claim is false.
2. *"Add scheduled (not per-PR) atheris fuzz-campaign CI workflow"* — the
   execution half per the design above (items 2-6); depends on #1 landing
   first so the local/CI code path can share the same collected-test
   surface rather than inventing a second invocation mechanism.

---

## Cross-cutting observations

- Beads `.11` and `.22` share exactly the coverage.json this pass used;
  `.11`'s tooling (`devtools lab test-economics`) is the more durable home
  for per-package coverage reporting going forward — `.22`'s
  per-file-worst-3 view is a complementary finer-grained cut the package
  tool doesn't currently produce, not a competing one.
- Bead `.20`'s biggest lesson is an infra gap (missing `git_head` on the
  dominant verify tier), not a test-quality finding — worth prioritizing
  ahead of building any ledger/marker machinery on top of it.
- Bead `.21`'s biggest lesson is a false-positive risk in naive AST
  classification (the `paths.*` idiom): any mechanical "worst offenders"
  scan over this codebase needs that idiom excluded or scored separately,
  or it will misdirect conversion effort at legitimate test-isolation code
  instead of real over-mocking.
- Bead `.18`'s biggest lesson is that a documented safety net (README's
  "runs in the normal test suite" claim) was not actually true under the
  current pytest config — worth a quick audit of other README/docstring
  claims about what "runs on every commit" elsewhere in the repo, though
  that's explicitly out of scope for this pass.
