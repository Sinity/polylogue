# Test Diet 01 handoff — query algebra and cardinality survivor

## Result

This package implements the strongest coherent local production-route survivor supported by the supplied snapshot. It adds an independent planted-fact oracle, four real-route laws, and one production correction discovered by those laws.

The correction is in the root CLI's non-search session-list branch. The branch compiled explicit Boolean expressions into `filter_kwargs["boolean_predicate"]`, but then manually forwarded every other filter to `ArchiveStore.list_summaries` and omitted that predicate. Consequently, a bare expression such as `sessions where title:td01-delete-cohort AND origin:codex-session` rendered an unfiltered page even though facade list/count and terminal actions used the intended selection. The patch now sends the already-typed filter map to `list_summaries`, preserving all existing filters and forwarding the Boolean predicate only when it exists.

The survivor begins with the repository's realized C-03 provider-native acquire/parse/index canary, clones its receipt-backed archive, and appends a separate known-fact manifest through the production archive writer. Expected membership, action rows, counts, partitions, pages, and destructive targets are computed only from that manifest; no production query path supplies its own expected answer.

## Snapshot identity and authority

| Field | Observed identity |
| --- | --- |
| Snapshot source | `/realm/project/polylogue` |
| Snapshot generation | `2026-07-17T043202Z` |
| Manifest branch | `master` |
| Commit / patch base | `f654480cadb7cc4c194704e24dfd483199547b35` |
| Reconstructed checkout | detached at the same commit |
| Branch delta | empty against `origin/master`; merge base is the same commit |
| Manifest dirty flag | `true` |
| Captured tree comparison | captured working-tree overlay produced no Git diff against the named commit |

The commit plus captured working tree is therefore the patch authority. The manifest's dirty flag conflicts with the content comparison; that contradiction is retained as evidence rather than silently normalized.

## Evidence inspected before implementation

Repository instructions and contracts inspected: `AGENTS.md` (symlink to `CLAUDE.md`), `CLAUDE.md`, `TESTING.md`, `CONTRIBUTING.md`, and `pyproject.toml`. These establish substrate-first semantics, the single Lark query grammar, strict typing, and focused managed verification.

Production paths followed through their dependencies:

- `polylogue/archive/query/expression.py`, `spec.py`, `plan.py`, `archive_execution.py`, and `unit_results.py`;
- `polylogue/cli/archive_query.py`, `query.py`, `click_app.py`, `query_verbs.py`, and `verb_cardinality.py`;
- `polylogue/api/` facade entry points and public query envelopes;
- `polylogue/storage/sqlite/archive_tiers/archive.py` and `polylogue/storage/sqlite/action_relation.py`;
- source parser models, archive writing, canonical identities, and result payloads needed to plant facts through production types.

Current tests and helpers inspected:

- `tests/unit/cli/test_query_expression.py`;
- `tests/unit/cli/test_query_exec_laws.py`;
- `tests/unit/cli/test_verb_cardinality.py`;
- `tests/unit/storage/test_archive_tiers_archive.py`;
- query security, parity, identity, miss-diagnostic, and runtime-filter tests;
- `tests/infra/query_cases.py`, `surfaces.py`, `semantic_facts.py`, and `workload_artifacts.py`.

The supplied Test Diet packet was also read broadly, especially `areas/query-composition.md`, `dossiers/exact-query-selection.md`, `11-test-proof-form-audit.md`, `08-capability-map.md`, and `test-suite-composition-and-scale-2026-07-16.md`. Its query dossier names commit `21f78b4db2ba62ff44b5f16dfab96067bc249b4c` and reports zero realized artifacts, so current source at `f654480...` supersedes stale path/API assumptions while preserving the dossier's behavioral obligations.

Relevant Beads records inspected: `polylogue-1xc.14.1`, `polylogue-b054.1.1.4`, `polylogue-xnkf`, `polylogue-yeq.3`, `polylogue-z9gh.2`, `polylogue-fnm`, `polylogue-z9gh.9.1`, and `polylogue-70qb`.

Relevant history inspected:

- `5d5edaf496d70b9372c7c2123c2e70a6ed4d34e6` — earlier Boolean-query repair;
- `7b5a5aa0589772ed9d7eb9632af1591cf22cdd33` — rank-pair actions by transcript order;
- `9f0e77116280571c14430b47986e7bd5b0e8be1a` — closure evidence for duplicate action fanout;
- `c6495ea29c92d54dae00b302cb17336d80b5162d` — shared receipt-backed query canary;
- `c20286459cf2c3d1e4c968a8584f13e7cd382ff2` — workload profiles and seeded real-pipeline artifacts;
- `89166362b9aee8c304b27a69f68ec1b74606f634` — production query evaluator;
- `478d6a77cf1cea1d63c180701d49099ad2cfacc3` — exact-session temporal bound.

## Mechanism

### Independent fact model

`tests/infra/query_manifest_oracle.py` is standard-library-only and deliberately imports no Polylogue parser, query spec, SQL relation, repository reader, or public payload type. It declares sessions and actions, then derives expected values directly:

- canonical session IDs and exact origin/title membership;
- expected action identities and nullable result fields;
- `is_error` partitions;
- deterministic offset pages.

The fixture plants these adversarial shapes:

- two `Bash` uses and two results sharing one `tool_id` in transcript order;
- a third `Bash` use with no result;
- the same repeated `tool_id` in another session;
- two Codex delete candidates plus a Claude title-shadow candidate;
- identical native IDs under Codex and Claude canonical origins;
- a retained control session.

### Real route

`tests/unit/cli/test_query_composition_laws.py` builds the existing C-03 artifact with `build_seeded_archive`, clones it with `clone_seeded_archive`, validates the receipt identity, and plants the independent facts with `ArchiveStore.write_parsed`. The laws then traverse:

1. Lark parsing and query lowering;
2. canonical SQLite action/session relations and repository list/count execution;
3. the asynchronous `Polylogue` facade;
4. the root CLI JSON list route;
5. CLI delete preview and apply;
6. post-action facade reads.

The CLI daemon fast path explicitly declines compiled Boolean predicates, so the repaired law exercises the authoritative local route. It does not claim daemon, HTTP, MCP, browser, or deployed-process coverage.

### Cardinality and mutation sensitivity

The canonical action relation ranks uses and results within `(session_id, tool_id)` by transcript order and joins equal ranks, preserving an unmatched use through a left join. The test's representative equality-join mutant creates a temporary plain relation and replaces the production relation selector only for the mutation witness. Two uses multiplied by two results produce four rows, plus the unmatched use for five; the production relation and independent oracle require three.

A second reversible mutation was applied in an isolated worktree to remove `boolean_predicate` immediately before the repaired `list_summaries` call. The session law failed with the default 20-row unfiltered page instead of the two oracle members. The leaked rows included the planted Claude shadow, both canonical-origin variants of `td01-shared`, C-03 irrelevant sessions, and unrelated planted controls. This is the specific wrong result the survivor detects.

## Changed files

| File | Change |
| --- | --- |
| `polylogue/cli/archive_query.py` | Replace a duplicated manual filter forwarding list with the existing typed `filter_kwargs`, thereby forwarding compiled Boolean predicates and reducing future call-site drift. No public signature or query-language change. |
| `tests/infra/query_manifest_oracle.py` | Add the independent known-fact manifest and pure expected-result evaluator. |
| `tests/unit/cli/test_query_composition_laws.py` | Add four real-route laws covering membership/count/identity, action rows/count/partitions/pages, preview/apply, and duplicate-join mutation sensitivity. |

No existing test or helper was deleted. `FILES/` is intentionally omitted because `PATCH.diff` completely and unambiguously carries all three changes.

## Acceptance matrix

| Mission obligation | Result | Evidence |
| --- | --- | --- |
| Start from current realized canary | PASS | Fixture builds and clones the current C-03 provider-native receipt-backed artifact. |
| Independent expected fact set | PASS | Standard-library manifest computes answers without production query imports. |
| Parsing and lowering | PASS | Session AST/compiled predicate and action terminal-source parsing are asserted. |
| SQL/repository execution | PASS | `query_actions`, `query_unit_counts`, session list/count, and canonical relation execute against SQLite. |
| Stable public read routes | PASS | Python facade and root CLI JSON list agree with the oracle. |
| Membership/count agreement | PASS | Exact canonical IDs and totals agree across facade, repository, and CLI. |
| Partition agreement | PASS | `group by is_error | count` equals the oracle partition and sums to total. |
| Pagination agreement | PASS | Two facade pages concatenate exactly to the stable unpaged order and oracle slices. |
| Preview/apply agreement | PASS | CLI dry-run IDs equal the oracle; apply count matches; post-state proves only those IDs disappeared. |
| Duplicate/missing-result cardinality | PASS | Repeated IDs rank-pair to two rows, unmatched use remains a third nullable row, cross-session decoy is excluded. |
| Representative anti-vacuity mutation | PASS | Dropped-predicate production mutant fails; plain equality join produces five instead of three. |
| Preserve exact identity | PASS | Same native ID under two origins resolves only the fully canonical Codex ID. |
| Preserve existing bounded-work canary | PASS | Complete owning test file run includes both current C-03 canaries. |
| Preserve parser/security/compatibility obligations | PASS for inspected cluster | Focused parser/CLI/storage cluster plus security, parity, identity, diagnostics, and runtime filters are green. |
| Full repository suite | UNVERIFIED | Not blanket-run; repository instructions explicitly prefer focused ownership selections. |
| Locked Nix/uv environment | UNVERIFIED | External package resolution was unavailable; verification used host Python plus declared-version dependencies. |
| Live daemon/HTTP/MCP/browser/deployment | UNVERIFIED | No operator process, secrets, or deployment was available or claimed. |
| Real testmon affected-selection proof | UNVERIFIED | Direct production mutation was proved, but the separate `polylogue-b054.1.1.4` testmon-gate obligation remains open. |

## Apply order

Apply the patch atomically against the exact snapshot commit:

```bash
git checkout f654480cadb7cc4c194704e24dfd483199547b35
git apply --check PATCH.diff
git apply PATCH.diff
```

The patch has no migration, generated-file, or data-order dependency. The production forwarding repair and its survivor tests should land together.

Recommended focused verification after application:

```bash
python -m ruff format --check \
  polylogue/cli/archive_query.py \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py
python -m ruff check \
  polylogue/cli/archive_query.py \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py
python -m mypy --strict \
  polylogue/cli/archive_query.py \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py
python -m devtools test tests/unit/cli/test_query_composition_laws.py -p no:randomly
```

Then run the owning files and query-contract files listed in `TESTS.md` through the project's managed environment.

## Patch reproducibility

`PATCH.diff` was generated from the three named paths only, checked with `git apply --check`, and applied to a fresh detached worktree at the exact snapshot commit. The resulting files byte-match the implementation worktree. Fresh-worktree `git diff --check`, Ruff format/check, `compileall`, strict changed-file mypy, and the four managed survivor laws all pass.

## Risks and limitations

The production change is intentionally narrow, but it centralizes an existing call onto `filter_kwargs`. Existing owning tests cover the old forwarding behavior; nevertheless, a release environment should rerun the full affected/testmon selection under the locked toolchain.

The new real-route laws cover SQLite, the Python facade, and local CLI read/action behavior. They do not make claims about HTTP or MCP transaction semantics, snapshot isolation across concurrent pages, cursor replay, cancellation, FTS/vector/lineage scale branches, or live archive resource ceilings. Those are separate open Bead obligations and would be unsafe to simulate as part of this local survivor.

Raw focused pytest required `-p no:randomly` because the available host's `pytest-randomly` setup raises `ValueError: Seed must be between 0 and 2**32 - 1` before the test body. The managed command with that explicit disablement passed. The workload helper emits Python 3.13 `fork()` deprecation warnings; no semantic failure occurred.

A full `devtools verify --quick` did not complete in this environment. Ruff stages completed after temporary command wrappers were supplied; the full-repository mypy stage was externally stopped after 240 seconds. Changed-file strict mypy completed successfully. This package does not convert that incomplete run into a pass claim.

## Value of another iteration

A small repair iteration would add modest confidence: rerun this patch under the exact locked Nix/uv environment, resolve the local `pytest-randomly` bootstrap problem, execute the true testmon affected gate with the reversible predicate mutation, and optionally add explicit sequence-predicate parity to the same oracle. That is verification hardening, not a redesign.

A substantial second pass would be materially different work: extend the shared bounded query transaction across daemon HTTP and future MCP-native surfaces; prove snapshot-stable cursor/page concatenation under concurrent writes; add FTS/vector/lineage and cancellation laws; and attach selective-plan/resource receipts at p50/p95/max generated workload tiers. That work could add significant value, but it belongs with `polylogue-yeq.3`, `polylogue-z9gh.2`, and `polylogue-z9gh.9.1`, not as an ungrounded expansion of this patch.
