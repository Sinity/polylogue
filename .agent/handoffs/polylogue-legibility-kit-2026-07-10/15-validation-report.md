# Validation report

This ledger separates what was executed from what remains a proposal. All commands ran against the supplied July 10, 2026 repository snapshots.

## Repository patch integrity

The Polylogue patch is based on `f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51`. The Sinex patch is based on `b70a08d9e51cecc9e76b92f955a88183e99842cf`.

For each project, a fresh detached worktree was created at the exact base commit. `git apply --check` succeeded, the patch was applied, and `git diff --check` succeeded afterward. The Polylogue patch changes 30 paths as represented by Git status, including the binary demo GIF; the Sinex patch changes nine paths.

## Polylogue checks executed

The focused validation set covered every production or test module changed by the patch:

```text
Ruff lint                         passed
Ruff format check                 passed
strict MyPy                       passed
focused pytest suite              41 passed in 17.34s
documentation-surface renderer    passed
static Pages renderer             passed
75-document command verifier      passed; no stale commands
rendered-site local-link check    passed
public-doc link/claim check       passed
git diff --check                  passed
```

The focused pytest set was:

```text
tests/unit/cli/test_demo_command.py
tests/unit/devtools/test_render_docs_surface.py
tests/unit/devtools/test_visual_vhs.py
tests/unit/site/test_renderer_behavior.py
```

The patched `polylogue demo tour` also ran successfully against a newly generated private-data-free archive. Its proof packet records:

```text
status                            passed
sessions                          11
messages                          43
assertions                        5
declared constructs               30/30 satisfied
absolute-path leaks               0
first evidence result             2.576 s
complete four-step tour           6.849 s
```

Its four user-facing steps are:

1. structural failure receipt;
2. failed-action aggregation from typed `is_error` fields;
3. composed lineage with inherited refs;
4. archive facets across origins.

The complete output is in [`polylogue-demo-tour/`](polylogue-demo-tour/report.md). The report itself distinguishes the deterministic product-contract proof from scale, universal provider fidelity, task-uplift, invoice, deletion, and Sinex-backend claims.

## Sinex checks executed

The Sinex patch changes documentation and comments but no runtime behavior. Static validation covered:

```text
public Markdown files             7 checked
relative links                    47 checked
public claims                     9 unique IDs parsed
roadmap-link policy               passed
git diff --check                  passed
```

The check also verifies that public planning language points to Beads rather than GitHub Issues.

## Machine-readable artifact checks

The following parsed successfully:

- both public claims ledgers;
- the demo portfolio;
- the Demo Packet v2 example;
- the Demo Packet v2 JSON Schema;
- the Beads launch-cut CSV, containing 28 rows;
- the worktree-lanes CSV, containing 16 rows.

The Demo Packet v2 example validates against the supplied Draft 2020-12 JSON Schema.


## Package integrity checks

The package-level checker resolved 67 package-relative Markdown links. The two proposed README artifacts were excluded from that package-relative pass because their links intentionally resolve inside their target repositories; those links were checked when rendering the patched repository surfaces.

The public-artifact scrub found no obvious execution-environment paths, access tokens, private-key markers, or credential-shaped strings in public narrative, visual, machine-readable, or proof artifacts. Operational fork prompts intentionally retain session artifact-directory output paths because they are designed to be launched as parallel ChatGPT forks in this environment.

The worktree bootstrap shell script, artifact-scrub shell script, GIF renderer, and deck generator all passed syntax checks. The package contains 16 primary and 16 alternate fork-prompt files.

## Presentation and image checks

The executive deck contains 13 PowerPoint slides and the PDF contains 13 pages. PDF preflight reports that it is openable, unencrypted, non-XFA, and not likely scanned.

Rendered image dimensions:

```text
Polylogue landing page            1440 × 916
Sinex landing page                1440 × 955
Resume This Bead surface          1440 × 975
Polylogue tour first frame        1080 × 720
```

The deck contact sheet and all four named images were visually inspected for clipping, overlap, unreadable type, and accidental private material.

## Explicitly not verified

This work does **not** claim:

- a full Polylogue test-suite pass;
- a Sinex Rust compile, Nix evaluation, service deployment, PostgreSQL migration, or NATS run;
- implementation of the proposed Incident 14:32 cross-project corpus;
- implementation of the full semantic transcript renderer;
- implementation of the Sinex-backed Polylogue data plane or rebuild proof;
- general agent-memory, retrieval, or task-success uplift;
- complete provider fidelity at all historical format variants;
- selective physical deletion across every material, projection, vector, cache, and report.

Rust, Cargo, Nix, PostgreSQL, and NATS were unavailable in the execution environment, so the Sinex contribution is a statically checked legibility and architecture patch rather than a runtime patch.

## Reproduction after patch application

Polylogue’s repository-native commands should be run from the patched checkout:

```bash
uv run ruff check <changed Python files>
uv run ruff format --check <changed Python files>
uv run mypy <changed Python files>
uv run pytest \
  tests/unit/cli/test_demo_command.py \
  tests/unit/devtools/test_render_docs_surface.py \
  tests/unit/devtools/test_visual_vhs.py \
  tests/unit/site/test_renderer_behavior.py
uv run python -m devtools render docs-surface
uv run python -m devtools render pages
uv run python -m devtools verify doc-commands
uv run python -m devtools render pages --check
```

Sinex should additionally receive its repository-native Rust, Nix, schema, and service checks on a machine with the declared toolchain and local services.
