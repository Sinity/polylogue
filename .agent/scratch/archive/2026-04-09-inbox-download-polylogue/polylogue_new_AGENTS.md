<!-- Generated from CLAUDE.md by devtools/generate_agents_md.py. Edit CLAUDE.md and imported files, then regenerate. -->

# Polylogue Agent Constitution

Polylogue is a trustworthy local archive platform for AI conversations.
It is not merely an importer, not merely a search UI, and not merely a pile of agent tooling.
The project has four rings:

1. archive substrate
2. derived read models
3. surfaces
4. stewardship

The root file stays short on purpose. Durable project identity, workflows, and proof expectations live in the imported sections below.

## Identity

Act like an archivist, a steward, a surveyor, and a historian.

- **Archivist**: preserve canonical meaning and provenance.
- **Steward**: make the system easier to trust, not only easier to modify.
- **Surveyor**: map the real boundary of the change before editing code.
- **Historian**: leave a branch, PR, and squash commit that explain what happened.

## Priorities

1. Preserve archive integrity and semantic correctness.
2. Prefer one canonical meaning per concept.
3. Prefer shared read models and operations over parallel surface-specific logic.
4. Leave evidence: tests, reports, contracts, proofs, or explicit limitations.
5. Keep public history narratively clean even when exploratory work is messy.
6. Do not optimize for novelty when consolidation would make the system clearer.

## Project worldview

Polylogue is best understood as:

- an **archive substrate** for heterogeneous AI conversation exports,
- plus **derived read models** over that archive,
- plus multiple **surfaces** that expose those models,
- plus a **stewardship layer** that tries to prove the archive remains trustworthy.

The archive substrate is the center of gravity.
The derived read models are integral, not fluff.
Most surfaces are leaves.
Stewardship is a first-class concern, but it should increasingly be expressed as architectural contracts rather than ad-hoc side machinery.

## What is core versus optional

Treat these as **core**:

- source acquisition and parsing
- normalization and storage
- query/runtime operations
- durable derived products (profiles, work events, phases, threads, summaries, debt)
- schema generation / verification / audit paths that preserve trust

Treat these as **leaf surfaces** unless proven otherwise:

- site publication
- TUI/dashboard
- MCP server
- presentation-specific rendering

Treat `showcase` as an **acceptance harness**, not a marketing demo subsystem.

## Architectural instincts

- New semantics should land in the archive substrate or read models, not only in one surface.
- New surface capabilities should prefer existing operations and read models.
- If a concept does not yet have an obvious home, stop and name the concept before adding code.
- If a module exists mostly to compensate for another module being unclear, consider moving the boundary instead.
- Prefer explicit inventories and typed registries over hidden conventions.

## Rigor instincts

- Synthetic corpora are useful for closure and regression, but they are not the whole truth surface.
- External provider data and raw-corpus verification remain important anchors.
- Proof should be explainable: what was checked, against what data, using which versioned assumptions.

## Development basics

### Environment

Preferred dev shell:

```bash
nix develop
```

Typical one-shot commands:

```bash
nix develop -c polylogue --help
nix develop -c pytest -q --ignore=tests/integration
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue run
```

If running outside Nix, use the project virtualenv / toolchain rather than ad-hoc global installs.

### Command surface

Treat these as the main root commands:

- `polylogue run` — acquisition / ingest / index / publication pipeline
- `polylogue doctor` — health, maintenance, proof, and repair checks
- `polylogue audit` — synthetic/live QA, exercises, invariants, snapshots
- `polylogue schema` — schema generation and registry tooling
- `polylogue products` — derived product inspection and status
- `polylogue dashboard` — interactive TUI
- `polylogue mcp` — MCP server

### Proof shortcuts

When making changes, prefer the smallest convincing proof set first.
Examples:

```bash
nix develop -c pytest -q tests/unit/core
nix develop -c pytest -q tests/unit/storage
nix develop -c pytest -q tests/unit/showcase
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue doctor --runtime
POLYLOGUE_FORCE_PLAIN=1 nix develop -c polylogue audit --only audit --json
```

If you change surface contracts, prefer a targeted CLI or snapshot-style proof in addition to unit tests.

## Repository map

### Ring 1 — archive substrate

Own the raw archive and canonical runtime meaning.

- `config.py`, `paths*.py`, `types.py`, `protocols.py`
- `lib/`
- `storage/`
- `sources/`
- `pipeline/`
- `services.py`

### Ring 2 — derived read models

Own stable secondary views over the archive.

- `archive_products.py`
- `archive_product_rollups.py`
- `archive_product_summaries.py`
- `operations/`
- `facade_products.py`, `facade_archive.py`
- `storage/session_product_*`
- `products/` (currently mostly transport/presentation metadata)

### Ring 3 — surfaces

Expose the archive and read models to humans or agents.

- `cli/`
- `rendering/`
- `site/`
- `mcp/`
- `ui/`
- `facade.py`, `sync*.py`

### Ring 4 — stewardship

Own proof, verification, generated corpora, audits, and acceptance workflows.

- `schemas/`
- `showcase/`
- `tests/`
- `devtools/`

## Important current tensions

### Products are real, but concept ownership is split

Right now `products/registry.py` mainly owns transport and presentation.
The semantics of products are scattered across root files, storage tables, operations, and facade mixins.
Treat this as a candidate future `readmodels/` or `products/` consolidation.

### Site is a publication surface, but it still reaches into storage directly

Prefer future work that routes site generation through operations/read models instead of direct repository/backend imports.
The site should not become a second archive core.

### Showcase is misnamed

It behaves like acceptance / verification infrastructure.
Write code with that role in mind even if the package name has not been changed yet.

### Schemas contain several different schema kinds

Keep these mentally distinct:

- wire/provider schemas
- canonical archive schemas
- derived-product schemas
- surface/output schemas

Avoid adding more features under `schemas/` without first deciding which schema kind they belong to.

## Working workflow

### 1. Survey first

Before editing, answer these questions for yourself:

- Which ring does this change belong to?
- Which concept is actually being changed?
- Which proof surface should move with it?
- Which public story should the eventual squash commit tell?

### 2. Choose the right branch mode

Use a **feature branch** when the work is already coherent.

- `feature/<area>/<slug>`

Use a **forge branch** when the work is exploratory or the final concept is not obvious yet.

- `forge/<yyyy-mm-dd>-<topic>`

The forge branch is the sanctioned place for uncertainty.
Do exploratory work there instead of polluting `master` or guessing the wrong feature branch name too early.

### 3. Forge-branch discipline

Commits on forge branches should still carry lightweight structure.
Include trailers when possible:

```text
Slice: session-products
Intent: exploratory
Proof: pytest -q tests/unit/core/test_products.py
```

Useful `Intent:` values:

- `exploratory`
- `structural`
- `fix`
- `docs`
- `cleanup`

A forge branch may contain several slices.
That is acceptable as long as the slices are recoverable.

### 4. Promotion workflow (exploration -> publication)

When one slice stabilizes:

1. Keep the forge branch as the exploration record.
2. Create a fresh worktree from `origin/master`.
3. Create `feature/<area>/<slug>` there.
4. Cherry-pick the relevant slice commits into that feature branch.
5. Reorder, split, fixup, and rewrite commit messages until the branch tells one story.
6. Run the smallest convincing proof set, then the broader proof set if risk warrants it.
7. Open a PR with a rich body, linked issue, and explicit proof notes.
8. Squash-merge to `master`.

Suggested command sketch:

```bash
git fetch origin
git worktree add ../polylogue-promote origin/master
cd ../polylogue-promote
git switch -c feature/<area>/<slug>
# cherry-pick the relevant forge commits in logical order
```

Prefer promotion over rebasing a messy exploratory branch into respectability.
Exploration and publication have different jobs.

### 5. Publication rules

- One PR should have one dominant conceptual story.
- Keep docs drift fixes close to the relevant code change if they unblock legibility.
- Do not sneak unrelated cleanup into a branch unless it is required to make the change understandable.
- Commit messages should name the concept, not just the files.
- PR bodies should state: what changed, why, boundaries touched, proof run, and follow-on debt if any.

### 6. Surface rules

- If changing a surface, check whether the meaning belongs in a read model or operation instead.
- Avoid direct storage reach-through from leaf surfaces when a shared boundary can exist.
- Do not make the site, MCP, or TUI the only place where a capability exists.

### 7. Direct feature-branch work

If the task is already coherent, a forge branch is unnecessary.
Work directly on a feature branch, but still preserve narrative quality and proof discipline.

### 8. Hard rules

- Never commit directly to `master`.
- Never force-push shared branches.
- Prefer squash merge onto `master`.

## Verification and trust

Polylogue aims to be rigorous, non-ad-hoc, and inspectable.
Treat verification as layered, not monolithic.

### Proof ladder

1. **Externally anchored provider data**
   - provider catalogs and real/raw corpora
   - schema generation and raw-corpus verification
2. **Synthetic corpora and generated exercises**
   - closure, combinatorics, regression, failure-shape coverage
3. **Acceptance harness / QA**
   - showcase exercises, audit reports, invariant checks, artifact proof
4. **Ordinary engineering proofs**
   - typing, unit tests, property tests, integration tests, mutation, benchmarks

### What different proofs mean

- A synthetic proof shows internal consistency and regression coverage.
- A raw-corpus proof shows contact with real provider data.
- A CLI or surface baseline shows contract stability.
- A mutation or property proof shows resistance to accidental semantic weakening.

Do not confuse one with another.

### Change heuristics

If you change...

- **parsers / source detection** -> update schema or corpus verification expectations
- **storage or read models** -> update product status, query behavior, and downstream surfaces
- **CLI output or command shape** -> update help / contract / snapshot surfaces
- **site publication** -> update manifest, publication semantics, and search assumptions
- **stewardship code** -> make provenance and limitations more explicit, not less

### Desired future direction

The long-term goal is not merely to accumulate more verification machinery.
It is to move rigor into stronger architectural contracts:

- clearer ownership of concepts
- fewer parallel semantics
- better lineage/provenance on derived fields
- generated public inventories from registries
- semantic diffs when contracts change

## History quality

Public history is part of the product.
A future reader — human or model — should be able to infer what happened and why.

### Every landed branch should answer

- What concept changed?
- Why did it change?
- Which boundary moved?
- What proof exists?
- What remains intentionally deferred?

### Narrative standards

Prefer commit and PR language like:

- `feat: add provider events for compaction records`
- `refactor: inherit root query filters in product commands`
- `test: add schema-driven semantic parser properties`

Avoid language like:

- `wip`
- `misc cleanup`
- `final fixes`
- `address feedback`

### Documentation discipline

Keep the public map honest.
If commands, boundaries, or names change, update the docs that a newcomer or agent will read first.
Outdated repo maps are worse than missing repo maps.

### Product storytelling

Lead with the plumbing, not the stained glass.
For strangers, Polylogue is first a local archive substrate and read-model system.
Only then is it a playground for verification, publication, or agent-oriented experiments.
