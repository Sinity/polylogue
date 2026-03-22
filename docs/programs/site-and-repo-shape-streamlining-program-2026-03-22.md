# Polylogue Site And Repo-Shape Streamlining Program

Date: 2026-03-22
Status: next planned execution slice
Role: concrete next program after the publication-control-plane implementation

## Why This Is Next

The publication-control-plane slice is now in place:

- typed `site` manifests
- durable `publications` records
- shared output-manifest scanning
- `site --json`

That means the remaining Step 5 work is no longer vague. It is now a clear
streamlining problem:

1. `polylogue/site/builder.py` is still a very large mixed-role module
2. the repo root still presents generated/runtime artifacts as peers of the
   codebase
3. publication/report surfaces still need a cleaner filesystem and operator
   topology around the now-correct control plane

Current evidence from the tree:

- `polylogue/site/builder.py` is `1363` LOC
- `polylogue/storage/repository.py` is `1206` LOC
- root generated/runtime directories still include:
  - `qa/` at about `47M`
  - `mutants/` at about `349M`
  - plus `dist/`, `result`, and `demos/`
- code proper is much smaller:
  - `polylogue/` about `4.6M`
  - `tests/` about `3.6M`
  - `docs/` about `4.3M`

So the next win is to make the code and repository shape match the cleaner
publication architecture that now exists.

## Goals

1. Make `site` internally legible by splitting scan/render/search/persist roles.
2. Quarantine generated/runtime artifact directories behind one intentional root.
3. Reduce repo-root overwhelm without cutting real verification capability.
4. Keep publication/report outputs on the shared control plane rather than
   growing more local ad hoc file conventions.

## Non-Goals

- schema package/version redesign as the main campaign
- `ConversationFilter` replacement
- large provider-parser rewrites
- removing showcase/QA as product surfaces

Those remain valid adjacent lanes, but they are not the sharpest next closure
move after the publication work.

## Program Order

### Step 1: Split `site/builder.py` By Role

Goal:

- reduce `site/builder.py` to a thin orchestration shell

Target internal seams:

- archive scan/read-model collection
- conversation page rendering
- index/dashboard rendering
- search materialization
- publication manifest persistence/materialization

Target shape:

- `site/builder.py` becomes orchestration only
- extracted modules carry the concrete mechanics
- shared dataclasses/models stay stable

Acceptance:

- builder becomes substantially smaller
- no duplicated repository/open-storage setup
- search and page rendering can be tested in narrower slices

### Step 2: Introduce One Generated-Artifact Root

Goal:

- stop presenting runtime/generated artifact directories as unrelated root peers

Target directories:

- `qa/`
- `mutants/`
- `result`
- any other runtime/generated output that is not source code or committed docs

Preferred shape:

- one explicit top-level quarantine such as `artifacts/`
- subtrees like:
  - `artifacts/qa/`
  - `artifacts/mutants/`
  - `artifacts/result/`

Rules:

- committed documentary evidence stays under `docs/`
- runtime/generated working-state outputs move out of the root clutter zone

Acceptance:

- root listing is meaningfully smaller and more readable
- commands/docs point at the new paths
- reset/cleanup flows know the new topology

### Step 3: Normalize Publication And Verification Output Conventions

Goal:

- make `site`, `qa`, and adjacent report surfaces feel like one family

Scope:

- manifest naming
- output-root naming
- durable-vs-ephemeral distinction
- CLI/documentation wording

Desired result:

- publication outputs are clearly materializations
- committed evidence is clearly documentary
- ephemeral/generated work products are clearly quarantined

### Step 4: Clean Repo Navigation Surfaces

Goal:

- make the repo read like one archive platform rather than code plus piles of
  artifacts

Scope:

- top-level README references
- docs references to artifact locations
- any stale `demos`/demo-era references that no longer reflect the real QA or
  showcase flow

Acceptance:

- the root and docs entrypoints explain where code, programs, evidence, and
  runtime artifacts live

## Implementation Constraints

### 1. Keep The New Publication Control Plane Canonical

Do not reintroduce loose local summary dicts or one-off manifest logic while
splitting `site`.

### 2. Treat Showcase As A First-Class Consumer

If output-location or manifest conventions are normalized, `showcase` must move
with them rather than being left as an older parallel world.

### 3. Prefer Moves And Collapses Over Compatibility Layers

This should be a cleanup program, not a “new path plus old path forever” plan.

### 4. Keep Durable Documentary Evidence In `docs/`

`docs/mutation-campaigns/` and `docs/benchmark-campaigns/` are documentary
evidence, not runtime scratch. Keep that distinction sharp.

## Suggested Commit Decomposition

1. `refactor: split site build orchestration from rendering and search`
2. `refactor: move runtime artifacts under a unified artifacts root`
3. `docs: normalize publication and generated-artifact topology`

## Exit Criteria

- `site/builder.py` is no longer a large mixed-role monolith
- runtime/generated artifacts are quarantined behind one clear root
- root-repo navigation is materially less overwhelming
- publication/report surfaces use one coherent location and naming policy
- docs and CLI surfaces describe the new topology directly
