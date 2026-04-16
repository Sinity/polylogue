---
created: "2026-04-12T23:58:00+02:00"
purpose: "Brainstorming note: existing capability fragments in Polylogue, schema-inference as proto-distillation, and lessons from Sinex xtask"
status: "active"
project: "polylogue"
---

# Existing Capabilities and xtask Lessons

## Main correction

Several of the previously proposed “future” capabilities do already exist in
some form.

The right move is not to invent a greenfield verification/meta layer.
The right move is to identify the existing capability fragments and decide
which one becomes the semantic root.

## Existing fragments already present in Polylogue

### 1. Real-data distillation already exists in proto form

The schema/inference operator surface is already doing the first half of
distillation:

- `polylogue/schemas/operator_inference.py`
  - `infer_schema(...)`
  - clustering of real samples
  - saved cluster manifests
- `polylogue/schemas/schema_inference.py`
  - structural inference
  - semantic-role inference
  - relational inference
  - sample loading from DB / sessions

So the previous “distill real data into synthetic spec” idea should not become a
parallel tool first.

It should likely extend this surface:

- current output:
  - provider schema package
  - cluster manifest
- desired future output:
  - `CorpusSpec`
  - pathology profile
  - provider-mix / topology distributions
  - feature density profile
  - giant-record and malformed/tolerated shape summaries

In other words:

- current schema inference ~= **artifact/schema distillation**
- future scenario distillation = **artifact/schema distillation + workload/assertion distillation**

### 2. Exercise generation already has meaningful introspection

Polylogue already has:

- `polylogue/showcase/generators.py`
  - CLI command-path introspection
  - generated command-help exercises
  - curated JSON-contract exercise generation
  - generated filter exercises
- `polylogue/showcase/dimensions.py`
  - multi-axis metadata

So “scenario compiler from introspection” is not speculative.
There is already a partial compiler.

The issue is that it currently compiles into `Exercise` too early.

### 3. Quality registry already exists in proto form

`devtools/quality_registry.py` already aggregates:

- validation lanes
- mutation campaigns
- benchmark campaigns

But it is an aggregation layer, not a semantic one.

The likely better future is:

- `ScenarioRegistry` / `CapabilityRegistry` becomes semantic root
- `QualityRegistry` becomes a rendered/projection view

### 4. Benchmark surface is already split into two forms

There is already a revealing duality:

- `devtools/benchmark_campaign.py`
  - durable artifacts
  - compare/baseline surface
  - mostly pytest-benchmark-domain oriented
- `devtools/benchmark_campaigns.py`
  - more realistic archive/workflow style runs

This split is not necessarily bad.
It suggests a distinction:

- micro/domain benchmark compiler target
- scenario/operator benchmark compiler target

What is missing is the common model feeding both.

## Lessons from Sinex xtask

Sinex’s `xtask` is useful here not because Polylogue should imitate Rust, but
because it shows what a strong control plane feels like.

Important patterns:

### 1. One obvious control plane

`xtask` is clearly the place where:

- verification policy
- exercise catalog
- benchmark contracts
- command introspection
- docs generation

all live together.

Polylogue is close, but not there yet:

- some of that is in `devtools`
- some in `polylogue/showcase`
- some in product CLI space

### 2. Exercises are treated as surface validation, not core ontology

In Sinex:

- `xtask exercise` is a strong execution/reporting surface
- it has tiers, audit baselines, regressions, manifests

But the command is very obviously tooling/control-plane territory.

That is a good clue for Polylogue:

- keep “exercise” as a strong execution/report/report-baseline concept
- but move the semantic ownership upward into the control plane

### 3. Baseline manifests are valuable

Sinex’s `QaManifest` idea is strong:

- small
- deterministic
- behavioral
- no volatile timings or paths

Polylogue should likely adopt an equivalent for:

- scenario pass/fail baselines
- maybe selected machine-contract surfaces

This is higher leverage than storing raw text output for everything.

### 4. Command catalog introspection should feed more than docs

Sinex has a real command catalog derived from clap introspection.
Polylogue already has pieces of this in:

- `devtools/command_catalog.py`
- `polylogue/cli/command_inventory.py`

The next step is to use command metadata for:

- help coverage
- JSON contract coverage
- completion metadata
- machine-surface audits
- maybe operator-map generation

## Stronger unifying idea than “scenario registry” alone

The likely root model is actually a small family:

- `CapabilitySpec`
  - what the system claims it can do
- `ArtifactSpec`
  - what durable states/artifacts exist
- `ScenarioSpec`
  - corpus + workload + assertions + presentation

This would let the repo model:

- command surfaces
- pipeline stages
- derived products
- indexes
- repairs
- exercises
- benchmarks

with one graph instead of parallel registries.

## Useful maps to generate

The system could likely auto-generate several high-value maps.

### 1. Artifact/dependency map

Nodes:

- raw payloads
- normalized conversations/messages/attachments
- action events
- session products
- aggregate products
- FTS/vector indexes
- render/site outputs

Edges:

- produced by
- invalidated by
- consumed by
- repaired by

### 2. Capability map

Rows:

- query
- run/acquire/parse/materialize/render/index
- doctor/repair
- schema/inference
- products
- qa/showcase
- benchmark/probe

Columns:

- user CLI
- machine CLI
- facade API
- MCP
- devtools

This would immediately show duplicated or missing surfaces.

### 3. Verification coverage map

For each capability/artifact pair:

- law/invariant tests
- scenario tests
- benchmarks
- live canaries
- showcase/QA

This is the “coloring inside the map” view.

### 4. Cost map

Per workload or scenario:

- wall
- RSS
- main scaling variable
- dominant artifact class
- current budget

This would turn performance work into something navigable.

## Concrete high-leverage direction

The best near-term architectural move is probably:

1. keep `devtools` as the control plane
2. extend the existing schema/operator inference surface toward corpus distillation
3. define a shared scenario/capability model in `devtools`
4. compile existing exercise/benchmark/lane surfaces from that model gradually

That preserves current investment instead of restarting the design.

## Non-goals

Avoid:

- replacing all existing registries at once
- building a giant generic framework before the first concrete slice
- moving everything under `polylogue` just for conceptual neatness

The repo already has enough machinery. The challenge is semantic unification.
