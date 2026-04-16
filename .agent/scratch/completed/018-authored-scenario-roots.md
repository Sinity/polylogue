# Authored Scenario Roots

## Context

This note records the substantial unification work that landed after the earlier
scenario-metadata and inferred-corpus phases.

The goal was to stop carrying multiple subtly different authored models for:

- named scenario-bearing sources
- executable control-plane scenarios
- synthetic benchmark runner dispatch

## What Landed

Two commits define the current boundary:

- `4d06ff7e` `refactor: compile inferred corpus scenarios`
- `283b4bdf` `refactor: unify authored executable scenarios`

### 1. Inferred corpus specs are no longer the top-level projection source

The authored/inferred flow now has three levels:

- `CorpusSpec`
  - one inferred or authored synthetic corpus variant
- `CorpusScenario`
  - one compiled named scenario grouping one or more `CorpusSpec`s by provider/version
- scenario projections / quality registry / schema CLI output
  - consume `CorpusScenario`, not loose inferred specs

Key effect:

- schema/operator inference now emits first-class inferred scenario objects
- quality/docs/report surfaces refer to compiled inferred scenarios directly

### 2. Named scenario sources now share one root

`polylogue/scenarios/sources.py` now defines `NamedScenarioSource`.

It is the shared authored root for scenario sources that have:

- `name`
- `description`
- shared scenario metadata
- projection identity derived from those fields

It now underpins:

- showcase exercise scenarios
- mutation campaigns
- executable control-plane scenarios via a devtools-specific subtype

### 3. Executable control-plane scenarios now share one root

`devtools/executable_scenarios.py` now defines `ExecutableScenario`.

It is the shared authored root for scenarios that carry:

- `name`
- `description`
- `execution: ExecutionSpec | None`

and derives common runtime-facing helpers:

- `command`
- `tests`
- `members`
- `runner`
- `is_composite`
- `is_runner`

It now underpins:

- validation lanes
- benchmark campaigns

### 4. Execution substrate now has four explicit kinds

`ExecutionSpec` now models:

- `COMMAND`
- `PYTEST`
- `COMPOSITE`
- `RUNNER`

This removed a real semantic leak:

- pytest-backed validation lanes used to be authored as generic `COMMAND`
- synthetic benchmarks used to route through bespoke `runner_name` fields

Now:

- validation lanes keep `PYTEST` semantics when authored that way
- synthetic benchmarks use `RUNNER` execution explicitly
- durable benchmark runtime consumes `campaign.execution` directly

## Important Consequences

The authored shape is now much cleaner:

- scenario-bearing roots are no longer re-declared independently in each family
- execution semantics are no longer split across lane-only and benchmark-only logic
- projection identity is now mostly structural rather than hand-reimplemented

The remaining direct `ScenarioProjectionSource` implementors are now:

- `NamedScenarioSource`
- `CorpusSpec`
- `CorpusScenario`

That is the intended stop boundary for this phase.

## What Is Intentionally Still Separate

These are not currently treated as architectural seams:

- `CorpusSpec`
  - dynamic, provider/version/scoped synthetic corpus descriptions
- `CorpusScenario`
  - compiled grouping over corpus variants
- `Exercise`
  - compiled artifact, not an authored source
- `CampaignResult`
  - benchmark artifact payloads, not authored roots
- validation lane category catalogs
  - organizational partition, not a second semantic root now that they all author `LaneEntry`

## Next Real Phase

The next meaningful architecture step is not another small seam cleanup.

It would be one of:

1. introduce a richer authored `ScenarioSpec` above execution/corpus sources
   - so lanes, benchmarks, showcase, and QA compile from one scenario model
2. expand schema/operator inference from `CorpusScenario` into fuller scenario compilation
   - inferred corpora + inferred workloads/assertions
3. widen the artifact/operation graph beyond the proven paths
   - so scenario compilation can target more of the runtime substrate directly

Those are new architecture phases, not leftover cleanup from the current one.
