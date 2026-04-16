# Control-Plane Compiler Architecture

## Core Observation

Polylogue already has most of the right pieces for a unifying verification and
operator architecture, but they are spread across multiple projection systems.

Examples:

- `polylogue/cli/command_inventory.py` inventories the Click tree
- `devtools/command_catalog.py` inventories the control-plane commands
- `polylogue/showcase/generators.py` compiles command introspection into
  exercises
- `polylogue/showcase/dimensions.py` classifies exercises across several axes
- `devtools/validation_lane_base.py` defines lane projections
- `devtools/benchmark_campaign.py` defines benchmark projections
- `devtools/benchmark_campaigns.py` is an older scenario-ish synthetic campaign
  runner
- `polylogue/showcase/qa_runner_stages.py` composes QA-time dynamic exercises

This is not random sprawl. It is a half-formed compiler architecture.

The problem is that the compiler does not yet have a single source language.

## Strong Claim

The right unifying design is:

- author a small semantic model
- compile all these projection surfaces from it

Not:

- keep hand-maintaining each registry and adding glue between them

## The Source Language

The best source language currently appears to be:

- `ArtifactNode`
- `OperationSpec`
- `ScenarioSpec`

with one important nuance:

- `OperationSpec` should be heavily derived from operation annotations and
  command/service introspection rather than manually authored in parallel

So the actual authored truth should be kept as small as possible.

## Existing Projection Systems And Their Future Role

### 1. Click command inventory

Current files:

- `polylogue/cli/command_inventory.py`
- `polylogue/cli/click_command_registration.py`

Current role:

- recursive discovery of the public Click tree

Future role:

- one input into `OperationSpec`
- source of surface path, help, option metadata, JSON support, completion
  capability, and whether a command is product-facing

This should no longer be used only for help exercises.

### 2. Devtools command catalog

Current file:

- `devtools/command_catalog.py`

Current role:

- curated inventory of developer/control-plane commands

Future role:

- second input into `OperationSpec`
- explicit control-plane operation registry

In the final shape, `command_catalog` should probably itself be generated from
annotated devtools operations, or at least validated against them.

### 3. Showcase exercises

Current files:

- `polylogue/showcase/exercise_models.py`
- `polylogue/showcase/generators.py`
- `polylogue/showcase/dimensions.py`

Current role:

- executable and presentable CLI-focused proof artifacts

Future role:

- compiled `ExerciseSpec` / `ExerciseArtifact`
- dimensions remain valuable, but should become derived facets from
  `ScenarioSpec` + `OperationSpec`, not the top-level authored taxonomy

Important point:

`ExerciseDimensions` is already evidence that the system wants richer
multi-axis modeling. It should not be discarded. It should be repositioned as a
projection vocabulary.

### 4. Validation lanes

Current file:

- `devtools/validation_lane_base.py`

Current role:

- executable validation bundles

Future role:

- compiled execution plans over scenarios and operations
- should stop being authored as bare command tuples where a higher-level
  scenario exists

Lanes are a scheduling/selection surface, not the semantic source of truth.

### 5. Benchmark campaigns

Current files:

- `devtools/benchmark_campaign.py`
- `devtools/benchmark_campaigns.py`

Current role:

- two overlapping benchmark worlds:
  - pytest-benchmark domain campaigns
  - synthetic archive scenario campaigns

Future role:

- one compiled benchmark projection system
- both micro-ish domain benchmarks and scenario benchmarks should fit under one
  inventory and one artifact schema

This split is one of the clearest signs that the repo is missing a unifying
source language.

### 6. QA orchestration

Current files:

- `polylogue/cli/commands/qa.py`
- `polylogue/showcase/qa_runner*.py`

Current role:

- composite workflow that mixes data mode, proof stages, exercise generation,
  capture, and snapshotting

Future role:

- operator-facing compiled workflow over scenario bundles

The current command is useful, but too orchestration-heavy to remain a semantic
root.

## Compiler View

The architecture should be thought of explicitly as a compiler pipeline.

### Frontend

Inputs:

- artifact declarations from runtime code
- operation annotations / inventories from runtime and devtools surfaces
- schema/operator inference output
- authored scenario definitions

### Mid-level representation

- artifact graph
- operation graph
- scenario graph
- coverage map
- cost map

This mid-level representation is the actual unifying win.

### Backends / projections

- showcase exercises
- validation lanes
- benchmark campaigns
- QA stage plans
- docs inventories
- completions and machine discovery
- coverage and cost maps

## Why This Is Better Than More Glue

If we keep the current approach, every new feature wants:

- command inventory changes
- showcase changes
- lane changes
- benchmark registry changes
- docs refresh logic

That is manageable only while the system is small.

The compiler architecture reduces that to:

- annotate/declare artifact or operation
- add or extend a scenario if this needs proof/benchmarking
- regenerate projections

That is much closer to a self-describing system.

## What Should Be Generated

The following should become generated or compiler-checked as much as practical:

- command and capability inventories
- benchmark campaign inventories
- validation-lane inventories
- exercise inventories
- help/JSON-contract/completion matrices
- coverage maps
- artifact dependency maps

If these remain largely hand-authored, the architecture is not yet unified.

## What Should Stay Hand-Authored

- artifact semantics
- scenario intent
- some benchmark budgets
- some presentation metadata

Trying to infer everything would create opaque magic. The right move is:

- explicit semantics
- automatic projection

## First Concrete Compiler Slice

The best first slice is:

1. define the action-event artifact graph
2. annotate the operations that probe/project/repair it
3. author scenario classes for its failure modes
4. compile:
   - a law-oriented test family
   - one benchmark scenario
   - one QA/exercise projection if useful
   - one map output

This would test the whole architecture on a path that already spans:

- substrate
- derived read model
- FTS
- doctor
- debt
- repair

## Second Slice As Compiler Falsifier

Then apply the same compiler path to:

- raw decode / validation / quarantine / reparse

If the same source language still works, the architecture is probably real.
If not, correct the language before rollout.

## Long-Term Direction

The most elegant end state is:

- `polylogue` exposes product/archive semantics
- `devtools` is the control-plane compiler and operator surface
- schema/operator inference distills real archives into synthetic corpus specs
- showcase, QA, lanes, and benchmarks are compiled projections
- maps provide explicit, exhaustive reasoning support instead of tribal memory

That is a genuinely unifying architecture, not merely a simpler pile.
