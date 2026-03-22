# Polylogue Docs Map

This directory currently contains four different kinds of material:

1. operator and architecture docs
2. active planning/program docs
3. analysis inputs and audits
4. generated campaign artifacts

Repo-local generated working-state artifacts now live under
[`../artifacts/README.md`](../artifacts/README.md) rather than in unrelated root
directories.

The flat root makes those easy to confuse. Start from this file instead of
reading `docs/` as one undifferentiated list.

## Primary Product And Operator Docs

- [architecture.md](./architecture.md) - repository architecture overview
- [cli-reference.md](./cli-reference.md) - CLI command reference
- [configuration.md](./configuration.md) - configuration and environment
- [data-model.md](./data-model.md) - archive/storage data model
- [internals.md](./internals.md) - internal mechanics
- [library-api.md](./library-api.md) - Python API surface
- [mcp-integration.md](./mcp-integration.md) - MCP integration notes
- [generate.md](./generate.md) - generation workflows
- [providers/README.md](./providers/README.md) - provider-specific docs

## Planning And Analysis

- [planning-and-analysis-map-2026-03-21.md](./planning-and-analysis-map-2026-03-21.md) - canonical index for plans, backlog docs, audits, and scratch notes
- [programs/README.md](./programs/README.md) - program and execution-plan docs
- [analysis/README.md](./analysis/README.md) - audits, research reports, and raw design inputs
- [programs/intentional-forward-program-2026-03-21.md](./programs/intentional-forward-program-2026-03-21.md) - current live execution program
- [programs/artifact-cohort-control-plane-program-2026-03-21.md](./programs/artifact-cohort-control-plane-program-2026-03-21.md) - latest executed subprogram

## Generated Campaign Artifacts

- [mutation-campaigns/README.md](./mutation-campaigns/README.md) - mutation testing artifacts
- [benchmark-campaigns/README.md](./benchmark-campaigns/README.md) - benchmark artifacts

## Historical Note

The March 5-7 closure/tasklist cluster now lives under:

- [archive/2026-03-05-07-closure-wave/README.md](./archive/2026-03-05-07-closure-wave/README.md)

The planning map marks which materials are current, reference-only, raw input,
or historical.
