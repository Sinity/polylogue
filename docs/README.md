# Polylogue Docs Map

This directory holds the current repository-facing documentation. Keep it
focused on material that is still actionable in the live codebase.

## Core References

- [architecture.md](./architecture.md) - repository architecture overview
- [cli-reference.md](./cli-reference.md) - generated CLI reference from live help output
- [configuration.md](./configuration.md) - configuration and environment
- [data-model.md](./data-model.md) - archive/storage data model
- [generate.md](./generate.md) - synthetic archive generation workflows
- [internals.md](./internals.md) - internal mechanics and code navigation
- [library-api.md](./library-api.md) - Python API surface
- [mcp-integration.md](./mcp-integration.md) - MCP integration notes
- [providers/README.md](./providers/README.md) - provider-specific docs
- [test-quality-workflows.md](./test-quality-workflows.md) - generated validation lanes, mutation campaigns, and benchmark campaigns
- [mutation-testing-baseline.md](./mutation-testing-baseline.md) - mutation workflow and baseline policy

## Local Artifact Convention

- [../artifacts/README.md](../artifacts/README.md) - local generated artifacts kept out of version control

Mutation and benchmark campaign outputs default to ignored local artifact paths
under `artifacts/` rather than committed `docs/` trees.

Historical planning corpora from the pre-rewrite repo are intentionally not
treated as live documentation here. Restore them selectively if they become
needed again, instead of leaving dead links in the default docs map.
