## Docs Surface

- `README.md` is the public entrypoint. Keep it accurate, concise, and free of stale CLI or workflow examples.
- `CONTRIBUTING.md` is the public workflow contract for branches, commits, issues, PRs, and repo governance.
- `TESTING.md` is the public testing guide. Do not duplicate large test inventories elsewhere.
- `docs/architecture.md` is the system-shape reference and is transcluded directly into `CLAUDE.md`.
- `docs/internals.md` is the working landmark map and is transcluded directly into `CLAUDE.md`.
- `docs/devtools.md` is the reference for generated surfaces, validation lanes, and local repo hygiene.
- the generated command catalog inside `docs/devtools.md` is rendered from the live `devtools` registry.
- `docs/README.md` and the generated documentation table in `README.md` are rendered from the shared docs registry.
- `docs/cli-reference.md` is generated from live help output.
- `docs/test-quality-workflows.md` is generated from live validation, mutation, and benchmark registries.
- `AGENTS.md` is generated locally from this root `CLAUDE.md` include set and should be refreshed whenever the included files change.
- Favor one maintained source document plus generated views over parallel handwritten summaries.
