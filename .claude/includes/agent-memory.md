## Agent Memory

Keep these repo facts loaded while working in Polylogue:

- The archive substrate and the derived read models are the core. Most user-facing surfaces are adapters over them.
- Preserve archive meaning and provenance before optimizing presentation or convenience.
- Put shared semantics in the substrate or read-model layer before adding surface-specific logic.
- Treat `showcase` as an acceptance harness, not as a separate product area.
- Keep `python -m devtools` as the single repo-maintenance entrypoint.
- Prefer generated docs from registries and live help output over duplicated handwritten summaries.
- Leave evidence for changes through tests, reproducible commands, or explicit limits.

When deciding where code belongs:

- acquisition, parsing, normalization, storage, and query logic belong in the archive substrate
- profiles, work events, phases, threads, and summaries belong in derived read models
- CLI, MCP, site, and UI code should reuse those lower layers
- verification, generated docs, and repo hygiene belong in `devtools/`, `showcase/`, or `tests/`

When in doubt:

- map the archive-facing boundary before editing
- check whether the concept already exists in a lower layer
- prefer explicit registries and typed contracts over hidden conventions
