# Benchmark Campaign Artifacts

Use `nix develop -c python -m devtools.benchmark_campaign list` to see campaign definitions.
Use `nix develop -c python -m devtools.benchmark_campaign run <campaign>` to record a fresh artifact.
Use `nix develop -c python -m devtools.benchmark_campaign compare <baseline.json> <candidate.json>` to compare two artifacts.

| Date | Campaign | Commit | Benchmarks | Runtime | Worst Regression | Markdown |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-03-12 | `pipeline` | `d833643dccdc` | 17 | 15.49s | - | [2026-03-12-pipeline.md](./2026-03-12-pipeline.md) |
| 2026-03-12 | `search-filters` | `d833643dccdc` | 11 | 28.88s | - | [2026-03-12-search-filters.md](./2026-03-12-search-filters.md) |
| 2026-03-12 | `storage` | `d833643dccdc` | 11 | 16.79s | - | [2026-03-12-storage.md](./2026-03-12-storage.md) |
