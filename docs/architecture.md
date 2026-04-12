# Polylogue Architecture

Polylogue is a local archive platform for AI conversations. The system has four rings:

1. archive substrate
2. derived read models
3. user and machine surfaces
4. verification and maintenance

## Rings

### 1. Archive Substrate

The substrate owns the archive's stored meaning:

- source acquisition and provider detection
- provider parsing and normalization
- SQLite persistence and search indexes
- archive-level query and runtime operations

Primary modules:

- `polylogue/sources/`
- `polylogue/pipeline/`
- `polylogue/storage/`
- `polylogue/lib/`
- `polylogue/operations/archive.py`

### 2. Derived Read Models

These are stored products over the archive:

- session profiles
- work events
- phases
- threads
- day and week summaries
- provider-level analytics and tag rollups

Primary modules:

- `polylogue/products/`
- `polylogue/storage/session_product_*.py`
- `polylogue/storage/repository_product_*.py`

### 3. Surfaces

These expose the archive and its products:

- CLI: `polylogue/cli/`
- Python API: `polylogue/facade.py`
- MCP server: `polylogue/mcp/`
- site generation: `polylogue/site/`
- dashboard and TUI: `polylogue/ui/`
- renderers: `polylogue/rendering/`

Most of these are leaf adapters. They should reuse archive operations and
derived products.

### 4. Verification and Maintenance

This ring keeps the archive inspectable and maintainable:

- schema generation and verification
- synthetic corpus generation
- showcase and deterministic acceptance exercises
- validation lanes
- mutation and benchmark campaigns

Primary modules:

- `polylogue/schemas/`
- `polylogue/showcase/`
- `devtools/`
- `tests/`

## Data Flow

```text
provider exports
  -> sources/
  -> pipeline/services/
  -> storage/
  -> products/
  -> CLI / Python API / MCP / site / dashboard
```

At the center is the archive database and its read models. Everything else
should either feed that center or read from it.

## Package Map

| Area | Responsibility |
| --- | --- |
| `polylogue/lib/` | Core models, filters, projections, IDs, dates, JSON helpers |
| `polylogue/storage/` | SQLite backend, repositories, search providers, product persistence |
| `polylogue/sources/` | Provider detection, acquisition, parser dispatch, raw payload handling |
| `polylogue/pipeline/` | Stage orchestration: acquire, schema, parse, materialize, render, index |
| `polylogue/products/` | Typed derived-product models and aggregation contracts |
| `polylogue/cli/` | Query-first CLI plus command families (`run`, `doctor`, `products`, `audit`, ...) |
| `polylogue/rendering/` | Markdown/HTML rendering and output-path logic |
| `polylogue/site/` | Static-site generation and publication manifests |
| `polylogue/mcp/` | Model Context Protocol server |
| `polylogue/showcase/` | Acceptance harness, seeded exercises, deterministic surface checks |
| `devtools/` | Repository tools for generated docs, validation, benchmarking, and hygiene |

## Placement Rules

Use these heuristics when adding or moving code:

- If it changes archive meaning, it belongs in `lib/`, `storage/`, `sources/`,
  `pipeline/`, or `products/`, not only in a surface.
- If it only presents existing archive meaning, it probably belongs in `cli/`,
  `mcp/`, `site/`, `rendering/`, or `ui/`.
- If it exists to prove, refresh, benchmark, or audit the repo, it belongs in
  `devtools/`, `showcase/`, or `tests/`.
- If a surface needs a new concept, define the concept in the substrate or
  product layer first.

## Stability Notes

- SQLite schema version is currently `v1` and remains fresh-only on mismatch.
- FTS uses `unicode61`; porter stemming is not assumed.
- The query surface is query-first at the root command and verb-oriented for
  archive actions.
- Use `devtools` for repo maintenance.
