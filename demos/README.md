# Demo Generation

Animated terminal screencasts for polylogue documentation and README.

## Quick Start

```bash
# Generate all demos (uses synthetic data — no real exports needed)
./demos/generate.sh

# Generate demos verbosely
./demos/generate.sh --verbose
```

## Prerequisites

- **VHS**: Terminal recorder from Charmbracelet. Available in the Nix devshell (`nix develop`), or install via `brew install vhs` / `go install github.com/charmbracelet/vhs@latest`.
- **Python environment**: `uv sync` or `nix develop` for polylogue dependencies.

## How It Works

The demo pipeline uses `SyntheticCorpus` (from `polylogue/schemas/synthetic.py`) to generate
realistic conversations for all supported providers. No real chat exports are needed.

## Tape Files

| Tape | Shows | Output |
|------|-------|--------|
| `01-overview.tape` | `polylogue` (no args) → stats panels | `artifacts/demos/01-overview.gif` |
| `02-run.tape` | `polylogue run --preview` → plan, then run | `artifacts/demos/02-run.gif` |
| `03-search.tape` | Query mode with filters | `artifacts/demos/03-search.gif` |
| `04-dashboard.tape` | TUI dashboard browsing | `artifacts/demos/04-dashboard.gif` |
| `05-site.tape` | Static HTML site generation | `artifacts/demos/05-site.gif` |

## Workflow

```
generate.sh
  ├── 1. seeds demo database (polylogue generate --seed → SyntheticCorpus)
  ├── 2. exports env vars for isolation
  ├── 3. runs each .tape file via vhs
  └── 4. copies durable docs assets to docs/assets/
```

## Re-recording a Single Tape

```bash
# Skip seeding, just re-record one tape
./demos/generate.sh --skip-seed --tape 03-search
```

## CI

The GitHub Actions workflow (`.github/workflows/demos.yml`) generates demos automatically. Local capture output lives under `artifacts/demos/`; durable published assets live under `docs/assets/`.
