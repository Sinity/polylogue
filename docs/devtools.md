# Developer Tools

Use `devtools` for routine repository maintenance. Call individual
`devtools/*.py` modules directly only when you are editing these tools.

It exposes both human and JSON discovery/status forms. Use the JSON forms for
scripts and agents.

## Command Ownership Policy

`devtools` is the repository control plane. It owns orchestration around local
repo readiness: generated-surface rendering, baseline verification, validation
lane dispatch, package/build checks, and branch/PR readiness gates.

Domain validation semantics belong in lab, schema, scenario, or insight
modules first. A `devtools` command may expose them only as a thin operator
entrypoint that delegates to the owning executable check implementation.

Routine command placement:

- keep repo state, rendering, packaging, and PR-readiness orchestration in
  `devtools`;
- keep archive/insight workflows in `polylogue` CLI/API surfaces;
- keep evidence/scenario behavior in verification modules with executable command entrypoints;
- prefer validation lanes and the ordinary verifier to compose executable
  checks rather than duplicating domain checks inside `devtools verify`.

<!-- BEGIN GENERATED: devtools-command-catalog -->
## Command Catalog

Use these discovery commands before scripting or dispatching subcommands:

```bash
devtools --help
devtools --list-commands
devtools --list-commands --json
devtools status
devtools status --json
```

## Core Loop

These are the commands worth remembering during normal repo work:

- `devtools status`: Check repo state, generated-surface drift, and the next default verification steps.
  Common forms: `devtools status`, `devtools status --json`, `devtools status --verify-generated`.
- `devtools test`: Run a specific test file, directory, or -k/-m selection in the inner loop, or inspect the latest full-run timing receipts, without invoking raw pytest.
  Common forms: `devtools test tests/unit/pipeline`, `devtools test -k hybrid`, `devtools test tests/unit/storage -x`, `devtools test --outliers 20`.
- `devtools why`: A verify failed, bootstrapped unexpectedly, or refused to run, and you want the cause without reading receipt JSON by hand.
  Common forms: `devtools why`, `devtools why --history 24`, `devtools why --run 20260817T213631Z-2709409-d5c6e72c`.
- `devtools verify`: Run the gates and tests locally before pushing. --quick stops at the static gates; --all runs the complete corpus.
  Common forms: `devtools verify`, `devtools verify --quick`, `devtools verify --all`.
- `devtools gate`: Run a single gate in isolation, or list the declared gates and which of them verify --quick runs.
  Common forms: `devtools gate --list`, `devtools gate layering`, `devtools gate mypy`.
- `devtools render`: Refresh or verify generated repo surfaces after changing docs, CLI help, declarations, or agent memory.
  Common forms: `devtools render all`, `devtools render all --check`, `devtools render cli-reference`.

### Core

| Command | Description |
| --- | --- |
| `devtools cache gc` | Preview or apply age-gated GC for the shared seeded-archive fixture cache. |
| `devtools status` | Render the devshell status view. |
| `devtools test` | Run focused pytest selections or inspect full-run timing outliers. |
| `devtools why` | Explain the most recent verification run, or where verification time went. |

### Verification

| Command | Description |
| --- | --- |
| `devtools gate` | Run one named invariant check. |
| `devtools scenario` | Run a named archive verification scenario. |
| `devtools schema-manifest` | Compare canonical SQLite schema manifests with archive tier files. |
| `devtools smoke` | Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow. |
| `devtools verify` | Run the local verification baseline: every quick gate, then the selected or complete test corpus. |

### Generated Surfaces

| Command | Description |
| --- | --- |
| `devtools render` | Refresh or verify one generated repository surface, or all of them. |

### Schema

| Command | Description |
| --- | --- |
| `devtools schema commit` | Persist a real full-corpus schema generation into committed provider packages. |
| `devtools schema compare` | Compare two committed schema package versions for a provider. |
| `devtools schema explain` | Explain a committed package element schema with evidence and annotations. |
| `devtools schema generate` | Generate provider schema packages and optional evidence clusters. |
| `devtools schema list` | List committed schema packages, versions, and evidence manifests. |
| `devtools schema parser-diff` | List observed provider wire keys that no parser references. |
| `devtools schema promote` | Promote a schema evidence cluster into a registered package version. |

### Benchmarking

| Command | Description |
| --- | --- |
| `devtools bench memory` | Measure query-memory envelopes on generated fixtures. |
| `devtools bench pipeline` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools bench query-envelope` | Measure repeated incident-scale query RSS, PSS, swap, and temp envelopes. |
| `devtools bench slo` | Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements. |

### Archive

| Command | Description |
| --- | --- |
| `devtools archive continuity-evidence` | Replay continuity scenarios and verify their query routes are discoverable. |
| `devtools archive index-fast-forward` | Plan and prove a declared index fast-forward against retained raw replay. |
| `devtools archive lineage-validation` | Validate lineage-count evidence before citing archive counts externally. |

<!-- END GENERATED: devtools-command-catalog -->

## Pattern Ratchet

Pattern baselines use `path:sha1` content anchors, where the digest is computed from the matched line's trimmed first line, so inserting or removing lines does not churn the baseline. Duplicate normalized lines are represented with a count suffix such as `path:sha1:2`; matches beyond the baselined multiset are new blocking debt, while anchors no longer matched remain shrink-only stale debt.

## Validation and Evidence

When changing semantics, validation, or surfaces:

```bash
devtools verify
devtools test tests/unit/path/to/test_file.py
devtools scenario run archive-smoke --tier 0
devtools scenario run reader-visual-smoke
```

Campaign outputs live under `.local/`, not in tracked docs trees.

## Local State Layout

- `.cache/`: disposable cache state.
- `.local/`: untracked local outputs such as campaigns, demo artifacts, and reports.
- `.venv/` and `.direnv/`: kept at the repo root because their tooling expects those locations.
- `.local/result`: preferred repo-local out-link for `nix build`; a top-level `result` symlink is just Nix's default ad-hoc out-link.

Keep new repo-local outputs in `.cache/` or `.local/` instead of adding new
top-level output roots.
