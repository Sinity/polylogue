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
- `devtools test`: Run a specific test file, directory, or -k/-m selection in the inner loop, or inspect the latest full-run timing receipts, without invoking raw pytest. Refuses in a checkout on the default branch unless given --on-default-branch. A selection naming eight or more test modules runs under xdist (-n 4) unless it passes -n or -p no:xdist. A selection with a fixed test order (-p no:randomly or --randomly-seed=N) that already passed on the identical tree is answered from its receipt; pass --rerun to run it anyway.
  Common forms: `devtools test tests/unit/pipeline`, `devtools test tests/unit/pipeline --rerun`, `devtools test -k hybrid`, `devtools test tests/unit/storage -x`, `devtools test --outliers 20`.
- `devtools why`: A verify failed, bootstrapped unexpectedly, or refused to run, and you want the cause without reading receipt JSON by hand.
  Common forms: `devtools why`, `devtools why --history 24`, `devtools why --run 20260817T213631Z-2709409-d5c6e72c`.
- `devtools verify`: Run the gates and bounded affected tests locally before pushing. --quick stops at static gates; --all runs the complete corpus at the explicit master/corpus boundary. Unknown or oversized affected plans are refused before pytest and name the count, reason, and next boundary. Pattern baselines use path:sha1:context_sha1[:count] content anchors, not source line numbers.
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
| `devtools smoke` | Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow. |
| `devtools verify` | Run every quick gate, then a bounded affected selection or the explicit complete test corpus. |
| `devtools verify api-parity` | Check CLI/MCP/Python semantic-operation parity and the library documentation. |
| `devtools verify cli-acceptance` | Render, lint and measure the public CLI acceptance surface. |
| `devtools verify provider-completeness` | Report provider/importer package completeness from OriginSpec declarations. |

### Generated Surfaces

| Command | Description |
| --- | --- |
| `devtools render` | Refresh or verify one generated repository surface, or all of them. |

### Schema

| Command | Description |
| --- | --- |
| `devtools schema closure` | Report which source files feed the derived schema identity. |
| `devtools schema commit` | Persist a real full-corpus schema generation into committed provider packages. |
| `devtools schema compare` | Compare two committed schema package versions for a provider. |
| `devtools schema explain` | Explain a committed package element schema with evidence and annotations. |
| `devtools schema frontier` | Declare, record and check the schema-source frontier. |
| `devtools schema generate` | Generate provider schema packages and optional evidence clusters. |
| `devtools schema list` | List committed schema packages, versions, and evidence manifests. |
| `devtools schema new` | Scaffold a typed declaration, adapter stub, contract skeleton, and landing plan. |
| `devtools schema parser-diff` | List observed provider wire keys that no parser references. |
| `devtools schema promote` | Promote a schema evidence cluster into a registered package version. |
| `devtools schema reconcile` | Account for every declared schema subject after a generation pass. |
| `devtools schema workload-profile` | Measure the aggregate-only synthetic workload profile of an origin from its real sources. |

### Benchmarking

| Command | Description |
| --- | --- |
| `devtools bench baseline` | List or record committed measurement receipts under tests/benchmarks/baselines/. |
| `devtools bench collection` | Measure what a pytest selection costs to collect, before any test runs. |
| `devtools bench fresh-build` | Build a fresh archive from a sealed corpus through polylogued run and write a comparable receipt. |
| `devtools bench ingest-throughput` | Measure ingest throughput against synthetic source records. |
| `devtools bench memory` | Measure query-memory envelopes on generated fixtures. |
| `devtools bench parser-census` | Parse a recorded source denominator with no archive and diff the result against the last census. |
| `devtools bench pipeline` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools bench query-envelope` | Measure repeated incident-scale query RSS, PSS, swap, and temp envelopes. |
| `devtools bench slo` | Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements. |

### Archive

| Command | Description |
| --- | --- |
| `devtools archive continuity-cold-model` | Grade cold, wire-only model plan formulation against the continuity registry. |
| `devtools archive continuity-evidence` | Replay continuity scenarios and verify their query routes are discoverable. |
| `devtools archive lineage-validation` | Validate lineage-count evidence before citing archive counts externally. |
| `devtools archive tool-outcome-census` | Classify every archived tool result by origin, construct, outcome and unknown reason. |
| `devtools archive tool-pairing-census` | Classify every tool call/result pairing gap against declared evidence. |

<!-- END GENERATED: devtools-command-catalog -->

Ordinary test and verification runs retain their first outcome and report. They do not automatically retry failures or promote a later isolated pass. A requested failure diagnostic is a separate observation and must retain both outcomes. `devtools test --rerun` still forces execution instead of reusing an identical green receipt.

Corpus launches retain a qualified ceiling of two workers, including when live memory limits are unreadable. The observed charge model still reports its arithmetic prediction; a wider width fitting below `memory.high` does not establish that it can avoid the previously observed reclaim stall. Focused launches retain their separate charge profile and ceiling.

Affected admission collects the final selected node IDs against a compatible snapshot of the checkout's testmon graph. New nodes in recorded files and each parametrization count as unknown when that environment has no execution record. Forced contract tests form a separate physical launch and are priced again when they overlap the affected launch. Collection writes only temporary graph and selection evidence; unsuccessful or incomplete evidence refuses admission. Recorded durations remain a floor for selections containing unknown nodes. Graph writers and readers share an environment key covering pytest configuration, conftest content and the effective Hypothesis profile. A changed policy requires explicitly authorized collection. Managed execution copies declared source bytes into independent inodes and mounts the copy readonly at the checkout path with bubblewrap. The child policy must match its declared graph key. Source bytecode lookups use a fresh empty readonly directory, including in workers, reruns and collection, so shared timestamp-valid caches cannot supply different executed bytes. A dedicated supervisor establishes kernel child-subreaper custody before each sole execution child. Independent admitted supervisor bytes execute from a sealed memfd; its private closure descriptor is withheld from descendants and its process is nondumpable. Every initial, rerun and collection attempt must provide an attempt-bound ECHILD closure and exact pidfd death before publication. Detached, opaque and further-user-namespace descendants remain kernel children; unrelated host peers are outside this observation boundary. Missing closure or forced supervisor death refuses authority and retains the source copy. The existing marker/birth sampler still owns memory attribution. Host PID and proc semantics remain available to shared job and temporary-file owners. Writable runtime artifacts stay under the declared `.cache` binding; source writes fail. Copy or policy mismatches, launch failures and unsettled descendants refuse execution authority while retaining the graph with an unavailable-authority marker. Only a completed authoritative full collection with terminal test evidence clears prior refusal.


Verification memory attribution uses the actual launched process group and a fresh child-only custody marker inherited by descendants and an explicitly requested diagnostic retry. Sharing a cgroup does not establish ownership. Each reading is bound to the kernel process start time; detached marked children remain attributable after reparenting, while reused PIDs cannot acquire prior custody. Environment comparison streams fixed-size chunks and retains only marker equality. Missing or changing process/custody/memory evidence is visibly incomplete or unmeasured in the existing memory receipt; incomplete evidence cannot corroborate the charge model. The sampler does not claim cgroup isolation.

## Pattern Ratchet

Pattern baselines use `path:sha1:context_sha1` content anchors, where the digest is computed from the matched line's trimmed first line, so inserting or removing lines does not churn the baseline. Duplicate normalized lines are represented with a count suffix such as `path:sha1:context_sha1:2`; matches beyond the baselined multiset are new blocking debt, while anchors no longer matched remain shrink-only stale debt.

## Validation and Evidence

When changing semantics, validation, or surfaces:

```bash
devtools verify
devtools test tests/unit/path/to/test_file.py
devtools scenario run archive-smoke --tier 0
devtools scenario run reader-visual-smoke
```

Campaign outputs live under `.local/`, not in tracked docs trees.

Pytest steps retain the original `process_exit` separately from the evidence
verdict. An existing unreadable or malformed report, selection, summary, or
event ledger produces `pytest_evidence_unavailable`; statistics publication
and mirror failures use the same diagnosis. The run-local step records
`evidence_error` with the phase, exception type, and message, and a process
exit of zero becomes a failed verification exit. Explicit interruption and
nonexecution diagnoses remain alongside this evidence error. `devtools why`
shows the recorded error; the canonical receipt carries only its phase and
type, without local paths or exception text.

## Checkout entry

The devshell configures Git hooks only for the entered checkout. It anchors
`core.hooksPath` to the common checkout's `.githooks`, enables worktree config,
and pins the current worktree's override. Each scope is written only when its
value differs. Entering a worktree leaves sibling worktree configs untouched.
Direnv uses the flake's bytecode and cleanup setup without repeating it.

Source fingerprint and lexical import-edge memos are disposable shared state
under `$XDG_CACHE_HOME/polylogue/source-fingerprints`, or
`~/.cache/polylogue/source-fingerprints` when XDG cache home is unset. Keys cover
repository-relative labels, exact source bytes, the algorithm version, Python's
AST format, and the fingerprint namespace. Equal source trees share entries;
changed content misses. Import edges store lexical bases and resolve them in
the current tree, so adding or removing an imported module still changes the
closure. Entries publish atomically with unique temporary files. An unavailable
or malformed memo recomputes the result; there is no checkout-local memo route.

## Local State Layout

- `.cache/`: disposable cache state.
- `.local/`: untracked local outputs such as campaigns, demo artifacts, and reports.
- `.venv/` and `.direnv/`: kept at the repo root because their tooling expects those locations.
- `.local/result`: preferred repo-local out-link for `nix build`; a top-level `result` symlink is just Nix's default ad-hoc out-link.

Keep new repo-local outputs in `.cache/` or `.local/` instead of adding new
top-level output roots.

Terminal verification publication resumes on the next verifier or `devtools why`
read if interruption separates the run receipt, history append, and durable
evidence append. Recovery preserves the original verdict and canonical receipt;
it does not rerun verification or grant new authority. The history's canonical
receipt also restores missing evidence after successful detail pruning. Both
append lanes serialize run identity checks with publication, so concurrent
finish and recovery produce one row per run in each lane. Recovery scans identities once per
lane for a batch and makes every appended row durable before advancing.


`devtools cache gc` returns one requested artifact page (`--page-size`, default
100) with `complete` and `next_cursor`; pass `--after <next_cursor>` to continue.
Deleting an earlier page does not shift the next page. Each page rechecks the
current declared reachability and existing cache, key and artifact leases.
Session archive fixtures hold the original shared cache lease from acquisition
through finalization; benchmark cloning holds it from acquisition through copy.

Applied GC atomically retires eligible trees into the existing private staging
namespace before unlinking any contents. Receipts are durably replaced before
deletion and after each artifact decision. An interrupted page reports
`interrupted`, incomplete deletion-byte evidence and no continuation: restart
the page to resume retired trees. Partial retired trees are disposable staging,
not corrupt published artifacts. Traversal and hashing observe cancellation
between streamed nodes and chunks; GC does not impose node or tree-depth caps.

The declared GC operation has no execution deadline (`timeout_seconds = 0`).
It remains owned by AgentCTL and stops through operator cancellation, preserving
the resumable retired tree and interruption receipt.

### Verification history destination

`POLYLOGUE_VERIFY_HISTORY_PATH` selects the shared history destination; unrelated custom paths remain supported. The operator environment may declare exact retired destinations in `POLYLOGUE_RETIRED_VERIFY_HISTORY_PATHS`, separated by the platform path separator. Writes to those destinations fail before creating directories. Use the current declared destination or an unrelated custom path. Historical files remain readable for provenance and run-ID-aware reconciliation; they are never redirected through an alias. Declared jobs supply the current destination explicitly.

The established managed history retirements are refused even when a long-lived
parent shell lacks `POLYLOGUE_RETIRED_VERIFY_HISTORY_PATHS`. That variable adds
operator-specific retirements; it cannot enable the established old addresses.
Unrelated custom destinations remain supported. Preexisting checkouts that
precede this guard must enter the current managed environment to inherit the
current destination.
