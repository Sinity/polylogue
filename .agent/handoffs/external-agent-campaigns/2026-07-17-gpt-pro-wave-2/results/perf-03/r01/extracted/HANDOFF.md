# CLI snappiness: startup profile, lazy import discipline, and daemon-first fast reads

## Delivery status

This package is a cohesive implementation against Polylogue snapshot commit
`536a53efac0cbe4a2473ad379e4db49ef3fce74d`. `PATCH.diff` changes 49 repository
paths (4,626 insertions, 632 deletions) and was generated with binary/full-index
Git diff support so the generated OpenAPI and topology artifacts apply correctly.

The patch was applied with `git apply --binary` to a new detached worktree at the
named commit. All 49 resulting files were SHA-256 compared with the audited
candidate and matched byte-for-byte. The clean-applied tree then passed the
mission-focused test set, full-repository Ruff checks, strict mypy on every
changed production module, every generated-render check, and Git diff hygiene.

## Mission outcome

The implementation attacks shell latency at all three requested layers:

1. Root and read-command import graphs no longer initialize Git version lookup,
   Pydantic payload/model trees, API and sync facades, runtime services, source
   discovery, SQLite backends, archive-tier DDL, or the Lark parser merely to
   dispatch Click or render help.
2. The query parser is lazy, memoized in-process, and backed by a content-addressed,
   versioned persistent Lark cache with corruption recovery, grammar-change
   invalidation, XDG behavior, and no-cache fallback.
3. Read-only CLI surfaces first try an identity-matched local daemon over AF_UNIX.
   Successful reads avoid event-loop creation and the direct SQLite/API stack;
   every miss, mismatch, timeout, malformed response, or unsupported shape
   silently returns control to the established direct path.

The root CLI cumulative interpreter-import measurement fell from **186.416 ms**
to **67.489 ms**, meeting the mission's approximate 150 ms target in this
container. A proven exact read through the warm daemon measured **399.778 ms**
median versus **1,250.029 ms** through the candidate direct SQLite path and
**1,916.780 ms** through the snapshot direct path. Subprocess wall numbers are
host-dependent and remain unverified on the operator machine.

## Snapshot identity

The attached project-state archive manifest reports:

- Generated at: `2026-07-17T180950Z`
- Branch: `master`
- Commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Dirty metadata flag: `true`

The archive's branch-delta summary names `origin/master` with the same merge base.
Its branch-delta patch, changed-file list, and branch-only commit log are all zero
bytes. The reconstructed tracked tree matches the named commit. Accordingly,
`PATCH.diff` is intentionally based on that commit, while the manifest's dirty
flag is retained here as an evidence caveat rather than silently discarded.

## Evidence inspected

Repository instructions and architecture:

- `CLAUDE.md` and `AGENTS.md`
- `polylogue/cli/click_command_registration.py` and lazy Click registration/doc
  behavior
- `polylogue/cli/click_app.py`, `query.py`, `query_verbs.py`, `archive_query.py`,
  `root_request.py`, `query_contracts.py`, and shared CLI types/rendering
- `polylogue/archive/query/expression.py` and the complete query grammar/lowering
  route
- `polylogue/archive/query/transaction.py` and archive read-context ownership
- `polylogue/cli/commands/status.py`, `status_diagnostics.py`, and `facets.py`
- `polylogue/cli/daemon_client.py`, daemon UDS transport, health identity, HTTP
  routing, route contracts, authentication, and archive APIs
- `polylogue/api`, `polylogue/api/sync`, services, readiness, version, paths,
  payload surfaces, SQLite package boundaries, and archive-tier schema/DDL
- generated CLI reference, OpenAPI, topology target, and topology-status machinery

Tests and history:

- Existing CLI, parser/lowering, version, schema registry, sync bridge, archive-tier
  DDL/write, status, daemon route-contract, security, and architecture suites
- Existing production UDS parity test and its seed infrastructure
- Beads `polylogue-20d`, `polylogue-20d.1`, `polylogue-20d.2`,
  `polylogue-20d.12`, and `polylogue-fko9`
- Relevant history including `3082c72f0` (hot daemon read routing), `81bfedd87`
  (interactive CLI latency), and the subsequent bounded-probe/parity lineage

## Design and implementation

### 1. Root startup and lazy import discipline

`polylogue.version` now exposes explicit lazy accessors and resolves Git-derived
build identity only when version data is actually requested. Root Click
registration uses a custom eager `--version` callback, so ordinary dispatch and
help do not pay the Git subprocess path.

Large public packages keep their import surface while deferring implementation:

- `polylogue.api` delegates implementation to `polylogue/api/facade.py`.
- `polylogue.api.sync` delegates to `polylogue/api/sync/facade.py`.
- `polylogue.cli`, `polylogue.daemon`, `polylogue.readiness`, and
  `polylogue.storage.sqlite` expose lazy package attributes instead of importing
  their full stacks.
- Public classes moved behind facades preserve their expected public module
  identity for introspection and serialization compatibility.

The root request and query-contract modules no longer import the Pydantic
`SessionQuerySpec` at runtime merely for annotations. Runtime services defer
backend/repository construction. The schema-version constants for source,
index, embeddings, user, and operations tiers moved to
`archive_tiers/versions.py`; the archive-tier package now loads only the one DDL
builder actually requested.

This keeps root dispatch and the daemon handshake on substrate-level modules.
Cold API/model/storage imports remain available at the exact direct-execution
boundary.

### 2. Grammar/model warm-cost design

`polylogue.archive.query.expression` no longer constructs `Lark(...)` while the
module imports. `_get_query_parser()` constructs the parser on first real parse
and memoizes it for the process.

The persistent cache lives at:

`$XDG_CACHE_HOME/polylogue/query-grammar/<digest>.lark-cache`

When `XDG_CACHE_HOME` is unset or empty, the root is `$HOME/.cache`, following
XDG semantics. The digest includes:

- the complete grammar text;
- a Polylogue cache-format version;
- the installed Lark version;
- Python major and minor version;
- parser mode;
- complete start-rule tuple; and
- parser options.

A grammar edit therefore selects a new cache file. A corrupt existing entry is
deleted and regenerated once. Read-only homes, directory creation failures,
cache read/write failures, and failed corruption repair fall back to
`Lark(..., cache=False)` without changing query behavior.

The measured candidate warm operation was 231.557 ms versus 241.946 ms for the
candidate cold operation, a 4.294% same-candidate warm improvement. The much
larger snapshot-to-candidate reduction reflects both persistent caching and the
removal of unrelated import work from the grammar module; the evidence is not
misrepresented as cache benefit alone.

### 3. Daemon-first read architecture

The import-light client uses stdlib AF_UNIX HTTP. Its canonical socket is:

`$XDG_RUNTIME_DIR/polylogue/daemon.sock`

with `/tmp/polylogue/daemon.sock` as the runtime-root fallback.

Detection is deliberately ordered from cheapest to most expensive:

1. honor `--no-daemon`, `POLYLOGUE_NO_DAEMON`, and `POLYLOGUE_DAEMON=off`;
2. check socket existence before configuration, storage, or build-version work;
3. resolve the same archive root used by direct SQLite and the layered bearer
   token without constructing runtime services or discovering ingestion sources;
4. issue the UDS health request and require exact resolved archive-root and index
   schema-version matches;
5. only after those match, resolve the local Polylogue build version and require
   `daemon_version` equality; and
6. issue the requested read.

The client and server also treat an unset or explicitly empty `XDG_RUNTIME_DIR` as `/tmp`, preventing relative-path discovery splits.

The client uses a 45 ms timeout per UDS request. A normal health-plus-read attempt
therefore has a nominal two-request budget below 100 ms. All optional-daemon
failures return `None`; the canonical direct executor remains responsible for
user-facing diagnostics and exit codes.

Invocation-injected archive roots and authentication tokens are preserved.
Configured bearer authentication is sent on private routes. A stale socket,
wrong archive, wrong schema, wrong build, unauthorized route, timeout, malformed
JSON, degraded route state, or unsupported output/query shape falls back.

The root query executor attempts the daemon synchronously before creating an
async bridge. A successful fast read therefore avoids event-loop construction as
well as API, source, model, and SQLite initialization.

### 4. Daemon command coverage

The patch enables the following read surfaces:

- Root/list and free-text search through `POST /api/cli/query`
- Canonical opaque-cursor resume for list pages
- Exact session summary through authenticated private `POST /api/cli/read`
- Finite `read --view messages` for JSON and NDJSON through the same private route
- Positional public-reference resolution through the existing resolver route
- Facets through the matched UDS daemon
- Full and compact status, preferring the matched UDS daemon before TCP discovery
- Bare-terminal recent-session triage

The private read route accepts only `summary` and `messages`, validates finite
pagination, resolves the session inside the archive read/transaction context,
and emits the canonical CLI payload. It is registered in the route contract and
private-read security matrix.

Writes never proxy. Structured/complex expressions, mutation modes, streaming,
non-stdout destinations, semantic-card Markdown message rendering, and read
views whose parity is not proven stay on the direct path.

### 5. Output and semantic parity

`polylogue/surfaces/archive_session.py` now owns the canonical summary/text and
search/list row projections shared by direct and daemon paths. This fixed real
second-pass defects in cursor shape and ranked-search fields while removing the
renderer duplication that could otherwise drift later.

The production AF_UNIX golden tests compare direct SQLite and daemon-served
results against the same seeded archive. They cover:

- non-empty list JSON;
- second-page cursor resume;
- free-text ranked hit and zero-result exit 2;
- JSON, NDJSON, CSV, YAML, Markdown, and plaintext list rendering;
- facets;
- exact summary;
- authenticated finite messages in JSON and NDJSON, including tokens, model,
  content blocks, tool input, pagination, and the established attachment/block
  metadata contract; and
- positional public references.

List/search envelopes permit only the documented `"source": "daemon"`
provenance field. Timestamp-only facets data is normalized. Other machine
payloads compare field-for-field, and the tests require real non-empty fixtures
so parity cannot pass on two empty outputs. `--verbose` emits
`served-by: daemon (uds, <ms>)` on proven fast-path use without altering stdout.

## Measurements

All measurements used the same locked Python 3.13.5 virtual-environment
interpreter. Medians are from seven fresh processes except exact reads, which use
five. Cumulative import rows are CPython `-X importtime` interpreter measurements.
Wall-clock rows are container/host dependent and **unverified** on the operator
machine.

### Cumulative interpreter import

| Scenario | Snapshot median | Candidate median | Reduction |
|---|---:|---:|---:|
| Root CLI | 186.416 ms | 67.489 ms | 63.797% |
| Query dispatch | 1,317.414 ms | 101.504 ms | 92.295% |
| Read verbs | 178.073 ms | 71.709 ms | 59.731% |
| Grammar module | 413.686 ms | 246.095 ms | 40.512% |
| Daemon client | 178.007 ms | 44.504 ms | 74.999% |
| Status module | 1,195.141 ms | 104.178 ms | 91.283% |

### Root command wall clock — unverified

| Scenario | Snapshot median | Candidate median | Exit | Reduction |
|---|---:|---:|---:|---:|
| Root `--help` | 275.307 ms | 129.449 ms | 0 | 52.980% |
| Strict command floor | 227.817 ms | 117.724 ms | 2 | 48.325% |

The command-floor status remains exactly 2.

### Grammar operation and process — process rows unverified

| Scenario | Snapshot median | Candidate median | Reduction |
|---|---:|---:|---:|
| Cold parse operation | 423.449 ms | 241.946 ms | 42.863% |
| Warm parse operation | 412.774 ms | 231.557 ms | 43.902% |
| Cold subprocess wall | 476.572 ms | 291.694 ms | 38.793% |
| Warm subprocess wall | 470.232 ms | 288.030 ms | 38.747% |

### Exact read wall clock — unverified

| Path | Median | Range |
|---|---:|---:|
| Snapshot direct SQLite | 1,916.780 ms | 1,859.212–2,192.172 ms |
| Candidate direct SQLite | 1,250.029 ms | 1,203.925–1,293.052 ms |
| Candidate matched daemon UDS | 399.778 ms | 373.497–414.158 ms |

The daemon provenance marker appeared in 5/5 samples. Relative to the snapshot
direct path, candidate direct improved 34.785% and matched-daemon reads improved
79.143%. Relative to candidate direct SQLite, the daemon path improved 68.019%.

## Reproduction

From an applied checkout with its locked environment available:

```bash
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
BASELINE_REPO="${BASELINE_REPO:?set BASELINE_REPO to the clean snapshot checkout}"

# Before/after cumulative imports.
"$PYTHON_BIN" devtools/profile_cli_startup.py \
  --python "$PYTHON_BIN" \
  --repo . \
  --baseline-repo "$BASELINE_REPO" \
  --section imports \
  --repeats 7 \
  --json

# Parser cold/warm cache measurements.
"$PYTHON_BIN" devtools/profile_cli_startup.py \
  --python "$PYTHON_BIN" \
  --repo . \
  --baseline-repo "$BASELINE_REPO" \
  --section grammar \
  --repeats 7 \
  --json

# Root help and strict command-floor wall timing.
"$PYTHON_BIN" devtools/profile_cli_startup.py \
  --python "$PYTHON_BIN" \
  --repo . \
  --baseline-repo "$BASELINE_REPO" \
  --section root-cli \
  --repeats 7 \
  --json

# Exact read. A matching production daemon must already be running.
ARCHIVE_ROOT="${ARCHIVE_ROOT:?set ARCHIVE_ROOT}"
SESSION_ID="${SESSION_ID:?set SESSION_ID}"
RUNTIME_DIR="${RUNTIME_DIR:-${XDG_RUNTIME_DIR:?set RUNTIME_DIR or XDG_RUNTIME_DIR}}"
"$PYTHON_BIN" devtools/profile_cli_startup.py \
  --python "$PYTHON_BIN" \
  --repo . \
  --section exact-read \
  --archive-root "$ARCHIVE_ROOT" \
  --session-id "$SESSION_ID" \
  --runtime-dir "$RUNTIME_DIR" \
  --repeats 5 \
  --json
```

The profiler refuses to label a candidate exact-read timing as daemon-served
unless every sample contains the production provenance marker. It also preserves
the selected virtual-environment interpreter path rather than resolving its
symlink to a different base interpreter.

## Changed files

### New production and tooling files

- `devtools/profile_cli_startup.py`
- `polylogue/api/facade.py`
- `polylogue/api/sync/facade.py`
- `polylogue/cli/daemon_reads.py`
- `polylogue/storage/sqlite/archive_tiers/versions.py`
- `polylogue/surfaces/archive_session.py`

### Modified production boundaries

- Root/API/sync/daemon/readiness/SQLite package initializers
- Root Click app, root request, query contracts, query executor, query verbs,
  archive query, daemon client, daemon UDS socket discovery, facets, status,
  status diagnostics, shared types
- Query expression/parser implementation
- Daemon HTTP and route contracts
- Services, version, date/storage dependency edges, archive-tier DDL modules,
  embeddings reconciliation, and payload projections

### Tests

- New `tests/unit/cli/test_cli_snappiness.py`
- New `tests/unit/devtools/test_profile_cli_startup.py`
- Expanded daemon client, real UDS golden parity, HTTP contract, HTTP security,
  and architecture-boundary tests

### Generated files

- `docs/openapi/search.yaml` — regenerated for `POST /api/cli/read`
- `docs/plans/topology-target.yaml` — regenerated for new module/dependency edges
- `docs/topology-status.md` — regenerated topology projection

`docs/cli-reference.md` did not change. `devtools render all --check` confirms the
lazy commands still expose their complete parameter surfaces to the generator.

## Acceptance matrix

| Requirement | Result | Evidence |
|---|---|---|
| Report snapshot identity and inspect source/tests/Beads/history | Pass | Identity and evidence sections above |
| Root dispatch/help around 150 ms interpreter time | Pass | 67.489 ms cumulative root import |
| Preserve strict command-floor semantics | Pass | Exit 2 in all seven root-wall samples and regression suite |
| Remove heavyweight root imports | Pass | Import profiles and explicit `sys.modules` regressions |
| Lazy Lark construction | Pass | Import regression and first-use parser tests |
| Persistent versioned grammar cache | Pass | Reuse, grammar invalidation, corrupt repair, XDG, and I/O fallback tests |
| Identity-matched daemon detection | Pass | Archive, schema, build, auth, stale socket, and mismatch tests |
| Timeout and silent direct fallback | Pass | Bounded client plus timeout/fallback regressions |
| Empty/unset XDG runtime consistency | Pass | Client/server socket-path regression |
| `--no-daemon` escape hatch | Pass | Flag and both environment forms tested |
| List/read/search/status/facets hot paths | Pass for proven shapes | Real UDS and focused adapter tests |
| Direct/daemon semantic parity | Pass | Production UDS golden matrix |
| Writes remain direct | Pass by routing design | Fast adapters accept read-only shapes only |
| Strict mypy | Pass | 37 changed production modules |
| Generated CLI/docs integrity | Pass | Every `devtools render all --check` target |
| Reproduction script shipped | Pass | `devtools/profile_cli_startup.py` plus tests |
| Apply-ready cohesive patch | Pass | Fresh `git apply --binary`, byte comparison, clean-tree gates |

## Apply order

Apply only to the exact clean snapshot:

```bash
git rev-parse HEAD
# Must print 536a53efac0cbe4a2473ad379e4db49ef3fce74d

test -z "$(git status --porcelain)"
git apply --check PATCH.diff
git apply --binary PATCH.diff
```

Then use the repository's locked environment and run the commands recorded in
`TESTS.md`. The patch contains the required generated artifacts, so generation
should be checked, not independently rerun and committed before the source patch
has applied.

## Risks, limitations, and remaining verification

- The operator's live 38 GB archive, live daemon, secrets, NixOS deployment, and
  real shell environment were not accessible. All wall-clock numbers are
  explicitly unverified outside this container.
- The final clean-applied acceptance suite is broad but not the complete
  repository test corpus. It covers 1,537 selected tests around every changed
  production route and its major dependencies. Full-repository Ruff and all
  generated checks did run.
- Exact summary and finite machine-message reads are daemon-enabled. Markdown
  semantic-card messages and the `raw`, `context`, `context-image`, `neighbors`,
  `correlation`, `temporal`, and `chronicle` views remain direct because their
  distinct payload/render contracts do not yet have golden parity.
- Complex DSL, mutation, stream, special projection, and non-stdout shapes stay
  direct by design. This preserves established validation and diagnostics rather
  than widening the proxy contract speculatively.
- The daemon result cache/post-ingest warming requested by Bead
  `polylogue-20d.12` is a separate feature and is not folded into this patch.
  This patch makes the warm daemon reachable cheaply; it does not add a second
  memoization layer.
- A live operator run should remeasure the shipped profiler and the repository's
  interactive SLO lane. A result outside budget would now be a small
  environment-specific repair or a separate daemon-cache/storage investigation,
  not evidence that the import/transport architecture failed to land.

Another iteration on this exact mission has low expected architectural value.
A small repair could respond to operator-only timing, authentication, or archive
shape evidence. A substantial next performance pass belongs to the separately
owned cursor/epoch-keyed daemon result cache and post-ingest warming work, not to
further widening this startup patch.
