# Test design and execution record

## Test strategy

The verification strategy is deliberately layered:

1. Import-boundary tests prove that root/daemon-facing imports do not initialize
   the cold model, API, source-discovery, storage, DDL, Click, or Git-version
   graphs.
2. Focused adapter tests prove daemon identity, authentication, timeouts, escape
   hatches, supported-shape selection, direct fallback, canonical cursors, and
   output rendering.
3. Production AF_UNIX golden tests start the real daemon HTTP server against a
   seeded archive and compare direct SQLite with daemon-served output.
4. Route-contract and security tests prove that the new private route is
   registered, validates input, enters the real archive read context, and obeys
   bearer-auth/host/origin policy.
5. Existing parser, version, schema, sync, archive-tier, status, and architecture
   suites protect the dependency edges touched by lazy loading.
6. Full-repository Ruff and generated-render checks protect style and generated
   products; strict mypy runs over every changed production Python module.
7. The frozen patch is applied to a clean worktree and the same gates run there.

## New and expanded test laws

| Test area | Production dependency exercised | Representative mutation/removal that must fail |
|---|---|---|
| Lazy package exports | `polylogue.api`, sync, CLI, daemon, readiness, SQLite package facades and introspection | Replace lazy `__getattr__` with eager imports or omit a public export |
| Root version deferral | `polylogue.version` and Click `--version` callback | Resolve Git/version metadata at module import |
| Runtime-services deferral | `polylogue.services` backend/repository properties | Restore top-level SQLite/repository imports |
| Daemon-down/stale-socket probe | `matching_daemon_client` ordering | Resolve config/build identity before checking or successfully probing the socket |
| UDS runtime-root consistency | client `default_daemon_socket_path` and server `daemon_socket_path` | Treat an explicitly empty `XDG_RUNTIME_DIR` as a relative path on either side |
| Archive/auth identity | active path owner, layered token, invocation-injected config | Construct source-discovering `RuntimeServices`, use a different root, or drop the token |
| Schema constants/DDL | archive-tier `versions.py` and lazy DDL mapping | Import all DDL modules to read one schema version or load every tier for one lookup |
| Lazy grammar parser | `_get_query_parser` | Construct `Lark` during module import |
| Grammar cache invalidation | `_query_parser_cache_path` | Remove complete grammar text, Lark/Python/parser/start/options, or cache-format version from the key |
| Cache reuse/recovery | `_build_query_parser` | Disable persistent reuse, retain a corrupt entry forever, or propagate cache-I/O failure into query semantics |
| Exact summary fast path | root query adapter and `POST /api/cli/read` | Import the cold executor/services before a proven daemon hit or accept a noncanonical payload |
| Finite messages fast path | `read_verb`, private route, canonical paginated message payload | Drop token/model/tool fields, widen metadata/attachment shape, ignore limit/offset, or use daemon output without parity validation |
| Public-reference fast path | existing `/api/refs/resolve` route | Import API facade for a daemon hit or accept a malformed resolution envelope |
| List/search fast path | `/api/cli/query`, canonical row projection, exit behavior | Proxy an unsupported structured shape, turn empty search exit 2 into success, or lose search rank/target fields |
| Cursor resume | import-light cursor decoder/encoder and list renderer | Emit a different opaque cursor, ignore decoded rank, or repeat page one |
| Timeout/fallback | UDS client and root direct executor boundary | Surface optional-daemon transport errors or skip canonical direct validation |
| Escape hatches | root flag and both environment forms | Probe the socket despite explicit disablement |
| Status/facets | matched UDS client before TCP/direct work | Prefer an unmatched daemon or bypass direct fallback |
| Real UDS parity | production `DaemonAPIUnixHTTPServer`, handler, transport, auth, routes, SQLite archive, renderers | Replace the server with a mock, compare empty fixtures, remove auth, alter route payloads, cursor, formatter, or provenance rules |
| Route contract/security | static route tables and handler methods | Omit `/api/cli/read`, allow unsupported views, bypass credential gating, or point the contract at a missing method |
| Surface/storage boundary | architecture dependency test | Import storage repositories from the new canonical surface module |
| Profiler correctness | shipped subprocess profiler | Resolve a venv symlink to the base interpreter, inherit daemon-disable variables, or claim daemon timing without every provenance marker |

The real-socket golden fixtures explicitly require non-empty list, search, summary,
and message results. They therefore cannot pass vacuously by comparing two empty
payloads.

## Final clean-apply test command

The following command ran from a new detached worktree after
`git apply --binary PATCH.diff`:

```bash
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
PYTHONPATH="$PWD" "$PYTHON_BIN" -m pytest -q --disable-warnings \
  tests/unit/cli/test_cli_snappiness.py \
  tests/unit/cli/test_daemon_client.py \
  tests/unit/cli/test_daemon_golden_parity.py \
  tests/unit/daemon/test_daemon_http_contracts.py \
  tests/unit/daemon/test_daemon_http_security.py \
  tests/unit/devtools/test_profile_cli_startup.py \
  tests/unit/architecture/test_surface_storage_boundary.py \
  tests/unit/cli/test_query_expression.py \
  tests/unit/core/test_version.py \
  tests/unit/core/test_version_runtime.py \
  tests/unit/core/test_schema_registry_versions.py \
  tests/unit/api/test_sync_bridge.py \
  tests/unit/storage/test_archive_tiers_ddl.py \
  tests/unit/storage/test_archive_tiers_write.py \
  tests/unit/cli/commands/test_status.py \
  tests/unit/cli/test_status.py \
  tests/unit/cli/test_status_diagnostics.py
```

Result:

```text
1536 passed, 1 skipped in 55.83s
```

The single skip is the existing embeddings-tier DDL test's declared fallback
when the runtime cannot load `sqlite-vec`. No selected test failed.

The selected set contains 1,537 collected cases and covers every changed
production route plus the existing parser/lowering, schema, version, sync,
archive-tier, status, security, and architecture contracts most likely to catch
composition errors.

## Production UDS golden matrix

`tests/unit/cli/test_daemon_golden_parity.py` starts
`DaemonAPIUnixHTTPServer` with `DaemonAPIHandler` and a real seeded SQLite archive.
The CLI connects through the production stdlib AF_UNIX client and passes the
health identity and configured bearer token.

Verified cases:

- List JSON direct/daemon structural parity with only daemon provenance removed
- Facets parity, excluding the independently generated timestamp
- Exact session summary field-for-field parity and daemon provenance proof
- Positional public-reference parity through the existing route
- Finite message JSON and NDJSON parity, including token counts, model,
  content-block/tool input, pagination, and established omitted metadata fields
- First and second list page parity using the exact direct opaque cursor
- NDJSON, CSV, YAML, Markdown, and plaintext list output parity
- Ranked free-text hit parity
- Zero-result search output and exit-status-2 parity
- Authenticated private route operation

## Static and generated checks on the clean-applied tree

Commands:

```bash
VENV_DIR="${VENV_DIR:-$PWD/.venv}"
"$VENV_DIR/bin/ruff" format --check .
"$VENV_DIR/bin/ruff" check .

mapfile -t changed_production < <(
  git diff --name-only 536a53efac0cbe4a2473ad379e4db49ef3fce74d -- \
    'polylogue/**/*.py'
)
PYTHONPATH="$PWD" "$VENV_DIR/bin/mypy" --strict \
  "${changed_production[@]}"

PYTHONPATH="$PWD" "$VENV_DIR/bin/python" -m devtools render all --check
git diff --check
```

Results:

```text
ruff format --check: 2211 files already formatted
ruff check: All checks passed
mypy --strict: Success: no issues found in 38 source files
render all --check: every target sync OK; generated local links resolve
git diff --check: clean
```

The render aggregate checked CLI reference, CLI output schemas, OpenAPI,
devtools reference, demo-corpus datasheet, quality reference, product workflows,
docs surface, MCP equivalence, MCP tool index, topology status, and pages.

`docs/cli-reference.md` remained byte-identical and required no regeneration.
The generated OpenAPI and topology files included in the patch are synchronized.

## Patch reconstruction proof

The frozen patch was produced from the complete staged candidate with:

```bash
git diff --cached --check
git diff --cached --binary --full-index \
  536a53efac0cbe4a2473ad379e4db49ef3fce74d > PATCH.diff
```

It was then tested in a new detached worktree:

```bash
PATCH_FILE="${PATCH_FILE:?set PATCH_FILE to the extracted PATCH.diff}"
CLEAN_WORKTREE="${CLEAN_WORKTREE:-/tmp/perf03-clean}"
git worktree add --detach "$CLEAN_WORKTREE" \
  536a53efac0cbe4a2473ad379e4db49ef3fce74d
cd "$CLEAN_WORKTREE"
git apply --check "$PATCH_FILE"
git apply --binary "$PATCH_FILE"
git diff --check
```

All 49 changed paths in that worktree were SHA-256 compared with the candidate;
there were zero missing or mismatched files. The static and pytest commands above
then ran from this reconstructed tree.

`PATCH.diff` is 6,160 lines and 232,107 bytes. Its standalone SHA-256 before ZIP
assembly is:

`bc4966804fab19d134efd32a60aa6acc4dc0a80e28aa6e7b2f9deeef1126ad2b`

## Measurement execution

The shipped profiler was run with the same locked Python 3.13.5 interpreter for
both snapshot and candidate. Import, grammar, and root-command rows used seven
fresh processes. Exact reads used five.

Completed measurement sections:

```bash
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
BASELINE_REPO="${BASELINE_REPO:?set BASELINE_REPO}"
"$PYTHON_BIN" devtools/profile_cli_startup.py \
  --python "$PYTHON_BIN" --repo . \
  --baseline-repo "$BASELINE_REPO" \
  --section imports --repeats 7 --json

"$PYTHON_BIN" devtools/profile_cli_startup.py \
  --python "$PYTHON_BIN" --repo . \
  --baseline-repo "$BASELINE_REPO" \
  --section grammar --repeats 7 --json

"$PYTHON_BIN" devtools/profile_cli_startup.py \
  --python "$PYTHON_BIN" --repo . \
  --baseline-repo "$BASELINE_REPO" \
  --section root-cli --repeats 7 --json

ARCHIVE_ROOT="${ARCHIVE_ROOT:?set ARCHIVE_ROOT}"
SESSION_ID="${SESSION_ID:?set SESSION_ID}"
RUNTIME_DIR="${RUNTIME_DIR:-${XDG_RUNTIME_DIR:?set RUNTIME_DIR or XDG_RUNTIME_DIR}}"
"$PYTHON_BIN" devtools/profile_cli_startup.py \
  --python "$PYTHON_BIN" --repo . \
  --section exact-read \
  --archive-root "$ARCHIVE_ROOT" \
  --session-id "$SESSION_ID" \
  --runtime-dir "$RUNTIME_DIR" \
  --repeats 5 --json
```

For the exact-read section, a production UDS daemon was started against the same
seeded five-tier archive. Health reported the expected archive root, index
schema 38, and build identity. The private route returned a canonical non-empty
session payload, and all five candidate daemon samples carried the production
`served-by: daemon (uds, <ms>)` marker. The daemon was shut down after profiling.

Measured medians are reproduced in `HANDOFF.md`. All subprocess wall times are
host-dependent and unverified on the operator's machine; cumulative CPython
import rows are the intended stable startup comparison.

## Verification not claimed

The complete repository test corpus was not run end-to-end on the final clean
worktree. The final claim is the 1,537-case changed-scope/dependency suite above,
not a fabricated full-suite result. The operator's live archive, live daemon,
NixOS service, secrets, and 38 GB page-cache behavior were not available.
