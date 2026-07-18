# Test design and execution record

## Certification design

The central test is `tests/unit/mcp/test_query_transaction_certification.py`. It creates 20 real `ParsedSession` records through `ArchiveStore.write_parsed`, each containing one message with a deterministic timestamp, deterministic identity, the query needle, and approximately 3,500 payload characters. It queries:

```text
messages where text:certificationneedle | sort by time asc
```

with page size 9.

The corpus is intentionally sized so the first MCP production serialization exceeds `MCP_RESPONSE_BUDGET_BYTES == 25_000`. The test does not infer overflow from a status field alone; it asserts `original_bytes > 25_000` and that the retained prefix is non-empty and smaller than the requested page.

It then drains API, MCP, and HTTP continuations to termination and compares their exact ordered message IDs to the independently recorded IDs returned during archive insertion. It also checks stable query/result refs, one-hour validity-window preservation, cursor interchange from API to MCP/HTTP, typed invalid/expired/stale outcomes, discovery example parsing, and forced deadline/work-budget receipts.

## Exact production dependencies exercised

| Test path | Production dependencies, not replicas |
| --- | --- |
| API drain | `Polylogue.query_units`; query grammar/compiler; `query_unit_request`; `query_units_transaction_request`; `QueryTransaction`; `execute_archive_read`; `InterruptibleSQLiteRead`; `ArchiveStore.open_existing`; terminal SQL executor; `query_unit_envelope` |
| MCP drain and overflow | registered `query_units` handler from `register_query_tools`; production callback hooks; same transaction/SQLite path; `_response_context`; `_json_payload`; `_bounded_item_page`; `_budget_envelope`; `MCPErrorPayload` translation |
| HTTP drain | real loopback `DaemonAPIHTTPServer` and `DaemonAPIHandler`; `_handle_query_units`; `_send_error`; synchronous transaction/execution-control path; same terminal SQL executor |
| Discovery grammar | registered `polylogue://capabilities/query` resource; production query metadata/declaration registry; `parse_unit_source_expression` |
| Deadline/budget | production `QueryExecutionContext`, deadline check, SQLite progress handler, work-budget exception mapping, transaction receipt, API exception, MCP error JSON, HTTP 503 JSON |
| Cancellation | production async `QueryTransaction.run`, worker cancellation/interrupt/drain path, and attached `QueryTransactionFailureReceipt` |
| Generated contracts | real Pydantic models, CLI schema renderer, OpenAPI renderer, and WebUI client generator |

The final verification used the real MCP SDK 1.28.1 and sqlite-vec 0.1.9; no import or production-dependency stubs were present. The selected tests executed the production registration manager, registered handler body, archive initialization, query parser/compiler, SQLite transaction path, response serialization, and loopback HTTP server. They did not launch a separate MCP stdio/client process.

## Anti-vacuity mutations

| Representative mutation/removal | Expected failing evidence |
| --- | --- |
| Reduce corpus text until the first response fits under 25,000 bytes | explicit `original_bytes > MCP_RESPONSE_BUDGET_BYTES` assertion fails |
| Remove MCP transaction context or terminal-page cursor minting | overflow drain omits the clipped suffix or terminates early; exact 20-ID equality fails |
| Reuse the storage continuation after MCP prefix trimming | offset jumps past omitted rows; ordered member equality and no-missing assertion fail |
| Let API/MCP/HTTP reconstruct filters or offset from caller parameters | continuation-only and cross-surface resume fail; stable ref and expected-member assertions fail |
| Remove archive epoch from result identity or validation | stale-cursor API/MCP/HTTP assertions fail; same-snapshot guard unit test fails |
| Remove expiry/checksum/canonical query validation | expired/tampered cursor tests cease returning the expected typed errors |
| Catch invalid cursor and execute an initial request | patched facade fallback raises; error payload would contain items or wrong code |
| Remove deadline/work-budget receipt mapping | API exception receipt, MCP receipt, or HTTP receipt assertions fail |
| Re-mint a new TTL on every page | issuance/expiration window equality assertions fail |
| Publish a discovery example not accepted by the grammar | `parse_unit_source_expression(example)` assertion names the failing example |

## Commands actually executed

### Isolated real dependency environment

```bash
/opt/pyvenv/bin/python -m venv --system-site-packages /tmp/polylogue-delivery-work/venv
/tmp/polylogue-delivery-work/venv/bin/python -m pip install -e '.[dev]'
/tmp/polylogue-delivery-work/venv/bin/python -m pip install setuptools
/tmp/polylogue-delivery-work/venv/bin/python -m pip check
```

The completed environment reported no broken requirements. Relevant resolved versions were MCP 1.28.1, sqlite-vec 0.1.9, pytest 9.1.1, Pydantic 2.13.4, Ruff 0.15.22, and Mypy 2.3.0.

### Final focused regression runs

```bash
/tmp/polylogue-delivery-work/venv/bin/python -m pytest -q \
  -o addopts='' -p no:randomly --timeout=90 \
  tests/unit/archive/query/test_transaction.py \
  tests/unit/mcp/test_query_transaction_certification.py \
  tests/unit/mcp/test_bounded_query_transport.py \
  tests/unit/mcp/test_server_surfaces.py
```

Observed result:

```text
103 passed in 22.83s
```

```bash
/tmp/polylogue-delivery-work/venv/bin/python -m pytest -q \
  -o addopts='' -p no:randomly --timeout=90 \
  tests/unit/daemon/test_daemon_http_contracts.py \
  tests/unit/daemon/test_web_reader.py::TestReaderQueryUnits \
  tests/unit/cli/test_cli_output_schemas.py \
  tests/unit/devtools/test_render_openapi.py \
  tests/unit/devtools/test_render_webui_client.py
```

Observed result:

```text
90 passed in 11.53s
```

Together these final post-repair runs executed 193 focused tests with no failures. File-level counts remained: 14 query transaction, 3 cross-surface certification, 3 bounded MCP transport, 83 registered MCP surface, 34 daemon HTTP contract, 12 Web reader `query_units`, and 44 generated CLI/OpenAPI/WebUI tests.

### Formatting, lint, strict typing, compilation, and generated drift

```bash
mapfile -t PYFILES < <(git apply --numstat PATCH.diff | awk '$3 ~ /\.py$/ {print $3}')
/tmp/polylogue-delivery-work/venv/bin/ruff check "${PYFILES[@]}"
/tmp/polylogue-delivery-work/venv/bin/ruff format --check "${PYFILES[@]}"
/tmp/polylogue-delivery-work/venv/bin/mypy "${PYFILES[@]}"
/tmp/polylogue-delivery-work/venv/bin/python -m compileall -q "${PYFILES[@]}"

/tmp/polylogue-delivery-work/venv/bin/python -m devtools.render_cli_output_schemas --check
/tmp/polylogue-delivery-work/venv/bin/python -m devtools.render_openapi --check
/tmp/polylogue-delivery-work/venv/bin/python -m devtools.render_webui_client --check

git diff --check
```

Observed result:

```text
All checks passed!                         # Ruff lint
17 files already formatted                # Ruff format check
Success: no issues found in 17 source files
render cli-output-schemas: sync OK: docs/schemas/cli-output
render openapi: sync OK: docs/openapi/search.yaml
render webui-client: sync OK: webui/src/api/generated.ts
```

Compilation and `git diff --check` exited successfully without diagnostics.

### Independent patch reconstruction

```bash
git worktree add --detach <temporary-path> bf8191b3f56aa40da8f271df7f3385c712825497
git -C <temporary-path> apply snapshot-dirty.patch
git -C <temporary-path> apply --check PATCH.diff
git -C <temporary-path> apply --check --whitespace=error-all PATCH.diff
git -C <temporary-path> apply PATCH.diff
python -m compileall -q <all 17 modified Python files in temporary worktree>
git -C <temporary-path> diff --check
```

Observed result:

- authority commit plus supplied dirty patch reconstruction: **passed**;
- supplied dirty patch identity: SHA-256 `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`, 2,272 bytes;
- ordinary and strict-whitespace `PATCH.diff` apply checks: **passed**;
- actual patch apply: **passed**;
- patched-tree compilation and whitespace checks: **passed**;
- all 20 patched paths matched the generating tree byte-for-byte;
- final `PATCH.diff`: SHA-256 `b3c5e888dba3c562931d9a8b13bb58417dbd3d676ee08919525de646d50667e1`, 111,521 bytes, 2,555 lines.

### Placeholder and input-copy scan

Added lines were scanned for `TODO`, `TBD`, `FIXME`, `XXX`, placeholder/pseudocode markers, `NotImplemented`, bare `pass`, source-archive names, and temporary/input paths. None were found. The only added ellipses are a pre-existing Protocol method body and the valid `tuple[MessageQueryRowPayload, ...]` type parameter. `PATCH.diff` contains no Git binary patch, tar/bundle signature, supplied archive, or copied source snapshot.

## Required complete-environment verification remaining

The focused tests and changed-file toolchain checks above are complete. Remaining operator-side verification is the repository-wide `devtools verify --quick` baseline, one real MCP stdio/client session, and the local live archive lane. No daemon deployment, browser, client rollout, secrets, NixOS configuration, incident archive, concurrent-ingest longevity run, or 4.85M-block performance test was available.
