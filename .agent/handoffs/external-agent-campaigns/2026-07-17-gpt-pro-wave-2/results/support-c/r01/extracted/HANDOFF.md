# Lane C handoff: transaction continuation certification across MCP/API/HTTP

## Mission

Certify one canonical bounded, resumable terminal-unit read transaction across the Python API, registered MCP tool, and daemon HTTP route using a deliberately overflowing real SQLite corpus. The package must prove lossless continuation, typed invalid/expired/stale failure, safe host-protection receipts, and parser-truthful MCP discovery without aliases or a test-only facade.

## Result

This package implements and certifies one canonical bounded/resumable `query_units` transaction across the Python API, registered MCP tool, and daemon HTTP route. The implementation keeps expression, structural session filters, page size, stable order, archive frame, query identity, result identity, validity window, and delivery offset inside one opaque `q2` continuation. All three adapters use the same constructor and decoder instead of reconstructing continuation state independently.

The certification corpus contains 20 real archived messages with approximately 3,500 characters each. Its first MCP result is deliberately larger than the production 25,000-byte response budget. The test asserts that overflow before accepting continuation evidence, then drains every member in the same order through API, MCP, and HTTP with no duplicates or omissions.

This is an implementation draft against the supplied snapshot, not a live-daemon, deployment, client-rollout, or 4.85M-block performance claim. This revision closes the earlier delivery gap by formatting, linting, and type-checking the changed Python files in a real project dependency environment, independently reconstructing the patch base, and sealing the required ZIP.

## Snapshot identity and patch base

The attached Chisel archive identifies itself as:

- generated: `2026-07-18T013442Z`
- source: `/realm/project/polylogue`
- branch: `master`
- commit: `bf8191b3f56aa40da8f271df7f3385c712825497`
- dirty: `true`
- supplied archive SHA-256: `47ad17ea5a44d148a3c58baa74da9cff650b51e46ceed38e41468810a0896df4`

The supplied working tree contained three unrelated uncommitted edits:

1. `polylogue/archive/query/unit_results.py`: type-safe aggregate sort narrowing;
2. `polylogue/daemon/http.py`: cast the selected peer connection to `socket.socket`;
3. `polylogue/hooks/__init__.py`: explicit `claude-code`/`codex` environment matching.

Those edits were preserved as the patch prerequisite, not copied into `PATCH.diff`. Their reconstructed patch is SHA-256 `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f` and 2,272 bytes. The reconstruction compared every tracked file present in the supplied working-tree archive against the authority commit: exactly those three files differed.

## Authority inspected

Repository guidance inspected: `CLAUDE.md`/`AGENTS.md`, especially the substrate-first rule, leaf-adapter boundaries, focused verification doctrine, strict typing gate, and generated schema requirements.

Production source followed end to end:

- query grammar and terminal execution: `polylogue/archive/query/expression.py`, `unit_results.py`, `execution_control.py`, `spec.py`, and query metadata/discovery;
- read ownership and storage: `polylogue/storage/sqlite/archive_tiers/archive.py` and the owned SQLite snapshot/progress-guard path;
- Python adapter: `polylogue/api/archive.py`;
- MCP registration, declarations, response context, serialization budget, and error translation: `polylogue/mcp/server_tools.py`, `server_support.py`, `payloads.py`, and `declarations/registry.py`;
- HTTP and server-rendered WebUI paths: `polylogue/daemon/http.py` and `webui.py`;
- shared payload/schema generation: `polylogue/surfaces/payloads.py`, CLI output schema rendering, OpenAPI rendering, and generated WebUI types.

Relevant tests inspected included the existing query transaction, MCP registered-surface, MCP response-budget, daemon HTTP contract, Web reader continuation, query grammar/discovery, schema renderer, and execution-control suites.

Relevant Beads inspected:

- `polylogue-z9gh.9.1` (`in_progress`, P0): names the shared bounded query transaction as the sole production read boundary, requires complete continuation state, stable query/result refs, useful overflow prefixes, cancellation, and cross-surface parity;
- `polylogue-rsad` (closed/superseded): preserves the live incident evidence that successful oversized MCP reads were replaced by unusable metadata-only responses;
- `polylogue-t46.3` (closed/superseded): records the adapter-divergence problem and requires one canonical execution/cursor owner;
- `polylogue-20d.5` (closed): explicitly superseded by `polylogue-z9gh.9.1` for remaining eager streaming/pagination work.

Relevant history inspected:

- `fb9073cfc801a6b4d8150e6998c96b966e39047c` — terminal query pipeline and shared executor;
- `113d1af972f102b6c07462869d4d3697771d047f` — MCP response-budget envelope;
- `9163d0134f3d334960e4c249c96c5671919a9a06` — bounded agent-facing archive reads;
- `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd` — continuity hardening;
- `36e629b541174964b0816f09dbf32a2e6ea5781e` — parser-truthful query discovery corpus;
- `d5de5285e95bcfe10279a3898321a83efa9a6bf5` — current HTTP continuation and generated client work;
- unmerged `8898356acaa12bc36817bf87cbab2b1b75c4b186` — earlier archive-epoch continuation work. Its intent was useful, but it predates the current HTTP route and accepted unframed legacy cursors, so current source and this mission take precedence.

## Mechanism

### Canonical request and identity

`query_units_transaction_request(...)` is now the only constructor used by API, MCP, HTTP, and the server-rendered WebUI for an initial terminal-unit read. It clamps page size, normalizes offset, carries the exact expression and structural session filters, fixes the projection/order contract, and binds the result to a conservative archive epoch.

`query_ref` identifies the logical query independent of page/offset. `result_ref` binds that logical query to projection, order, and archive epoch, while remaining stable across every page.

### Stable frame

The archive epoch combines:

- `index.db`: schema version, session count, maximum rowid, and maximum `sessions.updated_at_ms`;
- `user.db`: schema version, assertion count, maximum rowid, and maximum `assertions.updated_at_ms`.

A cursor is validated before work and the same epoch is recomputed inside the owned read-only SQLite snapshot before row selection. If the frame moved, continuation fails as `query_continuation_stale`; it does not execute offset pagination against a moving relation.

This is stable epoch-bound re-execution, not a materialized snapshot held across requests. Archive mutation deliberately invalidates the continuation instead of risking duplicate or skipped members.

### `q2` continuation

The versioned cursor contains the complete canonical request, result ref, original issuance time, expiration time, and a SHA-256 checksum. It has a one-hour default validity window. Rebased storage pages, MCP byte-budget prefixes, and failure retries preserve the original issuance/expiration window.

The checksum detects accidental corruption and the tests exercise that behavior. It is not a keyed authenticity mechanism and must not be described as one. Legacy `q1` tokens are rejected as typed invalid continuations; no compatibility alias was added.

### Failure receipt

Deadline, SQLite work-budget, and controlled cancellation outcomes preserve:

- execution `call_id` and execution receipt;
- operation;
- stable `query_ref` and `result_ref`;
- failed page offset;
- a retry continuation for that same offset when the transaction is framed.

API callers receive typed exceptions. MCP and HTTP serialize typed deadline/budget errors with the receipt. Async cancellation attaches the same `QueryTransactionFailureReceipt` to `asyncio.CancelledError` as `query_receipt`; a disconnected client cannot receive an HTTP response, so cancellation is certified at the shared execution boundary rather than claimed as a transport round trip.

### MCP byte overflow

The MCP response context now carries the live transaction request and continuation seed. When serialization exceeds 25,000 bytes, `_budget_envelope` returns the largest fitting item prefix and creates a `q2` cursor from the exact retained count. This also repairs the terminal-storage-page edge: if the storage page had no continuation but MCP omitted a suffix, MCP still emits an advancing canonical cursor rather than losing those members.

The clipped page and compact metadata deliberately suppress storage-relative continuations; the top-level MCP continuation is the sole authoritative advance instruction.

### Typed adapter behavior

- API: continuation-only resume; override attempts are typed `invalid_continuation` exceptions.
- MCP: continuation-only resume through the registered production handler; invalid/expired/stale and deadline/budget outcomes are typed `MCPErrorPayload` JSON.
- HTTP: continuation-only resume; invalid/expired is HTTP 400, stale is HTTP 409, and controlled deadline/budget failure is HTTP 503 with a receipt.

`QueryErrorPayload.receipt` is optional. Unrelated daemon errors continue to omit the field instead of gaining `"receipt": null`. CLI output schema, OpenAPI, and generated WebUI client types were regenerated.

## Decisions

1. Use one `q2` continuation contract and reject legacy/unframed `q1`; no compatibility aliases are added.
2. Use deterministic epoch-bound re-execution rather than attempting to hold a materialized SQLite snapshot across public requests; any relevant mutation fails typed as stale.
3. Bind the frame to both index sessions and user assertions because either can change terminal-query membership.
4. Treat the top-level MCP continuation as the sole advance instruction after byte clipping; storage-relative continuations are suppressed in clipped payloads.
5. Preserve the original issuance/expiration window across storage pages, MCP rebasing, and retry receipts rather than refreshing TTL per hop.
6. Keep transaction receipts optional in the shared error schema so unrelated daemon error wire shapes remain unchanged.
7. Attach cancellation identity at the shared async boundary; do not claim a disconnected HTTP or MCP caller can receive a normal post-cancellation response.

## Surface matrix

| Surface | Production entry exercised | First-page bound | Resume input | Invalid/expired behavior | Stable-frame behavior | Host-protection receipt |
| --- | --- | --- | --- | --- | --- | --- |
| Python API | `Polylogue.query_units` → shared constructor → `QueryTransaction` → `query_unit_envelope` → real SQLite terminal executor | requested row limit (`9` in certification) | `query_units(continuation=<q2>)` with no parameter reconstruction | typed `QueryContinuationError` / `QueryContinuationExpiredError`; no page-one fallback | typed `QueryContinuationStaleError` before or inside owned snapshot | typed deadline/budget exceptions; async cancellation carries `query_receipt` |
| MCP | registered `query_units` handler → same transaction/SQL path → `_json_payload` / `_budget_envelope` | row limit plus 25,000-byte production envelope; certification proves actual overflow and retained prefix | registered call with only `continuation=<q2>` | typed `MCPErrorPayload.code`; no items and no facade restart | typed `query_continuation_stale` | deadline/budget JSON includes receipt; cancellation receipt is attached at shared async boundary |
| HTTP | loopback `DaemonAPIHTTPServer` → `DaemonAPIHandler._handle_query_units` → same transaction/SQL path | requested row limit (`9` in certification) | `GET /api/query-units?continuation=<q2>` only | HTTP 400 with `query_continuation_expired` or `invalid_continuation`; no items | HTTP 409 `query_continuation_stale` | HTTP 503 deadline/budget envelope includes receipt |

## Acceptance matrix

| Mission criterion | Implementation evidence | Certification evidence | Status |
| --- | --- | --- | --- |
| Bounded first page plus lossless continuation on each public surface | one constructor/decoder; API/HTTP row pages; MCP prefix rebasing including terminal-page overflow | 20-message corpus; API and HTTP first page contains 9; MCP asserts original response >25,000 bytes and `0 < returned_items < 9` | implemented and locally exercised |
| Continuation exhausts one stable transaction without duplicates or omissions | stable query/result refs; exact offset advance; archive epoch checked inside owned snapshot | all three drains equal the expected ordered 20 message IDs; uniqueness and one ref pair asserted; API cursor resumed through MCP and HTTP | implemented and locally exercised |
| Expired/invalid cursor fails typed and never restarts | `q2` decoder, expiry, checksum, canonical query contract, override rejection | API exceptions, MCP typed JSON, HTTP 400; no `items`; facade restart paths patched to fail if invoked | implemented and locally exercised |
| Cancellation/deadline/budget retains identity | `QueryTransactionFailureReceipt`; retry cursor at same offset; transport mappings | deadline and VM-budget forced through API/MCP/HTTP; cancellation test uses real shared controlled read and inspects attached receipt | implemented; transport-disconnect cancellation remains inherently non-returning |
| MCP discovery examples parse real grammar | current discovery resource remains backed by query metadata registry/parser | every emitted unit example is passed to `parse_unit_source_expression` | implemented and locally exercised |

## Changed files

Production and generated contracts:

- `polylogue/archive/query/transaction.py`
- `polylogue/archive/query/unit_results.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py`
- `polylogue/api/archive.py`
- `polylogue/mcp/server_tools.py`
- `polylogue/mcp/server_support.py`
- `polylogue/mcp/payloads.py`
- `polylogue/mcp/declarations/registry.py`
- `polylogue/daemon/http.py`
- `polylogue/daemon/webui.py`
- `polylogue/surfaces/payloads.py`
- `docs/schemas/cli-output/query-error.schema.json`
- `docs/openapi/search.yaml`
- `webui/src/api/generated.ts`

Tests:

- `tests/unit/archive/query/test_transaction.py`
- `tests/unit/mcp/test_query_transaction_certification.py` (new)
- `tests/unit/mcp/test_bounded_query_transport.py`
- `tests/unit/mcp/test_server_surfaces.py`
- `tests/unit/daemon/test_web_reader.py`
- `tests/unit/daemon/test_daemon_http_contracts.py`

No complete replacement files are necessary; `FILES/` is intentionally omitted. The final `PATCH.diff` is 111,521 bytes and 2,555 lines, with SHA-256 `b3c5e888dba3c562931d9a8b13bb58417dbd3d676ee08919525de646d50667e1`.

### Proposed dominated deletions

None. The patch does not delete existing tests, helpers, retired-name guards, or production routes.

## Apply order

1. Check out `bf8191b3f56aa40da8f271df7f3385c712825497`.
2. Restore/apply the three supplied dirty edits listed above. The reconstructed prerequisite patch has SHA-256 `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`.
3. Apply `PATCH.diff` with `git apply PATCH.diff`.
4. Run the repository-managed verification in a complete locked environment, including formatting, lint, strict mypy, generated checks, and the focused tests in `TESTS.md`.

`PATCH.diff` is generated against the authority commit plus the supplied dirty state. Applying it directly to a clean authority commit without step 2 may fail in the two overlapping files and would also discard the operator's existing work.

## Verification performed

The exact commands and outcomes are recorded in `TESTS.md`. A fresh isolated environment installed the repository and development extras with the real MCP and sqlite-vec dependencies; `pip check` reported no broken requirements. The final post-repair verification was:

- **103 passed** in 22.83 seconds: query transaction, cross-surface certification, MCP overflow, and all registered MCP surface tests;
- **90 passed** in 11.53 seconds: daemon HTTP contracts, Web reader `query_units`, and CLI/OpenAPI/WebUI generated-contract tests;
- **193 focused tests passed in total**, without test failures or stubbed production dependencies;
- Ruff lint and format checks passed across all 17 changed Python files;
- strict Mypy passed across all 17 changed Python files;
- compilation of all 17 changed Python files passed;
- `render cli-output-schemas --check`, `render openapi --check`, and `render webui-client --check` all reported sync OK;
- `git diff --check` passed;
- an independent worktree reconstructed `bf8191b3...` plus the exact supplied dirty patch, passed ordinary and `--whitespace=error-all` apply checks, applied `PATCH.diff`, compiled the result, passed whitespace validation, and matched all 20 patched paths byte-for-byte with the generating tree.

The selected tests exercised production query parsing, real SQLite archive writes and reads, execution control, the registered MCP handler under MCP SDK 1.28.1, response serialization, sqlite-vec 0.1.9 archive initialization, and a real loopback daemon HTTP server. A separate MCP stdio/client process was not exercised.

## Risks and limitations

- Ruff and strict Mypy passed on every changed Python file, but the repository-wide `devtools verify --quick` baseline was not run. Unrelated repository-wide policy, coverage, packaging, or integration gates may still expose issues outside the focused change set.
- The cursor checksum is corruption detection, not cryptographic authorization. The existing daemon authentication boundary remains responsible for access control.
- Stability is conservative epoch-bound offset replay. Any relevant index/assertion frame change makes a cursor stale; this can reject a resume during concurrent ingestion rather than maintaining a long-lived materialized snapshot. That is the safe behavior for this patch but may be operationally noisy on a continuously mutating archive.
- The epoch assumes terminal query membership changes are reflected through `sessions` or `assertions`. A future independently mutable query-visible relation must be added to the epoch contract.
- Cancellation receipt propagation is certified at the shared async transaction. HTTP disconnects and cancelled MCP calls cannot deliver a normal response to the disconnected caller; operators can use the attached/server-side receipt, while deadline and budget failures are returned normally.
- No live daemon, real MCP stdio session, browser, secrets, NixOS deployment, production client, or incident-scale archive was available. Client rollout and live archive proof remain local-lane responsibilities.
- No real 4.85M-block performance claim is made.

## Value of another iteration

A normal second pass should be a **small repair pass** if it consists of running the full repository-wide `devtools verify --quick` baseline and exercising one real MCP stdio/client session. Formatting, lint, strict typing, generated drift, focused tests, and independent patch reconstruction are already complete for the changed files.

A **substantial second pass** becomes worthwhile only if live concurrent-ingest testing shows that the conservative epoch causes unacceptable stale-cursor rates, if a query-visible relation is found outside the epoch, or if deployed clients require a separately approved cursor migration strategy. Those findings would change the frame/snapshot design rather than merely polish this patch.
