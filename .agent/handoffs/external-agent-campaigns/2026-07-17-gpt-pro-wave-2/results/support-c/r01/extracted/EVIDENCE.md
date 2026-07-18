# Source, Beads, and history evidence

## Snapshot evidence

The attached Chisel overview and manifest report (supplied archive SHA-256 `47ad17ea5a44d148a3c58baa74da9cff650b51e46ceed38e41468810a0896df4`):

```text
Generated: 2026-07-18T013442Z
Source: /realm/project/polylogue
Git: master @ bf8191b3 dirty=true
```

The full commit is `bf8191b3f56aa40da8f271df7f3385c712825497`. The source snapshot's working tree, not a clean commit reconstruction, is the authority. Inspection found three supplied dirty edits, listed in `HANDOFF.md`; the final implementation diff deliberately excludes them and documents them as an apply prerequisite.

## Repository doctrine

`CLAUDE.md` states that substrate owns meaning and surfaces are leaf adapters. It directs semantic work into storage/insights or product layers before surface adaptation, uses strict mypy as the primary identifier/type net, prefers focused `devtools test`, and requires generated OpenAPI/CLI schema updates when embedded Pydantic models change.

The patch follows that doctrine by putting cursor/frame/receipt semantics in `polylogue/archive/query/transaction.py` and the owned storage snapshot, then making API/MCP/HTTP adapters consume those shared contracts. It adds no parallel production module or test-only facade.

## Current-source findings

### Shared executor existed, shared continuation reconstruction did not

Before this patch, all three `query_units` routes ultimately reached `QueryTransaction` and `query_unit_envelope`, but adapter ownership diverged:

- Python API accepted no continuation at all.
- MCP decoded a generic `q1` token locally, checked only a subset of fields, and reconstructed session filters/offset independently.
- HTTP decoded and validated continuation with its own separate formula.
- Query/result identity formulas were not consistently bound to an archive frame.

This meant the shared SQL executor did not by itself certify a shared resumable transaction.

### MCP overflow had a terminal-page data-loss edge

The existing response-budget wrapper could rebase a storage continuation when one existed. If an oversized storage page was logically terminal, however, the storage envelope had no continuation even when MCP kept only a prefix. The omitted suffix then had no advancing cursor.

The patch carries the live transaction request through the existing response-context seam and mints the exact retained-prefix cursor even for a terminal storage page. `tests/unit/mcp/test_bounded_query_transport.py::test_terminal_query_budget_trim_mints_q2_cursor_without_storage_continuation` isolates this production edge, while the cross-surface test proves full drain on a real corpus.

### Preflight-only epoch validation was insufficient

A cursor can be valid before a read connection opens and stale by the time SQL runs. The patch recomputes the same combined epoch inside the owned read snapshot before executing the page. A race that moves the archive now fails typed instead of being downgraded or replayed on a moving relation.

### User assertions can change terminal query membership

Terminal assertion/tag-related reads can depend on `user.db`, so an index-only epoch is incomplete. The combined epoch covers both `index.db` sessions and `user.db` assertions, and `ArchiveStore.read_query_epoch()` computes both from the owned snapshot/attached read-only user tier.

### Validity windows previously refreshed during transport rebasing

A cursor recreated on every page or MCP prefix could silently gain a new validity interval. The patch carries the decoded continuation seed through transaction, unit envelope, MCP budget trimming, and failure receipts so all descendants retain the original issuance/expiry.

### Error schema needed transport-safe optionality

Adding a receipt directly to `QueryErrorPayload` initially would make unrelated daemon errors emit `"receipt": null`. The final patch keeps the field in the typed/schema contract but explicitly omits it from unrelated error serialization. Only controlled transaction failures include it.

## Beads evidence

### `polylogue-z9gh.9.1`

Status `in_progress`, priority P0. Its description treats read failures as one implementation gap and requires the bounded query transaction to be the sole production read boundary. Its acceptance criteria require identical cross-surface plans/refs/cursors, complete continuation state, exact-once exhaustion, useful overflow evidence, and prompt cancellation/bounded execution.

Its July 15 incident notes record successful MCP reads being replaced wholesale by `response_budget_exceeded`, including empty continuation arguments. They explicitly require physical paging before full serialization, complete opaque state, useful prefix preservation, and interruptible work.

### `polylogue-rsad`

Closed as absorbed/superseded. It preserves the concrete oversized-response incident reports and requires that successful reads never become unrecoverable metadata-only envelopes. It calls for bounded first pages, stable refs, complete cursor arguments, and exact-once continuation.

### `polylogue-t46.3`

Closed as absorbed/superseded. It identifies API/MCP/daemon re-mapping and independent pagination/total/cursor ownership as the root divergence. Its acceptance criteria require one canonical plan and identical refs/page behavior across surfaces.

### `polylogue-20d.5`

Closed with the explicit reason: superseded by `polylogue-z9gh.9.1`, whose shared bounded query transaction owns the remaining eager streaming/pagination residues. It was treated as historical context rather than an independent current design authority.

## History evidence

- `fb9073cfc801a6b4d8150e6998c96b966e39047c` established terminal query pipeline execution through one executor.
- `113d1af972f102b6c07462869d4d3697771d047f` established the MCP response-budget wrapper that this patch repairs rather than replaces.
- `9163d0134f3d334960e4c249c96c5671919a9a06` and `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd` established the current bounded read/continuity substrate.
- `36e629b541174964b0816f09dbf32a2e6ea5781e` established parser-gated discovery examples; the certification consumes the emitted examples instead of creating a duplicate fixture catalog.
- `d5de5285e95bcfe10279a3898321a83efa9a6bf5` is newer than the unmerged epoch branch and contains the current HTTP continuation/generated-client route.
- unmerged `8898356acaa12bc36817bf87cbab2b1b75c4b186` demonstrated intended archive-epoch/result-ref binding, but it predated the current HTTP adapter, used `q1`, and tolerated empty legacy epochs. The present patch preserves the valid substrate intent while rejecting unframed/legacy cursors as the mission requires.

## Contradictions resolved

1. **Old branch versus current HTTP source.** The old branch assumed HTTP lacked continuation. Current master already had an HTTP continuation route. The patch repairs that current route and shares its decoder rather than cherry-picking stale adapter code.
2. **Legacy acceptance versus no aliases.** The old branch accepted an empty epoch for compatibility. The mission forbids aliases and requires invalid cursors to fail typed. `q2` therefore requires a frame and rejects `q1`/unframed input.
3. **“Stable transaction” versus mutable archive.** Holding a SQLite snapshot across public requests is not part of the current process model. The safe coherent substrate available here is epoch-bound deterministic re-execution: unchanged frame resumes exactly; changed frame fails stale. Documentation and tests make this explicit.
4. **Transport cancellation versus returned receipt.** A disconnected caller cannot receive a normal response. Deadline and budget receipts are returned through every surface; cancellation identity is attached at the shared async transaction and remains available server/API-side. No impossible HTTP response-after-disconnect claim is made.
5. **Generated model field versus existing error shape.** The receipt belongs in the shared typed contract, but ordinary errors must not change wire shape. The field is optional in schemas and omitted unless populated.

## Revision validation evidence

The final continuation revision was validated in a fresh isolated project environment with MCP 1.28.1, sqlite-vec 0.1.9, pytest 9.1.1, Pydantic 2.13.4, Ruff 0.15.22, and Mypy 2.3.0. The two final focused selections passed 103 and 90 tests respectively, for 193 total. Ruff lint/format, strict Mypy, compilation, and all three generated drift checks passed on the changed paths.

The dirty snapshot prerequisite was reconstructed independently from the bundle authority and working-tree archive. Exactly three tracked files differed; the resulting 2,272-byte patch has SHA-256 `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`. Applying that prerequisite followed by the final mission patch succeeded under both ordinary and strict-whitespace checks. All 20 patched paths then matched the generating tree byte-for-byte.

The final `PATCH.diff` is 111,521 bytes and 2,555 lines, SHA-256 `b3c5e888dba3c562931d9a8b13bb58417dbd3d676ee08919525de646d50667e1`. Added-line scans found no placeholder markers, bare `pass`, `NotImplemented`, source-archive names, temporary paths, Git binary patch, or copied snapshot content.

## Evidence boundaries

The certification provides local synthetic incident-shape proof, not operator live proof. It does not establish real archive scale, concurrent ingestion rates, deployment configuration, client migration, real MCP process behavior, or 4.85M-block performance. Those remain explicitly unverified.
