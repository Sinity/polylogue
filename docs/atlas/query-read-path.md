# Query and Read Path

## Area boundary

The query layer turns explicit user intent into a typed SQL plan and bounded
pages. Read surfaces expose stable archive objects and projections through
the CLI, API, MCP, and daemon without reimplementing substrate semantics
(`polylogue/archive/query/expression.py:1-80`; `polylogue/archive/query/transaction.py:1-100`).

## Route

```text
CLI/API/MCP request → filters + expression DSL → query plan → page/viewport
                                      ↓
                              index/read models
```

The CLI is query-first: root filters precede `find`, and query intent must be
signalled by `find`, a quoted expression, or field syntax. The query DSL is
lowered to SQL; it is not a grep-like post-filter. Pagination and cancellation
are part of the route contract (`polylogue/archive/query/transaction.py:1-100`).

Saved query definitions canonicalize the typed predicate wire shape before
persistence. Only declared grammar tokens and field aliases are normalized;
opaque operands and mapping keys stay exact. Boolean `kind=and/or` children
are sorted for identity, while directional sequence steps retain order. The
production evaluator executes the persisted predicate, so canonicalization
must preserve its selection meaning (`core/query_identity.py`).

Compact scalar alternatives such as `id:(A|B)` and `title:(alpha|beta)`
select the same session relation as explicit OR, including under NOT. Each
title alternative retains the ordinary substring-match semantics.
Quoted scalar operands remain single literals, including pipes and whitespace.

A temporal read with a resolved single-session reference retains that session
even when its original selection contains text or ranking criteria. An absent
reference selects a query set; a missing selected reference stays empty.
Selected temporal reads do not admit a vector snapshot; ranked query-set reads
retain their ordinary vector admission and named availability gaps.

## Shared read input

`ReadRequest.normalize` and `read_contract_schema` share the flat input
validation declaration. `{}` selects the summary preset. For example,
`{"preset":"dialogue","query":"needle","output_format":"json"}` overrides
selection and rendering; `views` is an array of declared view names. Unknown
keys, nested `selection`/`projection`/`render` objects, and malformed values
are refused. Query and projection owners additionally validate semantics.
Python callers pass an already lowered `SessionQuerySpec` through the explicit
`selection` keyword; it is not a machine input field.

Hydrated session rows obtain terminal state from the canonical structural
classifier without computing a full profile, costs, timing, or topics. Profile
construction reuses its existing actions with that same classifier. Missing
structural evidence remains `unknown`.

## Read identities

Use generated session, message, and block identities for exact reads. Use
public `origin` filters, not provider-wire names. Lineage-aware reads compose
parent prefixes and report cycle or dangling-branch-point status rather
than silently claiming completeness
(`_composed_transcript_plan` in `polylogue/storage/sqlite/archive_tiers/write.py:3399`).

Hydrated message classification reads textual markers only from declared `TEXT` blocks. Thinking and tool content remain in the complete display text without becoming prose classification evidence. Structured tool block types retain precedence; messages without blocks classify their supplied text. Explicit stored non-message types remain authoritative.

Ordinary block reads preserve the stored language and media type through typed records and domain hydration. Message streams plan inherited prefixes as bounded lineage segments and hydrate blocks and message-owned attachment references in batches on one read snapshot; closing a public stream closes its nested reader before releasing that connection.

Read frames retain the identity of the selected physical leaf through native
open, independently of subsequent active-pointer promotion. Resuming an idle
continuation rebinds a stale live frame before proving the anchor against current
rows; a predecessor's surviving anchor cannot authorize a successor page.

## Scoped ranked reads

Semantic and hybrid session requests qualify the canonical SQL and residual scope before ranking. The held ArchiveStore connection supplies current prose and occurrence identity to the retained-vector TEMP projection. Exact L2 scoring covers every eligible stored output; near takes the minimum over all stored seed outputs. Session witness selection, complete hybrid lane ranks and n-ary RRF precede the final session window (`polylogue/archive/query/archive_execution.py`, `polylogue/storage/search_providers/sqlite_vec_queries.py`, `polylogue/storage/sqlite/archive_tiers/archive.py`). TEMP relations use FILE storage selected at connection acquisition. Successful full lane settlement is explicit in `completed_lanes`; unavailable or failed lanes remain named gaps.

Session-level existential DSL filters stay independent of the semantic witness. Semantic under a correlated unit predicate or a non-session terminal source remains a typed compilation refusal (`polylogue/archive/query/expression.py`).

## Terminal outcome

Every row-bearing envelope carries one typed outcome -- `ok`, `empty`,
`degraded`, or `error` -- decided at the operation boundary
(`polylogue/surfaces/outcome.py:1-60`). `empty` means the declared scope
completed and holds zero rows; `degraded` means named gaps shaped the answer,
so absent rows may be the gap rather than the archive; `error` means no valid
answer. Surfaces serialize that decision and map it to their transport: the
CLI exit table is `OUTCOME_EXIT_CODES`, and the daemon's readiness chips are
projections of it. A surface that re-derives readiness from `not rows` has
reintroduced the ambiguity the type removes.

## Controlled-read boundary

A read may open the archive itself only through one of the two declared
read-boundary owners; every other direct open is an explicit writer-lease
open. The invariant, its owners and its change procedure are one stanza in
`doctrine.md`. No gate censuses direct opens any longer;
`tests/unit/daemon/test_surface_data_boundary.py` refuses connection openers in
the surface packages.

A read view that computes its own answer in-process is a different question
from this one: those are declared, and shrink-only, in
`polylogue/cli/read_view_registry.py:95-110`.

## Surface ownership

- `polylogue/cli/` owns command grammar and human/machine presentation.
- `polylogue/api/` owns the Python facade and typed result payloads.
- `polylogue/mcp/` owns capability-gated operation dispatch.
- `polylogue/daemon/` owns transport, lifecycle, and serialized mutation
  admission; its HTTP/UDS readers use the same product routes.
- `polylogue/analysis/` owns descriptor-driven derived projections.

Surfaces should call shared operations, query planners, and insight
descriptors. A surface-specific SQL query is a design smell unless its
read-model contract is explicitly owned there.

## Gotchas

New Click parameters on query verbs go last: positional shifts silently
reroute arguments. Stable JSON output requires schema and parity checks across
CLI/API/MCP. A successful page is not proof that derived models are current;
readiness and staleness are explicit fields. Never turn unavailable data into
an exact zero or infer tool failure from prose.

## Verification route

Begin with the focused query or surface test through `devtools test`. For a
cross-surface change, run the relevant CLI/API/MCP parity tests, pagination and
cancellation coverage, then the generated surface check. Use `devtools why` to inspect a managed verification refusal or
failure before interpreting a receipt.

## Large session evidence

`file-edits` and `web-content` share the byte-bounded evidence owner in
`polylogue/operations/evidence_window.py` and `evidence_payloads.py`.
Small rows remain ordinary objects in `rows`. An oversized row is delivered
as `row_fragment` instead, with its row offset and ordered field fragments.
Each field declares its name, `encoding` (`utf-8` or `json`), byte `offset`,
`total_bytes`, and `data_base64`. Concatenate decoded bytes in offset order;
decode UTF-8 only when the entire field is present, then parse JSON for
`encoding=json`. Empty strings, nulls, booleans, and structured patches keep
their normal projected values. The fragment's `complete` completes that row,
not necessarily the relation.

`returned`, `offset`, `next_offset`, and `total` count completed rows, never
fragments. A page can therefore have `rows=[]`, `returned=0`, and an advancing
continuation while delivering part of a row. Continue until the window's
`complete` is true. The token binds both row and field-byte coordinates to
the original result, snapshot, and expiry; changing the transport byte budget
does not change result identity. SQLite reads only bounded field slices on
resume. Evidence insert/update/delete advances the archive frame so callers
cannot silently assemble one row from two revisions. CLI, Python API, and MCP
preserve this contract; MCP supplies its smaller delivery budget to the owner.

Owned user-target existence probes and vector reads use the shared bounded
compute owner with `interactive-read` admission and their encoded request-byte
charge. Cancellation keeps the original result Future alive until its creator
has physically settled SQLite handles; a supplied vector snapshot remains on
its original creator. This charge describes request bytes, not decoded-result
memory. Vector publication and daemon transport waits retain their own owners.

## Thread search
The public thread insight route searches session identity, title, repository URL, branch, support level, and the payload's thread/member support signals before the requested result window. API, MCP registry projection, and insight exports share `ArchiveStore.iter_thread_insights`. Strong and moderate threads remain distinct; a root without a materialized profile remains readable from the same session evidence as an exact public thread read. Profile absence does not silently consume a page slot. Search does not hydrate all threads before selecting a page. The former async thread-list adapter and its row-only mapper are retired with their unused query DTO; exact retained thread-record reads still use their existing profile owner.
## Aggregate selection

`query.aggregate` reduces the same canonical distinct session relation used by scalar scope reads. Explicit IDs, lexical/action matching and structural predicates intersect before order, limit, sample and offset; count, statistics and grouping reduce that selected window, including `latest`. Named aggregates honor group-key sort direction before the result window. Multi-field group identities retain JSON null for missing values so a literal string such as `[missing]` remains a separate group. Content-excluded counts use the shared survivor walk and apply the requested window after exclusion. Ordinary list totals continue to count every survivor independently of their presentation page. Statistics with content exclusion remain a typed refusal.

Message-branch predicates retain the default top-level session scope; only explicit session lineage selectors or root choices change that scope. Row `fields`/`select` projections cannot be combined with `count` or `agg` terminals in either order. Boundary errors offer equivalent names accepted at the requested boundary, and DSL discovery uses the actual grammar metadata rather than internal plan attributes.

The CLI message walk narrows every continuation request to the remaining requested delivery, through the canonical session-read window contract. It preserves the returned continuation and next offset; it does not trim a wider page after advancing its cursor. Full exports continue using their bounded window size.

Repository attribution treats structured cwd, file and checkout paths as complete literal paths, including whitespace and punctuation. Writers, repository materialization, relative-path projection and stored profile names preserve those literal values, including a trailing blank in a checkout basename; a neighboring trimmed name is a separate repository. Git-root discovery observes the current filesystem on every call, honors Git ceilings and linked worktree markers, and never retains a cached absence or enclosing root across topology changes. Remote/name lexical parsing is separate; `file://` keeps the existing URL path-component interpretation: authority and fragment are excluded, and percent escapes are not decoded. Prose token extraction is not part of structured path normalization.

Session-list envelopes from both full sessions and summaries use the canonical row projection for repository and working-directory display names. Full repository URLs and working-directory paths remain in their declared domain fields; explicit presentation overrides remain supported.

Query-unit capability rows include their executable field names from the canonical unit metadata. Capability search covers these fields and existing operator and lowering bindings, so structural fields can be discovered without first guessing a unit.

Disconnected archive reads keep admission with the original worker until physical cleanup completes. Their bounded caller drain waits do not cancel that worker, and its eventual exception is observed without creating a second loop-level failure report.

Profile and latency insight readers normalize comma-separated repository and tag scopes through the same canonical CSV and session predicates as session queries. Values within one filter are alternatives; repository and tag filters intersect before ordering and pagination.

Session-list projection rows bind their public names to existing renderer profiles, CLI option and execution metadata, and evidence-family contracts. Adding a name with a declared renderer updates these consumers together; removing a name retires it from their public vocabularies. Unknown renderer families and collisions with another read view refuse at contract construction.

In-memory action-sequence matching streams completed all-pairs witnesses with one iterative search path. It does not cap intermediate candidates; the existential answer peeks the first completed witness from the same stream. Ordered, adjacent and elapsed-time edges retain their declared semantics.

Python transcript anchors resolve and read their bounded page on the same controlled archive snapshot, with canonical domain hydration and continuation framing. Exhaustive Python topology helpers explicitly request the complete selected graph on one repository snapshot; ordinary topology calls retain their bounded node/edge windows and continuation.

Warm daemon query results refresh relative-time display fields from the retained timestamps through the canonical row projection. Cached selection rows, ranking, continuation and snapshot authority remain unchanged.

Action rows and tool episodes retain the canonical `outcome_unknown_reason`. A paired result with an unknown verdict remains distinct from a missing result or an ambiguous association; episode caveats describe those separate evidence states. Append outcome reconciliation excludes empty tool IDs, matching the canonical association owner.

Tool-episode context reads the preceding and following three messages from the canonical composed transcript on the same read snapshot as the selected episode. Post-context starts after the paired result when present. Ordered prose blocks retain message boundaries and multiline text; `next_action` preserves the complete first post-result message text.

CLI session query pages carry `snapshot_epoch` from the pinned generation path and the complete canonical Index/User relation frame. List and ranked selection walks require that frame on every page and refuse `query_continuation_stale` if it changes, even when totals are unchanged. Query cache hits must match the current pinned frame before display decoration. User assertion changes invalidate the frame within the same Index generation. The frame covers Index/User selection inputs; semantic retrieval also depends on its separate pinned embeddings snapshot and remains outside cache reuse. Standalone unbound envelope builders omit the optional frame; a proven missing explicit scope runs the canonical query on the actual pinned reader and returns its empty verdict and frame. Failed lookups do not mint a frame.

An explicit session ID bypasses query selection only when identity is its sole predicate. Boolean, tag, repository and other filters still run through the canonical session selection for root transcript shortcuts and single/multiple-session action resolution. Empty filtered scopes retain their declared empty-page or cardinality refusal; they never read the named session solely because its ID exists.

CLI selection adapters carry the operation's rows, terminal outcome and answering authority together. Cardinality and mutation guards require an authoritative verdict before using IDs. Selector and dialogue-set display preserve the supplied outcome after rendering; successful row projections cannot erase gaps in the selection. Unbound missing-session lookup failures remain distinct from a pinned empty query.

Query-set summary and transcript exports select through resident `cli.query` and hydrate through bounded `session.read` domain pages. `session_projection=domain` uses the canonical session hydration fields, including tool blocks; the default `archive` projection keeps the existing archive transcript vocabulary. A selected read sends `selection_epoch`, the query's actual Index/User view token. The resident reader compares it before cache lookup or hydration on every page. Dialogue and registered per-session handlers carry the same declared epoch through their resident read requests; standalone unbound reads omit it. Domain transcript continuations bind their projection and selected epoch. A changed view refuses rather than combining sessions from different snapshots. The CLI stages the rendered set before output, then reports the original selector and hydration verdicts; a named gap remains degraded even when every returned session reads successfully.

This materialized client selection does not implement scalable resident mutation references or bound the memory required by a single fully rendered session. Existing operation-result bounds also remain an unmet transport obligation for an individual transcript row larger than one deliverable window.

Corpus compaction is an operator-budgeted projection over the pinned selected
sessions, without changing source evidence. Its executable degradation order
is clipping, exact adjacent run collapse, source-reference skeletons, item
drops with a manifest, then an index-only pack or typed envelope refusal.
Collapse keeps one original-text representation, an actual occurrence count
and every run member's reference; skeletons keep typed identity/provenance
without prose. Markdown exposes the same count, degradation and outcome as
the typed pack. Budget reductions and missing manifest detail are named gaps;
a no-item index-only pack stays degraded, and CLI exit follows that outcome.

Per-session included and dropped estimates partition original source prose.
Clip markers and run-count metadata consume the complete serialized-pack
budget but do not claim retained source tokens. Drop counters count reduction
events by stage (a source item may pass through several); token differences
are transferred once. Detailed omission rows may yield to the requested
budget with `omission_rows_truncated`; session totals remain exact unless
`session_token_totals_truncated` explicitly names missing totals. Budgets below
the final typed envelope are refused rather than enlarged.

## Assertion export and identity reset

Assertion export reads `created_at_ms, assertion_id` order through the resident
`user.assertions.export` v3 contract. The first request streams that pinned User
selection into a private SQLite relation, sorting once and counting inserted
rows. `limit` selects the chronological export prefix, including an explicitly
empty prefix; page size selects only a transport window. Later pages seek the
owned relation by ordinal, without rescanning or counting User assertions.
The opaque `selection_ref` binds the authenticated principal, filters, limit,
and assertion-only User frame. Assertion changes refuse continuation; unrelated
Index ingestion and User settings do not. The original attached User authority
remains required, including for an empty export. The final page remains
replayable until `user.assertions.export.release`; release reads no User tier.
Equivalent starts for the same principal, filters, limit and assertion frame
share one immutable image with independent release references held in a private
SQLite relation with a bounded page cache; abandoned starts add no resident
per-client entries. A release
cannot invalidate another client's final-page replay. Observing a newer
assertion revision retires older image bytes after page reads settle; their
remaining handles still refuse continuation as stale. The daemon deletes
abandoned current images after its exchanges physically settle on shutdown.
There is no selection expiry or population cap.
CLI JSON and JSONL exports stage the complete walk, release the image, then
publish output; failed and cancelled walks also release it.
Python callers use `iter_assertions_for_export`. One assertion's payload remains
proportional to one row.

Identity reset freezes the complete selector on private disk before publishing
bounded canonical plans through an Audit preview batch. The client retains only
that sealed request reference. Confirmation submits it to
`mutation.identity-reset.authorize`; apply submits only the resulting sealed
authorization request to `mutation.identity-reset`. Each continuity page holds
at most forty plans, each plan at most the declared mutation page size. Selection,
authorization and reservation complete before any session effect. New matching
arrivals remain outside the frozen selection. Target pages seek immutable Audit
part and target ordinals without decoding the complete selection.

There is no implicit execution deadline or confirmation expiry for this accepted
reset custody. Audit re-proves the originating sealed batch, authenticated
confirmation intent, principal, exact part/hash and uncancelled phase at issuance,
reservation and consumption, including journal replay. A reset authorization's
real issuance timestamp is also its expiry: it has no transferable standalone
lease. Other expired previews and authorizations retain their ordinary refusal.
Stale plans still refuse. Cancellation or refusal after an applied prefix reports
the partial batch, never a completed untouched suffix.

The typed historical receipt sums recorded suppression, absent-index and deleted
row counts across completed parts. Deleted rows belong to each completing apply;
recovery does not reconstruct a lifetime deletion total. CLI dry-run and JSON ID
arrays stream through staging; dry-run writes Audit previews but changes no
sessions. Python target memory and continuity payloads are bounded by a page;
Audit disk remains proportional to the selected population.

## Durable setting reads

`setting get` and `setting list` use the resident `user.settings.get` and
`user.settings.list` operations. They pin one read transaction in `user.db`
and verify archive identity before and after the read. Missing or corrupt User
authority refuses; an absent Index, Source or Audit tier does not withhold
durable preferences. Reads require the daemon and never fall back to the local
Python facade. An explicitly supplied Index version is observed through the supported read-only tier reader and compared before the User read; wrong versions or a missing Index refuse. Omitting the condition keeps the User-only route.

## Resident insight pages

The eleven registered `analyze insights` list commands call `insights.list`
through the daemon. Its closed discriminated request and result branches use
the registry's existing query and item models. The canonical page reader is
shared with the Python facade and executes on the resident pinned archive
snapshot: origin tag rollups are merged before paging, and cost estimates are
enriched and filtered before paging. Missing daemon or unavailable insight
authority refuses instead of opening a local archive. `ops insights status` also calls the resident `insights.readiness` route on
the same pinned reader, preserving the canonical selected coverage and convergence
verdict. Its named pending-convergence outcome remains visible with zero rows.
The `ops insights audit` command also uses resident `insights.rigor`: every registered product is sampled on that same pinned reader, including explicit uncovered or exempt entries. Per-product read failure remains a named degraded outcome. `ops insights hermes-health` uses the resident `insights.hermes_health` diagnostic and the configured Hermes source root. Missing or unreadable derived tiers remain measurement gaps rather than prerequisites for the diagnostic. Python callers share the operations composer. The `ops insights export` command uses resident `insights.export_bundle`; the Python facade uses the same composer. Version 2 JSONL bundles exhaust forward cursors on one pinned Index/User relation, merging origin tag rollups before the page cut. Each row is written directly to owner-private staging. Missing or divergent insight authority withholds its file and records the reason; valid degraded rows remain exportable. Cancellation or any exception before directory publication closes the original reader and removes staging. Publication completes the filesystem effect; it does not modify archive tiers. Individual insight objects and their nested fields still require memory proportional to one serialized row.


`ops insights fable-packet` compiles through resident `insights.fable_packet` on one pinned Index/User snapshot. Canonical delegation keyset pages and annotation pages exhaust the population without an outcome-changing row ceiling; cohort operands spill to disposable disk-backed SQLite, and the complete population digest is hashed row by row. Structural coverage is rescanned on the same pinned relation; only selected or labelled structural refs remain in memory. Declared template/stratum counts still require memory proportional to distinct output keys; matching annotation operands and label evidence remain proportional to label cardinality. The manifest binds the actual query frame. Missing annotation schema, batches or supported evidence remains a named `not_supported` packet and degraded outcome. Requested sample size and exact-template sensitivity retain their declared semantics; cancellation reaches each evidence page and compiler stage. The CLI does not open a local facade for this read.

Annotation imports return a committed immutable batch/count summary through CLI, MCP and Python, independent of later evidence reads. Resolving the returned `annotation-batch:` ref pages exact assertion references followed by validation errors through `items`, `total`, `offset` and `next_offset`. Error items preserve the original failure fields (including line and row key) and both failure/error ordinals; empty or non-array error documents remain complete failure items. The original User batch is the sole authority. Count and selected items use one User read transaction. SQLite parses stored JSON and may scan/sort entries for each ordered offset page; Python decodes only the selected window. Scalar and metadata previews retain their explicit partial-evidence descriptors. Collection preview caps and response-row duplication are retired.

## Thread search

The public thread insight route searches session identity, title, repository URL, branch, support level, and the payload's thread/member support signals before the requested result window. API, MCP registry projection, and insight exports share `ArchiveStore.iter_thread_insights`. Strong and moderate threads remain distinct; a root without a materialized profile remains readable from the same session evidence as an exact public thread read. Profile absence does not silently consume a page slot. Search does not hydrate all threads before selecting a page. The former async thread-list adapter and its row-only mapper are retired with their unused query DTO; exact retained thread-record reads still use their existing profile owner.


## Attachment library session scope

The attachment library applies a session filter to its canonical composed transcript before selecting the requested page. Parent-message references through the inherited branch cut are included; references after the cut and foreign messages are excluded. Rows retain their physical owning session, message and provider reference attribution. Both the pinned HTTP reader and asynchronous repository use one shared SQL selection and the existing snapshot/lineage owners. Incomplete lineage refuses the read instead of claiming a complete child-only library; an absent materialized session profile is not a lineage fault. Pages use the same newest-session, transcript-position, attachment-ID and reference-ID order. Planning retains segment metadata, without hydrating all messages or collecting every message ID. Existing offset and SQL relation scans can still scale with the selected relation.

Command-shape usage streams the pinned action relation through the existing shell
normalizer into a disposable SQLite fold. It retains execution multiplicity and
counts distinct sessions before selecting the requested aggregate page. Python
memory holds one command and the returned page; the fold and SQL sort can use
disk proportional to the selected actions and distinct session/shape pairs. Each
page still scans its selected input relation and counts aggregate groups to
resolve the declared Python slice operands, including negative or oversized
integers, before binding finite SQLite page bounds. Cancellation checks cover both
normalization and scratch SQL, and scratch resources settle before the read
returns. A repository filter projects that admitted repository; unfiltered
reads retain the alphabetical representative. MCP lowers its public `repo`
operand to the registry model's declared repository field.

Archive insight contract version 12 represents an unobserved materializer,
inference, or enrichment version as `null` (or omitted when the selected
renderer excludes null fields). A missing session profile cannot certify the
current materializer version. Available thread and latency projections remain
readable with unknown provenance; profile reads still require a profile row.
Recorded versions are returned unchanged. Query-time projections may declare
their own known projection version independently of profile materialization.

HTTP API and server-rendered session lists compile the same complete query
specification and execute through the canonical summary or search-envelope
route on every page. Explicit ordering, similarity and continuation operands
reach that owner without a first-page storage lowering. Ranked metadata comes
from the canonical envelope and its actual lane execution; the HTTP readiness
chip projects its terminal outcome. An unavailable search index remains an
explicit degraded HTTP envelope, including when no hits can be returned.

Ordinary list pages and totals share one pinned read and the canonical plan.
Page windows retain latest and sample semantics; ordinary totals count the
complete eligible scope, while latest reports at most one. Counts apply content filters in bounded
candidate batches without collecting the complete result. Random lexical
post-filtering streams distinct session identities; SQLite's FILE temporary
storage owns sorting and deduplication instead of an archive-sized Python set.

Canonical ranked search classifies missing or incomplete message FTS, SQLite contention, and unreadable storage as `SearchIndexUnavailableError` before surface rendering. Unrelated SQL failures propagate. HTTP consumes that typed refusal as an explicit degraded search envelope with unknown total, never an executed empty result.

Quoted repository operands preserve their literal whitespace, pipes and commas. Public repository CSV filters preserve each comma-delimited segment exactly; typed collections retain each member. Facets preserve stored repository names and root path characters; remote URL labels retain URL cleanup. Query explain field clauses expose the same quoted flag used by execution.

Typed session list and search scopes share the generic read's explicit-reference resolver. Exact IDs, unique prefixes and the outer `session:` namespace resolve before SQL filtering on the pinned reader. Action-lane lexical counts and rows use the same action filter, and hit payloads report that lane. Transcript domain windows retain their resolved session identity for message envelopes while continuations retain the original request selection. Their epoch checks use the repository's explicit Index path, including a selected Index that differs from the default root path.
