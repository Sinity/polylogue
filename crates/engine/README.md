# Provider state and convergence steps

`state::{claude,codex}` adapts the existing provider interpretation cores to
store-owned, pinned SQLite snapshots. It preserves typed storage/admission
failures through fallible provider callbacks and admits cumulative lookup and
encoded-delta traffic. Provider interpretation and serialization finish before
the writer transaction starts.

Each committed step retains its pending effects under its next predecessor
step number, including steps that do not advance a raw byte offset. Evidence
hashing, Claude contribution layout, Codex result assembly and canonical
publication remain explicit obligations. A provider reaching EOF does not clear
them. Required Claude body references come from the same pinned state basis;
the adapter performs no filesystem lookup.

`converge::PendingInstall` performs one provider preparation transaction per
owner turn: stage one admitted member, seal the complete delta, apply an
admitted member batch, or publish the scalar checkpoint. It rejects completion
claims. `recover_turn` discards an interrupted unsealed delta before reparse,
or finishes a sealed delta without interpreting against partly applied state.
The durable barrier blocks new source snapshots while existing pinned readers
retain their coherent predecessor view.

The tests use actual retained synthetic inputs, prepared writes, reopen and
owned-process exit. Store peer operations between turns establish a component
seam; they do not qualify a daemon or public-client end-to-end outcome.

The canonical worker returns its store read handles through the existing worker
join. Idle connections can retain SQLite schema caches; each parser turn opens
and validates a fresh source snapshot, then releases it before the actor applies
the result. Connection reuse does not resume an earlier view or change source,
generation, cancellation or publication checks.

Encoded-byte allowances are logical admission limits. Whole cells, source
record decoding, decoded objects, effects buffers, allocator/RSS use, SQLite
page work and arbitrarily large output fanout are not fully resource-qualified.
General Message/body selection, indexed contribution folding, search statistics
isolation, fairness and complete public-client behavior remain unfinished. The
limited selected terminal reference reader is described below; no live collection
is registered by this crate.

The model's `version_evidence::Framer` streams the owning source-evidence recipe
in admitted byte chunks. Its digest is distinct from normalized payload content
integrity. Providers still own exact membership/order/canonicalization; lost
ephemeral hash progress replays from zero. This prerequisite does not qualify
append-local hash updates or canonical publication.


`source_turn` computes one window from the existing provider cores. Its linear
commit guard releases framing continuation only after the actual prepared-store
publication receipt matches. `owner` connects explicit Codex and context-declared
Claude imports to one Writer actor,
one capture worker and one compute worker, with bounded metadata requests.
Restart enumerates retained imports and resumes or discards durable prepared
state before interpreting again. Peer requests can run between installation
turns; provider EOF and waiting tails keep canonical finalization pending.
Current scheduling drains one input before the next. The progress endpoint keeps canonical completion false. Fair per-input rounds,
arbitrary record capacity, physical IO/RSS and search completion remain open.

Codex effects use a fixed header and addressable typed units. A scalar cursor
reads one unit or boundary probe per step; whole inspection collects that same
cursor. The adapter layout version participates in the actor's parser basis.
`version_proof::Driver` pins one Codex evidence head and hashes the provider-owned
selected units in admitted slices outside writer transactions. Missing evidence,
count/slot mismatches and stale revisions cannot yield a digest. A lost process
replays hashing from zero. This proof alone does not certify source-field mapping,
canonical publication, lineage or search readiness.

`canonical_message` uses the provider-owned projection/presence/evidence cores
to prepare final-head supported Codex and Claude Messages with
typed supplied fields. Both providers use the same State/action/value/origin
producer and publication core. Claude work is scheduled by one-row keyset
lookups over retained pending locators; source ordinals determine Message and
block order, independently of scheduling key order. Archive revision and
provider evidence revision remain separate.
Worker computation produces one opaque action; the actor applies one value,
field, block, cursor or source-certificate transaction between peer commands.
The store selects its whole body only when no other input candidate competes.
Unsupported mappings stop with explicit pending diagnostics. Claude calls and
results use the existing staged block roots and immutable private summaries.
Signatures and derived outcome reasons are retained as non-ranking fields.
Global call pairing, native ToolItems and cross-input election remain pending.
Restart reconstructs preparation from its durable scalar cursor and active
value and origin producer. Before attaching a supported field origin, it retains
the exact record citation set, evaluates the source-owned selector against retained
bytes, and compares the streamed normalized digest and length with the ready field.
An immutable JSON descriptor transfers with that field and remains reachable after
parser-state replacement. Unknown selectors stay unavailable. This first path rebuilds roots and does not consume incremental dirty
Effects or implement general merge/fill. The actor has not yet scheduled historical
source-body retirement or full excision closure.

Value and evidence validation currently loop over a complete selected value or
Message on the worker with cooperative cancellation. Leaf and origin-descriptor windows serialize
one admitted field again for each chunk. Selector evaluation decodes admitted
whole source records; it does not establish a cumulative record-work bound. No whole-leaf Vec is required, but this
does not establish bounded worker computation, append locality or RSS. The CLI exposes only the selected terminal Message-reference consumer with
degraded coverage. Known Session listing/prose search and Session JSON reads are registered with degraded coverage. Export and the complete Claude Code demonstration remain unfinished.

Immutable import source-row settlement audits the retained step dispositions and
effects plus selected current source certificates. A fixed parser-token/basis
receipt in existing parser state survives restart; source EOF and canonical
iteration end merely start this audit. Unsupported effect/normalization cohorts
and retained malformed records stay unavailable. This does not settle Session
metadata/history/catalog, global freshness or search. Restart replays an unfinished
audit from its beginning. Claude's admitted whole Effects cell remains a whole
decode and may contain multiple bounded-admission obligations; universal worker
CPU/RSS/append-locality acceptance is still open.

Direct Session source qualification walks the pinned retained-input catalog,
not matching Message provenance. Claude main-context membership is source-owned;
unresolved subagent shapes remain named. Error counts use unique input/revision
record-start identities from committed disposition receipts, checked against the
same pinned parse accounting. Aggregate source receipt counts are not summed.
The read shares the selected-row snapshot, admits whole existing effect cells,
and checks its deadline/cancellation between catalog/cell/error steps. This does
not prove linked/global dependencies or full Session/search completeness.


`session_query` executes the registered typed Session subset against the existing pinned catalog. Admission dependencies qualify the searched population independently of matching rows. Selected human/model Text blocks build a query-local store-owned FTS corpus; all candidates are visited before matching/sorting/limit, while stored structured channels and origin readers retain their ownership. Queries expose known selected Message counts/times and retained hit references, never complete Session reductions. Current removal authority is checked again after evaluation. Canonical distinct-text relevance remains unavailable; the response discloses known activity order. General planner/paging, persistent search publication, linked subagent promotion, snippets and whole-corpus resource acceptance remain open.

Small field installation keeps the same source-selector verification. When the
normalized JSON value and its origin descriptor each fit the current value chunk
and read limit, and both chunks plus field key and next cursor fit metadata
admission, the worker verifies and hashes them before returning one typed field
operation. The store atomically installs their values, leases, field and cursor.
Existing producers and larger values retain the streaming path. Claude citations
coalesce only adjacent immediate-field donors with the same exact record range;
interleaved correction donors remain separate groups. These scheduling changes
preserve interpretation versions and immutable evidence recipes. They do not
establish a total worker-time or history-cost bound.
