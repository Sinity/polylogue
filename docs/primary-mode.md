# Primary mode

Polylogue on Sinex: what changes and what does not, what Polylogue sends, how it applies Sinex's confirmed stream, local state, write identity, several machines, Sinex's lifecycle applied to Polylogue's materials, moving user state, the cutover, and cross-system queries from Polylogue's side.

The producer client, the NATS ingress, the confirmed stream, refusals, kind registration and the lifecycle on the wire are Sinex's contract, in Sinex's [producers](https://github.com/Sinity/sinex/blob/master/docs/producers.md) and [lifecycle](https://github.com/Sinity/sinex/blob/master/docs/lifecycle.md) pages; this page states only Polylogue's side. The two modes are introduced in [architecture.md#modes](architecture.md#modes).

## What changes

Capture is identical in both modes ([capture.md](capture.md)); parsing, the parsed model, queries and every surface behave the same. What changes is where bytes and records are stored and how the local index is written.

- **Bytes go to Sinex as materials** through Sinex's producer client: each captured input, each appended step of a growing file as a part, export ZIPs whole. A document its source saves whole is a material sent as its content-defined chunks ([archive.md](archive.md)): the frame names the chunks in order and carries every chunk's encoding, and Sinex stores only the chunks it does not hold and records the material once the chunks read back as its bytes. The input's extents then name the chunks, and `index.db` catalogs each chunk's blob. No source byte is stored outside Sinex.
- **Confirmed originals drive local parsing.** Polylogue reconstructs its local model from Sinex-confirmed source materials. Sinex also retains declared native AI message, tool, terminal, file and work facts with source evidence and time quality, usable independently of Polylogue. A round trip through a copy of Polylogue's parsed product is not an intermediate authority; the local SQLite index remains reconstructible.
- **Authored records originate from request materials.** Each authored write is a request material (the write as received, with its request ID and computed author) plus its event. Its targets (sessions, messages, blocks, assertions, pages) are references in the payload with their `seen_hash`, never Sinex parents. A parser-upgrade replay of a session therefore archives the session's readings and their derived chain but never a note, judgment, highlight or page slot about it; those references resolve to the new readings by natural key. Marker assertions travel with their derived IDs, so the same marker sent from two hosts is one record.
- **Sinex-confirmed state alone enters `index.db`.** One durable subscription supplies confirmed originals, facts and lifecycle changes in commit order. Local parsing and application save the processed position in the same transaction as their complete result. A new stream that starts at a current-state origin first supplies an owner-certified indexed snapshot; the same writer then follows its deltas. This covers this host's writes, other hosts' writes and a rebuild alike. Polylogue's consumer has its own grant under Sinex's output rules, so it receives its own payloads whole; payloads too large for a message travel as material references and are fetched through the same client.
- **Lifecycle changes arrive on the stream** and are applied by the same consumer in commit order. An archived AI reading stops contributing to the local model. The affected source chain retracts through the ordinary writer, which selects the readings that other sources still supply; parsing the original bytes reconstructs the eligible contributions. Restoration makes the original contribution eligible again. Raw material lifecycle remains separate from the lifecycle of an interpretation. A tombstoned reading retains its reference. Session tables recompute for the sessions touched ([measures.md#rules](measures.md#rules)).
- **Sinex's lifecycle applies to Polylogue's materials** as to any source's, through Sinex's own surfaces. Polylogue itself offers no deletion in either mode ([architecture.md#deletion](architecture.md#deletion)).
- **Settings shared with Sinex.** The logical day, quiet hours and the interrupt budget are Sinex's settings; Polylogue has no value of its own for them, and its interrupting notices are shown by Sinex's deliverer ([operations.md#notices](operations.md#notices)).

Admission acknowledgments, publisher positions and locally processed positions
remain distinct. Acknowledgment proves Sinex admitted the write; publication
makes that committed state available to consumers; local progress proves the
corresponding parsing and application completed. Durable authored intent stays
available offline and retries with its original identity, without claiming
confirmed visibility or advancing processed progress before that work finishes.

## Producer connection and declarations

Before writing frame bytes, Polylogue checks Sinex's producer welcome with an
empty hello. This checks the protocol and sealing configuration before its
producer-owned kinds exist. It then registers its `polylogue.*` declarations
and requires the acknowledgment to name their exact definition hashes before
sending materials or events. Each event selects its declaration by name and
version. A refused hello or frame returns Sinex's reason. Retrying an operation
with a durable refusal returns the same reason from its verified acknowledgment.

Each acknowledgment is one typed result (admitted, kinds registered, replayed
or refused), and Polylogue checks that the result fits the operation it
answers: a kind registration is answered by its registered kinds (or, when it
declares nothing, by an empty admission), a replay by a replay, and every
other frame by an admission whose material is the prepared one. A complete
producer acknowledgment can carry no admitted-event position: an
empty frame, a repeated event or kind registration may admit no new events;
a replay may archive readings without admitting replacements. Polylogue keeps
that absent position in the original acknowledgment. New admissions and replacements
require their event position. A acknowledgment alone never advances the local index.
Ungrouped metadata operations can finish with their durable complete acknowledgment.
A group holding original source bytes also waits until the local index has
applied a complete prefix in the same scope through an owner head observed
after the acknowledgment. Sinex reads that head from its committed counter.
A head observed before publication, or an incomplete prefix, retains the group.

The producer queue retains its existing physical `reading_version` and
`export_keys` columns so committed pending operations keep their database and
membership. Those columns are inert: new operations write no reading version or
export keys, and source-head lookups depend only on physical source identity.
Parser output is not retained as a second reconstruction body in the queue.

Each complete stream position applies atomically. If a later position fails,
the earlier committed prefix stays visible and notifies readers. The writer
reports the failure with its last applied position before sending that cue;
a failed attempt with no committed progress produces no commit notification.
The capture loop returns queued writer errors on continuing passes even when
there are no new inputs to parse, while keeping the client's outstanding unacknowledged spool entries pending.

A capture turn pins the existing read scope (the token naming which confirmed state a turn may use) before using cached confirmed state and checks that same token before reporting completion. A reset during a turn, or between turns, discards reconstructible input progress, parser and discovery caches; watched identities and entries still awaiting acknowledgment stay owned until their actual acknowledgments settle. The next turn reloads confirmed facts under a fresh scope and repeats discovery. The outer loop retries a scope interruption as a failed turn; queued writer and reset failures remain errors.

A failed capture turn cannot mark earlier unacknowledged spool entries finished. The loop
keeps that continuation through a read or planning error until a completed
pass observes its actual state; batch mode cannot report idle from that error.
Affected paths remain scheduled through a failed turn and while confirmation
is pending, including inputs with no confirmed file facts yet. A missing provider
still consults its existing unacknowledged spool entries and scoped history. A gone observation
waits for the exact confirmed input and no pending predecessor, uses the ordinary
metadata acquisition route, and changes local file facts only after confirmation.

Declarations validate domain fields and original-source identity and name the
fields used by privacy hints and time rules. Native readers join those declared
fields directly. Sensitive domain values retain guarded field classes.

Each distinct typed link statement keeps its own source contribution. A later
statement without a branch point therefore cannot replace an earlier statement
that supplies it; the index merges them into one logical relation. Confirmed
export events retain the identities assigned before publication, even when
JSON object keys return in a different order. The original callback body and
its seal remain unchanged.

A reconnect supplies the actual native completion receipt saved atomically with
the applied position. Sinex reoffers that marker, then later data; it does not
redeliver the original history. The separate staged cursor permits broker ACKs
for durable partial pages and never substitutes for completed local progress.
A replayed old barrier cannot move the completed receipt beyond unapplied data.

The stream follower alone reads and acknowledges confirmed pages. It sends each
page to the writer for durable staging. If a complete position retracts local
source contributions, the follower resolves those ordinary batches outside the
serial writer, allowing queued workers to commit and release their leases. The
queued application rechecks the native scope, staged facts, completion evidence
and exact affected chains before committing lifecycle metadata, local effects
and completion together. Staging alone advances no visible completion. The
follower acknowledges only after this application succeeds. It drains
remaining stream or snapshot pages without an idle pause, checking stop and
synchronization requests between bounded pages. An empty or nonadvancing page
waits before polling again; failures retain their retry backoff. Event payload
references use at most sixteen concurrent grant-checked fetches, with results
applied in delivery order. A refused or erased event keeps its own position;
a lost connection fails the page before its deliveries can be acknowledged.
Acknowledging a page removes the producer spool operations that it confirmed. An
operation whose events Sinex admitted is removed only when every one of those
events has a fact in `index.db`; the applied position alone is not enough,
because the confirmed stream can lose its newest messages in a host crash while
later positions still arrive. Such an operation stays in the spool, with its
bytes, until the events reach `index.db`. Operations, acquisitions and acknowledgments
are prepared in dot-named directories and published by rename; opening the
spool removes any such directory a crash left behind. Preparing a
write does not start another stream read on the writer thread. Explicit
synchronization asks the follower for a page, drains ready captures on the writer,
then asks the follower again so newly prepared capture products can be applied.

A durable capture can remain pending while its confirmed input reaches the local
index. The capture loop reports this as pending progress and resumes the original
acquisition; it is complete only after confirmed application. Before planning
another acquisition, the sampler adopts the confirmed input and reading cursor.
One transition adopts logical progress for watched inputs and exact database
input IDs. It keeps parser continuation only at the same input, origin, chain
and reading cursor, invalidates cached tail bytes when their stored range changes,
and removes the unacknowledged spool entry only when the matching observation is covered.
A pending capture's raw byte cursor cannot advance that logical source state.
Confirmed application can remove the producer body before the capture loop
receives its result. The loop then adopts the exact confirmed input, file facts
and covered chain through that acquisition's actual range. A partial confirmed
prefix leaves the remaining bytes and reading pending; the next normal plan
continues from that prefix. Fully confirmed parsing needs no extra pass.
When a pass waits for a tool database, its later turns revisit that pass's
provider paths. Database completion cannot clear still-pending file capture.
Before assigning source revisions, the writer compares the capture with the
completed confirmed-stream history. If an unreadable record might describe an
input, new source revisions remain pending. Already readable facts remain available.
Identity claims sampled before provider I/O are checked against their original
scoped history cut. Later confirmations do not rebind that basis or turn missing
evidence into an alias claim. Parsed facts use the acquired body's original byte
range for their material anchors, even when those bytes add no new logical
extent. Facts outside that range need an existing confirmed source anchor.
Parsing an already confirmed chain uses the exact input, origin, chain, byte
range and material anchors. It does not sample a new provider head; several
database inputs can share one provider path. Inline attachments decoded by that
parser retain their own blob preparation. The sender publishes declared native
domain facts and waits for their authoritative admission before resolving the
ordinary local rows. One local transaction writes those rows, the actual
publication receipt and the parser's processed source position. Broker delivery,
native admission and local interpretation progress remain distinct.

Native source kinds contain their complete domain values and original-source
identity. Messages retain typed blocks, usage, production, provider attributes,
source order and sparse evidence; sessions retain title provenance, aliases and
environment facts. Documents, text fragments, events, explicit relationships,
copied prefixes, tool outcomes and repository facts retain their own typed
fields. The confirmed consumer records only the source identity, value hash,
range, lifecycle and referenced blobs needed to select a contribution. It never
turns the native body into local parser output. Native parser diagnostics retain
counts, status and stopped-program evidence, rather than local checkpoints or
row-comparison state. Authored and query records still use their exact durable
operation decoders.

Compact keyed-context changes belong to the local checkpoint transaction.
Private preparation retains them until the ordinary rows commit. If native
lifecycle hides a response needed by that optional cache, the transaction
discards the cache and opaque continuation while retaining the actual processed
byte position; a later parse obtains its context from the original prefix.
Missing historical context stays unrecorded. A confirmed event whose kind
definition is not the one Polylogue declares is refused.

Tool-to-terminal or tool-to-file associations require captured exact native
references, such as a `work.act` record naming both events. Sharing a session,
message, command, path or time does not establish that association. Ordinary
terminal and file watchers do not infer which AI tool action caused a change.
A batch waits for active local database work. Deferred parsing is local work,
not a confirmation wait: successful parsing can start a database read in the
same batch, while a failed parse stops that attempt. Errors in that attempt do
not clear another input's confirmation wait. Once only confirmed application
remains, the batch returns pending progress so its caller can wait for stream events.
Later batches consult the original unacknowledged spool entries until it is removed; an
unchanged database does not need another physical read to report that wait.

Confirmed payloads are checked against their current immutable kind declaration
and payload digest. The saved native context carries that digest and the
owner-provided source and producer origin.

Message and tool-call declarations request no Sinex text index. Their field
hints describe defaults for operator-configured output rules; they impose no
kind-wide admission class floor. Event frame names include declaration hashes.
A producer replay selects the exact declarations and parser/build identities
for its complete material sequence, and waits for the replacement acknowledgment
before updating its local index. Snapshot and live reads retain verified native
identity, material, record hash and lifecycle even for an older domain kind
whose body this build no longer reads. A parser upgrade replaces those output
kinds by name across their versions, using the owner's exact material sequence,
digests and live-set hash. It reconstructs values from original bytes.

A complete replacement output set need not mean a complete original input.
An incomplete appended tail can report waiting while replacing the outputs of
the complete prefix. Every previous domain contribution must fit inside that
prefix. An older opaque fact uses its whole original material extent as a
conservative bound; its parser diagnostic claims no additional domain values.
A refused or partial parse before that bound leaves the previous native output
set live. The bound and live-set hash are checked together before publication,
and Sinex revalidates that live set at admission. Archived unreadable facts
remain historical diagnostics rather than current refusal counts.

## Local state

What a host keeps locally in primary mode:
- `meta.json`, the archive's durable storage identity, containing its mode;
- `index.db`, primary cache format2, holding native identity, local interpretation receipts and its stream position;
- the embeddings store, local and backed up, since recomputing it costs money;
- the producer client's spool, where bytes and prepared writes wait until Sinex acknowledges them; it sits under Sinex's store root on the same host. Material bytes are acknowledged before the events that cite them.

NATS connections to non-loopback endpoints require native TLS before authentication, protecting original-byte uploads and downloads as well as producer credentials. Plain loopback endpoints support local forwarding; they ignore broker-discovered endpoints so that exception cannot redirect traffic to another host.

Material reads use the producer client's byte API: `fetch_material_bytes` requests a material ID, decoded-byte offset, optional exclusive end and positive byte budget. Each page returns metadata, bytes and its continuation offset without exposing Sinex's segment locations. The same-host sender hardlinks frozen spool inodes into the frame directory and submits one logical frame. With `upload_root` configured for a remote Sinex, it transfers encoded chunks from that local scratch before submitting the logical frame. A cross-filesystem or incompatible-group handoff copies the existing encoding; a pinned file remains readable if its spool name retires during handoff.

The producer spool keeps immutable encoded frame and body files. One SQLite queue owns operation order, acknowledged or refused outcomes, names, dependencies, group membership and readiness. A preparation syncs its files before one queue transaction publishes them; an acquisition publishes all members together. Acknowledgments are validated once and store their admitted event identities alongside the finished phase. Confirmed removal selects acknowledged candidates by their applied position, checks the recorded event identities against the local index, and removes each complete group in one transaction. Refused and conflicting operations remain as evidence, stop their own dependents, and leave source-head and authored stale-base queries.

A sender runs independently of confirmed-stream reads. It keeps several frames in flight, subscribes to each acknowledgment before transmission, and interleaves new independent captures with the oldest ready backlog. Durable queue changes wake the sender. Transient network or storage failures retry; a deterministic refusal remains diagnosed while unrelated operations continue. Shutdown cancels waits and preserves each unfinished operation's original identity and bytes.

There is no local ledger of inputs, no local segment log and no send cursor. The Sinex adapter is the Cargo feature `sinex`, off in a plain `cargo build` and included in the Nix package.

Every archive records its mode in `meta.json` before creating its stores. The
identity is independent of deployment configuration and survives rebuilding
the derived index. Opening compares the requested mode with this identity
before creating any file; a mismatch names both modes and refuses the open.
An archive's mode is never inferred from its database files. Configuration
selects the intended mode; it cannot relabel an existing archive.


### Online backup

Polylogue participates in Sinex's producer backup cut. Before Sinex fixes its
core position, the producer freezes transmission through durable acknowledgment
and freezes confirmed-stream spool confirmed removal. Capture continues preparing new
operations. After the core cut, Polylogue pins the accepted spool operations
under the queue mutex, including acquisition groups, original encoded frame and
body bytes, immutable identities and acknowledgments, and fixes a queue database read snapshot at the same cut. It copies that database snapshot and hashes the pinned files outside the mutex into an immutable producer artifact. The coordinator
acquires those bytes before releasing the freeze through Sinex's participant
protocol. A cancelled request stops pinning and copying and discards its partial
artifact. Unpublished temporary operations are excluded.

The producer artifact restores under the same producer identity alongside its
paired core cut. Receipts in that artifact precede the core cut; operations
without acknowledgments retry their original identities. A later live spool or a
producer label alone cannot replace this artifact. A missing participant leaves
the unacknowledged spool entries incomplete.

**Reads see what Sinex committed.** A write becomes visible to queries once its record arrives on the confirmed stream and is applied. The stale-base check of an authored write is the one exception: it runs against the applied position plus this host's prepared writes not yet confirmed, so two quick revisions on one host (an MCP memory `create` followed by `str_replace`) see each other ([assertions.md#currency](assertions.md#currency)).

Literal deliveries and advice already printed by a hook freeze their body and
receipt directly from the admitted request. They do not copy authored history or
select targets again. Other authored writes still prepare against the confirmed
state plus this host's durable pending effects. Both paths retain the original
request outcome on retry and apply the same validated frozen effects. Recording
these deliveries in standalone mode also leaves variant projection work to its
own mutations and derivation. Restored delivery snapshots keep later events from
other requests; the original request still owns its exact body and own receipt.

## Write identity

- A prepared write has a durable identity and immutable contents; a retry reuses both.
- A later write gets a new identity even when its content equals an earlier one.
- The reading order for a stable key is assigned or checked by Sinex, never taken from a host clock.
- Material digests keep deduplicating bytes.

## Degraded primary

When Sinex is unreachable or refuses this producer:
- reads answer from `index.db`, marked with the stream position they reflect and its lag;
- capture keeps collecting into the producer client's spool, and the sender retries with the same write identities;
- an authored write is prepared in the spool and answered with its identity and the state `pending` (exit 5, degraded); it becomes visible when Sinex commits it;
- the verdict is `degraded` with the reason `sinex blocked` and the error in plain words, and the shipped `Sinex blocked` rule fires after 10 minutes;
- Polylogue never falls back to standalone, whatever the cause, a broken configuration file included.

**Status in primary** adds a section: the mode, the Sinex endpoint and this host's producer identity; reachability and the time of the last acknowledgement (recorded in the producer spool, so a restart during an outage still says how long Sinex has been silent); prepared materials and writes waiting (count and bytes); the stream position applied and its lag; chains parsed but not yet confirmed.
Publication and read failures remain distinct observations in that status. A successful confirmed read does not clear an outstanding publication failure; only a successful publication retry clears it. Both reasons remain visible when both operations fail.

## Rebuild and parser upgrades

**Rebuild.** A new local `index.db` (one from another schema set aside at start, [operations.md](operations.md#replacing-an-index-from-another-schema), or one moved aside by hand) resets its subscription position and reads Sinex-confirmed records, with progress in `status`. The exact public primary Index1 schema is set aside for the original-material cache format2. Its older native contexts lack the material location and owner record hash needed by current reconstruction. Opening never obtains those fields by decoding the retired parsed body. The producer spool, including pending authored intent, stays under the separate Sinex store root; paid vectors remain in `embeddings.db`. Both retain their original identities and bytes.

The old cache remains until the native head has been applied and every stored chain has admitted local progress at its current source length. A known incomplete suffix can settle as waiting; a refused or failed attempt needs its actual program identity. Missing progress and unreadable native facts keep the cache. Retirement runs at ordinary local flush or open, rather than scanning the source catalog on each confirmed position.

A stream with origin zero starts with its ordinary feed. When Sinex requires a current projection, Polylogue clears its reconstructible cache and stages bounded snapshot pages on disk. The original producer queue, source bytes and paid vector stores retain their own custody.

Each snapshot page is fetched under the owner's pinned cut, scope, catalog and stream incarnation. Every fetched event must match its reference's identity, admission position, kind, definition, projection and source authority. Archived readings retain their original admission and lifecycle order. A page and its next cursor commit together; interruption resumes that cursor. Invalidated owner context discards the partial snapshot, reports the owner's reason as a failed turn, and starts again. Unavailable owner evidence leaves the staged pages and cursor intact for retry. Partial pages are invisible and advance no applied position.

The last page carries the owner's certificate for the complete indexed projection at position B, at or after the stream origin. The writer makes the entire staged snapshot visible and saves that certificate with applied position B in one transaction. Every later consumer, status and resume request supplies it. Before opening the SQL cut B, Sinex captures a native prefix N and binds its exact message identity in that certificate. The snapshot covers every admitted fact published through N, so the first subscription starts at N+1 even when B lies between completion markers. N is projection coverage, not a delivered completion. An empty prefix skips nothing; a missing or changed boundary invalidates the certificate for restart and requests a new projection. The origin itself never stands in for an applied client prefix or same-incarnation delivery evidence. A changed scope or stream incarnation replaces the older cache, including facts absent from the new snapshot. The snapshot supplies source metadata, native interpretation eligibility and authored effects. The ordinary parser reads confirmed materials to reconstruct the specialist model.
Its due-chain query supplies the selected format and recorded path even when
the source arrived after startup or belongs to another host. That path selects
the parser and identifies the input; parsing reads material bytes, never the
recorded path on the local filesystem. Rebuilding the local index does not republish already confirmed interpretations or revive covered absent readings.

**Parser upgrades.** A chain whose reading version changed is re-parsed from its materials, read through the producer client, and its facts are sent again under the same stable keys as a producer replay of those materials: the new readings become live and the previous ones are archived with their derived chains, which never include authored records about the session.

## Several machines

- Each host is its own producer identity (credential and frame namespace) under one Polylogue kind family.
- A chain is re-parsed by the host that captured it, or by another host reading its materials from Sinex under a declared ownership rule that names it.
- Convergence through stable keys holds whichever host sends.
- The currency of authored records follows the revision graph identically on every host; no host sees another's unconfirmed writes, so concurrent edits on two hosts become two visible heads, resolved by an ordinary edit that revises both ([assertions.md#currency](assertions.md#currency)).

## Moving user state

User state (assertions, their revisions and links, variants, pages, deliveries) moves between archives without Sinex as a bundle: `polylogue QUERY read --to DIR` writes the matched sessions with their assertions and variants, and a query over the user-data units writes user state alone ([reading.md#writing-out](reading.md#writing-out)). `polylogue import DIR` replays the records through the one write path ([assertions.md#write-path](assertions.md#write-path)), keeping each record's stored author with `via: import`; it is idempotent by request ID, and conflicting revisions become visible heads.

The predecessor's authored records are imported as a declared older format ([assertions.md](assertions.md#write-path)).

## Cutover

Moving an archive to primary mode is an ordinary technical act, run by whoever builds it once the comparison passes. (ruling 2026-10-27, cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:f1f39c29-7c7d-42e0-99ec-0c7a1b10b751).

`polylogue sinex cutover`:
1. checks that the primary daemon has collected every provider location and is idle;
2. sends every standalone input as materials, including inputs whose provider files are gone (imports, browser captures, older spools), and replays all authored state with its stored authors (`via: import`);
3. compares the two archives by natural keys (sessions, messages, attachments, authored records and their current revisions); any difference stops the act and is reported with the keys;
4. sets the standalone archive aside, moved to a recorded location and never deleted;
5. switches Polylogue's configured mode to `primary` and checks that `status` shows `primary` and `healthy`.

`--dry-run` performs steps 1 and 3 against what Sinex already holds and changes nothing.

Where a deployment manages Polylogue's configuration (a Nix module option, for example), the mode switch of step 5 is that deployment's change: whoever runs the cutover makes it through the deployment and applies it the deployment's way. The binary never edits a deployment's configuration.

## Cross-system queries

The Polylogue daemon evaluates queries that mix Polylogue and Sinex data, because it holds the AI text, which Sinex does not index; Sinex callers reach AI text through this route.

- Names are bound from the parsed query with the `pl.` and `sx.` prefixes, never from strings; namespace binding is defined by the shared language in common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md).
- Sinex parts are compiled for and run by Sinex through its API; results join on references and time.
- Every answer reports the positions it read in each product and their lag, so a mixed answer is reproducible and its staleness visible.
- `sx:` references are a form whose grammar Sinex owns ([references.md#forms](references.md#forms)).
