# Primary mode

Polylogue on Sinex: what changes and what does not, what Polylogue sends, how it applies Sinex's confirmed stream, local state, write identity, several machines, Sinex's lifecycle applied to Polylogue's materials, moving user state, moving to Sinex, and cross-system queries from Polylogue's side.

The producer client, the NATS ingress, the confirmed stream, refusals, kind registration and the lifecycle on the wire are Sinex's contract, in Sinex's [producers](https://github.com/Sinity/sinex/blob/master/docs/producers.md) and [lifecycle](https://github.com/Sinity/sinex/blob/master/docs/lifecycle.md) pages; this page states only Polylogue's side. The two modes are introduced in [architecture.md#modes](architecture.md#modes).

## What changes

Capture is identical in both modes ([capture.md](capture.md)); parsing, the parsed model, queries and every surface behave the same. What changes is where bytes and records are stored and how the local index is written.

- **Bytes go to Sinex as materials** through Sinex's producer client: each captured input, each appended step of a growing file as a part, export ZIPs whole. A document its source saves whole is a material sent as its content-defined chunks ([archive.md](archive.md)): the frame names the chunks in order and carries every chunk's encoding, and Sinex stores only the chunks it does not hold and records the material once the chunks read back as its bytes. The input's extents then name the chunks, and `index.db` catalogs each chunk's blob. No source byte is stored outside Sinex.
- **Confirmed originals drive local parsing.** Polylogue's local rows come from parsing source materials Sinex has confirmed. The events of each parse (`polylogue.*` events: messages, tool calls, sessions, documents, links and the rest) are prepared in the producer spool first; then the rows and the parser checkpoint commit in one index transaction, without waiting for Sinex to acknowledge those events or for them to come back on the stream. The spool sends them like any producer's frames, and a resend after a crash between the spool and the commit is acknowledged as a repeat. Sinex keeps these events with their source and time quality, usable independently of Polylogue; the local index never reads them back as rows and stays reconstructible from the originals.
- **Authored records originate from request materials.** Each authored write is a request material (the write as received, with its request ID and computed author) plus its event. Its targets (sessions, messages, blocks, annotations) are references in the payload with their `seen_hash`, never Sinex parents. A parser-upgrade replay of a session therefore archives the session's readings and their derived chain but never a note, tag, judgment or highlight about it; those references resolve to the new readings by natural key. Annotations from markup ([markup.md](markup.md)) travel with their derived IDs, so the same markup sent from two hosts is one record. Pages, their slots and members are annotations and travel as authored records like every other ([pages.md](pages.md#pages-elsewhere)).
- **Sinex-confirmed state alone enters `index.db`.** One subscription supplies confirmed originals, Polylogue's own events and lifecycle changes in commit order. Local parsing and application save the processed position in the same transaction as their complete result. A new stream that starts at a current-state origin first supplies a snapshot of Sinex's current state; the same writer then follows its deltas. This covers this host's writes, other hosts' writes and a rebuild alike. Open-class events arrive whole on the stream. An event of another class, or one too large for a stream message, arrives by ID and is fetched under Polylogue's own grant, so Polylogue receives its own payloads whole.
- **Lifecycle changes arrive on the stream** and are applied by the same consumer in commit order. Archiving an original material retracts the chains parsed from it through the writer, which keeps what other sources still supply; restoring it makes those chains contribute again when they are parsed. Archiving, restoring or tombstoning one of Polylogue's own `polylogue.*` events changes only Sinex's copy and the follower's record of that event: the local rows come from the original bytes, so they stay. Session tables recompute for the sessions touched ([measures.md#rules](measures.md#rules)).
- **Sinex's lifecycle applies to Polylogue's materials** as to any source's, through Sinex's own surfaces. Polylogue itself offers no deletion in either mode ([architecture.md#deletion](architecture.md#deletion)).
- **Settings shared with Sinex.** The logical day, quiet hours and the interrupt budget are Sinex's settings; Polylogue has no value of its own for them, and its interrupting notices are shown by Sinex's deliverer ([operations.md#notices](operations.md#notices)).

Admission acknowledgments, publisher positions and locally processed positions
remain distinct. An acknowledgment shows Sinex admitted the write; Sinex's
publisher makes that committed state available to consumers; local progress
shows the corresponding parsing and application completed. An authored write
saved in the spool stays available offline and retries with its original
identity, without reporting itself confirmed or advancing processed progress
before that work finishes.

## Producer connection and declarations

Before writing frame bytes, Polylogue checks Sinex's producer welcome with an
empty hello. This checks the protocol and sealing configuration before its
producer's own kinds exist. It then registers its `polylogue.*` declarations
and requires the acknowledgment to name their definition hashes before
sending materials or events. Each event selects its declaration by name and
version. A refused hello or frame returns Sinex's reason. Retrying an operation
with a stored refusal returns the same reason from its verified acknowledgment.

Each acknowledgment is one typed result (admitted, kinds registered, replayed
or refused), and Polylogue checks that the result fits the operation it
acknowledges: a kind registration is acknowledged by its registered kinds (or, when it
declares nothing, by an empty admission), a replay by a replay, and every
other frame by an admission whose material is the prepared one. A complete
producer acknowledgment can carry no admitted-event position: an
empty frame, a repeated event or kind registration may admit no new events;
a replay may archive readings without admitting replacements. Polylogue keeps
that absent position in the original acknowledgment. New admissions and replacements
require their event position. An acknowledgment alone never advances the local index.
Ungrouped metadata operations can finish with their stored complete acknowledgment.
A group holding original source bytes also waits until the local index has
applied a complete prefix in the same scope through a latest Sinex position observed
after the acknowledgment. Sinex reads that position from its committed counter.
A position observed before sending, or an incomplete prefix, keeps the group in the spool.

The producer queue is a SQLite database of format 2. Lookups of a source's latest queued capture
depend only on physical source identity, and parser output is not stored as a
second reconstruction body in the queue. A queue of another format is refused
when it opens: a format-1 queue is emptied by the build that wrote it (every
operation sent and confirmed) and then set aside.

Sinex numbers the readings of a key in the order it admits them, so the
frames of parsed events for one original chain are admitted in the order they
were prepared. Their identities begin with a prefix naming the chain, and a
frame is ready to send only when no earlier frame of its chain is still
pending in the queue. The sender therefore never has two frames of one chain
in flight, a retried frame cannot arrive after a later one, and frames of
different chains proceed concurrently. A refused frame was not admitted and
does not hold back the next one.

Each complete stream position applies atomically. If a later position fails,
the earlier committed prefix stays visible and notifies readers. The writer
reports the failure with its last applied position before sending that cue;
a failed attempt with no committed progress produces no commit notification.
The capture loop returns queued writer errors on continuing passes even when
there are no new inputs to parse, while keeping the client's outstanding unacknowledged spool entries pending.

A capture turn pins the existing read scope (the token naming which confirmed state a turn may use) before using cached confirmed state and checks that same token before reporting completion. A reset during a turn, or between turns, discards reconstructible input progress, parser and discovery caches; watched identities and entries still awaiting acknowledgment stay in the spool until their acknowledgments arrive. The next turn reloads confirmed events under a fresh scope and repeats discovery. The outer loop retries a scope interruption as a failed turn; queued writer and reset failures remain errors.

A failed capture turn cannot mark earlier unacknowledged spool entries finished. The loop
keeps that continuation through a read or planning error until a completed
pass observes its state; batch mode cannot report idle from that error.
Affected paths remain scheduled through a failed turn and while confirmation
is pending, including inputs with no confirmed file facts yet. A missing provider
still consults its existing unacknowledged spool entries and scoped history. A gone observation
waits for the confirmed input and no pending previous revision, uses the
metadata capture-read route, and changes local file facts only after confirmation.

Declarations validate domain fields and original-source identity and name the
fields used by privacy hints and time rules. Sinex's readers join those declared
fields directly. Sensitive domain values keep guarded field classes.

Each distinct typed link statement keeps its own source contribution. A later
statement without a branch point therefore cannot replace an earlier statement
that supplies it; the index merges them into one logical relation. Confirmed
export events keep the identities assigned before sending, even when
JSON object keys return in a different order. The original callback body and
its seal remain unchanged.

A reconnect names the completion marker saved atomically with the applied
position. Sinex reads that message back, reoffers it, then later data; it does
not redeliver the original history. Until the first marker after a snapshot,
the reconnect names the snapshot's start instead. The separate staged cursor permits broker ACKs
for staged partial pages and never substitutes for completed local progress.
A replayed old marker cannot move the completed marker beyond unapplied data.

The stream follower alone reads and acknowledges confirmed pages. It sends each
page to the writer for staging. If a complete position retracts local
source contributions, the follower resolves those batches outside the
serial writer, allowing queued workers to commit and release their key locks. The
queued application rechecks the Sinex read scope, staged events, completion marker
and affected chains before committing lifecycle metadata, local effects
and completion together. Staging alone advances no visible completion. The
follower acknowledges only after this application succeeds. It drains
remaining stream or snapshot pages without an idle pause, checking stop and
synchronization requests between bounded pages. An empty or nonadvancing page
waits before polling again; failures keep their retry backoff. Withheld
events use at most sixteen concurrent grant-checked fetches, with results
applied in delivery order; whole events need no fetch. A refused or erased event keeps its own position;
a lost connection fails the page before its deliveries can be acknowledged.
Acknowledging a page removes the producer spool operations that it confirmed. An
operation whose events Sinex admitted is removed only when every one of those
events has a row in `index.db`; the applied position alone is not enough,
because the confirmed stream can lose its newest messages in a host crash while
later positions still arrive. Such an operation stays in the spool, with its
bytes, until the events reach `index.db`. Operations, capture reads and acknowledgments
are prepared in dot-named directories and published by rename; opening the
spool removes any such directory a crash left behind. Preparing a
write does not start another stream read on the writer thread. Explicit
synchronization asks the follower for a page, drains ready captures on the writer,
then asks the follower again so newly prepared capture products can be applied.

A capture saved in the spool can remain pending while its confirmed input reaches
the local index. The capture loop reports this as pending progress and resumes the
original capture read; it is complete only after confirmed application. Before
planning another capture read, the sampler adopts the confirmed input and reading cursor.
One transition adopts logical progress for watched inputs and database
input IDs. It keeps parser continuation only at the same input, origin, chain
and reading cursor, invalidates cached tail bytes when their stored range changes,
and removes the unacknowledged spool entry only when the matching observation is covered.
A pending capture's byte cursor cannot advance that logical source state.
Confirmed application can remove the producer body before the capture loop
receives its result. The loop then adopts the confirmed input, file facts
and covered chain through that capture read's range. A partial confirmed
prefix leaves the remaining bytes and reading pending; the next normal plan
continues from that prefix. Fully confirmed parsing needs no extra pass.
When a pass waits for a tool database, its later turns revisit that pass's
provider paths. Database completion cannot clear still-pending file capture.
Before assigning source revisions, the writer compares the capture with the
completed confirmed-stream history. If an unreadable record might describe an
input, new source revisions remain pending. Rows already readable remain available.
Stated identities sampled before provider I/O are checked against their original
scoped history position. Later confirmations do not rebind that base or turn a missing
value into a stated alias. Parsed events use the read body's original byte
range for their record offsets, even when those bytes add no new logical
extent. Events outside that range need an existing confirmed record offset.
Parsing an already confirmed chain uses the same input, origin, chain, byte
range and record offsets. It does not sample the provider's current size again; several
database inputs can share one provider path. Inline attachments decoded by that
parser are materials of their own: the rows that cite them are written once the
stream has delivered those materials. Events Sinex already holds for the same
reading (a confirmed event with the same identity and value, or a confirmed
parse-progress event covering the event's range) are not sent again. The local
rows carry the commit position of the original material's acknowledgment.

Polylogue's kinds contain their complete domain values and original-source
identity. Messages keep typed blocks, usage, production, provider attributes,
source order and sparse stated fields; sessions keep title source, aliases and
environment facts. Documents, text fragments, events, explicit relationships,
copied prefixes, tool outcomes and repository facts keep their own typed
fields. The stream follower records only each confirmed event's identity,
record hash, value hash, range, lifecycle and referenced blobs: enough to decide
what still needs sending and to check a parser upgrade against Sinex's earlier
events. It never turns an event body into local rows. Parser diagnostics keep
counts, status and stopped-program details, rather than local checkpoints or
row-comparison state. Authored and query records still use their operation
decoders.

Compact keyed-context changes belong to the local checkpoint transaction.
Private preparation keeps them until the rows commit. Missing historical
context stays unrecorded. A confirmed event whose kind
definition is not the one Polylogue declares is refused.

Tool-to-terminal or tool-to-file associations require captured references
to both events as their origins wrote them, such as a `work.act` record naming both events. Sharing a session,
message, command, path or time does not establish that association.
Terminal and file watchers do not infer which AI tool action caused a change.
A batch waits for active local database work. Deferred parsing is local work,
not a confirmation wait: successful parsing can start a database read in the
same batch, while a failed parse stops that attempt. Errors in that attempt do
not clear another input's confirmation wait. Once only confirmed application
remains, the batch returns pending progress so its caller can wait for stream events.
Later batches consult the original unacknowledged spool entries until it is removed; an
unchanged database does not need another read of the file to report that wait.

Confirmed payloads are checked against their current immutable kind declaration
and payload digest. The saved Sinex context carries that digest and the
Sinex-provided source and producer origin.

Message and tool-call declarations request no Sinex text index. Their field
hints describe defaults for operator-configured output rules; they impose no
kind-wide admission class floor. Event frame names include declaration hashes.
A producer replay selects the declarations and parser/build identities
for its complete material sequence. The local index replaces the chain's rows
without waiting for the replay's acknowledgment. Snapshot and live reads keep the verified Sinex
identity, material, record hash and lifecycle even for an older domain kind
whose body this build no longer reads. A parser upgrade replaces those output
kinds by name across their versions, using Sinex's material sequence,
digests and live-set hash. It reconstructs values from original bytes.

A complete replacement output set need not mean a complete original input.
An incomplete appended tail can report waiting while replacing the outputs of
the complete prefix. Every previous domain contribution must fit inside that
prefix. An older event this build cannot read uses its whole original material extent
as a conservative bound; its parser diagnostic states no additional domain values.
A refused or partial parse before that bound leaves the previous output set in
Sinex live. The bound is checked against the follower's record before the replay
is prepared. The replay names the live set Sinex reports for the chain when it
is prepared, so it is prepared only when no earlier frame of that chain is
unacknowledged; Sinex checks that live set again at admission. Archived unreadable events
remain historical diagnostics rather than current refusal counts.

## Local state

What a host keeps locally in primary mode:
- `meta.json`, the archive's storage identity, containing its mode;
- `index.db`, the primary cache, holding the rows parsed from confirmed originals, the follower's record of Polylogue's confirmed events and its stream position;
- the embeddings store, local and backed up, since recomputing it costs money;
- the producer client's spool, where bytes and prepared writes wait until Sinex acknowledges them; it sits under Sinex's store root on the same host. Material bytes are acknowledged before the events that cite them.

NATS connections to non-loopback endpoints require built-in TLS before authentication, protecting original-byte uploads and downloads as well as producer credentials. Plain loopback endpoints support local forwarding; they ignore broker-discovered endpoints so that exception cannot redirect traffic to another host.

Material reads use the producer client's byte API: `fetch_material_bytes` requests a material ID, decoded-byte offset, optional exclusive end and positive byte budget. Each page returns metadata, bytes and its continuation offset without exposing Sinex's segment locations. The same-host sender hardlinks immutable spool inodes into the frame directory and submits one logical frame. With `upload_root` configured for a remote Sinex, it transfers encoded chunks from that local scratch before submitting the logical frame. A cross-filesystem or incompatible-group handoff copies the existing encoding; a pinned file remains readable if its spool name retires during handoff.

The producer spool keeps immutable encoded frame and body files. One SQLite queue owns operation order, acknowledged or refused outcomes, names, dependencies, group membership and readiness. A preparation syncs its files before one queue transaction makes them visible; a capture read makes all its members visible together. Acknowledgments are validated once and store their admitted event identities alongside the finished phase. Confirmed removal selects acknowledged operations by their applied position, checks the recorded event identities against the local index, and removes each complete group in one transaction. Refused and conflicting operations remain as a record, stop their own dependents, and leave the latest-capture and authored stale-base queries.

A sender runs independently of confirmed-stream reads. It keeps several frames in flight (at most one per parsed chain), subscribes to each acknowledgment before transmission, and interleaves new independent captures with the oldest ready backlog. Committed queue changes wake the sender. Transient network or storage failures retry; a deterministic refusal remains diagnosed while unrelated operations continue. Shutdown cancels waits and preserves each unfinished operation's original identity and bytes.

There is no local table of inputs, no local segment log and no send cursor. The Sinex adapter is the Cargo feature `sinex`, off in a plain `cargo build` and included in the Nix package.

Every archive records its mode in `meta.json` before creating its stores. The
identity is independent of deployment configuration and survives rebuilding
the derived index. Opening compares the requested mode with this identity
before creating any file; a mismatch names both modes and refuses the open.
An archive's mode is never inferred from its database files. Configuration
selects the intended mode; it cannot relabel an existing archive.


### Online backup

Polylogue participates in Sinex's producer backup position. Before Sinex fixes its
core position, the producer pauses sending until every frame in flight is
acknowledged, and pauses the removal of confirmed spool entries. Capture continues preparing new
operations. After the core backup position, Polylogue pins the accepted spool operations
under the queue mutex, including capture-read groups, original encoded frame and
body bytes, immutable identities and acknowledgments, and fixes a queue database read snapshot at the same position. It copies that database snapshot and hashes the pinned files outside the mutex into an immutable producer artifact. The coordinator
copies those bytes before ending the pause through Sinex's participant
protocol. A cancelled request stops pinning and copying and discards its partial
artifact. Temporary operations not yet in the queue are excluded.

The producer artifact restores under the same producer identity alongside its
paired core backup position. Acknowledgments in that artifact precede the core backup position; operations
without acknowledgments retry their original identities. A later live spool or a
producer label alone cannot replace this artifact. A missing participant leaves
the unacknowledged spool entries incomplete.

**Reads see what Sinex committed.** A write becomes visible to queries once its record arrives on the confirmed stream and is applied. The stale-base check of an authored write is the one exception: it runs against the applied position plus this host's prepared writes not yet confirmed, so two quick revisions on one host (an MCP memory `create` followed by `str_replace`) see each other ([annotations.md#current-revisions](annotations.md#current-revisions)).

Literal deliveries and advice already printed by a hook fix their body and
write result directly from the admitted request. They do not copy authored history or
select targets again. Other authored writes still prepare against the confirmed
state plus this host's pending effects saved in the spool. Both paths keep the
original request outcome on retry and apply the same validated recorded effects. Recording
these deliveries in standalone mode also leaves alternate derivation work to its
own mutations and derivation. Restored delivery snapshots keep later events from
other requests; the original request still keeps its own body and write result.

## Write identity

- A prepared write has a saved identity and immutable contents; a retry reuses both.
- A later write gets a new identity even when its content equals an earlier one.
- The reading order for a stable key is assigned or checked by Sinex, never taken from a host clock.
- Material digests keep deduplicating bytes.

## Degraded primary

When Sinex is unreachable or cannot accept producer writes:
- reads are served from `index.db`, marked with the stream position they reflect and its lag;
- capture keeps collecting into the producer client's spool, and the sender retries with the same write identities;
- material bytes not yet acknowledged are not parsed; chains already confirmed are still parsed, their rows commit and their events wait in the spool;
- an authored write is prepared in the spool and replied to with its identity and the state `pending` (exit 5, degraded); it becomes visible when Sinex commits it; setting writes keep their write result, including `id`, `repeat` and `pending`;
- the verdict is `degraded` with the reason `sinex blocked` and the error in plain words, and the shipped `Sinex blocked` rule fires after 10 minutes;
- Polylogue never falls back to standalone, whatever the cause, a broken configuration file included.

A confirmed-stream read denial withdraws cached answers and interrupts active reads. The cache stays unreadable across a restart. When Sinex grants access again the follower starts at position zero under that grant. A broker or database outage leaves the previous answers available with their applied position.

**Status in primary** adds a section: the mode, the Sinex endpoint and this host's producer identity; reachability and the time of the last acknowledgment (recorded in the producer spool, so a restart during an outage still says how long Sinex has been silent); prepared materials and writes waiting (count and bytes); the stream position applied and its lag; chains parsed but not yet confirmed.
Send and read failures remain distinct observations in that status. A successful confirmed read does not clear an outstanding send failure; only a successful send retry clears it. Both reasons remain visible when both operations fail.

## Rebuild and parser upgrades

Browser writes distinguish preparation from enqueue. If attached bytes are saved but unconfirmed, the reply is degraded with their capture `identity`; no action has been enqueued. If the authored chat action or provider import job is saved but unconfirmed, the reply carries its write result with `pending: true`, without looking up or claiming a confirmed queued action. A confirmed write returns the current action or job as before. Retries keep the corresponding capture or authored identity.

**Rebuild.** A new local `index.db` (one from another schema set aside at start, [installing.md](installing.md#replacing-an-index-from-another-schema), or one moved aside by hand) resets its subscription position and reads Sinex-confirmed records, with progress in `status`. Reconstruction reads the stored originals and confirmed records; nothing reads the set-aside cache. The producer spool, including pending authored intent, stays under the separate Sinex store root; paid vectors remain in `embeddings.db`. Both keep their original identities and bytes.

The set-aside cache remains until the stream is applied through Sinex's latest position and every stored chain has parser progress at its current source length. A known incomplete suffix can end as waiting; a refused or failed attempt needs its program identity. Missing progress and unreadable Sinex events keep it. Its removal runs at local flush or open, rather than scanning the source catalog on each confirmed position.

A stream with origin zero starts with the live stream. When Sinex requires a snapshot, Polylogue clears its reconstructible cache and stages bounded snapshot pages on disk. The producer queue, source bytes and paid vector stores are kept as they are.

Snapshot pages are read at one position B, at or after the stream origin, for one scope and stream instance. Each page carries its events whole or withheld; a withheld one is fetched and must match its ID, position and payload hash. Archived readings keep their original admission and lifecycle order. A page and its next cursor commit together; interruption resumes that cursor. When Sinex replies that the snapshot must restart (changed scope, catalog or stream instance, or a reading tombstoned since B), Polylogue discards the partial snapshot, reports the reason as a failed turn, and starts again. When Sinex is unavailable, the staged pages and cursor stay for retry. Partial pages are invisible and advance no applied position.

The last page names B and the last stream message Sinex saw before reading it. The writer makes the entire staged snapshot visible and saves that record with applied position B in one transaction. The next consumer request names that message, and Sinex starts delivery after it; every event that message or an earlier one delivered is in the snapshot. After the first completion marker, resuming names that marker instead. A changed scope or stream instance replaces the older cache, including events absent from the new snapshot. The snapshot supplies source metadata, Polylogue's own confirmed events and authored effects. The parser reads confirmed materials to rebuild the rows.
Its due-chain query supplies the selected format and recorded path even when
the source arrived after startup or belongs to another host. That path selects
the parser and identifies the input; parsing reads material bytes, never the
recorded path on the local filesystem. Rebuilding the local index does not send already confirmed parsed events again or revive covered absent readings.

**Parser upgrades.** A chain whose reading version changed is re-parsed from its materials, read through the producer client, as in standalone mode, and its rows are replaced locally. Its events are sent again under the same stable keys as a producer replay of those materials: the new readings become live and the previous ones are archived with their derived chains, which never include authored records about the session.

## Several machines

- Each host is its own producer identity (credential and frame namespace) under one Polylogue kind family.
- A chain is re-parsed by the host that captured it, or by another host reading its materials from Sinex under a declared ownership rule that names it.
- Convergence through stable keys holds whichever host sends.
- Which revisions of an authored record are current follows the revision graph identically on every host; no host sees another's unconfirmed writes, so concurrent edits on two hosts become two visible heads, resolved by an edit that revises both ([annotations.md#current-revisions](annotations.md#current-revisions)).

## Moving user state

User state (annotations, their revisions and links, alternates, deliveries) moves between archives without Sinex as a bundle: `polylogue 'repo:sinex' read --to DIR` writes the matched sessions with their annotations and alternates, and a query over the annotation units writes user state alone (`polylogue 'annotations where author:operator' read --to DIR`; [reading.md#formats-and-destinations](reading.md#formats-and-destinations)). `polylogue import DIR` replays the records through the one write path ([annotations.md#the-write-path](annotations.md#the-write-path)), keeping each record's stored author with `via: import`; it is idempotent by request ID, and conflicting revisions become visible heads.

Python Polylogue's authored records are imported as a declared older format ([capture.md#import](capture.md#import)).

## Moving to Sinex

Moving an archive to primary mode is a technical act, run once by whoever builds it when the comparison passes. It sits under `dev` because it is a one-time move, not an everyday act ([interfaces.md#discipline](interfaces.md#discipline)).

`polylogue dev move-to-sinex`:
1. checks that the primary daemon has collected every provider location and is idle;
2. sends every standalone input as materials, including inputs whose provider files are gone (imports, browser captures, older spools), and replays all authored state with its stored authors (`via: import`);
3. compares the two archives by natural keys (sessions, messages, attachments, authored records and their current revisions); any difference stops the act and is reported with the keys;
4. sets the standalone archive aside, moved to a recorded location and never deleted;
5. switches Polylogue's configured mode to `primary` and checks that `status` shows `primary` and `healthy`.

`--dry-run` performs steps 1 and 3 against what Sinex already holds and changes nothing.

Where a deployment manages Polylogue's configuration (a Nix module option, for example), the mode switch of step 5 is that deployment's change: whoever runs the move makes it through the deployment and applies it the deployment's way. The binary never edits a deployment's configuration.

## Cross-system queries

The Polylogue daemon evaluates queries that mix Polylogue and Sinex data, because it holds the AI text, which Sinex does not index; Sinex callers reach AI text through this route.

- Names are bound from the parsed query with the `pl.` and `sx.` prefixes, never from strings; namespace binding is defined by the shared language in common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md).
- Sinex parts are compiled for and run by Sinex through its API; results join on references and time.
- Every reply reports the positions it read in each product and their lag, so a mixed reply is reproducible and its staleness visible.
- `sinex:<kind>/<id>` references are a form whose grammar Sinex owns ([references.md#forms](references.md#forms)).

Query objects, execution begins and execution completions keep their logical IDs when their recorded format changes. Their named producer frames include the current Polylogue source version, so a current begin or completion never reuses a predecessor's immutable frame bytes. The ordinary writer, pending-product reader and preserved-record reconstruction use this same identity rule. A completion's kind version changes with the binding hashes it carries. A preserved cache is strictly an extraction input: its converted query products are published and confirmed before the normal fresh cache reconstruction. Its old derived tables are never retagged as current. A standalone register can be converted to the current register format. Original register, spool and paid output bytes remain preserved.

Line resolution uses its own declared live and restored record kinds through the same authored request path. Existing authored kind definitions and prepared bodies remain unchanged. Confirmed authored records retain their original order and live or historical execution status; pending authored materials retain their original source, identity and bytes on retry. They require no conversion or republication.
