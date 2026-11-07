# Primary mode

Polylogue on Sinex: what changes and what does not, what Polylogue sends, how it applies Sinex's confirmed stream, local state, write identity, several machines, Sinex's lifecycle applied to Polylogue's materials, moving user state, the cutover, and cross-system queries from Polylogue's side.

The producer client, the NATS ingress, the confirmed stream, refusals, kind registration and the lifecycle on the wire are Sinex's contract, in Sinex's [producers](https://github.com/Sinity/sinex/blob/master/docs/producers.md) and [lifecycle](https://github.com/Sinity/sinex/blob/master/docs/lifecycle.md) pages; this page states only Polylogue's side. The two modes are introduced in [architecture.md#modes](architecture.md#modes); the `polylogue.*` kinds will be listed in the [planned Sinex kinds reference](reference/README.md#planned-pages).

## What changes

Capture is identical in both modes ([capture.md](capture.md)); parsing, the parsed model, queries and every surface behave the same. What changes is where bytes and records are stored and how the local index is written.

- **Bytes go to Sinex as materials** through Sinex's producer client: each captured input, each appended step of a growing file as a part, export ZIPs whole. No source byte is stored outside Sinex.
- **Polylogue parses once** and sends complete, versioned `polylogue.*` facts keyed by Polylogue's stable IDs, together with its authored records and deliveries.
- **Authored records originate from request materials.** Each authored write is a request material (the write as received, with its request ID and computed author) plus its event. Its targets (sessions, messages, blocks, assertions, pages) are references in the payload with their `seen_hash`, never Sinex parents. A parser-upgrade replay of a session therefore archives the session's readings and their derived chain but never a note, judgment, highlight or page slot about it; those references resolve to the new readings by natural key. Marker assertions travel with their derived IDs, so the same marker sent from two hosts is one record.
- **The confirmed-stream consumer is the only way into `index.db`.** One durable subscription applies committed `polylogue.*` records to `index.db` in commit order, saving its stream position in the same transaction. It covers this host's writes, other hosts' writes and a rebuild alike. Polylogue's consumer has its own grant under Sinex's output rules, so it receives its own payloads whole; payloads too large for a message travel as material references and are fetched through the same client.
- **Lifecycle changes arrive on the stream** and are applied by the same consumer in commit order: an archived reading's rows leave `index.db`, so the index shows what Sinex shows as live; a restored reading's rows return; a newer reading replaces the old one's rows; a tombstoned reading keeps its reference and shows "tombstoned". Session tables recompute for the sessions touched ([measures.md#rules](measures.md#rules)).
- **Sinex's lifecycle applies to Polylogue's materials** as to any source's, through Sinex's own surfaces. Polylogue itself offers no deletion in either mode ([architecture.md#deletion](architecture.md#deletion)).
- **Settings shared with Sinex.** The logical day, quiet hours and the interrupt budget are Sinex's settings; Polylogue has no value of its own for them, and its interrupting notices are shown by Sinex's deliverer ([operations.md#notices](operations.md#notices)).

## Producer connection and declarations

Before writing frame bytes, Polylogue checks Sinex's producer welcome with an
empty hello. This checks the protocol and sealing configuration before its
producer-owned kinds exist. It then registers its `polylogue.*` declarations
and requires the acknowledgment to name their exact definition hashes before
sending materials or events. Each event selects its declaration by name and
version. A refused hello or frame returns Sinex's reason. Retrying an operation
with a durable refusal returns the same reason from its verified receipt.

A complete producer acknowledgment can carry no admitted-event position: an
empty frame, a repeated event or kind registration may admit no new events;
a replay may archive readings without admitting replacements. Polylogue keeps
that absent position in the original receipt. New admissions and replacements
require their event position. A receipt alone never advances the local index.
Ungrouped metadata operations can settle on their durable complete receipt.
A group holding original source bytes also waits until the local index has
applied a complete prefix in the same scope through an owner head observed
after the acknowledgment. Sinex reads that head from its committed counter.
A head observed before publication, or an incomplete prefix, retains the group.

Each complete stream position applies atomically. If a later position fails,
the earlier committed prefix stays visible and notifies readers. The writer
reports the failure with its last applied position before sending that cue;
a failed attempt with no committed progress produces no commit notification.
The capture loop returns queued writer errors on continuing passes even when
there are no new inputs to parse, while keeping outstanding custody pending.

Declarations validate the complete record and describe the outer projected
fields used by privacy hints and time rules. The complete record remains guarded;
the outer fields let Sinex join the record to its other domains.

Each distinct typed link statement keeps its own source contribution. A later
statement without a branch point therefore cannot replace an earlier statement
that supplies it; the index merges them into one logical relation. Confirmed
export events retain the identities assigned before publication, even when
JSON object keys return in a different order. The original callback body and
its seal remain unchanged.

The stream follower alone reads and acknowledges confirmed pages. It sends each
page to the writer and acknowledges it after durable application. Preparing a
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
and retires working custody only when the matching observation is covered.
A pending capture's raw byte cursor cannot advance that logical source state.
Confirmed application can retire the producer body before the capture loop
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
Parsing an already confirmed chain prepares reading products directly, using the
exact input, origin, chain, byte range and material anchors. It does not sample a
new provider head; several database inputs can share one provider path. Inline
attachments decoded by that parser retain their own blob preparation. Durable
product preparation advances the sampler's cached parser state only after the
matching reading version and cursor reach the confirmed index.
A batch waits for active local database work. Deferred parsing is local work,
not a confirmation wait: successful parsing can start a database read in the
same batch, while a failed parse stops that attempt. Errors in that attempt do
not clear another input's confirmation wait. Once only confirmed application
remains, the batch returns pending progress so its caller can wait for stream events.
Later batches consult the original producer custody until it is retired; an
unchanged database does not need another physical read to report that wait.

Confirmed payloads are checked against their current immutable kind declaration
and payload digest. The saved native context carries that digest and the
owner-provided source and producer origin.

Message and tool-call declarations request no Sinex text index. Their field
hints describe defaults for operator-configured output rules; they impose no
kind-wide admission class floor. Event frame names include declaration hashes.
A producer replay selects the exact declarations and parser/build identities
for its complete material sequence, and waits for the replacement acknowledgment
before updating its local index.

## Local state

What a host keeps locally in primary mode:
- `index.db`, a cache holding its stream position;
- the embeddings store, local and backed up, since recomputing it costs money;
- the producer client's spool, where bytes and prepared writes wait until Sinex acknowledges them; it sits under Sinex's store root on the same host. Material bytes are acknowledged before the events that cite them.

There is no local ledger of inputs, no local segment log and no send cursor. The Sinex adapter is a Cargo feature outside the default build.

### Online backup

Polylogue participates in Sinex's producer backup cut. Before Sinex fixes its
core position, the producer freezes transmission through durable acknowledgment
and freezes confirmed-stream spool retirement. Capture continues preparing new
operations. After the core cut, Polylogue pins the accepted spool operations
under the queue mutex, including acquisition groups, original encoded frame and
body bytes, immutable identities and receipts. It copies and hashes those pinned
files outside the mutex into an immutable producer artifact. The coordinator
acquires those bytes before releasing the freeze through Sinex's participant
protocol. A cancelled request stops pinning and copying and discards its partial
artifact. Rebuildable indexes and unpublished temporary operations are excluded.

The producer artifact restores under the same producer identity alongside its
paired core cut. Receipts in that artifact precede the core cut; operations
without receipts retry their original identities. A later live spool or a
producer label alone cannot replace this artifact. A missing participant leaves
producer custody incomplete.

**Reads see what Sinex committed.** A write becomes visible to queries once its record arrives on the confirmed stream and is applied. The stale-base check of an authored write is the one exception: it runs against the applied position plus this host's prepared writes not yet confirmed, so two quick revisions on one host (an MCP memory `create` followed by `str_replace`) see each other ([assertions.md#currency](assertions.md#currency)).

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

**Status in primary** adds a section: the mode, the Sinex endpoint and this host's producer identity; reachability and the time of the last acknowledgement; prepared materials and writes waiting (count and bytes); the stream position applied and its lag; chains parsed but not yet confirmed.
Publication and read failures remain distinct observations in that status. A successful confirmed read does not clear an outstanding publication failure; only a successful publication retry clears it. Both reasons remain visible when both operations fail.

## Rebuild and parser upgrades

**Rebuild.** Moving the local `index.db` aside and starting the daemon resets its subscription position and applies the confirmed stream from its start, with progress in `status`. A rebuild reads only the stream; it does not read materials back to re-parse them.

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

Moving an archive to primary mode is an ordinary technical act, run by whoever builds it once the comparison passes. Ruling, 2026-10-27: "you _did_ create current storage in the first place while developing. why would you now require my approval?" (cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:f1f39c29-7c7d-42e0-99ec-0c7a1b10b751).

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

- Names are bound from the parsed query with the `pl.` and `sx.` prefixes, never from strings; namespace binding is a law of the shared language in common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/query-language.md).
- Sinex parts are compiled for and run by Sinex through its API; results join on references and time.
- Every answer reports the positions it read in each product and their lag, so a mixed answer is reproducible and its staleness visible.
- `sx:` references are a form whose grammar Sinex owns ([references.md#forms](references.md#forms)).

## Laws

- **rebuild-from-stream**: Morrow in primary mode answers as standalone does, and again after a rebuild from the confirmed stream with no provider file present (CI, against a test Sinex stack from devkit).
- **user-state-rebuilds-from-stream**: every kind of authored record, rebuilt from the confirmed stream alone, reads identically.
- **outage-degrades-and-recovers**: during a Sinex outage reads answer with their position and lag, capture continues into the spool, an authored write returns `pending`, the verdict is `degraded: sinex blocked` and the notice fires once; on recovery everything is sent with its original write identities and the verdict returns to `healthy`.
- **write-identity**: a retry reuses identity and contents; a value changing A → B → A sends three writes and leaves A live; reading order never comes from the host clock.
- **hosts-converge**: two hosts sending the same session converge to one set of facts by stable keys; concurrent authored edits on two hosts appear as two heads.
- **cutover-compares-and-sets-aside**: `sinex cutover --dry-run` on Morrow (standalone, with an import whose file is gone and authored notes) reports equality by natural keys; the full act yields a primary archive with the same answers and the standalone archive set aside, not deleted.
- **no-bytes-outside-sinex**: a primary-mode host stores no source byte outside Sinex's store root (filesystem check after converging).
- **lifecycle-reaches-index**: on a test Sinex stack, archiving a session's readings removes the session from answers, restoring brings it back, and tombstoning leaves its reference marked tombstoned; a note, judgment and page slot on the session survive a parser-upgrade replay of it, still resolving to the new readings.
- **user-state-moves**: an archive with assertions, revisions, a withdrawal, a pass, a variant and a page, written out with `read --to DIR` and imported into a fresh archive with the same inputs, gives identical reads; importing twice changes nothing.
- **mixed-answer-reports-positions**: a mixed query's answer names the position it read in each product and their lag.
