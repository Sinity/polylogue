# Primary mode

Polylogue on Sinex: the one rule, the three parts primary mode adds (bytes kept in Sinex's store, the record shipped to Sinex, a reader of Sinex's log), what Sinex decides for Polylogue, parsed records in Sinex, visibility and status, local state, several machines, rebuild, moving user state, moving an archive onto Sinex, and queries across both products.

The ingress, frames and acknowledgments, the confirmed stream, refusals, kind registration and the lifecycle on the wire are Sinex's contract, in Sinex's [producers](https://github.com/Sinity/sinex/blob/master/docs/producers.md) and [lifecycle](https://github.com/Sinity/sinex/blob/master/docs/lifecycle.md) pages; this page states only Polylogue's side. Polylogue uses that contract as any outside program does: Sinex has no path that is Polylogue's alone. The two modes are introduced in [architecture.md#modes](architecture.md#modes).

This page is the design; primary mode is not built in this version. A build with the `sinex` feature refuses to start on an archive recorded as primary or with a configuration asking for it ([installing.md](installing.md#deployment)), and the parts below are written on the four operations of the record and the bytes: put, open, commit, and read after a position.

## The rule

**In both modes the record is the durable account of what this Polylogue captured and what its users authored, and the index follows the record** ([architecture.md](architecture.md#pipeline)). In primary mode the record converges with Sinex: everything in it is shipped to Sinex, everything Sinex decides is applied to it, and it can be rebuilt from Sinex.

Primary mode adds three parts beside the record. Each moves data one way, and none of them is on the path between a captured byte and a row a reader sees.

1. **Bytes are handed to Sinex's store and read from there**, so they exist once ([Bytes](#bytes)).
2. **The record is shipped to Sinex in commit order** ([Shipping the record](#shipping-the-record)).
3. **One reader of Sinex's committed log applies what did not originate here**: other hosts' records and Sinex's lifecycle changes ([What Sinex decides](#what-sinex-decides)).

- **What is the same code in both modes:** discovery and planning in the engine, capture and the batch it hands over, the parsers, the index with its schema, upkeep and build, the preparation of authored writes and their stale-base check, queries, reading, rendering, every surface, and the tests of all of these. None of them asks which mode it is in.
- **What differs by mode:** where bytes are put and opened; whether the daemon runs the shipper and the reader; one section of `status`; where the shared settings come from; and the removal of record rows that Sinex has tombstoned, which standalone never does.
- **Settings shared with Sinex.** The logical day, quiet hours and the interrupt budget are Sinex's settings; Polylogue has no value of its own for them, and its interrupting notices are shown by Sinex's deliverer, except `Sinex blocked`, which Polylogue shows itself ([operations.md#notices](operations.md#notices)).
- **Polylogue never falls back to standalone**, whatever the cause, a broken configuration file included. An archive records its mode when it is created; opening it in the other mode is refused, naming both, and the mode is never inferred from the files present.

## Bytes

- **Captured bytes are durable on this host before capture goes on.** They are written and synced to a holding directory, and the record rows that cite them commit as in standalone. Capture never waits for Sinex: a tool may delete its file while Sinex is down.
- **Then they are handed over.** Each input revision is sent to Sinex as one material, its blobs as the material's ordered chunks. Once Sinex has acknowledged it, the record names Sinex as the place of that blob and the held file is removed. From then on the bytes exist once, in Sinex's store.
- **Bytes are opened wherever they are**: from the holding directory until they are handed over, from Sinex afterwards, by material ID and byte range, in pages, under Polylogue's own grant. Another host's bytes, and the bytes of an export staged in Sinex and handed to Polylogue by reference, are read the same way. Nothing above the two byte operations knows which place answered.
- **The same bytes, cut the same way.** Blobs and their hashes are those of standalone ([archive.md](archive.md#stored-inputs)), so an input is the same input in both modes and material digests remove repeated bytes.
- **What a crash leaves.** A held file whose blob the record already places in Sinex is removed at the next start. A material sent whose acknowledgment was not recorded is sent again under the same name, and Sinex answers from its record of the first.
- **Reading stored bytes needs Sinex; reading the index does not.** Text is in the index ([archive.md](archive.md#text)), so sessions are read and searched with Sinex down. A raw read, a re-parse or a build of bytes Sinex holds waits for Sinex and says so.

## Shipping the record

- **Everything the record gains is sent to Sinex, in the order it was committed**: a blob's material before the rows that cite it, then the rows (inputs, chains, observations, authored records, requests) as events of Polylogue's kinds.
- **There is no send queue.** The record and the held bytes are what is waiting, as a source file is for any producer. The shipper keeps one position in the record, which moves when Sinex acknowledges.
- **A resend is never a second write.** A frame's name is derived from this host and the record position it carries, so sending it again after any crash or outage gets the answer Sinex recorded the first time.
- **An outage only delays.** The shipper retries without a limit of attempts or age. Meanwhile nothing else waits ([Visibility and status](#visibility-and-status)).
- **A record Sinex refuses stays in Polylogue.** A refusal of a frame's content is recorded with its reason against the record rows it carried and shown in `status`, and the shipper goes on: the rows are already this host's truth, and a refusal is a difference between the two products to be fixed in a declaration, not a reason to hold capture.
- **A Sinex that was reset or restored to an earlier backup is sent again what it lacks.** The shipper checks its position against the identity of the Sinex store it shipped to, and ships again from the record whatever that store no longer holds. Bytes that were handed over after the backup exist nowhere; `status` names their chains, their rows stay readable from the index, and no build can reproduce them. That exposure is the price of keeping bytes once: Sinex's backup is their backup.

## What Sinex decides

One reader follows Sinex's confirmed stream. The stream carries references (commit position, event ID, kind); the reader fetches the payloads in pages under Polylogue's grant.

- **It commits into the record what did not originate on this host:** other hosts' inputs, chains, observations and authored records; exports staged in Sinex and handed to Polylogue; and lifecycle changes. This host's own records it recognizes by producer identity and frame name, checks against what it holds, and does not apply again.
- **The index learns of them as it learns of anything: from the record.** A page's records and the mark of how far the reader has read commit together; the index then follows the record as it does for a local commit. There is no second path into the index, and no difference between a chain this host captured and one it learned of.
- **Archive and restore.** When Sinex archives a material, its chains count as absent: their readings leave the index and their sessions are composed from the readings that remain, so what another input still states stays shown. When Sinex archives a published record of a session or a message, that item is not shown, searched or counted while it is archived. A restore undoes either: the chain is read again from its bytes, the item is composed again.
- **Tombstone.** When Sinex tombstones a material, Polylogue removes the record rows that describe it (paths, observations, authored text among them), its readings, the texts that only those readings held together with their search entries, and rewrites any content group that held them, so no erased byte stays in a group beside others. This is the one case in which Polylogue removes anything, and it happens only on Sinex's decision ([architecture.md#deletion](architecture.md#deletion)). Bytes still in the holding directory cannot have been tombstoned, since Sinex has not seen them.
- **A change in what Polylogue may read.** When Sinex changes Polylogue's grant, answers keep coming from the current index until an index built under the new grant has caught up, and nothing computed under the old grant is sent to Sinex after the change begins.
- **A stopped reader stops nothing else.** Capture, the index and the shipper do not wait for it; a page it had not committed is delivered again.

## Parsed records in Sinex

AI work is present in Sinex beside everything else, as events its timeline, recall and derivations can read.

- **After a chain's reading commits in the index, the records composed from it are published**: the shown session, message and tool-call records, each keyed by its kind and its stable ID, with the inputs and chains that contributed carried in the payload. A message is one record in Sinex however many routes, machines and rewrites brought it.
- **Publication follows the index and never feeds it.** Nothing is queued before the index commits; a chain's state notes the parse version and length last published; published records are never read back into the index.
- **A parser upgrade is a replay.** A chain read again under a new parse version is published as a producer replay of that chain, so the new records become live and the previous ones are archived with what was derived from them. Authored records about a session name it by natural key and are untouched.
- **One host publishes an input's records:** the host that wrote the input ([Several machines](#several-machines)). A record composed from several hosts' inputs is published under one key and converges whichever host sends.
- Polylogue's kinds are declared in Polylogue's code and reach Sinex as an installed catalog ([kind registration](https://github.com/Sinity/sinex/blob/master/docs/producers.md#kind-registration)); Sinex knows none of them by name.

## Visibility and status

- **This host's own writes are visible at once.** A captured message is readable and searchable after the local commit, as in standalone, and a note or tag replies as saved. Neither waits for Sinex.
- **`status` shows what Sinex does not have yet:** the count and bytes of record rows and blobs not acknowledged and the age of the oldest; refusals with their reasons; the mode, the Sinex endpoint and this host's producer identity; whether Sinex is reachable and the time of the last acknowledgment, which is kept so a restart during an outage still says how long Sinex has been silent; how far the reader has read and its lag; and chains whose bytes Sinex no longer holds.
- **Sinex unreachable.** Polylogue works fully for this host: capture, the index, authored writes and every read of the index. Raw reads and re-parses of bytes Sinex holds wait. The verdict is `degraded` with the reason `sinex blocked` and the error in plain words, and the shipped `Sinex blocked` rule fires after 10 minutes. Sending and reading are reported apart: a successful read does not clear a failed send.

## Local state

What a host keeps in primary mode:
- `meta.json`, the archive's storage identity, containing its mode;
- `archive.db`, the record, as in standalone, with the place of each blob (held here, or in Sinex), the shipper's position and the reader's mark;
- the holding directory, for bytes Sinex has not acknowledged;
- `index.db` and its full-text files, the same as in standalone;
- the embeddings store, local and backed up, since recomputing it costs money.

A backup of a primary host is a backup of these files; nothing has to be paused or pinned for it, because nothing is deleted from the record when it is shipped. NATS connections to non-loopback endpoints require built-in TLS before authentication, protecting original bytes as well as producer credentials. The Sinex adapter is the Cargo feature `sinex`, off in a plain `cargo build` and included in the Nix package; a build without it has no primary mode and no code that asks for one.

## Write identity

- Every write has one saved identity and immutable contents; a retry reuses both.
- A later write gets a new identity even when its content equals an earlier one.
- Which chain of an input is newer is decided by the machine that wrote it ([archive.md](archive.md#stored-inputs)); the reading order of a published record's key is assigned or checked by Sinex. Neither is taken from a clock.
- Material digests keep removing repeated bytes.

## Several machines

Each host is its own producer identity (credential and frame namespace) under one Polylogue kind family, and the machine in its inputs' identities is that identity's host. A host's record holds its own rows and the rows it learned from Sinex; hosts that have shipped and read everything hold the same record and therefore the same index. Two invariants make that true without any order shared between hosts.

- **No input has two writers.** An input is tied to one machine or is its content ([archive.md](archive.md#stored-inputs)). Chains are numbered by their one writer, are never renumbered, and are never compared across hosts. A content-named input is the same bytes in one chain on every host: learning of it from another host is a repeat, and adds that host's observations. Two hosts that hold the same session through inputs of their own (a tool directory copied between machines) are two inputs of one session, composed like any two routes ([sessions.md](sessions.md#one-conversation-through-several-routes)).
- **Authored state is a function of the set of accepted records.** What a host ships and another applies is the accepted record with everything decided when it was accepted: its ID, the revisions it revises, the place it took, its number among the events of its job or delivery, whether a run was pre-registered. A host that applies it decides nothing again and numbers nothing. Applying is insert-if-absent by ID: the same ID with the same content is a repeat, and the same ID with other content is a conflict that is reported and never settled by which came first. Which revisions are current is a function of the revision graph ([annotations.md#current-revisions](annotations.md#current-revisions)); a setting's value is read from the current heads; where one of several equal statements is kept, the choice is by a value the statements carry. A host's commit positions order what that archive learned and when (its change feed), and never choose what a record says.
- **Concurrent edits stay visible.** No host sees another's writes before they are shipped and read, so two hosts that revise the same record at once produce two current revisions, shown as a conflict and resolved by an edit that revises both.
- **Any host can read any chain.** Parsers are pure and bytes are opened by ID, so a host indexes the chains it learned of exactly as it indexes its own.

## Rebuild and parser upgrades

- **A rebuild is the ordinary build.** An index from another build, or none, is built from the record with bytes opened from Sinex ([architecture.md](architecture.md#one-path)); there is no primary build path and nothing is re-sent.
- **Restoring the record.** A host with an empty record starts the reader from Sinex's snapshot. The same reader, with the same code, fills the record, this host's own earlier records included, and the index is then built. The host continues its own numbering after the highest position Sinex holds for it. What it had committed and not yet shipped when the record was lost is lost with it.
- **Parser upgrades.** A chain whose parse version changed is read again from its bytes, as in standalone, and published again as a replay ([Parsed records in Sinex](#parsed-records-in-sinex)).

## Moving user state

User state (annotations, their revisions and links, alternates, deliveries) moves between archives without Sinex as a bundle: `polylogue 'repo:sinex' read --to DIR` writes the matched sessions with their annotations and alternates, and a query over the annotation units writes user state alone (`polylogue 'annotations where author:operator' read --to DIR`; [reading.md#formats-and-destinations](reading.md#formats-and-destinations)). `polylogue import DIR` applies the records through the one write path ([annotations.md#the-write-path](annotations.md#the-write-path)), keeping each record's stored author with `via: import`; it is idempotent by ID, and conflicting revisions become visible heads. Records of an earlier store arrive the same way ([capture.md](capture.md#earlier-stores)).

## Moving to Sinex

A standalone archive moves onto Sinex in place. The archive is attached to Sinex: the shipper sends the record from its start and hands the bytes over. The index, the authored records and the paid vectors stay where they are, and nothing is parsed again. `status` shows what Sinex does not have yet while the move runs. The local segment files are kept until the archive has been compared with Sinex's copy, by natural key and byte hash, and are then set aside; until then every byte still exists on the host. Deployment configuration owns the change of mode.

## Cross-system queries

The Polylogue daemon evaluates queries that mix Polylogue and Sinex data, because it holds the AI text, which Sinex does not index; Sinex callers reach AI text through this route.

- Names are bound from the parsed query with the `pl.` and `sx.` prefixes, never from strings; namespace binding is defined by the shared language in common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md).
- Sinex parts are compiled for and run by Sinex through its API; results join on references and time.
- Every reply reports the positions it read in each product and their lag, so a mixed reply is reproducible and its staleness visible.
- `sinex:<kind>/<id>` references are a form whose grammar Sinex owns ([references.md#forms](references.md#forms)).
