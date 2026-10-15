# Decision: Polylogue on Sinex

Status: **decided.** Polylogue with Sinex off (SQLite plus blob files) is
fully working and the default; nothing in it needs Sinex to build, test or run.
For the operator, Polylogue runs on Sinex: Sinex keeps the durable evidence,
and Polylogue's local SQLite becomes a copy rebuilt from it. The mechanics
were agreed with the Sinex session (`/realm/inbox/spec/polylogue-sinex-joint/`,
`decisions.md`, cited as joint #N). `primary` mode ships right after the
real-corpus rebuild proof passes. Material configured as sealed is encrypted in Sinex's
store (joint #19, #23, #24): Polylogue reads sealed bytes only through
Sinex's API while their class is unlocked, subject to the sealed interpretation boundary below.
Decided by the spec authors from joint #7 and #10 and Bead `sinex-4j2.3`;
the operator may overrule.

The long form of this record, with per-section history and quotes, is
`evidence/sinex-decision-history.md`.

Sources: Polylogue epic `polylogue-303r` and the Python repository's
`docs/sinex-interop.md`; Sinex epic `sinex-4j2` (Phase B `sinex-4j2.3`) and
`sinex-zi6`; operator rulings 2026-04-01, 2026-05-25, 2026-06-27,
2026-07-08/10, 2026-07-12/13, 2026-09-17, 2026-09-25 and 2026-10-11.

**Sealed interpretation boundary.** Reading sealed bytes through Sinex's API does not make Polylogue's ordinary persistent parse/index pipeline safe for sealed content. Until an operator-only, session-owned ephemeral view covers indexes, vectors, cached results, retained reads and late output, sealed inputs are explicitly deferred/unsupported for interpretation and search. Plain source materials continue through the ordinary primary path. The lifetime and revocation contract belongs to Sinex's privacy owner (`sinex-spec/decisions/privacy.md`, Unlock lifetime, caches and publication); Polylogue holds no class key and cannot silently persist returned sealed plaintext.

## The rule

Sinex owns durable persistence and the lifecycle of the evidence; Polylogue
owns what the AI-work data means and how the product behaves (operator,
2026-07-13). Nothing irreplaceable stays only in local SQLite in backed
mode. Ruled out: a metadata-only bridge, an attachment opaque to Sinex, a
second conversation model inside Sinex, a duplicate query language in
Postgres, and session lineage encoded as Sinex derivation ancestry.

## Supported arrangements

The rewrite supports standalone `off` and native Sinex-backed `primary`. Permanent `mirror` is retired from the successor scope. Finite shadow comparison supplies its qualification role without maintaining a third product with local authority and a continuously synchronized Sinex copy. It qualifies native backing as described below. Existing predecessor installations are not changed by this decision. The successor refuses `mirror` configuration with an unsupported-mode error; it never maps it silently to `off` or `primary`.

## Modes

| Mode | Source of truth | What Polylogue does | What reads see |
|---|---|---|---|
| `off` (default) | local `archive.db` and blob store | no Sinex work at all | everything local |
| `primary` | Sinex, the only physical store of source material (joint #27) | places each new input into Sinex's store and announces it; keeps only a pending area of unconfirmed inputs; an input's newest sessions reach the local index only after Sinex confirms it | the local index, built by parsing materials read directly from Sinex's store; unconfirmed newest revisions wait, and sealed interpretation remains deferred as stated above |

- The mode and the Sinex address live in `$XDG_CONFIG_HOME/polylogue/sinex.toml`, not in
  `archive.db`, so a rebuild from Sinex can start from an empty archive.
  A `sinex.toml` that does not parse, or names an unknown mode, never stops
  `status` or reads: publishing pauses (in `primary`, new inputs wait in
  the pending area; nothing switches to `off` silently, which would start a
  second store), and `status` shows the error with the file's path
  (`polylogue-1tyq5.89`).
- Nothing local records what was sent: the publish step asks Sinex what it
  holds and sends the rest (`spec/convergence.md`).
- In `primary`, Polylogue keeps no copy of source material (operator ruling
  2026-10-12; joint #27). Because a provider may delete its file at any
  moment, the first durable write of a new input is a **pending area** that
  holds only inputs Sinex has not confirmed; each is deleted once
  `polylogue.have` answers complete, and `status` shows its size ("pending
  2 inputs, 14 MB, waiting for Sinex"). On the same host the input goes into
  Sinex's store by placement: Polylogue writes the blob file into Sinex's
  incoming directory and announces digest, size and revision; Sinex verifies,
  registers, then clones the verified bytes into a sinex-owned, read-only
  file in its store (a reflink on btrfs, so no copy cost; a rename would
  keep the operator as owner) and deletes the incoming file, and alone deletes anything it
  cannot verify. Polylogue reads plain materials directly from Sinex's store
  root, read-only, through `hashed-blobs`; sealed ones only through Sinex's
  API while their class is unlocked. sinexd runs as its own user, so access
  goes through ACLs the Sinex NixOS module sets, never through a wider
  umask: Sinex's incoming directory carries a default ACL granting sinexd
  read and delete, so a file Polylogue writes there under its 0077 umask is
  still readable to Sinex; the store root grants the operator read on plain
  materials only (Sinex `sources.md` §1.11). `archive.db` keeps Polylogue's own
  rows (inputs, revisions, observations, notes), referring to materials by
  digest, and publishes every row to Sinex as an insert-only record, so
  `archive.db` too can be restored from Sinex, with each row's published
  `local_seq` kept (`spec/storage/archive-db.md`, local order).
- `primary` is allowed only after finite shadow qualification and the rebuild proof below pass on the real corpus (joint #7, #10). Standalone stays complete and first; a missing partner does not block its development.

## Shadow qualification and cutover

A shadow comparison is a finite run against a fixed standalone archive basis and isolated Sinex endpoints, through the same native publisher and read-back adapter. It records the candidate pair, archive frontier, configuration and terminal result using the existing demonstration/job owners. It has no permanent daemon mode or automatic publisher in `off`. The source archive remains authoritative during this run. Synthetic examples precede authorized private samples and the real-corpus proof; a protocol-helper run alone does not qualify the actual producer/client route.

Compare rebuilt product answers, references, authored rows, deliveries and removal behavior, not just transferred counts. Retain all dependencies needed by that fixed basis and apply current privacy/removal rules to every copy and read. A failed run leaves standalone usable. Shadow copies are subject to their declared retention/disposal policy, not indefinite alternate authority.

Cutover is a separate controlled transition after qualification. Bind or pause incoming writes, record the final standalone basis, establish complete retained inputs and archive rows, rebuild and compare the native view, then explicitly select `primary`. New writes remain with the declared previous or pending owner until that switch. Neither changing configuration nor a transport acknowledgment proves this transition complete. Disposing of the predecessor source store needs its own conservation/recovery checks and authorization. Sinex outage never switches authority: pending bytes stay local, newest unconfirmed revisions wait, and already confirmed reads remain usable under current access/removal rules.

## What each side keeps

| Data | Backed mode | Sinex off |
|---|---|---|
| Inputs: provider files, exports, captures, hook spools, attachments | Sinex keeps the bytes | blob store |
| Normalized transcript: sessions, messages, blocks, tool calls, links, events | Sinex keeps immutable normalized segments and events pointing into them; Polylogue defines their meaning | rebuilt into `index.db` |
| `archive.db` rows: notes (pages included), tags, judgments, saved queries, kept results, settings, excisions | Sinex keeps them as insert-only events | `archive.db` |
| Context deliveries | travel inside the hook-spool inputs | derived from the hook spool into `index.db` |
| Embedding vectors | Polylogue's `embeddings.db`, the one local-only store: vectors cost money to recompute but are not irreplaceable, so they are backed up with the filesystem, not published | `embeddings.db` |
| Search index, facts, display order, everything derived | local `index.db`, rebuilt from Sinex | local `index.db` |
| Machine context (shell, git, browser, files) | Sinex; Polylogue reads and shows it | not available |

Pages need no move to Sinex later: they are notes, and backed mode already
keeps notes in Sinex. Decided by the spec authors from the definition of a
page (`spec/user-data.md`); the operator may overrule.

## What Polylogue sends (joint #3, #5)

Three kinds of immutable material, each with events anchored into it:

- provider input pieces: the exact bytes, growing files as appended pieces;
- normalized segments: export-format session and message objects, one per
  line, a head segment (session, lineage, usage) and transcript segments
  reused byte for byte across appends; segment boundaries follow input
  pieces; the head segment also carries the session's turns
  (`polylogue.turn`: id, session, start, end, prompts, answer; `spec/query.md`,
  turns), derived by `polylogue-model` and re-sent whenever the session is
  rewritten, so Sinex can join commands and other events to the turn's span
  (joint #22);
- record batches: `archive.db` rows only.

Deliveries travel once, as hook-spool input pieces and in the segments.
Sinex never re-parses provider formats; its automata may derive from
`polylogue.*` events. A revision is checked whole and refused whole if its
counts don't match its manifest; it is complete when its revision event is
live. Polylogue's reader selects a complete revision through its immutable
manifest and segments, so it sees the old revision or the new one, never half
(joint decision 26; initial transfer and large revisions, below). The codec is the export format and
lives in `polylogue-model`. The Python codec's bytes are not carried over.

## Identities (joint #4)

| Kind | Polylogue | In Sinex |
|---|---|---|
| Stable object | `<origin>:<native id>` and the message and block IDs derived from raw records (`spec/model.md`) | the equivalence key of its `polylogue.*` event, cited as `sinex:eq/polylogue/<kind>/<id>` with the ID percent-encoded |
| Resolved object | the object as Polylogue resolved it from every route that carries it (joint #28), with source references to each contributing input revision, the parser version and the latest excision applied | the live event for its equivalence key: a later publication with a changed record hash supersedes it, an unchanged one is suppressed; Sinex never merges routes itself |
| Source occurrence | input, revision chain and byte range, one per contributing input (`spec/export-format.md`, `sources`) | the segment line's coordinates |
| Sinex event | never used by Polylogue | Sinex's own table maps Polylogue IDs to its events; rows are added, never changed |

An exact citation records the source occurrence (with its chain) and the
parser version beside the object ID. Lineage is stored in Sinex as typed references (joint #5).

## References to Sinex context

Polylogue stores no reference to a Sinex derived object whose identity can
move: a derived session's or activity window's bounds change when a later
source is staged (Sinex's U4 prototype moved a session's start from 09:10:40
to 09:02:00), so bounds are not a key. Where Polylogue shows Sinex context
(the machine section of a briefing, a session placed among the operator's
activity), it stores a time and a scope and asks Sinex at read time for
"the window containing T", written in Sinex's form
`sinex:at/<time>/<output type>` (for example
`sinex:at/2026-09-14T07:16:30Z/activity.session.boundary`), or cites a
member event by its equivalence key.

## Deletion (joint #6)

- With Sinex off, excision is as in `spec/storage/archive-db.md`.
- In backed mode Polylogue removes local copies at once and sends Sinex an
  explicit **purge order**: the exact content event keys to tombstone (never
  the `polylogue.input` and `polylogue.revision` bookkeeping, which carries no
  content and stays live so the frontier never moves back, joint #29/#32),
  the materials
  to shred (only materials no remaining or re-published event still rests
  on: materials are content-addressed, so an unchanged object re-published
  lands on the material it had), and a replacement revision for each input shared with sessions
  that are not excised (rewritten as `spec/storage/archive-db.md` says, the
  excised lines turned into spaces of the same length). Sinex never works
  out the closure itself: it checks the order against each input's
  `sessions` list (refusing an order that drops a shared input instead of
  replacing it, or leaves a sole-session input's content bytes unshredded), tombstones exactly
  the listed keys and everything derived from them, refuses re-published
  events on shredded materials, and destroys the bytes through its privacy
  mechanism (crypto-shredding, `sinex-spec/decisions/privacy.md`). The joint
  contract test exercises both cases (`polylogue-sinex-joint/prototype/contract/`,
  the `joint-contract` flake check). `status` shows both steps as pending until Sinex
  confirms each.
- Excised digests and scopes are refused on re-import on both sides.
- Agents cannot excise in either mode; Sinex removal is operator-only too.

## Deliveries

A delivery records exactly what crossed into a model's context and what was
left out; the same briefing given twice is two deliveries. Whatever hands a
briefing to a session (the hook, the MCP server, a ping) appends a
`delivery` line to the hook spool; the spool is a stored input, so each
delivery is durable, becomes a `delivery` row in `index.db`, travels to
Sinex with the inputs and comes back in a rebuild (`spec/storage/index-db.md`,
`spec/sources/README.md`). The delivered text stays in the recipient's transcript;
the row holds its hash.

## Rebuilding from Sinex (joint #7)

- There is no rebuild command. In backed mode a deleted `index.db`
  converges from Sinex by the same loop that builds it from local inputs.
- Polylogue's `index.db` always rebuilds from raw inputs (served by Sinex),
  not from its own segments. Sinex automata own Sinex's Postgres views of
  Polylogue data and build them only when a Sinex query needs them.
- Disagreements are found by comparing state and shown in `status`.
- A rebuild never brings excised content back. Sinex's read-back returns
  no row an excision's purge archived, and Polylogue's restored `excision`
  rows are standing rules the loop applies to whatever arrives
  (`spec/storage/archive-db.md`, how excision works), so either guard alone
  suffices.
- The proof: the Morrow archive converged with Sinex off and converged from a
  Sinex holding the same inputs and rows give identical export documents (law
  38 in `plan/testing/laws.md`, run in both repositories), plus a real-corpus
  rebuild that must match before `primary` is allowed.

## Transport (joint #8, #26, #27, #29)

Polylogue publishes over Sinex's NATS ingress under the producer protocol
(`sinex-zi6`).

- **Asking first.** The publish step calls `polylogue.have` with the digests
  and revision IDs it would send; Sinex answers each as missing, checked
  (validated durable staging, admission incomplete) or complete, with typed failure and committed progress when admission is pending or stopped
  (`sinex-spec/spec/surfaces.md`, Polylogue producer reply contract). An
  unavailable staging read is not `missing`. Polylogue sends only what is missing,
  material first, then the revision. Retries are safe: material by digest,
  revisions by equivalence key. Nothing local records what was sent.
- **Material.** On the same host it is placed into Sinex's store in batches
  and only digests cross NATS (joint #27, above); from another host it
  travels in 256 KiB slices (under NATS's 1 MB message limit), cutting
  through records where needed, since anchors are byte ranges.
- **Revisions are framed** (joint #26): begin, parts, end, because one
  revision of a large session is tens of thousands of events (15–25 MB).
  Sinex stages the parts and validates the fixed revision at the end
  (stable-input rejection writes no events), then writes it in transactions sized to its live-latency
  budget with the `polylogue.revision` event last. A revision is complete
  when that event is live. `checked` is not irrevocable admission: each chunk
  and the closing event recheck current removal/privacy authority and required
  bytes. A later refusal stops remaining admission with visible typed reason
  and committed progress; it never rolls back earlier chunks as a whole or
  certifies completion. Transient availability/resource failures remain owed
  retryable work. The joint transport contract owns the failure schedules.
  `polylogue.input` and `polylogue.revision` are
  bookkeeping, not activity: Sinex keeps them in its self store, out of
  activity queries, and returns them through the producer's read-back so a
  restore can rebuild local order (joint #32). After sending a revision's end, Polylogue does
  not resend on a disconnect; it asks `have`, and resends from begin only
  if the answer is missing. Large revisions are first-publish ones and Codex
  rewrites (the largest rollout is 0.60 GB); an appending session publishes
  a small revision at most once a minute.
- **What a revision event carries**: the input and revision, the expected
  counts per kind, the materials, `local_seq` and `prev_local_seq` (the
  archive's write counter at the revision and at the one stored before it;
  `spec/storage/archive-db.md`), so Sinex can compute how far it holds
  Polylogue's state (joint #29). The immutable manifest names the complete logical record collection,
  including unchanged objects, by ordered normalized segment digests and
  exact ranges, with logical collection counts per kind. These are separate
  from the expected admission counts of the events submitted in this revision.
  Append revisions may reuse segments
  and reference their retained predecessors; every dependency needed to
  reconstruct the collection must remain registered. Every event names its
  admission revision in volatile `in_revision`, outside the record hash. An
  unchanged message can stay live under its first admission revision; this
  field does not determine membership in later complete revisions.
- **Confirmation** is `have` answering checked or complete, or the event on
  the confirmed stream. NATS accepting a frame is not confirmation: frames
  waiting in an ingress stream could be lost before admission in today's
  deployment (sinex-uz3h; Sinex law L50 forbids it).
- **Reading back** in `primary`: Sinex's producer read-back selects the
  latest completed revision for each input at the read's declared archive
  order, then reads its immutable manifest and segments. It never assembles
  that collection from the generic live-event table or filters membership
  by `in_revision`: partial writes can already have archived the preceding
  complete revision's events, and unchanged objects keep older admission
  revisions. While a newer revision is checked but incomplete, the last
  completed revision remains selectable, including by a cold reader after
  restart. With no completed revision, the input is pending, not an empty
  complete transcript. Such results have `up_to_date: false` and name the
  pending revision in `notes`.
  The Sinex producer adapter owns bounded manifest/dependency and segment
  pages. Each continuation pins the selected revision IDs, archive order,
  manifest digest and position; a newer completion cannot change an open
  read. Across inputs, Polylogue applies its existing resolved-object rules
  in archive order, with the replicated frontier establishing compatibility
  (joint #28/#29). Generic Sinex activity queries retain their mixed live
  view during admission.
  Admission completion and current readability are separate facts. Every
  page applies current tombstones, excisions, purge state and privacy access
  to all required dependencies. A blocked or missing dependency makes the
  selected view explicitly incomplete or unavailable; neither an older
  manifest nor an already issued continuation can restore its content.
  Metadata describing a completed revision remains readable without its
  blocked payload. This historical material read is not an archived-event
  fallback, and grants no access beyond Sinex's existing material/privacy
  authority.
  Polylogue keeps the last `commit_seq` it applied from the confirmed stream
  (in `index.db`'s `meta`, so a rebuild re-reads from the start) and drops any
  message at or below it: the stream is an in-order prefix of Sinex's commit
  log across outages too (Sinex law L44).
- **While Sinex is unreachable** the pending count is unmeasured ("—, Sinex
  unreachable since T"); `status` may add "N inputs retained in the pending area
  since the last successful check at T", a local measurement, not a remote pending
  count. Nothing is lost: unconfirmed inputs stay
  in the pending area (`primary`). Stable-input rejection is shown with Sinex's reason and is not retried
  unchanged; a relevant input, schema or code correction permits revalidation.
  Capacity, temporary missing bytes and resource failures remain pending and
  retry with backoff under the same revision, without a version change.
  Privacy/removal refusal needs its own authorized resolution; changing a
  producer version never lifts a tombstone or purge. `have` and status
  distinguish such stopped admission from active writing.
  Sinex's ingress stream has a byte limit, so a fast publisher waits rather
  than filling the disk; `status` then says "sinex primary · sending,
  waiting for Sinex to catch up".

**Initial-transfer sizing** (estimates from Sinex's measured parts, not a
measured run). Today's archive is about 73 GB. Placed in batches (temp
files, one filesystem sync, one announcement, Sinex adopting the batch with
one row transaction and one sync; the batched durability `storage/blobs.md`
uses for builds), it is 3–16 minutes of drive time unpaced at the content
store's measured 75–397 MiB/s. Synced file by file it would be about 2.5
hours, because the inputs are small (median 0.33 MiB; 8 MiB/s with a durable
put per blob), so placement is always batched, except a single live input
whose provider may delete it. From another host, through the NATS stream,
20–30 minutes. Shared IO pacing (joint #25) stretches all of these,
plausibly to an hour or more. Writing the events adds 25–150 ms per 1,000
events (a 78,000-event revision staged in 1.2–4.4 s and written in
1.9–11.5 s while live capture stayed inside its latency target), so the
roughly 6 million Claude Code records plus Codex's add minutes, not hours.
A finite qualification run publishes its fixed basis; native operation publishes later revisions as they arrive. Each revision is independently retried by asking what Sinex holds. Restart still performs that work and is not assumed free.

Everything Sinex-specific (the NATS client, the Sinex API client, the
producer credential) lives in one adapter crate, `polylogue-sinex`, behind
the Cargo feature `sinex`, outside the default build. Polylogue built without that feature has no NATS dependency.

## Shared code (joint #9, #11–#14)

| Crate | Home | Used by |
|---|---|---|
| `polylogue-model` (record types, export-format codec) | Polylogue repository | Sinex, to read what Polylogue sends |
| `sinex-protocol` (the producer protocol) | Sinex repository | `polylogue-sinex` |
| `hashed-blobs` (the content store's file layer) | Polylogue repository | Polylogue's blob store and Sinex's store for all data |

- Shared only when both projects really use a crate. Each stays in its
  owner's repository and is pinned by a semver git tag; stable Rust; no
  tokio, rusqlite, sqlx or async-nats in shared crates.
- `hashed-blobs`: BLAKE3 for both projects; seekable 4 MiB zstd frames; the
  row is written before the file; blobs are immutable (excision writes new
  blobs and repoints extents); deltas against one base, page patches as
  ordinary extents, the rebase threshold set per source; extra storage
  roots are a capability, not a requirement; sealed blobs (joint #19, #23,
  #24) are encrypted per blob under keyed names.
  Small blobs inline in `archive.db` belong to Polylogue's input layer, not
  the shared crate.
- Shared as patterns, not crates: test kits, the lint table, the ID and
  time rules. A small shared xtask core, with build and test history
  recording, is extracted once both repositories run the same code
  (joint #13).
- Sequencing: `hashed-blobs` is built in Polylogue's slice 1. The "On Sinex"
  slice comes right after Polylogue's slice 4 and is the same work as Sinex's
  slice 6; backfill of the existing corpus comes after the proof passes.
