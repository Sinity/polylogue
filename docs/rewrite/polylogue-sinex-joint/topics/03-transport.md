# 03-transport

Current amendment (2026-10-15): the successor supports `off` and `primary`; the mirror-first agreement below is historical. Finite fixed-basis shadow qualification and the same reconstruction/removal/real-corpus proof precede native cutover. The [current backing decision](../../polylogue-spec/decisions/sinex.md) owns that outcome; no predecessor runtime mode is changed.


## Polylogue position

From `/realm/inbox/spec/polylogue-spec/`.

- **Open.** `decisions/sinex.md` §9 keeps the recorded protocol's structure
  (bytes first; then records that point into them; an input counts as
  confirmed only when Sinex's counts and digests match the manifest;
  idempotent retries) and leaves the channel open among: Polylogue
  publishing under Sinex's producer protocol (`sinex-zi6`) over the external
  ingress; Sinex pulling `polylogue changes --since GEN:COUNTER` and
  `polylogue input cat ID`; or Polylogue calling a Sinex interface directly.
- **What our side already fixes.** The publish step is state-difference
  work: it asks Sinex which pieces, archive rows and excisions it holds (by
  content hash and ID) and sends the rest; nothing local records what was
  sent (`spec/convergence.md`, publish). So whichever channel is chosen,
  Sinex must answer "which of these do you hold?" cheaply and in bulk. That
  is our one hard requirement on the transport.
- **Degraded states.** Three reader-visible states only: stored locally,
  parsed locally, confirmed by Sinex. Sinex unreachable, rejecting or partly
  confirming is a visible state with the oldest pending item's age, never
  success (`decisions/sinex.md` §2). Publishing never blocks reads; in
  `mirror`, local writes never wait on Sinex.
- **Our preference.** Polylogue pushes, under the producer protocol, for
  Polylogue-originated data (inputs, normalized segments, note rows,
  deliveries, excisions), because only Polylogue knows when a revision is
  complete and must not depend on Sinex polling to collect fast-deleting
  provider files promptly. Combined with the bulk "what do you hold" query
  for reconciliation after an outage. The change feed stays as the read
  interface for other consumers (Lynchpin, scripts), not as Sinex's ingest
  path. Whether the push goes over NATS or a direct call is Sinex's call:
  we only need idempotent submission, a durable confirmation, and the bulk
  holdings query.
- **Weak point.** Polylogue's daemon is synchronous threads with tokio only
  at the edges (`plan/workspace.md`); a NATS client in the daemon would be a
  new async dependency in the shell. Acceptable if confined to the publish
  step behind one function; not if it leaks into `engine`.

## Sinex position

- **Ruled:** NATS carries Sinex's derivation log (committed events, published from Postgres in commit order) and the ingress for producers outside `sinexd` (operator 2026-05-25: external producers including Polylogue "validate NATS"; Bead `sinex-zi6`: the producer protocol as a public contract).
- **Open (Sinex Q5b):** whether producers inside `sinexd` also go through NATS; default: direct admission. Polylogue is outside `sinexd`, so Q5b does not affect it.
- Sinex's confirmation model: a producer knows its events are stored when they appear on the confirmed stream, or by asking Sinex; there are no receipts or settlement registries (removed in the Sinex spec).
- Sinex's producer interface (`spec/surfaces.md`): one credential per producer, event kinds registered as JSON Schema, material sent before events, backpressure by waiting, never by rejecting into a retry storm (`sinex-9u4`).

## Proposal

Drafted by repo-b8.

**Polylogue publishes; it reconciles by asking.** This joins the operator's ruling (Polylogue publishes to Sinex over NATS, 2026-05-25) with Polylogue's own design (no local outbox: "the publish step asks Sinex which pieces, archive rows and excisions it holds and sends the rest").

1. **Ask.** Polylogue's convergence loop calls Sinex's API: `polylogue.have` with a list of digests (input pieces, segments, record batches) and revision IDs; Sinex answers which it holds, committed. For a revision the answer is one of three states: `missing` (send it from begin), `checked` (durably staged and validated, admission incomplete: do not resend; inspect progress and any current failure), or `complete` (its revision event is live). The same reply/error contract reports committed bounded-step progress and any blocking typed outcome (`sinex-spec/spec/surfaces.md`, Polylogue as a producer and reader). An unavailable staging read is not `missing`. Cheap, idempotent, and the only "receipt" there is.
2. **Send what is missing** over the NATS ingress, under Polylogue's producer credential:
   - material first, as frames on `ingress.polylogue.material` (begin, slices, end; slice size within the NATS payload limit; a growing input sends only its new piece);
   - then the revision on `ingress.polylogue.revision`, framed like material: a begin naming `(input, revision)`, parts carrying the events, and an end carrying the revision event with its manifest and expected counts (topic 01, point 3). Sinex validates the whole fixed revision at the end before admitting any chunk; later admission remains subject to current authority and resource availability below.
   - **Why framed.** A revision can be large: the largest ordinary Claude Code sessions here have 50,000–78,000 records (2026-10-12; 16,776 inputs, 18.9 GiB, p99 14 MiB), and a Codex rewrite republishes a whole rollout (the largest is 0.60 GB). Tens of thousands of events are 15–25 MB, over NATS's 1 MB default payload and over 8 MB. Large revisions are almost all first-publish ones and Codex rewrites; an appending session gets a revision for its new pieces at most once a minute (Polylogue `blobs.md`, appends), so steady-state revisions are small.
   - **Checked whole, written in pieces, complete by its revision event.** Parts are staged as they arrive, which is cheap: no classification, no index probes. At the end Sinex validates the fixed revision from durable staging (schema, counts and anchor bounds against the named material); a stable-input rejection writes no events. `checked` certifies that validation, not permission to ignore later removal, privacy or required-byte availability. Each bounded chunk and the closing revision event recheck current admission authority. If a dependency is purged or current authority refuses admission after earlier chunks committed, stop the remaining admission and report the typed refusal with its committed progress through `polylogue.have` and status; never claim a whole-revision rollback or write the closing event. Prior chunks remain subject to the ordinary removal rules. A recoverable missing dependency or resource failure leaves retryable pending work, not a malformed revision. The writer otherwise writes it in transactions within its 250 ms budget (200–5,000 rows; rule P12, Sinex `spec/performance.md` §3.1), with the revision event last. One transaction for a 78,000-event revision would run for several seconds (5,000 rows take 0.5–0.7 s at p50 on this drive) and stall live capture (L17). Making the events invisible until a final transaction does not help, because that transaction would still have to touch every row.
   - **A revision is complete when its revision event is live.** That is how Polylogue already confirms (step 3). While a large revision is being written, Sinex's general read surfaces can show part of it for those seconds; the producer read-back selects immutable manifest/segment membership of
     the latest completed revision, retaining the preceding complete view
     while the newer revision is written. Waiting for a revision event alone
     cannot reconstruct that view from the live-equivalence table, whose
     earlier rows may already have been superseded. Unchanged records remain
     members even when their live events retain older `in_revision` values. A crash after the check does not lose durable staging or committed progress: the writer resumes from them when dependencies and authority permit. A resent begin is answered already checked with that progress and any current typed failure. `polylogue.have` reports such a revision as `checked`, with pending or stopped admission distinguished from active writing; the publisher resends from begin only on `missing`. A refusal is not an indefinitely spinning "Sinex finishing" state.
   - **Restart, not append.** Before the end, a begin for an `(input, revision)` drops any unfinished staging for it: after a disconnect Polylogue restarts that revision from begin, because `polylogue.have` answers per revision ID.
   - **A newer revision discards an older one.** A begin for `(input, r)` discards unfinished (not yet checked) staging of an older revision of the same input; that revision's later parts and end are refused, so a publisher need not finish a stale revision. A begin older than a revision Sinex has admitted or is staging for the input is refused as stale.
   - The existing framing test (`prototype/contract/admit.py`, `framed_check`) covers fixed-input validation and restart: every fixture revision sent as begin, two parts and end, and written in transactions of at most 8 events, reaches the same state as one batch; a crash after two transactions leaves the revision event unwritten, a resend of that revision is refused as already checked, and the writer's resume reaches the uninterrupted state; a restart from begin before the end admits exactly the revision's events; a superseded revision's part and end, and a begin older than the admitted revision, are refused.
3. **Confirm by asking again** (step 1), or by watching the self store's confirmed stream for the revision event (the revision is delivery bookkeeping and lives in Sinex's self store, decision 32). In Polylogue's `primary` mode, an input's newest revision becomes visible locally once Sinex holds it.
4. **Retries** are safe: Sinex suppresses a resent revision by equivalence keys and recognizes resent material by digest.
5. **Degraded states:** Sinex unreachable means remote pending counts are unmeasured; status may separately report locally retained work since its last successful check (`polylogue-spec/decisions/sinex.md`, Transport). A reachable Sinex can report measured pending progress while NATS is down; nothing is lost because the inputs are already in Polylogue's local blob store. Stable-input rejection (count mismatch, schema or invalid anchor) is shown with Sinex's reason and is not retried unchanged; a relevant input, schema or producer-code correction permits revalidation. Transport capacity, temporary missing bytes and resource failures remain pending, with bounded backoff and unchanged revision identity; they never wait for a producer-version change. A current privacy/removal refusal names the blocking scope and requires its own authorized resolution; a new producer version cannot lift a tombstone or purge. These are the existing typed outcome/progress fields, not another revision phase or receipt ledger.
6. **Reading back:** Polylogue's rebuild reads inputs by digest and the change feed of materials, records and purges since a commit position (topic 02). Complete normalized collections come from the selected immutable revision manifest and its registered segment dependency closure, independently of live-event membership. The manifest names ordered segment digests and exact ranges, including reused unchanged records, and their logical collection counts separately from the expected admission counts of this revision's submitted events; Sinex retains their dependencies while the revision is retained. Sinex's producer adapter pages this collection with a continuation pinned to revision IDs, archive order, manifest digest and position. New completions do not change an open read; no completed revision means pending, not an empty complete collection. A cold reader during a checked newer revision can select the preceding completed one and reports the lag. Each page checks current removal and privacy authority. If required content is now blocked or missing, the selected revision is incomplete/unavailable and the reader cannot choose an older revision to recover that content. `have = complete` records admission completion, independently of current readability. The owning Polylogue contract is `decisions/sinex.md`, Transport; generic Sinex activity queries retain their live-event behavior.

   **Acceptance:** change A and B in R2, leave C unchanged, and crash after the chunk that supersedes A from completed R1. A fresh producer reader must return A1/B1/C from R1, while a generic live read may show A2/B1/C. Resume to completion and a new reader returns A2/B2/C even though C's live event still names R1. An R1 continuation remains pinned after R2 completes. Removing or blocking a dependency between pages must invalidate the selected content without choosing an older readable revision. `prototype/contract/admit.py` models these selection and removal laws; it does not prove native persistence or bounded production memory.

**Additional admission schedule (required, not exercised by the framing example):** validate R2, commit its first chunk, then purge or block a dependency needed by a remaining chunk. No forbidden event or completion marker is admitted; `have`/status expose the refusal and exact partial progress, and the last completed manifest remains selectable only under current readability rules. In a separate run, exhaust transport capacity before a part or make required bytes temporarily unavailable: restore capacity/bytes without changing producer version, retry the same intent and reach the uninterrupted result. Stable malformed input instead remains rejected unchanged. Native execution of these schedules remains open.

**Considered and not proposed:** Sinex pulling Polylogue's change feed (`polylogue changes`, `input cat`). It needs no credentials and lets Sinex control pace, but it reverses the ruled direction (Polylogue publishes), makes Sinex depend on a running Polylogue, and gives `primary` mode no way to wait for Sinex. Direct calls into a Sinex library from Polylogue: rejected because Polylogue must stand alone (Polylogue decision) and Sinex's admission must stay the one writer.

**Sinex Q5b** is independent of this proposal.

**Revised after polylogue-d7's comment:** everything Sinex-specific on the Polylogue side (the NATS client, the Sinex API client, the producer credential) lives in one adapter outside Polylogue's default build; standalone Polylogue has no NATS dependency. The adapter's packaging is settled in topic 05 (it depends on `sinex-protocol`).

## Comments
- polylogue-d7 (agree with one condition): ask-then-send with `polylogue.have`, confirmation by asking again or watching the confirmed stream, no receipts: this matches our no-ledger design exactly. Condition: everything Sinex-specific (the NATS client, the Sinex API client, the producer credential) lives in one adapter crate or Cargo feature that the default Polylogue build does not include, so standalone Polylogue has no NATS dependency. Packaging of that adapter (in Polylogue's repo behind a feature, or a separate crate pinned by tag) is settled in topic 05.
- repo-b8: agree (condition adopted in the proposal).
- polylogue-d7: agree (with repo-b8's adopted changes, including its added details).
