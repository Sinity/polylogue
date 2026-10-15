# 02-projections

Current amendment (2026-10-15): the successor supports `off` and `primary`; the mirror-first agreement below is historical. Finite fixed-basis shadow qualification and the same reconstruction/removal/real-corpus proof precede native cutover. The [current backing decision](../../polylogue-spec/decisions/sinex.md) owns that outcome; no predecessor runtime mode is changed.


## Polylogue position

From `/realm/inbox/spec/polylogue-spec/`.

- Polylogue's derived store is `index.db`, rebuilt from inputs plus
  `archive.db` rows by the one convergence loop; there is no rebuild command
  (`spec/convergence.md`). In backed mode a deleted `index.db` converges from
  Sinex by the same loop; only where inputs and rows come from differs
  (`decisions/sinex.md` §8). Law 38 (`plan/testing.md`): Morrow converged
  standalone and converged from a Sinex holding the same inputs and rows give
  identical export documents and `status --json` counts, notes and
  deliveries included.
- Our spec says nothing about Postgres projections of Polylogue data. It
  assumes Polylogue reads only its own `index.db` and does not query
  Postgres. Open: whether `store::archive` becomes a trait with a Sinex
  implementation or Sinex simply serves the inputs and rows that module
  reads (`decisions/sinex.md` §8).
- Weak point: law 38 runs on the example archive only; `sinex-4j2.3`'s
  decisive proof runs on the real corpus and includes "use Polylogue
  normally, then rebuild, then explain every difference". Our plan names it
  ("proved by the cross-backend rebuild law on real data", slice 13) but
  has no procedure for the real-data run.

## Sinex position

- Sinex does not own Polylogue's projections. Polylogue's `index.db` is rebuilt from what Sinex holds (inputs and record batches, topic 01) by Polylogue's own loop.
- Postgres projections of the Polylogue domain: for one operator on one machine, Sinex sees no need; Polylogue's local SQLite index serves Polylogue's queries, and Sinex's own queries see `polylogue.*` events as ordinary events for cross-source joins. If shared Postgres projections are ever wanted, Polylogue owns their schema and code; Sinex's schema-in-code covers only Sinex's tables. Sinex will not duplicate Polylogue's query language.
- The decisive proof (`docs/sinex-interop.md`) is adopted as Sinex slice 6's acceptance (`/realm/inbox/spec/sinex-spec/plan/slices.md`). Sinex's side of it: every byte and record Polylogue sent comes back by digest and in commit order, and purges are visible in the feed.

## Proposal

**One owner per table; Polylogue's meaning is never re-implemented in Postgres.**

1. **Who writes what.**

   | Data | Written by | Lives in |
   |---|---|---|
   | Polylogue inputs (provider bytes, pieces), normalized segments | Polylogue, through Sinex's admission | Sinex materials |
   | `polylogue.*` events (sessions, messages, blocks, links, usage, events; one per normalized record, anchored into segments) | Polylogue, through admission | Sinex events |
   | Note rows, settings, excisions, deliveries | Polylogue, through admission | Sinex events (insert-only) |
   | `index.db` (search, facts, display order, everything derived) | Polylogue's writer only | local SQLite |
   | Postgres projections of Polylogue data | **Sinex automata only**, from the `polylogue.*` events, for Sinex's own cross-source reads (timelines, joins with shell, git, browser) | Sinex Postgres, Sinex's schema |

   Polylogue never writes Postgres tables directly and never reads Sinex
   projections for its own product behavior; it reads Sinex's materials and
   events (inputs by digest, record rows, the change feed; Sinex §9 "What
   Sinex hands back"). Sinex's projections may answer Sinex queries such as
   "which sessions touched this repo while this shell command ran", using
   only the typed facts in the events; they never re-derive transcript
   meaning (no second Polylogue parser, no second query language;
   `polylogue-303r`). This keeps one owner per fact: Polylogue owns meaning
   and `index.db`; Sinex owns storage, lifecycle and its own projections.

2. **What `index.db` rebuilds from.**
   - `off`: local blob store plus `archive.db`.
   - `mirror`: the same; Sinex holds a copy. A rebuild from Sinex is
     possible (and is what the proof below exercises) but not the normal
     path.
   - `primary`: Sinex's materials and record events, fetched by the same
     `store::archive` reads (inputs by digest, rows by ID), with plain
     materials read directly from Sinex's store root (joint #27); Polylogue
     keeps no blob store of its own, only a pending area of unconfirmed inputs. Polylogue re-runs its own parsers on the raw inputs;
     the normalized segments in Sinex are for other consumers and for the
     comparison below, not Polylogue's rebuild source. Rationale: parsers
     improve; rebuilding from Polylogue's earlier output would freeze old
     interpretations.

3. **When the two disagree.** A disagreement is any difference between
   what Polylogue derives now and what its published `polylogue.*` events
   say for the same stable ID and parser version, or between local and
   Sinex holdings of inputs and rows. It is found by state comparison, not
   recorded as it happens: the publish step's bulk "what do you hold" answer
   (topic 03) shows missing or extra pieces and rows; a re-publish after a
   parser-version change emits new interpretation events under the same
   equivalence keys (Sinex supersedes the old ones; no conflict). A
   genuine mismatch (same stable ID and parser version, different content)
   is a bug on one side and appears in `polylogue status` as a count with
   example IDs ("Sinex differs on 3 messages: …"), and as a failure of the
   comparison test. No repair command: the fix is a code fix, after which
   convergence removes the difference.

4. **The proof.** Two tests, one shared fixture.
   - **Example archive, every change, both repositories**: Morrow plus the
     Claude-Code-only home are published into a Sinex test instance;
     `index.db` is deleted; Polylogue converges from Sinex; the export
     documents, `status --json` counts, notes, deliveries and every
     reference (session, message, block, note, kept result) are identical to
     the standalone build (law 38). In Sinex's repository the same archive
     checks that the `polylogue.*` events and projections match the
     contract file. This is the contract test of Sinex §9 "keeping the
     contract in one place".
   - **Real corpus, once per backed-mode milestone** (Sinex slice 6
     acceptance; Polylogue slice 13): import the operator's corpus into
     Sinex; use Polylogue normally for a period (notes, judgments, briefings);
     delete every rebuildable Polylogue store; converge from Sinex; compare
     sessions, messages, blocks, links, usage, notes, deliveries and
     references; every difference is listed with its IDs and explained in
     the milestone report; the demos (`spec/presentation.md`) are rerun.
     Passing it is what allows `primary` (`questions.md` #25).

5. **What this asks of each spec.** Polylogue: write the real-corpus
   procedure into slice 13 and `plan/testing.md`; state in
   `decisions/sinex.md` that Polylogue rebuilds from raw inputs, not from its
   segments. Sinex: state that Polylogue projections are Sinex-owned,
   read-only to Polylogue, and built only from event facts; add the bulk
   holdings query to the producer interface.

## Comments
- repo-b8 (agree, with two notes):
  1. Agreed: Sinex projections of Polylogue data are Sinex-owned, built only by Sinex automata from the typed facts in `polylogue.*` events, read-only to Polylogue, and never re-derive transcript meaning. Sinex builds them only when a Sinex query needs them (one operator, one machine), not by default.
  2. Agreed that Polylogue rebuilds from raw inputs by re-running its parsers, and that the normalized segments serve other consumers (Sinex automata, cross-source joins) and the comparison. The bulk holdings query is the same `polylogue.have` call as topic 03; Sinex adds it to its producer interface. Purge in `primary` mode must also clear Polylogue's local cache copies; covered by topic 01 point 7.
- polylogue-d7: agree with repo-b8's two notes (projections built only when a Sinex query needs them; polylogue.have is the same call as topic 03's holdings query, in Sinex's producer interface).
