# 01-contract

## Polylogue position

From `/realm/inbox/spec/polylogue-spec/` (paths below are relative to it).

- **Division of work.** As `polylogue-303r` and `docs/sinex-interop.md`
  recorded: in backed mode Sinex keeps the inputs (provider bytes, exports,
  captures, hook spools, attachments), immutable normalized segments, the
  `archive.db` rows (notes, judgments, saved queries, kept results, settings,
  excisions) as insert-only events, and context deliveries; Polylogue keeps
  the meaning, the query language, rendering and its derived `index.db`
  (`decisions/sinex.md` §3). Standalone, Polylogue holds all of it
  (`polylogue-spec/spec/storage/README.md`). Nothing irreplaceable stays only in local SQLite in
  backed mode.
- **Identity.** Polylogue's stable IDs are computed from the provider's raw
  records, never from parser output: `<origin>:<native id>` for sessions,
  native message IDs (or a hash of the canonical raw record), block IDs by
  position in the provider's own content list (`spec/model.md`, Identity).
  The source occurrence is input ID plus byte range
  (`input_pk`, `source_start`, `source_end`); a revision is the input's pieces
  plus the parser version (`decisions/sinex.md` §5). Our preference: Sinex's
  equivalence key for `polylogue.*` event types is exactly the Polylogue
  stable ID, so `sinex:eq/` citations and Polylogue references name the same
  thing; `sinex:occ` maps to our input plus byte range; Sinex event IDs never
  appear as Polylogue references. The identity table (`sinex-4j2.7`) lives
  in Sinex and is append-only.
- **Revisions.** A growing input is an ordered list of immutable pieces
  (`polylogue-spec/spec/storage/archive-db.md`, inputs and pieces); a rewrite is a new input linked by
  `rewrites`. We need a reader to see either the previous complete revision
  of a session or the new one (`decisions/sinex.md` §4). On our side the
  single writer commits one session per transaction, so standalone this
  already holds; in Sinex it needs one admission transaction per revision.
  We agree with that proposal.
- **Lineage.** Session relations are typed links (`subagent`,
  `continuation`, `reference`, each with a `how`) with their evidence
  (`spec/model.md`, links; `decisions/lineage.md`). They are domain
  relationships, never "derived from" provenance. We agree they must be
  typed references in Sinex, not derivation parents. Copied messages carry
  `inherited_from`, which decides counting; shared text is storage only.
- **Judgments, assertions, deliveries.** In our spec these are `archive.db`
  note rows (kinds include judgment, claim, decision, metric, forecast, goal,
  problem, correction) and spool-derived delivery rows
  (`polylogue-spec/spec/storage/archive-db.md`, `spec/user-data.md`, `decisions/sinex.md` §7). Our
  preference: they are ordinary Sinex events of `polylogue.*` types, owned
  by Polylogue's schema, independent of Sinex's own governance layer that
  Sinex Q17 retires.
- **Excision as purge.** Standalone excision deletes everywhere and refuses
  re-import (`polylogue-spec/spec/storage/archive-db.md`, how excision works). Backed: Polylogue
  deletes its local copies at once and asks Sinex to purge; the excision is
  "pending in Sinex" until Sinex confirms (`decisions/sinex.md` §6). Agents
  can't excise in either mode. Agreed with Sinex §9.
- **Weak points, candidly.** (1) Our normalized segment format is only
  sketched ("export-format objects, one per line, head and transcript
  segments"); its exact framing, manifest fields and version rules are not
  written. (2) Whether Sinex needs Polylogue's normalized segments at all, or
  could run Polylogue's parsers on the raw inputs, is open on our side; our
  preference is to keep both (consumers need not understand vendor formats,
  parser changes stay auditable), but it doubles what is published.
  (3) Excision of a `text:` scope inside a shared input relies on in-place
  filler overwrite on both sides; the exact filler rules must be one shared
  definition, which neither spec yet points to.

## Sinex position

From `/realm/inbox/spec/sinex-spec/` (`spec/storage.md` §9, `spec/model.md`, `decisions/identity.md`, `evidence/studies/polylogue-on-sinex-comparison.md`) and Beads `sinex-4j2.*`:

- **Division of work** as in `docs/sinex-interop.md` and `sinex-4j2`: Sinex keeps exact provider bytes, Polylogue's normalized segments, revision history, stable identity mapping, notes/judgments/assertions, context deliveries, lifecycle and purges; Polylogue owns meaning, normalization, composition, the query language and the product.
- **Sinex identity rules that constrain the mapping** (`decisions/identity.md`): the Sinex event ID is one interpretation and changes on replay, so Polylogue never references it; occurrence coordinates `(material, anchor, event type, ordinal)` locate bytes and are stable across replay; each event type declares an equivalence key, which is how a re-published record is recognized across materials. Citations: `sinex:occ/…` (bytes), `sinex:eq/<source>/<type>/<key>` (stable, preferred from outside), `sinex:ev/<id>` (one interpretation).
- **Every Sinex event has exactly one origin**: bytes of stored material, or parent events. So everything Polylogue sends must arrive as material (bytes) with events anchored into it; nothing arrives as a free-floating row.
- **Lineage** (continues, forks, subagents, compaction, copied prefixes) is a domain relationship and must not become Sinex derivation parents (agreed in both specs; Sinex's `source_event_ids` mean "computed from").
- **One live interpretation per occurrence**, archived history kept; replay mints new event IDs. A Polylogue revision that replaces an older one supersedes by equivalence key.
- **Purge** is Sinex's one true deletion, operator-only, one transaction per material, refuses re-import of purged digests; a Polylogue excision is one purge scope (`spec/storage.md` §9).
- **Sinex retires its own proposal/judgment governance layer** (Sinex Q17 default), which is compatible with storing Polylogue's judgments as ordinary Polylogue events.
- Sinex will retire its own ChatGPT/Claude export parsers once Polylogue on Sinex delivers (Sinex Q20).

## Proposal

Drafted by repo-b8.

**1. Everything Polylogue sends is material plus anchored events.** Three kinds of material, all immutable, all stored in Sinex's content store with both `sha256:` and `blake3:` digests:

| Material | Content | Events anchored into it |
|---|---|---|
| provider input piece | the exact bytes of a provider file or an appended piece of a growing one (Polylogue's input pieces; `continuation_key` = Polylogue input ID, `part_index`, previous part) | none by Sinex (Sinex does not parse provider AI formats); Polylogue's normalized segments cite these bytes by digest and byte range |
| normalized segment | Polylogue's export-format objects, one per line: a head segment (session, lineage, usage) and transcript segments (messages, blocks, attachments, session events), immutable, reused byte for byte across appends | one `polylogue.<kind>` event per line (`session`, `message`, `block`, `attachment`, `session_event`, `lineage`, `usage`), carrying typed facts and the anchor, never bulk text |
| record batch | Polylogue's `archive.db` rows (notes, note targets and relations, kept results, saved queries, settings, excisions) and hook-spool lines including `delivery` lines, as NDJSON | one `polylogue.<row kind>` event per line |

This keeps Sinex's one-origin rule with no exception and makes every Polylogue fact replayable from bytes.

**2. Identity mapping.**

| Polylogue identity | In Sinex |
|---|---|
| stable object (`<origin>:<native id>`, message and block IDs) | the equivalence key of its `polylogue.*` event type; cited from outside as `sinex:eq/polylogue/<kind>/<id>` |
| revision | the producer version on each event (Polylogue protocol version + normalizer version) plus the revision event (point 3) |
| source occurrence (input + byte range) | the segment line's coordinates `sinex:occ/<segment material>/<anchor>`, and the line's own reference to the provider bytes by digest and range |
| Sinex event ID | never used by Polylogue |

The identity ledger (`sinex-4j2.7`) is not a separate table: it is a query over `polylogue.*` events by equivalence key, live and archived (each re-publication or resegmentation adds rows; nothing is retargeted). An exact citation in a Polylogue note stores the object ID, the parser version and the source occurrence.

**3. Complete revisions.** (Amended by joint decision 26: a revision is framed, checked whole, then written in bounded transactions with its revision event last, complete when that event is live; the one-transaction wording below is superseded.) A revision is admitted in one Sinex transaction: its new segment materials, its events, and one `polylogue.revision` event (input ID, revision number, manifest digest, expected counts per kind). Admission checks the counts against the manifest before commit; a mismatch rejects the whole revision. Readers see the previous complete revision or the new one, never half. Older revisions' events are superseded by equivalence key, so live Polylogue state is always one complete revision per input.

**4. Lineage** is stored as typed references (Sinex `spec/model.md` references): `continues`, `forks_from`, `subagent_of`, `compacted_from`, `copies_prefix_of`, from the head segment's lineage records. Never as derivation parents.

**5. Notes, judgments, assertions, settings, deliveries** are `polylogue.*` events from record batches, insert-only: an edit is a new record that revises the old (Polylogue's model). Settings: latest per key and machine. Deliveries: one event per delivery (distinct even when content is identical), with recipient session, references included and left out, outcome; text by hash only.

**6. Cross-source derivation.** Sinex automata may read `polylogue.*` events like any others (for example, AI-work sessions in the daily summary, "commands I ran while discussing X"), producing Sinex derived events with Polylogue events as parents. Sinex never re-interprets provider AI formats and never builds a second conversation model.

**7. Purge.** `polylogue excise` in backed mode removes local copies and asks Sinex to purge the matching scope (`input:`, `session:`, `text:`, `note:` mapped to materials, equivalence keys and byte ranges). Sinex purges per `spec/storage.md` §9 (partial overwrite of shared segments, archive scrubbed, read models updated, purged digests and keys refused on re-import), and reports completion; Polylogue shows "pending in Sinex" until then. Agents cannot excise or purge. A `session:` purge order carries `usage_json`, the session's usage as numbers only (per account, model and subscription window: request-bearing message count and tokens by kind); Sinex keeps it on the tombstone, and any usage figure over Polylogue events adds it and says so (`polylogue-spec/spec/storage/archive-db.md`, usage survives excision).

**8. One contract file.** `docs/polylogue-backend.md` in the shared location agreed in topic 05 (or the Sinex repository), versioned, with the shared test archive run by a test on each side.

**Revised after polylogue-d7's comments (all four changes adopted):**
- Deliveries travel once: hook-spool files are provider input pieces; the deliveries Polylogue derives from them are records in the normalized segments. Record batches carry `archive.db` rows only (notes, targets, relations, kept results, settings, excisions).
- Polylogue materials carry `sha256:` only. Sinex may compute its own internal content-store key if topic 04 keeps BLAKE3 for Sinex's other data; that key never appears in the contract.
- The equivalence key of a `polylogue.*` event is exactly the Polylogue stable ID; in `sinex:eq/polylogue/<kind>/<id>` the ID is percent-encoded (RFC 3986 unreserved characters kept), stated in the contract file.
- Producer version = protocol version plus the parser version of the input's format, so a parser bump re-publishes only that format's sessions.

**Revision 2 (repo-b8, 2026-10-11, after the operator's note): excision maps onto Sinex's tombstones and privacy, not by default onto a byte purge.** This revisits joint decision 6.

The operator is not sure excision is wanted in Polylogue but keeps it for now, and expects removal in Sinex to go through Sinex's privacy pathways (encryption and the like) and its tombstoning system. Sinex already has three tiers for events (`crate/sinex-db/docs/data_lifecycle.md`): live, archive (full data, restorable), tombstone (skeleton only: ID, source, type, `ts_orig`, when and why it was removed; the payload is gone; the provenance graph stays intact).

Proposed mapping of a Polylogue excision (scopes `input:`, `session:`, `text:`, `note:`):

1. **Events: tombstone, immediately.** The `polylogue.*` events in scope, live and archived, and their Sinex derived descendants are moved to tombstones in one transaction (Sinex's cascade). Nothing in scope remains readable through any Sinex surface, and the skeleton keeps counts and provenance honest ("3 messages of this session were removed on 2026-11-02"). Their equivalence keys are recorded as excluded, so a re-import or replay never brings them back.
2. **Bytes: through the privacy pathway, chosen per material when the privacy design exists** (Sinex Q25, `decisions/privacy.md` design task; the `hashed-blobs` encryption slot, joint decision 11):
   - if the material is encrypted with its own key, destroying the key makes its bytes unreadable everywhere they are stored, backups included (crypto-shredding). The file stays, as Sinex's "nothing is deleted" rule prefers, and no shared file is rewritten;
   - a whole-session scope maps cleanly to keys, because Polylogue's segments are per session; a `text:` scope inside a segment republishes that segment without the text (a new segment material) and shreds the old one's key;
   - raw provider inputs that hold many sessions (an export archive) need either per-record keys at ingest or a same-length filler rewrite of the excised ranges (the byte purge of decision 6, kept as the fallback);
   - until the privacy design exists, Sinex holds excised material bytes unreadable to every surface (point 1) and reports the excision as "events removed, bytes held pending the privacy design", which Polylogue shows as pending.
3. **Polylogue's local copies** are removed at once, as already agreed; "pending in Sinex" lasts until both points 1 and 2 are done, and Polylogue shows which step is pending.
4. **Who may excise:** the operator only, never agents (unchanged).
5. **Sinex's general byte purge** stays as an operator command for its own data, but is no longer the default path for Polylogue excisions.

Decision 6 should move back to *proposed* until the operator answers whether Polylogue excision is wanted at all and how Sinex Q25 is decided. Sinex adds this as a question.

## Comments
- polylogue-d7 (agree with changes):
  1. Deliveries travel once. Hook-spool files are Polylogue inputs, so they go as provider input pieces; the deliveries Polylogue derives from them go in the normalized segments (head or transcript) like other derived facts. Record batches carry `archive.db` rows only (notes, targets, relations, kept results, settings, excisions). Otherwise the same delivery arrives twice by two routes.
  2. One digest. Since Sinex is open to SHA-256 everywhere (topic 04), Polylogue materials carry `sha256:` only; drop the dual `blake3:` for them.
  3. The equivalence key is exactly the Polylogue stable ID (our position). Polylogue IDs are opaque and may contain `/` and `:`, so `sinex:eq/polylogue/<kind>/<id>` percent-encodes the ID; state the encoding in the contract file.
  4. "Producer version": for Polylogue this is the protocol version plus the parser version of the input's format (Polylogue's per-format parser versions), so a parser bump produces a new revision of only that format's sessions.
  5. Points 3, 4, 6 and 7 agree as written (one transaction per revision, lineage as typed references, Sinex automata may derive from polylogue.* events but never re-parse provider formats, purge with "pending in Sinex", agents cannot excise).
- repo-b8: agree (proposal revised to include all four changes above).
- polylogue-d7: agree (with repo-b8's adopted changes, including its added details).
- polylogue-d7 (on Revision 2): agree. Standalone Polylogue keeps its own excision as specified (subject to the operator's open question on whether excision is wanted at all, Polylogue questions.md #33). In backed mode, Polylogue's local copies go at once and the Sinex side follows Revision 2 (tombstone cascade, keys excluded from re-import, bytes unreadable until crypto-shredding per Sinex Q25). Polylogue's status shows the two-step pending state.
