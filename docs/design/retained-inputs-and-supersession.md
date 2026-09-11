# Retained inputs and safe supersession for convergence

Status: **decided** (polylogue-0qbdh). This document is the decision record
the seven dependent beads read before implementing: polylogue-ox0.1,
polylogue-cq1ql, polylogue-ximhz, polylogue-rovf5, polylogue-58jjk,
polylogue-2fr8s, polylogue-d5202.

Scope: what the archive retains for each observation of a source, and when
retained bytes may be retired. It authorizes no live blob deletion and no
wipe. It does not decide receipt volume (polylogue-gen6d) or the daemon's
derivation kernel (polylogue-bp12n.1).

Evidence head: `521888986`. Live-archive figures are read-only measurements
of `/realm/state/polylogue` taken 2026-09-11.

## 1. The question, and what the evidence answered

Two materially different things are retained the same way today. An
**append-only log** grows, so each observation of it is a byte prefix of the
next. A **mutable database member** changes, so successive observations are
unrelated byte images of one moving logical value. Both are stored as one
content-addressed blob per observation.

The bead asked whether that blobstore is excessive and should be simplified.
Measured, on the production live-append route, over one synthetic Codex
rollout grown to 1,096,453 bytes across 40 observations:

| configuration | retained distinct bytes | amplification | raw rows | appends accepted |
| --- | ---: | ---: | ---: | ---: |
| `append_warm` — continuity proof holds | 1,096,453 | **1.00×** | 41 | 40 / 40 |
| `full_only` — continuity unavailable | 22,499,957 | **20.52×** | 41 | 0 / 40 |
| `append_cold` — durable evidence only | 23,539,297 | **21.47×** | 79 | 1 / 40 |

The blobstore is **not** excessive. When the continuity proof holds, retained
bytes equal the source's own bytes exactly — one copy of each byte, no
duplicate, no prefix history. Content addressing is already carrying real
weight in production: 6,996 shared blob hashes save 28.10 GB on the live
archive. Every byte of excess is produced by **continuity lapsing**, which
forces a full snapshot, not by how a blob is represented.

**Decision.** Fix continuity; do not add a representation layer. Specifically,
no prefix-reference table, no reconstruction-on-read shim, no per-observation
history, no separate reindex engine. This is consistent with the operator's
close of polylogue-vzn6 ("do not promote `prune_once.py`, prefix-reference
storage, or receipt-based historical reconstruction into production").

## 2. Settled operator intent, and what this decides

**Settled before this bead** (not re-opened here):

- The fresh archive is rebuilt from actual sources; the current archive is not
  migrated forward (polylogue-reindex-2026.1).
- Prefix-reference storage and receipt-based historical reconstruction are
  rejected for production (polylogue-vzn6, closed).
- Ordinary daemon convergence is the only lifecycle; there is no repair
  product and no bulk reindex engine (polylogue-6kur, and the resolved item 6
  of `convergence-simplification-inventory.md`).
- Old snapshots are not retained to offer temporal queries.

**Decided here** — the six rules in §3, the scenario dispositions in §4, and
the retirement rule in §5.

**Deliberately left open** (named, not decided):

- The receipt-per-raw volume, measured below at 12.8 rows per raw. Owner:
  polylogue-gen6d.
- Whether the source-generation/item/attachment substrate becomes the general
  observation ledger for every origin, or stays an import-time structure.
  This decision needs only the scope and presence facts in §3, which that
  substrate can carry but does not uniquely own.

## 3. The chosen representation

### R1 — Retention follows the source's material law, not the capture route

Each declared source already states its law in `sources/origin_specs.py`. A
log member declares a `frontier_kind` (`exact-prefix`, or
`claude-header-body` for a mutable-header transcript); a database-shaped
origin declares a `DatabaseSourceCapability` with
`snapshot_method="logical_export"` and per-member `logical_tables`
(`origin_specs.py:770` `DatabaseMemberRule`, `:1786` for Codex). The retained
shape follows from that declaration:

- **Append-only log** — retain one baseline plus one delta per observation
  window. Owner: `sources/live/batch.py::_append_plan` →
  `sources/live/append_ingest.py::ingest_append_plans`.
- **Mutable database member** — retain one canonical logical export per
  *distinct logical revision of that member instance*. The export is
  complete for its declared scope and self-describing: it carries the
  member's tables, their DDL, their typed rows and an explicit `missing`
  list for declared tables the source did not have
  (`sources/sqlite_export.py:173`, header built at `:244`). Its blob hash is
  the member's logical revision.
- **Immutable artifact** (tool-result sidecar, export asset, memory
  document) — retain the bytes once, content-addressed, joined to its
  referent by a durable coordinate rather than by a live filesystem lookup.

### R2 — A full snapshot is the recovery path, not the steady state

A full capture is correct when, and only when, continuity cannot be proven.
It is not a cheap default: it costs 20.5× the content, measured. The design
target is therefore that the continuity proof does not lapse, and that a
lapse costs one snapshot rather than poisoning the rest of the session
(§6, D1).

### R3 — Scope is part of identity

The identity of an observed object is `(source scope, member, logical
coordinate)`. The source scope is the declared root or install the bytes came
from, not the basename and not a global "newest". Two roots are two scopes. A
value observed in scope X never supersedes the same-named object in scope Y,
in either acquisition order.

### R4 — Supersession replaces a value, never an object

A newer observation of scope X's member establishes the current value of
every object *present in it*. An object absent from that newer observation
becomes **absent-as-of-revision-R in scope X** — a presence fact about a
source — and stays archived and readable. Omission never cascades into
deletion of archived content.

Consequently a whole-table replace is not an acceptable projection
primitive. "The export states the complete thread set as of its revision, so
a thread Codex deleted must leave the projection too"
(`codex_state_projection.py:130`) confuses two facts: that Codex no longer
holds the thread, and that the archive no longer holds it. Only the first is
observed. The projection must express the first as presence state and keep
the archived object.

### R5 — Currency is decided by durable receipt order, never by wall clock or first sight

A live database that goes A → B → A re-mints A's content-derived raw id, so
`raw_sessions.acquired_at_ms` is the *first* time those bytes were seen, not
the latest. The durable receipt log is the authority: newest `raw_payload`
receipt first, ordered `(blob_refs.acquired_at_ms, blob_refs.rowid)`. This is
already implemented and documented at
`sources/codex_state_projection.py:56`; it is hereby the rule for every
latest-value projection, not a Codex-local trick.

### R6 — Disappearance is an observation, never a delete

A source root, file, or row that stops being observable records an absence
observation against its scope. Nothing archived is removed, no derived read
loses content, and no acquisition receipt is rewritten. A later reappearance
is an ordinary newer observation, not a resurrection path.

## 4. Scenario dispositions

"Today" is the observed behaviour at `521888986`. "Decided" is the required
outcome. Where they differ, §6 names the defect and its owner.

| # | Scenario | Decided retained result | Storage disposition | Today |
| --- | --- | --- | --- | --- |
| S1 | Strict-prefix log continuation | Baseline plus one delta per window; the chain replays to exactly the source bytes | 1.00× — one copy of each content byte | **Holds** when the cursor is live: measured 1.00×, 40/40 appends accepted |
| S2 | Divergent log (history rewrite) | The divergent observation becomes a new full baseline; the prior chain stays archived and readable as its own evidence | Both retained; the prior chain's bytes are **not** retirable — no retained successor contains them | Holds: `_classify_deduped_nodes` quarantines only the divergent suffix (`revision_authority.py:287`) |
| S3 | Truncated log | Truncation is not a prefix, so continuity is refused and a new baseline is captured | Both retained; prior bytes not retirable | Holds: `_record_append_cursor` refuses with "source replaced or truncated" (`batch.py:5249`) |
| S4 | A-B-A database values | A's second observation re-mints A's content hash and adds no blob; currency follows receipt order (R5) | Two blobs for three observations | Holds (`codex_state_projection.py:56`) |
| S5 | Row missing from a newer export | The object stays archived and readable; its presence in that scope becomes absent-as-of-R | No byte change | **Defect D2** — `write_thread_state_projection` deletes every projected row |
| S6 | Incomplete observation | Never an assertion of absence. A declared table the source lacked is carried in the export header's `missing` list; an item that could not be completed keeps a `pending`/`unknown_blocking` disposition and supersedes nothing | Retained, non-superseding | Partly: the export header records `missing` (`sqlite_export.py:228`); the projection does not consult it |
| S7 | Two source roots with disjoint objects | Two scopes, two current values; neither supersedes the other, in either acquisition order | Both retained | **Defect D2** — `latest_retained_state_export` picks one global newest across every `source_path` |
| S8 | Late or orphan sidecar | Joins by durable coordinate and reconverges; no duplicate message, and a still-missing sidecar stays an explicit outcome | One copy of the sidecar bytes | **Defect D3** — sidecar text is resolved from the original filesystem at parse time |
| S9 | Original source disappears after acquisition | Absence observation only; archived bytes and every derived read are unchanged | Nothing retired | Holds for acquired raw payloads; **fails** for anything only resolvable through a live sibling file (D3) |

## 5. When retained bytes may be retired

An immutable blob may be retired only when **all** of the following hold:

1. A retained successor covers the wanted content the blob uniquely
   evidences. What "covers" means follows the material law of R1:
   - **Log** — byte containment, proven by streamed byte comparison, not by
     size, path, timestamp or similarity
     (`archive/revision_authority.py::classify_historical_full_revision_streams`).
   - **Database member export** — logical containment within the same scope:
     every `(table, row identity)` the retired export uniquely evidences also
     appears in a retained successor export of that member instance, at the
     same value or a newer one. A newer export of a *shrinking* state does
     not cover the rows it dropped, so it never authorizes retiring its
     predecessor.
   - **Immutable artifact** — never retirable by supersession; it has no
     successor, only duplicates, which content addressing already collapses.
2. Every durable reference to the retired content still resolves through the
   ordinary production read route, with no reconstruction shim.
3. The provenance of the retired observation — that it was observed, when,
   and from which scope — remains readable.

Retirement is never motivated by age, by a retention window, or by a wish to
shrink a temporal-query surface. There is no retention policy in this design.

**No retirement mechanism is authorized or required by this decision.** Rule 1
is a constraint on any future retirement, not a request to build one. The
measurement in §7 is what would justify building one, and it does not: with
continuity intact, retained bytes already equal content bytes, so there is
nothing for a retirement pass to reclaim in a correctly-converged archive.

**Why this rule matters concretely.** The live archive is currently in the
state this rule exists to prevent: of 33,757 distinct raw payload hashes,
**31,998 (94.8%) have no physical blob**, covering 66.11 GB of logical
content. Those rows still claim their bytes; the bytes are gone. That is the
residue of the one-time 2026-08-24 prune (polylogue-vzn6) and is precisely
why the campaign rebuilds from actual sources instead of migrating this tier
forward. No future operation may leave the archive in that shape.

## 6. Named defects, with owners

### D1 — A continuity lapse poisons the rest of the session (no current owner)

`_resynthesize_cursor_from_source` recovers an append frontier from durable
`source.db` evidence when the disposable `ops.db` cursor is gone, but it
refuses when the accepted head is append-kind and was not reconstructed from
legacy offsetless rows (`sources/live/batch.py:4462`). Measured consequences,
`append_cold`: 21.47× amplification, 39 of 40 observations captured full, and
a residue of *deferred* append blobs that never join the accepted chain — the
`append_cold` run retains 79 rows for a 41-observation session, worse than
capturing every observation full.

The mechanism, probed directly over five ticks: once one full snapshot lands
beside an existing append, the newer full becomes `selected_baseline`, the
older full is `superseded`, and every append — the pre-existing one and each
newly planned one — is `deferred`. Successive plans re-issue from the same
durable baseline, so their windows overlap (observed: two appends both
starting at offset 1173), and `plan_revision_replay` can accept none of them.

This is not the case polylogue-yla8.9 closed. That bead authorized a full
snapshot to *fold* an accepted append chain when the two describe the same
bytes, and `_authorize_full_snapshot_fold`
(`storage/sqlite/archive_tiers/revision_governance.py:1716`) refuses unless
`full_size == frontier` — an **equal**-frontier snapshot. A recovery snapshot
is captured after the file has grown past the accepted append head, so it is
strictly longer, the fold does not apply, and the chain is stranded rather
than folded. The gap is the grown-frontier case, not the equal-frontier one.

Live corroboration: 1,900 append rows are `quarantined`, holding 6.65 GB —
more bytes than the 2,444 accepted append rows (3.94 GB).

Owners to name at implementation time:
`sources/live/batch.py::_resynthesize_cursor_from_source`,
`archive/revision_replay.py::plan_revision_replay`,
`sources/live/append_ingest.py::ingest_append_plans`. No existing bead owns
this; it is reported as a residual.

### D2 — Latest-value projection ignores source scope and deletes on omission

`sources/codex_state_projection.py:56` selects one newest state export across
every `source_path`, and `:130` replaces the whole projection (`:156`,
`DELETE FROM codex_thread_state`). Violates R3
and R4; breaks S5 and S7. Owner: **polylogue-ox0.1** (its AC1 and AC2 are
exactly these two scenarios). The same rules bind polylogue-58jjk for goals
and memories.

### D3 — Sidecar content is resolved from the live filesystem, not retained bytes

`sources/live/tool_result_sidecars.py` and
`sources/live/gemini_tool_output_sidecars.py` both state they perform no
acquisition-tier writes; the join reads sibling files under the original
session directory at parse time. The same retained transcript therefore
derives full output or only its truncated preview depending on whether the
original tree still exists. Violates R1's immutable-artifact rule and S9.
Owner: **polylogue-cq1ql**, with **polylogue-ximhz** for the Claude
index/history and ChatGPT export-asset equivalents and **polylogue-rovf5**
for harness memory documents.

### D4 — Declared coverage versus observed coverage

Whether a declared member's table set actually covers the wanted evidence is
a coverage question, not a retention question, and R1 does not answer it.
Owners: **polylogue-2fr8s** (Codex table classification) and
**polylogue-d5202** (schema observation for database and sidecar families).
Both may rely on R1's "the export is complete for its declared scope" and on
S6's rule that an incomplete observation never asserts absence.

### D5 — Receipt volume (named, not decided here)

The live source tier holds 550,458 receipt rows against 43,124 raw rows —
12.8 per raw. `raw_authority_census_plans` and
`raw_authority_census_post_plans` contribute 104,472 each. Owner:
**polylogue-gen6d**. This decision constrains it only by R6: compacting
receipts must not lose the provenance a retirement decision depends on.

## 7. Measurement record

### Live archive, read-only, 2026-09-11

`source.db` at `/realm/state/polylogue`, 43,124 `raw_sessions` rows,
100.14 GB of logical bytes, 33,757 distinct blob hashes / 72.05 GB.

| Measure | Value |
| --- | ---: |
| Distinct cohort bytes | 72.24 GB |
| Terminal-member bytes (the wanted content) | 44.64 GB |
| Superseded bytes, upper bound | 27.59 GB (38.2%) |
| Bytes already saved by content addressing | 28.10 GB across 6,996 shared hashes |
| Cohorts | 21,861, of which 4,308 have more than one observation |
| Chain lengths | 17,553 × 1; 4,091 × 2–4; 135 × 5–16; 38 × 17–64; 44 × 65+ |
| Superseded bytes by origin | codex-session 20.55 GB, claude-code-session 4.62 GB, chatgpt-export 1.66 GB |
| Append rows | 4,344 of 43,124 (10.1%): 2,444 accepted / 3.94 GB, 1,900 quarantined / 6.65 GB |
| Receipt rows | 550,458 total, 12.8 per raw |
| Physical blob store | 6,868 files / 6.11 GB |
| Raw payload hashes with no physical blob | 31,998 of 33,757 (94.8%), 66.11 GB |

### Bounded representation experiment

One synthetic Codex rollout, 40 observations, final size 1,096,453 bytes,
driven through `LiveBatchProcessor._append_plan` and `ingest_append_plans`
against a fresh temporary archive per configuration. Full results in §1.

`full_only` models an unavailable continuity proof (`ingest_cursor` wiped and
the processor rebuilt before every observation, durable resynthesis disabled);
`append_cold` is the same wipe with durable resynthesis left enabled, so
`source.db` evidence is the only continuity available; `append_warm` keeps the
ops.db cursor, which is the steady-state watcher. Because the two cold
configurations wipe the cursor before every observation, whether a full
capture writes a cursor afterwards cannot affect their totals — the
measurement isolates the continuity proof, not the cursor write.

A full capture is modeled at the archive write boundary
(`write_raw_payload` with a FULL envelope plus
`classify_raw_revision_cohort_for_live_watch`) rather than through
`_process_ingest_batch_sync`; the append configurations drive the production
route end to end.

Convergence work is unchanged by the representation: every configuration
replays 1,096,453 bytes, because replay parses the accepted chain's content
once regardless of how many superseded snapshots sit beside it. Storage, not
convergence work, is the axis the representation moves.

## 8. Production-route regressions required before rehearsal

Each must fail on the defect it pins, through ordinary acquisition and
convergence — not through a mocked join helper.

| # | Pins | Anti-vacuity condition | Home |
| --- | --- | --- | --- |
| V1 | A log observed across a continuity lapse retains bytes within a small constant of its final size | Reverting the append-head refusal, or disabling resynthesis, makes retained bytes exceed the bound | Extend `tests/unit/sources/test_live_append_cursor_resynthesis.py`; amplification pin beside `tests/integration/test_live_read_amplification.py` |
| V2 | No cohort accumulates append rows that can never join an accepted chain | Re-issuing a second append from an already-consumed baseline leaves a permanently deferred row | `tests/unit/sources/test_live_deferred_append_dedup.py` |
| V3 | Two source roots with disjoint objects both survive convergence in either order | Restoring the global-newest export selection makes one root's objects unreadable | polylogue-ox0.1's own selection |
| V4 | An object omitted from a newer export of the same scope stays readable | Restoring the whole-table replace makes it disappear | polylogue-ox0.1 |
| V5 | Full tool text, outcome and ownership reproduce with the original sidecar tree removed | Deleting the retained sidecar bytes, not the original path, is what turns it red | polylogue-cq1ql's named selectors |
| V6 | An incomplete observation supersedes nothing | Treating a `missing` declared table as an empty table makes a retained object disappear | polylogue-2fr8s / polylogue-d5202 |

## 9. Anti-goals

- No general event-sourcing system, and no per-update database history.
- No prefix-reference blob table, no reconstruction-on-read shim, and no
  revival of `prune_once.py`.
- No separate reindex engine; ordinary daemon convergence is the only
  lifecycle.
- No retention window, and no old-snapshot retention motivated by temporal
  queries.
- No live blob deletion or wipe is authorized by this decision.
