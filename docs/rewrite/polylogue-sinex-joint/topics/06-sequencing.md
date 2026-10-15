# 06-sequencing

Current amendment (2026-10-15): the successor supports `off` and `primary`; the mirror-first agreement below is historical. Finite fixed-basis shadow qualification and the same reconstruction/removal/real-corpus proof precede native cutover. The [current backing decision](../../polylogue-spec/decisions/sinex.md) owns that outcome; no predecessor runtime mode is changed.


## Polylogue position

From `/realm/inbox/spec/polylogue-spec/plan/slices.md`.

- **Polylogue's order.** Slices 1–4 give daily use and retire the Python
  version: Claude Code end to end (plus storing every raw input the old
  archive kept), the live daemon with Codex and hook briefings, the full
  query language and facts, web chats and migration. Slices 5–12 add the
  remaining providers, lineage and context, notes, schema inference,
  semantic search, the web reader, agent analytics and the workbench.
  Slice 13 is "On Sinex": `mirror` first, then the identity bridge and
  corpus backfill, then rebuild-from-Sinex proved by law 38 on real data,
  then `primary` only if that law holds and the operator wants it
  (`questions.md` #25), then deliveries, purge and ambient context. It
  "may move earlier".
- **Nothing in slices 1–4 waits on Sinex.** Durable access already goes
  through one module (`store::archive`), IDs are raw-derived and stable,
  every `archive.db` row has a global key, and there is a change feed and a
  byte anchor per message (`decisions/sinex.md`, consequences).
- **Our preference.** Keep Polylogue slices 1–4 independent of Sinex; agree
  the contract file and the shared test archive (Sinex §9, "keeping the
  contract in one place") as soon as both slice-1s exist, so both sides test
  against it early; move `mirror` up to right after Polylogue slice 4 if
  Sinex slices 1–2 are accepted by then, since `mirror` cannot lose data
  and starts the backfill clock. The decisive proof (topic 02) then gates
  `primary`.
- **Shared crates timing.** If the content-store and model crates are
  shared (topics 04, 05), they must exist before Polylogue slice 1's
  workers start, or Polylogue builds them first in its own workspace with
  no Polylogue dependency and Sinex adopts them later. Our preference is the
  second: Polylogue's slice 1 needs `polylogue-blobs` immediately, and Sinex
  can adopt a working crate rather than co-design one under a deadline.
- **Weak point.** Slice 13 is listed last of thirteen; if Polylogue's later
  slices run long, backed mode slips for months. Moving `mirror` earlier
  (above) is the remedy, but it is not yet written into `plan/slices.md`.

## Sinex position

- Sinex plan (`/realm/inbox/spec/sinex-spec/plan/slices.md`): new workspace skeleton; slice 1 shell and browser history (milestone: the operator searches real shell history within two weeks); slice 2 desktop context; slice 3 the Sinnix lake; slice 4 bulk exports; slice 5 live capture; slice 6 Polylogue on Sinex, starting only after standalone Polylogue is in daily use and Sinex slices 1-2 are accepted.
- Sinex slices 1-5 do not depend on Polylogue. Sinex's open blockers are its own operator questions (Q1, Q1b, Q5b, Q31, Q33).

## Proposal

Drafted by repo-b8.

**Principle:** neither project waits on the other except at one meeting point, and shared crates are built before either side would otherwise write its own version.

| Step | Polylogue | Sinex | Shared |
|---|---|---|---|
| 0 | spec settled | spec settled; operator answers Q1, Q1b, Q5b, Q31, Q33 | topics 04 and 05 decided; shared-crates repository created |
| 1 | slice 1 (Claude Code end to end) uses `polylogue-blobs` from the start | workspace skeleton + slice 1 use the shared blob crate as the content-store file layer | `polylogue-blobs` (specified, ~750 lines) built first, by whichever session starts first (test kits are shared as method only, per decision 9) |
| 2 | slices 2-4 standalone | slices 1-2 on real data | the segment codec crate, built when Polylogue's export format is stable (Polylogue slice 1 defines it) |
| 3 | standalone Polylogue in daily use; `input cat` (slice 1) and `changes` (slice 7) exist | slices 3-5 | contract file `polylogue-backend.md` drafted from topic 01; the shared test archive |
| 4 | slice 13 "On Sinex": moves to right after Polylogue is in daily use, not after slice 12 | slice 6 "Polylogue on Sinex" | both sides' contract tests green; decisive proof run (topic 02) |

- **Meeting point:** Polylogue slice 13 and Sinex slice 6 are the same work, done together once both sides are in daily use. Earlier than that, the transport (topic 03) has nothing real to carry.
- **If one side is late:** Polylogue keeps running standalone (no loss); Sinex keeps capturing everything else. Neither blocks.
- **Backfill** of the existing Polylogue corpus (`sinex-4j2.6`) comes after the decisive proof passes on the test archive and on a sample of real sessions.

**Revised after polylogue-d7's comments (adopted):**
- `polylogue-blobs` is built in Polylogue's slice 1, in the Polylogue repository; Sinex pins a tag when its new workspace starts. If Sinex's skeleton starts before the tag exists, Sinex codes against the crate's specified interface and pins once tagged; it does not write its own file layer.
- The segment codec is the export format, in `polylogue-model`; no separate crate.
- The meeting point is right after Polylogue's slice 4 (Polylogue in daily use) and Sinex's slices 1-2. `mirror` comes before `primary`; `primary` is allowed only after the real-corpus proof (topic 02) passes.
- Backfill after the decisive proof.

## Comments
- polylogue-d7 (agree with changes):
  1. `polylogue-blobs` is built in Polylogue's slice 1 (it is in our slice-1 plan already), inside the Polylogue repository; Sinex adopts it by pinning a tag when its new workspace starts. This reconciles your step 1 with our topic 05 rule that a crate is shared once both sides use it.
  2. The segment codec is the Polylogue export format, so it lives in `polylogue-model`; no separate crate.
  3. The meeting point is agreed: Polylogue's "On Sinex" slice moves to right after Polylogue is in daily use (after its slice 4), and `mirror` comes before `primary`; `primary` is allowed only after the real-corpus rebuild proof in topic 02 passes.
  4. Backfill after the decisive proof: agreed.
- repo-b8: agree (proposal revised to include all four changes above).
- polylogue-d7: agree (with repo-b8's adopted changes, including its added details).
- repo-b8 (correction, 2026-10-11): the operator chose to morph Sinex's current workspace (no new workspace, no rebuild branch); this does not change the sequencing above. "Test kit" in step 1 was a slip: per decision 9, test kits are shared as method only; Sinex's harness stays xtask's.
