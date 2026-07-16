# Polylogue Sol/Pro launch-dispatch handoffs (2026-07-15/16)

## What this is

28 `polylogue-sol-pro-launch-handoff*.zip` deliverables, manually downloaded
by the operator from ChatGPT Sol/Pro sessions between 2026-07-15 23:08 and
2026-07-16 07:02, deduplicated from 45 raw browser downloads by content hash
(17 were exact-duplicate re-downloads). Moved here from
`/realm/inbox/download/` — that directory is meant to be drained staging, not
permanent storage, and these are otherwise irreplaceable (see "Why these
aren't backups" below).

**Correction to an earlier assumption**: these were first read as 28
successive iterations of one mission (file sizes shrink 8.4MB→tens-of-KB
across the window at an identical filename). Checking `MANIFEST.json` inside
each zip disproved that — where a `mission` field is present, each one names a
**distinct** bead/topic (`polylogue-yyvg.5`, `polylogue-z9gh.1`,
`polylogue-ovme.1`, "authorize and receipt every destructive operation...",
"expose named-source freshness...", etc.). This looks like a coordinated
multi-mission dispatch batch (matches the `feature/browser/sol-pro-dispatch`
branch name — dispatching several Sol/Pro missions, not iterating one), not
one mission repeated. About half the manifests use a different schema
(`prompt_profile`/`base_revision`/`files` keys, no readable `mission` field)
— their actual mission is still unidentified; see `raw/` + `SHA256SUMS` and
open the zip's `MANIFEST.json`/`README.md` directly to identify those.

## Why these aren't backups — a real capture gap

Checked the live archive directly (not just message text): polylogue captured
all 28 source ChatGPT sessions in this window, and captured 29 attachments —
but every single one is a `polylogue-sol-pro-context-*.tar.gz`, the *input*
bundle the operator uploaded to seed each session. **Zero** attachments named
`*launch-handoff*` exist anywhere in `attachments`/`attachment_refs`.
Browser-capture acquires what gets uploaded *to* ChatGPT but has no path for
what ChatGPT offers back as a downloadable generated file. Filed as
`polylogue-s2x7` (P1 bug) — these 28 zips are named there as the real-world
regression fixture once one is unzipped into a browser-capture DOM fixture.

**Until `polylogue-s2x7` is fixed, this directory is the only copy of this
category of Sol/Pro output that will ever exist** — future runs will have the
same gap unless the capture-side fix lands first.

## Status: NOT adjudicated — do not apply patches wholesale

Following the precedent already set for a sibling provider in this repo
(`/realm/inbox/handoffs/polylogue-gemini-2026-07-16/README.md`, whose own
process-correction note reads: *"the earlier execution-oriented ZIP handoffs
were retired after AI Studio produced speculative, duplicated, truncated, and
artifact-generation-contaminated diffs"*) — patches inside these zips have
**not** been reviewed. Several contain applyable `PATCHES/*.patch` series;
none should be merged without the same review gate that caught the AI Studio
failures (see `/realm/inbox/handoffs/polylogue-gemini-2026-07-16/rejected-attempts/`
for what an unreviewed chat-UI deliverable looks like when it goes wrong).

## Contents

- `raw/<timestamp>-<hash8>.zip` — the 28 deduplicated originals, renamed by
  download timestamp + first 8 hex chars of their SHA-256 for stable,
  collision-free naming.
- `SHA256SUMS` — full hashes for integrity verification.

## Session correlation (chatgpt-export origin, live archive)

28 sessions captured 2026-07-15 21:34 → 2026-07-16 08:30. Named sessions
(most are titled "New chat" — the substantive ones):

| Time (local) | Session ID | Title |
| --- | --- | --- |
| 07-15 23:18 | `6a57f545` | Polylogue project progress |
| 07-16 00:36 | `6a58079b` | Polylogue Agent Query Fix |
| 07-16 01:19 | `6a580996` | Extension Receiver Pairing Stability |
| 07-16 02:24 | `6a582014` | Destructive Operation Authorization |
| 07-16 02:30 | `6a5823b5` | Mission Expose Source Freshness |
| 07-16 08:30 | `6a587a8c` | Live Action Proof Request |

Zip-download timestamps cluster in three bursts (23:08; 02:44–02:46;
03:55–03:56; 06:12–06:14; 07:02) rather than spreading evenly across the
8-hour window — consistent with the operator batch-downloading several
session deliverables back-to-back rather than one-per-session-as-it-finished.
Exact zip↔session pairing was not attempted here (would need the archived
session transcript's own reference to its `MANIFEST.json`/mission text,
cross-checked per zip) — the mission-name extraction above is the more
reliable correlation signal.

## Next steps (not done here)

1. Extract and read each zip's `MANIFEST.json`/`README.md`/`SUMMARY.md` to
   identify the ~14 zips whose mission wasn't recoverable from a single
   manifest-field grep.
2. Adjudicate each mission's design + patches against current master before
   considering any merge — treat exactly like the Gemini lane's
   analysis-before-execution discipline.
3. Fix `polylogue-s2x7` (browser-capture output-attachment gap) so future
   Sol/Pro runs don't require this manual-download workaround.
