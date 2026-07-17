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
successive iterations of one mission because the browser reused one download
filename. Manifest, transcript, patch-digest, and Bead correlation instead
identified a coordinated multi-mission dispatch batch. All 28 are now
identified in `INTEGRATION-LEDGER.json`, including manifests that use the
alternate `prompt_profile`/`base_revision`/`files` schema.

## Why these aren't backups — root cause is identified and already fixed upstream, not yet deployed

Checked the live archive directly: polylogue captured all 28 source ChatGPT
sessions and 29 *input* attachments (`polylogue-sol-pro-context-*.tar.gz`,
the bundles uploaded to seed each session), but **zero** `*launch-handoff*`
output attachments anywhere in `attachments`/`attachment_refs`.

This is tracked in full by **`polylogue-3v1`** ("Capture extension
reliability + status UX"), which already had a much deeper, evidence-rich
incident writeup than what I filed independently before finding it (filed
`polylogue-s2x7`, then closed as a duplicate once `3v1` turned up). `3v1`'s
own 2026-07-16 incident note: *"audited all 27 Sol Pro campaign
conversations. Current extension files parse to 2,551 messages while the
index exposed 205; 24/27 session projections mismatched, 22/27 ingest
cursors were permanently excluded after five transient failures... Branch
`feature/fix/browser-capture-replacement-reingest` now revives only changed
excluded observations and orders compatible browser snapshots by fidelity,
provider timestamp, stable message/attachment identities, and acquisition
time for attachment-only enrichment."*

**The fix is merged** — `d2573d438` `fix(capture): recover replaced browser
snapshots (#2930)`, 18 commits ahead of the branch this was being worked
from (hadn't fetched/rebased when the earlier, narrower diagnosis was made).
**It is not yet deployed against the live archive** — `3v1`'s own notes:
*"Live archive replay/deployed parity remains required before claiming
closure."* Confirmed independently: `polylogued.service` is currently
FAILED (SIGKILLed ~11:32, same day) — nothing is being reprocessed right
now.

**Operator caveat (2026-07-16)**: the whole GPT-Pro handoff/launch system is
being actively rewritten on this branch — re-check live source/bead state
before trusting this as current fact.

**Bottom line, unchanged**: this directory is — right now — the only copy of
this category of Sol/Pro output. Whether the merged fix, once deployed and
replayed, recovers these 28 from the underlying raw captures (plausible,
given the fix targets exactly this scenario) or they stay permanently lost
is what `polylogue-3v1`'s "live archive replay" step will determine — follow
that bead, not this one.

## Status: fully adjudicated

All 28 canonical ZIPs and all 325 older-corpus work units have terminal
processing decisions. `INTEGRATION-LEDGER.json` is the machine-readable
canonical-package ledger; `OLDER-INTEGRATION-LEDGER.json` records the older
handoff corpora. Both deliberately separate package processing from downstream
product delivery:

- `processing_state: processed` means the package was identified, read,
  adjudicated against current source, and routed durably.
- `disposition` records whether it merged, was incorporated as research or a
  pending delivery input, was subsumed/superseded, or was rejected.
- `delivery_state`, `delivery_constraint_kind`, and
  `next_delivery_prerequisite` explain why incorporated work that has not yet
  merged is waiting and identify its next owner.

The canonical result is 28/28 processed and zero unknown: four merged, four
research-incorporated, twelve incorporated with an explicit downstream
delivery prerequisite, two already subsumed, four superseded alternatives,
and two rejected. Every one of the twelve pending-delivery packages has its
full SHA recorded in its owning Bead. Nine await ordinary dependency delivery;
one awaits deployment, one is a verified CaptureJob draft preserved on
`feature/integration/capture-job-authority`, and one is an incorporated
MutationTransaction design awaiting route-census admission. “Awaiting” here
does not mean the handoff itself remains unprocessed.

Do not apply raw patches wholesale. The ledgers are the authority for whether
a patch was merged, preserved for reconciliation, incorporated as design
evidence, superseded, or rejected.

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
Exact package↔session correlation is recorded per row in
`INTEGRATION-LEDGER.json`; it combines transcript references with package
mission and manifest evidence rather than relying on download order.

## Remaining product delivery

Corpus processing is complete. Remaining work belongs to the Beads named by
the ledgers, not to a second handoff-intake pass. In particular, the preserved
CaptureJob draft must be reconciled only after provider adapters expose stable,
non-secret account handles; the other pending packages enter their owning
Beads when those Beads' declared dependencies or admission gates are met.

Capture reliability and live replay remain product work under the current
browser-capture Beads. Successful corpus adjudication does not itself prove
that future assistant-produced files will be captured without manual download.
