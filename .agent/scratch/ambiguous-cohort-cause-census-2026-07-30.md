---
created: 2026-07-30
purpose: >
  Determine, per origin, whether the polylogue-bu1i root cause (acquisition
  state contaminating attachment identity hash) generalizes beyond
  aistudio-drive, or whether other origins' equal-message-count "ambiguous"
  membership cohorts have different causes.
status: complete (initial census; three origins root-caused, four left as
  residue -- see polylogue-inoh)
project: polylogue
---

# Ambiguous membership cohort cause census

Read-only investigation against the live archive (`/realm/db/polylogue`,
opened `mode=ro`; never written). No conversation content, titles, or
message text appears below -- structural/aggregate fields only.

## Method

For each origin, joined `raw_session_memberships` (`decision='ambiguous'`) to
`raw_sessions` on `raw_id`, grouped by `logical_source_key` into cohorts, and
kept cohorts where every member row has the same `message_count` (the
population this investigation was scoped to -- pre-measured by the parent
bead, polylogue-bu1i). Within each cohort, deduplicated members by
`blob_hash` (byte-identical raws parse identically, so this collapses
acquisition-time re-scrapes for free) and parsed each distinct blob with the
PRODUCTION parser path: `polylogue.sources.dispatch.parse_payload` /
`parse_stream_payload`, routed exactly as
`polylogue.sources.revision_backfill._parse_one` routes it (including
`is_stream_record_provider` for Claude Code JSONL and
`provider_from_origin(origin, family_hint=capture_mode)` for provider
selection). Projected each parsed session with
`polylogue.pipeline.ids.session_revision_projection` and ran the production
`polylogue.archive.session_revision_membership.classify_membership_revisions`
end to end, plus targeted pairwise structural diffs (message id set/order
equality, per-id content equality, attachment key set equality, event hash
equality) to isolate the minimal delta.

Random sampling used a fixed seed (`20260730`) for reproducibility. Sample
sizes were chosen per origin's population size, not uniform.

## Population (equal-message-count ambiguous cohorts, full archive)

| origin | population | sampled | coverage |
| --- | --- | --- | --- |
| claude-ai-export | 566 | 40 | 7% |
| chatgpt-export | 129 | 35 | 27% |
| aistudio-drive | 151 | 30 (sanity re-check; 151/151 already proven by polylogue-bu1i) | 20% (100% via bu1i) |
| claude-code-session | 6 | 6 | 100% |
| hermes-session | 3 | 3 | 100% |
| grok-export | 1 | 1 | 100% |
| unknown-export | 2 | 2 | 100% |
| gemini-cli-session | 0 | -- | n/a (0 of 3 total ambiguous cohorts have equal message counts; out of this investigation's scope by construction) |

## Findings by origin

### aistudio-drive -- CONFIRMED same cause as polylogue-bu1i (sanity check)

30/30 sampled cohorts reproduce polylogue-bu1i's proven cause exactly:
attachment acquisition state (`inline_bytes`/`size_bytes` flipping from
unresolved to fetched) folds into `_attachment_hash_payload` identity,
producing disjoint equal-cardinality attachment hash sets and an ambiguous
verdict for a non-branch. polylogue-bu1i already established this at
100% (157/157 two-member cohorts); this sample corroborates it and adds no
new information. **This origin's full population is proven, not extrapolated.**

### claude-ai-export -- TWO NEW distinct causes, neither the bu1i cause

**21 of 40 sampled (52.5%) -- message order nondeterminism (polylogue-c429).**
Same message id SET, same per-message content (role/text/timestamp
byte-identical), but a DIFFERENT array order between the two Claude.ai
export vintages. `session_revision_projection.message_hashes` is an
order-sensitive tuple and `_strictly_dominates` requires an exact positional
prefix match, so identical content in a different sequence reads as a
branch. One reproduced example: 36/36 messages, 0 content diffs among all
36 shared ids, but the two revisions' `provider_message_id` sequences
differ entirely (one export appears chronologically sorted; the other
interleaves what look like edited-message sibling pairs).

**16 of 40 sampled (40%) -- synthetic attachment id instability
(polylogue-hith).** Same message content/order, same attachment COUNT, but
a disjoint attachment identity key set. Root cause traced to
`polylogue/sources/parsers/base_support.py:_make_attachment_id`, which
synthesizes `att-{hash(message_id:name:index)}` whenever the raw export
lacks a real attachment id -- and both the real-id presence and the
positional `index` vary across export vintages for the same physical
attachment.

3/40 sampled cohorts show a compound of one of the above plus a secondary
delta (attachment count difference or event-hash difference) not
separately root-caused. 0/40 were unclassifiable or genuinely divergent by
this census's criteria, though the 7% sample fraction means this is an
ESTIMATE (roughly 280-300 + 220-230 cohorts extrapolated across the two
causes, out of 566), not a census, and the two causes are not mutually
exclusive within a cohort.

### chatgpt-export -- ONE dominant NEW cause, distinct from claude-ai-export's

**33 of 35 sampled (94%) -- provider-reported generation-duration volatility
(polylogue-nuec).** Message and attachment content hashes are IDENTICAL
between the two export vintages in every one of these 33 cohorts; only
`session_events` differs, and in every sampled case the first differing
event is a `generation_lifecycle` entry whose `elapsed_duration_ms` value
(derived by the parser from the raw export's own
`finished_duration_sec`/`reasoning_start_time`/`reasoning_end_time` fields)
differs between exports with no consistent direction -- ruling out simple
clock skew. This is the cleanest finding in the census: the field is
explicitly labeled `duration_semantics: "provider_reported_elapsed"` (a
measurement, not conversation content) by the parser itself, yet it is
hashed into content identity.

1/35 sampled cohort combines this with a message-hash difference
(unexamined, out of scope for this pass). 1/35 sampled cohort now reads as
fully identical under the current classifier (message/event/attachment
hashes all equal) -- its recorded `ambiguous` decision appears stale;
see polylogue-9dxn (persisted-verdicts-never-re-derived, filed earlier the
same day) for the general mechanism that would explain this without a new
root cause.

27% sample fraction (the largest of any origin here); extrapolating 94%
across the 129-cohort population suggests roughly 120 cohorts share this
cause, but remains an estimate, not a census.

### Residue: claude-code-session, hermes-session, grok-export, unknown-export

Small populations (6, 3, 1, 2 respectively out of the equal-message-count
ambiguous set). Sampled to full coverage but NOT root-caused to the same
depth as the three origins above -- tracked as polylogue-inoh rather than
silently dropped. Headline observations, each caveated in the bead:

- claude-code-session: 5/6 and unknown-export: 2/2 sampled cohorts collapse
  to a SINGLE distinct blob-hash content group on re-parse (the recorded
  members are byte-identical to each other), which should not classify
  ambiguous under current logic at all -- suggests stale decisions (see
  polylogue-9dxn) or a raw sibling removed since the decision was recorded,
  neither confirmed here.
- hermes-session: 2/3 sampled cohorts are single-message conversations with
  identical message id/order/attachment keys; the actual content delta
  wasn't isolated at this population size. 1/3 hit a census-harness gap
  (hermes's SQLite-backed raw path wasn't handled by the generic parser
  call used here) -- a tooling gap, not evidence of anything.
- grok-export (full population, n=1): the one ambiguous cohort is a
  browser-capture DOM snapshot with genuinely different message id sets at
  equal count -- tentatively GENUINE ambiguity (a real captured-state
  divergence), not a misclassification, though n=1 is too small to
  generalize.

## Bottom line

- **aistudio-drive (151 cohorts): CONFIRMED same cause as polylogue-bu1i.**
  A fix there resolves this population's full 151.
- **claude-ai-export (566 cohorts): NEW causes, NOT the bu1i cause.** Two
  distinct defects (message-order nondeterminism ~53% sampled,
  synthetic-attachment-id instability ~40% sampled) together plausibly
  explain most of the population, but neither is the acquisition-state
  pattern bu1i already fixed -- a fix limited to bu1i's scope resolves
  approximately 0 of this population.
- **chatgpt-export (129 cohorts): NEW cause, NOT the bu1i cause.** A single
  dominant defect (provider-reported generation-duration volatility, ~94%
  sampled) that is architecturally simpler to fix than either
  claude-ai-export cause (exclude one volatile field from a hash, rather
  than relaxing a dominance comparison) but is its own separate defect.
- **claude-code-session (191 total, 6 in the equal-count population),
  hermes-session (4), grok-export (1), unknown-export (3): residue, not
  root-caused to the same standard.** Tracked in polylogue-inoh. At least
  one population (grok-export) appears to be genuinely ambiguous content,
  not misclassification.

## Beads filed

- polylogue-c429 -- claude-ai-export message-order nondeterminism
- polylogue-hith -- claude-ai-export synthetic attachment-id instability
- polylogue-nuec -- chatgpt-export generation-duration volatility in
  session-event identity
- polylogue-inoh -- residue tracking for the four small-population origins

All four `Ref polylogue-bu1i`. Full reproduction recipes (runnable against
the live read-only archive, production code only) are in each bead's
description.
