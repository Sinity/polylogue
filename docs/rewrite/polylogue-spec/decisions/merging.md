# Decision: the same conversation through several routes

Status: **decided**, following the operator's rejection of "latest wins" (2026-07-31).

A ChatGPT conversation can arrive through three exports taken months apart,
through live browser captures, and (rarely) a page-DOM fallback. A Claude Code
session is a live file and may also appear in hook events. A Codex rollout
can be rewritten in place, leaving two revisions. All routes for the same
`<origin>:<native id>` are one session.

**The rules are stated once, in `spec/sources/README.md`, "The same conversation
through several routes"** (union by message identity, a provisional-ID message
leaving it only when matched to a provider-ID message; the most faithful
route wins, then the more complete version; missing fields filled from the
other side; copies lose to originals; for a file rewritten in place, a
lossy rewrite (Codex) is merged with its earlier revisions and the older
revision wins on fields the rewrite damages, while an edit (AI Studio, Gemini
CLI, Hermes, Antigravity, Grok export) gives the session the newest
revision's messages only). This file records why.

## The tie-break

The tie-break is part of the rules in `spec/sources/README.md` (rule 2
across inputs, rule 5 between revisions of one input): content hash across
inputs, chain order between revisions, and never modification time, arrival
order or the input ID. A location input's ID is a UUIDv7 chosen when it was
first seen, so comparing IDs would be arrival order in disguise.

The chains of one input are one route, so route never separates them; rule 2
compares their versions by completeness first, and only on a tie does rule
5's chain order decide: the longer revision of a chain is later, and the
current chain is later than every older chain (copies, archive members,
migrated bytes, and a location's chains before its latest rewrite). Records
only an older chain holds are still added. A writer that keeps the first
version it sees meets this by writing an input's current chain before its
older ones; the result must not depend on that write order (law of order
independence).

## Why

The operator rejected "latest wins" ("maybe 'latest wins' was sloppy idea
altogether", 2026-07-31) and asked for differences to be resolved
decisively, recording a disagreement only when it is genuinely undecidable.
The route order follows how much each route preserves: the provider's own
API response and the live file carry every field; exports drop some
metadata and all tool structure (the July 2026 ChatGPT export drops tool and
system messages entirely); the DOM is text only.

The order is fixed (route, then completeness, then content) so that merging
does not depend on arrival order: the daemon can write each input as it
comes and still get the same result as a full rebuild (`spec/convergence.md`,
sessions built from several inputs; `plan/testing/laws.md`, order independence).

## Several routes are deliberate

Capturing the same conversation through several routes (a live file, hook
events, a browser capture, an export) is redundancy by design: each route is
a separate witness with its own provenance, and each is kept as an input.
Overlap is resolved by merging in derivation, by the rules above, never by
capturing less (operator ruling, 2026-10-11, stated for Sinex and applying
here alike).

Measured on the operator's data, no route is complete on its own: the July
2026 ChatGPT export dropped nodes the April export held; Codex rewrites
dropped reasoning, user messages and tool output that only older revisions
keep (`evidence/blobstore-v2.md`); and Sinex's fusion of the operator's zsh
and Atuin histories over 321 days found 6.6% of commands only in zsh, the
poorer source, while Atuin held repeats zsh drops on purpose
(`sinex-spec/prototype/fusion/`).
