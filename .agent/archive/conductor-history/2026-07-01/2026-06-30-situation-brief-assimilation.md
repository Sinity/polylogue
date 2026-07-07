# Situation Brief Assimilation

Source files read from `/realm/inbox/download`:

- `situation-brief (2).md`
- `2026-06-30_05-37-03_Claude_Chat_Document_review_and_attachment_assessment_-_Claude_-_https.md`
- `demo-spec-agent-recovery.md`
- `demo-spec-claim-vs-evidence.md`
- `methodology-post-scaffold.md`

## Operating Update

The brief changes priority more than architecture. The devloop's current
substrate-cleanup path is useful only when it directly helps a finished,
inspectable demonstration. Section 5.4 is the binding rule: finish one uplift or
finding artifact through the last mile, then stop instead of generalizing.

Correction from the operator: do not overcorrect this into "demo value instead
of substrate." Hard-to-fake value and substrate integrity have to be combined or
interleaved. A demo built on broken substrate is not honest; a substrate repair
that makes current or near-term artifacts truthful is not deferral. The
anti-pattern is getting lost in scaffold or generic substrate polish with no
pressure from a concrete artifact.

The adjacent 05:37 review log reinforces the same constraint in a different
register: the strongest missing piece was not more capability, but a compact
Polylogue-generated report over real agent logs. It also contains stale archive
figures from before dedupe/convergence work; the current local fact is the
active v18 archive at `/home/sinity/.local/share/polylogue` with 13,208 indexed
sessions and 3,833,656 messages. Do not reintroduce the older 16K-session frame
when writing demos or process notes.

## Demo Priority

1. Claim-versus-evidence report.
   Lowest-friction while the archive is already live: read-only over
   `index.db`, uses the structured outcome keystone, and can produce a
   publishable finding without a browser/control-arm setup.
2. Agent session self-recovery.
   Highest compounding value and best uplift demo, but requires a careful
   two-arm run and cross-stack pack assembly. Do not turn this into a recovery
   subsystem before one instance ships.

## Process Guard

Do not let scaffold work become the deliverable. Process improvements need an
executable consequence, and demo work needs a stop rule. The next loop should
prefer artifact completion over additional cleanup unless the cleanup is needed
to keep the artifact truthful.

Meta work remains legitimate when it prevents drift or catches stale process
state, but it should leave an executable check, a corrected packet, or a sharper
next slice. It should not become the visible proof in place of a live-archive
artifact.

## Current Application

Extend `scripts/agent_forensics.py` rather than create a new silo. The report
should add a structured failure follow-up section anchored on `actions.is_error`
and `actions.exit_code`, with refs for drill-down and caveats about heuristic
acknowledgment classification.

This has now shipped as the first instance:
`/realm/inbox/demos_polylogue/04-claim-vs-evidence/` contains `README.md`,
`report.md`, charts, and `structured_failure_followups.json`. The core headline
is occurrence-anchored: 42,046 failed structured outcomes and 11,713 immediate
next-turn `silent_proceed` classifications by the stated heuristic. Treat this
as a completed externalizable finding unless the operator asks for one specific
publication polish pass.
