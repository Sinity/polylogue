Title: "Public claims view: one rendered projection of findings, judgments, and evidence ancestry that cannot overstate support"

Result ZIP: `ann-05-claims-view-r01.zip`

## Mission

Implement bead `polylogue-3tl.16` (read its full record): public claims —
the numbers/behavioral statements that README, demos, findings pages, and
launch material publish — become a RENDERED VIEW over
`AssertionKind.FINDING` assertions, their evidence ancestry, judgments,
supersession, publication/privacy review, and evaluation/frame receipts.
There is NO second durable claims ledger; `docs/public-claims.yaml` (if
present) is a generated/import compatibility view only.

Why this matters right now: the operator is about to publish
claim-vs-evidence numbers externally. Every published number must be
traceable to a finding with live evidence status, and the rendering must
degrade honestly when evidence goes stale (e.g. after re-ingest changes
counts) — "supported (as of epoch E)" vs a silent stale number.

Deliver:

1. **Status projection**: stable public claim keys with status ∈
   {supported, partially_supported, not_supported, stale/needs-rerun,
   held_private, unknown, capability-only}, computed from finding
   assertions + judgments + supersession + the shared evidence-integrity
   verdict (bead `37t.14` owns integrity semantics — consume, don't
   re-walk ancestry or define circularity/staleness locally; if 37t.14's
   verdict surface doesn't exist yet in the snapshot, define the narrow
   interface you need and state it as a dependency).
2. **Presets**: README, launch, findings-page, and verified-export presets
   as parameterizations of ONE projection (bead AC: no per-surface
   reimplementation). Output: markdown + JSON renderings with per-claim
   status badges, evidence refs, and epoch/frame qualifiers.
3. **Blocking honesty**: broken, circular, stale, frame-incomplete,
   private-held, and unsupported evidence produce DISTINCT statuses and
   block an unqualified "supported" rendering (bead AC #2) — test each.
4. **First real population**: seed finding assertions for the
   claim-vs-evidence demo's headline claims (silent-proceed lower bound,
   handler-class split, per-origin inspection counts — take current values
   from `.agent/demos/claim-vs-evidence/` in the snapshot), each with
   evidence refs pointing at the demo artifacts, so the operator's outreach
   text can cite claim keys with live status.
5. Tests: status computation per blocking state; preset parity (same claim,
   same status across presets); staleness transition (evidence epoch
   advances → status degrades to needs-rerun, never silently stays
   supported).

## Constraints

- Depends conceptually on the judgment lifecycle (parallel job ann-04
  finishes 37t.12) — consume the MERGED PR #2791 lifecycle from the
  snapshot as authority; state interface assumptions where ann-04's
  residuals matter.
- No new durable tables if avoidable; findings/judgments live in the
  existing assertions substrate. Any new AssertionKind value triggers the
  known regeneration set (`render openapi`, `cli-output-schemas`,
  user_audit every-kind test).
- Renderings are public-repo content: sanitized.

## Deliverable emphasis

HANDOFF.md: projection architecture, status semantics table, preset
catalog, the seeded claim population (keys + current statuses), the 37t.14
interface assumption, and how README/outreach text should cite claim keys.
