# Beads Audit Swarm — Shared Brief

You are one auditor in a 16-agent swarm improving the **Polylogue Beads backlog itself**
(not the product). Repo: `/realm/project/polylogue`. Beads CLI: `bd`. Orchestrator applies
all mutations; **you are READ-ONLY on Beads — never run `bd update/create/close/dep/delete`.**

## Your job
Audit your assigned slice and return STRICT JSON findings (schema below). Ground EVERY finding
in evidence: quote the bead's own text (from your domain file), or cite a file:line from code/docs,
or a `bd show <id>` result. No hand-waving. Be bold but precise.

## Doctrine you must honor (from operator memory)
- **Execution-grade contract**: a ready bead has description=*why+evidence*, design=*how: files,
  algorithm, pitfalls*, acceptance=*checkable done-state + verify commands*, and PR-shaped size.
  P0/P1 and ready-frontier P2 beads should meet this; deep P3 backlog need not yet.
- **Priority frame**: proof campaigns > campaign enablers > correctness > surface hygiene.
  The P0 campaign epics (sru/tf2/jxe) are CLOSED; `polylogue-cfk` (uplift re-run) + `polylogue-3tl`
  (external legibility) are the live top-of-frame. Flag tier-4 polish that displaces tier-1 work.
- **GH issues are external-refs, NOT mirrors.** `gh-1807` umbrella was DROPPED ("doctrine lives in
  docs"). `gh-2317/2309` folded into surface-algebra (jnj). Closed GH issues cited as active = stale.
- **Don't invent work.** A "missing bead" must be implied by an epic's OWN acceptance criteria, a
  design doc, or a concrete code/docs gap — not by your preference.
- Don't propose overwriting careful operator-authored fields; propose ADDITIVE design/acceptance
  where absent, or a specific rewrite only when the current text is wrong/contradictory.

## What to look for
1. **Under-specification** — P1 / ready-frontier beads missing design or acceptance. Propose the
   actual text (execution-grade), not "add acceptance criteria".
2. **Gaps / missing beads** — work an epic's AC implies but no child covers. Propose full spec.
3. **Inconsistencies** — contradictory scope between a bead and its epic; stale refs (dropped #1807,
   closed epics); wrong type (a bug filed as task); acceptance that doesn't match description.
4. **Dependency / priority** — missing deps (bead B clearly needs A first), likely cycles, priority
   that violates the frame, epic whose children don't cover its AC.
5. **Reorg** (authorized) — beads that should be re-parented under a different epic, or a cluster of
   standalone beads that implies a MISSING epic.

## Output — return ONLY this JSON (no prose around it)
```json
{
  "domain": "<your domain key>",
  "summary": "<=3 sentences: health of this slice",
  "underspecified": [{"id":"","missing":["design"|"acceptance"],"proposed_design":"","proposed_acceptance":"","why":""}],
  "gaps": [{"proposed_title":"","type":"feature|task|bug|decision","priority":0,"parent":"<epic id>","why":"","design":"","acceptance":"","deps":["<id>"]}],
  "inconsistencies": [{"id":"","kind":"stale-ref|scope-conflict|wrong-type|acc-mismatch|other","evidence":"","fix":""}],
  "dep_priority": [{"id":"","issue":"missing-dep|likely-cycle|priority-violation|epic-ac-gap","detail":"","proposed":""}],
  "reorg": [{"id":"","change":"reparent|reprioritize|new-epic","from":"","to":"","rationale":""}]
}
```
Keep each array to your highest-confidence items (quality over volume; ~3-8 per array max).
Empty arrays are fine. Return within ~6 minutes — do not deep-crawl the codebase.
