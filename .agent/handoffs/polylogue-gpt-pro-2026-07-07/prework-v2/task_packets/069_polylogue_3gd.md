# 069. polylogue-3gd — Activation layer: the agent-side setup that makes the substrate get used at all

Priority/type/status: **P1 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **blocked-hard**.

Hard blockers: polylogue-d1y, polylogue-pj8

## What the bead says

Operator directive (2026-07-03), verbatim intent: everything built here — assertions, annotation protocol, blackboard, judge/note verbs, MCP tools, remote control — is WASTED WORK if agents do not actually use it, and adoption is behavioral: models use what their context reminds them of. The fate to avoid is inert substrate. The fix is an unapologetically LARGE agent-side activation layer: the operator explicitly authorizes 10K+ tokens (possibly 30-50K including injected state) of instructions + state in global CLAUDE.md / skills / hooks — teaching what exists, when to reach for it, how to phrase it, with worked examples — plus MEASURED adoption so 'is it used' is a number, not a hope. This bead owns the whole layer end-to-end and coordinates the pieces that exist as parts (pj8 cookbook, d1y hooks install, 37t.4 preamble, 37t.2 protocol spec).

## Existing design note

(1) GLOBAL CLAUDE.md SECTION (sinnix-side, dots/claude/ — renders to all agents via render-agents): a substantial 'Polylogue substrate' chapter (~3-6K tokens of standing instruction) structured as: WHAT EXISTS (archive, memory, blackboard, protocol — one paragraph each with the mental model); WHEN-TRIGGERS (a trigger->action table: 'starting work in a repo -> read your preamble, it is evidence', 'about to re-derive how X works -> polylogue find/prior-art flow', 'discovered something durable -> ::lesson marker or polylogue note', 'finished a slice -> markers become candidates automatically; check bd + blackboard', 'stuck/failed -> postmortem flow'); HOW (the exact seven t8t flows with copy-paste invocations); PROTOCOL SPEC (the 37t.2 kinds with 5 worked examples inline — models imitate examples, not descriptions); NORMS (evidence tiers, refs-over-dumps, archive-root pitfall). Not a pointer to a skill — the core lives IN standing context; the skill carries the long tail. (2) SKILL + MCP PROMPTS: pj8 executes under this bead's umbrella. (3) INJECTED STATE: 37t.4/37t.11 preamble is the dynamic half — repo brief, due lessons, messages, affordance index; the CLAUDE.md chapter is the static half; they cross-reference ('your preamble below is compiled from the archive — every line resolves via resolve_ref'). (4) HOOKS: d1y install as the distribution mechanism. (5) VERIFICATION — the part that prevents decay: adoption metrics from affordance-usage (MCP-tool diversity per session, marker emission rate, note/judge invocations, blackboard reads) tracked WEEKLY as a small rendered report; session-cut recovery drills (37t.7 note) as the behavioral test; and an A/B instinct check — the stc experiment machinery can compare activation-chapter-on vs off on real work when tuning size. Iterate the chapter based on measured usage, not taste; the 10-50K budget question is answered by the adoption curve. (6) GROUNDING: current global CLAUDE.md already has the shape to extend (world-model + operational chapters, @-includes, render-agents pipeline); the polylogue chapter lands as dots/claude/world-model/polylogue-substrate.md included from the index — same mechanism, sinnix repo, coordinated commit.

## Acceptance criteria

The substrate chapter exists in dots/claude and renders to Claude+Codex; trigger table, seven flows, and five protocol examples present; preamble cross-reference live; the weekly adoption report renders from affordance-usage with baseline captured BEFORE the chapter ships (so the delta is measurable); one month later the report shows material adoption movement or the chapter is revised (decay-watch is part of acceptance, not an afterthought).

## Static mechanism / likely defect

Issue description localizes the mechanism: Operator directive (2026-07-03), verbatim intent: everything built here — assertions, annotation protocol, blackboard, judge/note verbs, MCP tools, remote control — is WASTED WORK if agents do not actually use it, and adoption is behavioral: models use what their context reminds them of. The fate to avoid is inert substrate. The fix is an unapologetically LARGE agent-side activation layer: the operator explicitly authorizes 10K+ tokens (possibly 30-50K including injected state) of instructions + state in global CL… Design direction: (1) GLOBAL CLAUDE.md SECTION (sinnix-side, dots/claude/ — renders to all agents via render-agents): a substantial 'Polylogue substrate' chapter (~3-6K tokens of standing instruction) structured as: WHAT EXISTS (archive, memory, blackboard, protocol — one paragraph each with the mental model); WHEN-TRIGGERS (a trigger->action table: 'starting work in a repo -> read your preamble, it is evidence', 'about to re-derive …

## Source anchors to inspect first

- `polylogue/coordination/envelope.py` — Coordination envelope model exists; harden it as the shared payload.
- `polylogue/coordination/payloads.py` — Coordination payload types should stay small and evidence-ref oriented.
- `polylogue/coordination/rendering.py` — Rendered advisories should be scheduler-mediated, not chat spam.
- `tests/unit/coordination/test_envelope.py` — Existing envelope tests are the starting verification lane.
- `polylogue/mcp/server_prompts.py:219` — MCP prompt registration exists and can surface cookbook/roles.
- `polylogue/cli/commands/agents.py` — CLI agent commands are the operator-facing entry point.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:31` — ASSERTION_DEFAULT_STATUS is ACTIVE, so missing status currently means trusted active.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:641` — upsert_blackboard_note passes author_kind and no explicit status into upsert_assertion.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — upsert_assertion is the single write chokepoint to patch.
- `polylogue/api/contracts/assertions.py` — Check public assertion request/response contract after changing default status behavior.

## Implementation plan

1. (1) GLOBAL CLAUDE.md SECTION (sinnix-side, dots/claude/ — renders to all agents via render-agents): a substantial 'Polylogue substrate' chapter (~3-6K tokens of standing instruction) structured as: WHAT EXISTS (archive, memory, blackboard, protocol — one paragraph each with the mental model)
2. WHEN-TRIGGERS (a trigger->action table: 'starting work in a repo -> read your preamble, it is evidence', 'about to re-derive how X works -> polylogue find/prior-art flow', 'discovered something durable -> ::lesson marker or polylogue note', 'finished a slice -> markers become candidates automatically
3. check bd + blackboard', 'stuck/failed -> postmortem flow')
4. HOW (the exact seven t8t flows with copy-paste invocations)
5. PROTOCOL SPEC (the 37t.2 kinds with 5 worked examples inline — models imitate examples, not descriptions)
6. NORMS (evidence tiers, refs-over-dumps, archive-root pitfall).
7. Not a pointer to a skill — the core lives IN standing context

## Tests to add

- Acceptance proof: The substrate chapter exists in dots/claude and renders to Claude+Codex
- Acceptance proof: trigger table, seven flows, and five protocol examples present
- Acceptance proof: preamble cross-reference live
- Acceptance proof: the weekly adoption report renders from affordance-usage with baseline captured BEFORE the chapter ships (so the delta is measurable)
- Acceptance proof: one month later the report shows material adoption movement or the chapter is revised (decay-watch is part of acceptance, not an afterthought).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
