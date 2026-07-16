# 102. polylogue-7aw — Ingest agent configuration as a source family (skills, CLAUDE.md, hooks)

Priority/type/status: **P2 / feature / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Treat agent configuration as a corpus polylogue versions, queries, and correlates with session outcomes: CLAUDE.md/AGENTS.md revisions, skills, hook configs — config-over-time x outcome-over-time is the continual-learning dataset (composes with the self-experimentation rail). New source family with real design work (identity, content-hash versioning, correlation keys to sessions via repo+time). Git history already carries much of it — the parser may be a git-log walker rather than a file watcher.

## Existing design note

FULL SCOPE (upgraded from notes 2026-07-03; operator: we should have full control/understanding of CLAUDE.md regardless of replacement ambitions): (1) CAPTURE: CLAUDE.md files (global dots/claude tree incl. @-included world-model/operational files, per-project CLAUDE.md, rendered AGENTS.md), skills (~/.claude/skills + repo skills), hooks configs, and settings.json families ingest as a config-artifact source family — versioned by content hash, timestamped, watcher-covered (they change via sinnix commits AND live edits). Each session then joins to the CONFIG STATE it ran under (session_agent_policies table exists — verify shape, extend to reference config-artifact hashes): 'which rules were in force when this session ran' becomes queryable, which is the precondition for every claim about instruction efficacy. (2) UNDERSTAND: skill-invocation tracking (Skill tool calls are structural blocks) -> which skills fire, how often, with what outcomes; per-instruction-file attention is NOT directly observable (models do not cite which CLAUDE.md line they obeyed) — the honest proxies are behavioral: rule-violation rates for rules with structural signatures (rvh machinery), and A/B via the experiment substrate (stc) when a rule change is contentious. (3) THE REPLACEMENT PATH (2192 provocation, operationalized): static CLAUDE.md content classifies into: identity/norms (stays static — cheap, always-on), reference material (belongs in skills — on-demand), state-like content (env facts, project status — should be INJECTED, it goes stale as text), and lessons/corrections (should be assertions on the SRS curve — they are judged memory wearing a config costume). Migration is measured, not big-bang: pick one state-like section of the operator's global CLAUDE.md (e.g. the projects table or session-recall notes), replace with a scheduler-injected equivalent compiled from the archive, run the drills, compare; each migrated section shrinks static context and gains freshness. The activation-layer chapter is deliberately the COUNTER-example: instructions-about-the-substrate stay static because they must precede any query. (4) EDIT FRICTION: with capture + classes in place, 'polylogue config-doc' surfaces render current config state with provenance; edit affordances route to the owning files (sinnix dots) — no shadow editing.

## Acceptance criteria

Config artifacts ingest with content-hash versioning and watcher coverage on the live machine; a session from last week resolves to the exact CLAUDE.md/skill versions it ran under; skill-invocation report renders (which skills, frequency, outcome mix); one state-like global-CLAUDE.md section migrated to injected-equivalent with drill-verified parity and the static text retired; the classification of the operator's current global CLAUDE.md into the four content classes is committed as the migration map.

## Static mechanism / likely defect

Issue description localizes the mechanism: Treat agent configuration as a corpus polylogue versions, queries, and correlates with session outcomes: CLAUDE.md/AGENTS.md revisions, skills, hook configs — config-over-time x outcome-over-time is the continual-learning dataset (composes with the self-experimentation rail). New source family with real design work (identity, content-hash versioning, correlation keys to sessions via repo+time). Git history already carries much of it — the parser may be a git-log walker rather than a file watcher. Design direction: FULL SCOPE (upgraded from notes 2026-07-03; operator: we should have full control/understanding of CLAUDE.md regardless of replacement ambitions): (1) CAPTURE: CLAUDE.md files (global dots/claude tree incl. @-included world-model/operational files, per-project CLAUDE.md, rendered AGENTS.md), skills (~/.claude/skills + repo skills), hooks configs, and settings.json families ingest as a config-artifact source family —…

## Source anchors to inspect first

- `polylogue/sources/dispatch.py` — Current origin/source dispatch logic; target for OriginSpec consolidation.
- `polylogue/sources/import_preflight.py` — Preflight/readiness should report origin strictness and ambiguity.
- `polylogue/sources/provider_completeness.py` — Provider completeness is adjacent to OriginSpec readiness.
- `polylogue/sources/parsers/base.py` — Parser base contracts should be folded into OriginSpec.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. FULL SCOPE (upgraded from notes 2026-07-03
2. operator: we should have full control/understanding of CLAUDE.md regardless of replacement ambitions): (1) CAPTURE: CLAUDE.md files (global dots/claude tree incl.
3. @-included world-model/operational files, per-project CLAUDE.md, rendered AGENTS.md), skills (~/.claude/skills + repo skills), hooks configs, and settings.json families ingest as a config-artifact source family — versioned by content hash, timestamped, watcher-covered (they change via sinnix commits AND live edits).
4. Each session then joins to the CONFIG STATE it ran under (session_agent_policies table exists — verify shape, extend to reference config-artifact hashes): 'which rules were in force when this session ran' becomes queryable, which is the precondition for every claim about instruction efficacy.
5. (2) UNDERSTAND: skill-invocation tracking (Skill tool calls are structural blocks) -> which skills fire, how often, with what outcomes
6. per-instruction-file attention is NOT directly observable (models do not cite which CLAUDE.md line they obeyed) — the honest proxies are behavioral: rule-violation rates for rules with structural signatures (rvh machinery), and A/B via the experiment substrate (stc) when a rule change is contentious.
7. (3) THE REPLACEMENT PATH (2192 provocation, operationalized): static CLAUDE.md content classifies into: identity/norms (stays static — cheap, always-on), reference material (belongs in skills — on-demand), state-like content (env facts, project status — should be INJECTED, it goes stale as text), and lessons/corrections (should be assertions on the SRS curve — they are judged memory wearing a config costume).

## Tests to add

- Acceptance proof: Config artifacts ingest with content-hash versioning and watcher coverage on the live machine
- Acceptance proof: a session from last week resolves to the exact CLAUDE.md/skill versions it ran under
- Acceptance proof: skill-invocation report renders (which skills, frequency, outcome mix)
- Acceptance proof: one state-like global-CLAUDE.md section migrated to injected-equivalent with drill-verified parity and the static text retired
- Acceptance proof: the classification of the operator's current global CLAUDE.md into the four content classes is committed as the migration map.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
