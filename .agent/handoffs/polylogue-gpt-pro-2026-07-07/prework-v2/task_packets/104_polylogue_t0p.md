# 104. polylogue-t0p — Capture the rest of Claude Code: todos, file-history, prompt history, debug artifacts

Priority/type/status: **P2 / task / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Session JSONL is one artifact among many that Claude Code writes, and the others answer questions transcripts cannot: ~/.claude/todos/*.json = the agent's live PLAN state per session (task lists with status — plan-vs-execution comparison becomes structural); file-history/ = pre-edit file snapshots (ground truth for the yrx changes view, catches what tool-log reconstruction misses); history.jsonl = the operator's prompt history across sessions (paste-detection and prompt-reuse analytics); history-summaries/, debug/, mcp-logs/ (MCP failure forensics), ide/ locks, jobs/, ccusage/stats caches (cost cross-checks — lpl already eyes stats-cache.json). None are captured; several are pruned by the harness on its own schedule, so what is not ingested is eventually LOST.

## Existing design note

Priority order by evidence value: (1) TODOS: small JSON, session-keyed, trivially parsed -> new artifact kind + a session-linked read model (plan items with status transitions when multiple snapshots exist); analytics: plan-completion rate by session outcome (a construct-valid 'did it do what it planned' measure — feeds claim-vs-evidence). (2) FILE-HISTORY: content-addressed capture into the blob store keyed to session+path (dedup makes this cheap); yrx gains a ground-truth lane (diff reconstruction cross-checked against actual snapshots — discrepancy is itself a finding). (3) PROMPT HISTORY: ingest as operator-authored evidence rows (privacy note: already local, same tier as transcripts). (4) MCP-LOGS/DEBUG: lower value, opt-in artifact kinds for failure forensics only. Each lands via the artifact-taxonomy path (classify_artifact + watcher roots) — no bespoke silos; each gets a fidelity declaration (fs1.3 pattern). VERIFY current dir shapes against the live ~/.claude on the operator machine first; these are undocumented harness internals that move (the deobfuscation checkout in ~/.claude is prior art for shape archaeology).

## Acceptance criteria

Todos and file-history ingest end-to-end from the live machine into artifact kinds with provenance; plan-vs-outcome measure registered (9l5.7) with tier=structural; yrx cross-check lane reports agreement rate between reconstructed and snapshot diffs; watcher covers the new roots; fidelity declared per artifact kind.

## Static mechanism / likely defect

Issue description localizes the mechanism: Session JSONL is one artifact among many that Claude Code writes, and the others answer questions transcripts cannot: ~/.claude/todos/*.json = the agent's live PLAN state per session (task lists with status — plan-vs-execution comparison becomes structural); file-history/ = pre-edit file snapshots (ground truth for the yrx changes view, catches what tool-log reconstruction misses); history.jsonl = the operator's prompt history across sessions (paste-detection and prompt-reuse analytics); history-summaries/, debug/… Design direction: Priority order by evidence value: (1) TODOS: small JSON, session-keyed, trivially parsed -> new artifact kind + a session-linked read model (plan items with status transitions when multiple snapshots exist); analytics: plan-completion rate by session outcome (a construct-valid 'did it do what it planned' measure — feeds claim-vs-evidence). (2) FILE-HISTORY: content-addressed capture into the blob store keyed to sess…

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

1. Priority order by evidence value: (1) TODOS: small JSON, session-keyed, trivially parsed -> new artifact kind + a session-linked read model (plan items with status transitions when multiple snapshots exist)
2. analytics: plan-completion rate by session outcome (a construct-valid 'did it do what it planned' measure — feeds claim-vs-evidence).
3. (2) FILE-HISTORY: content-addressed capture into the blob store keyed to session+path (dedup makes this cheap)
4. yrx gains a ground-truth lane (diff reconstruction cross-checked against actual snapshots — discrepancy is itself a finding).
5. (3) PROMPT HISTORY: ingest as operator-authored evidence rows (privacy note: already local, same tier as transcripts).
6. (4) MCP-LOGS/DEBUG: lower value, opt-in artifact kinds for failure forensics only.
7. Each lands via the artifact-taxonomy path (classify_artifact + watcher roots) — no bespoke silos

## Tests to add

- Acceptance proof: Todos and file-history ingest end-to-end from the live machine into artifact kinds with provenance
- Acceptance proof: plan-vs-outcome measure registered (9l5.7) with tier=structural
- Acceptance proof: yrx cross-check lane reports agreement rate between reconstructed and snapshot diffs
- Acceptance proof: watcher covers the new roots
- Acceptance proof: fidelity declared per artifact kind.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
