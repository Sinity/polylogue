---
created: 2026-06-28
purpose: Audit read/find CLI flags against the query DSL — which are obsolete (selection belongs in DSL), which expose a real DSL gap, which are legitimately presentational
status: active — operator gave implicit sign-off to execute collapses (2026-06-28)
read-with: context-pack-gratuitous-2026-06-28.md
---

# DSL vs flags audit

## Principle (operator, algebra-vs-hardcoded)
SELECTION (which sessions/units, filtered/bounded/sorted) belongs in the QUERY DSL — one expressive
substrate. DELIVERY/RENDERING (where output goes, what format, which view) belongs in flags. Any flag that
SELECTS or FILTERS is a candidate for DSL absorption. Make the DSL more expressive where it has a real gap;
remove flags that merely duplicate it.

## DSL already covers (EXPRESSION_FIELD_REGISTRY, fields.py — the tokens that exist today)
session: repo_names(`repo:`), origins(`origin:`), since/until, query_terms(free text), tags(`tag:`),
title, cwd_prefix, referenced_path, parent_id, root, sidechain, continuation, has_paste/thinking/tool_use
(`has:`), has_types, has_branches, similar_text/similar_session_id, sample, latest.
bounds/shape: limit, offset, sort, reverse, min_messages/max_messages, min_words/max_words, typed_only.
unit scopes (pipe): `messages/actions/blocks/assertions/files/runs/observed-events/context-snapshots where …`
with role:/text:/group by FIELD | count | sort by. action_terms/tool_terms/action_sequence.

## SMOKING GUN
`cli/read_views/context.py:run_read_context_pack` does `del request` — it DISCARDS the upstream `find`
query entirely and re-selects via its own flags. So `find <QUERY> then read --view context-pack` ignores
QUERY. Every context-pack selection flag duplicates an existing DSL token.

## Classification of read-verb flags
### A. REDUNDANT with DSL → remove; route selection through the upstream `find` query
- `--project-path`  → `cwd:`/cwd_prefix
- `--project-repo`  → `repo:`
- `--since` / `--until` → since/until
- `--pack-origin`   → `origin:`
- `--pack-query`    → free-text query_terms
- `--max-sessions`  → `limit:`
- `--message-role`  → `messages where role:` (unit-scope select)
- `--material-origin` → message material predicate (add token if missing; see B)
- `--message-type`  → message_type
(`--max-messages` is a per-session render bound — borderline; see C.)
These exist ONLY because the context-pack/messages views re-implement selection instead of consuming the
query. Collapse: views take their session set from the executed query; drop the flags.

### B. GENUINE DSL GAPS → make the DSL more expressive (this is the high-leverage part)
1. **Related-unit projection / "include X with sessions"** — `--include-assertions` (and the pack's bundling
   of assertions/messages/actions per session) is a PROJECTION: "return sessions WITH their assertions".
   The DSL pipe switches unit scope (`sessions where … | assertions where …`) but cannot ATTACH a related
   unit to each session in output. PROPOSAL: a projection/`with` clause, e.g.
   `sessions where repo:polylogue with assertions, actions` → each session row carries its joined units.
   This single extension absorbs `--include-assertions`, `--include-candidates`, and the context-pack's
   per-session bundling — i.e. context-packs become expressible queries.
2. **token budget** — no `--max-tokens`/DSL bound on accumulated OUTPUT size (only count bounds
   min/max_messages, limit). PROPOSAL: a DSL bound `limit tokens N` (or `--max-tokens` as a general read
   modifier) with honest omission accounting. General — every multi-result read wants it.
3. **material origin token** — confirm a DSL token for messages.material_origin exists; if not, add
   `material:` (note: material_origin itself has construct-validity issues, see construct-validity-audit).

### C. LEGITIMATELY presentational (keep as flags — NOT selection)
- `--view`, `--to`/destination, `--format`, `--out`, `--fields`, `--all`/`--first` (delivery cardinality),
  `--no-redact`, `--no-code-blocks`, `--no-tool-calls`, `--no-tool-outputs`, `--no-file-reads`,
  `--prose-only` (render shaping of an already-selected session), `--window-hours`/`--related-limit`/
  `--since-hours`/correlation knobs (these parameterize a VIEW's computation, not session selection —
  though related_limit/window are arguably query-like and could move later).

## Execution plan (sign-off implicit per operator)
1. **DSL `with <units>` projection clause** (gap B1) — the keystone: makes "session + related units" a query,
   which is what context-pack IS. Highest leverage; absorbs include-assertions/candidates + pack bundling.
2. **Token bound** (gap B2) — `limit tokens N` / general `--max-tokens` with omissions.
3. **Collapse context-pack + messages views** to consume the upstream query selection; remove class-A flags;
   re-express `compile_context`/`build_context_pack` as thin shims over `find … then read` (retire the
   ContextSpec/ContextImage/compiler trio + mcp/context_pack.py 15 DTOs once equivalent).
4. Verify each step against the dev archive: the same output reachable via plain `find … then read …`.

This is the same substrate-not-interpreter shape as the insights redesign. Each step is a real artifact;
do them in throughput order, verifying on real data.
