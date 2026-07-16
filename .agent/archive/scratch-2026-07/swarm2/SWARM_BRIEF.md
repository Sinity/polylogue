# Swarm brief — CLI / DSL / hot-daemon / composer (2026-07-05)

You are one agent in a design/research swarm for **Polylogue** (a local AI-session
archive with a query-first DSL). Repo: `/realm/project/polylogue`. You are
READ-ONLY on the codebase; your deliverable is a design/research document.

## Shared context you must build on (decisions already made this session)

**Read algebra baseline.** The read surface is `Query × Projection × Render`
(`4p1`, realized in `polylogue/surfaces/projection_spec.py`):
- **Query** = a Lark LALR grammar (`polylogue/archive/query/expression.py:623`,
  `_QUERY_GRAMMAR`): boolean `and/or/not`, field predicates, FTS (`~`), semantic
  (`semantic:`), count/date/time, structural `exists <unit>(...)` over units
  {observed-event, context-snapshot, message, action, block, assertion, file,
  run}, and `seq(A -> B -> C)`. A **pipeline** is hand-split on `|` OUTSIDE the
  grammar (`_split_pipeline_stages`, expression.py:1436): stages
  {session_scope, sort, limit, offset, group, count, transform, terminal};
  terminal actions {read, analyze, select, mark, delete, continue}.
- **Projection** (`ProjectionSpec`) = `body_policy` {full, omit-tool-outputs,
  authored-dialogue, metadata-only} + `exclude_block_kinds` + EvidenceFamily set
  {sessions, messages, blocks, actions, raw, context, chronicle, neighbors,
  correlation, temporal, assertions}. Chosen today via **named views** (summary,
  transcript, dialogue, messages, raw, context, context-image, chronicle,
  neighbors, correlation, temporal, timeline) — a closed preset vocabulary, NOT
  composable.
- **Render** (`RenderSpec`) = format {markdown,json,ndjson,html,obsidian,org,
  yaml,plaintext,csv} × destination {terminal,stdout,browser,clipboard,file} ×
  `layout` × out-path.
- Retrieval lanes: lexical (FTS bm25) / semantic (vector) / hybrid (RRF k=60).
- `fnm` bead roadmap (designed, mostly unbuilt): fnm.1 aggregates, fnm.2 unit
  predicates/windows, fnm.3 SEQ modifiers, fnm.6 terminal->projection, fnm.7
  child-count, fnm.8 lineage-scope, fnm.9 pipeline-as-subquery, fnm.10
  fields/select stage, fnm.11 cross-surface parity, fnm.12 macros, fnm.13
  set-algebra (designed: `docs/design/query-set-algebra.md`).

**Measured facts.**
- The CLI is NOT thin: `cli/` reaches directly into storage/pipeline/archive
  **45** times vs **18** via `polylogue.api`. Real substrate logic lives in the
  CLI. **Decision: `t46` "contracts own surfaces" (make the CLI thin, route
  everything through one contract) is done FIRST — assume it as a prerequisite.**
- Import tax: importing `cli.click_app` costs ~240 ms (click ~120 ms;
  `polylogue.version`→inspect/importlib.metadata ~83 ms; logging ~43 ms).
  `polylogue --help` ~0.35 s; `ops status` ~1.4 s. A Python CLI floor is ~150 ms.
- Daemonless READS work today (direct-archive handlers), but importing the Python
  substrate is unavoidable.

**Architecture direction (the frame to design within).**
- **Powerful, resident "hot" daemon is the target.** It may be MEMORY-HUNGRY and
  feature-rich — do NOT constrain to a small RAM budget; design for capability.
  It is acceptable to REQUIRE the daemon (keep a `--no-daemon` break-glass only).
- **Thin client over a fast local protocol (UDS).** The client (CLI/TUI) should
  do NO substrate work; it speaks a wire protocol to the warm daemon. Because it
  speaks a protocol (not FFI/importing Python), the client could later be
  written in Go/Rust for a sub-10 ms floor. The substrate stays Python.
- **The composer / live-preview is the headline UX**: an interactive surface
  where the user dynamically composes a query (structural completion of fields,
  values, operators, lanes, pipeline stages, set-ops) and sees results PREVIEW
  live as they type — only possible with the warm daemon answering
  `complete(partial)` and `preview(spec)` in single-digit ms.
- **No bare-query support.** The operator does NOT want a bare single token to
  fall into query mode; search is via an explicit verb / the composer. (So the
  "did-you-mean typo hint" is moot.)
- **Fewer named views.** Prefer a composable projection algebra (fields / fold /
  window / budget) and user-defined views + render layouts as `user.db`
  config/macros (generalize `fnm.12`); built-in views become thin defaults.

## Output contract
- Write your deliverable to `/realm/project/polylogue/.agent/scratch/swarm2/<YOUR-ID>.md`
  (e.g. `A2_protocol.md`). Ground claims in real files (`file:line`) where
  relevant; distinguish evidence from proposal; for research, cite sources.
- Be concrete and opinionated; give a recommendation, not a survey. Note open
  questions needing operator input.
- Return a tight (<=200 word) summary as your final message.
- Do NOT edit code or beads. Do NOT run the daemon or heavy tests.
