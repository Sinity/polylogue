# C10 — Interactive query-composer UX (the headline feature)

Status: **design proposal**. Scope: the keystroke → structural-completion →
live-preview → commit loop for composing Polylogue read queries against the warm
hot daemon. Read-only survey of existing surfaces; distinguishes **[evidence]**
(grounded in code) from **[proposal]**.

Frame (from `SWARM_BRIEF.md`): warm memory-hungry daemon; thin client over UDS;
`complete(partial)` + `preview(spec)` answer in single-digit ms; no bare-query;
composable projection algebra + user macros/views in `user.db`.

---

## 1. What already exists (build on, don't reinvent)

**[evidence] The completion brain is already built and composer-shaped.**
`polylogue/archive/query/completions.py` exposes `query_completion_candidates(kind, incomplete, unit, field)`
over 12 kinds (`QUERY_COMPLETION_KINDS`, completions.py:39): `field`,
`structural-unit`, `structural-field`, `terminal-source`, `terminal-field`,
`pipeline-stage`, `projection-unit`, `projection-field`, `count-operator`,
`numeric-operator`, `date-operator`, `action`. Every candidate
(`QueryCompletionCandidate`, completions.py:59) already carries the fields a
live composer needs:

- `insert` / `replace_start` / `replace_end` — span-replacement, not naive append.
- `display`, `group`, `description`, `score` — for a grouped, ranked popup.
- `danger` — destructive terminals (`delete`, `mark`) flagged (completions.py:539).
- `preview_command` — **a candidate can already carry a preview**. The schema
  anticipated this UX.
- `stale`, `unsupported_reason`, `payload_model` — honesty + drill-down.

**[evidence] The grammar and pipeline seams.** Query is a Lark LALR grammar
(`expression.py:623`, `_QUERY_GRAMMAR`); the pipeline is hand-split on `|`
*outside* the grammar (`_split_pipeline_stages`, expression.py:1436), and `with`
projection units split similarly (`_iter_top_level_with_positions`,
expression.py:1488). Set-algebra (fnm.13) lands as `| intersect (SUBQUERY)`
binary stages at that same pre-Lark layer (`docs/design/query-set-algebra.md`
§5 Design A).

**[evidence] `explain_query_expression`** (server_tools.py:1016) already returns
a parse/lowering explanation, and field lowerings carry human `plan_description`
labels (`fields.py`, e.g. `_label("origin")`). That is the raw material for a
parse/explain strip.

**[evidence] A minimal preview TUI already exists.** `ui/tui/screens/search.py`
is Textual: an `Input` over a split `DataTable` (results) + `Markdown` (preview),
routing through `TUIReadSurface.search_sessions` → the same `SessionListResponse`
envelope as CLI/MCP/web. It is submit-then-render (no live preview, no
completion). The composer is the *evolution* of this screen, not a new app.

**The one real gap:** completion today is **kind-driven** — the caller must tell
it which of the 12 contexts applies. A live composer must **infer the context
from the parse state at the cursor**. That inference (`context_at(text, cursor)`)
is the new primitive the daemon must own (§3.1). Everything downstream already
exists.

---

## 2. Recommended interaction model (the headline)

**One daemon-backed live-preview composer, exposed as two RPCs that every client
shares.** The composer is not a bespoke TUI feature; it is a *protocol* —
`complete` + `preview` — and the Textual TUI, CLI inline completion, and a future
Go/Rust client are all thin renderers of it. This keeps the "one read algebra"
(4p1) honest: there is exactly one brain deciding what you can type next and what
it would return.

Two daemon RPCs (thin client sends text + cursor; daemon does all substrate work):

```
complete(text, cursor, session) -> { context, candidates[], error_span? }
preview(text, session, stage?, budget) -> { relation_at_stage, explain, cost, cardinality, sample_rows[] }
```

Design tenets:

1. **The pipeline IS the interaction spine.** A query is a chain of
   relation→relation stages (`find … | group by … | intersect (…) | read`). The
   composer makes each `|` boundary a first-class, previewable checkpoint. You
   build left-to-right and *watch the relation change shape* at every stage. This
   is the whole point of the warm daemon: preview-at-stage in single-digit ms.

2. **Two densities, not two apps.** The operator rejected bare-query. So there is
   one composer with a **quick lane** (open on a verb, type an FTS/semantic term,
   Enter → results — feels like search) and a **full lane** (the same surface, but
   you keep typing `|` stages and the preview keeps refining). Quick is just full
   with a short pipeline. No mode-switch cliff; `|` is the graduation.

3. **Preview is bounded, read-only, and honest.** Preview never runs a
   destructive terminal, never pays for a full export, never embeds a corpus. It
   caps rows (default 20), uses count-only plans for aggregates, and labels
   sampled/estimated results as such. Commit is the *only* path that executes the
   real terminal unbounded.

4. **Completion is context-inferred and span-aware.** No mode key to say "I'm
   completing a field now." The daemon parses up to the cursor, decides the
   context, and returns candidates with `replace_start/end` so accepting one
   surgically rewrites the right span (already supported, completions.py:72).

---

## 3. The keystroke → completion → preview → commit loop

### 3.0 Timing / cancellation model [proposal]

Two independent debounced streams over the UDS socket, each carrying a monotonic
`seq`; the daemon drops any request whose `seq` is stale for its stream:

| Stream | Trigger | Debounce | Daemon work |
|---|---|---|---|
| `complete` | every keystroke / cursor move | ~0 ms (send immediately, cancel in-flight) | parse-to-cursor + candidate lookup (pure metadata, no DB) — target < 3 ms |
| `preview` | idle after typing | 120 ms idle | plan + bounded execute of the relation at the cursor's stage — target < 30 ms warm |

Completion is cheap (metadata only) so it fires eagerly; preview is DB-touching
so it waits for a typing pause. Both are cancellable: a newer keystroke
supersedes an older preview mid-flight (the daemon checks `seq` before the
expensive execute step). This is *only* affordable because the daemon is warm and
resident — a cold `polylogue` process (~240 ms import, brief §Import tax) could
never do per-keystroke completion.

### 3.1 `context_at(text, cursor)` — the new primitive [proposal]

The composer's intelligence is a function that, given the raw text and cursor
offset, returns *which of the 12 completion kinds applies and with what
unit/field binding*. It reuses the existing split seams:

1. Split into pipeline stages (`_split_pipeline_stages`) + `with`-clauses; locate
   the stage containing the cursor and its **stage role** (session_scope / sort /
   group / set-op-operand / terminal / …).
2. Within that stage, do a **partial parse** of the text left of the cursor to
   classify the token under construction:
   - bare word, no `:` → `field` (or `terminal-source` / `pipeline-stage` if the
     stage position expects a verb) — dispatch to `query_field_candidates`.
   - `origin:` with cursor after the colon → **value** completion for that field
     (see §3.2 gap).
   - inside `exists <unit>(` → `structural-field` bound to `unit`.
   - inside `<source> where` → `terminal-field`.
   - after `|` at a fresh stage → `pipeline-stage` + set-op verbs
     (`intersect (`, `union (`, `except (`) per set-algebra §7.
   - inside a set-op's `(` … → recurse: it's a full sub-query context.
   - after `with ` → `projection-unit`; inside `with unit(` → `projection-field`.
3. Return `{ kind, unit?, field?, replace_start, replace_end, incomplete }` and
   call the existing `query_completion_candidates(...)`.

This is the missing glue. It is a *dispatcher over machinery that already exists*,
not new domain logic. Ship it in the substrate (`archive/query/`) so CLI, TUI,
and MCP inline-complete identically (the `fnm.11` parity requirement).

### 3.2 Value completion — the honest gap [proposal]

Today the 12 kinds complete **field names, units, operators, verbs** — the
*structural* vocabulary. They do **not** complete **field values** (`origin:cod…`
→ `codex-session`; `tag:aut…` → `auth-refactor`; `model:…`). Value completion is
what makes a composer feel alive, and it is inherently *live* (values come from
the archive, not a static registry). Add a 13th kind, `field-value`, resolved by
the daemon against `index.db` — cheap `DISTINCT` reads over small enum-like
columns (origin, model, tool, repo, project) and prefix scans over `tags`. This
is the one completion that *requires* the warm daemon (a static CLI can't know
your tag vocabulary). Cap + cache per column; mark `stale` if the index moved.

### 3.3 Preview-at-stage — the core mechanic [proposal]

The preview pane shows **the relation as it exists at the cursor's pipeline
stage**, by truncating the pipeline at the current stage boundary and executing
that prefix bounded:

- Cursor in `find auth | group by model| read` while editing `group by model`:
  once the stage parses, preview shows the **grouped** relation (model → count);
  while it's still `group by mod`, preview shows the **pre-group** rows plus a
  ghost hint "→ will group by model".
- Each committed `|` stage becomes a **chip on a stage rail** annotated with its
  output cardinality (`auth ›142  | group by model ›7 groups | read`). Clicking /
  arrowing a chip moves the preview focus to that stage's output — you can inspect
  the relation *before* and *after* any transform without deleting text.
- Terminal stages (`read`, `analyze`, `select`) preview their **projection** on a
  bounded sample (e.g. the first matched session rendered with the current
  view/fields), so you see the actual output shape, not just a count.

Preview honesty: the strip labels the relation `≈142 (capped at 500-row probe)`
when the operand cap bit, and `sampled` when rows are a sample. Aggregates run
count-only. Never silently truncate (mirrors set-algebra §6 pagination note).

### 3.4 Set-algebra composition & preview [proposal]

Set-ops are `| intersect (SUBQUERY)` binary stages (set-algebra §5 Design A).
In the composer, opening a set-op paren spawns a **nested operand sub-composer**:
the operand `(…)` gets its own inline completion context (recurse in §3.1) and its
own cardinality readout. The strip renders the live set relationship:

```
A = find auth            ›142
B = intersect (test)     ›88
A ∩ B                    ›37     (left-ranked; A's order restricted, set-algebra §3.1)
```

This is the payoff of live preview for set-algebra: the user sees `|A|`, `|B|`,
and the combined size **before committing**, so an accidental `except` that zeroes
the set (`A except A = ∅`, §3.2) is visible immediately, not after a run. Grain
mismatch (§4, P1 fail-closed) surfaces as an inline error in the strip with the
suggested `| sessions` lift, not a post-hoc traceback.

### 3.5 Commit / graduation

Preview is a bounded read; **commit** is the graduation to a real run:

- **Enter on a complete, terminal-bearing query** → execute unbounded with the
  real terminal. `read`/`analyze`/`select` render to the chosen destination; the
  preview pane expands to the full result.
- **A pipeline with no terminal** defaults to `read` on Enter (the quick lane).
- **Destructive terminals** (`delete`, `mark`; `danger=True`, completions.py:539)
  require an explicit confirm step — commit shows the affected-row count from the
  last preview and a y/N gate. Preview never executed them.
- **Export** = commit with a render spec: a `>` / "send to" affordance opens a
  render picker (format × destination: markdown/json/…/file/clipboard/browser,
  per brief Render layer). Export re-executes unbounded through the same spec, so
  what you previewed is what you export.
- Every commit writes a **recall entry** to `user.db` (query text + resolved spec
  + result fingerprint + timestamp). Naming a recall entry promotes it to a
  **macro** (`@name`, fnm.12) usable as a set-op operand or query prefix.

### 3.6 History / recall

- **Ctrl-R reverse-search** over `user.db` recall entries: fuzzy-match prior query
  strings, preview the recalled query live before loading it into the composer for
  editing. (The warm daemon makes recall-preview free.)
- **`@`-completion**: typing `@` anywhere a query is valid completes saved macros
  (a 14th completion source over `user.db`), so composed cohorts are reusable in
  set-ops (`@arm_pack | except (@arm_raw)`, set-algebra §7).
- Recall entries store the *resolved spec*, not just text, so a recalled query
  survives grammar evolution and shows what it actually did.

---

## 4. Screen layout

```
┌─ Polylogue composer ─────────────────────────────────────── daemon ●warm ─┐
│ QUERY                                                                       │  ← query input line (editable)
│  find auth | group by model | read▏                                        │
│  ── stage rail ──────────────────────────────────────────────────────────  │
│  [find auth ›142] [group by model ›7] [read]                               │  ← per-stage cardinality chips
├─ context: pipeline-stage (after '|')  · valid ─────────────────────────────┤  ← parse/explain strip
│  ▸ read        render matched sessions            (terminal)               │
│  ▸ analyze     run analysis over the relation     (terminal)               │  ← completion candidates
│  ▸ select      pick rows for a follow-up action   (terminal)               │     (grouped, ranked, span-aware)
│  ▸ intersect ( combine with another query's set   (set-op)                 │
├─ PREVIEW  ·  relation @ group-by  ·  ≈142 sessions → 7 groups  ·  ~18ms ───┤  ← live preview pane
│  model                    sessions                                          │
│  claude-opus-4                 61                                           │
│  gpt-5.2-codex                 44                                           │
│  claude-sonnet-4               22                                           │
│  … 4 more groups                                                            │
├─ lexical(bm25) · budget 500-row probe · cost n/a · ⏎ run  ^R recall  ^E export ┤ ← status/keybinding footer
└────────────────────────────────────────────────────────────────────────────┘
```

Five regions, top-to-bottom: **(1) query line** with an editable cursor; **(2)
stage rail** showing each `|`-stage as a chip with its output cardinality; **(3)
parse/explain strip** = current completion context + validity (`valid` /
`incomplete` / `error @col`); **(4) completion candidates** — grouped, ranked,
`danger` in red, drawn inline under the cursor's stage; **(5) preview pane** with a
one-line meta header (which stage, cardinality, preview latency); **(6) footer**
= active retrieval lane, preview budget, cost estimate, and the commit/recall/
export keys. The completion list and preview coexist — completion is
transient/keystroke-driven, preview is the persistent workspace.

---

## 5. Concrete states

### State A — mid-field value completion (the quick lane, gap §3.2)

User typed `find origin:cod` in a fresh composer. Context inferred as
`field-value` for `origin`; daemon returns live distinct values; preview already
reflects the partial (incomplete) query as a best-effort structural filter.

```
┌─ Polylogue composer ─────────────────────────────────────── daemon ●warm ─┐
│ QUERY                                                                       │
│  find origin:cod▏                                                           │
│  [find …]                                                                   │
├─ context: field-value (origin:)  · incomplete ────────────────────────────┤
│  ▸ codex-session            412 sessions          origin                   │  ← values from index.db,
│  ▸ codex-cloud-session       19 sessions          origin                   │     each with live counts,
│    (2 values, prefix 'cod')                        ↩ accept replaces 'cod'  │     span-replace 'cod'
├─ PREVIEW  ·  relation @ find  ·  ≈431 sessions (prefix match)  ·  ~11ms ────┤
│  2026-07-04  codex-session   "refactor ingest batch"                        │
│  2026-07-04  codex-session   "topology edge repair"                         │
│  2026-07-03  codex-cloud     "schema v2 evolution"                          │
│  … showing 3 of ≈431 (sampled)                                              │
├─ lexical · probe 500 · ⏎ accept+run  ^R recall  Tab next  Esc clear ───────┤
└────────────────────────────────────────────────────────────────────────────┘
```

Accepting `codex-session` rewrites the span (`replace_start/end`) to
`find origin:codex-session ▏` and the preview re-runs exact.

### State B — building a pipeline stage (full lane, preview-at-stage §3.3)

User has `find touches:ids.py | ` and is choosing the next stage. The stage rail
shows the upstream cardinality; the preview shows the **pre-stage** relation
because the new stage isn't typed yet; the strip offers stage verbs.

```
┌─ Polylogue composer ─────────────────────────────────────── daemon ●warm ─┐
│ QUERY                                                                       │
│  find touches:ids.py | ▏                                                    │
│  [find touches:ids.py ›38]  [ | … ]                                         │
├─ context: pipeline-stage (after '|')  · incomplete ───────────────────────┤
│  ▸ group by      fold rows into groups            group                    │
│  ▸ sort          order the relation               sort                     │
│  ▸ limit         cap row count                     limit                   │
│  ▸ intersect (   ∩ with another query's result    set-op                   │
│  ▸ union (       ∪ with another query's result     set-op                  │
│  ▸ except (      A minus another query's result    set-op                  │
│  ▸ read          render (terminal)                 terminal                │
├─ PREVIEW  ·  relation @ find (upstream of new stage)  ·  38 sessions  ·  ~9ms ┤
│  2026-07-02  codex     "ids.py hashing NFC bug"      142 msg                │
│  2026-06-30  claude    "content-hash boundary"        88 msg                │
│  … 36 more                                                                  │
├─ lexical · probe 500 · pick a stage or ⏎ read  ^R recall ──────────────────┤
└────────────────────────────────────────────────────────────────────────────┘
```

### State C — set-algebra operand with live set sizes (§3.4)

User is composing `find auth | intersect (` and typing the operand sub-query. The
nested operand has its own completion context; the strip shows the live
`|A|/|B|/|A∩B|` readout so the set relationship is visible before commit.

```
┌─ Polylogue composer ─────────────────────────────────────── daemon ●warm ─┐
│ QUERY                                                                       │
│  find auth | intersect ( semantic:"token budget"▏ )                        │
│  [find auth ›142]  [∩ intersect ( … )]                                      │
├─ context: query (set-op operand, grain=session)  · valid ─────────────────┤
│  ▸ semantic:"…"   vector lane      ▸ ~"…"  FTS lane    ▸ date> …            │
│    (operand is a full sub-query — normal completion resumes inside '(')     │
├─ SET PREVIEW  ·  session grain  ·  left-ranked  ·  ~24ms ──────────────────┤
│   A = find auth                       ›142                                  │
│   B = semantic:"token budget"         › 63                                  │
│   ───────────────────────────────────────                                  │
│   A ∩ B                               › 29    (A's order restricted)        │
│  ─ member sample ─                                                          │
│   2026-06-28  codex   "auth token budget accounting"                        │
│   2026-06-21  claude  "session cost + auth headers"                         │
├─ hybrid-aware · operands capped 500 each · ⏎ run intersection  ^E export ──┤
└────────────────────────────────────────────────────────────────────────────┘
```

If the operand were message-grain while `A` is session-grain, the strip would
instead show: `error: grain mismatch — wrap operand in \`| sessions\` to lift`
(set-algebra §4 P1), with a one-key "apply fix" that inserts the lift.

---

## 6. Keybindings [proposal]

Modeless-first, discoverable, terminal-idiomatic:

| Key | Action |
|---|---|
| *(typing)* | edits query line; fires `complete` eagerly, `preview` on 120 ms idle |
| `Tab` / `S-Tab` | next / prev completion candidate (or accept-common-prefix if one) |
| `Enter` (candidate focused) | accept candidate (span-replace), stay in composer |
| `Enter` (query complete) | **commit / run** the query with its terminal (default `read`) |
| `|` | start a new pipeline stage (rail gains a chip; preview re-anchors) |
| `(` after a set-op verb | open nested operand sub-composer |
| `Ctrl-→ / Ctrl-←` | move preview focus across stage-rail chips (inspect stage outputs) |
| `Ctrl-R` | reverse-search recall history (preview before load) |
| `Ctrl-E` | export — open render-spec picker (format × destination) |
| `Ctrl-L` | toggle retrieval lane (lexical / semantic / hybrid) for the active leg |
| `Ctrl-Space` | force-open completion (when popup was dismissed) |
| `Esc` | dismiss completion popup → second `Esc` clears the line |
| `?` (empty line) | cheat-sheet overlay of fields/units/stages from the registries |

No global modes; the only "mode" is *inside `(`* (operand sub-composer), which is
structural, not a keystroke toggle.

---

## 7. Why this shape (recommendation & rationale)

1. **Protocol, not feature.** Making `complete` + `preview` daemon RPCs (with
   `context_at` as the missing inference glue, §3.1) means the composer TUI, CLI
   inline completion, and a future Go/Rust thin client are *renderers of one
   brain*. This is the only way to honor `fnm.11` cross-surface parity and the
   thin-client direction without three divergent completion implementations.

2. **The pipeline is the UX.** Preview-at-stage (§3.3) turns the abstract
   relation-algebra (`R → R` stages, set-algebra §2) into something you *watch*.
   The cardinality-annotated stage rail is the single most valuable affordance:
   it makes "where did my rows go?" answerable at a glance, which is the #1 pain
   of any query language.

3. **Live set sizes de-risk set-algebra.** The `|A|/|B|/|A∩B|` readout (§3.4) is
   the composer earning its keep — set-ops are exactly where users guess wrong,
   and preview shows the answer before commit.

4. **Bounded preview + explicit commit** keeps the warm daemon fast and safe:
   per-keystroke completion is metadata-only; preview is a small capped probe;
   only commit pays full freight, and destructive terminals gate on a confirm
   backed by a real preview count.

5. **Reuse over invention.** The candidate schema (span replace, danger,
   preview_command), the completion kinds, the pipeline split seams, the
   `explain` output, and the Textual split-pane screen all already exist. The net
   new code is: `context_at`, one `field-value` completion kind + a `@macro`
   source, the `preview(spec, stage)` truncation, and the TUI wiring. The design
   deliberately sits *on top of* the substrate the brief says is already being
   made thin (t46).

---

## 8. Open questions (need operator input)

1. **Preview cost visibility.** Should the strip show a live *token/$ estimate*
   for semantic-lane previews (embedding probe cost), or is preview always free
   because it reuses cached embeddings and never embeds new text? (Recommend:
   preview never embeds — it only vector-searches existing rows; label cost `n/a`.)
2. **Preview budget default.** 20 rows / 500-row probe cap — tunable per surface,
   or fixed? Larger caps give truer cardinality but cost latency. (Recommend:
   fixed small default, `Ctrl-B` to bump for a one-off deep count.)
3. **Quick-lane default terminal.** Enter on a terminal-less query → `read`.
   Confirm `read` (not `analyze`/`summary`) is the right zero-friction default.
4. **`field-value` freshness.** Cache distinct-value lists per column in the
   daemon and invalidate on ingest, or read live each time? (Recommend: cache +
   `stale` flag; enum-like columns are tiny and change rarely.)
5. **CLI inline completion parity.** Does the shell (bash/zsh) completion path
   consume the same `context_at`, or is full live-preview TUI-only with the shell
   getting structural completion but no preview? (Recommend: shell gets
   `complete` only; preview needs a persistent pane → TUI/`--interactive`.)
6. **Graduation to a saved view.** When a user builds a projection they like
   (`fields …` / `with …`), should commit offer "save as view `@name`" inline, so
   composed projections become the user-defined views the brief wants (replacing
   built-in named views)? (Recommend: yes — this is how the view vocabulary grows
   from usage.)
