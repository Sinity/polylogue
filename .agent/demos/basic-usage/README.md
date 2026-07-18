# Basic Usage — The Features Actually Work

Generated: 2026-07-18
Archive: Polylogue's deterministic demo archive (`polylogue demo seed`), index schema v40 at capture time.

Eight short, real walkthroughs proving ordinary Polylogue features work
end-to-end, not just in unit tests. Every command below was actually run; every
output is the real result, copied verbatim (only line-wrapping added for
readability). Each walkthrough states exactly what it proves and nothing more.

**Why the demo archive, not the live one:** every command here is real and
copy-pasteable, but it runs against the private-data-free deterministic demo
archive rather than the operator's live archive, for two reasons: (1) this
repository is public — the live archive is the operator's actual work
history and its content must not appear here; (2) at the time this suite was
built, the live archive's derived index was in a known, unrelated degraded
state (see `docs/findings/claim-vs-evidence.md`) that would have made any
live numbers non-representative anyway. The same commands work identically
against a real archive — only the data differs.

Reproduce the whole archive first:

```bash
export POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-basic-usage-demo
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json
```

Then every command below runs unmodified with that `POLYLOGUE_ARCHIVE_ROOT` exported.

## 1. Find — fielded query + pipeline aggregate

File: [`01-find-query.txt`](01-find-query.txt)

```
polylogue --origin codex-session find "sessions where origin:codex-session" then select --json
polylogue find "actions where tool:bash | group by origin | count"
```

The first command is a fielded DSL query (`origin:codex-session`) returning
the matched session refs. The second is a pipeline query: it scopes to
`actions` (not `sessions`), filters by tool name, groups by origin, and
aggregates — the DSL's `sessions where … | <unit>s where … | group by … |
count` grammar (`archive/query/expression.py`), not a bespoke script.

**What this proves:** the query-first CLI's fielded predicates and multi-stage
pipeline grammar both execute against a real archive and return real,
distinct result shapes (a JSON row list vs. a grouped count table).

## 2. Read — exact-ref transcript

File: [`02-read.txt`](02-read.txt)

```
polylogue find id:codex-session:demo-receipts then read --view transcript
```

An exact-ref query (`id:<session>`) piped into `read --view transcript`. The
transcript itself is worth reading closely: the assistant claims "All tests
pass" immediately after a tool result shows `exit_code: 1` and a failing
test — a small, self-contained illustration of exactly the claim-vs-evidence
gap this lane's Phase 0/1 work investigates, captured as normal archive
content rather than manufactured for the point.

**What this proves:** exact-ref resolution and the `transcript` render view
work end-to-end, and the resulting text is faithful to the underlying
structured tool-result evidence (not reconstructed from prose).

## 3. Search — FTS hit with snippet and provenance

File: [`03-search.txt`](03-search.txt)

```
polylogue find "clock" then select --json
```

An unfielded, quoted free-text query runs through the same FTS5 index this
CLI's `find` verb is backed by (`archive/query/expression.py`'s `near:`/bare
free-text lowering). Every hit carries a stable session ref
(`origin:native_id`), so a caller can always resolve a search hit back to its
exact source. (The MCP `search` tool, demonstrated separately in section 7,
additionally returns a highlighted snippet and message-level ref per hit —
the CLI's `find` surface intentionally keeps this compact for terminal use.)

**What this proves:** free-text search resolves across origins (a ChatGPT
export, a Claude Code session, and a Codex session all matched "clock" from
genuinely different underlying text) and every result is provenance-bearing,
not a bag of disconnected snippets.

## 4. Resume — continuation command generation

File: [`04-resume.txt`](04-resume.txt)

```
polylogue find id:codex-session:demo-receipts then continue
```

**What this proves:** Polylogue can generate the exact runtime-native resume
invocation (`codex resume <id>`) for an archived session, not just read it —
continuity, not just archaeology.

## 5. Cost — disjoint token/cost accounting

File: [`05-cost-usage.json`](05-cost-usage.json)

```
polylogue analyze usage --format json
```

The full output is long (it deliberately keeps provider-event rows, origin
cumulative counters, and per-model rollups as separate evidence streams
rather than collapsing them); the load-bearing excerpt is
`logical_pricing_lanes`, which reports, per pricing provenance:

```json
{
  "provenance": "priced",
  "row_count": 4,
  "usage": {
    "input_tokens": 56000,
    "output_tokens": 10500,
    "cached_input_tokens": 280000,
    "cache_write_tokens": 42000,
    "reasoning_output_tokens": 0,
    "total_tokens": 388500
  },
  "catalog_priced_subtotal_usd": 2.835
}
```

**What this proves:** cost/usage accounting keeps input, output, cached, and
cache-write token lanes disjoint (never silently folded together, matching
the documented Codex-inclusive-token/Claude-cache-token pitfalls) and
reports pricing provenance (`priced` vs. `estimate_only`) per row rather than
one blended number.

## 6. Lineage — composed fork read

File: [`06-lineage.txt`](06-lineage.txt)

```
polylogue find id:codex-session:demo-lineage-fork then read --view transcript
polylogue find id:codex-session:demo-lineage-parent then read --view transcript
```

The fork session's transcript already contains the parent's two turns
("Map the demo lineage base context." / "I have the base context and can
branch the analysis.") composed ahead of its own divergent tail — reading
the fork alone gives the full coherent conversation, not just the child's
new turns. `demo-lineage-parent`'s own transcript is included for direct
comparison: the fork's first two turns are byte-identical to the parent's,
proving genuine prefix composition rather than duplicated storage that
happens to render similarly.

**What this proves:** forked sessions are read as parent-prefix + child-tail
composition (the archive's `session_links` lineage model), not stored or
displayed as disconnected fragments.

## 7. MCP — search → get_session_summary round-trip

File: [`07-mcp-roundtrip.json`](07-mcp-roundtrip.json)

A real MCP client/server exchange over stdio JSON-RPC (the same protocol any
MCP-speaking agent client uses), driven by
`devtools/continuity_replay.py`'s `StdioMCPContinuityRoute` against the demo
archive:

```python
async with StdioMCPContinuityRoute(archive_root) as route:
    search_result = await route.invoke("search", {"query": "clock", "limit": 3})
    # search_result["hits"][0]["session"]["id"] == "chatgpt-export:cross-material-duplicate-01"
    summary = await route.invoke("get_session_summary", {"id": "chatgpt-export:cross-material-duplicate-01"})
```

The MCP surface is mid-rewrite by a parallel lane (#3056 retired the
`archive_list_sessions`/`archive_search_sessions` aliases the same week this
suite was built), so this demo deliberately exercises the two canonical
tools least likely to move — `search` and `get_session_summary` — rather
than the full tool catalog. See `docs/mcp-reference.md` for the current
complete tool list.

**What this proves:** the MCP surface answers the identical question the CLI
answers (search → resolve a session summary) through a real protocol
round-trip, not a mocked handler — this is the continuity surface real agent
clients (not humans at a terminal) actually use.

## 8. Status/health — daemon and archive readiness

Files: [`08-status-health.txt`](08-status-health.txt),
[`08-status-health.debt.json`](08-status-health.debt.json)

```
polylogue status --daemon-url http://127.0.0.1:1
polylogue ops debt list --format json
```

(`--daemon-url` points at an unreachable port so this demo reproduces
identically for a reader with no daemon running at all — `polylogue status`
normally talks to a live `polylogued` daemon's HTTP API first and only falls
back to bounded local SQLite checks when the daemon is unreachable.)

`ops debt list` names concrete, actionable freshness/convergence gaps rather
than a pass/fail flag — in this demo archive: 14 sessions pending embedding
catch-up, `messages_fts` not yet query-ready, and (because this suite seeds
`--with-overlays`) a candidate finding assertion from the claim-vs-evidence
demo overlay still awaiting operator judgment, each with its own evidence
ref and a copy-pasteable remediation command.

**What this proves:** archive health is queryable without a running daemon —
per-tier schema versions, row counts, raw-materialization debt, and FTS/
embedding backfill progress are all real, structural facts read directly
from the five SQLite tiers, not decorative placeholder text — and debt is
reported as named, actionable rows, not a single opaque health flag.

## Regenerate everything

```bash
export POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-basic-usage-demo
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
devtools workspace basic-usage-demo-check --archive-root "$POLYLOGUE_ARCHIVE_ROOT"
```

The check command re-runs every command above against a fresh seed and
asserts each output has the expected *shape* (non-empty hit lists, expected
JSON keys, expected session refs) — not exact volatile counts, since the
demo corpus's construct coverage numbers can grow without invalidating any
of these walkthroughs. See [`COLD_READER_GATE.md`](COLD_READER_GATE.md) for
the fresh-reader verification prompt.

## Files

- `01-find-query.txt`
- `02-read.txt`
- `03-search.txt`
- `04-resume.txt`
- `05-cost-usage.json`
- `06-lineage.txt`
- `07-mcp-roundtrip.json`
- `08-status-health.txt`
- `08-status-health.debt.json`
- `COLD_READER_GATE.md`
- `README.md` — this file.
