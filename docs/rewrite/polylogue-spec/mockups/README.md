# Web reader mockups

Static, self-contained HTML (inline CSS, one small copy-to-clipboard script,
light and dark through `prefers-color-scheme`). Open the files directly in a
browser. All data is from the Morrow example archive
(`examples/archive/FACTS.md` and the raw files), with the clock at 2026-10-08
12:00 Europe/Warsaw. Links between the five files work; all other links are
placeholders.

| File | Shows | Spec |
|---|---|---|
| `index.html` | overview: counts, recent sessions with lineage hints, recent notes, sources and the unrecognized `atis-latch` record, search box | `spec/surfaces/web.md`, the overview, and bare `polylogue` / `status` examples |
| `search.html` | results for `chrono`: 3 sessions, 7 hits, the subagent hit under its parent, the decision note on its hit, the defaults that excluded hits | `spec/surfaces/web.md` search; `spec/query.md` defaults |
| `session.html` | reader for session #2: tool calls collapsed with outcome first and expandable (`<details>`), the failing `cargo test` with exit 101 and 3m 43s, the subagent link, thinking as one dim line, the decision note beside its message, a lineage panel, a continuation line, and the compaction, branch and inherited-prefix indicators | `spec/rendering.md`; `spec/surfaces/cli.md`, read |
| `cost.html` | cost by day (bars, unknown shown hatched), by model, by origin, most expensive sessions | `spec/surfaces/web.md` cost views; `spec/analysis.md` |
| `page.html` | later slice: the investigation page `page:morrow-next-friday` with a question, sections, an Evidence message reference (rendered from the archive, not copied), a kept result, a live query, a Findings note from a `::finding` marker, members, a ping composer showing the exact resume command, and a decision box | `spec/user-data.md` "Pages, workspaces and investigations"; `evidence/pages-and-workbench.md` |

Every view has "copy as CLI command" (`spec/surfaces/web.md`).

## Spec gaps found while drawing

1. **The page example has no backing in the archive.** In `spec/user-data.md`, the Evidence quote ("On a Friday, `next friday` returns the same date") does not match the archive message (`msg_01a20000a11`: "Found a bug: "next friday" evaluated on a Friday returns the same day …"). Session #6 also contains no `::finding` marker. The mockup quotes the real message and labels the finding as illustrative. Either add the marker to #6 in `build.py` or mark the example as hypothetical.
2. **The investigation scaffold has two section lists.** In `user-data.md`, `page new --template investigation` prints "Goal, Assumptions, Evidence, Findings, Decision". The minimal model lists eight sections (adding Alternatives, Plan and Open questions). The mockup uses a subset.
3. **Session #2 has no compaction and no branches**, but the reader must show both. The mockup shows them in a side panel, taken from #4 (compaction at 14:40, 182k → 14.5k tokens) and #7 (branch at 21:04). A single example session with all the indicators might be clearer.
4. **No web routes for pages or notes.** `spec/surfaces/web.md` lists `/`, `/search`, `/s/SESSION`, `/cost` and `/live` only, so the page mockup has no defined URL.
5. **When tool durations are shown is not stated.** `rendering.md` writes `[duration]` in brackets, as if optional. The CLI example shows a duration only on the failing call. The mockup shows durations on every call with a measurable one, which looks odd for synthetic Read calls (6m 30s). The rule should say which calls show a duration.
6. **Cost by origin and "most expensive sessions"** are not specified anywhere. They were computed here: claude-code $5.86 including the subagent, codex $0.33.
7. **The per-session cost column with subagents:** the session list shows $2.85 for #2, while the summary shows "+ $0.16 in 1 subagent". The spec does not say whether list rows include subagent cost; the mockup leaves it out of the list.
8. **Where notes appear in the web reader:** "notes beside the messages they target" could mean a right margin or inline. The mockup puts them right-aligned on the message header, as the CLI does.
9. **Members and roles:** `@parser` → #4 is invented. The spec gives no rule for which session a member points to after that session is resumed into a new one.
