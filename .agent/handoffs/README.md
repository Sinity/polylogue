# Polylogue external-agent handoffs

Tracked home for corpora produced by external AI sessions (ChatGPT/GPT-Pro,
Gemini/AI Studio, etc.) that Beads notes reference by path — design docs,
research bundles, prework packets, launch deliverables. Moved out of the
gitignored `.agent/scratch/` on 2026-07-16 once each corpus was confirmed
still load-bearing (referenced by open beads' `notes`/`design` fields) and
not safely archivable. See `polylogue-60v8` for the ongoing scratch-cleanup
check and `.agent/archive/scratch-2026-07/` for corpora that had zero live
references and were retired instead.

| Directory | Source date | What it is |
| --- | --- | --- |
| `polylogue-gpt-pro-2026-07-06/` | 2026-07-06 | R&D corpus (bundles/, DR reports) digested into the tech-tree bead graph (epic `polylogue-ejm3`, closed) |
| `polylogue-gpt-pro-2026-07-07/` | 2026-07-07 | The big one: 194 per-bead prework task packets (`prework-v2/task_packets/`) backing ~188 open/closed beads' `[Prework packet]` notes, plus `upgrade-setup/` (delivery-gate program corpus) and a captured 811-msg session. The old `task_packets/task_packets/` self-symlink workaround was collapsed to a single segment during this move — bead notes were rewritten to match. |
| `polylogue-gpt-pro-2026-07-07-design-reports/` | 2026-07-07 | Design reports (MCP surface collapse, one-read-contract cut, storage-twins SQL inventory) from the same wave |
| `polylogue-deep-research-2026-07-09/` | 2026-07-09 | Deep-research lanes (D01 positioning, D02 retrieval stack, D07 eval lane) — see `00-INDEX.md` |
| `polylogue-legibility-kit-2026-07-10/` | 2026-07-10 | External-legibility launch kit v1 — complete: 16 fork-prompts, mockups, patches, decks, and real generated evidence (a demo-tour archive with actual `source.db`/`index.db`/etc). See `00-START-HERE.md`. |
| `polylogue-legibility-kit-v2-2026-07-10/` | 2026-07-10 | v2 ("second edition") of the above — **not a superset of v1**, it's provably incomplete per its own `MISSING-FROM-DOWNLOAD.txt` (missing the entire fork-prompts corpus, iteration audit, validation report). Don't archive v1 on the assumption v2 replaced it. `polylogue-3tl.18` owns adjudicating/retiring this whole parallel-control-plane pattern. |
| `polylogue-readme-positioning-2026-07-14/` | 2026-07-14 | README/positioning draft work — see `START-HERE-polylogue.md` |
| `polylogue-session-snapshot-2026-07-08/` | 2026-07-08 | A single captured ChatGPT session (html/md/messages.json/router-stream.js) used as an analysis fixture |
| `polylogue-sol-pro-2026-07-15/` | 2026-07-15/16 | 28 Sol/Pro launch-dispatch deliverables (design+patches), currently the only surviving copy of that output — see its own `README.md` and `polylogue-3v1` for the capture-gap status |

For the Gemini/AI Studio equivalent, see `/realm/inbox/handoffs/polylogue-gemini-2026-07-16/` (outside this repo — not yet unified here, tracked separately).
