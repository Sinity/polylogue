# Polylogue documentation

These pages are Polylogue's design: its behaviour, model, formats, and
the reasons for them. They describe the product as designed. What exists in a
given build is described by CLI help, MCP tool schemas and module
documentation. How work is done in this repository is in
[`AGENTS.md`](../AGENTS.md).

Three files here are inputs to the current build rather than design pages:
[agent-manual.md](agent-manual.md), the card the current MCP server sends as
its instructions, [agent-guide.md](agent-guide.md), the guide the current
build appends to its operation reference, and the generated
[derived-data reference](derived.md). Their locations are inputs to the
runtime and its tests. The target card and tool descriptions are
[agents.md](agents.md#mcp-server).

## Pages

| Page | Covers |
|---|---|
| [architecture](architecture.md) | purpose, principles, the two modes, data flow, the stores, durability, the pipeline (the record and the index, readings, pure parsers, composition, one path for upkeep and build) and its targets, configuration and clocks, the store boundary, deletion |
| [capture](capture.md) | where inputs come from: locations, origins, hooks, tool databases, import, AI Studio through Drive, capture status |
| [extension](extension.md) | the browser extension: live capture, provider import, its own UI, chat actions |
| [archive](archive.md) | stored inputs and their chains, readings and covered chains, text, evolution, parsers and parse versions, goldens, field accounting, format drift and `dev schema`, trace data |
| [sessions](sessions.md) | the parsed model: sessions and what identifies them, messages, names and references, message authorship, times, session links, compaction, attachments, logical sessions |
| [references](references.md) | the one reference grammar and its rules |
| [interfaces](interfaces.md) | the operation table, the line grammar, verb words, chains, the verbs and commands, option names and exit codes |
| [latency](latency.md) | latency classes and budgets, deadlines, the daemon's states, warm state |
| [terminal](terminal.md) | the orientation screen, picker, row numbers, footers, refusals, output, help and `--explain` |
| [completion](completion.md) | completion at every position, where its values come from, and shell registration |
| [query](query.md) | selections, Polylogue's units and fields, defaults, times, paging, deadlines, replies, facets, read-only SQL |
| [transforms](transforms.md) | conditions over related rows, transforms after `\|`, attached rows, combining queries |
| [saved queries](saved-queries.md) | questions, runs, `save`, saved names, shipped saved queries, kept results, figures, findings, watches |
| [search](search.md) | full-text search, semantic ranking and `embed`, fusion, the brief of a search |
| [reading](reading.md) | `read`: views (including a page's contents and trace), show and hide, ranges and seen marks, paging, formats and destinations, alternates, compare and blind |
| [transcripts](transcripts.md) | how a transcript is composed and rendered: branches, compaction, headers, tool calls, thinking, attachments, value states, rendering for models |
| [annotations](annotations.md) | everything authored: the record, kinds, links, current revisions, the write path, `note`, `tag`, `edit`, `withdraw`, annotation lines, checked claims, memory and methods |
| [pages](pages.md) | pages: slots, sections, forms, writing and reading pages, query cells, authorship and proposals, history and checkpoints, names and renames, finding pages, members, reply placement |
| [page text](page-text.md) | Markdown with references; links, citations, embeds and inclusions; cite by quote and the quote check; backlinks; when a source changes; grounding |
| [page working copy](page-working-copy.md) | a page as one Markdown file: editing in your editor, grammar, check-in and conflicts, importing Markdown, writing pages out as a site |
| [page templates](page-templates.md) | templates and parameters, shipped pages and templates, investigations, experiments |
| [authorship](authorship.md) | who wrote a record and how each channel decides |
| [judging](judging.md) | `judge`: verdicts, comparisons, blind judging, rankings, agreement, sampled judging, forecasts, rubrics and passes |
| [alternates](alternates.md) | translations and other renderings of content |
| [markup](markup.md) | annotations, page placements and sends written inside messages |
| [measures](measures.md) | derived data and measures, and the catalog of derived products |
| [agents](agents.md) | the agent contract, trust rendering, the skill file, the three MCP tools |
| [live sessions](live-sessions.md) | live state, the process watcher, peers, tasks, control channels |
| [agent context](agent-context.md) | `brief` and its forms, budgets, deliveries and advisories |
| [session control](session-control.md) | `send`, `answer`, `interrupt`, `resume`, `read --follow`, `start`, roles, hosted sessions, notices to sessions, delegation |
| [web](web.md) | the workbench frame, keys, inspector, live behaviour, access and routes |
| [web designs](web-design.md) | visual baselines, artboards and views to draw |
| [web reader](web-reader.md) | session reading and control, reports, sharing, documents and messages |
| [web pages](web-pages.md) | the page list and page view, citing while reading, editing in place, proposals, history |
| [web views](web-views.md) | home, running, search, notes, comparisons, judging passes, pages and workspaces, timelines, cost, analysis, status and settings |
| [operations](operations.md) | the daemon, storage failure, idle work, `status` and `--verify`, notices, the change feed |
| [installing](installing.md) | `install`, configuration and `set`, `embed`, `demo`, several machines, deployment, upgrades, backup and restore |
| [primary mode](primary-mode.md) | Polylogue on Sinex: one record in both modes, bytes kept in Sinex's store, the record shipped, the reader of Sinex's log, several machines, and the move to it |
| [testing](testing.md) | test archives, properties tested, fault and crash tests, fuzzing, benchmarks |
| [development](development.md) | building, the dev shell, xtask, CI, the deployment modules |
| [vocabulary](vocabulary.md) | each concept and its one name, and the removed words |
| [set aside](set-aside.md) | words whose job another line does, and what the operator ruled out |

The shared mechanics (blobs and segments, the query grammar, the operation and
failure types, the web shell, signals) are documented in
[polylogue-sinex-common](https://github.com/Sinity/polylogue-sinex-common),
one README per crate. Sinex's pages are in
[Sinex's docs](https://github.com/Sinity/sinex/blob/master/docs/README.md).

## Conventions

### Design rules

Each page states the decided behaviour in its own words. Task history and
source-session references stay in Beads. Public documentation contains no
private session excerpts or identifiers. A rule that rests on a default the
operator may still change says "(default; open operator question)".

### Generated artifacts

Generated artifacts are changed through their declarations, not by hand.

### Links

Inside the repository, links are relative. Links into Sinex or
polylogue-sinex-common are absolute GitHub URLs on `master`. Headings linked
from another repository keep their wording.
