# Polylogue documentation

These pages are Polylogue's design: its behaviour, model, formats, and
the reasons for them. They describe the product as designed. What exists in a
given build is described by CLI help, MCP tool schemas and module
documentation. How work is done in this repository is in
[`AGENTS.md`](../AGENTS.md).

Two current-build artifacts also live here: the embedded
[agent manual](agent-manual.md), the card MCP sends as its instructions, to
which `polylogue manual` appends the [agent guide](agent-guide.md) and the
generated operation reference, and the generated [derived-data reference](derived.md). Their current
locations are inputs to the runtime and its tests.

## Pages

| Page | Covers |
|---|---|
| [architecture](architecture.md) | purpose, principles, the two modes, data flow, the stores, durability, the pipeline and its targets, configuration and clocks, the store boundary, deletion |
| [capture](capture.md) | where inputs come from: locations, origins, hooks, tool databases, import, AI Studio through Drive, capture status |
| [extension](extension.md) | the browser extension: live capture, backfill, its own UI, chat actions |
| [archive](archive.md) | stored inputs, versioned rows, evolution, parsers and reading versions, goldens, field accounting, format drift |
| [sessions](sessions.md) | the parsed model: sessions, messages, message authorship, times, session links, compaction, attachments, logical sessions |
| [references](references.md) | the one reference grammar and its rules |
| [assertions](assertions.md) | everything authored: assertions, kinds, current revisions, authorship, judgment, variants, markup, memory |
| [pages](pages.md) | Pages: layouts, members and roles, agents working through pages, ping records |
| [query](query.md) | Polylogue's units and fields, query objects, paging, deadlines, answers, watches |
| [search](search.md) | full-text search, semantic ranking, fusion, recall |
| [reading](reading.md) | `read`, composition, views, budgets, formats, writing out, rendering for models |
| [measures](measures.md) | derived data and measures, and the catalog of derived products |
| [agents](agents.md) | the agent contract, trust rendering, manual, MCP and live sessions |
| [agent context](agent-context.md) | packs, budgets, brief, takeover, recall, deliveries and advisories |
| [session control](session-control.md) | resume, live-session control, pings, notices, hand-off and delegation |
| [interfaces](interfaces.md) | the command table, grammar, latency, warm state and exit codes |
| [completion](completion.md) | shell and query completion, the catalog and shell registration |
| [terminal](terminal.md) | picker, row addressing, output and help |
| [web](web.md) | the workbench frame, keys, inspector, live behaviour, access and routes |
| [web designs](web-design.md) | visual baselines, artboards and views to draw |
| [web reader](web-reader.md) | session reading and control, reports, sharing, documents and messages |
| [web views](web-views.md) | home, running, search, notes, Pages, workspaces, comparisons, timelines, cost, analysis, status and settings |
| [operations](operations.md) | the daemon, status, notices, configuration, deployment, backup |
| [primary mode](primary-mode.md) | Polylogue on Sinex |
| [testing](testing.md) | test archives, properties tested, fault and crash tests, fuzzing, benchmarks |
| [development](development.md) | building, the dev shell, xtask, CI, the deployment modules |
| [vocabulary](vocabulary.md) | each concept and its one name |
| [set aside](set-aside.md) | dropped writers and the format rules worth keeping |

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
