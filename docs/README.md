# Polylogue documentation

These pages are Polylogue's design: its behaviour, model, formats and laws, and
the reasons for them. They describe the product as designed. What exists in a
given build is described by CLI help, MCP tool schemas and module
documentation. The [reference index](reference/README.md) distinguishes
current artifacts from planned generated pages; the difference is work,
tracked outside the repository. How work is done in this repository is in
[`AGENTS.md`](../AGENTS.md).

Two current-build artifacts also live here: the embedded
[agent manual](agent-manual.md), whose command reference is appended at run
time, and the generated [derived-data reference](derived.md). Their current
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
| [assertions](assertions.md) | everything authored: assertions, kinds, currency, authorship, judgment, variants, markup, memory |
| [pages](pages.md) | Pages: layouts, members and roles, agents working through pages, ping records |
| [query](query.md) | Polylogue's units and fields, query objects, paging, deadlines, answers, watches |
| [search](search.md) | full-text search, semantic ranking, fusion, recall |
| [reading](reading.md) | `read`, composition, views, budgets, formats, writing out, rendering for models |
| [measures](measures.md) | derived data and measures, and the catalog of derived products |
| [agents](agents.md) | the agent contract: brief, packs, takeover, deliveries, MCP, live sessions, pings |
| [interfaces](interfaces.md) | the command table and its transports, the CLI, exit codes |
| [web](web.md) | the web workbench and its visual baselines |
| [operations](operations.md) | the daemon, status, notices, configuration, deployment, backup |
| [primary mode](primary-mode.md) | Polylogue on Sinex |
| [testing](testing.md) | test archives, law suites, fault and crash tests, fuzzing, benchmarks |
| [vocabulary](vocabulary.md) | each concept and its one name |

The shared mechanics (blobs and segments, the query grammar, the operation and
failure types, the web shell, signals) are documented in
[polylogue-sinex-common](https://github.com/Sinity/polylogue-sinex-common),
one README per crate. Sinex's pages are in
[Sinex's docs](https://github.com/Sinity/sinex/blob/master/docs/README.md).

## Conventions

### The operator's words

Only the operator's own words are labelled. Everything unlabelled is the
design. A ruling is cited once, on the page that owns the rule it binds, after
the sentence that states the rule:

> **Polylogue has no deletion.** Ruling, 2026-10-28: "probably just do not
> have deletion in the first place in polylogue, for now"
> (cc-queued:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f@2026-10-27T23:14:04Z).

The date is the operator's local date; a locator's time is UTC. The quote is
verbatim, with the operator's spelling, cut with "…", normally under thirty
words. When the design sentence already says everything, the citation is
`(ruling DATE, LOCATOR)` with no quote. Only
the latest deliberate statement is cited. A rule that rests on a default the
operator may still change says "(default; open operator question)".

### Locators

Every locator resolves in a primary source.

| Locator | Resolves in |
|---|---|
| `cc:<session>:<message>` | a Claude Code user turn: the transcript `<session>.jsonl`, the message's `uuid`; in Polylogue, `polylogue read message:claude-code:<session>:<message>` |
| `cc-queued:<session>@<UTC time>` | a message typed while the agent was busy: the `queue-operation` line of that transcript at that time |
| `cc-reject:<session>:<message>` | text typed when rejecting a tool call, in that transcript |
| `codex:<session>@<UTC time>` | a Codex rollout |
| `chatgpt:<conversation>:<message>`, `claude.ai:<conversation>:<message>`, `aistudio:<prompt file>:<turn>` | the provider's export, and the Polylogue archive |
| `journal YYYY-MM-DD HH:MM` | the operator's journal, by entry time |
| `notes:<file>` | the operator's personal notes, by file name |
| `polylogue@<commit>:<path>`, `sinex@<commit>:<path>` | the predecessor's code in the repository's history |

Sinex's and polylogue-sinex-common's documentation use the same conventions
and link here.

### Laws

Every area page ends with `## Laws`. Each law is one bullet: a bold key, then
the property and how it is checked. A law's full name is `<page>/<key>`
(`archive/re-parse-keeps-every-version`). Keys are plain words and stay stable;
there are no law numbers.

A test that checks a law carries the key in its name
(`re_parse_keeps_every_version`, with a suffix when several tests check one
law). Finding a law's tests is a name search. `cargo xtask laws` lists each
law and the tests whose names carry its key, and the laws with none; it fails
nothing. Laws of the shared query language live in the `query-language` crate
and run against Polylogue's catalog too.

### Generated pages

The [planned pages](reference/README.md#planned-pages) will be rendered from
the declarations by `cargo xtask docs`, committed and checked as goldens.
That generator is not implemented yet. Until a reference generator exists,
its table stays in the area page that owns the subject. Existing generated
artifacts are changed through their declarations, not by hand.

### Links

Inside the repository, links are relative. Links into Sinex or
polylogue-sinex-common are absolute GitHub URLs on `master`. Headings linked
from another repository keep their wording.
