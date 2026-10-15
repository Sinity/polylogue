# Polylogue spec and build plan

Working directory for the rewrite of Polylogue in Rust: what the product is,
how it should behave, and how to build it one verified piece at a time.

Status (2026-10-13): the layered spec, examples, prototypes and composition skeleton are available. The next delivery work follows the [joint realization plan](../polylogue-sinex-joint/REALIZATION-PLAN.md): preserve the current repository implementation, prepare a fresh executable bootstrap, then deliver independently useful vertical slices. Slice packets now follow that route; bootstrap adequacy and actual interfaces must be established before dispatch; a compiling skeleton is not a runnable product. Product questions remain in [questions.md](questions.md).
Progress is in [LOG.md](LOG.md). The deterministic specification and prototype checks are declared in the root flake
(`nix flake check git+file:///realm/inbox/spec`): the Morrow facts and their
builder, the query examples and composition laws, the law index, the vision
catalog, the outline, the skeleton and its types, the inventory of the
Python version, the visual tokens, and the prototype's tests.

## Reading order

0. [DIGEST.md](DIGEST.md): the 20-minute version.
1. [spec/purpose.md](spec/purpose.md): what Polylogue is for and what is out of scope.
2. [spec/principles.md](spec/principles.md): the rules that decide design questions.
3. [spec/architecture.md](spec/architecture.md): the system on one page, and one session traced end to end.
4. [spec/workflows.md](spec/workflows.md): the real uses, walked through with commands.
5. [vision/README.md](vision/README.md): what grows on the substrate, and which uses justify it;
   [vision/synthesis.md](vision/synthesis.md): the uses that justify slices 1–4, who meets them, and what real data has and has not shown.
6. [outline/README.md](outline/README.md): the whole system as components, each placed in a slice.
7. [plan/slices.md](plan/slices.md), [plan/slice-01.md](plan/slice-01.md) and `plan/slice-01-starts-here.md`: the build order, the first slice, and where a builder starts.
8. Then the reference files for the area you work on.

## Joint design with Sinex

The mechanics of running Polylogue on Sinex, and the code the two share,
were agreed with the Sinex session in `/realm/inbox/spec/polylogue-sinex-joint/`
(`decisions.md`, cited as joint #N). The Sinex spec is in
`/realm/inbox/spec/sinex-spec/`.

## Files

| File | What it holds |
|---|---|
| `DIGEST.md` | the short version |
| `questions.md` | the operator's one open question (6 and 7, order of work), and where the 44 settled ones were recorded |
| `LOG.md` | timestamped log of how the spec was built |
| **spec/** | |
| `spec/purpose.md` | purpose, users, jobs, non-goals, related projects |
| `spec/principles.md` | the design rules |
| `spec/architecture.md` | the system on one page; one session end to end |
| `spec/workflows.md` | real uses with commands and outputs, and the real needs by frequency |
| `spec/model.md` | entities, identity rules, worked example |
| `spec/sources/*.md` | every provider: locations, formats, mapping tables, merging (split by provider) |
| `spec/storage/*.md` | the blob store and the three SQLite files, with DDL |
| `spec/convergence.md` | the single update loop |
| `spec/query.md` | the query language: grammar, units, fields, examples |
| `spec/surfaces/*.md` | CLI, MCP and hooks, web reader, extension and clients (split by surface) |
| `spec/rendering.md` | views, message and block rendering, ranges, formats |
| `spec/export-format.md` | the one JSON shape for a session document |
| `spec/analysis.md` | usage and cost, tool outcomes, time, drift, schema inference, embeddings |
| `spec/user-data.md` | notes, markers, saved queries, briefings, resume, judgment, pages |
| `spec/behavior.md` | speed targets, memory, failures, privacy, observability |
| `spec/presentation.md` | how Polylogue is explained to someone new: README, demo, first outside user |
| `spec/glossary.md` | the terms used in a specific sense |
| **decisions/** | |
| `decisions/README.md` | index of every decision, who made it and when |
| `decisions/*.md` | raw retention, storage files, process, lineage, one message per response, merging, schema validation, query parser, repository, Sinex |
| **plan/** | |
| `plan/method.md` | how slices are built: design, review, parallel work, verification |
| `plan/slices.md` | the slices in order |
| `plan/slice-01.md`, `plan/slice-01-prompts.md` | slice 1 in full, and its worker prompts |
| `plan/testing.md`, `plan/testing/laws.md`, `plan/testing/real-data-cases.md` | test layers and the example archive; every law, numbered; real-data cases |
| `plan/beads.md` | pointer to the Beads switchover table in `evidence/beads/` |
| `plan/rewrite-rules.md` | one rule per past bug class, each with its enforcement |
| `plan/lessons.md` | process mistakes not to repeat |
| `plan/workspace.md` | Rust crates, dependencies, lints, xtask, Nix |
| `plan/migration.md`, `plan/compare-old.md` | moving from the Python version; comparing parsers on real files |
| `plan/AGENTS.md.draft` | the rules file for the new repository |
| **examples/** | the Morrow example archive (`archive/`, with FACTS.md and a checker), scrubbed real excerpts (`raw/`) |
| `examples/prices.toml` | the example price table |
| `examples/archive/expected/` | slice outputs the checker verifies |
| **prototype/** | the executable slice-1 skeleton: the per-record parse framework, a Claude Code parser with incremental carry, a writer into the spec's `index.db` schema; its tests check the Morrow slice-1 counts, record accounting (law 120) and append equivalence at 3,234 split points (`prototype/README.md`; the flake check `polylogue-prototype`) |
| **prototypes/ingest-bench/** | the first-build benchmark on the operator's real Claude Code and Codex data; results in `evidence/ingest-benchmark.md` |
| **SET-ASIDE.md** | features and designs not built now, each with what would bring it back; nothing there is rejected |
| **prototype-probes/** | throwaway parser probes, evidence only |
| **mockups/** | older static drafts (the design canvas supersedes them) |
| **audit/** | the whole-codebase audit of the Python version: per-unit reports, verifications, and the summaries (`SUMMARY.md`, `spec-additions.md`, `rewrite-rules-additions.md`, `bloat-patterns.md`, `naming.md`, and their `-rerun` counterparts) |
| **evidence/** | reference material, not part of the spec: the operator's own words by period, the Beads reconciliation (`evidence/beads/`, including the switchover table), real-data reports, reviews, research, ideas, and `integration-status.md` |

## How evidence is weighed

The operator's views changed over the year. When statements conflict, the
most recent deliberate ruling wins; earlier ones explain how it got there.
Corrections the operator gives while this spec is being written override
everything else. The operator's own words outrank agent-written designs,
which are used only as a source of details. Where the spec authors settled
a question from the definitions and evidence, the record says so, and the
operator may overrule it.

## Sources used

- The operator's own messages: Claude Code and Codex sessions in Polylogue
  checkouts (Dec 2025 – Oct 2026), ChatGPT, claude.ai and AI Studio threads
  about Polylogue, journal entries.
- The current Python codebase (`/realm/project/polylogue/repo`), its docs
  and all 4,085 of its Beads.
- Earlier agent-written designs, used for details, not as a template.
- Real provider data on this machine and the pre-reset archive.
- Sinex, Lynchpin and Sinnix, for integration and code layout.
