# Polylogue strategy, agent docs, UX, and xtask notes

## 1. Core judgment

Polylogue becomes much more coherent if it is described as a **trustworthy local archive platform for AI conversations** rather than as a search tool, a static site generator, or an agent playground.

That gives it four explicit rings:

1. **Archive substrate** — ingestion, normalization, storage, query/runtime services.
2. **Derived read models** — profiles, work events, phases, threads, summaries, debt.
3. **Surfaces** — CLI, library/sync API, static publication site, MCP, optional TUI.
4. **Stewardship** — schema generation, raw-corpus verification, QA, proof, synthetic corpora, mutation/benchmark/test infrastructure.

Seen that way:

- the derived products are integral;
- the stewardship layer is integral;
- MCP is a legitimate leaf adapter;
- the TUI is genuinely optional;
- the site is coherent but boundary-leaky;
- `showcase` is integral in function but misnamed;
- the real coherence problem is not feature count, but **concept ownership and surface drift**.

## 2. Replacement coding-agent docs

### 2.1 What the docs system should optimize for

The always-loaded root document should not try to contain the whole project.
It should do three jobs well:

1. lock in project identity and priorities,
2. point the agent at the right boundaries,
3. activate the correct workflow and proof behavior.

The heavy detail should be split into imported and path-scoped documents, with executable workflows moved into skills.

### 2.2 Why not make one gigantic root file?

For Claude Code specifically, the official guidance still favors concise, well-structured `CLAUDE.md` files, with a target of under 200 lines per file and larger projects split using imports, rules, or nested files. Claude loads root/project memories as context, not as hard configuration, so larger files can reduce adherence. citeturn667595view0turn667595view1turn667595view2

For Codex, `AGENTS.md` is read before work begins, and Codex also supports skills with progressive disclosure so the full skill instructions are loaded only when needed. citeturn667595view4turn667595view5turn667595view6

So the best architecture is not “make the one root file huge.” It is:

- a **small always-loaded constitution**,
- **imported topic files**,
- **nested per-subtree guidance**,
- **skills for executable workflows**,
- a **generated-and-committed `AGENTS.md`** for tools that do not understand Claude-style `@path` transclusion.

### 2.3 Recommended file design

Keep these roles separate.

`CLAUDE.md`
: tiny root constitution; the thing that is always present.

`.claude/includes/*.md`
: imported durable memory modules: worldview, basics, repo map, workflows, verification, history.

`polylogue/<subsystem>/CLAUDE.md`
: path-scoped guidance for local work in high-risk or high-ambiguity areas.

`.claude/skills/<name>/SKILL.md`
: executable playbooks for repeated workflows.

`AGENTS.md`
: generated from the same source and committed, so Codex-style agents consume the same project memory.

### 2.4 Constitution-like sections that make sense for Polylogue

A good constitution for this repo should encode character and invariants, not just commands.

Recommended sections:

- **Identity**: archivist, steward, surveyor, historian.
- **Priorities**: canonical meaning, trust, shared boundaries, proof, narrative history.
- **Worldview**: archive substrate + derived read models + surfaces + stewardship.
- **Core vs optional**: what is first-class and what is a leaf.
- **Architectural instincts**: prefer shared semantics, avoid surface-only behavior, name concepts before coding them.
- **Rigor instincts**: distinguish synthetic closure from raw-corpus evidence.
- **History instincts**: public history is part of the product.

### 2.5 What should be a skill, not a rule

Rules are passive context. Skills are active playbooks.

For Polylogue, at least these skills make sense:

- `promote-slice`: convert exploratory work into a clean feature branch / PR narrative.
- `surface-audit`: check for docs drift, parallel semantics, and boundary leaks after surface edits.
- `query-archive`: use CLI JSON or the sync/library API instead of ad-hoc SQL during development.
- `proof-plan`: choose a minimal convincing proof set based on changed paths.
- `refresh-agent-surface`: regenerate `AGENTS.md`, command references, and inventories, then verify no drift.
- `product-boundary-review`: for changes touching products/read models, verify versioning, provenance, freshness, and repair semantics.

The first three are implemented in the patch I produced.

### 2.6 Branch and GitHub workflow that matches exploratory work

Your instinct about exploratory work is right: often you do not know the right branch story until you are already inside the work.

The cleanest answer is to distinguish **exploration** from **publication**.

Use two modes:

- `feature/<area>/<slug>` when the conceptual story is already clear.
- `forge/<yyyy-mm-dd>-<topic>` when the work is exploratory.

Forge branches are where uncertainty is allowed. They should still carry some lightweight structure in commits:

- `Slice: <concept>`
- `Intent: exploratory|structural|fix|docs|cleanup`
- `Proof: <what was run>`

When a slice stabilizes, do not try to "sanitize the messy branch in place". Promote it:

1. keep the forge branch as the exploration record,
2. create a fresh worktree from `origin/master`,
3. create a clean feature branch there,
4. cherry-pick just the slice commits,
5. reorder/split/fixup until the branch tells one story,
6. open a PR,
7. squash merge to `master`.

That gives you three virtues at once:

- you do not lose the real exploratory history,
- your public history stays clean,
- the agent has a sanctioned workflow for uncertainty instead of silently doing the wrong thing.

### 2.7 Extra agent-doc ideas that go beyond the obvious

These are the most promising non-obvious additions.

**A concept inventory.**
One generated page that names the core nouns of the repo and where they live. For example: archive substrate, read model, product, package schema, artifact proof, acceptance harness, publication manifest. A lot of incoherence in Polylogue comes from concept dispersion; this inventory would fight that directly.

**Negative-space rules.**
A short section naming things the agent should *not* do: do not invent site-only semantics, do not query SQLite directly when a stable surface exists, do not leave docs drift after renaming a command.

**Proof receipts.**
For substantial changes, the agent should leave a structured receipt in the PR body or commit trailers: changed concept, boundary moved, proof run, remaining debt.

**A docs freshness gate.**
The public map is currently too easy to rot. Add a skill or CI check that regenerates `AGENTS.md`, command inventories, and maybe a docs surface snapshot, then fails if checked-in docs drift.

**Boundary sentinels.**
Use import-boundary tests or static checks so the site and TUI cannot reach into storage when that would bypass the intended read-service boundary.

## 3. Patch produced

I prepared a concrete patch that does the following:

- replaces the root `CLAUDE.md` with a short constitution-style root;
- adds imported memory modules for basics, worldview, repo map, workflows, verification, and history;
- adds path-scoped `CLAUDE.md` files for `storage`, `schemas`, `site`, `showcase`, `operations`, and `products`;
- adds skills for `promote-slice`, `query-archive`, and `surface-audit`;
- adds `devtools/generate_agents_md.py`;
- starts tracking a generated `AGENTS.md` instead of ignoring it.

This is intentionally a documentation/agent-memory patch rather than an invasive code move.

## 4. README redesign

The current README has useful information, but it still feels like a feature inventory rather than a polished entrance.

The README should do six jobs in order:

1. make the project legible in one screen,
2. provide the correct mental model,
3. show a quick path to visible value,
4. demonstrate why it is better than naive alternatives,
5. map the major surfaces,
6. hand off to deeper docs.

### 4.1 Recommended top-level README structure

1. **Hero**
   - one-sentence value proposition
   - short subtitle with the correct category: local archive substrate + search + derived read models + publication
   - hero image/collage

2. **Why this exists**
   - one paragraph on fragmented vendor exports and why archive normalization matters

3. **Why not just grep?**
   - because grep assumes you already have one normalized corpus and does not know providers, sessions, tools, reasoning blocks, products, or publication

4. **90-second quickstart**
   - minimal install
   - minimal ingest
   - one query
   - one product view
   - one site publish

5. **What Polylogue is**
   - the four-ring architecture diagram

6. **Key capabilities**
   - ingest
   - query and filters
   - derived read models
   - publication site
   - agent-facing surfaces

7. **Examples**
   - not toy snippets; real output fragments and screenshots

8. **Docs map**
   - “start here” links, not a dump of every document

9. **Project status / non-goals**
   - what is stable, what is experimental, what is intentionally local-first

### 4.2 Specific visuals to produce

The repo currently lacks committed images and demos. That hurts first impression.

Recommended assets:

- `docs/assets/hero-4up.png`
  - four-panel hero collage:
    - search results
    - conversation page
    - session profile / product view
    - publication site homepage

- `docs/assets/architecture-rings.svg`
  - the four-ring system diagram

- `docs/assets/provider-map.png`
  - provider support / normalization diagram

- `docs/assets/search-modes.png`
  - lexical vs semantic vs product navigation mockup

- `docs/assets/site-information-architecture.svg`
  - home / search / providers / sessions / threads / tags / reports

### 4.3 Specific screencasts to produce

These should be short and task-oriented. Use the repo’s existing VHS/demo machinery where possible.

1. `docs/demos/01-quickstart.cast`
   - generate a synthetic demo archive
   - run one query
   - show one conversation

2. `docs/demos/02-ingest-real-exports.cast`
   - drop exports in inbox
   - run `polylogue run`
   - show provider counts and new archive stats

3. `docs/demos/03-cross-provider-search.cast`
   - demonstrate filters, provider narrowing, and one semantic retrieval example

4. `docs/demos/04-derived-products.cast`
   - show session profiles, work events, phases, or threads and why they are useful

5. `docs/demos/05-publish-site.cast`
   - build a static site and browse the result

For each screencast, also produce:

- a poster PNG,
- a short caption,
- the exact CLI transcript,
- the source fixture/workspace used.

### 4.4 Deeper docs reorg

Right now the docs surface mixes product docs with planning and historical records.
That should be split clearly.

Recommended top-level docs split:

- `docs/getting-started/`
- `docs/concepts/`
- `docs/guides/`
- `docs/reference/`
- `docs/architecture/`
- `docs/operations/`
- `docs/archive/` (historical planning / old records only)

The current `docs/README.md` should stop trying to be a mixed directory map for product docs, planning, and generated artifacts all at once.

## 5. UI / UX and site generation

### 5.1 What the site should be

The site should not feel like a generic static dump.
It should be framed as **archive publication**: a stable, browsable, legible read-side of the archive.

### 5.2 Information architecture

A coherent publication site for Polylogue would likely have these primary sections:

- Home
- Search
- Providers
- Sessions
- Threads
- Tags
- Reports
- About / Manifest

Where:

- **Search** is the main retrieval surface,
- **Sessions** and **Threads** expose derived read models,
- **Reports** exposes archive-wide rollups and stewardship artifacts,
- **Manifest** explains provenance and build/version facts for the published site.

### 5.3 Conversation page redesign

Current conversation pages are readable, but still fairly plain.

A stronger conversation page would add:

- sticky metadata rail (provider, date, counts, tags, links to products);
- role chips and collapsible tool/reasoning blocks;
- code block copy buttons and anchors;
- related conversations / neighboring thread items;
- breadcrumbs back to provider/session/thread pages;
- explicit disclosure when content has been transformed or redacted.

### 5.4 Search design

Polylogue needs to distinguish three retrieval modes more explicitly.

**Lexical / faceted retrieval**
: titles, previews, tags, provider filters, date ranges, content affordances.

**Semantic relatedness**
: “show me nearby conversations / nearby sessions / similar threads”.

**Product navigation**
: browse not by raw conversation text but by derived entities like session profiles or work events.

For a static site, lexical search is the natural primary mode.
Pagefind or a similar static index makes sense there.

True embedding search is less natural in a purely static publication surface. A better static compromise would be:

- precomputed related items,
- “semantic lenses” pages,
- nearest-neighbor links baked into the build,
- optional local/live search mode if the user runs a local server.

### 5.5 Visual system

The site currently hardcodes many CSS tokens even though `ui/theme.py` already exists.
That creates unnecessary visual drift.

Recommended direction:

- export theme tokens to generated CSS variables,
- keep a restrained neutral palette,
- use provider colors sparingly as accents,
- keep reading width tighter on conversation pages,
- use a consistent card/nav/table language across dashboard, conversation, and list pages.

### 5.6 Crispness heuristics for the whole product

To make the whole thing feel tighter:

- make the archive substrate and read models the visible center,
- demote optional surfaces from the top-line story,
- unify terminology across CLI, docs, and pages,
- ensure every user-visible surface can be explained in one sentence.

## 6. MCP versus skills / CLI / API

MCP is a reasonable external integration surface.
It makes sense when Polylogue is being exposed to other tools like Claude Desktop or a separate agent host.

But MCP should probably not be the primary internal agent surface for developers working *inside this repo*.

For internal coding-agent work, the better default is:

1. **skills** for workflow activation,
2. **stable CLI JSON contracts** for shell-level access,
3. **`SyncPolylogue` / library API** for in-process structured access.

That fits the repo’s library-first intent more naturally.

A useful design principle here is:

- **MCP for outside-in integration**,
- **CLI/API/skills for inside-the-repo work**.

If agent-facing access is awkward today, the answer is likely not “force more MCP”. The answer is to harden a few explicit machine-facing local surfaces.

## 7. Architecture reorg roadmap

I would not start with a giant package rename. I would do it in phases.

### Phase 0 — conceptual cleanup

- Start describing the repo in terms of the four rings.
- Rename `showcase` in docs mentally to acceptance / verification harness.
- Make “derived read model” or “product” a first-class concept in docs.
- Make the public map honest.

### Phase 1 — product / read-model consolidation

The highest-value code reorg is to give derived products a real home.

Today, product semantics are dispersed across:

- `archive_products.py`
- `archive_product_rollups.py`
- `archive_product_summaries.py`
- `operations/`
- `facade_products.py`
- `storage/session_product_*`
- `products/registry.py`

Recommended direction:

- either expand `products/` into the true read-model subsystem,
- or introduce a new `readmodels/` package and migrate product semantics there.

That subsystem should own:

- definitions/contracts,
- materializers,
- storage/status/freshness,
- provenance/versioning,
- registries/inventories,
- repair hooks,
- shared query surfaces.

### Phase 2 — leaf surfaces consume shared boundaries

The next reorg target is the site and TUI.

They should consume operations/read-model boundaries, not storage internals.
That avoids parallel semantics and makes publication/search behavior more trustworthy.

MCP is already closer to the desired adapter shape than the site is.

### Phase 3 — stewardship boundary cleanup

`schemas/` currently bundles several schema kinds together.
Even before moving files, start naming them distinctly:

- provider/wire schemas,
- canonical archive schemas,
- derived-product schemas,
- surface/output schemas.

Eventually, either split them physically or make the substructure much clearer.

### Phase 4 — rigorous structural checks

Add checks that move rigor from “good habits” into enforced boundaries:

- import-boundary tests,
- docs freshness checks,
- semantic diff reports when contract versions change,
- generated inventories from registries,
- field-level provenance / lineage on derived data.

## 8. Polylogue rigor: what is already true, and what still is not

Your clarification is important: the schema pipeline is not purely circular if schema generation is anchored in external/provider data and later used as the reference set. The codebase reflects that direction; `schema_inference.py` explicitly describes generation from real provider samples, and the verification models are framed around raw-corpus verification and artifact proof. So the “self-generated abstraction all the way down” critique was too strong. The stronger critique is narrower: some assurance still proves internal closure more than cross-surface architectural integrity.

What is already true:

- strong test volume and typed discipline,
- schema generation from real data samples,
- raw-corpus verification and artifact proof machinery,
- acceptance/QA workflows,
- mutation/benchmark/property/invariant machinery,
- version/provenance thinking in derived tables.

What is not yet fully true:

- concept ownership is still dispersed,
- surfaces still bypass shared boundaries in places,
- the public docs surface is not itself derived or guarded strongly enough,
- some proofs still validate components in isolation more than the architectural contract between them.

The next level of rigor would come from:

- explicit read-model declarations,
- surface conformance tests,
- import-boundary enforcement,
- lineage/provenance for derived fields,
- generated public inventories,
- semantic diff gates.

## 9. xtask: what it already is

`xtask` is not over-counted because of the exercise catalog.
The exercise subtree is relatively small compared with the overall command/control mass.
The big areas are the command plane, sandbox, history, and the large status/doctor/history surfaces.

Functionally, `xtask` is already five things at once:

1. task runner,
2. environment/runtime control plane,
3. diagnostics and observability cockpit,
4. sandbox/test lab,
5. agent-context and documentation generator.

That means the most coherent way to describe it is not “cargo wrapper”.
It is closer to a **developer operating system** or **engineering control plane**.

That framing explains why it is so large.

## 10. xtask: what it could become in a more ambitious form

The most promising direction is to make that identity explicit rather than pretending it is a humble helper.

### 10.1 A workflow engine, not just a command set

Right now it has many commands, but the higher-level workflow model is still implicit.
It could grow a first-class notion of:

- work item,
- proof plan,
- branch state,
- runtime state,
- incident state,
- publication state.

Then `xtask` would not just *run commands*; it would advance engineering state machines.

### 10.2 A memory-bearing engineering cockpit

The history DB is already valuable. A more ambitious form would make it the brain of the system.

Potential additions:

- recommend the smallest convincing proof set based on changed paths and past failures,
- warn when a developer is about to repeat a flaky or expensive failing workflow,
- summarize recurring debt by subsystem,
- attach command/proof history to PR preparation.

### 10.3 Branch and history stewardship

The same promotion ideas described for Polylogue could be implemented in `xtask` form:

- forge-branch introspection,
- slice clustering by touched paths and commit trailers,
- feature-branch promotion via worktree + cherry-pick,
- PR body generation from proof receipts,
- squash-commit suggestion based on branch story.

### 10.4 Engineering digital twin

The `docs snapshot` and `docs agents` commands already hint at this.
The ambitious version would package:

- source snapshot,
- current docs/agent memory,
- diagnostics snapshot,
- recent command history,
- runtime state,
- recommended next actions,

into one coherent machine-consumable artifact.

### 10.5 Policy engine

The system already has partial agent-facing JSON contracts.
Push that further:

- stable schemas for status/doctor/history output,
- explicit compatibility promises,
- policy bundles for CI, local dev, and agent use,
- machine-readable “why this failed and what to do next” guidance.

### 10.6 Sinex dogfooding

A very ambitious direction would be for `xtask` to emit structured engineering events into Sinex itself, so the workspace literally observes its own development process.
That would let Sinex analyze build/test/infra behavior as one of its own domains.

## 11. Recommended next steps

If I were sequencing this pragmatically, I would do:

1. land the agent-doc patch,
2. regenerate and commit `AGENTS.md`,
3. rewrite the README around the correct product identity,
4. add the visual assets / screencasts,
5. make the site consume stronger shared read-model boundaries,
6. consolidate product/read-model ownership,
7. add docs freshness and boundary enforcement checks.

That gets the repo more coherent before attempting an invasive package move.
