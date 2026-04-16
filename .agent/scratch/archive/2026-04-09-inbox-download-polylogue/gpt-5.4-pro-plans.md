I looked much closer, and the picture is clearer now.

The first important point is that Polylogue is not large because of the obvious “extra” surfaces. By runtime code mass, the biggest subsystems are `storage` at about 16.1k code-ish lines, `schemas` at about 9.6k, `cli` at about 6.9k, `lib` at about 6.8k, `pipeline` at about 6.0k, and `sources` at about 5.3k. By comparison, `mcp` is about 1.1k, `site` about 1.6k, and the actual `ui/tui` code is only about 300. So the repo feels mystery-house-like mostly because the archive substrate and the stewardship layer got big, not because MCP or the TUI exploded.

In a rough manual clustering of the runtime package, I get something like this: about 39k code-ish lines in the archive substrate, about 11.9k in assurance/stewardship, about 5.5k in derived products, and about 5k in access/publication adapters. That is already a strong clue that the project’s real center of gravity is not “viewer for chat logs.” It is closer to “trustworthy local archive platform for AI conversations.”

That matters for judging coherence. A lot of the things that can look like sprawl actually do belong together, but they are currently named and packaged in a way that hides the logic.

The most coherent interpretation of Polylogue is this: there is a raw heterogeneous corpus coming from providers; that corpus gets normalized into a local archive; then the archive produces derived read models; then multiple surfaces expose those models; and a stewardship layer tries to prove the whole thing is valid enough to trust. If you read the repo that way, a lot of the “why is this here?” questions get easier.

The session-product and analytics work is genuinely integral under that interpretation. The repo does not treat them like random add-ons. It has versioned public contracts in `archive_products.py`, persistent materialized tables like `session_profiles`, `session_work_events`, `session_phases`, `work_threads`, `session_tag_rollups`, and `day_session_summaries`, plus FTS indexes and provenance/version fields in the DDL. The profile rows themselves carry evidence, inference, and enrichment payloads. That is not ad hoc tagging. It is a real read-model system over the archive. If Polylogue were only “ingest and grep,” then this would be sprawl. If Polylogue is “archive plus archive intelligence,” then this is core.

MCP is also genuinely integral, and structurally it is one of the cleaner parts. It mostly sits on top of `ArchiveOperations` and exposes the archive to external agents. That is directly on-mission. It behaves like a proper adapter.

The site builder is more ambiguous. Conceptually it belongs: publishing and browsing an archive is a natural surface. But structurally it is less coherent than MCP because it imports `ConversationRepository` and `SQLiteBackend` directly in places like `site/builder.py`, `site/scan.py`, and `site/conversation_pages.py`. So it acts less like a leaf adapter and more like its own parallel app. I would not call it false to the project, but I would call it boundary-leaking.

The TUI is the opposite. It is genuinely somewhat sprawl-like, but it is small. The actual `ui/tui` runtime code is tiny relative to the repo. So the issue is not that it made the project huge. The issue is that it adds another bespoke interaction surface, with its own state and test burden, for something that is not obviously central. If you want a very crisp project identity, the TUI is a good candidate to demote to optional or plugin-like status.

`showcase` is one of the best examples of “integral but badly framed.” Functionally it is not a showcase at all. It is an acceptance harness. It seeds workspaces, generates fixtures, runs the real CLI through a boundary, checks invariants, and produces QA reports. `qa_runner_workflow.py` even gates later stages on schema audit results. That is stewardship and acceptance testing, not a demo layer. So I would not cut it. I would rename and reposition it.

The `schemas` package is also mostly integral, but too omnibus. Right now it bundles at least four different concerns: provider/wire-format contracts and harmonization, semantic and relational inference, verification/proof workflows, and synthetic corpus generation. Those all relate to trustworthiness, but the package is trying to be too many things under one heading. That makes a coherent stewardship layer look like random adjacent machinery.

The part that most harms coherence is not feature count. It is concept dispersion. The clearest example is “products.” The `products` package is only the registry and generic transport layer. The registry docstring more or less says so: rendering and dispatch are generic there, but product semantics remain elsewhere. And indeed they are spread across `archive_products.py`, `archive_product_rollups.py`, `archive_product_summaries.py`, `facade_products.py`, repository/product read paths, and `storage/session_product_*`. So one of the project’s most important concepts does not have one real home. That makes an integral feature feel like sprawl.

So if I classify the current features by coherence:

Integral, and probably should remain first-class: sources, pipeline, storage, query/runtime services, derived session/archive products, schema/verification machinery, MCP.

Coherent but should be treated as leaf surfaces rather than centers of gravity: site generation, rendering.

Small but genuinely optional/sprawl-like: TUI.

Integral in function but misnamed and therefore confusing: showcase.

Stewardship tooling that belongs in the project but should be clearly marked as such: validation lanes, QA workflows, mutation/benchmark campaigns, operator schema tools.

The best way to make this more coherent is not mostly deleting code. It is to make the architecture admit what the project has become.

I think the right center of gravity is: Polylogue is a local archive substrate plus derived read models plus stewardship. In other words, not “chat export searcher,” and not “grab bag of AI workflow tools,” but “evidence-bearing archive platform.”

If you adopt that explicitly, the codebase wants to be organized into four rings.

First, the archive substrate: source acquisition, parsing, normalization, storage, repository/query, and runtime services.

Second, derived read models: profiles, work events, phases, threads, summaries, debt, analytics, enrichments.

Third, surfaces: CLI, facade, MCP, site, rendering, optional TUI.

Fourth, stewardship: schemas, verification, proof, synthetic corpora, acceptance exercises, validation lanes, mutation/benchmark campaigns.

Right now all four rings exist, but they are braided together in names and imports. The project gets much more coherent if those rings become explicit.

The two strongest concrete refactors would be these.

One, turn derived products into a real subsystem. Move the contracts, materializers, storage/read paths, status, and registry into one package with internal structure, instead of splitting “product semantics” between root files, storage, and a tiny `products/registry.py`. Whether you call it `products` or `readmodels`, it should actually own the concept.

Two, make site and TUI use the same read-service boundary that MCP already uses. MCP already behaves like an adapter over common archive operations. Site and TUI should stop reaching into storage directly. Once they consume the same read-model layer as everything else, they stop looking like side quests.

A third cleanup, more conceptual than mechanical, would be to split “schemas” into distinct schema kinds. Right now that word is overloaded. You really have at least four schema domains: wire/provider schemas, canonical archive schemas, derived-product schemas, and surface/output schemas. Making those explicit would reduce a lot of the current fog.

A fourth one: separate “user-facing archive CLI” from “operator/stewardship CLI.” The root command surface currently mixes ordinary archive use with doctoring, schema management, QA, and publication. That is part of why the project feels like all of its tools are in one garage. Even if you keep one binary, the conceptual split should be sharper.

On `xtask`, I reran the count more carefully, and the answer is no: the earlier impression was not caused by overcounting tests or the exercise catalog. If anything, the more careful recount makes `xtask` look slightly larger.

`xtask/src` alone is about 82,226 physical lines and about 71,669 code-ish lines. `xtask/tests` adds about 7,085 physical and 5,879 code-ish. `xtask/macros` adds about 1,511 physical and 1,303 code-ish. So the total Rust in `xtask` is about 90,905 physical and 78,921 code-ish lines.

The exercise subsystem is only about 4,526 physical and 3,990 code-ish inside `xtask/src/commands/exercise`. So that is not what is making `xtask` huge. The big masses are `xtask/src/commands` at about 26.5k code-ish, `xtask/src/sandbox` at about 14.6k, and `xtask/src/history` at about 8.6k, plus large root modules like `coordinator.rs` and `preflight.rs`. The largest single files are things like `src/history/db.rs`, `src/commands/doctor.rs`, `src/commands/history.rs`, and `src/commands/status.rs`. So `xtask` is genuinely a large control plane, not mostly an exercise catalog in disguise. Non-Rust docs and fixtures are only around 232 KB total, with the largest single one being a snapshot file around 109 KB, so the size is not coming from bulky assets either.

On your main goal—making Polylogue rigorous, non-ad-hoc, and verifiable—I think the right answer is: substantially yes in spirit and tooling, but not yet in the strongest architectural sense.

It is already more rigorous than average by a wide margin. There is strict mypy over `polylogue`, a 90% coverage floor in CI, named validation lanes for different contract surfaces, mutation tooling, schema audit workflows, raw-corpus verification, artifact-proof reporting, synthetic corpus generation, acceptance-style exercise runs, and derived products that carry version and provenance metadata. That is real rigor. This is not a pile of unstructured vibe-coded glue.

But it is not yet “inherently valid” in the stronger sense where the structure itself forces correctness and coherence.

The biggest reason is circularity. A lot of the assurance stack is endogenous. The project defines schemas, uses them to generate synthetic corpora, and then uses those corpora to test schema- and parser-driven behavior. That is useful for regression and closure, but it is not independent evidence. It proves consistency with your own abstractions more than it proves fidelity to reality.

The second reason is that architectural boundaries are not enforced. The site and TUI surfaces still bypass the common operations/read-model layer. So even if each surface is individually tested, the architecture does not yet guarantee that they all consume the same semantics in the same way.

The third reason is concept placement. “Products,” “schemas,” and “showcase” are all important concepts, but they are not yet represented by crisp ownership boundaries. That means you have rigorous components without a fully rigorous conceptual map.

The fourth reason is that the public legibility surface is not itself under proof. Docs and command help are not fully derived and checked as authoritative artifacts, which is why drift was able to happen. If the public map can rot, then a real slice of coherence is still informal.

So my judgment is: Polylogue is already contract-rich, instrumented, and stewarded. It is not yet intrinsically self-validating. You are building the scaffolding for that, but some of the rigor still lives in surrounding control planes rather than in the architecture itself.

The most promising new concepts to push it further are these.

First, make derived products explicit as a projection system. Each projection should declare its source dependencies, keying, contract version, materializer version, storage schema, freshness rule, reconciliation query, and consumer surfaces. Then status, repair, docs, and even CLI/MCP registration can be generated from those definitions. This would turn “products” from scattered code into a real algebra of read models.

Second, define a surface conformance matrix. The same canonical query or product request should be runnable through facade, CLI JSON, MCP, site data endpoints, and the TUI backend, with equivalence checks on IDs, ordering, counts, fields, and provenance. Right now you have mixed-consumer contract testing in spirit; this would make it much more formal.

Third, adopt a three-band evidence model for verification. Synthetic corpora should prove contract closure. Frozen redacted real corpora should prove realism and backward compatibility. Live dogfood should prove operational health and budget behavior. Those three bands should be distinct, and a feature passing one should not be mistaken for passing the others.

Fourth, treat architecture itself as a verified artifact. Add import-boundary tests so that surfaces cannot import storage directly, or assurance code cannot leak into runtime layers where it does not belong. This would move some correctness from “tested after the fact” to “structurally prevented.”

Fifth, make derivations explainable at field level. Your products already have evidence/inference/enrichment payloads; extend that idea into first-class lineage objects so a consumer can ask why a profile, phase, or work event says what it says, and get links back to raw messages, parser families, and inference versions. That would make “verifiable” mean inspectable, not just test-covered.

Sixth, generate the public map. CLI reference, product inventory, schema inventory, dependency graph, and perhaps the architecture overview should be derived from code/registries and checked in CI. That closes an important gap between internal rigor and external coherence.

Seventh, add semantic diff gates. When a schema or materializer version changes, run canonical corpora through old and new versions and produce an explicit diff report of what changed in normalized outputs and derived products. That makes semantic drift visible and governed rather than accidental.

Eighth, model the archive lifecycle as a state machine. Raw imported, normalized, projections stale, projections fresh, embeddings stale, publication stale, etc. Maintenance and repair commands then become state transitions with allowed preconditions. That would make the control plane much less ad hoc.

If I compress all of this into one sentence, it is this: Polylogue already has the pieces of a rigorous archive system, but the next leap is to stop expressing rigor mostly as extra machinery around the project and start expressing it as fewer, stronger, more explicit architectural contracts inside the project.

The encouraging part is that this does not require abandoning what you built. It mostly requires naming the real thing correctly and making the boundaries line up with the reality that is already there.

I went ahead and turned this into concrete artifacts.

The two main ones are [a comprehensive strategy/report memo](sandbox:/mnt/data/polylogue_strategy_and_docs_plan.md) and [a patch that drops in a new agent-docs system for Polylogue](sandbox:/mnt/data/polylogue_agent_docs.patch). I also pulled out the generated files so you can inspect them directly: [new `CLAUDE.md`](sandbox:/mnt/data/polylogue_new_CLAUDE.md) and [generated `AGENTS.md`](sandbox:/mnt/data/polylogue_new_AGENTS.md).

The short version of my recommendation is this:

Polylogue should not use “one giant root agent doc.” Even with big context windows, the better design is a small always-loaded constitution, imported topic modules, path-scoped nested docs, and skills for executable workflows. That matches how Claude Code actually wants `CLAUDE.md` to be used: it treats those files as context rather than hard config, recommends concise/project-focused docs, and explicitly suggests splitting larger instruction sets with imports or scoped rules. Codex, meanwhile, reads `AGENTS.md` before work begins, and its skills are progressive-disclosure tools, which makes a generated-and-committed `AGENTS.md` plus skills a very natural cross-agent setup. ([Claude API Docs][1])

So the system I actually patched in is:

A tiny root `CLAUDE.md` that carries identity and project worldview. Imported memory modules for basics, repo map, workflows, verification, and history. Nested `CLAUDE.md` files in `storage`, `schemas`, `site`, `showcase`, `operations`, and `products`. Skills for “promote exploratory work into a clean branch,” “query the archive through stable surfaces,” and “audit surface drift.” A generated `AGENTS.md` committed to the repo so Codex-style agents see the same project memory Claude gets via transclusion. Claude’s own docs say a project `CLAUDE.md` should cover build/test commands, architecture, coding standards, and workflows, so I kept those in the imported modules rather than stuffing them all into the root file. ([Claude API Docs][1])

One subtle but important thing: I changed the design so `AGENTS.md` is tracked rather than ignored. If `AGENTS.md` is only locally generated and not committed, Codex won’t reliably benefit from it. Since Codex reads `AGENTS.md` before work, that file needs to exist in the actual repo state, not just as a local convenience. ([OpenAI Developers][2])

On your constitution idea specifically: yes, I think that is the right framing for Polylogue. The repo responds better to “what kind of agent are you in this house?” than to “here is a bag of commands.” The right characteristics are archivist, steward, surveyor, and historian. That is in the patch. The reason this works better is that it gives the agent decision principles when the task is ambiguous. Commands help once the path is known; character helps before the path is known.

Your branch-history concern also has a clean answer. I think you want two explicit modes: feature branches for coherent work, and forge branches for exploratory work. In the patch I used `feature/<area>/<slug>` for already-coherent work and `forge/<yyyy-mm-dd>-<topic>` for exploration. The trick is not to pretend exploration is already publication. Let the forge branch be messy but tagged with lightweight structure like `Slice:`, `Intent:`, and `Proof:` trailers. Then when one slice stabilizes, promote it into a fresh feature branch from `origin/master` using a new worktree plus cherry-pick. That preserves the real exploratory record while keeping public history narratively clean. It is much better than trying to retroactively “make the messy branch respectable.”

On Polylogue’s actual product coherence, my view hasn’t changed, but I can state it more tightly now.

The project is most coherent when treated as four rings: archive substrate, derived read models, surfaces, stewardship. The things that are truly integral are the substrate, the derived products, and the stewardship layer. MCP is a valid leaf adapter. The TUI is optional. The site is coherent in purpose but currently too entangled with storage internals. The biggest conceptual weakness is still product/read-model ownership being split across too many places. The next serious architecture move should be to consolidate derived product semantics into a real subsystem, whether that stays `products/` or becomes `readmodels/`. I did not do that invasive code move in this pass because it is high-risk without full end-to-end validation; I stayed on the safer, higher-leverage documentation/agent-memory layer.

Your clarification on schemas matters, and I’m updating my judgment accordingly. The stronger circularity critique was too harsh. The repo does in fact anchor schema generation in external/provider data and then verify against raw corpora. The code reflects that direction. So the real remaining critique is narrower: not “it’s circular,” but “some rigor still lives around the architecture rather than inside enforced architectural boundaries.” In other words, the issue is less about provenance of the schemas and more about concept ownership, surface conformance, docs freshness, and import boundaries.

That leads to the next layer of rigor I’d push for. Not just more tests or more proof machinery, but stronger structural contracts. Generated inventories from registries. Import-boundary tests so site and TUI stop bypassing the intended service/read-model layer. Semantic diff reports when a contract version changes. Field-level provenance on derived data. Docs freshness checks so command/help/doc drift becomes harder. Those are the things that make the system “inherently valid” at a deeper level.

For the README and public docs, I think the repo needs a much more polished front door. The current README is feature-rich, but it still reads like a capabilities list. It should instead do this in order: say what Polylogue is in one sentence, explain why it exists, explain why it is not just grep, show a 90-second quickstart, show the four-ring architecture, then show the main surfaces. The report includes a specific structure and a concrete visual plan.

The visual/doc package I’d build is: a four-panel hero image, a rings architecture diagram, a provider-normalization map, a search-modes mockup, and five short screencasts. The screencasts should be exactly these: synthetic quickstart, real export ingest, cross-provider query, derived products tour, and static-site publish. Each should have a poster PNG, fixed transcript, and reproducible fixture/workspace. That is how you make the repo feel like a product instead of a mystery house.

On UI/UX, the site should be reframed as archive publication, not just “generate static site.” Its primary IA should probably be Home, Search, Providers, Sessions, Threads, Tags, Reports, and Manifest/About. Search should explicitly separate lexical/faceted retrieval from semantic relatedness from product navigation. For a static site, lexical search should be the primary mode, and semantic search should probably appear as precomputed relatedness or “lenses,” not pretend to be a fully live embedding system unless you are running a local server. The current site also duplicates visual tokens that already exist in `ui/theme.py`; I’d unify those so the visual language is one system across CLI-adjacent surfaces and publication pages.

On MCP, I don’t think “MCP everywhere” is the right answer. I’d keep MCP as the external integration surface for tools outside the repo. For in-repo coding agents, the better default is skills plus stable CLI JSON plus the sync/library API. That is more aligned with Polylogue’s library-first intent and avoids forcing agents through an extra network-style protocol when they are already inside the codebase.

On `xtask`, the closer look confirmed the earlier intuition. It is not big because of the exercise catalog. The exercise subtree is comparatively small. The real mass is in the command plane, sandbox, history/analytics, and status/doctor surfaces. So `xtask` is not really “too large task runner”; it is already drifting toward a developer operating system. That is not a bug. It is the correct interpretation of what it has become.

The more ambitious form of `xtask` would lean into that identity. It could become an engineering control plane with first-class workflow state, proof planning, branch/PR/history stewardship, engineering memory, and machine-readable action recommendations. The really interesting future is for it to become a memory-bearing cockpit: source snapshot, docs/agent memory, diagnostics snapshot, runtime state, recent command history, and recommended next actions, all packaged into one machine-consumable artifact. The fact that it already has `xtask docs agents` and `xtask docs snapshot` means it is already halfway there.

So my recommendation stack is:

Land the agent-doc patch first. Then commit the generated `AGENTS.md`. Then rewrite the README around the correct product identity. Then add the visual assets and screencasts. Then tighten the site around publication and stronger shared boundaries. Then do the larger read-model/product consolidation.

The patch I made is safe and ready to inspect. The larger code reorg I’ve left as a design plan rather than pretending I could responsibly land it blind.

[1]: https://docs.anthropic.com/en/docs/claude-code/memory "How Claude remembers your project - Claude Code Docs"
[2]: https://developers.openai.com/codex/guides/agents-md "Custom instructions with AGENTS.md – Codex | OpenAI Developers"

