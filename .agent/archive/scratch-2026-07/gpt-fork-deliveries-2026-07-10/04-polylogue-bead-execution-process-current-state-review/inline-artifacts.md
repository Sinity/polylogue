## Turn 56 (role=assistant) — answer to 'describe all beads in order they should be executed, at the end, describe what precisely is the end state of the system'

I treated “beads” as the Beads issue records in the archive. Closed beads are already executed, so the execution plan covers the 397 still-active beads: 396 open plus 1 in progress. I generated a full ordered execution table with every bead, its priority/type/status, parent bead, hard prerequisites, intent, and done state:

[Full Beads execution order — Markdown](sandbox:/mnt/data/polylogue_beads_execution_order.md)  
[Full Beads execution order — CSV](sandbox:/mnt/data/polylogue_beads_execution_order.csv)

The ordering rule I used is: finish hard `blocks` dependencies first, inherit blocker state through parent-child hierarchy, execute children before closing parent/epic beads, then break ties by in-progress state, priority, type, and ID. Informational relationships like `related`, `relates-to`, `discovered-from`, and `supersedes` are not execution blockers.

The first execution run begins with the active/deploy-gating and highest-risk P1 beads:

`polylogue-s7ae.6` → classify the aborted full verify before coordination deploy.  
`polylogue-8jg9.4` → fix the ops-doctor blob-cleanup race.  
`polylogue-9e5.28` → make rigor audit iterate the product registry, not only declared contracts.  
`polylogue-9e5.29` → prevent numeric zero claims over absent evidence.  
`polylogue-9e5.30` → tag prose-mined forensic fields as text-derived.  
`polylogue-cpf.5` → propagate weakest temporal provenance through aggregates.  
`polylogue-cpf.6` → add a clock seam and audit `sort_key_ms` handling.  
`polylogue-kwsb.1` → harden daemon/capture Host, Origin, token, and spool controls.  
`polylogue-37t.15` → force non-user assertion writes through candidate/non-injected policy.  
`polylogue-83u.4` → classify missing referenced blobs in the production backup.  
`polylogue-cfk` → rerun the uplift experiment with fresh handoff packs.  
`polylogue-0v9p`, `polylogue-arso`, `polylogue-rlsb`, `polylogue-d4zk` → land the content-variant substrate and surfaces.  
`polylogue-d1y`, `polylogue-pj8`, `polylogue-s7ae.2`, `polylogue-ahqd`, `polylogue-37t.11`, `polylogue-s7ae.3`, `polylogue-s7ae.5` → install hooks, ship coordination MCP/cookbook surfaces, scheduler, coordination messages, adoption proof, and two-agent live proof.

After that, the plan proceeds through scale-hardening, lineage, query/read algebra, interactive performance, audit/provenance, web cockpit, analytics, interop/export, and frontier/vision features. The full artifact contains all 397 beads in exact order.

The precise end state after executing the whole schedule is this:

Polylogue has no remaining open/in-progress executable beads from this snapshot. All child beads are closed before their parent/epic closures, and no blocked Beads remain blocked in this tech-tree slice.

The archive substrate is scale-hardened. Blob cleanup cannot delete leased or in-flight blobs; production-backup blob debt is classified, restored where recoverable, or explicitly accepted; FTS/search/read-model drift is measured; large archive rebuilds and schema bumps use safe blue-green derived-tier replacement instead of long degraded windows.

Durable and derived storage tiers obey their intended regimes. `source.db` and `user.db` evolve through additive migrations with backup-manifest gates. `index.db` and `embeddings.db` rebuild or blue-green swap from durable evidence. `ops.db` remains disposable operational state.

Evidence honesty is enforced at the field level. Numeric fields distinguish absent evidence, true zero, and not applicable. Temporal aggregates propagate the weakest timestamp source. Prose-mined forensic facts carry text-derived caveats. The rigor audit covers every registered insight product or records an explicit exemption.

Agent coordination is operational. Hooks are installable and health-checked. MCP roles and prompts are discoverable. Agents can write benign, attributed records through the correct role. Candidate assertions go through one safe chokepoint. A context scheduler controls what enters agent context. A reproducible proof shows two agents in separate worktrees on one repo detecting overlap, exchanging a scoped message, receiving context injection, and producing a handoff packet.

Content variants are first-class archive objects. Language detection, preference facts, variant refs/nodes/alignment/storage, variant-aware projection/query/render profiles, and UX for creating/reviewing/messaging about variants all compose with the archive.

Session lineage is truthful and rebuild-stable. Shared content is stored once and counted once. Compaction boundaries, effective context, subagent/main-session distinctions, transcript completeness, physical identity collisions, compaction loss, and regrounding surfaces are represented explicitly.

The read surface is unified. CLI, daemon, MCP, Python API, web UI, and generated docs use the same Query × Projection × Render contract instead of parallel behavior. Surface-side math and duplicate dispatch paths are either deleted or routed through contracts.

Interactive performance is productized. CLI startup is light, common queries route through a hot daemon path, daemon caching/warming and push channels support live UIs, bulk ingest has bounded resource envelopes, and SLO/regression gates measure latency continuously.

The web workbench becomes an evidence cockpit. It supports evidence baskets, citable reports, verified exports, session replay, timeline/firehose views, daily narratives, live tailing, analytics views, long-session navigation, and citation anchors on top of the same evidence contracts.

Analytics answer “so what” questions without overclaiming. Work-graph units, outcome/pathology measures, temporal trends, process mining, survival analysis, prediction/calibration, experiment hosting, information-theoretic measures, trajectory-quality scoring, and model-drift observation are available with uncertainty and construct-validity limits.

Interop and export breadth expand. OriginSpec owns ingestion strictness and dispatch. Additional origins and trace formats become importable where scheduled. Polylogue export/import is content-hash idempotent. Outbound citation/export lanes exist for git notes, PR/issue citations, SARIF-style exports, PKM profiles, federation, and research/eval formats.

External legibility is closed. A stranger can understand the product from the first screen of the README, run the one-command demo, inspect proof artifacts, verify claims against a public claims ledger, and cite published findings.

---

## Turn 114 (role=assistant) — answer to 'assess the bead-set-implied plan's quality, completeness, readiness for implementation'

Overall: the bead-set-implied plan is a strong architectural backlog and a credible dependency-aware execution map. It is not yet a fully implementation-ready delivery plan. Treat it as a complete inventory plus a safe topological ordering, not as a sprint schedule or release plan.

My scorecard:

| Dimension | Assessment |
| --- | --- |
| Backlog coverage | High. It includes all 397 non-closed executable issue beads from the snapshot. |
| Dependency hygiene | Good. I found no missing dependency references and no hard dependency cycles. |
| Architectural coherence | High. The work clusters around real product doctrines: evidence honesty, storage tier separation, provenance, query/render unification, agent coordination, lineage truth, interop, and external legibility. |
| Implementation readiness | Medium. P1/P2 safety and bug beads are mostly actionable; later P3/P4 feature beads are often roadmap/spec material. |
| Detail quality | Uneven but above average. Many beads have intent, design, and acceptance criteria; 72 active beads still lack acceptance criteria. |
| Scheduling quality | Weak-to-medium. Too many beads are “ready now,” so dependency ordering alone does not produce a useful delivery sequence. |
| End-state precision | Medium. The end state is coherent, but several clauses are product-vision statements rather than directly verifiable completion conditions. |

The biggest strength is that the bead set is not a generic wishlist. It is evidence-shaped. Many items point to real observed failures, named source files/modules, explicit verification expectations, and product invariants. For example, the first P1 items are not vague “improve reliability” work; they identify concrete hazards like unsafe blob cleanup, numeric claims over absent evidence, temporal provenance laundering, unsafe agent assertion promotion, missing production blobs, and an aborted full verification run. That is implementation-friendly.

The second major strength is thematic completeness. The plan covers the full system lifecycle: ingest, capture, storage, blobs, derived indexes, embeddings, assertions, preferences, query algebra, rendering, daemon, CLI, MCP, web UI, docs, demos, release proof, analytics, interop, and external publication. It also treats “truthfulness of evidence” as a cross-cutting product constraint rather than a documentation afterthought. That gives the plan a real spine.

The dependency graph itself is clean, but under-constraining. There are 623 exported dependency edges, including 125 `blocks` edges and 362 parent-child edges. After filtering to active work, the hard-blocker graph leaves 314 of 397 beads in wave 0. That means roughly 79% of the active backlog appears executable immediately. In practice, that is not true from an engineering-management perspective. It means the graph captures explicit blockers, but not enough architectural, migration, release, ownership, or sequencing constraints.

That is the most important flaw: the plan has false serial precision. A numbered order from 1 to 397 looks like “do this, then this, then this,” but most early ordering among unrelated wave-0 beads comes from priority/type/id tie-breaking, not real causality. The ordering is safe as a deterministic queue. It is not strong evidence that, say, every wave-0 web, query, analytics, content-variant, or interop task should actually be started before later blocked work unlocks.

The P1 layer is the most ready. All 32 P1 active beads have acceptance criteria. That part can be implemented almost directly, especially the stop-the-line work: classify the aborted verify, fix unsafe blob cleanup, harden daemon/capture security, prevent unsafe agent assertion activation, classify/restore missing blob debt, and repair rigor/provenance/numeric/temporal honesty gaps. This is the best-defined slice of the plan.

The P2 layer is mostly ready but needs local inspection before coding. There are 181 P2 beads; only 5 lack acceptance criteria. Many P2 items include enough design detail to start, but they span schema, derived models, UI, daemon paths, query contracts, release gates, and migration behavior. These should be batched into capability releases, not executed as isolated tickets.

The P3/P4 layer is substantially less ready. Among P3 beads, 60 of 166 lack acceptance criteria. Among P4 beads, 7 of 18 lack acceptance criteria. That is normal for horizon/backlog material, but it means the plan’s long tail is not implementation-ready. Those beads need design expansion, dependency normalization, test strategy, and milestone assignment before they become executable.

The bead detail is generally strong. Across the 397 active beads, 359 have descriptions, 363 have design text, and 325 have acceptance criteria. Around 80% mention verification/testing somewhere, and around 56% have explicit verify/test language in acceptance criteria. That is much better than a typical issue tracker. The detail weakness is not absence; it is inconsistency. Some beads are crisp enough to code; others are architectural slogans with partial execution shape.

There are some backlog-hygiene issues worth fixing before serious execution. The Beads summary has inconsistent readiness signals: the snapshot summary says 317 ready issues, while `ready_ids` contains 100, and blocked counts appear as 79 in one place and 83 after blocker propagation. That does not invalidate the plan, but it means readiness should be regenerated and linted before using it as an execution source of truth. Also, one open epic lacks acceptance criteria, and at least one open epic-like item has no open children, which suggests a few hierarchy/type cleanups are needed.

Completeness is strong at product-scope level and weaker at delivery-plan level. The bead set covers what the system should become, but it does not fully specify how the work is released. Missing delivery-layer information includes owners, estimates, batch size, branch/PR boundaries, migration order, rollback strategy, compatibility matrix, demo/proof cut lines, release gates, and “do not start before X stabilizes” soft dependencies. Those are not usually stored in a bead title/description, but they are necessary for execution.

The plan also does not fully integrate the 22 Beads memory records into the execution order. Some memories encode process invariants, such as bead field-update discipline, automatic convergence expectations, and backlog linting. Those are not executable beads, but they should affect implementation. Ignoring them risks violating project-specific operating rules while still “following the plan.”

The end state is directionally correct but not precise enough as a verification target. Statements like “Polylogue is the local system of record for AI work” and “the web workbench becomes an evidence cockpit” describe the intended product, but they are not directly testable. A precise end state needs a claims ledger mapping each final-state sentence to bead IDs, schema versions, commands, fixtures, proof artifacts, and required passing validation lanes.

The highest-risk areas are schema/migration coupling, derived-tier rebuild correctness, data-loss paths, security boundaries, surface-contract drift, and performance regressions. The plan notices most of these risks, which is good. But because many related items are not hard-linked, an implementer could accidentally build UI or analytics on top of unstable contracts, then pay integration debt later.

The best way to execute this plan is not bead #1 through bead #397. The right execution shape is phased:

First, close operational safety and evidence-honesty gates: verify classification, blob safety, daemon security, assertion safety, missing blob debt, temporal/numeric/provenance rigor, and backup/degraded-state correctness.

Second, stabilize substrate contracts: storage tier separation, migration/rebuild invariants, source/origin specs, query/projection/render contracts, assertion/candidate policy, and schema ownership.

Third, build product surfaces on those contracts: coordination substrate, content variants, lineage truth, MCP/CLI/API convergence, webui v2, evidence baskets, live tailing, reports, and export lanes.

Fourth, expand analytics and interop: work-graph units, outcome measures, process mining, prediction/calibration, external trace formats, federation, PKM/export, and research-tooling formats.

Fifth, close external legibility: README, one-command demo, public claims ledger, install matrix, proof cards, docs site, launch kit, and published findings.

So the verdict is: high-quality backlog, strong architectural plan, medium implementation readiness, and uneven detail. It is ready for immediate execution on P1/P2 safety and correctness work. It is not ready to execute blindly as a 397-step linear plan. Before broad implementation, it needs a short normalization pass: reconcile readiness counts, run bead lint fresh, add missing acceptance criteria for P2+ work, convert hidden architectural dependencies into explicit `blocks` edges, split large horizon features, and define release gates.

---

## Turn 158 (role=assistant) — answer to 'describe state of the project implied by full beads-set much more thoroughly, descriptively, including examples / simulations'

The full bead set implies Polylogue is already a substantial working system, not a greenfield plan. But it is also not yet a finished product. Its current shape is: a local-first evidence archive for AI work, with real ingestion, normalized storage, query surfaces, demo artifacts, reports, and tests, now being expanded into a full “system of record” for agent work: context memory, coordination substrate, forensic analytics, citation engine, web cockpit, and public proof surface.

The most accurate one-sentence state is:

Polylogue is a self-dogfooding AI-work archive that already proves it can turn raw agent/chat history into structured evidence, but the bead set is pushing it from “powerful local forensic tool” into “trusted operating layer for multi-agent development.”

The scale implied by the bead set is large. The archive snapshot contains 492 issue beads plus 22 durable memory records. Of the issues, 95 are closed, 396 are open, and 1 is in progress. The active work is 397 beads: 32 P1, 181 P2, 166 P3, and 18 P4. The repository snapshot itself is also large: hundreds of core/storage files, hundreds of tests, CLI/MCP/daemon/web/docs/devtools surfaces, generated artifacts, all-ref history, and demo shelves. The active private archive mentioned in closed proof beads was at schema v23 with 16,498 physical sessions and 4,142,175 messages during the July 2026 proof campaigns.

That matters because the bead set is not describing a toy app. It is describing a system already operating against a real, messy, multi-million-message personal AI-work corpus.

The project’s core object is not “a chat transcript.” It is evidence. A transcript is only one source. Polylogue wants to preserve raw provider exports, browser captures, tool calls, tool results, assistant messages, user prompts, action outcomes, model/provider usage, costs, attachments, blobs, session lineage, compactions, branches, work items, assertions, annotations, query runs, findings, analyses, variants, and external artifacts. All of those become typed things that can be queried, cited, rendered, compared, and fed back into agents.

The existing system already has several important pieces. It has a local archive rooted in SQLite tiers. It has durable source/user tiers and derived index/embedding tiers. It has normalized sessions, messages, blocks, actions, and parsed tool outcomes. It has CLI surfaces, MCP surfaces, daemon/web surfaces, browser capture/extension work, devtools, docs rendering, generated OpenAPI/CLI schemas, and a Beads-backed development loop. It has proof artifacts under `.agent/demos`, including claim-vs-evidence, agent-forensics, handoff-pack, and uplift experiment artifacts.

The closed P0 campaigns show what the project can already do. The claim-vs-evidence campaign converted a research question into product capability: “how often do agents proceed after failed tool calls?” became queryable through action-unit outcome fields and `followup_class`. A report could be regenerated against the active archive, with structured failures, calibrated markers, sensitivity windows, seeded reproduction, and a cold-reader gate. The forensics campaign regenerated usage/cost reporting on the current archive, separating physical-session tokens from logical high-water tokens and separating stored/provider-priced cost from catalog/API-equivalent estimates. The uplift experiment tested whether a handoff pack improved continuation-agent performance; the result was diagnostic and negative because the pack was stale, which then created follow-up product work around freshness.

That tells us something important about the project culture: Polylogue treats a failed or negative result as a product signal, not as an embarrassment. The stale uplift pack did not get hidden; it became evidence that context packs need freshness, successor links, progress-visible regeneration, and better continuation-time assembly.

The bead set also implies a very strong doctrine: Polylogue must not overclaim. This appears everywhere. Numeric fields need field-level evidence contracts. Empty backing rows must not render as zero. Prose-mined facts must carry `text_derived` provenance. Temporal aggregates must propagate the weakest timestamp source instead of laundering fallback dates into authoritative provider timestamps. Public claims need a claims ledger with evidence refs and statuses. Reports must state sample frames, calibration, caveats, and non-claims. This is not merely “add tests”; it is a product philosophy.

A typical dishonest version of this system would say: “The agent silently ignored 41,886 errors.” Polylogue’s intended version says something closer to: “The archive contains 41,886 structured failed tool outcomes. This sampled report inspected 5,000 origin-stratified failures. Under this marker classifier, the next-turn silent-proceed lower bound is X, with this calibration precision/recall, this sensitivity window, and these non-claims.” That level of evidence hygiene is one of the strongest signals in the bead set.

The current danger zone is also clear. The project has real capabilities, but it has several stop-the-line reliability and trust problems.

The first stop-the-line issue is deployment confidence. `polylogue-s7ae.6` is in progress because a full `devtools verify` run was aborted at 74% after a coordination commit. Until every failure is classified as coordination-caused, fixed, or pre-existing, the coordination layer is not deploy-clean.

The second is blob and attachment integrity. There is an open P1 to classify 39,586 missing referenced blobs in the production backup. Another P1 says `ops doctor cleanup_orphans` can delete an in-flight leased blob. Another asks browser capture to preserve uploaded attachment bytes going forward. The bead set’s implication is blunt: Polylogue already reasons about attachments and blobs, but its byte-integrity story is not yet trustworthy enough. The final state must guarantee that when the archive says “this message had this attachment,” the referenced bytes either exist, are recoverable, or are explicitly marked as missing with a classified reason.

The third is daemon/capture security. A P1 bead calls out Host/Origin checks, receiver tokens, and spool governors. The implied current state is that localhost capture/daemon surfaces are powerful enough to be dangerous if exposed incorrectly. Since this archive may contain all of a user’s AI work, the project needs DNS-rebinding protection, CSRF/auth discipline, receiver tokens, bounded spooling, and secret/excision paths.

The fourth is agent-write safety. The project wants agents to write useful evidence back into the archive, but not to corrupt trusted user memory. That is why `upsert_assertion` needs a single chokepoint: non-user authors must land as candidates with `inject:false`. In the final state, an agent can say “I think this file is related to this bug,” but it cannot silently inject that as trusted user context.

The fifth is scale. There are many closed scale fixes already, including the index rebuild partial-readiness bug and topology graph optimization, but the open scale-hardening epic says the system still has bugs that only appear on real-scale archives. This is exactly what one would expect from a tool that works against millions of messages: fixture success is not enough.

The project’s storage model is one of its central architectural axes. The bead memories make the doctrine explicit: durable tiers evolve through additive migrations and backup-manifest gates; derived tiers are rebuilt or blue-green swapped; `ops.db` is disposable. `source.db` and `user.db` are durable truth. `index.db` and `embeddings.db` are derived read models. A healthy Polylogue does not ask the operator to manually babysit normal convergence. FTS coherence, source-to-index materialization, insight refresh, and embedding catch-up should be automatic daemon/startup/write-path invariants. CLI commands are diagnostic or break-glass tools, not routine maintenance.

A simulation of the target rebuild behavior:

A schema bump lands for `index.db`. The user continues using the web UI and CLI. Reads still hit the old index generation. A fresh generation builds beside it from durable `source.db` and `user.db`. New writes are captured and replayed into the new generation. When the new generation is converged, Polylogue pauses writes for a tiny swap window, flips the active pointer, and reaps the old generation later. During the whole process, status surfaces say exactly what is happening: “layout ready,” “schema ready,” “materialization 72%,” “serving old generation,” “new generation warming,” not a false `archive_ready=true` over a partial corpus.

That is the difference between a hobby indexer and a local evidence database.

The target query/read model is another major transformation. The bead set wants one read algebra: Query × Projection × Render. Query decides what objects are selected. Projection decides what fields, joins, variants, budgets, context windows, and evidence forms are included. Render decides how the result appears: CLI table, JSON, MCP payload, web card, report, citation, docs artifact, etc. The goal is to delete parallel behavior across CLI, MCP, daemon, Python API, web UI, and generated docs.

In the target state, this kind of thing should be natural:

`actions where is_error:true | group by session.origin, followup_class | count`

The same semantic query should feed a CLI aggregate, a web chart, an MCP answer, a docs report, and a regression test. There should not be one implementation for CLI, another for web, another for report generation, and another hidden in a demo script.

The bead set also implies a richer object universe. Object refs expand beyond sessions/messages/blocks/actions into query runs, result sets, findings, cohorts, analyses, annotation batches, variants, variant nodes, compaction events, work episodes, delegations, artifacts, and external citations. A mature Polylogue object can be addressed, resolved, rendered, cited, compared, superseded, and used as evidence.

A simulation of target evidence navigation:

A user asks, “Did any previous agent already debug the duplicate action rows bug?”

Polylogue searches by repo, file paths, symbols, and action/tool outcomes. It returns an evidence basket:

`issue:polylogue-xnkf` — actions view fans out on duplicate `tool_ids`.  
`session:claude-code/...` — agent inspected the affected SQL view.  
`block:...` — exact SQL fragment cited.  
`action:...` — failed test result.  
`finding:...` — derived explanation: one logical action becomes up to N×M rows because duplicate tool IDs multiply joins.  
`artifact:...` — previous patch or report.  

The user can open the result in the web UI, export a citable report, or ask an agent to resume from it. The agent receives a bounded context pack, not an unstructured dump. Every included claim links back to an object ref. If some evidence is text-derived rather than raw, the payload says so.

That is the “system of record” ambition.

The agent coordination substrate is the current P1 spine. It is not merely “agents can chat.” It is an evidence-backed coordination layer over sessions, repos, worktrees, branches, work item refs, resource episodes, context injections, messages, handoffs, proof artifacts, and freshness/provenance. Beads is an adapter, not a hard dependency. The system should work even if the repo uses GitHub issues, another tracker, or no tracker.

A simulation of the target multi-agent scenario:

Agent A starts in `/repo/polylogue` on worktree `coordination-a` and claims a bead about MCP prompts. Agent B starts in the same repo on worktree `coordination-b` and begins touching the context scheduler. The daemon sees both sessions, their repo roots, branches, changed files, current beads, and recent actions. The coordination envelope says: “same repo peer active; overlapping area: context/MCP; possible collision: generated schemas.” Agent A posts a scoped message: “I’m regenerating MCP schemas after adding prompts; avoid render churn until I push.” Agent B does not get spammed in a chatroom. Instead, the context scheduler injects a subtle advisory at the next appropriate boundary. Later, Agent A produces a handoff packet with changed files, tests, unresolved risks, and evidence refs. Agent B’s context includes that handoff because it is relevant, fresh, and within budget.

The proof bead for this is concrete: two agents, separate worktrees, one repo; overlap detection; exactly one scoped message; scheduler-mediated injection; reproducible before/after JSON envelope artifacts.

That implies the future Polylogue is not just for reviewing past work. It becomes active infrastructure while work is happening.

The context/memory loop is related but broader. The `polylogue-37t` epic implies agents should not simply start from a blank prompt or a pile of raw logs. They should receive judged, bounded, provenance-carrying context. Declared claims become candidates; candidates are judged; safe material becomes preamble/context; context assembly is deterministic and budgeted; every injection is logged. The bead set wants a context scheduler as the single arbiter for everything that enters an agent’s context.

A simulation of target context assembly:

A new coding agent enters the repo. Polylogue compiles context from several sources: current repo, active bead, recent sessions touching the same files, unresolved failures, accepted user preferences, relevant handoffs, fresh coordination messages, and maybe a compacted-session regrounding pack. It refuses stale or rejected assertions. It includes candidate material only if the profile allows it. It emits a ledger: “these 12 items were considered; these 5 were included; these 7 were excluded for stale/superseded/rejected/budget reasons.” Running the same inputs twice yields byte-identical context.

That is a very different model from “grep some logs and paste a summary.”

The content-variant system is a separate P1 product axis. The bead set wants language-aware transformed archive objects with alignment. Variants are not assertions. A translation, transliteration, simplification, or summary is a first-class variant object over a source ref. It has kind, language, status, coverage, provenance, and alignment edges. Alignment can represent one-to-one, one-to-many, many-to-one, omitted, and partial mappings.

A simulation:

A user has a mixed Polish/English ChatGPT session. The raw session remains unchanged. Polylogue detects block/message/session language with confidence. The user asks for an English simplified variant of a dense Polish explanation. A variant is created targeting `message:...` and `block:...`. The variant text is stored separately. Its nodes align to the original blocks. The web reader can show original, variant, or side-by-side. Search can find either original Polish terms or English variant terms. If an agent creates a candidate variant, the user can accept, reject, or supersede it. The source content is never overwritten.

That tells us the archive is intended to preserve original evidence while allowing user-friendly transformed views.

Session lineage truth is another major theme. Polylogue is tracking provider sessions, branch points, shared prefixes, compactions, subagents, logical threads, and physical sessions. The bead set says shared content must be stored once and counted once. External counts should use logical grain with physical figures footnoted. Physical-session token totals and logical high-water token totals must not be conflated. This is especially important because the forensics campaign already encountered token/cost double-counting risks.

A simulation:

Claude creates a long coding session. It compacts. A subagent runs. A later continuation shares the same prefix. A naive archive counts all repeated prefix messages in every physical transcript and inflates messages/tokens/cost. Target Polylogue stores shared content once, represents physical sessions separately from logical composed sessions, records compaction boundaries, and can render either grain honestly:

“Physical sessions: 3. Logical composed work thread: 1. Shared prefix: 1,240 messages. Unique post-branch content: 380 messages. Token usage: provider-reported physical total X; logical high-water estimate Y; billing reconciliation not claimed.”

That is the level of precision implied by the lineage beads.

Compaction lifecycle is even more explicit. Today, Polylogue mostly observes compaction after the fact. The target state treats compaction as an event with pre-compaction snapshots, loss forensics, and regrounding. It wants to know what was in context before compaction, what survived, what was transformed, what was forgotten, and whether forgotten items later became needed.

A simulation:

Before compaction, a hook captures the assembled context payload and stores it as a blob. If the hook is unavailable, Polylogue records a manifest of refs instead and marks the weaker snapshot source. After compaction, Polylogue compares retained and lost items: file paths, object refs, claims, tasks, recent failures. It ranks “lost but later needed” items. Later an agent asks `compaction_forgot --top 10`, gets stable refs, and requests `compaction_reground`. The scheduler injects a bounded regrounding pack into context. The next compaction preserves the regrounding assertion because it is explicitly represented.

That is not a normal transcript archive feature. It is OS-like memory management for agent work.

The web workbench target is also far beyond a search page. The current web UI appears to exist but has architecture and reliability debt. The future web UI is an evidence cockpit: result lists, reader views, timelines, live tailing, evidence baskets, citable reports, session replay, analytics, partial/stale/error states, citation anchors, and eventually webui architecture v2 with generated typed API clients, SSE/cache, routing, tokens, palettes, and reproducible builds.

A simulation:

The user opens the web workbench during an index rebuild. Instead of every panel independently showing “Failed to fetch,” the page has one global liveness/convergence model. It says: “Daemon reachable. Facets stale by 2m14s. Index generation rebuilding, 68%. Results served from old generation. Live tail active.” The user searches for a failed test. They drag three sessions, two actions, and one finding into an evidence basket. The report builder emits a Markdown/HTML report with citations to exact blocks and typed drift states. If a cited block changed after re-ingest, the citation says “content hash drifted” instead of silently pointing at a different thing.

That is the intended surface: not a dashboard for vibes, but a workbench for evidence.

The analytics target is broad. The `polylogue-9l5` and related work-graph beads imply Polylogue wants to answer “so what?” questions about AI work. Not just “what happened?” but “what patterns lead to success, failure, churn, cost, latency, correction, abandonment, or quality?” The planned units include delegations, episodes, artifact edges, turn pairs, entity mentions, world effects, verification runs, projects, topic clusters, and cross-origin threads. The planned methods include temporal trends, process mining, survival analysis, prediction/calibration, experiment hosting, information-theoretic measures, and trajectory-quality scoring.

A simulation:

The user asks, “Which workflows waste the most agent time?”

Polylogue does not just count tokens. It identifies work episodes, joins tool failures, test reruns, corrections, compactions, delegation edges, file churn, and final outcomes. It reports:

“Refactor episodes touching storage twins have high correction density and high rebuild latency. Median time-to-green is X. Episodes with early `devtools verify --quick` plus focused tests have lower abandonment. Prediction confidence is low for web UI work because outcome labels are sparse. Here are the five episodes and exact evidence refs.”

The bead set repeatedly says these analytics must carry construct-validity limits. That means the system should say when a metric is weak, not just draw a chart.

Interop is another implied future. Polylogue is not meant to ingest only one provider. The bead set names Hermes bridge, provider usage, origin breadth, external trace formats, Beads issue history, browser capture, agent configuration, PKM/export, git notes, PR/issue citations, SARIF-style exports, federation, research/eval formats, and a versioned normalized session interchange schema. The target system becomes both importer and exporter of AI-work evidence.

A simulation:

A user imports Claude Code logs, Codex state, ChatGPT browser captures, Gemini/Grok browser sessions, Beads issue history, and Hermes runtime spans. Polylogue normalizes them into common sessions/messages/blocks/actions while preserving origin-specific raw evidence and parser fingerprints. A later export emits a self-contained evidence packet: normalized session schema, cited blocks, raw-source provenance, verification run, report, and stable object refs. A GitHub PR can link to the authoring session’s postmortem bundle. A research tool can consume the normalized trace without reverse-engineering every provider.

That is why the normalized session model and OriginSpec work are important. Without them, every surface becomes a one-off parser.

External legibility is a P1 program because the project currently appears to have more substance than public clarity. The bead set says a stranger should be able to understand the project from the README’s first screen, run the one-command demo, and cite a published finding. This includes a public claims ledger, docs-site publishing lane, install matrix, proof cards, GitHub page polish, launch kit, Datasette lane, codebase atlas, docs IA cleanup, and anti-grep explanation.

The implied public positioning is not “chat history viewer.” It is closer to:

“Polylogue is the local system of record for AI work. It preserves raw evidence, normalizes provider traces, lets you query tool use and outcomes, builds citable reports, and feeds verified context back into agents.”

The anti-grep proof card matters because the obvious skeptic says: “Why not just grep `~/.claude`?” The answer implied by the beads is: grep finds text; Polylogue resolves structure. It knows tool calls versus results, exit codes versus prose claims, physical versus logical sessions, compaction boundaries, model/provider cost provenance, citation anchors, follow-up behavior after failures, attachments/blobs, and derived work units. Grep cannot tell you whether an assistant acknowledged a failed command within the next three turns, grouped by provider, with calibrated caveats.

The Beads memories add another layer: the project has an explicit agent-development operating model. A bead is supposed to be execution-grade before implementation: why/evidence, design/how, checkable acceptance, verify commands, PR-shaped size. There are rules about schema bumps, shell traps, search hygiene, devloop daemons, Beads field-update discipline, area labels, and backlog lint. This means the project is not just building Polylogue; it is building a disciplined loop for agents to build Polylogue using Polylogue.

That self-reference is central. Polylogue archives agent work. Beads direct agent work. Agents use Polylogue to recover context. Polylogue produces evidence about agent work. That evidence changes the product. The product then improves the next agent run.

A concrete dogfood loop looks like this:

An agent runs a long devloop and fails tests. Polylogue captures the session, tool calls, errors, files touched, and resulting commits. A claim-vs-evidence report later finds that the agent proceeded after a failed tool result without acknowledging it. That becomes a finding. The finding becomes a query capability. The query capability becomes an MCP prompt or devloop-review check. Future agents receive an advisory when they are about to repeat the same pattern. If the advisory helps, an uplift experiment measures it. If the experiment fails because the context pack is stale, freshness becomes a product bead.

That loop is the heart of the project.

The full bead set also implies a final end state with several invariants:

First, every important user-visible claim is evidence-backed or explicitly labeled as unmeasured, capability-only, stale, text-derived, sampled, estimated, or degraded.

Second, raw evidence is never casually overwritten. Derived views can be rebuilt. Variants can be created. Summaries can be attached. But original source content remains addressable and citable.

Third, agents can contribute memory, tags, variants, messages, and assertions, but their writes are gated, attributed, non-injected by default, and reviewable.

Fourth, storage operations are safe under real conditions: daemon death, rebuilds, large corpora, partial indexes, blob leases, schema bumps, interrupted ingestion, and provider format drift.

Fifth, every surface speaks the same contracts. CLI, web, MCP, daemon, generated docs, Python API, and reports should not implement separate truths.

Sixth, the system is useful while work is happening, not only after the fact. Hooks, capture, daemon convergence, context scheduling, coordination envelopes, and live tailing make Polylogue an active development companion.

Seventh, the archive can forget on purpose. “Keep everything” is not allowed to mean “secrets are immortal.” Excision and secret hygiene become first-class.

Eighth, public readers can reproduce and cite the important claims. The docs site, claims ledger, proof artifacts, one-command demo, and install matrix make the project externally inspectable.

The final system “feels” like this:

You start a coding session. Polylogue already knows the repo, branch, worktree, open work items, past sessions, unresolved failures, relevant docs, accepted preferences, rejected stale claims, and active peer agents. It captures the session live. When you search, it returns structured evidence, not just keyword hits. When an agent writes memory, it is a candidate with provenance. When a report gives a number, the denominator and evidence source are visible. When an index rebuild runs, the UI tells the truth. When a blob is missing, the system says which reference broke and whether it is recoverable. When you publish a claim, the claim ledger links it to proof. When another agent resumes next week, it gets a bounded, fresh, cited context pack instead of folklore.

The most important unresolved reality is that the bead set is ahead of the product. It encodes a coherent target state, and some impressive pieces are already real, but the current project is still in a hardening phase. The P1 layer is dominated by trust gates: verify classification, blob safety, daemon security, missing blob classification, evidence-honesty bugs, agent-write safety, lineage truth, content variants, coordination proof, and external legibility.

So the project implied by the full bead set is not “nearly done with a few features left.” It is more like this:

A serious working prototype has crossed into real-product territory. It has enough data, proof, and surface area to be valuable. It has also become powerful enough that correctness, security, provenance, rebuild safety, and public claim discipline now dominate. The bead set is the transition plan from impressive dogfood tool to trustworthy local infrastructure for AI work.

---

## Turn 220 (role=assistant) — answer to a request to adjust the plan into gates/lanes/readiness rules (54,086 chars) — the most substantial single re-planning delivery in the thread

Here is the adjusted plan: the bead list is no longer treated as a single 397-step queue. It becomes a delivery system with gates, lanes, readiness rules, explicit hidden dependencies, and concrete proof requirements.

A “gate” here means a check that must pass before later work can ship. A “lane” means a group of work that can move in parallel with other groups. A “release” means a coherent batch that leaves the system safer or more useful than before. A “proof artifact” means a durable file, report, test log, fixture, demo output, or command result that shows the work really happened.

The adjusted execution model is:

The original bead order remains the safe dependency order.

The implementation plan is organized into releases.

Each release has a strict entry condition, exit condition, verification suite, and rollback/degrade story.

Every bead is either immediately implementable, needs a small specification pass, needs acceptance criteria, or is parked as horizon work.

P1 and safety P2 beads are not allowed to wait behind attractive feature work.

Public claims, analytics, launch material, agent memory injection, browser posting, and evidence exports are blocked until their evidence and safety foundations are in place.

The plan’s new backbone is this:

1. Normalize the backlog and readiness rules.
2. Close the trust floor: no false claims, no unsafe blob deletion, no unsafe agent memory writes, no unsafe daemon/capture access.
3. Harden storage, blobs, rebuilds, migrations, and degraded states.
4. Establish the shared read/evidence contract.
5. Ship context, memory, and agent coordination on top of safe writes.
6. Ship content variants, lineage truth, and compaction recovery.
7. Make live intake, capture, daemon, and interactive surfaces fast and reliable.
8. Build the web evidence cockpit.
9. Build analytics only after evidence honesty and statistical metadata exist.
10. Expand interop and export.
11. Make the project publicly legible and launchable.
12. Leave speculative P4 work parked behind explicit future gates.

The important change is that “ready” no longer means “has no hard blocker in Beads.” It means “safe to implement now without inventing missing contracts.”

The bead set contains 397 active beads: 32 P1, 181 P2, 166 P3, and 18 P4. All 32 P1 beads have acceptance criteria. That is good: the urgent layer is executable. The weak spot is the long tail: 72 active beads lack acceptance criteria, including 5 P2 beads, 60 P3 beads, and 7 P4 beads. Those are not implementation-ready until they receive checkable completion conditions.

The five active P2 beads that need acceptance criteria before implementation are:

| Bead | Needed acceptance criteria |
| --- | --- |
| `polylogue-rsad` | MCP responses over size budget return metadata-only summaries, continuation handles, and explicit “open/read next” affordances. Boilerplate is removed. Tests prove large payloads do not flood the agent context. |
| `polylogue-2qx` | `OriginSpec` exists as the sole origin dispatch contract. Each origin declares strictness, detector, parser, fixture, schema mapping, and docs. Dispatch order is deterministic and tested against ambiguous fixtures. |
| `polylogue-37t.5` | Local OpenAI-compatible embedding provider works through the same abstraction as cloud providers. Context retrieval can opt into it. Secrets are not logged. Retrieval fixtures prove local/cloud parity at the interface level. |
| `polylogue-rii.2` | Hook events and OTLP spans materialize into queryable evidence tables with stable refs, idempotent replay, parser fingerprints, and fixture coverage. |
| `polylogue-fs1.4` | Hermes forensic report emits citable findings over imported Hermes sessions, with coverage/fidelity caveats, raw evidence refs, and regeneration command. |

Those five should be updated before they enter a sprint.

The P3/P4 beads without acceptance criteria should not be coded directly. They should either be converted into execution-grade beads or deliberately parked. Some of them are good ideas, but “good idea” is not enough for this system. Anything that touches claims, memory, analytics, ingestion, or public surfaces needs an evidence contract before code.

The comprehensive execution adjustment is below.

## 0. Global execution rules

These rules apply to every release.

First, a child bead closes before its parent epic closes. An epic is not done because “the main idea landed.” An epic is done when all children are closed or intentionally split out, and its terminal acceptance criteria pass.

Second, durable evidence beats derived convenience. Raw source rows, user state, blob bytes, object refs, and provenance are more important than fast views, dashboards, search indexes, summaries, or reports. If a derived surface disagrees with durable evidence, the derived surface is wrong.

Third, every number is a claim. A count, percentage, duration, cost, token total, confidence value, trend, or score must carry its denominator, evidence source, and absence semantics. “No backing rows” must not render as zero.

Fourth, every time has a source. Provider timestamp, capture timestamp, file timestamp, fallback date, synthetic date, and inferred date are different things. Aggregates must carry the weakest time source inside them.

Fifth, every agent-written fact is untrusted until judged. Agent writes can be useful candidates. They cannot silently become active, injected, trusted memory.

Sixth, every browser/daemon route is hostile until proven safe. Localhost does not mean safe. Capture surfaces need Host checks, Origin checks, tokens, bounded spooling, and clear auth behavior.

Seventh, every importer is a contract, not a script. An origin needs a detector, parser, raw fixture, normalized fixture, fidelity declaration, schema mapping, and drift sentinel.

Eighth, every public claim needs a public evidence reference. README, docs, launch posts, demos, reports, and screenshots must link to the evidence that supports them or mark the claim as capability-only, unmeasured, stale, sampled, or not supported.

Ninth, every user-facing read path goes through the same contract. CLI, daemon, MCP, Python API, web UI, generated docs, and reports must not each implement their own truth.

Tenth, refactors are allowed when they reduce risk or unlock a release. Cosmetic cleanup does not outrank safety, evidence integrity, migration correctness, or public proof.

## 1. Backlog normalization release

This is the planning correction that makes the bead set implementation-ready.

Purpose: turn the bead inventory into a controlled delivery source.

Included work:

`polylogue-8jg9.1` becomes a standing backlog-hygiene invariant. It should lint readiness, blocked state, missing acceptance criteria, stale parent/child state, and impossible “ready” counts.

The readiness discrepancy must be reconciled. The snapshot contains conflicting signals: one summary says 317 ready issues, another count says 100 ready, and inherited blocked count differs from direct blocked count. The implementation plan should pick one definition:

A bead is “ready to implement” only when all of these are true:

It is open or in progress.

It has no active hard blocker.

It has acceptance criteria.

It has enough design or code-locality information to start safely.

It does not depend on an unmodeled product contract.

It is small enough to fit into one PR or has been split.

The output of this normalization release should be a `delivery_manifest` or equivalent artifact with these fields:

Bead ID.

Release assignment.

Lane assignment.

Readiness grade.

Hard blockers.

Soft blockers added by this plan.

Required proof artifact.

Verification lane.

Rollback or degraded-mode note.

Public-claim impact.

Migration impact.

Security/privacy impact.

Data-loss impact.

A practical grading system:

Grade A means implement now. It has description, design, acceptance criteria, and no active blockers. There are 166 active P1/P2 beads in this shape.

Grade B means implement after a local code inspection. It has acceptance criteria, but its design or description is incomplete. There are 42 active P1/P2 beads in this shape.

Grade C means do not implement yet. It is P1/P2 but lacks acceptance criteria. There are 5 of these.

Grade D means horizon-ready. It is P3/P4 with acceptance criteria, useful for future planning but not part of the safety spine.

Grade E means idea/spec work. It is P3/P4 without acceptance criteria or with excessive scope.

Exit criteria for this release:

Backlog lint reports one consistent ready count.

All active P1 and P2 beads have acceptance criteria.

Every release below has a manifest row.

Every P3/P4 bead is either assigned to a future release or explicitly parked.

Every new hidden blocker listed in this plan is either added to Beads or recorded in the delivery manifest.

## 2. Release A — trust floor

Purpose: make the archive safe enough to trust.

This release closes the urgent layer where the system can currently lie, lose bytes, expose private data, or let agents self-promote claims.

Primary beads:

`polylogue-s7ae.6` — classify the aborted full verification run before coordination deploy.

`polylogue-8jg9.4` — prevent ops-doctor blob cleanup from deleting in-flight leased blobs.

`polylogue-9e5.28` — rigor audit must cover the product registry, not only declared contracts.

`polylogue-9e5.29` — absent numeric evidence must not render as zero.

`polylogue-9e5.30` — prose-mined forensic fields must carry `text_derived` provenance.

`polylogue-cpf.5` — temporal aggregates must propagate the weakest timestamp source.

`polylogue-cpf.6` — relative-date parsing needs an injected clock seam and `sort_key_ms` audit.

`polylogue-kwsb.1` — daemon/capture Host, Origin, token, and spool hardening.

`polylogue-37t.15` — all non-user assertion writes become candidate, non-injected assertions.

`polylogue-83u.4` — classify the 39,586 missing referenced blobs.

Supporting P2 work:

`polylogue-1xc.12` for full-text-search drift and coherence checks.

`polylogue-20d.4` for structured-query routing parity.

`polylogue-4ts.3`, `.4`, `.6` for lineage correctness and transcript completeness signals.

`polylogue-a7xr.1`, `.2`, `.3`, `.6` for connection leaks, stale profile disagreement, unordered prose reconstruction, and duplicated date parsing.

`polylogue-f2qv.1`, `.2`, `.4`, `.5`, `.3`, `polylogue-ivsc`, `polylogue-xy95`, `polylogue-5hf` for usage/cost honesty.

`polylogue-peo` for daemon crash forensics.

`polylogue-4be` for restore drills.

`polylogue-8jg9.2` for blob-GC race fixtures.

`polylogue-9e5.3`, `.4`, `.19` for column honesty, writer race audit, and storage-layer correctness scenarios.

`polylogue-b0b` for replacing keyword outcome/pathology heuristics with structural evidence.

Exit criteria:

A full `devtools verify` run is either green or every failure is classified as fixed, pre-existing, or intentionally deferred with a bead reference.

A malicious cross-origin GET to the daemon is refused.

A foreign Host header is refused.

Unauthenticated capture POST is refused.

Forged-token capture POST is refused.

The web shell and extension still work through legitimate routes.

A leased uncommitted blob survives both normal blob GC and ops-doctor cleanup.

Every one of the 39,586 missing blob references is classified by table, reference type, origin, recoverability, and decision.

Direct-file-recoverable blobs are restored with SHA-256 verification.

Quantitative insight products over empty backing rows render `None`, `unknown`, `uncovered`, or equivalent; never `0`.

Temporal aggregate tests prove “weakest source wins.”

Prose-mined forensic fields are visibly caveated.

Agent writes through blackboard/MCP/API land as candidate, non-injected assertions unless the author is the user.

Simulation:

A browser tab controlled by an attacker tries to fetch the local daemon through DNS rebinding. Before this release, it may read archive data because localhost GET routes are not centrally protected. After this release, the request dies at the first middleware check. The archive does not care that the request came from “localhost”; it checks Host and Origin policy before dispatch.

Another simulation:

An agent says, “The project uses blue-green rebuilds already,” and posts that as a memory. Before this release, the claim can become active agent-authored memory. After this release, it becomes a candidate assertion with `inject:false`. It may appear in a review queue. It cannot become trusted context by itself.

Another simulation:

A report asks, “How many failed tool calls had no assistant acknowledgement?” If the backing rows are missing because an index did not materialize, the answer cannot be “0.” It must say “uncovered” or “unknown.” A zero is reserved for the case where there was evidence and the count is truly zero.

## 3. Release B — storage, rebuild, and byte-integrity floor

Purpose: make Polylogue safe under real archive scale.

This release turns storage from “works on fixtures” into “survives large private archives, interrupted rebuilds, schema changes, blob races, and daemon death.”

Primary beads and groups:

`polylogue-1xc` scale-hardening epic.

`polylogue-1xc.8` schema rebuild-safety scenario.

`polylogue-6wnh` large Codex append thread-refresh cost.

`polylogue-ma2` FK-supporting index for cleanup paths.

`polylogue-xgw` schema hygiene for evidence-cockpit reads.

`polylogue-fie` archive scaling doctrine.

`polylogue-20d.15` bulk ingest throughput and resource envelope.

`polylogue-b5l` blue-green index rebuilds.

`polylogue-60i5` durable-tier batch migration coordination.

`polylogue-8jg9` operational resilience epic.

`polylogue-peo` daemon crash forensics.

`polylogue-4be` restore drill.

`polylogue-83u.3` preserve uploaded attachment bytes in live browser capture.

`polylogue-83u.6` attachment acquisition census.

`polylogue-83u.2` non-inline attachment acquisition.

`polylogue-jsy` blob hash validation cleanup.

`polylogue-83u.5` blob-store compression, after integrity is proven.

The release adds a hard rule:

Compression, optimization, and storage cleanup never precede byte-integrity proof. First prove bytes are present and classified. Then compress.

New blocker edges to add:

Make `polylogue-b5l` depend not only on `polylogue-20d.15`, but also on rebuild-safety proof from `polylogue-1xc.8`, blob-GC safety from `polylogue-8jg9.4`/`polylogue-8jg9.2`, and restore proof from `polylogue-4be`.

Make `polylogue-83u.5` depend on `polylogue-83u.4`, `polylogue-83u.2`, and `polylogue-83u.3`.

Make any public backup/restore claim depend on `polylogue-4be`, `polylogue-83u.4`, and `polylogue-b5l`.

Exit criteria:

A derived-tier rebuild cannot show a partial corpus as archive-ready.

A schema bump has a declared durable-tier migration path or derived-tier rebuild path.

A full index generation can be rebuilt beside the old generation and swapped only when complete.

Writes during rebuild are either replayed, blocked briefly with a loud state, or explicitly safe.

WAL growth and write-lock duration are bounded by a named resource envelope.

A live-copy probe or large synthetic corpus reproduces the bug classes that previously only appeared at scale.

Backups are restored in a drill, not merely created.

Blob references either resolve to bytes, resolve to classified recoverable missing state, or resolve to documented irrecoverable state.

Simulation:

The user starts Polylogue after a schema bump. `source.db` and `user.db` are durable; they migrate additively with a backup manifest. `index.db` is derived; it begins a fresh generation. The web UI does not say “ready” over a half-built corpus. It says: “serving old index generation; new generation 64%; writes replay queue 118 events.” When the new generation converges, Polylogue swaps generations. No user or agent sees a phantom 3,000-session archive when the source has 16,000 sessions.

## 4. Release C — one read contract and first-class evidence objects

Purpose: stop every surface from inventing its own archive semantics.

This release makes “read something from Polylogue” a shared contract:

Query selects objects.

Projection chooses fields, joins, variants, budgets, and context windows.

Render decides CLI table, JSON, MCP payload, web card, report, Markdown, or citation.

Primary beads:

`polylogue-4p1` — one read algebra: Query × Projection × Render.

`polylogue-4p1.1` — route daemon split-archive fast path through `SessionQuerySpec.from_params`.

`polylogue-fnm` — one query grammar.

`polylogue-fnm.11` — pipeline/clause parity across units and generated support matrix.

`polylogue-fnm.1`, `.10`, `.12`, `.13`, `.14` — aggregates, field selection, macros, set algebra, compact transforms.

`polylogue-ap7` — semantic transcript rendering shared across CLI/web.

`polylogue-t46` — contracts own surfaces.

`polylogue-t46.3`, `.4`, `.5`, `.6`, `.8` — collapse duplicate CLI/MCP/daemon execution paths.

`polylogue-jnj.1`, `.5`, `.x7d`, plus later `polylogue-jnj` work — surface algebra and rendering contracts.

`polylogue-rxdo.1`, `.2`, `.3`, `.4` — ObjectRefs, durable query identity, query-run telemetry, finding assertions.

`polylogue-svfj` — block content-hash citation anchors with typed drift states.

`polylogue-bby.15` — evidence basket to citable report to verified export, but only after evidence refs exist.

New blocker edges to add:

Make `polylogue-bby.15` depend on `polylogue-rxdo.1`, `.2`, `.3`, `.4`, `polylogue-4p1`, `polylogue-fnm.11`, and `polylogue-svfj`.

Make `polylogue-3tl.16` public claims ledger depend on `polylogue-rxdo.4`, `polylogue-svfj`, `polylogue-9e5.28`, `.29`, `.30`, `polylogue-cpf.5`, and `polylogue-cpf.6`.

Make every report/demo bead under `polylogue-212` and `polylogue-3tl.4` depend on the evidence object layer if it emits citable claims.

Exit criteria:

A query issued through CLI, daemon, MCP, and Python API selects the same rows.

Projection behavior is not reimplemented with per-surface flags.

A rendered citation can resolve to a block by content hash.

If cited content changes after re-ingest, the resolver returns a typed drift state instead of silently pointing to new content.

A query run is addressable as an object.

A result set is addressable as an object.

A finding is an assertion with evidence, lifecycle, and provenance.

A report can carry evidence refs without bespoke glue code.

Simulation:

The user runs a query in the web UI: “tool failures followed by assistant silence.” They save the result as a result set. Later the CLI opens the same result set and renders it as Markdown. MCP reads it as structured context. A docs report cites three blocks from it. No surface recomputes a different answer. If one cited block’s content hash changed after a re-import, the report says the citation drifted.

## 5. Release D — agent context, memory, and coordination

Purpose: make Polylogue useful while agents are working, not only after the fact.

This release is where Polylogue becomes an active coordination layer. It is blocked on the trust floor because unsafe agent writes and unsafe daemon routes would poison the whole idea.

Primary beads:

`polylogue-37t.15` — already in trust floor; single safe assertion-write chokepoint.

`polylogue-37t.12` — judgment queue for candidate assertions.

`polylogue-37t.11` — context scheduler: the one arbiter for what enters agent context.

`polylogue-d1y` — one-command hooks install and hook liveness monitoring.

`polylogue-pj8` — agent query cookbook and MCP prompt discoverability.

`polylogue-s7ae.2` — pre-deployment MCP/hook coordination batch.

`polylogue-ahqd` — observe MCP write adoption.

`polylogue-s7ae.3` — coordination messages and scheduler-mediated advisories.

`polylogue-s7ae.5` — live proof with two agents, separate worktrees, one repo.

`polylogue-3gd` and `polylogue-3gd.1` — activation layer and adoption telemetry.

`polylogue-t8t` — seven core agent workflows walked end-to-end.

`polylogue-37t.1`, `.2`, `.3`, `.4`, `.6`, `.7`, `.8` — assertion lifecycle, inline annotation, reboot-with-refs, SessionStart rollout, devshell entry, failure loop, resume routing.

`polylogue-rii.1`, `.2`, `.3`, `polylogue-x4s`, `polylogue-4c0`, `polylogue-7fj` — live substrate intake, work events, Beads issue history, and session-to-bead links.

New blocker edges to add:

Make `polylogue-s7ae` deployment depend on `polylogue-kwsb.1`, not only coordination-specific beads. Coordination over an unsafe daemon is not deployable.

Make `polylogue-bfv` advisory hooks depend on `polylogue-kwsb.1`, `polylogue-37t.15`, `polylogue-37t.11`, `polylogue-d1y`, `polylogue-20d.1`, and `polylogue-20d.12`.

Make `polylogue-3gd` depend on `polylogue-37t.11` as well as hooks and cookbook, because activation without scheduler control risks noisy or unsafe context.

Make `polylogue-rii.2` depend on `polylogue-2qx` once `OriginSpec` becomes the ingest contract.

Exit criteria:

Agent-authored claims enter as candidates.

Candidate review works in bulk.

Context assembly is deterministic from declared inputs.

Context assembly emits a ledger: included items, excluded items, reasons, budget usage, and trust class.

Hooks can be installed with one command.

Hook liveness is visible.

MCP prompts and roles are discoverable.

A two-agent proof exists: separate worktrees, same repo, overlap detected, one scoped message sent, context injected by the scheduler, handoff packet produced.

The proof artifact is reproducible and includes before/after JSON envelopes.

Simulation:

Agent A starts on a storage bead. Agent B starts on a web evidence-basket bead. Polylogue sees the same repo, different worktrees, overlapping files around read contracts. Agent A sends: “I am changing ObjectRef shape; avoid export assumptions until this PR lands.” Agent B does not receive random chatter. The scheduler injects the advisory only when relevant. Later, Agent A leaves a handoff packet containing changed files, tests, remaining risks, and evidence refs. Agent B’s context includes that handoff because it is fresh, scoped, and under budget.

## 6. Release E — content variants, preferences, and transformed views

Purpose: allow translated, simplified, summarized, or otherwise transformed content without confusing it with original evidence.

Primary beads:

`polylogue-0v9p` — language detection and preference facts.

`polylogue-arso` — content variant substrate: refs, nodes, alignment, storage.

`polylogue-rlsb` — variant-aware projection, query, and reader render profiles.

`polylogue-d4zk` — user/agent UX for creating, reviewing, messaging about variants.

`polylogue-4smp` — content variants epic closure.

Related preference/configuration beads:

`polylogue-y4c` — configuration doctrine and DB-backed runtime preferences.

`polylogue-6kh` — query-scope preferences.

`polylogue-w8db` — configuration doctrine epic.

`polylogue-y8w`, `polylogue-3xx`, `polylogue-1jc` — reading preferences, verb behavior, learned defaults.

New blocker edges to add:

Make `polylogue-d4zk` also depend on `polylogue-37t.15` if agents can create variant candidates. Agent-created variants need the same safety boundary as assertions.

Make variant-aware public rendering depend on `polylogue-svfj`, because variants must cite source content with drift-aware refs.

Make learned preferences depend on the judgment queue, not raw behavioral inference. Learned defaults should be candidates, not silent settings changes.

Exit criteria:

A content variant targets a public object ref.

Variant kinds are closed vocabulary: translation, transliteration, simplification, summary, and any explicitly added kind.

Variant nodes can represent transformed content at session, message, block, span, or assertion-body grain.

Alignment edges can represent one-to-one, one-to-many, many-to-one, omitted, expanded, reordered, and partial mappings.

Original text and variant text are labeled distinctly in CLI, web, MCP, query, and export.

Agent-created variants are candidates where review is required.

A fixture shows a heavily annotated mixed-language session with original text, translated variant, simplified variant, and clickable alignment back to the source.

Simulation:

A Polish message is translated into English. The original block stays unchanged. The English text is a variant node. The variant says which source block it translates, which parts are summarized, and which parts were omitted. Search can match the English variant, but a report cannot cite the translation as original evidence. It cites the source block and may also show the variant as a view.

## 7. Release F — lineage truth and compaction recovery

Purpose: represent long AI sessions as they actually happen: branches, shared prefixes, compactions, continuations, subagents, and physical/logical grains.

Primary beads:

`polylogue-4ts.3` — distinguish subagent auto-compaction from main-session compaction.

`polylogue-4ts.4` — lineage composition reads in one transaction.

`polylogue-4ts.6` — completeness signal for silently truncated transcripts.

`polylogue-4ts.5` — compaction boundary range columns and effective-context derivation.

`polylogue-4ts.7` — physical session identity collision beneath origin/native ID.

`polylogue-4ts` — lineage truth epic.

`polylogue-gjg.1` — compaction events and loss-item derived tables.

`polylogue-gjg.2` — pre-compaction snapshot capture.

`polylogue-gjg.3` — deterministic loss forensics.

`polylogue-gjg.4` — `compaction_forgot` and `compaction_reground` surfaces.

`polylogue-gjg` — compaction lifecycle epic.

`polylogue-h6r` — stable agent identity tuple.

`polylogue-2qx` — OriginSpec, needed for origin identity correctness.

New blocker edges to add:

Make `polylogue-gjg.1` depend on `polylogue-4ts.5`.

Make `polylogue-gjg.2` depend on `polylogue-d1y`.

Make `polylogue-gjg.4` depend on `polylogue-37t.11`, because regrounding is context injection and must pass through the scheduler.

Make `polylogue-4ts.7` depend on `polylogue-2qx`.

Exit criteria:

Shared prefixes are stored once.

Shared prefixes are counted once at logical grain.

Physical session counts remain available but are footnoted when used externally.

A composed session read is transactionally consistent.

Transcript truncation cannot happen silently.

Compaction boundaries are visible.

Effective context can be derived.

A pre-compaction snapshot or manifest exists where hooks allow it.

Loss forensics can rank “lost but later needed” items.

A regrounding pack can survive the next compaction.

Simulation:

A long session compacts after 90,000 tokens. Later, a continuation loses an important constraint. Polylogue can show the exact pre-compaction context refs, what survived, what was lost, what was later needed, and a suggested regrounding pack. The next agent receives that pack through the scheduler with citations, not as folklore.

## 8. Release G — live intake, capture, daemon, and interactive performance

Purpose: make the front door fast, live, and honest.

Primary beads:

`polylogue-20d` interactive performance epic.

`polylogue-20d.1` — CLI-to-daemon fast path over Unix domain socket.

`polylogue-20d.2` — defer heavy imports off CLI startup.

`polylogue-20d.5` — streaming reads.

`polylogue-20d.6` — live full-ingest catch-up latency and WAL shape.

`polylogue-20d.10` — post-filter efficiency.

`polylogue-20d.12` — daemon result cache and post-ingest warming.

`polylogue-20d.13` — daemon push channel.

`polylogue-20d.14` — named latency budgets and regression gates.

`polylogue-20d.15` — bulk ingest throughput and resource envelope.

`polylogue-th0` — interactive-surface test harness.

`polylogue-yeq` — advanced verification lanes.

Capture-specific work:

`polylogue-jlme` — capture extension reliability.

`polylogue-3v1.1` — concurrent browser-capture instances.

`polylogue-3v1` — extension reliability and status UX.

`polylogue-90y` — in-page overlay.

`polylogue-ptx` — browser-capture posting with attachments.

`polylogue-b1n` — webui-driven posting.

`polylogue-da1` — provider format-drift sentinel.

New blocker edges to add:

Make `polylogue-ptx` depend on `polylogue-kwsb.1`, `polylogue-83u.3`, `polylogue-83u.4`, and `polylogue-3v1.1`.

Make `polylogue-b1n` depend on `polylogue-ptx`, `polylogue-kwsb.1`, and the context/write safety layer if it can create assertions or messages.

Make live web features depend on `polylogue-20d.13` rather than polling.

Make daemon-cache features depend on convergence/invalidation proof.

Exit criteria:

CLI startup has a named budget and measured regression gate.

Hot-path CLI queries can route through a persistent daemon.

Common reads stream instead of blocking on full materialization.

Daemon push events exist for live UI features.

Cache invalidation is cursor-keyed or otherwise evidence-safe.

Capture extension reports spool health, completeness, and gaps.

Multiple extension instances are attributed and deduplicated.

Provider format drift is detected from live ingest shape changes.

Interactive tests exercise pty flows, completions, fuzzy pickers, and slow/missing routes.

Simulation:

The user runs `polylogue find failed tests` repeatedly while a live browser capture is appending messages. The first query may warm caches. The second is fast. The web UI receives a push event that new capture data landed. It does not poll blindly. If the daemon dies, the UI shows heartbeat/crash state rather than pretending the archive is stale or empty.

## 9. Release H — web evidence cockpit

Purpose: turn the web UI from a reader into a place where evidence is assembled, inspected, cited, and exported.

Primary beads:

`polylogue-bby.11` — web UI architecture v2.

`polylogue-bby.8` — perceived performance.

`polylogue-bby.15` — evidence basket to citable report to verified export.

`polylogue-yrx` — session changes view.

`polylogue-bby.1` — responsiveness under slow/missing routes.

`polylogue-bby.2` — query completions and expression explain.

`polylogue-bby.3` — aggregate analytics views.

`polylogue-bby.4` — live session tailing.

`polylogue-bby.5` — long-session navigation.

`polylogue-bby.10` — timeline and firehose.

`polylogue-bby.12` — session replay.

`polylogue-bby.13` — day page.

`polylogue-bby.6` — interaction debt cleanup.

`polylogue-bby.14` — pinboard; keep this P4 behind cockpit basics.

New blocker edges to add:

Make `polylogue-bby.11` block major web feature expansion. Architecture v2 should land before piling more fragile UI state onto the existing shell.

Make `polylogue-bby.3` depend on `polylogue-9l5.7` if it renders statistical measures.

Make `polylogue-bby.4` and `.10` depend on `polylogue-20d.13`.

Make `polylogue-bby.12` depend on lineage completeness signals from `polylogue-4ts`.

Exit criteria:

The web UI has a single typed API client generated from the same contract other surfaces use.

It shows daemon status, capture status, stale/partial/degraded state, and index convergence state.

It can build an evidence basket from sessions, messages, blocks, actions, query runs, result sets, and findings.

It can export a report with citations.

It can explain a query expression.

It can handle long sessions without freezing.

It can show live tailing when daemon push is available.

Visual smoke tests pass on seeded corpora.

Simulation:

The user searches for “failed verify then commit.” The web UI shows a result list, a session reader, and a basket. The user drags a failed command, the assistant’s next message, and a later commit into the basket. The report builder emits a Markdown report with content-hash citations. One citation has drifted since import; the report marks it with a drift warning. The user can still see the old evidence state if retained, or an honest “unresolved drift” state if not.

## 10. Release I — analytics, experiments, and measured learning

Purpose: let the archive answer “so what?” without overclaiming.

This release must not proceed before evidence honesty, query contracts, usage/cost honesty, and statistical metadata exist.

Primary beads:

`polylogue-1vpm.1` — delegation unit.

`polylogue-1vpm.2` — episode unit.

`polylogue-1vpm.4` — turn-pair unit.

`polylogue-1vpm` — work-graph units epic.

`polylogue-9l5.1` — outcome-conditioned analytics.

`polylogue-9l5.2` — cross-provider comparative analytics.

`polylogue-9l5.6` — tool-episodes projection.

`polylogue-9l5.7` — statistics substrate and measure registry.

`polylogue-9l5.8` — temporal analytics.

`polylogue-stc` — experiment hosting.

`polylogue-9l5.13` — activity spans materializer.

`polylogue-9l5.15` — triage frontier.

Then later:

`polylogue-9l5.10` process mining.

`polylogue-9l5.11` predictive advisories.

`polylogue-9l5.12` information-theoretic and graph measures.

`polylogue-9l5.3` pathology epidemiology.

`polylogue-9l5.4` token-economy analytics.

`polylogue-9l5.5` saved views.

`polylogue-9l5.9` survival analysis.

`polylogue-9l5.14` efficiency measure pack.

`polylogue-h10` prediction/calibration tracking.

P4 analytics stay parked:

`polylogue-9l5.16` trajectory quality index.

`polylogue-9l5.17` model-drift observatory.

New blocker edges to add:

Make all `polylogue-9l5.*` statistical outputs depend on `polylogue-9l5.7`.

Make `polylogue-9l5.7` depend on evidence-honesty beads: `polylogue-9e5.28`, `.29`, `.30`, `polylogue-cpf.5`, `.6`, and `polylogue-f2qv` usage/cost correctness where costs/tokens appear.

Make `polylogue-9l5.1`, `.2`, `.6`, and `.13` depend on `polylogue-1vpm` units where they use episodes, delegations, or turn pairs.

Make `polylogue-stc` depend on `polylogue-rxdo` query/result/finding identity and `polylogue-9l5.7`.

Make `polylogue-h10` depend on `polylogue-h6r`, because predictions need stable actor identity.

Exit criteria:

Every measure is registered.

Every measure declares sample frame, evidence tier, known confounds, coverage limits, and uncertainty form.

Outputs render footnotes for evidence tier and construct validity.

Cost and token analytics distinguish provider-stored, provider-priced, catalog-estimated, API-equivalent, subscription-credit, cached, uncached, reasoning, and completion lanes where applicable.

Experiments have declared arms, preregistered metrics, paired analysis where appropriate, raw per-pair tables, and cold-reader evidence.

No “magic score” is presented as truth. Composite scores are reward-shaping or triage aids only.

Simulation:

The user asks, “Which agent workflows are most wasteful?” Polylogue does not answer with a single vibe score. It identifies episodes, tools, retries, test failures, duration, costs, and outcomes. It says: “Storage migration episodes have high retry density and high time-to-green. Evidence tier: structural tool calls plus test results. Confounds: large sessions and partial capture. Here are the five episodes.” The user can open each episode and inspect exact evidence.

## 11. Release J — embeddings and semantic retrieval

Purpose: make vector search earn its cost.

Primary beads:

`polylogue-mhx.1` — provider abstraction.

`polylogue-mhx.2` — embedding target policy.

`polylogue-mhx.3` — retrieval quality eval lane.

`polylogue-mhx.4` — semantic recall leg in context compilation.

`polylogue-mhx.5` — semantic analytics surfaces.

`polylogue-mhx.6` — storage/spend efficiency.

`polylogue-mhx` — embedding substrate epic.

Related beads:

`polylogue-0k6` — changed-text full-replace regression.

`polylogue-0ns` — bound embedding work in large sessions.

`polylogue-37t.5` — local OpenAI-compatible embedding lane.

New blocker edges to add:

Make semantic context injection depend on `polylogue-37t.11`.

Make semantic analytics depend on `polylogue-9l5.7`.

Make embedding storage/spend optimization depend on retrieval quality measurement, not the other way around.

Exit criteria:

The system can run with no embedding provider.

The system can run with local or cloud OpenAI-compatible providers.

Embedding targets are explicit: what gets a vector, at what grain, and why.

Changed text invalidates and replaces vectors correctly.

Large sessions do not trigger unbounded embedding work.

FTS, vector, and hybrid retrieval are compared before vector claims are made.

Semantic retrieval included in agent context carries evidence refs and reason codes.

Simulation:

An agent asks for prior sessions similar to “blue-green rebuild swap bug.” Polylogue retrieves candidates with hybrid search. It shows whether each candidate came from text match, vector similarity, or both. The scheduler chooses three examples under budget and logs why the other candidates were excluded.

## 12. Release K — interop, origin breadth, and export

Purpose: make Polylogue a two-way evidence system, not a roach motel.

Primary beads:

`polylogue-l4kf` — ecosystem interop epic.

`polylogue-2qx` — OriginSpec.

`polylogue-l4kf.1` — `polylogue-export` origin and content-hash idempotent envelope.

`polylogue-l4kf.2` — federation manifest and selective sync.

`polylogue-l4kf.3` — git notes, PR/issue footers, SARIF pathology export.

`polylogue-r47` — Obsidian/PKM export.

`polylogue-4g5` — HPI/Promnesia.

`polylogue-611` — Grok importer.

`polylogue-fs1.8` — Nous Chat browser-capture adapter.

`polylogue-ale` — external link archival.

`polylogue-7k7` — research-tooling export.

`polylogue-uiw` — target origin set and generic OpenAI-chat-shape detector.

`polylogue-wmj` — OTel GenAI trace export.

`polylogue-0cg` — OTel GenAI ingest.

Hermes bridge:

`polylogue-fs1`, `.2`, `.3`, `.4`, `.5`, `.6`, `.7`, `.9`, `.10`.

Codex/Claude related:

`polylogue-ox0`.

`polylogue-t0p`.

`polylogue-7aw`.

`polylogue-da1`.

New blocker edges to add:

Make every new origin depend on `polylogue-2qx`.

Make `polylogue-l4kf.1` depend on `polylogue-svfj` and `polylogue-rxdo` object identity.

Make `polylogue-l4kf.3` depend on `polylogue-rxdo.4`, because outbound findings need evidence-backed finding assertions.

Make federation depend on content-hash idempotent export/import.

Make research export depend on fidelity declarations and leakage gates.

Exit criteria:

Each origin has detector, parser, fixture, schema mapping, docs, and fidelity declaration.

Ambiguous origin dispatch is deterministic and tested.

Provider format drift is detected.

Export/import of Polylogue’s own interchange envelope is content-hash idempotent: importing an export does not create duplicate facts.

Outbound citations work in developer-native surfaces: git notes, PR/issue footers, SARIF-like outputs.

Interop exports preserve evidence refs, caveats, and drift states.

Simulation:

A user exports a package of sessions, findings, and report artifacts to a collaborator. The collaborator imports it into another Polylogue archive. Shared content hashes prevent duplication. Findings remain linked to evidence. Missing private blobs are marked as intentionally omitted, not silently lost. Later, a PR footer links to the same finding object.

## 13. Release L — external legibility, demos, and launch

Purpose: make a stranger understand, run, and cite Polylogue.

Primary beads:

`polylogue-3tl` — external legibility epic.

`polylogue-3tl.12` — README de-meta pass.

`polylogue-3tl.13` — schema-versioning docs reconciliation.

`polylogue-3tl.16` — public claims ledger.

`polylogue-3tl.4` — findings publishing lane.

`polylogue-3tl.7` — install matrix.

`polylogue-3tl.9` — docs/visuals ownership.

`polylogue-3tl.15` — anti-grep proof card.

`polylogue-3tl.6` — normalized session model as interchange schema.

`polylogue-3tl.8` — GitHub surface polish.

`polylogue-3tl.10` — launch kit.

`polylogue-ttu` — docs information architecture.

`polylogue-6bu` — docs-site verification lane.

`polylogue-6l6` — docs/theme/release proof polish.

`polylogue-y0b` — generated codebase atlas.

`polylogue-45i` — Datasette lane.

Demo portfolio:

`polylogue-212.1` — post-hoc forensic Q&A.

`polylogue-212.2` — “The receipts.”

`polylogue-212.3` — “Where did the money actually go.”

`polylogue-212.4` — “Behavioral archaeology.”

`polylogue-212.8` — honesty anti-demo.

`polylogue-212.6`, `.7`, `.9` — continuation, finding packet, fable-as-foreman demo material.

Experiment rerun:

`polylogue-cfk` — re-run two-arm uplift with freshness-fixed packs.

New blocker edges to add:

Make `polylogue-3tl.10` depend on `polylogue-cfk`, `polylogue-3tl.16`, `polylogue-3tl.7`, `polylogue-3tl.4`, and the demo portfolio.

Make every demo with numeric or behavioral claims depend on trust-floor evidence beads.

Make public README claim publication depend on the claims ledger.

Make normalized session schema publication depend on OriginSpec and read/evidence refs.

Exit criteria:

The README first screen says what category Polylogue owns: local system of record for AI work.

A stranger can run the one-command demo successfully.

A stranger can cite a finding URL.

Public claims ledger lists every README/launch claim with status and evidence ref.

Install works across the declared package managers and operating systems.

Docs site has link integrity and cache verification.

At least two evidence artifacts are published.

At least one anti-demo shows that Polylogue can say “not supported” instead of forcing a finding.

A cold-reader pass confirms an outsider can recover the product purpose, proof path, and claim boundaries.

Simulation:

A new reader lands on GitHub. They do not see “chat viewer” or “AI memory thing.” They see: “Polylogue is a local system of record for AI work.” They run the demo. It imports a small trace, queries failed tool calls, builds a citable report, and shows why grep cannot answer the same question. The public claims ledger marks “improves agent performance” as capability/under-test unless the refreshed uplift experiment supports a stronger claim.

## 14. Release M — substrate consolidation and codebase simplification

Purpose: pay down architectural duplication only where it removes real delivery risk.

Primary beads:

`polylogue-a7xr` — substrate consolidation.

`polylogue-hiu` — collapse storage twins onto sync core behind async adapter.

`polylogue-pf1` — sync/async divergence diff.

`polylogue-exb` — layering: substrate rings import API facade.

`polylogue-1fp` — facade decomposition.

`polylogue-a7xr.5`, `.10`, `.11`, `.12`, `.14`, `.15`, `.16` — specific consolidation chores.

`polylogue-dab`, `.1` — run-projection cache materialization cleanup.

`polylogue-c9y` — package topology legibility.

`polylogue-ca4` — DuckDB optional OLAP decision.

`polylogue-dx1` — daemon HTTP substrate decision.

`polylogue-f94` — kill or commit TUI.

`polylogue-utf` — devtools surface economy.

`polylogue-4822` — curated SDK and frozen public models.

New blocker edges to add:

Make major consolidation depend on the read/evidence contract. Do not refactor the old shape deeply if the contract is about to move.

Make `polylogue-hiu` depend on `polylogue-pf1` and `polylogue-exb`, as already implied, and also on migration/rebuild proof from Release B.

Make SDK publication depend on stable public object refs and normalized session schema.

Exit criteria:

There is one storage core with explicit async boundary.

Public models are frozen at the external-consumer boundary.

God modules are split along actual capability seams.

Dead abstractions are deleted or adopted.

Developer commands are usage-ranked and consolidated.

The codebase atlas is generated and drift-checked.

This release should not be allowed to consume the safety budget. It becomes valuable after the contracts stabilize.

## 15. Release N — horizon and vision work

Purpose: keep speculative work visible without letting it distort the implementation spine.

Parked or late-gated examples:

`polylogue-2jj` — IssueBench.

`polylogue-c36` — native compilation probe.

`polylogue-gqx` — desktop presence.

`polylogue-lu1` — ambient theming.

`polylogue-bby.14` — pinboard.

`polylogue-9l5.16` — trajectory quality index.

`polylogue-9l5.17` — model-drift observatory.

`polylogue-fnm.9` — pipeline-as-subquery composition.

`polylogue-l4kf.2` — federation.

`polylogue-l4kf.3` — outbound provenance exports.

These are not bad. They are late. Their entry condition is that the evidence floor, read contract, storage floor, and public claim discipline already exist.

## Implementation packets

The work should be executed as PR-shaped packets, not giant epics.

A good packet has:

One primary bead.

Optional tightly related child beads.

One migration decision, or none.

One proof artifact.

One verification command set.

One rollback/degrade note.

One user-visible behavior change.

One documentation update if a public contract changed.

Recommended packet structure:

Packet 1: verification classification  
Primary bead: `polylogue-s7ae.6`  
Output: full verify log, failure classification table, fixes or pre-existing references.

Packet 2: blob cleanup safety  
Primary beads: `polylogue-8jg9.4`, `polylogue-8jg9.2`  
Output: leased-blob race fixture, doctor path safe behavior, no deletion of in-flight blobs.

Packet 3: evidence-honest numbers  
Primary beads: `polylogue-9e5.28`, `.29`, `.30`  
Output: product-registry audit coverage, field-level evidence contracts, prose-derived caveats.

Packet 4: time honesty  
Primary beads: `polylogue-cpf.5`, `.6`  
Output: weakest-source propagation tests, clock seam, `sort_key_ms` audit table.

Packet 5: daemon/capture security  
Primary bead: `polylogue-kwsb.1`  
Output: central Host/Origin/token/spool controls, negative security tests, extension still working.

Packet 6: assertion-write safety  
Primary bead: `polylogue-37t.15`  
Output: all non-user writes candidate/non-injected, rejected candidate cannot be self-revived by agent.

Packet 7: missing blob debt  
Primary beads: `polylogue-83u.4`, later `.2`, `.3`, `.6`  
Output: classification table, restored subset, irrecoverable decision record.

Packet 8: read algebra cutover  
Primary beads: `polylogue-4p1`, `.1`, `polylogue-t46.3`, `polylogue-jnj.1`  
Output: CLI/daemon/MCP parity tests.

Packet 9: evidence object refs  
Primary beads: `polylogue-rxdo.1`, `.2`, `.3`, `.4`, `polylogue-svfj`  
Output: query-run/result-set/finding refs and content-hash citation resolver.

Packet 10: scheduler and coordination proof  
Primary beads: `polylogue-37t.12`, `.11`, `polylogue-s7ae.3`, `.5`  
Output: context ledger, scoped message proof, two-agent worktree demo.

Packet 11: variants substrate  
Primary beads: `polylogue-0v9p`, `polylogue-arso`, `polylogue-rlsb`, `polylogue-d4zk`  
Output: translated/simplified session fixture with alignment.

Packet 12: blue-green rebuild  
Primary beads: `polylogue-20d.15`, `polylogue-b5l`, `polylogue-1xc.8`  
Output: no partial-ready state, generation swap, resource envelope.

Packet 13: web evidence basket  
Primary beads: `polylogue-bby.11`, `.15`, `.8`  
Output: basket, report, export, visual smoke.

Packet 14: stats registry  
Primary beads: `polylogue-9l5.7`, `polylogue-stc`  
Output: registered measures with evidence tier, sample frame, uncertainty, confounds.

Packet 15: public proof  
Primary beads: `polylogue-3tl.16`, `.4`, `.7`, `.12`, `polylogue-cfk`  
Output: claims ledger, published finding, install matrix, refreshed uplift report.

## Verification matrix

The plan needs a standing verification suite organized by failure class.

Safety verification:

Host/Origin/CSRF/token tests.

Capture token forgery tests.

Spool bound tests.

Secret/excision tests.

Agent-write candidate/non-injected tests.

Rejected-candidate resurrection tests.

Data-integrity verification:

Blob lease race tests.

Blob reference resolver tests.

Missing blob classifier tests.

Restore drill.

SHA-256 byte verification.

Content-hash citation drift tests.

Storage verification:

Schema migration dry runs.

Blue-green derived-tier rebuild tests.

Partial-index readiness tests.

WAL/resource envelope tests.

Large synthetic corpus test.

Live-copy probe where safe.

Query verification:

CLI/daemon/MCP/Python parity tests.

Query grammar metamorphic tests.

Projection/render parity tests.

Pipeline support matrix.

Set algebra equivalence tests.

Lineage verification:

Branch matrix.

Shared-prefix dedup.

Physical/logical count comparison.

Compaction boundary derivation.

Completeness signal tests.

Subagent/main-session distinction tests.

Analytics verification:

Measure registry tests.

Empty backing row tests.

Unknown/null density tests.

Sample-frame footnote tests.

Uncertainty rendering tests.

Cost/token lane partition tests.

UI verification:

Reader visual smoke.

Slow/missing route behavior.

Long-session navigation.

Live tail with daemon push.

Evidence basket export.

Interactive pty/completion/fuzzy-picker tests.

Public proof verification:

One-command demo.

Docs link integrity.

Claims ledger coverage.

Install matrix.

Cold-reader pass.

## Exact added dependency policy

The dependency graph should be strengthened with these rules.

Public reports depend on evidence honesty. Anything that publishes a number, rate, cost, trend, behavioral finding, or leaderboard depends on `polylogue-9e5.28`, `.29`, `.30`, `polylogue-cpf.5`, `.6`, and relevant `polylogue-f2qv` cost/usage beads.

Agent context depends on write safety. Anything that injects context into an agent depends on `polylogue-37t.15` and `polylogue-37t.11`.

Agent advisories depend on daemon safety. Anything that listens live and advises an agent depends on `polylogue-kwsb.1`, `polylogue-d1y`, and daemon liveness/performance work.

Browser posting depends on capture security and blob integrity. `polylogue-ptx` and `polylogue-b1n` depend on `polylogue-kwsb.1`, `polylogue-83u.3`, `polylogue-83u.4`, and spool/concurrency work.

Web evidence export depends on object refs. `polylogue-bby.15` depends on `polylogue-rxdo.*`, `polylogue-svfj`, `polylogue-4p1`, and `polylogue-fnm.11`.

Analytics depend on statistical metadata. `polylogue-9l5.*` outputs depend on `polylogue-9l5.7`.

Experiments depend on query/result identity. `polylogue-stc` depends on `polylogue-rxdo.2`, `.3`, and `polylogue-9l5.7`.

Interop depends on OriginSpec. New importers/exporters depend on `polylogue-2qx` or must explicitly justify why not.

Federation depends on idempotent export/import. `polylogue-l4kf.2` depends on `polylogue-l4kf.1`.

Outbound developer citations depend on finding assertions. `polylogue-l4kf.3` depends on `polylogue-rxdo.4`.

Compaction regrounding depends on scheduler control. `polylogue-gjg.4` depends on `polylogue-37t.11`.

Public launch depends on claims ledger and demos. `polylogue-3tl.10` depends on `polylogue-3tl.16`, `polylogue-3tl.4`, `polylogue-3tl.7`, `polylogue-cfk`, and demo portfolio closure.

## What not to do

Do not implement web cockpit features before the read/evidence contract is stable.

Do not implement analytics dashboards before the measure registry exists.

Do not publish launch claims before the claims ledger exists.

Do not add new capture/posting channels before daemon/capture security lands.

Do not compress blobs before missing blob debt is classified and byte integrity is proven.

Do not let agents inject context before write safety and the scheduler exist.

Do not refactor large storage modules before the migration/rebuild safety floor is in place.

Do not treat P4 vision beads as blocked implementation. They are product direction until their gates open.

Do not close epics because the “main idea” landed. Close them only when terminal acceptance passes.

## Final precise end state

After the adjusted plan completes, Polylogue is no longer merely a local archive or chat-search tool.

It is a local evidence operating layer for AI work.

It captures raw AI work from multiple origins.

It preserves source evidence and blob bytes honestly.

It classifies missing evidence rather than hiding it.

It separates durable truth from derived indexes.

It rebuilds derived tiers without showing partial archives as ready.

It has one read contract across CLI, daemon, MCP, Python API, web UI, reports, and docs.

It has stable object references for sessions, messages, blocks, actions, queries, query runs, result sets, findings, variants, annotations, cohorts, analyses, episodes, delegations, and artifacts.

It has content-hash citation anchors with drift states.

It can build citable evidence baskets and reports.

It can say “not supported” when evidence is missing.

It can run live while agents work.

It can coordinate multiple agents in one repo without tracker lock-in.

It lets agents contribute candidate memory, variants, findings, and messages without letting them self-promote into trusted user memory.

It compiles bounded, cited, deterministic context packs.

It records why context was included or excluded.

It models session lineage, branches, shared prefixes, physical sessions, logical sessions, compactions, and subagents truthfully.

It can recover from compaction loss by showing what was forgotten and generating regrounding packs.

It supports translated, simplified, transliterated, and summarized content variants without confusing transformed text with original evidence.

It measures work outcomes, costs, retries, tool failures, episodes, and experiments with uncertainty and construct-validity metadata.

It imports from more origins through declared origin contracts.

It exports evidence back out through citable interchange, developer-native surfaces, research formats, and knowledge tools.

It has a public README, demo, claims ledger, install matrix, published findings, and cold-reader proof that a stranger can understand, run, and cite the system.

The project state after this adjustment is implementation-ready in the sense that there is a clear path from “urgent trust repairs” to “publicly legible system of record,” with each later layer blocked on the foundation it needs. The original bead order remains useful, but the real plan is now gated, phased, and proof-driven.

---

