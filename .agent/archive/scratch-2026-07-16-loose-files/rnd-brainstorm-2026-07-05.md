---
created: "2026-07-05"
purpose: "Parallel R&D brainstorm (Opus ultrathink + 8 grounded zone-agents) — ideas + GPT-pro prompt fan-out for the polylogue program. Scope deliberately orthogonal to the swarm2 daemon/composer/DSL/cost/lineage set."
status: "active"
project: "polylogue"
---

# Polylogue R&D brainstorm — 2026-07-05

Inputs: my ultrathink pass (68 ideas, 8 zones) + 8 background zone-agents grounded in
live beads/code/raw-log. Deduped below. Scope avoids what swarm2 already owns
(hot-daemon/UDS/thin-client/lifecycle, CLI-thinning/PolylogueService contract,
live-preview composer, query-DSL grammar/set-algebra/projection, cost-token math,
lineage dedup, attachment bytes, scale/WAL, bead-set meta-audit).

Reintegration note (operator-corrected): GPT-pro webui chats are **auto-captured into
the archive by the browser extension** (extension → receiver :8765 → spool → daemon →
archive). There is **no manual export step**. The flywheel is: lynchpin bundle → fan
out chats → they self-capture → distill into beads → coding agents execute. The only
manual work is the *analytical* half (link captured R&D sessions to lanes; distill).

---

## Convergent themes (hit independently by ≥2 agents — highest signal)

1. **Queries / findings as first-class stored objects.** self-ref: `query:<id>` ObjectRef,
   content-hashed result-set snapshots, query→query dependency edges, standing-queries as
   change-detectors. product: `finding` as an `AssertionKind` (claim + producing-query +
   provenance + URL, `find kind:finding`), findings-as-tests manifest. UX: content-hash
   citation anchors as the atom everything cites. → **the convergent frontier.**
2. **Construct-validity as a queryable substrate, not a markdown header.** trust:
   `finding_provenance` table + sample-frame drift detector (re-run each finding's COUNT on
   every converger pass, auto-stamp "population 412→1,088, re-run before citing"). product:
   findings-as-tests (CI re-runs the number ± tolerance). content: summary must carry
   citation edges (coverage>0) or be rejected at write. Brand line: **"every number resolves
   to bytes / no regex over prose."**
3. **Content-hash citation anchors that survive re-ingest.** UX + ecosystem (CIF) both landed
   on stable `session:message:block` anchors (not `position`) as the load-bearing atom — the
   honest fix for the fork-position-shift + recovery-digest-fabrication class.
4. **The "so what" analytics is mostly one materializer away.** The keystone substrate already
   exists (`tool_duration_ms`, `tool_result_is_error`/`exit_code` v16, `actions` view,
   `workflow_shape`, `terminal_state`, `session_links`). Gap-taxonomy, retry-storm,
   cost-per-outcome, delegation-yield are new *measures*, not new plumbing.
5. **Recursive-safety gate.** self-ref + trust: agent-authored assertions default to
   CANDIDATE + `inject:false` until operator-promoted. Load-bearing now that the archive
   auto-captures and feeds itself — prevents the archive laundering its own hallucinations
   back into context (the recovery-report fabrication class, structurally).
6. **Corpus-compaction pack (`find <q> | compact`).** analytics: token-budgeted, decision-dense
   digest (drop tool-spam via `material_origin`, keep human turns + decisions + error→fix +
   outcomes, dedup replayed prefixes, fit a context window, emit a drop-manifest). This is the
   flywheel enabler — it packages the archive's own sessions for the next GPT-pro lane.

---

## Ideas by zone (deduped; NEW = no bead yet; else epic/bead anchor)

### A. Product identity / positioning / legibility (3tl, 212, 9e5)
- Hero-finding discipline: ONE robust+surprising+clickable number as the spearhead (README hero
  + launch post + lead demo all point at it), not a finding *catalog* with no front door. — NEW
- `finding` as first-class AssertionKind; findings-as-tests provenance manifest (stale finding
  fails the devloop, not the reader). — NEW / sibling of 3tl.9
- Head-to-head "lie detector" demo: same question to naive grep/log-tailer (fabricates from
  prose) vs polylogue (answers from structure or refuses). — 212
- Interactive "click-the-number" Artifact: click a headline stat → decompose number→query→
  structural field→raw bytes. The moat as a shareable link, 10-second cold-reader proof. — NEW
- Claims-ledger dogfood: every self-marketing claim is itself a `finding` with evidence status
  (proven/capability/aspirational), published; radical-honesty as brand. — NEW
- Category-defense bake-off vs LangSmith/Langfuse/Helicone (live/single-provider/online) and
  Rewind/QS (surveillance) — offline, cross-provider, reconstructed-from-exports. — 3tl
- [RADICAL] Flight-recorder / black-box positioning: the forensic recorder you consult after a
  crash (post-hoc, structural, "what did the agent actually do"). No competitor owns it. — NEW
- [RADICAL] Public honesty benchmark: claim-vs-evidence as a NAMED public eval + leaderboard
  others run their models against; personal archive → public science instrument. — NEW
- [RADICAL] Prompt/meta-workflow distillery: "your history is training data for how you should
  work" — mine history into better parametrized prompts (operator's stated dream). — NEW
- `polylogue tour` / `explain-project`: the CLI narrates its own category + runs the hero demo
  for a cold reader. — 3tl
- Self-hosting proof badge: polylogue's own repo as reference deployment, live "N of our
  decisions are citable" counter. — NEW
- Weekly auto "state of my AI work" report — recurring proof artifact (dogfood + marketing). — 9l5/212
- Ingest Simon Willison's `llm` SQLite logs as a source — manufactures its own citation path
  into the Datasette/`llm` community. — l4kf
- Honesty demo: showcase what polylogue *refuses* to claim (unfetched attachments, stale rows). — 9e5/83u

### B. Deep analytics / "so what" (9l5) — most are one measure away
- Gap-taxonomy materializer: partition each session's wallclock into typed spans (edit/compile/
  test/rebuild/model-thinking/human-idle) from inter-event gaps + tool identity + command regex.
  The literal raw-log-1841 ask. — 9l5.8
- Long-call time-sink ranking: tool calls whose wall-duration > N min, labeled by command class
  (full-test/nix-switch/rebuild), per session, rolled up per repo. — NEW
- Retry-storm / guess-and-check detector: edit→same-target-test→edit cycles where
  `tool_result_is_error` stays true and error signature doesn't change = churn. — 9l5.9
- Agent-efficiency score with peer baseline: `tool_active_minutes/wallclock` z-scored against a
  per-repo, per-`workflow_shape` cohort (judge like-work, not global). — 9l5.2/.7
- Tool-error recovery latency: first-error → next-success on same tool/target, per model. — NEW
- Context-gluttony index: whole-file `Read` (no offset/limit) + bytes ingested vs targeted
  `Grep`/`Glob` (raw-log-2001 "skillful grep"). — NEW
- Cost-per-outcome join: `terminal_state` × `session_costs` → "$ per *successful* session",
  "spend burned on abandoned+stuck". The sharpest so-what. — 9l5.1
- Delegation yield (subagent ROI): child cost/tokens/wallclock vs parent terminal success. — NEW
- Rework half-life / churn survival: time until a touched file is re-edited later; short = thrash. — 9l5.9
- Measure→prose narration (local-LLM): `analyze --narrate` passes only the numeric result to
  gemma/deepseek → grounded prose, never invents facts. — 9l5/mhx.5/1944
- Pathology epidemiology over time: prevalence per ISO-week × model + changepoint on upgrades. — 9l5.8
- Wasted-test-run detector: long test span that produced no caught error = wasted minutes
  (quantifies the blanket-run anti-pattern). — NEW
- Abandonment autopsy clustering: cluster stuck/abandoned by (last-tool, last-error-signature). — 9l5.1
- Context-switch / interruption tax: abandon-A→spawn-B→return-A patterns; re-orientation cost. — NEW
- Thinking-vs-doing drift: `thinking_duration_ms` vs `tool_duration_ms` trended per model — early
  "this model got worse for my work" signal. — 9l5.8
- Model head-to-head from real corpus: same task-shape across models — cost/turns/errors/success. — 9l5
- Per-session "so what" scorecard `reader_panel`: efficiency + gap-sparkline + retry count +
  cost-per-outcome + delegation-yield in one slot. — NEW

### C. Self-referential substrate (s7ae, 37t, rii) — the frontier
- Promote saved queries to `query:<id>` ObjectRef so any assertion can target/cite a query
  (a query accrues annotations/judgments/supersession). — 37t / NEW
- Materialized result-set snapshots in ops.db `(query_id, frontier_ids, result_hash, ran_at)` →
  `read query:abc --diff` ("3 in / 1 out since last run"); assertions attach to a frozen answer. — NEW
- Query→query dependency edges + invalidation (storage layer, not grammar). — NEW
- Standing queries as change-detectors: `context_policy_json {watch:true}` re-runs on quiet
  window; a delta writes a CANDIDATE assertion (archive's own alerting). — feeds rii
- Self-mine BLOCKER/LESSON/DECISION assertions → discovered beads (cluster same-target,
  `bd create` with evidence_refs). — 37t
- Recursive-safety gate: agent-authored assertions default CANDIDATE + `inject:false`. — 7aw (load-bearing)
- Ingest polylogue's own dev sessions into a `self` workspace (cwd-prefix == POLYLOGUE_ROOT). — NEW
- Read-access log as a queryable unit source: most re-read sessions, saved queries returning
  nothing, recall packs never opened (dead-memory detection); the signal the scheduler needs. — 37t
- Context scheduler: budget-aware inject selection ranked by staleness × attention ×
  topic-proximity (embedding of current session); decay by `staleness_json`. — 37t
- Programmable `context_policy_json` windows: `{inject, when:[start|on_topic|on_error], ttl_days,
  max_injections, cooldown}` — assertion as scheduled memory primitive. — NEW
- Annotation recipes: parameterized assertion templates (a postmortem recipe auto-attaches
  decision/blocker/lesson skeletons pre-filled from `session_phases`/`get_pathologies`). — NEW
- Belief-history read over `supersedes_json`: `belief_timeline(target_ref)` shows reversal history
  so an agent stops re-litigating settled reversals. — NEW
- Live recall packs bound to a `query:<id>` (recompute membership live, keep human narrative). — NEW
- Semantic dedup of saved queries via embeddings (avoid a 40th near-duplicate view). — NEW
- Reverse "annotation density" insight (session → assertions-about-it): most-annotated,
  unresolved-blockers, zero-human-touch. — rii
- Blackboard posts carry `query:<id>` refs → coordinator posts a query, workers run-and-report
  (notepad → task bus). — s7ae
- Synthetic "meta-session" of the archive's own runtime (ingest/convergence/read events as blocks)
  → every read-model applies to polylogue operating on itself, zero new insight code. — NEW

### D. Ecosystem / interop / breadth (l4kf, 7aw, fs1)
- Publish **CIF (Conversation Interchange Format)**: content-addressed transcripts with portable
  `session:message:block` anchors + per-source fidelity declaration; the neutral hub spec every
  export/import profile targets. — l4kf
- `polylogue-export` as a first-class re-ingestable Origin: round-trip reconstructs identical
  ids → free correctness invariant + federation primitive. — NEW
- `.well-known/ai-sessions` federation manifest + selective content-hash sync (local-first peer
  sharing, dedup via idempotency). — NEW
- Emit `git notes` (refs/notes/polylogue) attaching a session citation to the commit it produced
  (`git log --notes=polylogue`). — NEW (outbound counterpart to 7xv)
- `polylogue cite` → gh PR/issue provenance footer + reverse `session_link` (PR ↔ session). — NEW
- Export mined pathologies as **SARIF** → GitHub code-scanning surfaces agent-behavior defects. — NEW
- Signed session attestation `(content_hash, anchor, sig)` → tamper-evident citations. — NEW (pairs ale)
- MCP sessions/messages/blocks as **resources** (`resource://` + subscriptions), not only tools —
  the embeddable, live-updating continuity primitive. — NEW
- Recall-packs / saved-views as MCP **prompts** (slash-command surface) — zero-friction injection. — NEW
- `llms.txt`/`AGENTS.md` continuity artifact for MCP-less harnesses (Aider/Cursor/Zed read a file). — NEW
- Attach Sinex/ActivityWatch ground-truth (window-focus, Atuin, file-activity) to a session as
  observed-events by time window — transcript says what the agent did; this says what the human did. — NEW
- Cross-origin "same investigation" stitching (repo + time-window + embedding), edge=thematic not
  prefix-sharing: Cursor→Claude Code→ChatGPT as one logical unit. — NEW
- **Cursor parser** (chats in `state.vscdb` SQLite, binary keyed blobs — a SQLite-shaped acquire
  path, unlike every current file detector). Largest missing corpus. — NEW
- Continue.dev dev-data JSONL (purpose-built for export); Zed via ACP (source + live lane);
  OpenCode/Warp stores; Aider `.aider.chat.history.md` (first prose-shaped parser). — l4kf
- Datasette read-only publish of the archive (already SQLite; QS/Willison ecosystem for free). — NEW
- IDE Timeline/SCM provider: "which sessions touched this file" in the editor. — NEW
- Eval-corpus export: promptfoo test-cases + SWE-agent `.traj` downstream of CIF. — 7k7/fs1.5
- Wire grok-export (reserved origin, no parser). — 611/l4kf

### E. Content transform / multimodal / embeddings (4smp, mhx)
- `mechanical` vs `generative` variant provenance axis (OCR/transliteration trusted 100%; LLM
  translation/summary lossy). The trust axis is the point. — 4smp
- Variant staleness keyed to `source_content_hash` (mark stale/orphaned on source hash change,
  never auto-repaint). — 4smp
- Summary variants must carry citation edges (coverage>0) or be rejected at write. — 4smp
- Coverage-gap "dark matter" rendering: the source spans a summary did NOT cover ("what got
  dropped"). — 4smp
- Faithfulness score via embedding drift (cosine between summary node and its cited source span). — mhx+4smp
- Per-block language detection (not per-session) driving translation defaults; exclude code/tool
  blocks by construction. — 0v9p
- Glossary-pinned translation: deterministic pre/post substitution of domain terms, span-preserving. — 4smp
- Image blocks → caption variant (searchable) + separate OCR variant with bbox alignment
  (mechanical) — two variant kinds over one image, never conflated. — 4smp+mhx
- Audio transcript as a first-class origin (per-block audio offsets, "jump to audio at this line")
  — the operator's oldest itch. — NEW
- Embedding eval harness mining ground truth from lineage: resume/fork pairs = semantically-near,
  few shared tokens → **free labeled positives** for recall@k (FTS vs vector vs hybrid). — mhx
- Multi-provider embedding shootout on a frozen probe set (Voyage vs local nomic/bge/gemma);
  second vec0 space per provider with hard provenance. — mhx
- Reranker as a measured post-retrieval stage (bge-reranker top-50→top-10), promoted only if it
  lifts precision@10 on the probe set. — mhx
- Tool-result-aware chunking: embed a distilled head+tail+error variant, keep raw block as evidence. — mhx+4smp
- Retrieval unit = `action` (embed call-intent + outcome jointly via the actions view). — mhx
- Semantic diff across a fork's divergent tail as topic-drift vector ("did the fork abandon the
  goal?"). — NEW
- Hierarchical coarse→fine retrieval (session-summary embeddings → action/message embeddings). — mhx
- `variant`/`coverage` as queryable DSL units (`sessions where variant:translation and coverage<0.8`). — 4smp

### F. Trust / verifiability / privacy (cpf, kwsb, 9e5)
- `finding_provenance` as a queryable `index.db` table (cursor-id/position, measure+version, git
  SHA, sample-frame predicate, run-date) + sample-frame drift auto-flag. — cpf/3tl.4
- `trust_class ∈ {OPERATOR, SYSTEM, QUOTED}` on every context-injectable row; the compiler cannot
  concatenate QUOTED (attacker-controllable) into an OPERATOR block without a visible fence + ref. — cpf/37t.11
- Injection-tripwire fixture: known injection strings must round-trip through `compile_context`/
  `compose_context_preamble` as fenced QUOTED or the test fails. — cpf
- Confidence + evidence-ref REQUIRED (NOT NULL) at assertion write. — cpf/9e5.1
- Secret-scan at the READ boundary (fold into read, per operator): emit spans as `secret_candidate`
  overlays with a reveal affordance; keep bytes durable, never re-surface a pasted key by accident. — kwsb/27m
- Secret candidates → judgment queue as assertions (never auto-redact); acceptance → excision,
  rejection → suppress pattern → labeled corpus tunes the detector. — 27m
- Verifiable excision with content-hash tombstone (records old content_hash, recomputes new;
  idempotency can't resurrect the removed span; auditor can prove exactly-this-hash-removed). — 27m
- One shared mutation-audit contract across excise/reset/delete/MCP-admin (actor, ref,
  dry-run-diff hash, reason, --yes provenance); fixes the reset-before-preview bug class. — kwsb/jnj.5
- Forget-audit reconciliation: converger invariant re-scans index/FTS/embeddings/blob for the
  removed hash and fails loudly on reappearance. — kwsb/27m
- Degradation-rung in every read payload envelope (full/partial-index/embeddings-stale/
  evidence-only); EVIDENCE-ONLY floor is sacred (raw reads always answer). — cpf
- Claim-vs-evidence as a standing property test over insight OUTPUT: no rendered insight string
  carries a quantitative claim whose backing rows are empty/NULL. — 9e5
- "Unverified candidate" as a first-class render mode distinct from "fact" (codifies the text-mined
  recovery-report fix into the type system). — 9e5
- Semantic-version-bump invalidation cascade for findings/assertions (old-version findings → stale
  + diff banner; append-only supersession). — cpf/3tl.4
- Adversarial self-audit shipped as `polylogue audit --adversarial <insight>` (generates the
  strongest counter-frame as `caveat` assertions before you cite). — 9e5
- Injected-context provenance receipt: `build_context_image` attaches a manifest of what it
  contains + why (source refs, trust classes, freshness per component). — cpf/37t.11
- `lossy_grouping` marker on the origin projection (GEMINI+DRIVE→AISTUDIO_DRIVE is non-injective). — 9e5

### G. UX beyond the composer (bby, jnj)
- Citable-report builder ("evidence basket"): accrete selected blocks into a live Markdown doc
  with provenance footnotes, exportable as the deliverable — the missing "report" end of bby. — NEW
- Content-hash citation anchors (`cite this block`, keyed on hash not position) + citation-integrity
  verifier (re-resolve on export, flag drifted/deleted/quarantined). — NEW
- Real force-directed lineage/topology graph (replace BFS text list): edge-kind coloring, quarantined
  cycle-break nodes flagged, branch-point highlighted, click-to-refocus. — complements bby.5
- Assertion/evidence overlay graph + marginalia layer in the reader (marks/corrections as sticky
  margin notes) — the durable user.db tier is currently invisible at read time. — NEW
- Year-heatmap calendar (contribution-graph) → click into day-page (bby.13); the navigable temporal
  index. — bby
- Unified temporal scrubber: one zoom spine over year-heatmap → firehose (bby.10) → replay (bby.12). — meta over bby
- `material_origin` texture gutter (minimap strip colored by authoredness) — reveals a 4k-msg
  session's human/assistant/tool texture. — extends bby.5
- Delegation call-tree render layout (nested collapsible cards, not flat inline splice; after #2545). — NEW
- Specimen-gallery browse mode (grid of session-glyph cards for visual triage of 200 sessions). — NEW
- Side-by-side fork/parent diff pane (surface `compare_sessions`, highlight only divergent tail). — NEW
- Ambient theming from pywal/`prefers-color-scheme` on the web reader (extend `il` palette to web). — jnj
- Progressive-enhancement / server-rendered read routes (readable on phone / `curl | pandoc`);
  bake into arch-v2 (bby.11) as a constraint. — feeds bby.11
- Obsidian-vault read-view profile (wikilink-cross-referenced Markdown, one file per session,
  lineage as `[[…]]` backlinks; reuses canonical-URL projection). — NEW/jnj
- Cmd-K command palette (kill the last `window.prompt()`, bby.6). — extends bby.6
- Voice recall loop (push-to-talk → local Whisper → DSL; TTS read-back of resume-briefs). — NEW
- "Now playing" ambient strip (reader header + optional Waybar module live-tailing daemon SSE). — rides bby.4

### H. Radical reframes / moonshots
- Polylogue as a local RL/eval environment: sessions = labeled trajectories (task, tool-calls,
  outcome, cost, corrections in user.db); `polylogue eval export --format atropos`. — fs1
- Verifiable-reward RL tasks auto-mined from CI-passing sessions (`tool_result_exit_code=0` → the
  verify command IS the reward). — eval-env
- Reward model from your corrections (CORRECTION assertions = human-preference dataset over agent
  behavior; train a tiny local scorer "would Sinity flag this turn"). — eval-env
- Session replay/reproduction: re-execute recorded tool-calls against a fresh checkout at the
  session's git SHA → determinism, "worked-then-breaks-now" regressions, RL rewards. First step: a
  dry replayer printing the executable plan (no execution). — replay
- Counterfactual re-run ("what if model X"): freeze human turns + initial repo state, re-drive with
  a different model, diff cost/turns/outcome. — replay
- Spec-cards per session (initial SHA, human intent, acceptance signal, final diff) → any session
  becomes a portable benchmark item without transcript leakage. — eval-env
- Trajectory quality index (latency + correction density + abandonment + tool-error + phase thrash
  → one 0–1 score) as RL reward-shaping AND personal dashboard. — eval-env/observatory
- Model-drift observatory: same task-shape across (model, month) → cost/turns/error drift. — observatory
- Pathology-driven auto-guardrails: recurring failures → injected `caveat` assertions ("last 3× you
  edited convergence_stages.py you broke X"). — resident (links #1498 cascade)
- Standing "resident intellect" agent mining the archive nightly (context = polylogue MCP +
  assertions; drops ≤5 findings/day via save_annotation). — resident
- Self-improving CLAUDE.md/skill synthesis from repeated corrections (weekly digest drafts a
  candidate memory snippet; operator reviews, never auto-commits). — resident
- Cross-project agent memory: `recall(task_hint)` MCP tool returning most-similar prior sessions +
  their corrections/lessons across ALL repos. — cross-project-memory
- Personal-model distillation substrate ("golden turns" subset as SFT/preference data). — personal-model
- Privacy-preserving cross-archive federation: exchange only derived statistics/embeddings/
  pathology-rates, never content. — federation
- Analysis-recipe library (parameterized DSL pipelines you name/share/run: `run-recipe <name>`). — recipes
- The umbrella: polylogue as substrate for a personal AI that knows your whole collaboration
  history (curated "resident" MCP profile bundling recall + assertions + lynchpin). — resident

---

## GPT-pro prompt fan-out (deduped, final)

Prepend the **standing depth contract** to every prompt:

> You have the forked context: the polylogue bundle + our swarm2 daemon/composer/thin-client
> designs + our discussion of queries/findings-as-objects, delegation units, annotation recipes,
> and browser-auto-capture. Ground every claim in the bundle (cite file:line / bead id); tag
> [evidence] vs [proposal]. Be opinionated — one defended recommendation + the runner-up you
> rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate
> change, (3) the full direction. End with **Open questions for the operator** and **What's
> missing** (a claim unverified, a source unread, a modality not run). Snapshot is dated; live code
> may have moved — flag assumptions to verify locally. Don't re-derive swarm2.

### Design [A] (analysis over the bundle)
A1. **"So what" analytics + gap-taxonomy materializer.** Design `session_gap_profile` partitioning
    each session's wallclock into typed spans (edit/compile/test/rebuild/model-thinking/human-idle)
    from inter-event gaps + tool identity + command classification; the gap decision table, the
    STRICT schema, registration as a measure (grouping×window×uncertainty) with construct-validity
    metadata, and a compact sparkline. Then the measure family it seeds (retry-storm, recovery
    latency, gluttony, cost-per-outcome, delegation-yield).
A2. **Queries+findings as first-class objects (the convergent frontier).** Storage+ref model:
    `query:<id>` and `finding` ObjectRef/AssertionKinds; content-hashed result-set snapshots (which
    tier?); query→query dependency edges + invalidation; standing-query change-detection emitting
    CANDIDATE assertions; findings-as-tests manifest (CI re-runs the number ± tolerance). DDL
    placement across the 5 tiers, invariants, read/write API — without touching the DSL grammar.
    Enumerate the recursive-loop failure modes.
A3. **Context scheduler + memory loop + recursive-safety gate.** Given a token budget, a topic
    embedding, a read-access log, and assertions with `context_policy_json`/`staleness_json`/
    `confidence`/`author_kind`: the ranking function, the programmable-window schema (when/ttl/
    max_injections/cooldown), and the gate that keeps agent-authored assertions out of context until
    operator promotion. Prove it prevents self-hallucination feedback.
A4. **Delegation graph as substrate.** The `delegation` unit + extraction from Claude-Code/Codex
    subagent structure + artifact/scratchpad linking + delegation-card projection + analytics DSL
    (`delegations where … | group by subagent_model_family | count`). Generalize the Fable demo.
A5. **Session replay/reproduction critique.** Re-run recorded tool-calls against the repo at the
    recorded SHA for determinism / RL rewards / model A-B. Enumerate the correctness/safety/
    nondeterminism problems (side effects, network, time, FS state), which session types are safely
    replayable, and the smallest trustworthy first slice (dry replayer).
A6. **Content-variant model + honesty invariants.** Typed variants over refs (translation/
    transliteration/simplification/summary/caption/OCR) with alignment edges (incl. partial),
    a `mechanical` vs `generative` provenance axis, `source_content_hash` staleness, coverage>0
    write invariant, and query surfaces that guarantee variant text is never returned as evidence.
A7. **Construct-validity as a live surface.** `finding_provenance` table + sample-frame drift
    detector + `trust_class` (OPERATOR/SYSTEM/QUOTED) on injected context + the injection-tripwire
    fixture. Make honesty a query + a regression gate, not a doc convention.
A8. **Redaction/forget done right.** Fold secret-scan into `read` (spans as `secret_candidate`
    overlays → judgment queue), verifiable excision with hash tombstone, one shared mutation-audit
    contract, and a converger reconciliation that proves no derived tier retained the removed hash.
A9. **Webui evidence cockpit: anchors → report.** Citation-anchor scheme (content-hash, survives
    re-ingest + fork-position shift), evidence-basket→citable-report data flow, and citation-
    integrity verifier (drift/quarantine handling). Anchor format, resolution algorithm, minimal API.
A10. **Unified temporal-navigation instrument.** Year-density heatmap + scrubbable firehose +
    per-day narrative + per-session replay under one zoom model; the zoom state machine, aggregation
    tiers (year→week→day→session→message), data contracts per level, and forks as timeline branching.
A11. **Corpus-compaction pack (`find <q> | compact`).** Token-budgeted, decision-dense digest for
    feeding an external LLM: block-selection scoring (drop tool-spam via material_origin, keep human
    turns + decisions + error→fix + outcomes), replayed-prefix dedup, per-session boundary markers,
    and a fidelity drop-manifest. This is the R&D-flywheel enabler.
A12. **Own-export as a first-class source + CIF round-trip + federation.** The case for/against making
    the archive's own export re-ingestable (round-trip id identity as a correctness invariant +
    federation primitive); where non-injective origin collapse or lossy fidelity declarations break
    it; manifest-pull vs feed-push federation.
A13. **Product spearhead + positioning.** Rank candidate hero-findings (authored-user-word share,
    replay-duplication %, subscription-vs-API cost gap, abandonment mortality, Codex reasoning
    inflation) on {robustness, surprise, clickability, resolves-to-bytes}; output the single
    spearhead + headline + reproduce command + 2 backups, plus the flight-recorder/honesty-benchmark
    framing vs named competitors.
A14. **Prompt/meta-workflow distillery.** From my highest-value past sessions, induce 5–8 general
    *parametrized* meta-prompts (params: repo, task-type, risk-tier) that would have beaten what I
    actually typed; for each, the pattern, the sessions it's distilled from, and a falsifiable A/B test.
A15. **The R&D flywheel (corrected).** Design the fan-out + reintegration loop given that browser-auto-
    capture already ingests GPT-pro chats: lane clustering, making each lane's captured sessions a
    retrievable cohort (title stub / tag), the recursive-safety gate on distilled assertions, and the
    distill→bead step. No manual ingest.

### Deepresearch [DR] (web)
D1. **Competitive/landscape scan + positioning.** Tools that archive/analyze AI-agent sessions
    (exporters, LLM-observability: LangSmith/Langfuse/Helicone, agent-trace, eval harnesses,
    Rewind/QS). White space for "system of record for AI work"; assess the flight-recorder + public
    honesty-benchmark framings and what naming lands in the Latent-Space/Karpathy discourse.
D2. **Local-LLM + embedding + reranker stack on a consumer box (13700K + ~10GB GPU).** Models for
    classify/summarize/narrate a 40GB personal corpus (gemma/qwen/deepseek/flash-lite) and embeddings/
    rerankers (nomic/bge-m3/gte vs Voyage baseline): decision matrix over quality (MTEB + code),
    footprint, throughput, quantization, licensing. + a self-labeled recall@k eval from fork/resume pairs.
D3. **Embedding & retrieval for long tool-heavy transcripts.** SOTA chunking, dense/hybrid/late-
    interaction, and eval for very long agent sessions (not chat-sized). Does vector beat BM25 here,
    and how do you prove it locally (mhx.3)? How methods degrade on code + tool-output vs prose.
D4. **AI-session harness + interchange-format landscape.** Where Cursor/Zed/Continue/Aider/OpenCode/
    Cline/Windsurf/Amp/Warp store sessions + schema shape + open-vs-reverse-engineered + stability;
    and the format standards (OTel GenAI, Zed ACP, llms.txt/AGENTS.md, promptfoo/inspect/SWE `.traj`).
    Which ONE interchange standard should a neutral local archive champion vs merely adapt to.
D5. **Portable content-addressed citation-anchor standard.** Prior art (W3C Web Annotation selectors,
    git notes, DOI/PID, IPFS/Nostr content addressing, Gwern link archival); what makes an anchor
    durable, tamper-evident, re-resolvable across independent archives; failure modes (re-parse drift,
    schema change, redaction).
D6. **Cost/pricing & subscription-credit models.** Current API vs subscription-credit accounting
    across Anthropic/OpenAI/Google; honest dual-view presentation; pricing-catalog sourcing (f2qv).
D7. **RL/eval-environment landscape + trajectory formats.** Atropos/verifiers/NeMo ATOF-ATIF/OpenAI
    evals/terminal-bench/SWE-bench: input schema, reward model, whether human-corrections are
    first-class, and how a local ~10k-session corpus (tool-calls, exit codes, corrections, SHAs)
    lowers in with minimal fabrication. The single most leverageable export target + exact JSONL shape.
D8. **Personal-model training/distillation from own history.** Corpus size/quality for useful SFT vs
    preference-tuning (DPO/reward-model) of a local model; data-quality gating; federation techniques
    for comparing behavioral statistics without sharing transcripts. A minimal pipeline over 40GB.
D9. **Agent memory & context-engineering prior art.** How leading agent frameworks manage long-term
    memory, context assembly, provenance — what the context scheduler (37t) should steal.
D10. **Construct-validity as a product feature.** dbt tests / Great Expectations / metric-store
    semantic layers / postmortem tooling: how best-in-class systems expose sample-frame, denominator
    provenance, and staleness so a number can't be silently miscited. Concrete design for auto-flagging
    findings whose population predicate drifted.
D11. **Verifiable deletion in append-only content-addressed stores.** Tombstoning, hash-of-removed,
    crypto-shredding, Merkle-consistency proofs; interaction with content-hash idempotency + derived-
    index rebuilds; what an auditor can prove without retaining bytes; resurrection failure modes.
D12. **Summary-faithfulness/attribution scoring.** Embedding-drift cosine vs NLI entailment vs
    citation-coverage; SOTA for detecting hallucinated/unsupported summary sentences cheaply + locally;
    degradation on code + tool-output.
D13. **Persisted queries as first-class objects — prior art.** Datalog/materialized-view engines,
    dbt lineage, Datasette/Steampipe saved queries, graph views, event-sourced read models: patterns
    for query→query invalidation, result-set snapshotting, standing-query alerting on single-writer SQLite.
D14. **Big-graph in-browser + progressive-enhancement + ambient theming.** Force-directed vs
    hierarchical node-link rendering at ~16k nodes; SPA-plus-hydration usable on mobile/curl; pywal/
    prefers-color-scheme piped into a locally-served app. Techniques, ceilings, failure modes.

Run split: fork the current GPT-pro session for continuation lanes (A2, A3, A4, A5, A11, A15, D9);
fresh chats + bundle for the rest. 14 [A] + 14 [DR] = 28 chats — well inside the quota budget.

---
---

# WAVE 2 — 14 agents (3 deepen the convergent frontier, 11 open new dimensions)

## New convergent themes (wave-2 — hit across lanes)

7. **The measure algebra is now explicit.** Analytics = a 5-tuple `⟨reducer(column) over unit-frame × grouping × window × comparison × uncertainty⟩` — the catalog is a *cartesian product*, not 16 hand-written `analyze` modes. Highest-leverage first: cache-amplification ratio, latency three-lane share, engaged-vs-wall efficiency, tool-mix entropy, credit-vs-API divergence. Every "cross-model" / "longitudinal" variant is a grouping/window swap on a base measure, never new code.
8. **Queries+findings-as-objects got a full spec.** `query:<hash>` ObjectRef (hash over the planned AST post-macro-expansion, mirroring content-hash idempotency); `AssertionKind.FINDING` reusing the pathology candidate→judge lifecycle verbatim; `result_sets` snapshots in the *derived* tier (droppable by `reset --index`); `query_edges` DAG persisted from the set-algebra EXPLAIN nodes; a `StandingQueryStage` in the converger; **findings-as-tests** (a promoted finding with `value.expected` becomes a re-runnable invariant). Multiple other lanes (search self-testing saved-queries, proactive standing-queries, attention frontier-as-query) independently assume this.
9. **Recursive-safety is a coherent subsystem, not a flag.** Across self-capture / proactive / attention / queries-deep: agent-authored assertions default CANDIDATE+inject:false; a **closed-loop laundering gate** (a claim whose evidence resolves *only* to other agent-authored sessions is quarantined until an external-grounded citation or human judgment breaks the cycle); `author_kind` differential trust in the inject gate; provenance-cycle quarantine reusing `TopologyEdgeStatus`; never alert/surface on `generated_context_pack` material. This is what makes auto-capture-of-own-R&D safe.
10. **"Wire what already exists" beats "build new."** proactive found a full 5-backend notification fan-out (only carrying ops alerts); search found `score_components` already carries per-lane RRF decomposition; incident found postmortem-bundle machinery; economics found lease/GC + retention already present. Most wave-2 wins are *routing an existing pipe to a new signal*.
11. **The episode is the missing keystone unit.** episode + missing-units + self-capture converge: a logical task spanning sessions/tools/time, derived above `session_links`, stitched by a 4-signal scorer with a hard false-merge floor, the right grain for cost-per-outcome and "how I actually solved X."
12. **Construct-validity became executable everywhere.** measure-catalog's coverage-precondition registry; meta-quality's "insight claiming a number over zero rows must FAIL" CI gate; ingestion's byte-fidelity ratio; temporal's `time_confidence` field. The brand ("every number resolves to bytes") is now a test lane, a column, and a gate.

## Ideas by lane (curated strongest ~5 each)

### W2-Q — Queries/findings-as-objects (deep)
- `query:<hash>` keyed on planned AST post-macro-expansion (dedup, cache-key, "have I run this"). — NEW
- `result_sets` snapshots in index.db (derived), Merkle root over sorted keys for O(1) change-detect; only a promoted finding referencing one is durable. — NEW
- `AssertionKind.FINDING` reuses pathology candidate→judge lifecycle + inject-gating verbatim (zero new lifecycle code). — NEW
- Findings-as-tests: promoted finding with `value.expected` → a `ConvergenceStage` re-runs it, emits `finding-drift` candidate on divergence. — NEW
- Content-anchor evidence as `session:X@<content_hash>` so a re-materialize mismatch flags possibly-stale instead of lying; provenance-cycle quarantine via `TopologyEdgeStatus`. — NEW

### W2-M — Measure catalog (deep)
- The 5-tuple algebra as the registry contract; catalog = product not list. — 9l5.7
- 16 measures spec'd (cache-amplification, thinking-tax, latency three-lane, interaction-latency asymmetry, engaged-vs-wall, stuck-tool density, compaction pressure, substantive density, outcome-conditioned cost, tool-mix entropy, workflow-shape transition matrix, session-redundancy zstd ratio, thread-cost Gini, subagent fan-out, pathology epidemiology, credit-vs-API divergence). — 9l5.x
- Uncertainty layer: Wilson for proportions, bootstrap CI for median/pXX, coverage-tier gate; one-line honest footnote (`n=…, coverage=evidence, timing_provenance=structural`) from one registry declaration. — 9l5.7
- Denominator/unit-frame is the #1 construct-validity trap — the registry must enforce it. — 9l5.7

### W2-S — Self-capturing recursive loop (deep)
- Synthetic runtime meta-session (convergence/ingest/debt events as a session tree) → every insight applies to polylogue on itself. — NEW
- R&D cohort auto-minted per agent wave (session-tree ∪ repo ∪ time-window), judged/diffed/cited as a unit. — NEW
- Closed-loop laundering gate (THE invariant); hallucination-provenance cohort down-weights sessions that historically fabricated. — NEW
- Design-chat→bead→PR provenance chain (resolve_ref on a bead walks back to the originating GPT-pro design chat). — NEW
- Loop-latency as a native measure (design-chat ingest → bead → PR merged → postmortem); cohort drift/novelty score vs prior waves. — NEW

### W2-Act — Activation & adoption
- `polylogue install` one-command idempotent hook wiring (all 16 CC + 6 Codex events) + `polylogue doctor` hook-liveness heartbeat. — d1y
- Adoption measured FROM the archive: count `mcp__polylogue__*` tool_use per session/repo → adoption-rate insight; "why isn't this used" diagnosis for configured-but-zero-usage repos. — NEW
- `what_now(intent)` router + primary-tool tiering (progressive disclosure over 130 tools). — pj8
- Assertions-as-injected-continuity (assertions > CLAUDE.md), token-budgeted ranked preamble; adoption canary as a dogfood CI gate. — 3gd
- A/B activation experiment rail (config versioned + adoption measured → empirical, not faith). — 3gd/7aw

### W2-Inc — Incident/compaction/resilience
- `CompactionEvent` as a first-class archived object; loss-forensics as a construct measure; always-on JSONL-boundary fallback snapshot (degrade honestly when the hook is absent). — gjg
- `compaction_forgot(session_id)` MCP tool; content-addressed compaction snapshots via blob dedup (near-free). — gjg
- `daemon_lifecycle` table + heartbeat-backed liveness (not pid-file truth); crash-recovery convergence-debt reconciliation. — peo
- Postmortem bundle ingests the incident (daemon deaths, debt strandings, compaction events overlapping the window). — 8jg9
- The voluntary-handoff / involuntary-compaction / cross-session-resumption triad = one OS-memory story under 37t. — gjg/37t

### W2-T — Temporal doctrine
- Name the four times as a typed enum (source/ingest/derived-order/event); cross-kind comparison → mypy error. — cpf.1
- `sort_key_provenance` (explicit/inherited/fell-back-to-ingest/synthesized-0) so "ordered by real time" is queryable; the epoch-0 vanish bug. — NEW
- LIVE BUG: `RELATIVE_BASE = datetime.now()` at import → `since:7d` anchors to process-start, unpatchable by frozen_clock. — NEW
- Skew-tolerance band (co-temporal within band); total stable tiebreak `(sort_key_ms, session_id, position, block_position)` — a citation-anchor prerequisite. — NEW
- `time_confidence` on read payloads; tz-unknown-by-default (`source_tz`/`tz_provenance`); backfill must never move `sort_key_ms`; `docs/doctrine/time.md`. — cpf

### W2-Econ — Archive economics & longevity
- Storage-growth ledger insight (bytes/session/day by tier×origin + trend + 1/3/10yr projection). — NEW
- Promote/re-parent zstd blob compression (36GB→~4-7GB, address unchanged) OUT of the attachment epic; per-origin trained dictionaries. — 83u.5
- Access-temperature tier orthogonal to durability (cold >18mo → level-19 recompress + `cold/` shard); yearly epoch: repack fully-cold years into frozen read-only segments. — NEW
- Embeddings are the second storage bomb: int8/binary quantize vec0 (4×), embed only human+assistant blocks, drop cold; measure the derived-rebuild cost (the "rebuildable ≠ free" liability). — NEW
- `reacquirability` class on blobs (drop re-derivable-from-`~/.claude` first) + citable drop-tombstones + per-origin bytes/session SLO circuit-breaker. — 83u.4

### W2-Ing — Ingestion robustness & coverage
- Byte-fidelity ratio per raw_session (parsed_bytes/blob_size) — silent-drop becomes queryable. — NEW
- Round-trip reconstruction check vs blob_hash; unparsed-key census (ranked ignored provider keys). — NEW
- Detection-ambiguity score + misclassification tripwire (run other detectors post-parse); streaming-window blind-spot guard (islice 32). — NEW
- `parser_fingerprint` on raw_sessions → force reprocess on parser improvement (fidelity gains backfill). — NEW
- Zero-message parse-success anomaly detector; `unknown/grok-export` volume funnel; decode-failure taxonomy (encoding/framing/schema). — NEW

### W2-Ep — Cross-tool episode reconstruction
- `episode`/`episode_members` table in index.db, content-hashed member set; a tier ABOVE topology (lineage stays leaf). — NEW
- 4-signal scorer (repo × time-kernel × embedding × shared-artifact) with contributions in `evidence_json`; shared-artifact (file/SHA/error-fingerprint) dominates. — NEW
- Confidence tiers with a hard false-merge floor (linked > corroborated > candidate); anti-stitch negative evidence. — NEW
- Episode→commit/PR attribution (`produced_ref`); episode-level cost/effort rollup honoring material_origin + lineage-dedup. — NEW
- Episode as a stitch-hypothesis assertion (confirm/split/reject feeds the scorer); lynchpin terminal-artifact glue (a commit with no in-window session is still a member). — NEW

### W2-U — Missing data-model units
- episode; turn-pair (VIEW); entity-mention/reference (table); artifact produced/consumed (VIEW→table); world-effect cause→effect edge (VIEW). — NEW
- verification-run outcome (exit_code keystone); tool-outcome + retry-chain (extend `actions`); correction edge anchored to the block it corrects. — extend/NEW
- project (repo ∪ worktrees ∪ cwd ∪ `g-p-` gizmo_id) as durable user.db dim; topic/theme cluster (HDBSCAN over vec0). — NEW
- cross-origin thread (formalize `threads`); semantic handoff edge (materialize resume graph); phase segment as rows (extend session_phases). — extend
- goal/intent + decision-object as construct-gated candidates (recursive-safety); spend-episode cost VIEW. — NEW

### W2-Search — Relevance & explainability
- Lineage-aware result dedup/collapse (one thread ≠ N near-identical hits) with `variant_count`. — NEW
- `explain search`/`--why` narrating existing `score_components`; `AssertionKind.RELEVANCE` judgments from `mark` → the LtR corpus. — NEW
- `devtools lab search-eval` (nDCG@10/MRR/recall per lane, gate weight/k changes); weighted RRF; query-intent classifier before lane routing. — NEW
- Did-you-mean / query relaxation in miss-diagnostics (each with the count it would yield); field-boosted bm25 (title/opening > tool-log); recency×relevance knob. — NEW
- MMR diversity re-rank; score-calibration for cross-query comparability (never show raw bm25 as %); `weak_evidence` band on single-lane deep-rank hits. — NEW

### W2-Prox — Proactive/ambient surfacing
- Semantic alert tier on the EXISTING `HealthAlert` fan-out (add a `Notice` severity + `content` family); zero new channel. — NEW
- Standing queries as `daemon_events` producers (`notify_on: appeared|disappeared|count_crossed`); `polylogue brief --since 24h` = a query over the event ledger (the oracle habit, deterministic). — NEW
- "You're repeating a past mistake" real-time nudge (embed live tail → find_similar vs pathology/lesson sessions that ended badly, cite the prior failure). — NEW (highest-value ambient signal)
- Three-cadence policy table (on-event/daily/weekly) + per-family token bucket + `SUPPRESSION`-assertion snooze; recursive-safety (no alert on generated context); now-quiet deferral reusing hot-file logic. — NEW
- MCP-injected ambient preamble (brief the next agent before it starts); ambient-event replay audit ("what did you NOT tell me"). — NEW

### W2-Att — Attention/triage frontier
- A context-free `frontier` view ranking all 16.7k (invert cwd-coupled find_resume_candidates) + one `worth_reviewing_score` per logical session. — NEW
- `AssertionKind.triaged` (resumed/wont_resume/archived/snoozed:<until>) → the frontier is a true inbox that empties; snooze-with-wake. — NEW
- Durable-vs-disposable classifier as a first-class axis; "started and never finished" = terminal-state + no lineage/file follow-up. — NEW
- Cluster loose ends by thread/repo into cards (triage topics not rows); surface the actual open-question text with a citation-anchor. — NEW
- Blocker-centric lane ("N sessions blocked on the same thing" via embedding-cluster of blocker prose); inverted-U attention-urgency curve; frontier honesty (render low-confidence as a guess). — NEW

### W2-MQ — Meta-quality (proving polylogue's own correctness)
- Identity round-trip + injectivity metamorphic property (the `run_ref` global-PK collision #2464 class); content-hash idempotency suite (NFC adversaries, user-metadata excluded). — NEW
- Lineage compose⊕identity law + no-double-count invariant at scale (the ~32% dup #2467 as an executable law). — NEW
- FTS5 trigger-coherence metamorphic test; construct-validity CI gate (insight over zero rows must FAIL); recovery-digest fabrication regression. — NEW
- Daemon-vs-direct differential lane (same corpus → byte-identical index state); convergence idempotency + `false_means_pending` fixpoint; SIGKILL fault-injection at each stage boundary. — NEW
- Scale-shaped synthetic corpus generator as first-class infra (the missing 1xc.1 substrate); WAL/commit-boundary assertion helper; CHECK↔Literal drift property. — 1xc infra

## Prompt additions (wave-2) — titles + tag + focus (ask me to expand any to paste-ready)

Design [A]:
- A16 [A] Measure catalog + construct-validity registry — fill the 5-tuple registry row (construct, formula, evidence-tier, denominator, top-2 confounds, coverage precondition) for the 16 measures; flag provenance-mixing denominators. (sharper A1)
- A17 [A] Query→finding object-model migration — canonical AST hashing, result_sets tier placement + `reset --index` interaction, supersession-versioning vs content-addressed ids; DDL + the dropped-snapshot failure mode. (sharper A2)
- A18 [A] Recursive-safety invariant for a self-ingesting archive — the closed-loop predicate, quarantine state machine, author-kind trust, release conditions; SQL-expressible check + too-loose/too-tight failure modes.
- A19 [A] Runtime meta-session schema — project daemon/convergence/debt events into a queryable synthetic session tree; identity/hashing, tier, new self-analytics unlocked.
- A20 [A] `CompactionEvent` schema + loss-forensics diff — structural retained/lost/transformed classification, rank-by-later-reference, honest degradation on JSONL-boundary snapshots.
- A21 [A] Four-times stable ordering — total, skew-tolerant comparator with per-key provenance; identical corpora → identical order; backfill sorts by history; invariant tests.
- A22 [A] Tiering + retention policy engine — durability × access-temperature classification driving recompress/drop/repack; tombstone schema, space-pressure trigger, lease/GC-safe invariants.
- A23 [A] Adoption-measurement layer — signals distinguishing "used" vs "invisible", adoption-rate controlling for irrelevant repos, the zero-usage failure taxonomy.
- A24 [A] Progressive-disclosure/intent-routing over 130 MCP tools — primary-tagging vs `what_now()` router vs prompts vs skill; selection-accuracy/token tradeoffs.
- A25 [A] Ingest-fidelity metric — byte round-trip loss vs intended omission, cheap on multi-GB JSONL, one coverage number that never hides truncation/depth-cap drops.
- A26 [A] Provider misclassification detector + ambiguity score — winner-vs-runner-up margin, bounded cost on sampled JSONL, late-signal-past-window failure mode.
- A27 [A] Episode confidence/merge model — 4-signal scorer with a hard false-merge floor, confidence tiers, anti-stitch signals, per-edge auditable evidence.
- A28 [A] Table-vs-VIEW boundary for derived units — decision matrix (recursion depth, embedding-join cost, rebuild cost, query-unit exposure) applied to the 16 proposed units.
- A29 [A] Derived-intent construct-validity gate — candidate→verified state machine for goal/decision/phase units; citation-anchor requirement; the recovery-digest incident as a test case.
- A30 [A] Explainable ranking UX — "why ranked here" template over `score_components`, calibrated cross-query confidence (no raw bm25 %), `weak_evidence` threshold.
- A31 [A] Semantic notification policy layer — operational-vs-content taxonomy, per-family cadence + token-bucket + durable snooze, "worth interrupting" scoring, no-self-alert constraint.
- A32 [A] `worth_reviewing_score` + triage inbox state model — decomposable score, inverted-U urgency curve, force-to-zero conditions; triaged/snooze-with-wake state transitions + the "today's frontier" query.
- A33 [A] Metamorphic/property-test taxonomy — the algebraic laws (identity round-trip, hash idempotency, lineage compose⊕no-double-count, FTS coherence); per-law relation + generator + oracle, ranked by bug-class caught.

Deepresearch [DR]:
- D15 [DR] Metric-as-composable-object prior art — dbt-metrics/MetricFlow, PostHog, Cube.js, Honeycomb: dimension/window/comparison/uncertainty algebra + the traps (Simpson's, ratio-of-ratios, censored means).
- D16 [DR] Self-ingesting loops without error amplification — model-collapse mitigation, agent memory poisoning, provenance/taint tracking, retrieval down-weighting of self-generated content.
- D17 [DR] Compaction handling across agent frameworks — does any framework snapshot full pre-compaction context, measure loss, or re-ground from an external store? Prior art for "compaction as a first-class observable event."
- D18 [DR] Daemon-death forensics & heartbeat liveness — systemd Restart, faulthandler stack dumps, atexit clean-vs-vanish, heartbeat-age vs pid-file; reconciling half-finished derived work after unclean crash.
- D19 [DR] Event-time vs ingest-time vs processing-time — Datomic/OTel/Git/IMAP/journald conventions for clock skew, unknown tz, backfilled records, temporal-uncertainty display.
- D20 [DR] Relative-date & calendar-bucketing failure modes — import-time-frozen bases, DST/NTP-corrupted durations, UTC-midnight buckets misassigning local evening activity; hermetic time-injection patterns.
- D21 [DR] Compression/retention/dedup for personal append-only content-addressed stores — zstd trained dicts vs generic, FastCDC economics, restic/borg/Kopia incremental over immutable chunks, embedding int8/binary recall, time-decay sampling.
- D22 [DR] The "rebuildable tier" fallacy — when delete-and-rebuild beats retaining (re-embed $ + CPU-hours + rebuild risk); how search engines/feature-stores/vector DBs budget reindex; tiered rebuild-on-demand.
- D23 [DR] Silent field-drop & source-drift in ETL/archiving — dovecot/notmuch, ES ingest pipelines, Singer/Airbyte, OTel collectors: residual capture, dead-letter queues, coverage/lag, reprocess-on-parser-improvement.
- D24 [DR] Cross-tool episode/record-linkage — activity-timeline reconstruction, session segmentation, entity-resolution confidence, trace stitching without a shared id; adaptive temporal boundaries + false-link prevention.
- D25 [DR] Commit↔reasoning attribution — signals attributing a commit to the session that produced it; separating human vs machine authorship; cost/effort per shipped unit spread across tools + untranscribed steps.
- D26 [DR] Knowledge-graph/backlink/EAV schema prior art — Roam/Obsidian, Datomic, Gerrit change-objects, OTel span↔resource: mention edges, cause→effect provenance, episode clustering without storing content N×.
- D27 [DR] RRF tuning + result diversification — choosing k / per-lane weights / score normalization; offline relevance judgment sets; MMR vs cluster-then-rank vs canonical-collapse over forked document trees.
- D28 [DR] Notification fatigue & proactive-surfacing — interruptibility (Horvitz), digest-vs-realtime, peripheral-awareness UI, "repeating a past mistake" nudging; thresholds + empirical fatigue-onset.
- D29 [DR] Saved-query-as-subscription — incremental view maintenance / Materialize / Debezium / changedetection: appeared/disappeared/count-crossed, dedup by stable identity, no re-notification storms after backfill.
- D30 [DR] Human triage of large heuristic queues — email imbox, issue triage, SOC alert de-dup, read-later: scoring, snooze, clustering, zero-inbox pressure, staleness decay, dismissal audit; what fails when items are immutable.
- D31 [DR] Persisted-queries-as-objects — dbt exposures/snapshots, Malloy, Dagster asset checks, Great Expectations, Datasette canned queries, LookML, Metabase: query identity, result-set versioning, staleness triggers, test-bound-to-query.
- D32 [DR] Scale-only + fault-injection testing for embedded DBs — WAL-blowup regressions, crash-consistency fuzzing at txn boundaries, deterministic simulation (FoundationDB/TigerBeetle/Antithesis), property state-machines for GC/lease races; what's cheap on a synthetic corpus and what it misses.
- D33 [DR] Construct-validity as an executable substrate — Great Expectations / dbt tests / Monte Carlo / semantic-diff / authorship-gating: assert non-empty backing rows, non-fabricated text-mined events, no lineage double-count; a minimal self-verifying-archive gate.

Total fan-out now: 33 [A] + 33 [DR] = 66 prompts. Recommended first batch (highest leverage, run these ~20 first): A16, A17, A18, A2/A17-merge, A1/A16-merge, A11, A20, A32, A27, A13, A14, D1, D2, D7, D15, D16, D21, D24, D30, D33.
